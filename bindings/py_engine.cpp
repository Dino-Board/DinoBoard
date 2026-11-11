#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../engine/core/game_interfaces.h"
#include "../engine/core/rng_salt.h"
#include "../engine/core/schema_hash.h"
#include "../engine/core/game_registry.h"
#include "../engine/core/heuristic.h"
#include "../engine/core/heuristic_registry.h"
#include "../engine/core/feature_encoder.h"
#include "../engine/core/masked_state.h"
#include "../engine/core/viewer_state.h"
#include "../engine/core/snapshot_io.h"
#include "../engine/infer/batched_belief_evaluator.h"
#include "../engine/infer/batched_policy_value_evaluator.h"
#include "../engine/infer/evaluator_pool.h"
#include "../engine/infer/onnx_belief_evaluator.h"
#include "../engine/infer/onnx_policy_value_evaluator.h"
#include "../engine/infer/split_policy_value_evaluator.h"
#include "../engine/runtime/selfplay_runner.h"
#include "../engine/runtime/arena_runner.h"
#include "../engine/runtime/heuristic_runner.h"
#include "../heuristic/mcts_pv_heuristic.h"
#include "../engine/search/net_mcts.h"
#include "../engine/search/root_noise.h"
#include "../engine/search/temperature_schedule.h"

namespace py = pybind11;

using namespace board_ai;

namespace {

// Parse the user-facing string into the OpponentSelection enum. Unknown
// values raise — never silently fall back to puct (no_silent_degradation).
inline search::OpponentSelection parse_opponent_selection(
    const std::string& s) {
  if (s == "puct") return search::OpponentSelection::kPuct;
  if (s == "prior" || s == "frozen_prior") {
    return search::OpponentSelection::kFrozenPrior;
  }
  throw std::invalid_argument(
      "opponent_selection: unknown value '" + s +
      "' (expected 'puct' or 'prior')");
}

// seat_configs flag that must be a real bool. cast<bool>() would also accept
// 0 / 1 / None and so hide a mistyped config value.
inline bool require_bool(const py::handle& value, const char* key) {
  if (!py::isinstance<py::bool_>(value)) {
    throw std::invalid_argument(
        std::string("seat_configs: '") + key + "' must be a bool");
  }
  return value.cast<bool>();
}

// Tracker adapter helpers: convert the old (state_before, action, state_after)
// trio into the tracker's new event-only input. Games with a registered
// public_event_extractor produce the event stream; games without one pass
// empty vectors (tracker becomes effectively a no-op for those games, which
// matches their pre-migration behavior since they had no hidden info).
inline void tracker_init(IBeliefTracker& bt, const GameBundle& /*bundle*/,
                         IGameState& state, int perspective,
                         const AnyMap& payload = {}) {
  // Two-step init: walker `apply_public_snapshot` (the broadcast
  // public snapshot path) is the responsibility of the caller; here
  // we just hand the
  // perspective-private bootstrap payload to the tracker. The tracker
  // writes any private slots into `state` and seeds its own memory by
  // reading public facts from `state`.
  bt.init(state, perspective, payload);
}

inline void tracker_observe(IBeliefTracker& bt, const GameBundle& bundle,
                            const IGameState& before, ActionId action,
                            const IGameState& after, int perspective) {
  PublicEventTrace trace;
  if (bundle.public_event_extractor) {
    trace = bundle.public_event_extractor(before, action, after, perspective);
  }
  const int actor = before.current_player();
  bt.observe_public_event(actor, bundle.rules->public_action(before, action), trace.events);
}

// One belief tracker per seat for single-truth-state game loops (eval vs
// heuristic). Each is initialized once from its own seat's masked view and
// fed every public event from its own perspective, so no seat's search can
// see another seat's private cards.
class SeatTrackers {
 public:
  SeatTrackers(const GameBundle& bundle, const IGameState& state) {
    if (!bundle.belief_tracker) return;
    for (int p = 0; p < state.num_players(); ++p) {
      auto t = bundle.belief_tracker->clone();
      auto view = make_masked_state(state, state.schema_ref(), p);
      t->init(*view, p, AnyMap{});
      per_seat_.push_back(std::move(t));
    }
  }
  bool active() const { return !per_seat_.empty(); }
  IBeliefTracker* get(int seat) const {
    if (!active()) return nullptr;
    if (seat < 0 || seat >= static_cast<int>(per_seat_.size())) {
      throw std::out_of_range("SeatTrackers::get: seat out of range");
    }
    return per_seat_[static_cast<size_t>(seat)].get();
  }
  void observe(const GameBundle& bundle, const IGameState& before,
               ActionId action, const IGameState& after) {
    for (size_t p = 0; p < per_seat_.size(); ++p) {
      tracker_observe(*per_seat_[p], bundle, before, action, after, static_cast<int>(p));
    }
  }

 private:
  std::vector<std::unique_ptr<IBeliefTracker>> per_seat_;
};

std::any py_to_any(const py::handle& obj);

AnyMap py_dict_to_any_map(const py::dict& d) {
  AnyMap m;
  for (const auto& [k, v] : d) {
    const std::string key = py::cast<std::string>(k);
    m[key] = py_to_any(v);
  }
  return m;
}

std::any py_to_any(const py::handle& obj) {
  if (py::isinstance<py::bool_>(obj)) return py::cast<bool>(obj);
  if (py::isinstance<py::int_>(obj))  return py::cast<int>(obj);
  if (py::isinstance<py::float_>(obj)) return py::cast<double>(obj);
  if (py::isinstance<py::str>(obj))   return py::cast<std::string>(obj);
  if (py::isinstance<py::dict>(obj)) {
    return py_dict_to_any_map(py::reinterpret_borrow<py::dict>(obj));
  }
  if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
    // Inspect first element to decide vector<int> vs vector<any>.
    py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);
    bool all_int = !seq.empty() ? py::isinstance<py::int_>(seq[0]) : false;
    if (all_int) {
      for (const auto& item : seq) {
        if (!py::isinstance<py::int_>(item) || py::isinstance<py::bool_>(item)) {
          all_int = false;
          break;
        }
      }
    }
    if (all_int) {
      std::vector<int> out;
      out.reserve(seq.size());
      for (const auto& item : seq) out.push_back(py::cast<int>(item));
      return out;
    }
    std::vector<std::any> out;
    out.reserve(seq.size());
    for (const auto& item : seq) out.push_back(py_to_any(item));
    return out;
  }
  throw std::runtime_error("py_to_any: unsupported Python type");
}

py::object any_to_py(const std::any& val) {
  if (val.type() == typeid(int))
    return py::cast(std::any_cast<int>(val));
  if (val.type() == typeid(float))
    return py::cast(std::any_cast<float>(val));
  if (val.type() == typeid(double))
    return py::cast(std::any_cast<double>(val));
  if (val.type() == typeid(bool))
    return py::cast(std::any_cast<bool>(val));
  if (val.type() == typeid(std::string))
    return py::cast(std::any_cast<std::string>(val));
  if (val.type() == typeid(std::vector<int>))
    return py::cast(std::any_cast<std::vector<int>>(val));
  if (val.type() == typeid(std::vector<std::vector<int>>)) {
    py::list lst;
    for (const auto& inner :
         std::any_cast<std::vector<std::vector<int>>>(val)) {
      lst.append(py::cast(inner));
    }
    return lst;
  }
  if (val.type() == typeid(std::vector<std::any>)) {
    py::list lst;
    for (const auto& item : std::any_cast<std::vector<std::any>>(val))
      lst.append(any_to_py(item));
    return lst;
  }
  if (val.type() == typeid(std::vector<AnyMap>)) {
    py::list lst;
    for (const auto& item : std::any_cast<std::vector<AnyMap>>(val)) {
      py::dict d;
      for (const auto& [k, v] : item) d[py::cast(k)] = any_to_py(v);
      lst.append(d);
    }
    return lst;
  }
  if (val.type() == typeid(AnyMap)) {
    py::dict d;
    for (const auto& [k, v] : std::any_cast<AnyMap>(val))
      d[py::cast(k)] = any_to_py(v);
    return d;
  }
  throw std::runtime_error(
      std::string("any_to_py: unsupported type: ") + val.type().name());
}

template <typename T>
py::array_t<T> to_ndarray(const std::vector<T>& v) {
  return py::array_t<T>(static_cast<py::ssize_t>(v.size()), v.data());
}

// `compact`: features as a float32 ndarray and legal_mask as a uint8
// ndarray instead of Python lists (~8x less memory, no per-element
// PyFloat allocation), for the training pipeline's compact replay. Fast
// (playout-cap) samples carry no features / legal_mask in compact mode:
// they never enter the replay buffer, and with long games they dominate
// the per-step payload.
py::dict sample_to_py(const runtime::SelfplaySample& s, bool compact = false) {
  py::dict d;
  d["ply"] = s.ply;
  d["player"] = s.player;
  d["action_id"] = s.action_id;
  d["z"] = s.z;
  d["z_values"] = s.z_values;
  d["policy_action_ids"] = s.policy_action_ids;
  d["policy_action_visits"] = s.policy_action_visits;
  if (compact) {
    if (s.full_search) {
      d["features"] = to_ndarray(s.features);
      py::array_t<std::uint8_t> mask(static_cast<py::ssize_t>(s.legal_mask.size()));
      auto* m = mask.mutable_data();
      for (std::size_t i = 0; i < s.legal_mask.size(); ++i) m[i] = s.legal_mask[i] != 0.0f ? 1 : 0;
      d["legal_mask"] = mask;
    }
  } else {
    d["features"] = s.features;
    d["legal_mask"] = s.legal_mask;
  }
  d["tail_solved"] = s.tail_solved;
  d["full_search"] = s.full_search;
  d["simulations"] = s.simulations;
  d["heuristic_simulations"] = s.heuristic_simulations;
  return d;
}

// `compact`: every field as an ndarray (features float32, the counts int32,
// hand_counts shaped [N-1, K]) — the belief buffer holds replay_buffer_size
// of these, and as Python lists each one costs ~8x more.
py::dict belief_sample_to_py(const runtime::BeliefSample& s, bool compact = false) {
  py::dict d;
  d["ply"] = s.ply;
  d["observer"] = s.observer;
  if (compact) {
    const std::size_t rows = s.hand_counts.size();
    const std::size_t k = rows ? s.hand_counts[0].size() : 0;
    py::array_t<int> hand({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(k)});
    auto* h = hand.mutable_data();
    for (std::size_t r = 0; r < rows; ++r) {
      if (s.hand_counts[r].size() != k) {
        throw std::runtime_error("belief_sample_to_py: ragged hand_counts rows");
      }
      std::copy(s.hand_counts[r].begin(), s.hand_counts[r].end(), h + r * k);
    }
    d["features"] = to_ndarray(s.features);
    d["hand_counts"] = hand;
    d["remaining"] = to_ndarray(s.remaining);
    d["alive_per_opp"] = to_ndarray(s.alive_per_opp);
    return d;
  }
  d["features"] = s.features;
  py::list hand_counts;
  for (const auto& row : s.hand_counts) hand_counts.append(py::cast(row));
  d["hand_counts"] = hand_counts;
  d["remaining"] = s.remaining;
  d["alive_per_opp"] = s.alive_per_opp;
  return d;
}

py::dict result_to_py(const runtime::SelfplayEpisodeResult& result,
                      const GameBundle& bundle, bool compact_samples = false) {
  py::list samples;
  for (const auto& s : result.samples)
    samples.append(sample_to_py(s, compact_samples));

  py::dict out;
  if (bundle.final_score_extractor && result.final_state) {
    out["final_scores"] = bundle.final_score_extractor(*result.final_state);
  }
  out["winner"] = result.winner;
  out["draw"] = result.draw;
  // Per-seat terminal z, length num_players. Pipeline converts to
  // (z+1)/N shares for the unified per-seat tally (`pX=` log + eval
  // wins). Empty only when num_players==0 (degenerate).
  out["terminal_z_values"] = result.terminal_z_values;
  out["total_plies"] = result.total_plies;
  out["samples"] = samples;
  py::list belief_samples;
  for (const auto& bs : result.belief_samples)
    belief_samples.append(belief_sample_to_py(bs, compact_samples));
  out["belief_samples"] = belief_samples;
  out["belief_label_class_count"] = result.belief_label_class_count;
  out["tail_solve_attempts"] = result.tail_solve_attempts;
  out["tail_solve_completed"] = result.tail_solve_completed;
  out["tail_solve_successes"] = result.tail_solve_successes;
  out["tail_solve_total_ms"] = result.tail_solve_total_ms;

  if (!result.custom_stats.empty()) {
    py::dict stats;
    for (const auto& [k, v] : result.custom_stats) stats[py::cast(k)] = v;
    out["custom_stats"] = stats;
  }

  if (result.trace_enabled) {
    out["trace_perspective"] = result.trace_perspective;
    // initial_observation: game-defined AnyMap
    py::dict init_obs;
    for (const auto& [k, v] : result.initial_observation) init_obs[py::cast(k)] = any_to_py(v);
    out["initial_observation"] = init_obs;
    // initial_belief_snapshot: tracker's serialize() output
    py::dict init_bs;
    for (const auto& [k, v] : result.initial_belief_snapshot) init_bs[py::cast(k)] = any_to_py(v);
    out["initial_belief_snapshot"] = init_bs;
    // observation_trace: list of per-ply records
    py::list trace_list;
    for (const auto& t : result.observation_trace) {
      py::dict entry;
      entry["ply"] = t.ply;
      entry["actor"] = t.actor;
      entry["action"] = static_cast<int>(t.action);
      py::list ev_list;
      for (const auto& [kind, payload] : t.events) {
        py::dict e;
        e["kind"] = kind;
        py::dict p;
        for (const auto& [pk, pv] : payload) p[py::cast(pk)] = any_to_py(pv);
        e["payload"] = p;
        ev_list.append(e);
      }
      entry["events"] = ev_list;
      // Truth-side public snapshot. Empty for games without
      // public_state_applier registered.
      py::dict snap;
      for (const auto& [k, v] : t.public_snapshot) snap[py::cast(k)] = any_to_py(v);
      entry["public_snapshot"] = snap;
      py::dict bs;
      for (const auto& [k, v] : t.belief_snapshot_after) bs[py::cast(k)] = any_to_py(v);
      entry["belief_snapshot_after"] = bs;
      trace_list.append(entry);
    }
    out["observation_trace"] = trace_list;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Shared helpers — deduplicate boilerplate across binding functions.
// ---------------------------------------------------------------------------

struct PerSeatBundles {
  std::vector<std::unique_ptr<GameBundle>> bundles;
  std::vector<IBeliefTracker*> trackers;
  std::vector<IGameState*> states;
};

PerSeatBundles create_per_seat_bundles(
    const std::string& game_id, std::uint64_t seed, int num_players) {
  PerSeatBundles r;
  r.bundles.reserve(static_cast<size_t>(num_players));
  r.trackers.reserve(static_cast<size_t>(num_players));
  r.states.reserve(static_cast<size_t>(num_players));
  for (int p = 0; p < num_players; ++p) {
    auto pb = std::make_unique<GameBundle>(
        GameRegistry::instance().create_game(game_id, seed));
    r.trackers.push_back(pb->belief_tracker.get());
    r.states.push_back(pb->state.get());
    r.bundles.push_back(std::move(pb));
  }
  return r;
}

runtime::SelfplayConfig build_selfplay_config(
    int simulations, int world_cap, float c_puct, double temperature,
    double dirichlet_alpha, double dirichlet_epsilon,
    int dirichlet_on_first_n_plies, int max_game_plies,
    bool tail_solve_enabled, int tail_solve_depth_limit,
    std::int64_t tail_solve_node_budget, float tail_solve_auxiliary_weight,
    double temperature_initial, double temperature_final,
    int temperature_decay_plies,
    double heuristic_guidance_ratio, double heuristic_temperature,
    double training_filter_ratio,
    search::OpponentSelection opp_sel,
    bool safe_mode,
    int top_k,
    std::optional<float> fpu_reduction,
    bool uct2_arrival_visits) {
  runtime::SelfplayConfig cfg{};
  cfg.simulations = simulations;
  cfg.world_cap = world_cap;
  cfg.c_puct = c_puct;
  cfg.fpu_reduction = fpu_reduction;
  cfg.uct2_arrival_visits = uct2_arrival_visits;
  cfg.temperature = temperature;
  cfg.dirichlet_alpha = dirichlet_alpha;
  cfg.dirichlet_epsilon = dirichlet_epsilon;
  cfg.dirichlet_on_first_n_plies = dirichlet_on_first_n_plies;
  cfg.max_game_plies = max_game_plies;
  cfg.tail_solve_enabled = tail_solve_enabled;
  cfg.tail_solve_config.depth_limit = tail_solve_depth_limit;
  cfg.tail_solve_config.node_budget = tail_solve_node_budget;
  cfg.tail_solve_config.auxiliary_weight = tail_solve_auxiliary_weight;
  cfg.heuristic_guidance_ratio = heuristic_guidance_ratio;
  cfg.heuristic_temperature = heuristic_temperature;
  cfg.training_filter_ratio = training_filter_ratio;
  cfg.opponent_selection = opp_sel;
  cfg.safe_mode = safe_mode;
  cfg.top_k = top_k;
  if (temperature_initial >= 0.0 || temperature_final >= 0.0) {
    cfg.temperature_schedule.enabled = true;
    if (temperature_initial >= 0.0) {
      cfg.temperature_schedule.has_initial = true;
      cfg.temperature_schedule.initial = temperature_initial;
    }
    if (temperature_final >= 0.0) {
      cfg.temperature_schedule.has_final = true;
      cfg.temperature_schedule.final_ = temperature_final;
    }
    cfg.temperature_schedule.decay_plies = temperature_decay_plies;
  }
  return cfg;
}

struct BeliefEvalResult {
  std::unique_ptr<infer::OnnxBeliefEvaluator> evaluator;
  const IBeliefEvaluator* ptr = nullptr;
};

BeliefEvalResult create_belief_evaluator(
    const GameBundle& bundle, const std::string& caller_name) {
  BeliefEvalResult r;
  if (bundle.belief_feature_extractor && !bundle.belief_model_path.empty()) {
    r.evaluator = std::make_unique<infer::OnnxBeliefEvaluator>(
        bundle.belief_model_path);
    if (!r.evaluator->is_ready()) {
      throw std::runtime_error(
          caller_name + ": failed to load belief model '" +
          bundle.belief_model_path + "': " + r.evaluator->last_error());
    }
    r.ptr = r.evaluator.get();
  }
  return r;
}

py::dict run_selfplay_episode_py(
    const std::string& game_id,
    std::uint64_t seed,
    const std::string& model_path,
    int simulations,
    int world_cap,
    float c_puct,
    double temperature,
    double dirichlet_alpha,
    double dirichlet_epsilon,
    int dirichlet_on_first_n_plies,
    int max_game_plies,
    bool tail_solve_enabled,
    int tail_solve_depth_limit,
    std::int64_t tail_solve_node_budget,
    float tail_solve_auxiliary_weight,
    double temperature_initial,
    double temperature_final,
    int temperature_decay_plies,
    double heuristic_guidance_ratio,
    double heuristic_temperature,
    double training_filter_ratio,
    int trace_perspective,
    const std::string& opponent_selection,
    const std::string& belief_model_path,
    const std::string& heuristic_selector,
    bool safe_mode,
    int top_k,
    std::optional<float> fpu_reduction,
    bool uct2_arrival_visits) {
  // Validate before releasing the GIL — exception propagation is cleaner.
  const auto opp_sel = parse_opponent_selection(opponent_selection);
  py::gil_scoped_release release;

  auto bundle = GameRegistry::instance().create_game(game_id, seed);
  if (!belief_model_path.empty()) bundle.belief_model_path = belief_model_path;
  const board_ai::IHeuristic* heuristic_ptr = nullptr;
  if (!heuristic_selector.empty() && bundle.heuristics.has(heuristic_selector)) {
    heuristic_ptr = &bundle.heuristics.get(heuristic_selector);
  }
  // Public-event extractor feeds two consumers:
  //   - per-perspective trackers (pp_trackers below) — observe events
  //     each ply for every seat.
  //   - trace_belief_tracker — records a specific perspective's
  //     snapshot for regression tests.
  // Must be populated whenever the game registers it, regardless of
  // trace_perspective. The tracker bootstrap snapshot is now produced
  // by the framework walker (make_masked_state) — no per-game extractor.
  runtime::PublicEventExtractor trace_extractor = bundle.public_event_extractor;
  // For tracing we need a SECOND bundle (and its belief_tracker) dedicated
  // to the traced perspective, separate from both the main tracker and the
  // pp_trackers.
  // Tracing only requires public_event_extractor — that's the source
  // of the per-ply event stream + public_snapshot. trace_belief_tracker
  // is optional: tracker-bearing games (Splendor / LL) get one for
  // belief_snapshot_after; snapshot-only games (Azul) trace without a
  // tracker (belief_snapshot_after stays empty).
  std::unique_ptr<GameBundle> trace_bundle;
  IBeliefTracker* trace_bt = nullptr;
  if (trace_perspective >= 0) {
    if (!trace_extractor) {
      py::gil_scoped_acquire acquire;
      throw std::runtime_error(
          "run_selfplay_episode: trace_perspective >= 0 but game '" + game_id +
          "' did not register a public_event_extractor");
    }
    trace_bundle = std::make_unique<GameBundle>(
        GameRegistry::instance().create_game(game_id, seed));
    trace_bt = trace_bundle->belief_tracker.get();  // may be nullptr
  }

  if (model_path.empty()) {
    throw std::invalid_argument("run_selfplay_episode: model_path must not be empty");
  }
  auto evaluator = std::make_unique<infer::OnnxPolicyValueEvaluator>(
      model_path, bundle.encoder.get());
  if (!evaluator->is_ready()) {
    throw std::runtime_error("run_selfplay_episode: failed to load model: " + evaluator->last_error());
  }
  const search::IPolicyValueEvaluator* eval_ptr = evaluator.get();
  // Single-model selfplay: factory returns the same evaluator for every seat.
  runtime::PolicyEvaluatorFactory eval_factory =
      [eval_ptr](int /*player*/) -> const search::IPolicyValueEvaluator& {
    return *eval_ptr;
  };

  // Belief plumbing: the extractor is always exposed when registered (so
  // emit can produce BeliefSamples even before a belief.onnx is trained).
  // The evaluator is only constructed when a model path is set; without
  // it, the tracker's prepare_for_root is a no-op and randomize_unseen
  // falls back to uniform sampling.
  const IBeliefFeatureExtractor* belief_extractor_ptr =
      bundle.belief_feature_extractor.get();
  auto belief_eval = create_belief_evaluator(bundle, "run_selfplay_episode");
  const int num_players_sp = bundle.state->num_players();
  // Single-episode selfplay: all seats share the same belief evaluator.
  std::vector<const IBeliefEvaluator*> per_seat_belief_evals_sp;
  if (belief_eval.ptr) {
    per_seat_belief_evals_sp.assign(
        static_cast<size_t>(num_players_sp), belief_eval.ptr);
  }

  auto cfg = build_selfplay_config(
      simulations, world_cap, c_puct, temperature,
      dirichlet_alpha, dirichlet_epsilon, dirichlet_on_first_n_plies,
      max_game_plies, tail_solve_enabled, tail_solve_depth_limit,
      tail_solve_node_budget, tail_solve_auxiliary_weight,
      temperature_initial, temperature_final, temperature_decay_plies,
      heuristic_guidance_ratio, heuristic_temperature,
      training_filter_ratio, opp_sel, safe_mode, top_k, fpu_reduction,
      uct2_arrival_visits);

  auto pp = create_per_seat_bundles(game_id, seed, num_players_sp);

  auto result = runtime::run_selfplay_episode(
      *bundle.state, *bundle.rules, *bundle.value_model, eval_factory, cfg, seed,
      pp.trackers,
      pp.states,
      bundle.public_state_applier,
      bundle.encoder.get(),
      bundle.tail_solver.get(),
      bundle.adjudicator,
      bundle.auxiliary_scorer,
      heuristic_ptr,
      bundle.training_action_filter,
      bundle.tail_solve_trigger,
      bundle.episode_stats_extractor,
      trace_perspective,
      trace_bt,
      trace_extractor,
      bundle.events_only_extractor,
      belief_extractor_ptr,
      per_seat_belief_evals_sp,
      bundle.belief_label_extractor.get());

  py::gil_scoped_acquire acquire;
  return result_to_py(result, bundle);
}

// Multi-episode batch binding. Loads evaluators (and optional belief
// evaluator) once, then runs one episode per seed in `seeds` and returns
// the resulting dicts under the `episodes` key.
//
// `model_paths` is a per-seat list of length num_players. Same-path
// entries are deduplicated so mirror selfplay (all seats == latest) only
// loads one evaluator.
//
// `latest_seat` (default -1 = mirror) names the seat carrying the latest
// network in pool selfplay. When >= 0 and the latest seat's path differs
// from the pool path, the binding wraps the latest seat's evaluator in a
// SplitPolicyValueEvaluator so that during the latest seat's MCTS:
//   - perspective_player == latest_seat: policy + value from latest
//   - perspective_player != latest_seat: policy from opponent network,
//                                        value from latest network
// Mirror tasks pass -1 and skip the wrapper entirely.
//
// Sample filtering by latest_seat is caller-side; this binding just
// returns one episode dict per seed in the same order.
py::dict run_selfplay_episodes_batch_py(
    const std::string& game_id,
    const std::vector<std::uint64_t>& seeds,
    const std::vector<std::string>& model_paths,
    int simulations,
    int world_cap,
    float c_puct,
    double temperature,
    double dirichlet_alpha,
    double dirichlet_epsilon,
    int dirichlet_on_first_n_plies,
    int max_game_plies,
    bool tail_solve_enabled,
    int tail_solve_depth_limit,
    std::int64_t tail_solve_node_budget,
    float tail_solve_auxiliary_weight,
    double temperature_initial,
    double temperature_final,
    int temperature_decay_plies,
    double heuristic_guidance_ratio,
    double heuristic_temperature,
    double training_filter_ratio,
    const std::string& opponent_selection,
    const std::string& belief_model_path,
    const std::string& heuristic_selector,
    int latest_seat,
    bool safe_mode,
    int top_k,
    const std::vector<std::string>& belief_model_paths,
    std::optional<float> fpu_reduction,
    bool uct2_arrival_visits) {
  const auto opp_sel = parse_opponent_selection(opponent_selection);
  py::gil_scoped_release release;

  if (seeds.empty()) {
    py::gil_scoped_acquire acquire;
    throw std::invalid_argument(
        "run_selfplay_episodes_batch: seeds must not be empty");
  }

  // Master bundle owns the encoder for the whole batch. Per-episode
  // bundles below carry their own state + rules but share the encoder
  // pointer through the evaluator.
  auto master = GameRegistry::instance().create_game(game_id, seeds[0]);
  if (!belief_model_path.empty()) master.belief_model_path = belief_model_path;
  const board_ai::IHeuristic* heuristic_ptr = nullptr;
  if (!heuristic_selector.empty() && master.heuristics.has(heuristic_selector)) {
    heuristic_ptr = &master.heuristics.get(heuristic_selector);
  }
  const int num_players = master.state->num_players();

  if (static_cast<int>(model_paths.size()) != num_players) {
    py::gil_scoped_acquire acquire;
    throw std::invalid_argument(
        "run_selfplay_episodes_batch: model_paths size " +
        std::to_string(model_paths.size()) + " != num_players " +
        std::to_string(num_players));
  }
  for (size_t i = 0; i < model_paths.size(); ++i) {
    if (model_paths[i].empty()) {
      py::gil_scoped_acquire acquire;
      throw std::invalid_argument(
          "run_selfplay_episodes_batch: model_paths[" + std::to_string(i) +
          "] must not be empty");
    }
  }

  // Dedup evaluators by model_path: mirror selfplay collapses N seats to
  // one ORT session.
  std::unordered_map<std::string, infer::OnnxPolicyValueEvaluator*> eval_by_path;
  std::vector<std::unique_ptr<infer::OnnxPolicyValueEvaluator>> evaluators;
  std::vector<const search::IPolicyValueEvaluator*> seat_eval_ptrs(num_players);
  for (int p = 0; p < num_players; ++p) {
    const auto& path = model_paths[p];
    auto it = eval_by_path.find(path);
    if (it == eval_by_path.end()) {
      auto ev = std::make_unique<infer::OnnxPolicyValueEvaluator>(
          path, master.encoder.get());
      if (!ev->is_ready()) {
        py::gil_scoped_acquire acquire;
        throw std::runtime_error(
            "run_selfplay_episodes_batch: failed to load model '" + path +
            "': " + ev->last_error());
      }
      eval_by_path.emplace(path, ev.get());
      seat_eval_ptrs[p] = ev.get();
      evaluators.push_back(std::move(ev));
    } else {
      seat_eval_ptrs[p] = it->second;
    }
  }
  // Pool selfplay split evaluator: when `latest_seat >= 0`, the latest
  // seat's MCTS uses opponent network for the policy at non-latest
  // perspective_player nodes but the latest network for value
  // everywhere. Mirror tasks (latest_seat < 0) skip this entirely and
  // use the dedup'd OnnxPolicyValueEvaluator directly.
  std::unique_ptr<infer::SplitPolicyValueEvaluator> split_eval;
  if (latest_seat >= 0) {
    if (latest_seat >= num_players) {
      py::gil_scoped_acquire acquire;
      throw std::invalid_argument(
          "run_selfplay_episodes_batch: latest_seat " +
          std::to_string(latest_seat) + " >= num_players " +
          std::to_string(num_players));
    }
    // Pick any non-latest seat as the opponent network; pool selfplay
    // ships one frozen pool path on every non-latest seat, so they all
    // dedup to the same evaluator.
    int opp_seat = -1;
    for (int p = 0; p < num_players; ++p) {
      if (p != latest_seat) { opp_seat = p; break; }
    }
    if (opp_seat >= 0 &&
        seat_eval_ptrs[opp_seat] != seat_eval_ptrs[latest_seat]) {
      split_eval = std::make_unique<infer::SplitPolicyValueEvaluator>(
          seat_eval_ptrs[latest_seat],
          seat_eval_ptrs[opp_seat],
          latest_seat);
      seat_eval_ptrs[latest_seat] = split_eval.get();
    }
  }
  runtime::PolicyEvaluatorFactory eval_factory =
      [&seat_eval_ptrs](int player) -> const search::IPolicyValueEvaluator& {
    return *seat_eval_ptrs[static_cast<size_t>(player)];
  };

  // Per-seat belief evaluators. Two modes:
  //   1. belief_model_paths empty (or all-empty): fall back to the master
  //      bundle's single belief_model_path — every seat shares one
  //      evaluator. This is the legacy mirror behavior.
  //   2. belief_model_paths non-empty: must be size num_players. Each
  //      non-empty entry loads its own OnnxBeliefEvaluator (dedup'd by
  //      path so seats sharing a path share a session). Empty entries
  //      → that seat falls back to uniform sampling.
  // The `belief_model_path` (singular) parameter remains as the master-
  // level override used to populate `bundle.belief_model_path` for
  // per-episode bundles below; seat routing is handled separately via
  // the per_seat_belief_evals_batch vector.
  std::vector<std::unique_ptr<infer::OnnxBeliefEvaluator>> belief_evals_owned;
  std::unordered_map<std::string, IBeliefEvaluator*> belief_eval_by_path;
  std::vector<const IBeliefEvaluator*> per_seat_belief_evals_batch;
  if (master.belief_feature_extractor) {
    bool any_per_seat = false;
    if (!belief_model_paths.empty()) {
      if (static_cast<int>(belief_model_paths.size()) != num_players) {
        py::gil_scoped_acquire acquire;
        throw std::invalid_argument(
            "run_selfplay_episodes_batch: belief_model_paths size " +
            std::to_string(belief_model_paths.size()) + " != num_players " +
            std::to_string(num_players));
      }
      for (const auto& bp : belief_model_paths) {
        if (!bp.empty()) { any_per_seat = true; break; }
      }
    }
    if (any_per_seat) {
      per_seat_belief_evals_batch.assign(
          static_cast<size_t>(num_players), nullptr);
      for (int p = 0; p < num_players; ++p) {
        const auto& bp = belief_model_paths[p];
        if (bp.empty()) continue;
        auto it = belief_eval_by_path.find(bp);
        if (it != belief_eval_by_path.end()) {
          per_seat_belief_evals_batch[p] = it->second;
          continue;
        }
        auto ev = std::make_unique<infer::OnnxBeliefEvaluator>(bp);
        if (!ev->is_ready()) {
          py::gil_scoped_acquire acquire;
          throw std::runtime_error(
              "run_selfplay_episodes_batch: failed to load belief model '" +
              bp + "' for seat " + std::to_string(p) + ": " +
              ev->last_error());
        }
        per_seat_belief_evals_batch[p] = ev.get();
        belief_eval_by_path.emplace(bp, ev.get());
        belief_evals_owned.push_back(std::move(ev));
      }
    } else {
      // Legacy path: single belief evaluator from master bundle, shared
      // across all seats.
      auto belief_eval = create_belief_evaluator(
          master, "run_selfplay_episodes_batch");
      if (belief_eval.ptr) {
        per_seat_belief_evals_batch.assign(
            static_cast<size_t>(num_players), belief_eval.ptr);
        belief_evals_owned.push_back(std::move(belief_eval.evaluator));
      }
    }
  }

  auto cfg = build_selfplay_config(
      simulations, world_cap, c_puct, temperature,
      dirichlet_alpha, dirichlet_epsilon, dirichlet_on_first_n_plies,
      max_game_plies, tail_solve_enabled, tail_solve_depth_limit,
      tail_solve_node_budget, tail_solve_auxiliary_weight,
      temperature_initial, temperature_final, temperature_decay_plies,
      heuristic_guidance_ratio, heuristic_temperature,
      training_filter_ratio, opp_sel, safe_mode, top_k, fpu_reduction,
      uct2_arrival_visits);

  std::vector<runtime::SelfplayEpisodeResult> results;
  results.reserve(seeds.size());
  for (std::uint64_t seed : seeds) {
    auto bundle = GameRegistry::instance().create_game(game_id, seed);
    if (!belief_model_path.empty()) bundle.belief_model_path = belief_model_path;

    auto pp = create_per_seat_bundles(game_id, seed, num_players);

    auto result = runtime::run_selfplay_episode(
        *bundle.state, *bundle.rules, *bundle.value_model, eval_factory, cfg, seed,
        pp.trackers,
        pp.states,
        bundle.public_state_applier,
        bundle.encoder.get(),
        bundle.tail_solver.get(),
        bundle.adjudicator,
        bundle.auxiliary_scorer,
        heuristic_ptr,
        bundle.training_action_filter,
        bundle.tail_solve_trigger,
        bundle.episode_stats_extractor,
        /*trace_perspective=*/-1,
        /*trace_belief_tracker=*/nullptr,
        bundle.public_event_extractor,
        bundle.events_only_extractor,
        bundle.belief_feature_extractor.get(),
        per_seat_belief_evals_batch,
        bundle.belief_label_extractor.get());
    results.push_back(std::move(result));
  }

  py::gil_scoped_acquire acquire;
  py::list episodes;
  for (auto& r : results) {
    episodes.append(result_to_py(r, master));
  }
  py::dict out;
  out["episodes"] = episodes;
  return out;
}

// run_concurrent_selfplay_step — Phase 1 of PARALLEL_MCTS_VLOSS_PLAN.
//
// Spawns (num_self_threads + num_pool_threads) OS sampling threads sharing
// a BatchedEvaluator pool. Each thread has a fixed mode (self / pool); all
// threads pull seeds from a single shared counter (work-stealing). When a
// thread grabs a seed, it runs one episode in its own mode:
//   - self thread: every seat uses `latest_model_path` (mirror selfplay)
//   - pool thread: pick a random latest_seat + a random opp from
//     `pool_paths` (deduped to opponent_pool_max_distinct entries),
//     then run with mixed per-seat models
//
// Both modes feed leaf evals into the shared EvaluatorPool — `latest`
// evaluator (one instance) plus up to `opponent_pool_max_distinct` opp
// evaluators created on demand. Pool thread's split-evaluator setup
// (latest seat sees opponent policy at non-latest perspective_player) is
// applied via SplitPolicyValueEvaluator like run_selfplay_episodes_batch
// does.
//
// `num_self_threads + num_pool_threads`: total sampling threads. Caller
// computes from `cli --workers` and `opponent_pool_self_ratio`.
//
// Returns: { "episodes": [episode_dict, ...] } where each ep dict carries
// `latest_seat: -1` for mirror runs and `latest_seat >= 0` for pool runs
// (matching run_selfplay_episodes_batch's labeling for downstream sample
// filtering).
py::dict run_concurrent_selfplay_step_py(
    const std::string& game_id,
    const std::vector<std::uint64_t>& seeds,
    const std::string& latest_model_path,
    const std::vector<std::string>& pool_paths,
    int num_self_threads,
    int num_pool_threads,
    int opponent_pool_max_distinct,
    int simulations,
    int world_cap,
    float c_puct,
    double temperature,
    double dirichlet_alpha,
    double dirichlet_epsilon,
    int dirichlet_on_first_n_plies,
    int max_game_plies,
    bool tail_solve_enabled,
    int tail_solve_depth_limit,
    std::int64_t tail_solve_node_budget,
    float tail_solve_auxiliary_weight,
    double temperature_initial,
    double temperature_final,
    int temperature_decay_plies,
    double heuristic_guidance_ratio,
    double heuristic_temperature,
    const std::string& heuristic_selector,
    double training_filter_ratio,
    const std::string& opponent_selection,
    const std::string& belief_model_path,
    std::size_t batch_size,
    std::size_t batch_timeout_us,
    std::size_t max_queue_size,
    int intra_op_num_threads,
    int num_inference_threads,
    const std::vector<std::string>& providers,
    const std::vector<int>& devices,
    bool safe_mode,
    int top_k,
    const std::vector<std::string>& pool_belief_paths,
    int leaf_batch_size,
    int fast_simulations,
    double full_search_prob,
    std::optional<float> fpu_reduction,
    bool compact_samples,
    bool uct2_arrival_visits) {
  const auto opp_sel = parse_opponent_selection(opponent_selection);
  py::gil_scoped_release release;

  if (seeds.empty()) {
    py::gil_scoped_acquire acquire;
    throw std::invalid_argument(
        "run_concurrent_selfplay_step: seeds must not be empty");
  }
  if (leaf_batch_size < 1) {
    py::gil_scoped_acquire acquire;
    throw std::invalid_argument(
        "run_concurrent_selfplay_step: leaf_batch_size must be >= 1");
  }
  const std::size_t total_threads_for_queue =
      static_cast<std::size_t>(num_self_threads + num_pool_threads);
  if (max_queue_size < total_threads_for_queue * static_cast<std::size_t>(leaf_batch_size)) {
    py::gil_scoped_acquire acquire;
    throw std::invalid_argument(
        "run_concurrent_selfplay_step: max_queue_size (" + std::to_string(max_queue_size) +
        ") < threads (" + std::to_string(total_threads_for_queue) + ") * leaf_batch_size (" +
        std::to_string(leaf_batch_size) + "); the evaluator queue would overflow");
  }
  if (num_self_threads < 0 || num_pool_threads < 0 ||
      num_self_threads + num_pool_threads <= 0) {
    py::gil_scoped_acquire acquire;
    throw std::invalid_argument(
        "run_concurrent_selfplay_step: num_self_threads + num_pool_threads must be > 0");
  }
  if (latest_model_path.empty()) {
    py::gil_scoped_acquire acquire;
    throw std::invalid_argument(
        "run_concurrent_selfplay_step: latest_model_path must not be empty");
  }
  if (num_pool_threads > 0 && pool_paths.empty()) {
    py::gil_scoped_acquire acquire;
    throw std::invalid_argument(
        "run_concurrent_selfplay_step: num_pool_threads > 0 but pool_paths empty");
  }
  if (opponent_pool_max_distinct < 1) {
    py::gil_scoped_acquire acquire;
    throw std::invalid_argument(
        "run_concurrent_selfplay_step: opponent_pool_max_distinct must be >= 1");
  }

  // Master bundle owns the encoder + bundle metadata for the whole step.
  auto master = GameRegistry::instance().create_game(game_id, seeds[0]);
  if (!belief_model_path.empty()) master.belief_model_path = belief_model_path;
  const int num_players = master.state->num_players();

  // Pick the actual opponent set used this step: dedup pool_paths, then
  // draw opponent_pool_max_distinct of them uniformly without replacement
  // (seeded by seeds[0], so different steps cycle through the whole pool,
  // matching the legacy path's uniform rng.choice over pool_paths).
  // active_opp_belief_paths[i] is the belief model path paired with
  // active_opp_paths[i]. Empty string means "no belief model for that
  // opponent" (falls back to uniform sampling).
  std::vector<std::string> active_opp_paths;
  std::vector<std::string> active_opp_belief_paths;
  {
    std::vector<size_t> candidates;
    std::unordered_set<std::string> seen;
    for (size_t pi = 0; pi < pool_paths.size(); ++pi) {
      const auto& p = pool_paths[pi];
      if (p.empty() || p == latest_model_path) continue;
      if (seen.insert(p).second) candidates.push_back(pi);
    }
    std::mt19937_64 pick_rng(board_ai::rng::derive_subseed(
        seeds[0], "concurrent_selfplay.opponent_pool"));
    const size_t take = std::min(
        candidates.size(), static_cast<size_t>(opponent_pool_max_distinct));
    for (size_t k = 0; k < take; ++k) {
      std::uniform_int_distribution<size_t> pick(k, candidates.size() - 1);
      std::swap(candidates[k], candidates[pick(pick_rng)]);
    }
    candidates.resize(take);
    std::sort(candidates.begin(), candidates.end());
    for (const size_t pi : candidates) {
      active_opp_paths.push_back(pool_paths[pi]);
      // Pair with belief path if pool_belief_paths is long enough.
      active_opp_belief_paths.push_back(
          pi < pool_belief_paths.size() ? pool_belief_paths[pi] : std::string());
    }
  }

  // Construct shared evaluator pool. Build latest + active opp evaluators
  // up front so failures surface before threads spin up.
  infer::EvaluatorPool eval_pool;
  infer::BatchedEvaluatorConfig batched_cfg{};
  batched_cfg.batch_size = batch_size;
  batched_cfg.batch_timeout_us = batch_timeout_us;
  batched_cfg.max_queue_size = max_queue_size;
  batched_cfg.intra_op_num_threads = intra_op_num_threads;
  batched_cfg.num_inference_threads = num_inference_threads;
  batched_cfg.inter_op_num_threads = 1;
  batched_cfg.providers = providers;
  batched_cfg.devices = devices;
  search::IPolicyValueEvaluator* latest_eval = nullptr;
  std::vector<search::IPolicyValueEvaluator*> opp_evals;
  try {
    latest_eval = eval_pool.get_or_create(
        latest_model_path, master.encoder.get(), batched_cfg);
    for (const auto& p : active_opp_paths) {
      opp_evals.push_back(eval_pool.get_or_create(p, master.encoder.get(), batched_cfg));
    }
  } catch (const std::exception& e) {
    py::gil_scoped_acquire acquire;
    throw std::runtime_error(
        std::string("run_concurrent_selfplay_step: evaluator creation failed: ") + e.what());
  }

  // mcts_pv guidance heuristic evaluator (Phase 2 follow-up): when the
  // selected guidance heuristic is a mcts_pv with model_path, the
  // per-bundle OnnxPolicyValueEvaluator is replaced inside each worker
  // thread's bundle.heuristics with this shared BatchedEvaluator so all
  // sampling threads' guidance MCTS calls batch together. heuristic types
  // without model_path (rule, hand-written PV) are left untouched.
  search::IPolicyValueEvaluator* guidance_eval = nullptr;
  std::string guidance_model_path;
  if (heuristic_guidance_ratio > 0.0 && !heuristic_selector.empty()
      && master.heuristics.has(heuristic_selector)) {
    auto* h = const_cast<board_ai::IHeuristic*>(
        &master.heuristics.get(heuristic_selector));
    if (auto* pvh = dynamic_cast<board_ai::heuristic::MctsPVHeuristic*>(h)) {
      const std::string& mp = pvh->model_path_hint();
      if (!mp.empty()) {
        if (max_queue_size < total_threads_for_queue *
                                 static_cast<std::size_t>(pvh->leaf_batch_size())) {
          py::gil_scoped_acquire acquire;
          throw std::invalid_argument(
              "run_concurrent_selfplay_step: max_queue_size (" +
              std::to_string(max_queue_size) + ") < threads (" +
              std::to_string(total_threads_for_queue) + ") * guidance heuristic '" +
              heuristic_selector + "' leaf_batch_size (" +
              std::to_string(pvh->leaf_batch_size()) +
              "); the evaluator queue would overflow");
        }
        try {
          guidance_eval = eval_pool.get_or_create(
              mp, pvh->encoder_hint(), batched_cfg);
          guidance_model_path = mp;
        } catch (const std::exception& e) {
          py::gil_scoped_acquire acquire;
          throw std::runtime_error(
              std::string("run_concurrent_selfplay_step: guidance evaluator "
                          "creation failed for '") + mp + "': " + e.what());
        }
      }
    }
  }

  // Per-pool-opp split evaluator: when a pool seat (non-latest) uses opp_X,
  // the latest seat's MCTS still wants `latest` for value but `opp_X`'s
  // policy at non-latest perspective_player nodes (matches
  // run_selfplay_episodes_batch's split semantics). Pre-build one split
  // evaluator per opp.
  std::vector<std::unique_ptr<infer::SplitPolicyValueEvaluator>> split_evals;
  // split_eval_for_opp[i] = split(latest, opp_evals[i], latest_seat=*)
  // We don't bind a specific latest_seat here because each pool episode
  // chooses its own; SplitPolicyValueEvaluator's latest_seat parameter is
  // baked into the constructor though, so we'll build one *per (opp, seat)*
  // lazily inside the worker. For now just keep opp evaluators ready.

  // Belief evaluator: BatchedBeliefEvaluator so all sampling threads
  // share one queue + EP path (CPU or CoreML matching the PV evaluators).
  std::unique_ptr<infer::BatchedBeliefEvaluator> belief_evaluator;
  const IBeliefEvaluator* belief_evaluator_ptr = nullptr;
  if (master.belief_feature_extractor && !master.belief_model_path.empty()) {
    try {
      belief_evaluator = std::make_unique<infer::BatchedBeliefEvaluator>(
          master.belief_model_path, batched_cfg);
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw std::runtime_error(
          "run_concurrent_selfplay_step: failed to load belief model '" +
          master.belief_model_path + "': " + e.what());
    }
    belief_evaluator_ptr = belief_evaluator.get();
  }

  // Pool-opponent belief evaluators: one BatchedBeliefEvaluator per distinct
  // belief_model_path across pool opponents. Dedup'd so two opponents
  // sharing the same belief.onnx share one evaluator.
  std::unordered_map<std::string, std::unique_ptr<infer::BatchedBeliefEvaluator>>
      pool_belief_evaluators;
  // opp_belief_eval_ptrs[i] = evaluator for active_opp_belief_paths[i],
  // nullptr if that opponent has no belief model.
  std::vector<const IBeliefEvaluator*> opp_belief_eval_ptrs(
      active_opp_belief_paths.size(), nullptr);
  if (master.belief_feature_extractor) {
    for (size_t i = 0; i < active_opp_belief_paths.size(); ++i) {
      const auto& bp = active_opp_belief_paths[i];
      if (bp.empty()) continue;
      // If it's the same as the latest belief, reuse that evaluator.
      if (belief_evaluator && bp == master.belief_model_path) {
        opp_belief_eval_ptrs[i] = belief_evaluator_ptr;
        continue;
      }
      auto it = pool_belief_evaluators.find(bp);
      if (it != pool_belief_evaluators.end()) {
        opp_belief_eval_ptrs[i] = it->second.get();
      } else {
        try {
          auto be = std::make_unique<infer::BatchedBeliefEvaluator>(
              bp, batched_cfg);
          opp_belief_eval_ptrs[i] = be.get();
          pool_belief_evaluators.emplace(bp, std::move(be));
        } catch (const std::exception& e) {
          py::gil_scoped_acquire acquire;
          throw std::runtime_error(
              "run_concurrent_selfplay_step: failed to load pool belief model '" +
              bp + "': " + e.what());
        }
      }
    }
  }

  auto cfg = build_selfplay_config(
      simulations, world_cap, c_puct, temperature,
      dirichlet_alpha, dirichlet_epsilon, dirichlet_on_first_n_plies,
      max_game_plies, tail_solve_enabled, tail_solve_depth_limit,
      tail_solve_node_budget, tail_solve_auxiliary_weight,
      temperature_initial, temperature_final, temperature_decay_plies,
      heuristic_guidance_ratio, heuristic_temperature,
      training_filter_ratio, opp_sel, safe_mode, top_k, fpu_reduction,
      uct2_arrival_visits);
  cfg.leaf_batch_size = leaf_batch_size;
  cfg.fast_simulations = fast_simulations;
  cfg.full_search_prob = full_search_prob;
  cfg.encode_fast_samples = !compact_samples;

  // ------------------------------------------------------------
  // Thread pool: shared seed counter (work-stealing). Each thread fixed in
  // self/pool mode; whoever grabs a seed runs that mode. Self/pool ratio
  // emerges naturally from thread speed × thread count (faster mode runs
  // more episodes — matches plan v0.7 §1.5.2).
  // ------------------------------------------------------------
  const std::size_t total_episodes = seeds.size();
  std::vector<runtime::SelfplayEpisodeResult> results(total_episodes);
  std::vector<int> result_latest_seat(total_episodes, -1);   // -1 = self/mirror
  std::vector<std::exception_ptr> errors(total_episodes);
  std::atomic<std::size_t> next_idx{0};

  // Per-thread (deterministic) RNGs for pool-mode opp/seat selection.
  // Seeded from seeds[0] + thread_id so reruns of the same step replay
  // identical pool draws.
  auto run_one_episode =
      [&](std::size_t idx, int thread_id, bool is_pool) {
    const std::uint64_t seed = seeds[idx];
    auto bundle = GameRegistry::instance().create_game(game_id, seed);
    if (!belief_model_path.empty()) bundle.belief_model_path = belief_model_path;

    // mcts_pv guidance heuristic hot-swap: replace per-bundle ONNX session
    // with the shared BatchedEvaluator from the pool, so all sampling
    // threads' guidance MCTS calls feed one batched queue. No-op when
    // guidance heuristic isn't a mcts_pv with model_path, or when the
    // ratio is 0 (selfplay didn't load it).
    board_ai::IHeuristic* heuristic_ptr = nullptr;
    if (heuristic_guidance_ratio > 0.0 && !heuristic_selector.empty()
        && bundle.heuristics.has(heuristic_selector)) {
      heuristic_ptr = const_cast<board_ai::IHeuristic*>(
          &bundle.heuristics.get(heuristic_selector));
      if (guidance_eval) {
        if (auto* pvh = dynamic_cast<board_ai::heuristic::MctsPVHeuristic*>(
                heuristic_ptr)) {
          pvh->replace_evaluator(guidance_eval);
        }
      }
    }

    auto pp = create_per_seat_bundles(game_id, seed, num_players);

    // Decide per-seat evaluator routing.
    // Mirror: every seat uses latest_eval.
    // Pool: pick latest_seat + opp from active_opp_paths via thread-local
    //       RNG seeded by (seeds[0], thread_id, idx). Wrap latest seat's
    //       evaluator in SplitPolicyValueEvaluator so its MCTS uses opp
    //       policy at non-latest perspective.
    int chosen_latest_seat = -1;
    std::vector<const search::IPolicyValueEvaluator*> seat_eval_ptrs(
        static_cast<size_t>(num_players), latest_eval);

    // Per-seat belief evaluator vector. Mirror: all seats use latest.
    // Pool: latest seat uses latest, opponent seats use paired belief.
    std::vector<const IBeliefEvaluator*> per_seat_belief_evals;
    if (belief_evaluator_ptr) {
      per_seat_belief_evals.assign(
          static_cast<size_t>(num_players), belief_evaluator_ptr);
    }

    if (is_pool && !opp_evals.empty()) {
      std::mt19937_64 thread_rng(
          board_ai::rng::derive_subseed(
              seeds[0], "concurrent_selfplay.pool_dispatch",
              static_cast<std::uint64_t>(thread_id) * 1009ULL + idx));
      std::uniform_int_distribution<int> seat_dist(0, num_players - 1);
      std::uniform_int_distribution<size_t> opp_dist(0, opp_evals.size() - 1);
      chosen_latest_seat = seat_dist(thread_rng);
      const size_t chosen_opp_idx = opp_dist(thread_rng);
      auto* opp_eval = opp_evals[chosen_opp_idx];
      // No split evaluator: latest seat uses latest, opp seats use opp.
      // SplitPolicyValueEvaluator's "opp policy + latest value" hybrid is
      // a Phase 1.x optimization; we keep it off by default because it
      // doubles eval calls per leaf — a serious cost under shared-queue
      // batched inference. Re-enable later if measurements warrant it.
      for (int p = 0; p < num_players; ++p) {
        seat_eval_ptrs[static_cast<size_t>(p)] =
            (p == chosen_latest_seat) ? latest_eval : opp_eval;
      }

      // Per-seat belief: latest seat keeps latest belief, opponent seats
      // use the belief model paired with this opponent's checkpoint.
      const IBeliefEvaluator* opp_belief =
          (chosen_opp_idx < opp_belief_eval_ptrs.size())
              ? opp_belief_eval_ptrs[chosen_opp_idx]
              : nullptr;
      if (belief_evaluator_ptr || opp_belief) {
        if (per_seat_belief_evals.empty()) {
          per_seat_belief_evals.assign(
              static_cast<size_t>(num_players), nullptr);
        }
        for (int p = 0; p < num_players; ++p) {
          per_seat_belief_evals[static_cast<size_t>(p)] =
              (p == chosen_latest_seat) ? belief_evaluator_ptr : opp_belief;
        }
      }
    }

    runtime::PolicyEvaluatorFactory eval_factory =
        [&seat_eval_ptrs](int player) -> const search::IPolicyValueEvaluator& {
      return *seat_eval_ptrs[static_cast<size_t>(player)];
    };

    results[idx] = runtime::run_selfplay_episode(
        *bundle.state, *bundle.rules, *bundle.value_model, eval_factory,
        cfg, seed,
        pp.trackers,
        pp.states,
        bundle.public_state_applier,
        bundle.encoder.get(),
        bundle.tail_solver.get(),
        bundle.adjudicator,
        bundle.auxiliary_scorer,
        heuristic_ptr,
        bundle.training_action_filter,
        bundle.tail_solve_trigger,
        bundle.episode_stats_extractor,
        /*trace_perspective=*/-1,
        /*trace_belief_tracker=*/nullptr,
        bundle.public_event_extractor,
        bundle.events_only_extractor,
        bundle.belief_feature_extractor.get(),
        per_seat_belief_evals,
        bundle.belief_label_extractor.get());
    result_latest_seat[idx] = chosen_latest_seat;
  };

  auto worker_fn = [&](int thread_id, bool is_pool) {
    while (true) {
      const std::size_t idx = next_idx.fetch_add(1, std::memory_order_relaxed);
      if (idx >= total_episodes) break;
      try {
        run_one_episode(idx, thread_id, is_pool);
      } catch (...) {
        errors[idx] = std::current_exception();
      }
    }
  };

  std::vector<std::thread> threads;
  const int total_threads = num_self_threads + num_pool_threads;
  threads.reserve(static_cast<size_t>(total_threads));
  // Pool threads with empty pool_paths fall back to mirror (treat as self).
  const bool effective_pool = num_pool_threads > 0 && !opp_evals.empty();
  for (int t = 0; t < num_self_threads; ++t) {
    threads.emplace_back(worker_fn, t, /*is_pool=*/false);
  }
  for (int t = 0; t < num_pool_threads; ++t) {
    threads.emplace_back(worker_fn, num_self_threads + t,
                         /*is_pool=*/effective_pool);
  }
  for (auto& th : threads) th.join();

  for (std::size_t i = 0; i < total_episodes; ++i) {
    if (errors[i]) {
      py::gil_scoped_acquire acquire;
      std::rethrow_exception(errors[i]);
    }
  }

  py::gil_scoped_acquire acquire;
  py::list episodes;
  for (std::size_t i = 0; i < results.size(); ++i) {
    py::dict ep = result_to_py(results[i], master, compact_samples);
    results[i] = {};
    ep["latest_seat"] = result_latest_seat[i];
    episodes.append(ep);
  }
  py::dict out;
  out["episodes"] = episodes;
  py::dict stats;
  stats["num_self_threads"] = num_self_threads;
  stats["num_pool_threads"] = num_pool_threads;
  stats["active_opp_paths"] = active_opp_paths;
  stats["total_episodes"] = static_cast<int>(total_episodes);
  out["stats"] = stats;
  return out;
}

py::dict run_arena_match_py(
    const std::string& game_id,
    std::uint64_t seed,
    const std::vector<std::string>& model_paths,
    const std::vector<int>& simulations_list,
    const std::vector<int>& world_cap_list,
    double temperature,
    int max_game_plies,
    bool tail_solve,
    int tail_solve_depth_limit,
    std::int64_t tail_solve_node_budget,
    float tail_solve_auxiliary_weight,
    const std::vector<std::string>& opponent_selection_list,
    const std::string& belief_model_path,
    const std::vector<py::dict>& seat_configs,
    bool safe_mode,
    int top_k) {
  // seat_configs is the per-seat "AI configuration" form. When non-empty
  // it is the authoritative source for every per-seat MCTS knob (model_path,
  // simulations, world_cap, temperature, opponent_selection, tail_solve_*).
  // When empty, fall back to the flat-parameter form for backward compat.
  const bool use_seat_configs = !seat_configs.empty();

  // Resolve effective per-seat knobs into stable C++ vectors before
  // releasing the GIL (we cannot touch py::dict without the GIL).
  const size_t n_seats = use_seat_configs ? seat_configs.size() : model_paths.size();
  if (n_seats == 0) {
    throw std::invalid_argument(
        "run_arena_match: seat_configs and model_paths both empty");
  }

  std::vector<std::string> sc_model_paths(n_seats);
  std::vector<int> sc_simulations(n_seats, 200);
  std::vector<int> sc_world_cap(n_seats, 0);
  std::vector<double> sc_temperature(n_seats, temperature);
  std::vector<std::string> sc_opponent_selection(n_seats, "puct");
  std::vector<int> sc_tail_solve(n_seats, 0);
  std::vector<int> sc_ts_depth(n_seats, tail_solve_depth_limit);
  std::vector<std::int64_t> sc_ts_budget(n_seats, tail_solve_node_budget);
  std::vector<float> sc_ts_aux(n_seats, tail_solve_auxiliary_weight);
  std::vector<float> sc_c_puct(n_seats, 1.4f);
  std::vector<std::optional<float>> sc_fpu_reduction(n_seats);
  std::vector<bool> sc_uct2_arrival_visits(n_seats, true);
  std::vector<bool> sc_safe_mode(n_seats, safe_mode);
  std::vector<int> sc_top_k(n_seats, top_k);
  // Per-seat belief model paths: extracted from seat_configs before GIL
  // release. Falls back to the top-level belief_model_path.
  std::vector<std::string> sc_belief_paths(n_seats, belief_model_path);

  if (use_seat_configs) {
    for (size_t i = 0; i < n_seats; ++i) {
      const py::dict& d = seat_configs[i];
      if (!d.contains("model_path")) {
        throw std::invalid_argument(
            "run_arena_match: seat_configs[" + std::to_string(i) +
            "] missing required 'model_path'");
      }
      sc_model_paths[i] = d["model_path"].cast<std::string>();
      if (d.contains("simulations")) sc_simulations[i] = d["simulations"].cast<int>();
      if (d.contains("world_cap")) sc_world_cap[i] = d["world_cap"].cast<int>();
      if (d.contains("c_puct")) sc_c_puct[i] = d["c_puct"].cast<float>();
      if (d.contains("fpu_reduction")) sc_fpu_reduction[i] = d["fpu_reduction"].cast<std::optional<float>>();
      if (d.contains("uct2_arrival_visits")) sc_uct2_arrival_visits[i] = require_bool(d["uct2_arrival_visits"], "uct2_arrival_visits");
      if (d.contains("temperature")) sc_temperature[i] = d["temperature"].cast<double>();
      if (d.contains("opponent_selection")) sc_opponent_selection[i] = d["opponent_selection"].cast<std::string>();
      if (d.contains("tail_solve_enabled")) sc_tail_solve[i] = d["tail_solve_enabled"].cast<bool>() ? 1 : 0;
      if (d.contains("tail_solve_depth_limit")) sc_ts_depth[i] = d["tail_solve_depth_limit"].cast<int>();
      if (d.contains("tail_solve_node_budget")) sc_ts_budget[i] = d["tail_solve_node_budget"].cast<std::int64_t>();
      if (d.contains("tail_solve_auxiliary_weight")) sc_ts_aux[i] = d["tail_solve_auxiliary_weight"].cast<float>();
      if (d.contains("safe_mode")) sc_safe_mode[i] = d["safe_mode"].cast<bool>();
      if (d.contains("top_k")) sc_top_k[i] = d["top_k"].cast<int>();
      if (d.contains("belief_model_path")) sc_belief_paths[i] = d["belief_model_path"].cast<std::string>();
    }
  } else {
    if (!opponent_selection_list.empty() &&
        opponent_selection_list.size() != model_paths.size()) {
      throw std::invalid_argument(
          "run_arena_match: opponent_selection_list size " +
          std::to_string(opponent_selection_list.size()) +
          " != model_paths size " + std::to_string(model_paths.size()));
    }
    for (size_t i = 0; i < n_seats; ++i) {
      sc_model_paths[i] = model_paths[i];
      if (i < simulations_list.size()) sc_simulations[i] = simulations_list[i];
      if (i < world_cap_list.size()) sc_world_cap[i] = world_cap_list[i];
      if (i < opponent_selection_list.size()) sc_opponent_selection[i] = opponent_selection_list[i];
      sc_tail_solve[i] = tail_solve ? 1 : 0;
      sc_safe_mode[i] = safe_mode;
    }
  }
  py::gil_scoped_release release;

  std::vector<search::OpponentSelection> opp_sel_per_player;
  opp_sel_per_player.reserve(n_seats);
  for (const auto& s : sc_opponent_selection) {
    opp_sel_per_player.push_back(parse_opponent_selection(s));
  }

  auto bundle = GameRegistry::instance().create_game(game_id, seed);
  if (!belief_model_path.empty()) bundle.belief_model_path = belief_model_path;

  for (size_t i = 0; i < n_seats; ++i) {
    if (sc_model_paths[i].empty()) {
      py::gil_scoped_acquire acquire;
      throw std::invalid_argument(
          "run_arena_match: seat " + std::to_string(i) + " model_path must not be empty");
    }
  }

  std::vector<std::unique_ptr<infer::OnnxPolicyValueEvaluator>> evaluators;
  std::vector<const search::IPolicyValueEvaluator*> eval_ptrs;
  for (size_t i = 0; i < n_seats; ++i) {
    auto ev = std::make_unique<infer::OnnxPolicyValueEvaluator>(
        sc_model_paths[i], bundle.encoder.get());
    if (!ev->is_ready()) {
      py::gil_scoped_acquire acquire;
      throw std::runtime_error(
          "run_arena_match: failed to load model_" + std::to_string(i) + ": " + ev->last_error());
    }
    eval_ptrs.push_back(ev.get());
    evaluators.push_back(std::move(ev));
  }

  std::vector<runtime::ArenaPlayerConfig> player_configs;
  for (size_t i = 0; i < n_seats; ++i) {
    runtime::ArenaPlayerConfig cfg{};
    cfg.simulations = sc_simulations[i];
    cfg.world_cap = sc_world_cap[i];
    cfg.c_puct = sc_c_puct[i];
    cfg.fpu_reduction = sc_fpu_reduction[i];
    cfg.uct2_arrival_visits = sc_uct2_arrival_visits[i];
    cfg.temperature = sc_temperature[i];
    cfg.opponent_selection = opp_sel_per_player[i];
    cfg.safe_mode = sc_safe_mode[i];
    cfg.top_k = sc_top_k[i];
    if (sc_tail_solve[i]) {
      if (!bundle.tail_solver) {
        py::gil_scoped_acquire acquire;
        throw std::invalid_argument(
            "run_arena_match: tail_solve_enabled=true but game " + game_id +
            " has no tail_solver registered.");
      }
      if (!bundle.tail_solve_trigger) {
        py::gil_scoped_acquire acquire;
        throw std::invalid_argument(
            "run_arena_match: tail_solve_enabled=true but game " + game_id +
            " has no tail_solve_trigger registered.");
      }
      cfg.tail_solve_enabled = true;
      cfg.tail_solve_config.depth_limit = sc_ts_depth[i];
      cfg.tail_solve_config.node_budget = sc_ts_budget[i];
      cfg.tail_solve_config.auxiliary_weight = sc_ts_aux[i];
      cfg.tail_solver = bundle.tail_solver.get();
      cfg.tail_solve_trigger = bundle.tail_solve_trigger;
    }
    player_configs.push_back(cfg);
  }

  IBeliefTracker* arena_bt = bundle.belief_tracker.get();
  const size_t n_eval = eval_ptrs.size();

  // Per-seat belief evaluators: dedup by path so seats sharing the same
  // belief model share one ORT session.
  std::unordered_map<std::string, std::unique_ptr<infer::OnnxBeliefEvaluator>>
      belief_eval_by_path;
  std::vector<const IBeliefEvaluator*> per_seat_belief_evals(n_seats, nullptr);
  const IBeliefFeatureExtractor* belief_extractor_ptr = nullptr;
  if (bundle.belief_feature_extractor) {
    for (size_t i = 0; i < n_seats; ++i) {
      const auto& bp = sc_belief_paths[i];
      if (bp.empty()) continue;
      auto it = belief_eval_by_path.find(bp);
      if (it != belief_eval_by_path.end()) {
        per_seat_belief_evals[i] = it->second.get();
      } else {
        auto ev = std::make_unique<infer::OnnxBeliefEvaluator>(bp);
        if (!ev->is_ready()) {
          py::gil_scoped_acquire acquire;
          throw std::runtime_error(
              "run_arena_match: failed to load belief model '" + bp +
              "' for seat " + std::to_string(i) + ": " + ev->last_error());
        }
        per_seat_belief_evals[i] = ev.get();
        belief_eval_by_path.emplace(bp, std::move(ev));
      }
    }
    // Activate belief_extractor if at least one seat has a belief evaluator.
    for (size_t i = 0; i < n_seats; ++i) {
      if (per_seat_belief_evals[i]) {
        belief_extractor_ptr = bundle.belief_feature_extractor.get();
        break;
      }
    }
  }

  auto pp = create_per_seat_bundles(game_id, seed, bundle.state->num_players());

  auto result = runtime::run_arena_match(
      *bundle.state, *bundle.rules, *bundle.value_model,
      [&eval_ptrs, n_eval](int player) -> const search::IPolicyValueEvaluator& {
        return *eval_ptrs[static_cast<size_t>(player) % n_eval];
      },
      player_configs, max_game_plies, seed,
      arena_bt, bundle.adjudicator,
      bundle.public_event_extractor,
      pp.trackers,
      pp.states,
      bundle.public_state_applier,
      bundle.events_only_extractor,
      belief_extractor_ptr,
      per_seat_belief_evals);

  py::gil_scoped_acquire acquire;
  py::dict out;
  out["winner"] = result.winner;
  out["draw"] = result.draw;
  out["terminal_z_values"] = result.terminal_z_values;
  out["total_plies"] = result.total_plies;
  py::list actions;
  for (auto a : result.action_history) actions.append(static_cast<int>(a));
  out["action_history"] = actions;
  py::list ply_stats;
  for (const auto& ps : result.ply_stats) {
    py::dict d;
    d["tail_solved"] = ps.tail_solved;
    d["tail_solve_value"] = ps.tail_solve_value;
    ply_stats.append(d);
  }
  out["ply_stats"] = ply_stats;
  return out;
}

// run_concurrent_arena_batch — Phase 2 batched/threaded version of
// run_arena_match for tools/eval_model.py. Spawns num_threads OS threads,
// each pulling tasks from a shared atomic counter. Tasks share a single
// EvaluatorPool so all distinct model_paths across all tasks dedup to one
// BatchedPolicyValueEvaluator each. providers (cpu / coreml / cuda) lets
// the entire batch hit GPU.
//
// Each task is a py::dict with the same shape run_arena_match accepts
// (seat_configs + seed + max_game_plies + belief_model_path), but the
// evaluator construction is hoisted into the shared pool here.
//
// Returns: { "results": [{winner, draw, total_plies, action_history, ...},
// ...], "stats": {...} } — same per-game shape as run_arena_match.
py::dict run_concurrent_arena_batch_py(
    const std::string& game_id,
    const std::vector<py::dict>& tasks,
    int num_threads,
    int max_game_plies,
    std::size_t batch_size,
    std::size_t batch_timeout_us,
    std::size_t max_queue_size,
    int intra_op_num_threads,
    int num_inference_threads,
    const std::vector<std::string>& providers,
    const std::vector<int>& devices,
    bool safe_mode) {
  if (tasks.empty()) {
    throw std::invalid_argument(
        "run_concurrent_arena_batch: tasks must not be empty");
  }
  if (num_threads <= 0) {
    throw std::invalid_argument(
        "run_concurrent_arena_batch: num_threads must be > 0");
  }

  // Pre-extract task data (need GIL for py::dict access). Each task carries
  // its own seed + per-seat configs. We dedup model paths across all tasks
  // into a single set up front so the EvaluatorPool warm-up can fail-fast
  // before threads spin up.
  struct TaskData {
    std::uint64_t seed = 0;
    std::vector<std::string> seat_model_paths;
    std::vector<int> seat_simulations;
    std::vector<int> seat_world_cap;
    std::vector<float> seat_c_puct;
    std::vector<std::optional<float>> seat_fpu_reduction;
    std::vector<bool> seat_uct2_arrival_visits;
    std::vector<double> seat_temperature;
    std::vector<std::string> seat_opp_selection;
    std::vector<int> seat_tail_solve;
    std::vector<int> seat_ts_depth;
    std::vector<std::int64_t> seat_ts_budget;
    std::vector<float> seat_ts_aux;
    std::vector<bool> seat_safe_mode;
    std::vector<int> seat_top_k;
    std::vector<int> seat_leaf_batch;
    std::string belief_model_path;
    // Per-seat belief model paths. Falls back to task-level belief_model_path.
    std::vector<std::string> seat_belief_paths;
  };
  std::vector<TaskData> task_data;
  task_data.reserve(tasks.size());
  std::unordered_set<std::string> all_model_paths;
  std::unordered_set<std::string> all_belief_paths;
  for (size_t ti = 0; ti < tasks.size(); ++ti) {
    const py::dict& t = tasks[ti];
    TaskData td;
    if (!t.contains("seed")) {
      throw std::invalid_argument(
          "run_concurrent_arena_batch: tasks[" + std::to_string(ti) +
          "] missing 'seed'");
    }
    td.seed = t["seed"].cast<std::uint64_t>();
    if (t.contains("belief_model_path")) {
      td.belief_model_path = t["belief_model_path"].cast<std::string>();
    }
    if (!t.contains("seat_configs")) {
      throw std::invalid_argument(
          "run_concurrent_arena_batch: tasks[" + std::to_string(ti) +
          "] missing 'seat_configs'");
    }
    auto seats = t["seat_configs"].cast<std::vector<py::dict>>();
    const size_t n_seats = seats.size();
    if (n_seats == 0) {
      throw std::invalid_argument(
          "run_concurrent_arena_batch: tasks[" + std::to_string(ti) +
          "] seat_configs is empty");
    }
    td.seat_model_paths.resize(n_seats);
    td.seat_simulations.resize(n_seats, 200);
    td.seat_world_cap.resize(n_seats, 0);
    td.seat_c_puct.resize(n_seats, 1.4f);
    td.seat_fpu_reduction.resize(n_seats);
    td.seat_uct2_arrival_visits.resize(n_seats, true);
    td.seat_temperature.resize(n_seats, 0.0);
    td.seat_opp_selection.resize(n_seats, "puct");
    td.seat_tail_solve.resize(n_seats, 0);
    td.seat_ts_depth.resize(n_seats, 5);
    td.seat_ts_budget.resize(n_seats, 10000000LL);
    td.seat_ts_aux.resize(n_seats, 0.0f);
    td.seat_safe_mode.resize(n_seats, safe_mode);
    td.seat_top_k.resize(n_seats, 0);
    td.seat_leaf_batch.resize(n_seats, 1);
    // Default per-seat belief paths to task-level belief_model_path.
    td.seat_belief_paths.resize(n_seats, td.belief_model_path);
    for (size_t i = 0; i < n_seats; ++i) {
      const py::dict& d = seats[i];
      if (!d.contains("model_path")) {
        throw std::invalid_argument(
            "run_concurrent_arena_batch: tasks[" + std::to_string(ti) +
            "].seat_configs[" + std::to_string(i) + "] missing 'model_path'");
      }
      td.seat_model_paths[i] = d["model_path"].cast<std::string>();
      if (td.seat_model_paths[i].empty()) {
        throw std::invalid_argument(
            "run_concurrent_arena_batch: empty model_path in task " +
            std::to_string(ti) + " seat " + std::to_string(i));
      }
      all_model_paths.insert(td.seat_model_paths[i]);
      if (d.contains("simulations")) td.seat_simulations[i] = d["simulations"].cast<int>();
      if (d.contains("world_cap")) td.seat_world_cap[i] = d["world_cap"].cast<int>();
      if (d.contains("c_puct")) td.seat_c_puct[i] = d["c_puct"].cast<float>();
      if (d.contains("fpu_reduction")) {
        td.seat_fpu_reduction[i] = d["fpu_reduction"].cast<std::optional<float>>();
      }
      if (d.contains("uct2_arrival_visits")) {
        td.seat_uct2_arrival_visits[i] =
            require_bool(d["uct2_arrival_visits"], "uct2_arrival_visits");
      }
      if (d.contains("temperature")) td.seat_temperature[i] = d["temperature"].cast<double>();
      if (d.contains("opponent_selection")) td.seat_opp_selection[i] = d["opponent_selection"].cast<std::string>();
      if (d.contains("tail_solve_enabled")) td.seat_tail_solve[i] = d["tail_solve_enabled"].cast<bool>() ? 1 : 0;
      if (d.contains("tail_solve_depth_limit")) td.seat_ts_depth[i] = d["tail_solve_depth_limit"].cast<int>();
      if (d.contains("tail_solve_node_budget")) td.seat_ts_budget[i] = d["tail_solve_node_budget"].cast<std::int64_t>();
      if (d.contains("tail_solve_auxiliary_weight")) td.seat_ts_aux[i] = d["tail_solve_auxiliary_weight"].cast<float>();
      if (d.contains("safe_mode")) td.seat_safe_mode[i] = d["safe_mode"].cast<bool>();
      if (d.contains("top_k")) td.seat_top_k[i] = d["top_k"].cast<int>();
      if (d.contains("leaf_batch_size")) td.seat_leaf_batch[i] = d["leaf_batch_size"].cast<int>();
      if (td.seat_leaf_batch[i] < 1) {
        throw std::invalid_argument(
            "run_concurrent_arena_batch: leaf_batch_size must be >= 1 in task " +
            std::to_string(ti) + " seat " + std::to_string(i));
      }
      if (d.contains("belief_model_path")) td.seat_belief_paths[i] = d["belief_model_path"].cast<std::string>();
    }
    // Collect all distinct belief paths for pool warm-up.
    for (const auto& bp : td.seat_belief_paths) {
      if (!bp.empty()) all_belief_paths.insert(bp);
    }
    task_data.push_back(std::move(td));
  }

  int max_leaf_batch = 1;
  for (const auto& td : task_data) {
    for (int lb : td.seat_leaf_batch) max_leaf_batch = std::max(max_leaf_batch, lb);
  }
  if (max_queue_size < static_cast<std::size_t>(num_threads) *
                           static_cast<std::size_t>(max_leaf_batch)) {
    throw std::invalid_argument(
        "run_concurrent_arena_batch: max_queue_size (" + std::to_string(max_queue_size) +
        ") < num_threads (" + std::to_string(num_threads) + ") * leaf_batch_size (" +
        std::to_string(max_leaf_batch) + "); the evaluator queue would overflow");
  }

  // Master bundle holds the encoder; all evaluators share it (same as
  // run_concurrent_selfplay_step).
  auto master = GameRegistry::instance().create_game(game_id, task_data[0].seed);

  py::gil_scoped_release release;

  // EvaluatorPool: one BatchedPolicyValueEvaluator per distinct model_path.
  infer::EvaluatorPool eval_pool;
  infer::BatchedEvaluatorConfig batched_cfg{};
  batched_cfg.batch_size = batch_size;
  batched_cfg.batch_timeout_us = batch_timeout_us;
  batched_cfg.max_queue_size = max_queue_size;
  batched_cfg.intra_op_num_threads = intra_op_num_threads;
  batched_cfg.num_inference_threads = num_inference_threads;
  batched_cfg.inter_op_num_threads = 1;
  batched_cfg.providers = providers;
  batched_cfg.devices = devices;
  std::unordered_map<std::string, search::IPolicyValueEvaluator*> eval_by_path;
  try {
    for (const auto& path : all_model_paths) {
      eval_by_path[path] = eval_pool.get_or_create(
          path, master.encoder.get(), batched_cfg);
    }
  } catch (const std::exception& e) {
    py::gil_scoped_acquire acquire;
    throw std::runtime_error(
        std::string("run_concurrent_arena_batch: evaluator creation failed: ") + e.what());
  }

  // Belief evaluators: dedup belief_model_path across tasks too.
  std::unordered_map<std::string, std::unique_ptr<infer::BatchedBeliefEvaluator>> belief_evals;
  if (master.belief_feature_extractor) {
    try {
      for (const auto& bp : all_belief_paths) {
        if (bp.empty()) continue;
        belief_evals[bp] = std::make_unique<infer::BatchedBeliefEvaluator>(
            bp, batched_cfg);
      }
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw std::runtime_error(
          std::string("run_concurrent_arena_batch: belief evaluator creation failed: ") + e.what());
    }
  }

  const std::size_t total_tasks = task_data.size();
  std::vector<runtime::ArenaMatchResult> results(total_tasks);
  std::vector<std::exception_ptr> errors(total_tasks);
  std::atomic<std::size_t> next_idx{0};

  auto worker_fn = [&]() {
    while (true) {
      const std::size_t idx = next_idx.fetch_add(1, std::memory_order_relaxed);
      if (idx >= total_tasks) break;
      try {
        const TaskData& td = task_data[idx];
        const size_t n_seats = td.seat_model_paths.size();

        auto bundle = GameRegistry::instance().create_game(game_id, td.seed);
        if (!td.belief_model_path.empty()) {
          bundle.belief_model_path = td.belief_model_path;
        }

        auto pp = create_per_seat_bundles(game_id, td.seed,
                                          bundle.state->num_players());

        // Per-seat evaluators routed via the shared pool.
        std::vector<const search::IPolicyValueEvaluator*> seat_eval_ptrs(n_seats, nullptr);
        for (size_t i = 0; i < n_seats; ++i) {
          seat_eval_ptrs[i] = eval_by_path.at(td.seat_model_paths[i]);
        }

        // Per-seat ArenaPlayerConfig.
        std::vector<runtime::ArenaPlayerConfig> player_configs;
        player_configs.reserve(n_seats);
        for (size_t i = 0; i < n_seats; ++i) {
          runtime::ArenaPlayerConfig cfg{};
          cfg.simulations = td.seat_simulations[i];
          cfg.world_cap = td.seat_world_cap[i];
          cfg.c_puct = td.seat_c_puct[i];
          cfg.fpu_reduction = td.seat_fpu_reduction[i];
          cfg.uct2_arrival_visits = td.seat_uct2_arrival_visits[i];
          cfg.temperature = td.seat_temperature[i];
          cfg.opponent_selection = parse_opponent_selection(td.seat_opp_selection[i]);
          cfg.safe_mode = td.seat_safe_mode[i];
          cfg.top_k = td.seat_top_k[i];
          cfg.leaf_batch_size = td.seat_leaf_batch[i];
          if (td.seat_tail_solve[i]) {
            cfg.tail_solve_enabled = true;
            cfg.tail_solve_config.depth_limit = td.seat_ts_depth[i];
            cfg.tail_solve_config.node_budget = td.seat_ts_budget[i];
            cfg.tail_solve_config.auxiliary_weight = td.seat_ts_aux[i];
            cfg.tail_solver = bundle.tail_solver.get();
            cfg.tail_solve_trigger = bundle.tail_solve_trigger;
          }
          player_configs.push_back(cfg);
        }

        // Per-seat belief evaluators routed via the shared belief pool.
        std::vector<const IBeliefEvaluator*> per_seat_belief_evals(n_seats, nullptr);
        const IBeliefFeatureExtractor* belief_extractor_ptr = nullptr;
        if (bundle.belief_feature_extractor) {
          for (size_t i = 0; i < n_seats; ++i) {
            const auto& bp = td.seat_belief_paths[i];
            if (bp.empty()) continue;
            auto it = belief_evals.find(bp);
            if (it != belief_evals.end()) {
              per_seat_belief_evals[i] = it->second.get();
              belief_extractor_ptr = bundle.belief_feature_extractor.get();
            }
          }
        }

        results[idx] = runtime::run_arena_match(
            *bundle.state, *bundle.rules, *bundle.value_model,
            [&seat_eval_ptrs](int player) -> const search::IPolicyValueEvaluator& {
              return *seat_eval_ptrs[static_cast<size_t>(player) %
                                     seat_eval_ptrs.size()];
            },
            player_configs, max_game_plies, td.seed,
            bundle.belief_tracker.get(), bundle.adjudicator,
            bundle.public_event_extractor,
            pp.trackers, pp.states,
            bundle.public_state_applier,
            bundle.events_only_extractor,
            belief_extractor_ptr,
            per_seat_belief_evals);
      } catch (...) {
        errors[idx] = std::current_exception();
      }
    }
  };

  std::vector<std::thread> threads;
  const int actual_threads = std::min(num_threads, static_cast<int>(total_tasks));
  threads.reserve(static_cast<size_t>(actual_threads));
  for (int t = 0; t < actual_threads; ++t) {
    threads.emplace_back(worker_fn);
  }
  for (auto& th : threads) th.join();

  for (std::size_t i = 0; i < total_tasks; ++i) {
    if (errors[i]) {
      py::gil_scoped_acquire acquire;
      std::rethrow_exception(errors[i]);
    }
  }

  py::gil_scoped_acquire acquire;
  py::list out_results;
  for (auto& r : results) {
    py::dict d;
    d["winner"] = r.winner;
    d["draw"] = r.draw;
    d["terminal_z_values"] = r.terminal_z_values;
    d["total_plies"] = r.total_plies;
    py::list actions;
    for (auto a : r.action_history) actions.append(static_cast<int>(a));
    d["action_history"] = actions;
    py::list ply_stats;
    for (const auto& ps : r.ply_stats) {
      py::dict pd;
      pd["tail_solved"] = ps.tail_solved;
      pd["tail_solve_value"] = ps.tail_solve_value;
      ply_stats.append(pd);
    }
    d["ply_stats"] = ply_stats;
    out_results.append(d);
  }
  py::dict out;
  out["results"] = out_results;
  return out;
}

// `draw` label of a finished eval-vs-heuristic game. winner() < 0 only says
// there is no single winner: in a 3-4 player game several seats can share
// first place (or last), which is a win for some and a loss for the others,
// not a draw. A draw is "no single winner AND every seat got the same
// terminal value". Same answer as before for every 2-player result.
bool terminal_draw(int winner, const std::vector<float>& terminal_values) {
  if (winner >= 0) return false;
  for (const float v : terminal_values) {
    if (std::abs(v - terminal_values.front()) > 1e-6f) return false;
  }
  return true;
}

py::dict run_constrained_eval_vs_heuristic_py(
    const std::string& game_id,
    std::uint64_t seed,
    const std::string& model_path,
    int simulations,
    int world_cap,
    float c_puct,
    double temperature,
    int model_is_player,
    bool constrained,
    double heuristic_temperature,
    const std::string& opponent_selection,
    bool tail_solve_enabled,
    int tail_solve_depth_limit,
    std::int64_t tail_solve_node_budget,
    float tail_solve_auxiliary_weight,
    const std::string& belief_model_path,
    const std::string& heuristic_selector,
    bool safe_mode,
    int top_k,
    int max_game_plies,
    std::optional<float> fpu_reduction,
    bool uct2_arrival_visits) {
  const auto opp_sel = parse_opponent_selection(opponent_selection);
  py::gil_scoped_release release;

  auto bundle = GameRegistry::instance().create_game(game_id, seed);
  if (!belief_model_path.empty()) bundle.belief_model_path = belief_model_path;
  if (heuristic_selector.empty() || !bundle.heuristics.has(heuristic_selector)) {
    py::gil_scoped_acquire acquire;
    throw std::runtime_error(
        "run_constrained_eval_vs_heuristic: heuristic '" +
        heuristic_selector + "' not registered for game '" + game_id + "'");
  }
  const board_ai::IHeuristic& eval_heuristic = bundle.heuristics.get(heuristic_selector);

  if (model_path.empty()) {
    throw std::invalid_argument("run_constrained_eval_vs_heuristic: model_path must not be empty");
  }
  auto model_eval = std::make_unique<infer::OnnxPolicyValueEvaluator>(
      model_path, bundle.encoder.get());
  if (!model_eval->is_ready()) {
    throw std::runtime_error("run_constrained_eval_vs_heuristic: failed to load model: " + model_eval->last_error());
  }
  const search::IPolicyValueEvaluator* eval_ptr = model_eval.get();

  auto belief_eval = create_belief_evaluator(
      bundle, "run_constrained_eval_vs_heuristic");
  const IBeliefEvaluator* belief_evaluator_ptr = belief_eval.ptr;

  std::unique_ptr<runtime::FilteredRulesWrapper> filtered_rules;
  if (constrained && bundle.training_action_filter) {
    filtered_rules = std::make_unique<runtime::FilteredRulesWrapper>(
        *bundle.rules, bundle.training_action_filter);
  }

  auto state = bundle.state->clone_state();
  int ply = 0;
  std::vector<ActionId> action_history;
  std::vector<runtime::ArenaPlyStats> ply_stats_vec;

  std::mt19937_64 rng(seed ^ 0xBEEF);
  SeatTrackers trackers(bundle, *state);

  while (!state->is_terminal() && ply < max_game_plies) {
    const int cp = state->current_player();

    if (cp == model_is_player) {
      const IGameRules& rules_for_model =
          filtered_rules ? *filtered_rules : *bundle.rules;
      const auto legal = rules_for_model.legal_actions(*state);
      if (legal.empty()) break;

      search::NetMctsConfig mcts_cfg{};
      mcts_cfg.simulations = simulations;
      mcts_cfg.world_cap = world_cap;
      mcts_cfg.c_puct = c_puct;
      mcts_cfg.fpu_reduction = fpu_reduction;
      mcts_cfg.uct2_arrival_visits = uct2_arrival_visits;
      mcts_cfg.opponent_selection = opp_sel;
      mcts_cfg.safe_mode = safe_mode;
      mcts_cfg.top_k = top_k;
      if (trackers.active()) mcts_cfg.root_belief_tracker = trackers.get(cp);
      if (bundle.belief_feature_extractor && belief_evaluator_ptr) {
        mcts_cfg.belief_extractor = bundle.belief_feature_extractor.get();
        mcts_cfg.belief_evaluator = belief_evaluator_ptr;
      }
      // Sim-tracker descent maintenance — events-only flavor.
      if (bundle.events_only_extractor) {
        mcts_cfg.events_only_extractor = bundle.events_only_extractor;
      }

      if (tail_solve_enabled && bundle.tail_solver) {
        bool try_ts = false;
        if (bundle.tail_solve_trigger) {
          try_ts = bundle.tail_solve_trigger(*state, ply);
        } else {
          try_ts = true;
        }
        if (try_ts) {
          mcts_cfg.tail_solve_enabled = true;
          mcts_cfg.tail_solve_config.depth_limit = tail_solve_depth_limit;
          mcts_cfg.tail_solve_config.node_budget = tail_solve_node_budget;
          mcts_cfg.tail_solve_config.auxiliary_weight = tail_solve_auxiliary_weight;
          mcts_cfg.tail_solve_config.margin_scorer = bundle.auxiliary_scorer;
          mcts_cfg.tail_solver = bundle.tail_solver.get();
        }
      }

      search::NetMcts mcts(mcts_cfg);
      search::NetMctsStats stats{};
      const std::uint64_t mcts_seed = board_ai::rng::derive_subseed(
          seed, "py_engine.replay_mcts", static_cast<std::uint64_t>(ply));
      mcts.search_root(*state, rules_for_model, *bundle.value_model,
                        *eval_ptr, &stats, mcts_seed);

      ActionId chosen = search::select_action_from_visits(
          stats.root_actions, stats.root_action_visits, temperature,
          seed ^ static_cast<std::uint64_t>(ply), top_k, legal[0]);

      action_history.push_back(chosen);
      ply_stats_vec.push_back({stats.tail_solved, stats.tail_solve_value});
      std::unique_ptr<IGameState> state_before;
      if (trackers.active()) state_before = state->clone_state();
      bundle.rules->do_action_fast(*state, chosen, rng);
      if (trackers.active()) trackers.observe(bundle, *state_before, chosen, *state);
    } else {
      auto masked = make_masked_state(*state, state->schema_ref(), cp);
      const IBeliefTracker* hr_tracker = cp >= 0 ? trackers.get(cp) : nullptr;
      auto hr = eval_heuristic.pick(*masked, hr_tracker, *bundle.rules, rng());
      if (hr.actions.empty()) break;

      double u01 = static_cast<double>(rng() & 0xFFFFFFFF) / 4294967296.0;
      std::size_t idx = runtime::sample_heuristic_index(
          hr.scores, heuristic_temperature, u01);
      ActionId chosen = hr.actions[idx];
      action_history.push_back(chosen);
      ply_stats_vec.push_back({false, 0.0f});
      std::unique_ptr<IGameState> state_before;
      if (trackers.active()) state_before = state->clone_state();
      bundle.rules->do_action_fast(*state, chosen, rng);
      if (trackers.active()) trackers.observe(bundle, *state_before, chosen, *state);
    }
    ++ply;
  }

  int winner = -1;
  bool draw = true;
  std::vector<float> terminal_z_values;
  if (state->is_terminal()) {
    winner = state->winner();
    terminal_z_values = bundle.value_model->terminal_values(*state);
    draw = terminal_draw(winner, terminal_z_values);
  } else if (bundle.adjudicator) {
    winner = bundle.adjudicator(*state);
    draw = (winner < 0);
    // Non-terminal adjudication is a timeout/anti-stalling fallback. Until a
    // game supplies richer custom adjudication values, penalize every seat.
    const int n = state->num_players();
    terminal_z_values.assign(static_cast<size_t>(n), -1.0f);
  }

  py::gil_scoped_acquire acquire;
  py::dict out;
  out["winner"] = winner;
  out["draw"] = draw;
  out["terminal_z_values"] = terminal_z_values;
  out["total_plies"] = ply;
  py::list actions;
  for (auto a : action_history) actions.append(static_cast<int>(a));
  out["action_history"] = actions;
  py::list pstats;
  for (const auto& ps : ply_stats_vec) {
    py::dict d;
    d["tail_solved"] = ps.tail_solved;
    d["tail_solve_value"] = ps.tail_solve_value;
    pstats.append(d);
  }
  out["ply_stats"] = pstats;
  return out;
}

// run_concurrent_eval_vs_heuristic_batch — Phase 2 batched/threaded version
// of run_constrained_eval_vs_heuristic. Spawns num_threads OS threads each
// running one m_vs_h game from a shared atomic counter, sharing one
// BatchedPolicyValueEvaluator pool for the model side. Heuristic side runs
// synchronously inside each worker thread (no extra coordination — fast
// rule heuristics are <1ms, MCTS-PV heuristics like quoridor's rule_mcts_v1
// run a hand-written PV that's CPU-only and uses thread-local state, so
// they don't compete for the BatchedEvaluator).
py::dict run_concurrent_eval_vs_heuristic_batch_py(
    const std::string& game_id,
    const std::string& model_path,
    const std::string& heuristic_selector,
    const std::vector<py::dict>& tasks,   // list of {seed, model_is_player}
    int num_threads,
    int simulations,
    int world_cap,
    float c_puct,
    double temperature,
    double heuristic_temperature,
    int max_game_plies,
    bool tail_solve_enabled,
    int tail_solve_depth_limit,
    std::int64_t tail_solve_node_budget,
    float tail_solve_auxiliary_weight,
    const std::string& opponent_selection,
    const std::string& belief_model_path,
    std::size_t batch_size,
    std::size_t batch_timeout_us,
    std::size_t max_queue_size,
    int intra_op_num_threads,
    int num_inference_threads,
    const std::vector<std::string>& providers,
    const std::vector<int>& devices,
    bool constrained,
    bool safe_mode,
    int top_k,
    int leaf_batch_size,
    std::optional<float> fpu_reduction,
    bool uct2_arrival_visits) {
  if (tasks.empty()) {
    throw std::invalid_argument(
        "run_concurrent_eval_vs_heuristic_batch: tasks must not be empty");
  }
  if (num_threads <= 0) {
    throw std::invalid_argument(
        "run_concurrent_eval_vs_heuristic_batch: num_threads must be > 0");
  }
  if (leaf_batch_size < 1) {
    throw std::invalid_argument(
        "run_concurrent_eval_vs_heuristic_batch: leaf_batch_size must be >= 1");
  }
  if (max_queue_size < static_cast<std::size_t>(num_threads) *
                           static_cast<std::size_t>(leaf_batch_size)) {
    throw std::invalid_argument(
        "run_concurrent_eval_vs_heuristic_batch: max_queue_size (" +
        std::to_string(max_queue_size) + ") < num_threads (" +
        std::to_string(num_threads) + ") * leaf_batch_size (" +
        std::to_string(leaf_batch_size) + "); the evaluator queue would overflow");
  }
  if (model_path.empty()) {
    throw std::invalid_argument(
        "run_concurrent_eval_vs_heuristic_batch: model_path must not be empty");
  }

  const auto opp_sel = parse_opponent_selection(opponent_selection);

  // Pre-extract per-task (seed, model_is_player) under the GIL.
  struct TaskData {
    std::uint64_t seed = 0;
    int model_is_player = 0;
  };
  std::vector<TaskData> task_data;
  task_data.reserve(tasks.size());
  for (size_t ti = 0; ti < tasks.size(); ++ti) {
    const py::dict& t = tasks[ti];
    TaskData td;
    if (!t.contains("seed") || !t.contains("model_is_player")) {
      throw std::invalid_argument(
          "run_concurrent_eval_vs_heuristic_batch: tasks[" +
          std::to_string(ti) + "] missing 'seed' or 'model_is_player'");
    }
    td.seed = t["seed"].cast<std::uint64_t>();
    td.model_is_player = t["model_is_player"].cast<int>();
    task_data.push_back(td);
  }

  // Master bundle holds the encoder.
  auto master = GameRegistry::instance().create_game(game_id, task_data[0].seed);
  if (!belief_model_path.empty()) master.belief_model_path = belief_model_path;

  if (heuristic_selector.empty() || !master.heuristics.has(heuristic_selector)) {
    throw std::runtime_error(
        "run_concurrent_eval_vs_heuristic_batch: heuristic '" +
        heuristic_selector + "' not registered for game '" + game_id + "'");
  }

  py::gil_scoped_release release;

  // Shared BatchedEvaluator pool. Single model_path = single evaluator,
  // shared across all worker threads.
  infer::EvaluatorPool eval_pool;
  infer::BatchedEvaluatorConfig batched_cfg{};
  batched_cfg.batch_size = batch_size;
  batched_cfg.batch_timeout_us = batch_timeout_us;
  batched_cfg.max_queue_size = max_queue_size;
  batched_cfg.intra_op_num_threads = intra_op_num_threads;
  batched_cfg.num_inference_threads = num_inference_threads;
  batched_cfg.inter_op_num_threads = 1;
  batched_cfg.providers = providers;
  batched_cfg.devices = devices;
  search::IPolicyValueEvaluator* shared_eval = nullptr;
  try {
    shared_eval = eval_pool.get_or_create(
        model_path, master.encoder.get(), batched_cfg);
  } catch (const std::exception& e) {
    py::gil_scoped_acquire acquire;
    throw std::runtime_error(
        std::string("run_concurrent_eval_vs_heuristic_batch: evaluator failed: ") + e.what());
  }

  // Belief evaluator (single, shared).
  std::unique_ptr<infer::BatchedBeliefEvaluator> belief_evaluator;
  const IBeliefEvaluator* belief_evaluator_ptr = nullptr;
  if (master.belief_feature_extractor && !master.belief_model_path.empty()) {
    try {
      belief_evaluator = std::make_unique<infer::BatchedBeliefEvaluator>(
          master.belief_model_path, batched_cfg);
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      throw std::runtime_error(
          std::string("run_concurrent_eval_vs_heuristic_batch: belief failed: ") + e.what());
    }
    belief_evaluator_ptr = belief_evaluator.get();
  }

  // Heuristic-side mcts_pv evaluator: when the selected heuristic is a
  // mcts_pv with model_path (splendor's pv_512x4, azul's pv_512x4 etc.),
  // load the shared BatchedEvaluator into the pool here so all workers'
  // heuristic.pick() leaf evals batch together. Hot-swap onto each
  // worker's bundle.heuristics happens in run_one_episode below.
  // No-op for rule heuristics or hand-written PV (no model_path).
  search::IPolicyValueEvaluator* heuristic_eval = nullptr;
  std::string heuristic_model_path;
  {
    auto* h = const_cast<board_ai::IHeuristic*>(
        &master.heuristics.get(heuristic_selector));
    if (auto* pvh = dynamic_cast<board_ai::heuristic::MctsPVHeuristic*>(h)) {
      const std::string& mp = pvh->model_path_hint();
      if (!mp.empty()) {
        if (max_queue_size < static_cast<std::size_t>(num_threads) *
                                 static_cast<std::size_t>(pvh->leaf_batch_size())) {
          py::gil_scoped_acquire acquire;
          throw std::invalid_argument(
              "run_concurrent_eval_vs_heuristic_batch: max_queue_size (" +
              std::to_string(max_queue_size) + ") < num_threads (" +
              std::to_string(num_threads) + ") * heuristic '" +
              heuristic_selector + "' leaf_batch_size (" +
              std::to_string(pvh->leaf_batch_size()) +
              "); the evaluator queue would overflow");
        }
        try {
          heuristic_eval = eval_pool.get_or_create(
              mp, pvh->encoder_hint(), batched_cfg);
          heuristic_model_path = mp;
        } catch (const std::exception& e) {
          py::gil_scoped_acquire acquire;
          throw std::runtime_error(
              std::string("run_concurrent_eval_vs_heuristic_batch: heuristic mcts_pv "
                          "evaluator failed for '") + mp + "': " + e.what());
        }
      }
    }
  }

  struct WorkerResult {
    int winner = -1;
    bool draw = false;
    int total_plies = 0;
    std::vector<ActionId> action_history;
    std::vector<float> terminal_z_values;
  };
  const std::size_t total_tasks = task_data.size();
  std::vector<WorkerResult> results(total_tasks);
  std::vector<std::exception_ptr> errors(total_tasks);
  std::atomic<std::size_t> next_idx{0};

  auto worker_fn = [&]() {
    while (true) {
      const std::size_t idx = next_idx.fetch_add(1, std::memory_order_relaxed);
      if (idx >= total_tasks) break;
      try {
        const TaskData& td = task_data[idx];
        auto bundle = GameRegistry::instance().create_game(game_id, td.seed);
        if (!belief_model_path.empty()) bundle.belief_model_path = belief_model_path;
        // Hot-swap heuristic mcts_pv evaluator onto the shared one (if
        // applicable).
        auto* eval_h_mut = const_cast<board_ai::IHeuristic*>(
            &bundle.heuristics.get(heuristic_selector));
        if (heuristic_eval) {
          if (auto* pvh = dynamic_cast<board_ai::heuristic::MctsPVHeuristic*>(
                  eval_h_mut)) {
            pvh->replace_evaluator(heuristic_eval);
          }
        }
        const board_ai::IHeuristic& eval_heuristic = *eval_h_mut;
        // Constrained: legal_actions for the model side go through
        // training_action_filter (matches old run_constrained_eval_vs_heuristic
        // when constrained=true). Heuristic side always uses unfiltered rules
        // so the heuristic itself sees the full action set.
        std::unique_ptr<runtime::FilteredRulesWrapper> filtered_rules;
        if (constrained && bundle.training_action_filter) {
          filtered_rules = std::make_unique<runtime::FilteredRulesWrapper>(
              *bundle.rules, bundle.training_action_filter);
        }
        const IGameRules& rules_for_model =
            filtered_rules ? static_cast<const IGameRules&>(*filtered_rules)
                           : static_cast<const IGameRules&>(*bundle.rules);

        auto state = bundle.state->clone_state();
        int ply = 0;
        std::vector<ActionId> action_history;
        std::mt19937_64 rng(td.seed ^ 0xBEEF);
        SeatTrackers trackers(bundle, *state);

        while (!state->is_terminal() && ply < max_game_plies) {
          const int cp = state->current_player();

          if (cp == td.model_is_player) {
            // Model side: shared BatchedEvaluator MCTS. When constrained=true,
            // legal_actions goes through training_action_filter (rules_for_model
            // = filtered wrapper) so the model's MCTS sees a narrowed action set.
            const auto legal = rules_for_model.legal_actions(*state);
            if (legal.empty()) break;

            search::NetMctsConfig mcts_cfg{};
            mcts_cfg.simulations = simulations;
            mcts_cfg.world_cap = world_cap;
            mcts_cfg.c_puct = c_puct;
            mcts_cfg.fpu_reduction = fpu_reduction;
            mcts_cfg.uct2_arrival_visits = uct2_arrival_visits;
            mcts_cfg.opponent_selection = opp_sel;
            mcts_cfg.safe_mode = safe_mode;
            mcts_cfg.top_k = top_k;
            mcts_cfg.leaf_batch_size = leaf_batch_size;
            if (trackers.active()) mcts_cfg.root_belief_tracker = trackers.get(cp);
            if (bundle.belief_feature_extractor && belief_evaluator_ptr) {
              mcts_cfg.belief_extractor = bundle.belief_feature_extractor.get();
              mcts_cfg.belief_evaluator = belief_evaluator_ptr;
            }
            if (bundle.events_only_extractor) {
              mcts_cfg.events_only_extractor = bundle.events_only_extractor;
            }
            if (tail_solve_enabled && bundle.tail_solver) {
              bool try_ts = bundle.tail_solve_trigger
                  ? bundle.tail_solve_trigger(*state, ply) : true;
              if (try_ts) {
                mcts_cfg.tail_solve_enabled = true;
                mcts_cfg.tail_solve_config.depth_limit = tail_solve_depth_limit;
                mcts_cfg.tail_solve_config.node_budget = tail_solve_node_budget;
                mcts_cfg.tail_solve_config.auxiliary_weight = tail_solve_auxiliary_weight;
                mcts_cfg.tail_solve_config.margin_scorer = bundle.auxiliary_scorer;
                mcts_cfg.tail_solver = bundle.tail_solver.get();
              }
            }

            search::NetMcts mcts(mcts_cfg);
            search::NetMctsStats stats{};
            const std::uint64_t mcts_seed = board_ai::rng::derive_subseed(
                td.seed, "py_engine.concurrent_eval_mcts",
                static_cast<std::uint64_t>(ply));
            mcts.search_root(*state, rules_for_model, *bundle.value_model,
                             *shared_eval, &stats, mcts_seed);
            ActionId chosen = search::select_action_from_visits(
                stats.root_actions, stats.root_action_visits, temperature,
                td.seed ^ static_cast<std::uint64_t>(ply), top_k, legal[0]);
            action_history.push_back(chosen);
            std::unique_ptr<IGameState> state_before;
            if (trackers.active()) state_before = state->clone_state();
            bundle.rules->do_action_fast(*state, chosen, rng);
            if (trackers.active()) trackers.observe(bundle, *state_before, chosen, *state);
          } else {
            // Heuristic side: synchronous in this worker thread.
            auto masked = make_masked_state(*state, state->schema_ref(), cp);
            const IBeliefTracker* hr_tracker = cp >= 0 ? trackers.get(cp) : nullptr;
            auto hr = eval_heuristic.pick(*masked, hr_tracker, *bundle.rules, rng());
            if (hr.actions.empty()) break;
            double u01 = static_cast<double>(rng() & 0xFFFFFFFF) / 4294967296.0;
            std::size_t hi = runtime::sample_heuristic_index(
                hr.scores, heuristic_temperature, u01);
            ActionId chosen = hr.actions[hi];
            action_history.push_back(chosen);
            std::unique_ptr<IGameState> state_before;
            if (trackers.active()) state_before = state->clone_state();
            bundle.rules->do_action_fast(*state, chosen, rng);
            if (trackers.active()) trackers.observe(bundle, *state_before, chosen, *state);
          }
          ++ply;
        }

        WorkerResult& wr = results[idx];
        wr.total_plies = ply;
        wr.action_history = std::move(action_history);
        if (state->is_terminal()) {
          wr.winner = state->winner();
          wr.terminal_z_values = bundle.value_model->terminal_values(*state);
          wr.draw = terminal_draw(wr.winner, wr.terminal_z_values);
        } else if (bundle.adjudicator) {
          wr.winner = bundle.adjudicator(*state);
          wr.draw = (wr.winner < 0);
          // Non-terminal adjudication is a timeout/anti-stalling fallback.
          // Until a game supplies richer custom adjudication values, penalize
          // every seat.
          const int n = state->num_players();
          wr.terminal_z_values.assign(static_cast<size_t>(n), -1.0f);
        } else {
          wr.draw = true;
          wr.terminal_z_values.assign(
              static_cast<size_t>(state->num_players()), 0.0f);
        }
      } catch (...) {
        errors[idx] = std::current_exception();
      }
    }
  };

  std::vector<std::thread> threads;
  const int actual_threads = std::min(num_threads, static_cast<int>(total_tasks));
  threads.reserve(static_cast<size_t>(actual_threads));
  for (int t = 0; t < actual_threads; ++t) {
    threads.emplace_back(worker_fn);
  }
  for (auto& th : threads) th.join();

  for (std::size_t i = 0; i < total_tasks; ++i) {
    if (errors[i]) {
      py::gil_scoped_acquire acquire;
      std::rethrow_exception(errors[i]);
    }
  }

  py::gil_scoped_acquire acquire;
  py::list out_results;
  for (auto& r : results) {
    py::dict d;
    d["winner"] = r.winner;
    d["draw"] = r.draw;
    d["terminal_z_values"] = r.terminal_z_values;
    d["total_plies"] = r.total_plies;
    py::list actions;
    for (auto a : r.action_history) actions.append(static_cast<int>(a));
    d["action_history"] = actions;
    out_results.append(d);
  }
  py::dict out;
  out["results"] = out_results;
  return out;
}

py::dict run_heuristic_episode_py(
    const std::string& game_id,
    std::uint64_t seed,
    double temperature,
    int max_game_plies,
    const std::string& heuristic_selector) {
  py::gil_scoped_release release;

  auto bundle = GameRegistry::instance().create_game(game_id, seed);
  if (heuristic_selector.empty() || !bundle.heuristics.has(heuristic_selector)) {
    py::gil_scoped_acquire acquire;
    throw std::runtime_error(
        "run_heuristic_episode: heuristic '" + heuristic_selector +
        "' not registered for game '" + game_id + "'");
  }
  const board_ai::IHeuristic& heuristic = bundle.heuristics.get(heuristic_selector);

  auto pp = create_per_seat_bundles(game_id, seed, bundle.state->num_players());

  auto result = runtime::run_heuristic_episode(
      *bundle.state, *bundle.rules, *bundle.value_model,
      bundle.encoder.get(),
      heuristic,
      temperature, max_game_plies, seed,
      bundle.auxiliary_scorer, bundle.adjudicator,
      pp.trackers,
      pp.states,
      bundle.public_state_applier,
      bundle.public_event_extractor);

  py::gil_scoped_acquire acquire;
  return result_to_py(result, bundle);
}

// Per-seat heuristic match: each seat plays its own named heuristic with
// its own temperature. Used by platform/tools/eval_model.py to run
// heuristic-vs-heuristic benchmarks (e.g. lkf_v1 vs rule_v1).
py::dict run_heuristic_vs_heuristic_py(
    const std::string& game_id,
    std::uint64_t seed,
    int max_game_plies,
    const std::vector<std::string>& heuristic_names,
    const std::vector<double>& temperatures) {
  py::gil_scoped_release release;

  auto bundle = GameRegistry::instance().create_game(game_id, seed);
  const int n = bundle.state->num_players();
  if (static_cast<int>(heuristic_names.size()) != n) {
    py::gil_scoped_acquire acquire;
    throw std::runtime_error(
        "run_heuristic_vs_heuristic: heuristic_names size " +
        std::to_string(heuristic_names.size()) +
        " != num_players " + std::to_string(n) +
        " for game '" + game_id + "'");
  }
  if (static_cast<int>(temperatures.size()) != n) {
    py::gil_scoped_acquire acquire;
    throw std::runtime_error(
        "run_heuristic_vs_heuristic: temperatures size " +
        std::to_string(temperatures.size()) +
        " != num_players " + std::to_string(n) +
        " for game '" + game_id + "'");
  }

  std::vector<const board_ai::IHeuristic*> per_seat_heuristics(
      static_cast<size_t>(n), nullptr);
  for (int p = 0; p < n; ++p) {
    const std::string& name = heuristic_names[static_cast<size_t>(p)];
    if (name.empty() || !bundle.heuristics.has(name)) {
      py::gil_scoped_acquire acquire;
      throw std::runtime_error(
          "run_heuristic_vs_heuristic: heuristic '" + name +
          "' not registered for game '" + game_id + "' (seat " +
          std::to_string(p) + ")");
    }
    per_seat_heuristics[static_cast<size_t>(p)] = &bundle.heuristics.get(name);
  }

  auto pp = create_per_seat_bundles(game_id, seed, n);

  auto result = runtime::run_heuristic_episode_per_seat(
      *bundle.state, *bundle.rules, *bundle.value_model,
      bundle.encoder.get(),
      per_seat_heuristics, temperatures,
      max_game_plies, seed,
      bundle.auxiliary_scorer, bundle.adjudicator,
      pp.trackers,
      pp.states,
      bundle.public_state_applier,
      bundle.public_event_extractor);

  py::gil_scoped_acquire acquire;
  py::dict out = result_to_py(result, bundle);
  py::list actions;
  for (const auto& s : result.samples) actions.append(static_cast<int>(s.action_id));
  out["action_history"] = actions;
  return out;
}

// Encode the bundle's initial state from a specific perspective seat,
// optionally after init'ing the bundle's belief tracker for a possibly
// DIFFERENT perspective and applying a sequence of (actor, action) public
// events on that tracker. Used by OB-002 regression test: a tracker bound
// to seat A who learned an opponent's hand via Priest must not leak that
// knowledge into the encoder output for seat B.
py::dict encode_state_for_perspective_py(
    const std::string& game_id,
    std::uint64_t seed,
    int encode_player,
    int tracker_perspective,
    py::list event_actions) {
  py::gil_scoped_release release;

  auto bundle = GameRegistry::instance().create_game(game_id, seed);
  if (encode_player < 0 || encode_player >= bundle.state->num_players()) {
    py::gil_scoped_acquire acquire;
    throw std::runtime_error("encode_state_for_perspective: encode_player out of range");
  }

  // Drive both bundle.state (truth) and bundle.belief_tracker (perspective
  // observer) forward by replaying actions. This populates tracker's
  // known_hand from real public events (e.g. Priest reveals) without us
  // synthesizing event payloads by hand.
  bool tracker_init_done = false;
  if (bundle.belief_tracker && tracker_perspective >= 0) {
    tracker_init(*bundle.belief_tracker, bundle, *bundle.state, tracker_perspective);
    tracker_init_done = true;
  }

  py::gil_scoped_acquire acquire_for_list;
  std::vector<std::pair<int, ActionId>> pairs;
  pairs.reserve(py::len(event_actions));
  for (auto h : event_actions) {
    auto t = h.cast<py::tuple>();
    pairs.emplace_back(t[0].cast<int>(), t[1].cast<ActionId>());
  }
  py::gil_scoped_release release2;

  std::mt19937_64 step_rng(
      board_ai::rng::derive_subseed(seed, "py_engine.encode_replay_step"));
  for (const auto& [actor, action] : pairs) {
    auto before = bundle.state->clone_state();
    bundle.rules->do_action_fast(*bundle.state, action, step_rng);
    if (tracker_init_done && bundle.public_event_extractor) {
      auto ev = bundle.public_event_extractor(*before, action, *bundle.state, tracker_perspective);
      bundle.belief_tracker->observe_public_event(
          actor, bundle.rules->public_action(*before, action), ev.events);
    }
  }

  std::vector<float> features;
  auto masked = make_masked_state(*bundle.state,
                                  bundle.state->schema_ref(), encode_player);
  bundle.encoder->encode_features(*masked, encode_player,
                                  bundle.belief_tracker.get(), &features);

  py::gil_scoped_acquire acquire;
  py::dict out;
  out["features"] = features;
  out["tracker_perspective"] =
      bundle.belief_tracker ? tracker_perspective : -1;
  return out;
}

py::dict encode_state_py(
    const std::string& game_id,
    std::uint64_t seed) {
  py::gil_scoped_release release;

  auto bundle = GameRegistry::instance().create_game(game_id, seed);
  const int player = bundle.state->current_player();
  const auto legal = bundle.rules->legal_actions(*bundle.state);

  std::vector<float> features;
  std::vector<float> legal_mask;
  bundle.encoder->encode(*bundle.state, player, bundle.belief_tracker.get(), legal, &features, &legal_mask);

  const bool is_terminal = bundle.state->is_terminal();
  const int action_space = bundle.encoder->action_space();
  const int feature_dim = bundle.encoder->feature_dim();

  py::gil_scoped_acquire acquire;
  py::dict out;
  out["features"] = features;
  out["legal_mask"] = legal_mask;
  out["legal_actions"] = legal;
  out["current_player"] = player;
  out["is_terminal"] = is_terminal;
  out["action_space"] = action_space;
  out["feature_dim"] = feature_dim;
  return out;
}

py::dict tail_solve_py(
    const std::string& game_id,
    std::uint64_t seed,
    int perspective_player,
    int depth_limit,
    std::int64_t node_budget) {
  py::gil_scoped_release release;

  auto bundle = GameRegistry::instance().create_game(game_id, seed);
  if (!bundle.tail_solver) {
    py::gil_scoped_acquire acquire;
    throw std::runtime_error("tail_solve: game '" + game_id + "' has no tail_solver registered");
  }

  search::TailSolveConfig cfg{};
  cfg.depth_limit = depth_limit;
  cfg.node_budget = node_budget;
  // The standalone tail_solve binding is a test-only correctness probe; it
  // intentionally leaves speed/margin tiebreaks at zero so tests assert raw
  // win/loss/draw values, not bonuses. Production paths (selfplay/arena/eval)
  // pass their own weights via run_*_match.

  auto ts = bundle.tail_solver->solve(
      *bundle.state, *bundle.rules, *bundle.value_model,
      perspective_player, cfg);

  py::gil_scoped_acquire acquire;
  py::dict out;
  out["value"] = ts.value;
  out["best_action"] = ts.best_action;
  out["nodes_searched"] = ts.nodes_searched;
  out["elapsed_ms"] = ts.elapsed_ms;
  out["budget_exceeded"] = ts.budget_exceeded;
  return out;
}

class GameSessionWrapper {
 public:
  GameSessionWrapper(const std::string& game_id, std::uint64_t seed,
                     const std::string& model_path, bool use_filter,
                     const std::string& belief_model_path = "")
      : game_id_(game_id), seed_(seed), model_path_(model_path) {
    py::gil_scoped_release release;
    bundle_ = std::make_unique<GameBundle>(
        GameRegistry::instance().create_game(game_id, seed));
    if (!belief_model_path.empty()) {
      bundle_->belief_model_path = belief_model_path;
    }
    if (!model_path.empty()) {
      evaluator_ = std::make_unique<infer::OnnxPolicyValueEvaluator>(
          model_path, bundle_->encoder.get());
      if (!evaluator_->is_ready()) {
        throw std::runtime_error("GameSession: failed to load model: " + evaluator_->last_error());
      }
    }
    if (bundle_->belief_feature_extractor &&
        !bundle_->belief_model_path.empty()) {
      belief_evaluator_ = std::make_unique<infer::OnnxBeliefEvaluator>(
          bundle_->belief_model_path);
      if (!belief_evaluator_->is_ready()) {
        throw std::runtime_error(
            "GameSession: failed to load belief model '" +
            bundle_->belief_model_path + "': " +
            belief_evaluator_->last_error());
      }
    }
    if (use_filter && bundle_->training_action_filter) {
      filtered_rules_ = std::make_unique<runtime::FilteredRulesWrapper>(
          *bundle_->rules, bundle_->training_action_filter);
    }
    bt_ = bundle_->belief_tracker.get();
    // Per-perspective AI views: MCTS searches on these, never on the truth
    // state (bundle_->state). Skipped when external_obs_mode_ is set
    // (AI API path) — in that mode bundle_->state IS the AI view.
    init_ai_views_();
  }

  void init_ai_views_() {
    if (external_obs_mode_) return;
    const int n = bundle_->state->num_players();
    ai_views_.resize(static_cast<size_t>(n));
    ai_trackers_.resize(static_cast<size_t>(n));
    ai_encoders_.resize(static_cast<size_t>(n));
    ai_evaluators_.resize(static_cast<size_t>(n));
    for (int p = 0; p < n; ++p) {
      auto extra = GameRegistry::instance().create_game(game_id_, seed_);
      ai_trackers_[p] = std::move(extra.belief_tracker);
      ai_encoders_[p] = std::move(extra.encoder);

      // Bootstrap each AI view from the truth state via the walker:
      // viz=1 slots flow through; viz=0 slots are placeholder. This is
      // the cleanest "nature action 0" snapshot — game-agnostic.
      ai_views_[p] = make_masked_state(*bundle_->state,
                                       bundle_->state->schema_ref(), p);
      if (ai_trackers_[p]) {
        tracker_init(*ai_trackers_[p], *bundle_, *ai_views_[p], p);
      }
      if (!model_path_.empty() && ai_encoders_[p]) {
        ai_evaluators_[p] = std::make_unique<infer::OnnxPolicyValueEvaluator>(
            model_path_, ai_encoders_[p].get());
        if (!ai_evaluators_[p]->is_ready()) {
          throw std::runtime_error(
              "GameSession: failed to load model (ai_view[" + std::to_string(p) +
              "]): " + ai_evaluators_[p]->last_error());
        }
      }
    }
  }

  // Advance ai_view for a single perspective. Hidden-info games: snapshot
  // overwrites the public part and the tracker observes the events. Fully-
  // public games (no extractor registered) just run do_action_fast on the
  // seat — there is no snapshot to apply. The session's viz=0 slots are
  // intentionally NOT freshened: nothing on the decision side reads them
  // (MCTS sims sample at sim entry, hash uses kHiddenHashSentinel for viz=0,
  // encoder reads masked-state placeholders).
  void advance_ai_view_(int perspective, const IGameState& truth_before,
                        ActionId action) {
    if (perspective < 0 || perspective >= static_cast<int>(ai_views_.size())) return;
    if (!ai_views_[perspective]) return;
    const int actor = truth_before.current_player();
    const ActionId observed = bundle_->rules->public_action(truth_before, action);
    if (!bundle_->public_event_extractor) {
      const std::uint64_t view_step_seed = board_ai::rng::derive_subseed(
          seed_, "session.view_step",
          static_cast<std::uint64_t>(ply_count_) * 17ULL +
              static_cast<std::uint64_t>(perspective));
      std::mt19937_64 view_step_rng(view_step_seed);
      bundle_->rules->do_action_fast(*ai_views_[perspective], action, view_step_rng);
      if (ai_trackers_[perspective]) {
        ai_trackers_[perspective]->observe_public_event(actor, observed, {});
      }
      return;
    }
    PublicEventTrace trace = bundle_->public_event_extractor(
        truth_before, action, *bundle_->state, perspective);
    ai_views_[perspective]->begin_step_for_session_observe();
    if (bundle_->public_state_applier && !trace.public_snapshot.empty()) {
      bundle_->public_state_applier(
          *ai_views_[perspective], trace.public_snapshot, perspective);
    }
    if (ai_trackers_[perspective]) {
      ai_trackers_[perspective]->observe_public_event(
          actor, observed, trace.events);
    }
  }

  bool is_terminal() const { return bundle_->state->is_terminal(); }
  bool is_turn_start() const { return bundle_->state->is_turn_start(); }
  int current_player() const { return bundle_->state->current_player(); }
  int winner() const { return bundle_->state->winner(); }
  std::vector<int> winners() const { return bundle_->state->winners(); }
  std::vector<int> losers() const { return bundle_->state->losers(); }
  int num_players() const { return bundle_->state->num_players(); }
  std::string game_id() const { return game_id_; }

  // Expose perspective hash so tests can directly assert hash-scope
  // invariants (e.g. "internal RNG state must NOT affect public hash" —
  // the BUG-028 anti-pattern). Returns the truth state's hash; for
  // information-set hashing through ai_views_, callers can use
  // state_hash_for_perspective_ai_view.
  std::uint64_t state_hash_for_perspective(int player) const {
    return bundle_->state->state_hash_for_perspective(player);
  }

  // Expose step_count so tests can assert framework wrappers around
  // do_action_fast / do_action_deterministic / undo_action keep the
  // counter strictly monotonic across rules-path transitions.
  std::uint32_t step_count() const { return bundle_->state->step_count(); }

  py::dict get_state_dict() {
    if (!bundle_->state_serializer) {
      throw std::runtime_error("get_state_dict: game '" + game_id_ + "' has no state_serializer registered");
    }
    py::gil_scoped_release release;
    auto m = bundle_->state_serializer(*bundle_->state);
    py::gil_scoped_acquire acquire;
    py::dict out;
    for (const auto& [k, v] : m) out[py::cast(k)] = any_to_py(v);
    return out;
  }

  // Keeps a leaf only where the two probe serializations agree (see
  // engine/core/viewer_state.h); anything else becomes None. Containers
  // of different shape are dropped as a whole.
  static py::object keep_agreeing_leaves(py::handle a, py::handle b) {
    if (py::isinstance<py::dict>(a) && py::isinstance<py::dict>(b)) {
      py::dict da = py::reinterpret_borrow<py::dict>(a);
      py::dict db = py::reinterpret_borrow<py::dict>(b);
      py::dict out;
      for (auto item : da) {
        out[item.first] = db.contains(item.first)
            ? keep_agreeing_leaves(item.second, db[item.first])
            : py::object(py::none());
      }
      for (auto item : db) {
        if (!da.contains(item.first)) out[item.first] = py::none();
      }
      return out;
    }
    if (py::isinstance<py::list>(a) && py::isinstance<py::list>(b)) {
      py::list la = py::reinterpret_borrow<py::list>(a);
      py::list lb = py::reinterpret_borrow<py::list>(b);
      if (py::len(la) != py::len(lb)) return py::none();
      py::list out;
      for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(py::len(la)); ++i) {
        out.append(keep_agreeing_leaves(la[i], lb[i]));
      }
      return out;
    }
    if (a.get_type().is(b.get_type()) && a.equal(b)) {
      return py::reinterpret_borrow<py::object>(a);
    }
    return py::none();
  }

  // What seat `viewer` may be shown of the state: the game serializer's
  // output with every leaf that depends on a slot hidden from `viewer`
  // replaced by None. Built from two probe states that contain no hidden
  // truth value at all (engine/core/viewer_state.h), so it is a function of
  // the viewer-visible slots only. `get_state_dict()` stays the truth dump
  // for tests and tools; anything sent to a player must use this one.
  //
  // Top-level keys starting with '_' are test-only by convention and are
  // omitted. `current_player` / `is_terminal` / `winner` are public by the
  // IGameState contract and are taken from the real state (a serializer
  // evaluating them on a probe can be thrown off by the sentinels).
  py::dict get_public_state_dict(int viewer) {
    if (!bundle_->state_serializer) {
      throw std::runtime_error("get_public_state_dict: game '" + game_id_ + "' has no state_serializer registered");
    }
    AnyMap ma;
    AnyMap mb;
    {
      py::gil_scoped_release release;
      viz::ViewerProbes probes = viz::make_viewer_probes(*bundle_->state, viewer);
      ma = bundle_->state_serializer(*probes.a);
      mb = bundle_->state_serializer(*probes.b);
    }
    py::dict out;
    for (const auto& [k, v] : ma) {
      if (!k.empty() && k[0] == '_') continue;
      auto it = mb.find(k);
      if (it == mb.end()) {
        out[py::cast(k)] = py::none();
      } else {
        out[py::cast(k)] = keep_agreeing_leaves(any_to_py(v), any_to_py(it->second));
      }
    }
    for (const auto& [k, v] : mb) {
      if (!k.empty() && k[0] == '_') continue;
      if (ma.find(k) == ma.end()) out[py::cast(k)] = py::none();
    }
    if (out.contains("current_player")) out["current_player"] = bundle_->state->current_player();
    if (out.contains("is_terminal")) out["is_terminal"] = bundle_->state->is_terminal();
    if (out.contains("winner")) out["winner"] = bundle_->state->winner();
    return out;
  }

  // True iff `seat` currently sees a slot that `viewer` does not, i.e.
  // `seat`'s legal actions / search statistics may depend on information
  // hidden from `viewer`.
  bool seat_sees_more_than(int seat, int viewer) const {
    return viz::seat_sees_more_than(*bundle_->state, seat, viewer);
  }

  // Test support: (field name, idx) of every slot hidden from `viewer`.
  py::list _hidden_slots(int viewer) {
    py::list out;
    const auto& schema = bundle_->state->schema_ref();
    viz::for_each_hidden_slot(
        *bundle_->state, schema, viewer,
        [&](viz::FieldId fid, const std::vector<int>& idx, const viz::VizTensor&) {
          out.append(py::make_tuple(
              schema.fields[static_cast<size_t>(fid)].name, idx));
        });
    return out;
  }

  // Test support: the encoder's features and legal mask of the session state
  // from `perspective`, through the same masked path search uses (without a
  // belief tracker, so only for encoders that do not read one).
  py::dict _encode(int perspective) {
    if (perspective < 0 || perspective >= bundle_->state->num_players()) {
      throw std::runtime_error("_encode: perspective out of range");
    }
    std::vector<float> features, legal_mask;
    {
      py::gil_scoped_release release;
      const auto legal = bundle_->rules->legal_actions(*bundle_->state);
      if (!bundle_->encoder->encode(*bundle_->state, perspective, nullptr, legal, &features, &legal_mask)) {
        throw std::runtime_error("_encode: encoder produced an invalid feature length");
      }
    }
    py::dict out;
    out["features"] = features;
    out["legal_mask"] = legal_mask;
    return out;
  }

  // Test support: dual of _force_write_field.
  py::object _read_field(const std::string& name, const std::vector<int>& idx) {
    const viz::FieldId fid =
        viz::schema_lookup(bundle_->state->schema_ref(), name);
    if (fid == viz::kInvalidFieldId) {
      throw std::runtime_error("_read_field: unknown field '" + name + "'");
    }
    return any_to_py(bundle_->state->read_field_slot(fid, idx));
  }

  // Test hook: directly invoke the game's public_state_applier on a given
  // snapshot. Used by test_public_snapshot_round_trip to verify the applier
  // is a correct inverse of the extractor without going through
  // apply_observation.
  void apply_public_snapshot(py::dict snapshot, int receiver_seat) {
    if (!bundle_->public_state_applier) {
      throw std::runtime_error(
          "apply_public_snapshot: game '" + game_id_ +
          "' has no public_state_applier registered");
    }
    AnyMap snap = py_dict_to_any_map(snapshot);
    py::gil_scoped_release release;
    bundle_->public_state_applier(*bundle_->state, snap, receiver_seat);
  }

  py::dict get_action_info(ActionId action) {
    if (!bundle_->action_descriptor) {
      throw std::runtime_error("get_action_info: game '" + game_id_ + "' has no action_descriptor registered");
    }
    auto m = bundle_->action_descriptor(action);
    py::dict out;
    for (const auto& [k, v] : m) out[py::cast(k)] = any_to_py(v);
    return out;
  }

  std::vector<ActionId> get_legal_actions() {
    py::gil_scoped_release release;
    const IGameRules& rules = filtered_rules_ ? *filtered_rules_ : *bundle_->rules;
    return rules.legal_actions(*bundle_->state);
  }

  std::vector<ActionId> get_all_legal_actions() {
    py::gil_scoped_release release;
    return bundle_->rules->legal_actions(*bundle_->state);
  }

  void apply_action(ActionId action) {
    py::gil_scoped_release release;
    const int actor = bundle_->state->current_player();
    std::unique_ptr<IGameState> state_before = bundle_->state->clone_state();
    bundle_->rules->do_action_fast(*bundle_->state, action, step_rng_);
    if (bt_) {
      tracker_observe(*bt_, *bundle_, *state_before, action, *bundle_->state,
                      actor);
    }

    // Advance each perspective's AI view via the public-event protocol.
    if (!external_obs_mode_) {
      const int n = static_cast<int>(ai_views_.size());
      for (int p = 0; p < n; ++p) {
        advance_ai_view_(p, *state_before, action);
      }
    }
    ++ply_count_;
  }

  // Combined action + events step for the AI API. Sequence:
  //   1. begin_step_for_session_observe() bumps step_count for DAG
  //      acyclicity (do_action_fast is intentionally skipped — the AI
  //      session never runs rules).
  //   2. public_state_applier(snapshot) overwrites session state_'s public
  //      fields from the truth snapshot.
  //   3. belief_tracker.observe_public_event(actor, action, events) — the
  //      events list is fed only to the tracker; it does not mutate state.
  //
  // After apply_observation returns:
  //   - session state_'s public fields equal the truth snapshot exactly
  //     (test_public_snapshot_round_trip);
  //   - session state_'s viz=0 slots retain whatever was last written there;
  //     nothing on the decision side reads them. MCTS sims sample at sim
  //     entry on a cloned sim_tracker; the hash mixes kHiddenHashSentinel
  //     for viz=0; the encoder reads masked-state placeholders.
  //   - `state_hash_for_perspective(own)` on the session is byte-equal to
  //     running the same observation stream on any other seed
  //     (test_public_hash_excludes_internal_rng / test_api_belief_matches_selfplay).
  //
  // `events` is a list of {"kind": str, "payload": dict} entries describing
  // public observations from this transition. `public_snapshot` is the
  // truth-side dump of all public fields; when non-empty + game has
  // public_state_applier, it overwrites session state_'s public fields.
  // Fully-public games (tictactoe, quoridor) register no applier.
  void apply_observation(ActionId action,
                         py::list events,
                         py::dict public_snapshot) {
    if (!bundle_->public_state_applier) {
      throw std::runtime_error(
          "apply_observation: game '" + game_id_ +
          "' has no public_state_applier registered");
    }
    std::vector<std::pair<std::string, AnyMap>> event_list;
    for (py::handle item : events) {
      py::dict d = py::cast<py::dict>(item);
      std::string kind = py::cast<std::string>(d["kind"]);
      AnyMap payload = py_dict_to_any_map(py::cast<py::dict>(d["payload"]));
      event_list.emplace_back(std::move(kind), std::move(payload));
    }

    AnyMap snap_map;
    bool have_snapshot = false;
    if (public_snapshot && py::len(public_snapshot) > 0) {
      snap_map = py_dict_to_any_map(public_snapshot);
      have_snapshot = true;
    }

    py::gil_scoped_release release;
    external_obs_mode_ = true;
    const int actor = bundle_->state->current_player();
    bundle_->state->begin_step_for_session_observe();
    if (have_snapshot) {
      bundle_->public_state_applier(*bundle_->state, snap_map, api_perspective_);
    }

    if (bt_) {
      std::vector<PublicEvent> events_v(event_list.begin(), event_list.end());
      bt_->observe_public_event(actor, action, events_v);
    }
    ++ply_count_;
  }

  // Test/integration helper: apply an action on the truth state and ALSO
  // return the public-event trace (events + public_snapshot) for a given
  // perspective. This is what selfplay's trace machinery records per-ply,
  // exposed in a step-driven form so tests can drive a ground-truth session
  // and forward the resulting trace to a separate API/AI session.
  //
  // The trace dict shape matches `observation_trace[i]` from
  // run_selfplay_episode: {"events": [...], "public_snapshot": {...}},
  // plus "action": the public projection of `action` that observers (and
  // apply_observation) receive. For deterministic games (no extractor)
  // returns empty list / dict.
  py::dict apply_action_with_trace(ActionId action, int perspective) {
    std::vector<std::pair<std::string, AnyMap>> event_list;
    AnyMap snap_map;
    bool have_extractor = false;
    ActionId observed = action;
    {
      py::gil_scoped_release release;
      const int actor = bundle_->state->current_player();
      std::unique_ptr<IGameState> state_before = bundle_->state->clone_state();
      observed = bundle_->rules->public_action(*state_before, action);
      bundle_->rules->do_action_fast(*bundle_->state, action, step_rng_);
      if (bundle_->public_event_extractor) {
        have_extractor = true;
        PublicEventTrace trace = bundle_->public_event_extractor(
            *state_before, action, *bundle_->state, perspective);
        for (auto& ev : trace.events)
          event_list.emplace_back(std::move(ev.first), std::move(ev.second));
        snap_map = std::move(trace.public_snapshot);
      }
      if (bt_) {
        tracker_observe(*bt_, *bundle_, *state_before, action, *bundle_->state,
                        actor);
      }
      if (!external_obs_mode_) {
        const int n = static_cast<int>(ai_views_.size());
        for (int p = 0; p < n; ++p) {
          advance_ai_view_(p, *state_before, action);
        }
      }
      ++ply_count_;
    }
    py::dict out;
    py::list ev_list;
    for (const auto& [kind, payload] : event_list) {
      py::dict e;
      e["kind"] = kind;
      py::dict p;
      for (const auto& [pk, pv] : payload) p[py::cast(pk)] = any_to_py(pv);
      e["payload"] = p;
      ev_list.append(e);
    }
    out["events"] = ev_list;
    py::dict snap;
    if (have_extractor) {
      for (const auto& [k, v] : snap_map) snap[py::cast(k)] = any_to_py(v);
    }
    out["public_snapshot"] = snap;
    out["action"] = observed;
    return out;
  }

  // Test/integration helper: extract the initial observation for a given
  // perspective from the truth state, the way the partner-side server would
  // before sending it to the AI. Two-section wire shape, structurally
  // identical to per-ply snapshot:
  //   {"public_snapshot": <walker viz::serialize_public_snapshot(state, perspective)>,
  //    "tracker_init":    <tracker.pack_init_payload(state, perspective)>}
  // The public_snapshot half carries every (idx, value) pair where
  // viz[idx, perspective]=1 plus the full perspective viz slice;
  // perspective-internal tracker bootstrap rides tracker_init.
  py::dict extract_initial_observation(int perspective) {
    AnyMap pub;
    AnyMap tk_init;
    {
      py::gil_scoped_release release;
      viz::serialize_public_snapshot(
          *bundle_->state, bundle_->state->schema_ref(), perspective, pub);
      if (bt_) {
        tk_init = bt_->pack_init_payload(*bundle_->state, perspective);
      }
    }
    py::dict pub_d;
    for (const auto& [k, v] : pub) pub_d[py::cast(k)] = any_to_py(v);
    py::dict tk_d;
    for (const auto& [k, v] : tk_init) tk_d[py::cast(k)] = any_to_py(v);
    py::dict out;
    out["public_snapshot"] = pub_d;
    out["tracker_init"] = tk_d;
    return out;
  }

  // External-engine interop (GameBundle::external_state_loader): replace the
  // truth state, mid-game, with one exported by another engine of the same
  // game. Each AI view is rebuilt from it the way a peer session bootstraps
  // from extract_initial_observation; loaded models are kept.
  void load_external_state(const std::string& json) {
    if (!bundle_->external_state_loader) {
      throw std::runtime_error("load_external_state: game '" + game_id_ +
                               "' registers no external_state_loader");
    }
    if (external_obs_mode_) {
      throw std::runtime_error(
          "load_external_state: this session holds an AI view "
          "(apply_initial_observation); load into a truth session");
    }
    py::gil_scoped_release release;
    bundle_->external_state_loader(*bundle_->state, json);
    for (int p = 0; p < static_cast<int>(ai_views_.size()); ++p) {
      ai_views_[p] = make_masked_state(*bundle_->state,
                                       bundle_->state->schema_ref(), p);
      if (ai_trackers_[p]) {
        const AnyMap payload =
            bt_ ? bt_->pack_init_payload(*bundle_->state, p) : AnyMap{};
        tracker_init(*ai_trackers_[p], *bundle_, *ai_views_[p], p, payload);
      }
    }
  }

  // The pending decision as (other engine's option key, atom sequence)
  // pairs, restricted to the atoms already applied. See
  // GameBundle::external_option_lister.
  std::vector<std::pair<std::string, std::vector<int>>> external_options() const {
    if (!bundle_->external_option_lister) {
      throw std::runtime_error("external_options: game '" + game_id_ +
                               "' registers no external_option_lister");
    }
    std::vector<std::pair<std::string, std::vector<int>>> out;
    {
      py::gil_scoped_release release;
      for (auto& [key, atoms] : bundle_->external_option_lister(*bundle_->state)) {
        out.emplace_back(std::move(key), std::vector<int>(atoms.begin(), atoms.end()));
      }
    }
    return out;
  }

  // Partner-provided initial observation. Two-section wire shape mirrors
  // per-ply snapshot: `public_snapshot` is wholesale-applied via
  // `viz::apply_public_snapshot` (same primitive per-ply uses);
  // `tracker_init` is handed to `tracker.init` so it can write any
  // perspective-internal tracker memory. The session's viz=0 slots are
  // never touched here — sim-entry `randomize_unseen` on a clone is the
  // only path that reads them.
  void apply_initial_observation(int perspective_player, py::dict initial_obs) {
    AnyMap obs_map = py_dict_to_any_map(initial_obs);
    auto pub_it = obs_map.find("public_snapshot");
    auto tk_it = obs_map.find("tracker_init");
    if (pub_it == obs_map.end()) {
      throw std::runtime_error(
          "apply_initial_observation: missing 'public_snapshot' field");
    }
    if (pub_it->second.type() != typeid(AnyMap)) {
      throw std::runtime_error(
          "apply_initial_observation: 'public_snapshot' must be an AnyMap");
    }
    const AnyMap& pub = std::any_cast<const AnyMap&>(pub_it->second);
    AnyMap tk_payload;
    if (tk_it != obs_map.end()) {
      if (tk_it->second.type() != typeid(AnyMap)) {
        throw std::runtime_error(
            "apply_initial_observation: 'tracker_init' must be an AnyMap");
      }
      tk_payload = std::any_cast<const AnyMap&>(tk_it->second);
    }
    py::gil_scoped_release release;
    external_obs_mode_ = true;
    api_perspective_ = perspective_player;
    viz::apply_public_snapshot(
        *bundle_->state, bundle_->state->schema_ref(),
        perspective_player, pub);
    if (bt_) {
      bt_->init(*bundle_->state, perspective_player, tk_payload);
    }
  }

  // Adapter-only escape hatch: write a single schema slot directly.
  //
  // Used by `games/6nimmit/judge_adapter/adapter.py` to bootstrap an initial GT
  // state from a judge-supplied "this is your hand" payload before
  // calling `extract_initial_observation(seat)` to ship to the AI side.
  // Bypasses do_action_fast / viz on purpose — caller is responsible for
  // ensuring the writes match a state that the corresponding rules could
  // legally have produced (i.e. don't write something that would later
  // confuse legal_actions).
  //
  // Underscore prefix marks this as not part of the AI Pipeline contract
  // — production code (selfplay, web, AI API, training) must not call it.
  // Tests / external-judge adapters only.
  void _force_write_field(const std::string& name,
                          const std::vector<int>& idx,
                          py::object value) {
    std::any v;
    if (py::isinstance<py::int_>(value)) {
      v = std::any(value.cast<int>());
    } else if (py::isinstance<py::bool_>(value)) {
      v = std::any(value.cast<bool>());
    } else {
      throw std::runtime_error(
          "_force_write_field: value must be int or bool (got " +
          std::string(py::str(value.get_type().attr("__name__"))) + ")");
    }
    py::gil_scoped_release release;
    const viz::FieldId fid =
        viz::schema_lookup(bundle_->state->schema_ref(), name);
    if (fid == viz::kInvalidFieldId) {
      throw std::runtime_error("_force_write_field: unknown field '" + name + "'");
    }
    bundle_->state->write_field_slot(fid, idx, v);
  }

  // Test hook for the tail-solver contract "do_action_deterministic +
  // undo_action restores the state exactly". For every legal action `a` of
  // the truth state, on a fresh clone: do a; (if not terminal) do + undo up
  // to `nested` legal replies; undo a. Returns what an untouched clone
  // ("baseline") and each round-tripped clone look like — state dict, legacy
  // full hash, every perspective hash, step_count — so the test can compare.
  // The session itself is not modified.
  py::dict _debug_do_undo_roundtrip(int nested) {
    struct Row {
      ActionId action = -1;
      AnyMap state;
      std::uint64_t full_hash = 0;
      std::vector<std::uint64_t> perspective_hashes;
      std::uint32_t step_count = 0;
    };
    auto capture = [&](ActionId action, const IGameState& s) {
      Row r;
      r.action = action;
      if (bundle_->state_serializer) r.state = bundle_->state_serializer(s);
      r.full_hash = s.state_hash();
      for (int p = 0; p < s.num_players(); ++p) {
        r.perspective_hashes.push_back(s.state_hash_for_perspective(p));
      }
      r.step_count = s.step_count();
      return r;
    };
    Row baseline;
    std::vector<Row> rows;
    {
      py::gil_scoped_release release;
      const IGameRules& rules = *bundle_->rules;
      baseline = capture(-1, *bundle_->state->clone_state());
      for (ActionId a : rules.legal_actions(*bundle_->state)) {
        auto clone = bundle_->state->clone_state();
        const UndoToken tok = rules.do_action_deterministic(*clone, a);
        if (!clone->is_terminal()) {
          int done = 0;
          for (ActionId b : rules.legal_actions(*clone)) {
            if (done++ >= nested) break;
            const UndoToken tok_b = rules.do_action_deterministic(*clone, b);
            rules.undo_action(*clone, tok_b);
          }
        }
        rules.undo_action(*clone, tok);
        rows.push_back(capture(a, *clone));
      }
    }
    auto to_py = [](const Row& r) {
      py::dict d;
      d["action"] = static_cast<int>(r.action);
      py::dict st;
      for (const auto& [k, v] : r.state) st[py::cast(k)] = any_to_py(v);
      d["state"] = st;
      d["state_hash"] = r.full_hash;
      d["perspective_hashes"] = r.perspective_hashes;
      d["step_count"] = r.step_count;
      return d;
    };
    py::dict out;
    out["baseline"] = to_py(baseline);
    py::list rt;
    for (const auto& r : rows) rt.append(to_py(r));
    out["roundtrips"] = rt;
    return out;
  }

  // Return the belief tracker's serialized state as a dict. Canonical form:
  // two trackers with semantically identical beliefs return equal dicts.
  // Empty dict if no tracker or tracker holds no explicit state.
  py::dict get_belief_snapshot() {
    py::dict out;
    if (!bt_) return out;
    AnyMap m = bt_->serialize();
    for (const auto& [k, v] : m) out[py::cast(k)] = any_to_py(v);
    return out;
  }

  void configure_tail_solve(bool enabled, int depth_limit, std::int64_t node_budget,
                            float auxiliary_weight) {
    ts_enabled_ = enabled;
    ts_depth_limit_ = depth_limit;
    ts_node_budget_ = node_budget;
    ts_auxiliary_weight_ = auxiliary_weight;
  }

  py::dict get_ai_action(int simulations, double temperature,
                         float c_puct = 1.4f,
                         bool cover_root_edges = false,
                         std::string opponent_selection = "puct",
                         double temperature_initial = -1.0,
                         double temperature_final = -1.0,
                         int temperature_decay_plies = 0,
                         int world_cap = 0,
                         double dirichlet_alpha = 0.0,
                         double dirichlet_epsilon = 0.0,
                         int dirichlet_on_first_n_plies = 0,
                         int top_k = 0,
                         std::optional<float> fpu_reduction = std::nullopt,
                         bool uct2_arrival_visits = true) {
    const auto opp_sel = parse_opponent_selection(opponent_selection);
    py::gil_scoped_release release;

    // Resolve effective temperature from schedule + current ply. When
    // initial/final < 0 the schedule is disabled and `temperature` is
    // used as-is (back-compat with callers that don't pass schedule).
    search::TemperatureSchedule t_sched{};
    if (temperature_initial >= 0.0 || temperature_final >= 0.0) {
      t_sched.enabled = true;
      if (temperature_initial >= 0.0) {
        t_sched.has_initial = true;
        t_sched.initial = temperature_initial;
      }
      if (temperature_final >= 0.0) {
        t_sched.has_final = true;
        t_sched.final_ = temperature_final;
      }
      t_sched.decay_plies = temperature_decay_plies;
    }
    const double effective_temperature = search::resolve_linear_temperature(
        t_sched, temperature, static_cast<int>(ply_count_));

    const IGameRules& rules = filtered_rules_ ? *filtered_rules_ : *bundle_->rules;
    const int cp = bundle_->state->current_player();

    // Select the search state / evaluator / tracker.
    // External-obs mode (AI API): bundle_->state IS the AI view.
    // GameSession mode: route through perspective-specific ai_views_[cp].
    const IGameState* search_state = nullptr;
    const search::IPolicyValueEvaluator* eval_ptr = nullptr;
    IBeliefTracker* search_bt = nullptr;
    if (external_obs_mode_) {
      search_state = bundle_->state.get();
      eval_ptr = evaluator_.get();
      search_bt = bt_;
    } else {
      if (cp < 0 || cp >= static_cast<int>(ai_views_.size()) || !ai_views_[cp]) {
        throw std::runtime_error(
            "GameSession.get_ai_action: ai_view for current player not initialized");
      }
      search_state = ai_views_[cp].get();
      eval_ptr = ai_evaluators_[cp].get();
      search_bt = ai_trackers_[cp].get();
    }

    const auto legal = rules.legal_actions(*search_state);
    if (legal.empty()) {
      py::gil_scoped_acquire acquire;
      return py::dict();
    }

    if (!eval_ptr) {
      throw std::runtime_error(
          "GameSession.get_ai_action: no model loaded — create session with model_path");
    }

    // ISMCTS: root-sampling hidden info + DAG per-acting-player keying.
    search::NetMctsConfig mcts_cfg{};
    mcts_cfg.simulations = simulations;
    mcts_cfg.world_cap = world_cap;
    mcts_cfg.c_puct = c_puct;
    mcts_cfg.fpu_reduction = fpu_reduction;
    mcts_cfg.uct2_arrival_visits = uct2_arrival_visits;
    {
      const auto noise = search::resolve_root_dirichlet_noise(
          static_cast<float>(dirichlet_alpha),
          static_cast<float>(dirichlet_epsilon),
          dirichlet_on_first_n_plies, static_cast<int>(ply_count_));
      mcts_cfg.root_dirichlet_alpha = noise.alpha;
      mcts_cfg.root_dirichlet_epsilon = noise.epsilon;
    }
    mcts_cfg.cover_root_edges = cover_root_edges;
    mcts_cfg.opponent_selection = opp_sel;
    if (search_bt) {
      mcts_cfg.root_belief_tracker = search_bt;
    }
    if (bundle_->belief_feature_extractor && belief_evaluator_) {
      mcts_cfg.belief_extractor = bundle_->belief_feature_extractor.get();
      mcts_cfg.belief_evaluator = belief_evaluator_.get();
    }
    // Sim-tracker descent maintenance — events-only flavor.
    if (bundle_->events_only_extractor) {
      mcts_cfg.events_only_extractor = bundle_->events_only_extractor;
    }

    if (ts_enabled_ && bundle_->tail_solver) {
      int ply = static_cast<int>(ply_count_);
      bool try_ts = false;
      if (bundle_->tail_solve_trigger) {
        try_ts = bundle_->tail_solve_trigger(*search_state, ply);
      } else {
        try_ts = true;
      }
      if (try_ts) {
        mcts_cfg.tail_solve_enabled = true;
        mcts_cfg.tail_solve_config.depth_limit = ts_depth_limit_;
        mcts_cfg.tail_solve_config.node_budget = ts_node_budget_;
        mcts_cfg.tail_solve_config.auxiliary_weight = ts_auxiliary_weight_;
        mcts_cfg.tail_solver = bundle_->tail_solver.get();
      }
    }

    search::NetMcts mcts(mcts_cfg);
    search::NetMctsStats stats{};
    const std::uint64_t mcts_seed = board_ai::rng::derive_subseed(
        seed_, "session.mcts", static_cast<std::uint64_t>(ply_count_));
    mcts.search_root(*search_state, rules, *bundle_->value_model,
                      *eval_ptr, &stats, mcts_seed);

    // Deterministic action_seed: any non-deterministic component (e.g.
    // wall-clock) here would make argmax tie-breaks flaky across runs
    // even at temperature=0. Derive from (seed_, ply_count_) instead so
    // the session's action sequence is fully reproducible from its seed.
    std::uint64_t action_seed = mcts_seed ^ 0xBF58476D1CE4E5B9ULL;
    ActionId chosen = search::select_action_from_visits(
        stats.root_actions, stats.root_action_visits,
        effective_temperature, action_seed, top_k, legal[0]);

    py::gil_scoped_acquire acquire;
    py::dict out;
    out["action"] = chosen;
    out["action_info"] = get_action_info(chosen);
    py::dict st;
    st["simulations"] = stats.simulations_done;
    st["best_value"] = stats.best_action_value;
    py::list rv;
    for (double v : stats.root_values) rv.append(v);
    st["root_values"] = rv;
    py::dict wm;
    for (size_t ei = 0; ei < stats.root_actions.size(); ++ei) {
      if (ei < stats.root_edge_values.size()) {
        py::list vals;
        for (double v : stats.root_edge_values[ei]) vals.append(v);
        wm[py::cast(static_cast<int>(stats.root_actions[ei]))] = vals;
      }
    }
    st["action_values"] = wm;
    // Expose root visit distribution so tests can compare the full
    // MCTS policy across paths (selfplay vs API), not just argmax.
    // This is the signal that would catch a subtle info leak biasing
    // one path's priors without flipping the top pick.
    py::list root_actions_py;
    py::list root_visits_py;
    for (size_t ei = 0; ei < stats.root_actions.size(); ++ei) {
      root_actions_py.append(static_cast<int>(stats.root_actions[ei]));
      int v = ei < stats.root_action_visits.size() ? stats.root_action_visits[ei] : 0;
      root_visits_py.append(v);
    }
    st["root_actions"] = root_actions_py;
    st["root_action_visits"] = root_visits_py;
    st["tail_solved"] = stats.tail_solved;
    st["tail_solve_value"] = stats.tail_solve_value;
    st["tail_solve_attempted"] = stats.tail_solve_attempted;
    st["tail_solve_completed"] = stats.tail_solve_completed;
    st["tail_solve_elapsed_ms"] = stats.tail_solve_elapsed_ms;
    // Outcome encoded so the web layer can render "0% 残局已求解 / 100% / 平局"
    // without re-running the solver. Mirrors TailSolveOutcome enum integer.
    st["tail_solve_outcome"] = static_cast<int>(stats.tail_solve_outcome);
    st["dag_reuse_hits"] = stats.dag_reuse_hits;
    st["expanded_nodes"] = stats.expanded_nodes;
    st["max_depth_reached"] = stats.max_depth_reached;
    st["simulations"] = stats.simulations_done;
    out["stats"] = st;
    return out;
  }

  py::dict get_heuristic_action(const std::string& selector,
                                double temperature) {
    if (selector.empty() || !bundle_->heuristics.has(selector)) {
      throw std::runtime_error(
          "get_heuristic_action: heuristic '" + selector +
          "' not registered for game '" + game_id_ + "'");
    }
    const board_ai::IHeuristic& heuristic = bundle_->heuristics.get(selector);

    ActionId chosen;
    {
      py::gil_scoped_release release;
      // GameSession's heuristic call routes through the acting player's
      // truth-free ai_view (same view MCTS reads from). Mask once more
      // before dispatch so viz=0 reads land on placeholder.
      // External-obs mode (AI API): bundle_->state IS the AI view, as in
      // get_ai_action; ai_views_ stopped advancing when observations began.
      const int cp = bundle_->state->current_player();
      IGameState* base = bundle_->state.get();
      if (!external_obs_mode_ && bundle_->public_event_extractor &&
          cp >= 0 && cp < static_cast<int>(ai_views_.size()) && ai_views_[cp]) {
        base = ai_views_[cp].get();
      }
      auto masked = make_masked_state(*base, base->schema_ref(), cp);
      // Use the perspective's ai_tracker (kept in sync via observe_public_event
      // on every advance), not bundle_->belief_tracker which is never init'd
      // in GameSession non-external mode and would feed a blank belief into
      // any tracker-aware heuristic.
      const IBeliefTracker* hr_tracker = nullptr;
      if (external_obs_mode_) {
        hr_tracker = bt_;
      } else if (cp >= 0 && cp < static_cast<int>(ai_trackers_.size())) {
        hr_tracker = ai_trackers_[cp].get();
      }
      auto hr = heuristic.pick(
          *masked, hr_tracker, *bundle_->rules, seed_ ^ 0xDEAD);
      if (hr.actions.empty()) {
        py::gil_scoped_acquire acquire;
        return py::dict();
      }

      const std::uint64_t sample_seed = board_ai::rng::derive_subseed(
          seed_, "session.heuristic_sample",
          static_cast<std::uint64_t>(ply_count_));
      double u01 = static_cast<double>(sample_seed & 0xFFFFFFFF) / 4294967296.0;
      const size_t idx = runtime::sample_heuristic_index(
          hr.scores, temperature, u01);
      chosen = hr.actions[idx];
    }

    py::dict out;
    out["action"] = chosen;
    out["action_info"] = get_action_info(chosen);
    return out;
  }

  py::dict apply_ai_action(int simulations, double temperature) {
    auto result = get_ai_action(simulations, temperature);
    if (result.contains("action")) {
      apply_action(py::cast<ActionId>(result["action"]));
    }
    return result;
  }

  // Test-only: run MCTS with the on_sim_step callback enabled, capturing
  // the cloned sim_tracker's serialize() at every descent step of every
  // sim. Returns the root tracker snapshot taken at sim entry (before any
  // descent step) plus a list of (sim, depth, sim_tracker_snapshot) tuples.
  // Used by test_sim_tracker_descent_maintained to assert the sim_tracker
  // actually evolves during descent (not frozen at root).
  //
  // This routes through the same `get_ai_action` MCTS config plumbing —
  // no path divergence — but with `on_sim_step` set to a recorder. The
  // returned action is discarded; callers care only about the traces.
  py::dict _debug_run_mcts_with_sim_traces(int simulations) {
    const IGameRules& rules = filtered_rules_ ? *filtered_rules_ : *bundle_->rules;
    const int cp = bundle_->state->current_player();

    const IGameState* search_state = nullptr;
    const search::IPolicyValueEvaluator* eval_ptr = nullptr;
    IBeliefTracker* search_bt = nullptr;
    if (external_obs_mode_) {
      search_state = bundle_->state.get();
      eval_ptr = evaluator_.get();
      search_bt = bt_;
    } else {
      if (cp < 0 || cp >= static_cast<int>(ai_views_.size()) || !ai_views_[cp]) {
        throw std::runtime_error(
            "_debug_run_mcts_with_sim_traces: ai_view for current player not initialized");
      }
      search_state = ai_views_[cp].get();
      eval_ptr = ai_evaluators_[cp].get();
      search_bt = ai_trackers_[cp].get();
    }

    if (!eval_ptr) {
      throw std::runtime_error(
          "_debug_run_mcts_with_sim_traces: no model loaded");
    }
    if (!search_bt) {
      throw std::runtime_error(
          "_debug_run_mcts_with_sim_traces: game has no belief tracker registered");
    }

    // Capture root tracker snapshot BEFORE the search call — this is
    // what every sim's clone starts from after randomize_unseen seeds it.
    AnyMap root_snapshot = search_bt->serialize();

    // Recorder: appends (sim_index, step_index, serialize()) tuples.
    struct Record { int sim; int step; AnyMap snap; };
    std::vector<Record> records;

    search::NetMctsConfig mcts_cfg{};
    mcts_cfg.simulations = simulations;
    mcts_cfg.c_puct = 1.4f;
    mcts_cfg.root_belief_tracker = search_bt;
    if (bundle_->events_only_extractor) {
      mcts_cfg.events_only_extractor = bundle_->events_only_extractor;
    }
    mcts_cfg.on_sim_step = [&records](int sim, int step,
                                       const IGameState&,
                                       const IBeliefTracker* tr) {
      if (!tr) return;
      records.push_back(Record{sim, step, tr->serialize()});
    };

    {
      py::gil_scoped_release release;
      search::NetMcts mcts(mcts_cfg);
      search::NetMctsStats stats{};
      const std::uint64_t mcts_seed = board_ai::rng::derive_subseed(
          seed_, "session.mcts.debug", static_cast<std::uint64_t>(ply_count_));
      mcts.search_root(*search_state, rules, *bundle_->value_model,
                        *eval_ptr, &stats, mcts_seed);
    }

    py::dict out;
    py::dict root_dict;
    for (const auto& [k, v] : root_snapshot) root_dict[py::cast(k)] = any_to_py(v);
    out["root_tracker"] = root_dict;

    py::list traces;
    for (const auto& r : records) {
      py::dict entry;
      entry["sim"] = r.sim;
      entry["step"] = r.step;
      py::dict snap;
      for (const auto& [k, v] : r.snap) snap[py::cast(k)] = any_to_py(v);
      entry["tracker"] = snap;
      traces.append(entry);
    }
    out["sim_traces"] = traces;
    return out;
  }

  // Test-only: run MCTS with the on_descent_edge callback enabled and
  // return every descent step as (sim, step, parent_node, edge_index,
  // child_node, uct2_parent_visits), in execution order. Node ids are
  // per-search DAG indices (0 = root). test_uct2_arrival_visits recounts
  // the parent visits from these paths.
  py::dict _debug_run_mcts_with_edge_traces(int simulations,
                                            bool uct2_arrival_visits,
                                            int leaf_batch_size) {
    const IGameRules& rules = filtered_rules_ ? *filtered_rules_ : *bundle_->rules;
    const int cp = bundle_->state->current_player();

    const IGameState* search_state = nullptr;
    const search::IPolicyValueEvaluator* eval_ptr = nullptr;
    IBeliefTracker* search_bt = nullptr;
    if (external_obs_mode_) {
      search_state = bundle_->state.get();
      eval_ptr = evaluator_.get();
      search_bt = bt_;
    } else {
      if (cp < 0 || cp >= static_cast<int>(ai_views_.size()) || !ai_views_[cp]) {
        throw std::runtime_error(
            "_debug_run_mcts_with_edge_traces: ai_view for current player not initialized");
      }
      search_state = ai_views_[cp].get();
      eval_ptr = ai_evaluators_[cp].get();
      search_bt = ai_trackers_[cp].get();
    }
    if (!eval_ptr) {
      throw std::runtime_error(
          "_debug_run_mcts_with_edge_traces: no model loaded");
    }

    struct Record { int sim; int step; int parent; int edge; int child; int visits; };
    std::vector<Record> records;

    search::NetMctsConfig mcts_cfg{};
    mcts_cfg.simulations = simulations;
    mcts_cfg.c_puct = 1.4f;
    mcts_cfg.uct2_arrival_visits = uct2_arrival_visits;
    mcts_cfg.leaf_batch_size = leaf_batch_size;
    if (search_bt) {
      mcts_cfg.root_belief_tracker = search_bt;
    }
    if (bundle_->events_only_extractor) {
      mcts_cfg.events_only_extractor = bundle_->events_only_extractor;
    }
    mcts_cfg.on_descent_edge = [&records](int sim, int step, int parent,
                                          int edge, int child, int visits) {
      records.push_back(Record{sim, step, parent, edge, child, visits});
    };

    search::NetMctsStats stats{};
    {
      py::gil_scoped_release release;
      search::NetMcts mcts(mcts_cfg);
      // Same seed as get_ai_action: with its default arguments (and no
      // belief model) the two run the identical search, so tests can tie
      // get_ai_action's switch to the counts recorded here.
      const std::uint64_t mcts_seed = board_ai::rng::derive_subseed(
          seed_, "session.mcts", static_cast<std::uint64_t>(ply_count_));
      mcts.search_root(*search_state, rules, *bundle_->value_model,
                        *eval_ptr, &stats, mcts_seed);
    }

    py::list edges;
    for (const auto& r : records) {
      edges.append(py::make_tuple(r.sim, r.step, r.parent, r.edge, r.child, r.visits));
    }
    py::dict out;
    out["edges"] = edges;
    out["simulations"] = stats.simulations_done;
    out["root_action_visits"] = stats.root_action_visits;
    return out;
  }

 private:
  std::string game_id_;
  std::uint64_t seed_;
  std::string model_path_;
  std::unique_ptr<GameBundle> bundle_;
  std::unique_ptr<infer::OnnxPolicyValueEvaluator> evaluator_;
  std::unique_ptr<infer::OnnxBeliefEvaluator> belief_evaluator_;
  std::unique_ptr<runtime::FilteredRulesWrapper> filtered_rules_;
  IBeliefTracker* bt_ = nullptr;
  // Perspective seat that bt_ (the session-level tracker on bundle_->state)
  // is bound to. Set by apply_initial_observation; defaults to 0 when the
  // session was constructed without an external observation. Used as the
  // observer argument to randomize_unseen on bundle_->state in
  // external_obs_mode.
  int api_perspective_ = 0;
  std::size_t ply_count_ = 0;
  // GT step rng. Seeded directly from seed_ (no subseed derivation) so
  // selfplay's gt_step rng with the same episode_seed produces identical
  // truth draws — `test_public_snapshot_round_trip` replays selfplay
  // traces through GameSession.apply_action and depends on this match.
  std::mt19937_64 step_rng_{seed_};
  bool ts_enabled_ = false;
  int ts_depth_limit_ = 10;
  std::int64_t ts_node_budget_ = 200000;
  float ts_auxiliary_weight_ = 0.0f;

  // Per-perspective AI views. Populated by the constructor via the
  // initial-observation protocol (unless external_obs_mode_ flips on
  // first — see comments at init_ai_views_). get_ai_action routes MCTS
  // through these so searches never read hidden fields from bundle_->state.
  std::vector<std::unique_ptr<IGameState>> ai_views_;
  std::vector<std::unique_ptr<IBeliefTracker>> ai_trackers_;
  std::vector<std::unique_ptr<IFeatureEncoder>> ai_encoders_;
  std::vector<std::unique_ptr<infer::OnnxPolicyValueEvaluator>> ai_evaluators_;
  bool external_obs_mode_ = false;
};

py::dict test_belief_tracker_py(
    const std::string& game_id,
    std::uint64_t seed,
    int plies,
    int randomize_trials) {
  py::gil_scoped_release release;

  auto bundle = GameRegistry::instance().create_game(game_id, seed);
  if (!bundle.belief_tracker) {
    py::gil_scoped_acquire acquire;
    throw std::runtime_error("test_belief_tracker: game must have belief_tracker");
  }

  auto state = bundle.state->clone_state();
  IBeliefTracker* bt = bundle.belief_tracker.get();
  tracker_init(*bt, bundle, *state, 0);

  std::mt19937_64 action_rng(seed);
  std::mt19937_64 step_rng(
      board_ai::rng::derive_subseed(seed, "test_belief_tracker.gt_step"));
  int actual_plies = 0;
  for (int i = 0; i < plies && !state->is_terminal(); ++i) {
    auto legal = bundle.rules->legal_actions(*state);
    if (legal.empty()) break;
    const size_t idx = action_rng() % legal.size();
    ActionId chosen = legal[idx];
    auto state_before = state->clone_state();
    bundle.rules->do_action_fast(*state, chosen, step_rng);
    tracker_observe(*bt, bundle, *state_before, chosen, *state, 0);
    actual_plies += 1;
  }

  auto extract_deck_ids = [&](const IGameState& gs) -> std::vector<int> {
    if (!bundle.state_serializer) return {};
    auto m = bundle.state_serializer(gs);
    std::vector<int> result;
    auto it = m.find("_test_all_deck_ids");
    if (it != m.end()) {
      result = std::any_cast<std::vector<int>>(it->second);
    }
    return result;
  };

  auto extract_tableau_ids = [&](const IGameState& gs) -> std::vector<int> {
    if (!bundle.state_serializer) return {};
    auto m = bundle.state_serializer(gs);
    std::vector<int> result;
    auto it = m.find("_test_tableau_ids");
    if (it != m.end()) {
      result = std::any_cast<std::vector<int>>(it->second);
    }
    return result;
  };

  std::vector<int> original_deck = extract_deck_ids(*state);
  std::vector<int> tableau_ids = extract_tableau_ids(*state);

  AnyMap belief_snapshot = bt->serialize();
  AnyMap original_state_map;
  if (bundle.state_serializer) {
    original_state_map = bundle.state_serializer(*state);
  }

  // Observer-perspective hash before / after each randomize_unseen: lets
  // tests assert that sampling leaves every observer-visible slot untouched.
  const std::uint64_t original_hash = state->state_hash_for_perspective(0);
  std::vector<std::uint64_t> trial_hashes;
  std::vector<std::vector<int>> trial_decks;
  std::vector<AnyMap> trial_state_maps;
  trial_decks.reserve(static_cast<size_t>(randomize_trials));
  trial_state_maps.reserve(static_cast<size_t>(randomize_trials));
  for (int t = 0; t < randomize_trials; ++t) {
    auto clone = state->clone_state();
    std::mt19937_64 trial_rng(board_ai::rng::derive_subseed(
        seed, "test_belief_tracker.trial", static_cast<std::uint64_t>(t + 1)));
    bt->randomize_unseen(*clone, /*observer=*/0, trial_rng);
    trial_hashes.push_back(clone->state_hash_for_perspective(0));
    trial_decks.push_back(extract_deck_ids(*clone));
    if (bundle.state_serializer) {
      trial_state_maps.push_back(bundle.state_serializer(*clone));
    } else {
      trial_state_maps.push_back({});
    }
  }

  py::gil_scoped_acquire acquire;
  py::dict out;
  out["plies"] = actual_plies;
  out["original_deck"] = original_deck;
  out["tableau_cards"] = tableau_ids;
  py::list trials;
  for (const auto& d : trial_decks) {
    trials.append(py::cast(d));
  }
  out["trial_decks"] = trials;
  out["original_hash"] = original_hash;
  out["trial_hashes"] = trial_hashes;

  py::dict bs;
  for (const auto& [k, v] : belief_snapshot) bs[py::cast(k)] = any_to_py(v);
  out["belief_snapshot"] = bs;

  py::dict orig_state;
  for (const auto& [k, v] : original_state_map) orig_state[py::cast(k)] = any_to_py(v);
  out["original_state"] = orig_state;

  py::list trial_states;
  for (const auto& m : trial_state_maps) {
    py::dict d;
    for (const auto& [k, v] : m) d[py::cast(k)] = any_to_py(v);
    trial_states.append(d);
  }
  out["trial_states"] = trial_states;
  return out;
}

}  // namespace

PYBIND11_MODULE(dinoboard_engine, m) {
  m.doc() = "DinoBoard C++ engine bindings";

  m.def("run_selfplay_episode", &run_selfplay_episode_py,
      py::arg("game_id"),
      py::arg("seed"),
      py::arg("model_path") = "",
      py::arg("simulations") = 200,
      py::arg("world_cap") = 0,
      py::arg("c_puct") = 1.4f,
      py::arg("temperature") = 1.0,
      py::arg("dirichlet_alpha") = 0.3,
      py::arg("dirichlet_epsilon") = 0.25,
      py::arg("dirichlet_on_first_n_plies") = 30,
      py::arg("max_game_plies") = 500,
      py::arg("tail_solve_enabled") = false,
      py::arg("tail_solve_depth_limit") = 5,
      py::arg("tail_solve_node_budget") = 10000000LL,
      py::arg("tail_solve_auxiliary_weight") = 0.0f,
      py::arg("temperature_initial") = -1.0,
      py::arg("temperature_final") = -1.0,
      py::arg("temperature_decay_plies") = 0,
      py::arg("heuristic_guidance_ratio") = 0.0,
      py::arg("heuristic_temperature") = 0.0,
      py::arg("training_filter_ratio") = 1.0,
      py::arg("trace_perspective") = -1,
      py::arg("opponent_selection") = std::string("puct"),
      py::arg("belief_model_path") = std::string(""),
      py::arg("heuristic_selector") = std::string(""),
      py::arg("safe_mode") = false,
      py::arg("top_k") = 0,
      py::arg("fpu_reduction") = py::none(),
      py::arg("uct2_arrival_visits").noconvert() = true);

  m.def("run_selfplay_episodes_batch", &run_selfplay_episodes_batch_py,
      py::arg("game_id"),
      py::arg("seeds"),
      py::arg("model_paths"),
      py::arg("simulations") = 200,
      py::arg("world_cap") = 0,
      py::arg("c_puct") = 1.4f,
      py::arg("temperature") = 1.0,
      py::arg("dirichlet_alpha") = 0.3,
      py::arg("dirichlet_epsilon") = 0.25,
      py::arg("dirichlet_on_first_n_plies") = 30,
      py::arg("max_game_plies") = 500,
      py::arg("tail_solve_enabled") = false,
      py::arg("tail_solve_depth_limit") = 5,
      py::arg("tail_solve_node_budget") = 10000000LL,
      py::arg("tail_solve_auxiliary_weight") = 0.0f,
      py::arg("temperature_initial") = -1.0,
      py::arg("temperature_final") = -1.0,
      py::arg("temperature_decay_plies") = 0,
      py::arg("heuristic_guidance_ratio") = 0.0,
      py::arg("heuristic_temperature") = 0.0,
      py::arg("training_filter_ratio") = 1.0,
      py::arg("opponent_selection") = std::string("puct"),
      py::arg("belief_model_path") = std::string(""),
      py::arg("heuristic_selector") = std::string(""),
      py::arg("latest_seat") = -1,
      py::arg("safe_mode") = false,
      py::arg("top_k") = 0,
      py::arg("belief_model_paths") = std::vector<std::string>{},
      py::arg("fpu_reduction") = py::none(),
      py::arg("uct2_arrival_visits").noconvert() = true);

  m.def("run_concurrent_selfplay_step", &run_concurrent_selfplay_step_py,
      py::arg("game_id"),
      py::arg("seeds"),
      py::arg("latest_model_path"),
      py::arg("pool_paths") = std::vector<std::string>{},
      py::arg("num_self_threads"),
      py::arg("num_pool_threads") = 0,
      py::arg("opponent_pool_max_distinct") = 3,
      py::arg("simulations") = 200,
      py::arg("world_cap") = 0,
      py::arg("c_puct") = 1.4f,
      py::arg("temperature") = 1.0,
      py::arg("dirichlet_alpha") = 0.3,
      py::arg("dirichlet_epsilon") = 0.25,
      py::arg("dirichlet_on_first_n_plies") = 30,
      py::arg("max_game_plies") = 500,
      py::arg("tail_solve_enabled") = false,
      py::arg("tail_solve_depth_limit") = 5,
      py::arg("tail_solve_node_budget") = 10000000LL,
      py::arg("tail_solve_auxiliary_weight") = 0.0f,
      py::arg("temperature_initial") = -1.0,
      py::arg("temperature_final") = -1.0,
      py::arg("temperature_decay_plies") = 0,
      py::arg("heuristic_guidance_ratio") = 0.0,
      py::arg("heuristic_temperature") = 0.0,
      py::arg("heuristic_selector") = std::string(""),
      py::arg("training_filter_ratio") = 1.0,
      py::arg("opponent_selection") = std::string("puct"),
      py::arg("belief_model_path") = std::string(""),
      py::arg("batch_size") = std::size_t{32},
      py::arg("batch_timeout_us") = std::size_t{50},
      py::arg("max_queue_size") = std::size_t{256},
      py::arg("intra_op_num_threads") = 2,
      py::arg("num_inference_threads") = 1,
      py::arg("providers") = std::vector<std::string>{},
      py::arg("devices") = std::vector<int>{},
      py::arg("safe_mode") = false,
      py::arg("top_k") = 0,
      py::arg("pool_belief_paths") = std::vector<std::string>{},
      py::arg("leaf_batch_size") = 1,
      py::arg("fast_simulations") = 0,
      py::arg("full_search_prob") = 1.0,
      py::arg("fpu_reduction") = py::none(),
      py::arg("compact_samples") = false,
      py::arg("uct2_arrival_visits").noconvert() = true);

  m.def("run_arena_match", &run_arena_match_py,
      py::arg("game_id"),
      py::arg("seed"),
      py::arg("model_paths") = std::vector<std::string>{},
      py::arg("simulations_list") = std::vector<int>{},
      py::arg("world_cap_list") = std::vector<int>{},
      py::arg("temperature") = 0.0,
      py::arg("max_game_plies") = 500,
      py::arg("tail_solve") = false,
      py::arg("tail_solve_depth_limit") = 10,
      py::arg("tail_solve_node_budget") = std::int64_t{200000},
      py::arg("tail_solve_auxiliary_weight") = 0.0f,
      py::arg("opponent_selection_list") = std::vector<std::string>{},
      py::arg("belief_model_path") = std::string(""),
      py::arg("seat_configs") = std::vector<py::dict>{},
      py::arg("safe_mode") = false,
      py::arg("top_k") = 0);

  m.def("run_concurrent_arena_batch", &run_concurrent_arena_batch_py,
      py::arg("game_id"),
      py::arg("tasks"),
      py::arg("num_threads"),
      py::arg("max_game_plies") = 500,
      py::arg("batch_size") = std::size_t{32},
      py::arg("batch_timeout_us") = std::size_t{50},
      py::arg("max_queue_size") = std::size_t{256},
      py::arg("intra_op_num_threads") = 2,
      py::arg("num_inference_threads") = 1,
      py::arg("providers") = std::vector<std::string>{},
      py::arg("devices") = std::vector<int>{},
      py::arg("safe_mode") = false);

  m.def("run_concurrent_eval_vs_heuristic_batch",
      &run_concurrent_eval_vs_heuristic_batch_py,
      py::arg("game_id"),
      py::arg("model_path"),
      py::arg("heuristic_selector"),
      py::arg("tasks"),
      py::arg("num_threads"),
      py::arg("simulations") = 200,
      py::arg("world_cap") = 0,
      py::arg("c_puct") = 1.4f,
      py::arg("temperature") = 0.0,
      py::arg("heuristic_temperature") = 0.0,
      py::arg("max_game_plies") = 500,
      py::arg("tail_solve_enabled") = false,
      py::arg("tail_solve_depth_limit") = 5,
      py::arg("tail_solve_node_budget") = std::int64_t{10000000},
      py::arg("tail_solve_auxiliary_weight") = 0.0f,
      py::arg("opponent_selection") = std::string("puct"),
      py::arg("belief_model_path") = std::string(""),
      py::arg("batch_size") = std::size_t{32},
      py::arg("batch_timeout_us") = std::size_t{50},
      py::arg("max_queue_size") = std::size_t{256},
      py::arg("intra_op_num_threads") = 2,
      py::arg("num_inference_threads") = 1,
      py::arg("providers") = std::vector<std::string>{},
      py::arg("devices") = std::vector<int>{},
      py::arg("constrained") = false,
      py::arg("safe_mode") = false,
      py::arg("top_k") = 0,
      py::arg("leaf_batch_size") = 1,
      py::arg("fpu_reduction") = py::none(),
      py::arg("uct2_arrival_visits").noconvert() = true);

  m.def("run_constrained_eval_vs_heuristic", &run_constrained_eval_vs_heuristic_py,
      py::arg("game_id"),
      py::arg("seed"),
      py::arg("model_path"),
      py::arg("simulations") = 200,
      py::arg("world_cap") = 0,
      py::arg("c_puct") = 1.4f,
      py::arg("temperature") = 0.0,
      py::arg("model_is_player") = 0,
      py::arg("constrained") = true,
      py::arg("heuristic_temperature") = 0.0,
      py::arg("opponent_selection") = std::string("puct"),
      py::arg("tail_solve_enabled") = false,
      py::arg("tail_solve_depth_limit") = 10,
      py::arg("tail_solve_node_budget") = 200000LL,
      py::arg("tail_solve_auxiliary_weight") = 0.0f,
      py::arg("belief_model_path") = std::string(""),
      py::arg("heuristic_selector") = std::string(""),
      py::arg("safe_mode") = false,
      py::arg("top_k") = 0,
      py::arg("max_game_plies") = 500,
      py::arg("fpu_reduction") = py::none(),
      py::arg("uct2_arrival_visits").noconvert() = true);

  m.def("run_heuristic_episode", &run_heuristic_episode_py,
      py::arg("game_id"),
      py::arg("seed"),
      py::arg("temperature") = 0.0,
      py::arg("max_game_plies") = 200,
      py::arg("heuristic_selector") = std::string(""));

  m.def("run_heuristic_vs_heuristic", &run_heuristic_vs_heuristic_py,
      py::arg("game_id"),
      py::arg("seed"),
      py::arg("max_game_plies") = 200,
      py::arg("heuristic_names") = std::vector<std::string>{},
      py::arg("temperatures") = std::vector<double>{});

  m.def("encode_state", &encode_state_py,
      py::arg("game_id"),
      py::arg("seed") = 0xC0FFEE);

  m.def("encode_state_for_perspective", &encode_state_for_perspective_py,
      py::arg("game_id"),
      py::arg("seed") = 0xC0FFEE,
      py::arg("encode_player") = 0,
      py::arg("tracker_perspective") = -1,
      py::arg("event_actions") = py::list());

  m.def("tail_solve", &tail_solve_py,
      py::arg("game_id"),
      py::arg("seed"),
      py::arg("perspective_player") = 0,
      py::arg("depth_limit") = 10,
      py::arg("node_budget") = 200000LL);

  m.def("test_belief_tracker", &test_belief_tracker_py,
      py::arg("game_id"),
      py::arg("seed"),
      py::arg("plies") = 20,
      py::arg("randomize_trials") = 10);

  m.def("game_metadata", [](const std::string& game_id) -> py::dict {
    py::gil_scoped_release release;
    auto bundle = GameRegistry::instance().create_game(game_id, 0);
    const int num_players = bundle.state->num_players();
    const int action_space = bundle.encoder->action_space();
    const int feature_dim = bundle.encoder->feature_dim();
    const bool has_public_state_applier = static_cast<bool>(bundle.public_state_applier);
    const bool has_tail_solver = static_cast<bool>(bundle.tail_solver);
    const bool has_tail_solve_trigger = static_cast<bool>(bundle.tail_solve_trigger);
    const bool has_final_score_extractor = static_cast<bool>(bundle.final_score_extractor);
    const bool has_belief_extractor =
        static_cast<bool>(bundle.belief_feature_extractor);
    const int belief_feature_dim =
        has_belief_extractor ? bundle.belief_feature_extractor->feature_dim() : 0;
    const int belief_logit_count =
        has_belief_extractor ? bundle.belief_feature_extractor->output_logit_count()
                             : 0;
    py::gil_scoped_acquire acquire;
    py::dict out;
    out["num_players"] = num_players;
    out["action_space"] = action_space;
    out["feature_dim"] = feature_dim;
    // Capability flag: snapshot-path games register a public_state_applier;
    // fully-public games (tictactoe, quoridor) don't. The REST AI API uses
    // this to dispatch between the action_id-only path and the full
    // apply_observation path. extract_initial_observation /
    // apply_initial_observation are walker-driven and available for every
    // game, so no separate has_initial_observation_applier flag is needed.
    out["has_public_state_applier"] = has_public_state_applier;
    out["has_tail_solver"] = has_tail_solver;
    out["has_tail_solve_trigger"] = has_tail_solve_trigger;
    out["has_final_score_extractor"] = has_final_score_extractor;
    out["has_belief_extractor"] = has_belief_extractor;
    out["belief_feature_dim"] = belief_feature_dim;
    out["belief_logit_count"] = belief_logit_count;
    if (const TokenLayout* tl = bundle.encoder->token_layout()) {
      py::dict layout;
      layout["num_tokens"] = tl->num_tokens;
      layout["token_dim"] = tl->token_dim;
      layout["num_token_types"] = tl->num_token_types;
      layout["num_entity_ids"] = tl->num_entity_ids;
      layout["num_pointer_heads"] = tl->num_pointer_heads;
      layout["token_types"] = tl->token_types;
      layout["action_token"] = tl->action_token;
      layout["action_head"] = tl->action_head;
      layout["num_anchor_ids"] = tl->num_anchor_ids;
      layout["anchor_action"] = tl->anchor_action;
      layout["variable_tokens"] = tl->variable_tokens;
  layout["token_positions"] = tl->token_positions;
      out["token_layout"] = layout;
    } else {
      out["token_layout"] = py::none();
    }
    return out;
  }, py::arg("game_id"));

  // The encoder's board symmetries (IFeatureEncoder::symmetries, see
  // EncoderSymmetry) for training-time augmentation; [] if it has none.
  m.def("encoder_symmetries", [](const std::string& game_id) -> py::list {
    std::vector<EncoderSymmetry> syms;
    {
      py::gil_scoped_release release;
      syms = GameRegistry::instance().create_game(game_id, 0).encoder->symmetries();
    }
    py::list out;
    for (const auto& s : syms) {
      py::dict d;
      d["feature_map"] = s.feature_map;
      d["anchor_map"] = s.anchor_map;
      d["position_affine"] = std::vector<int>(s.position_affine.begin(), s.position_affine.end());
      d["column_map"] = s.column_map;
      d["action_map"] = s.action_map;
      out.append(d);
    }
    return out;
  }, py::arg("game_id"));

  // Resolved settings of a registered heuristic, read from the object the
  // engine actually instantiates (not from heuristic.json), so benchmark
  // logs can state what the heuristic opponent searches with. "mcts_pv"
  // entries report the NetMctsConfig their pick() uses; anything else is an
  // opaque C++ rule and reports only its type.
  m.def("heuristic_settings", [](const std::string& game_id,
                                 const std::string& name) -> py::dict {
    std::unique_ptr<GameBundle> bundle;
    {
      py::gil_scoped_release release;
      bundle = std::make_unique<GameBundle>(
          GameRegistry::instance().create_game(game_id, 0));
    }
    if (name.empty() || !bundle->heuristics.has(name)) {
      throw std::runtime_error(
          "heuristic_settings: heuristic '" + name +
          "' not registered for game '" + game_id + "'");
    }
    const board_ai::IHeuristic& h = bundle->heuristics.get(name);
    py::dict out;
    out["name"] = name;
    const auto* pvh =
        dynamic_cast<const board_ai::heuristic::MctsPVHeuristic*>(&h);
    if (!pvh) {
      out["type"] = "rule";
      return out;
    }
    const search::NetMctsConfig cfg = pvh->search_config();
    // float -> double would print 1.6f as 1.600000023841858.
    auto as_config_number = [](float x) {
      return std::round(static_cast<double>(x) * 1e6) / 1e6;
    };
    out["type"] = "mcts_pv";
    out["model_path"] = pvh->model_path_hint();
    out["simulations"] = cfg.simulations;
    out["world_cap"] = cfg.world_cap;
    out["c_puct"] = as_config_number(cfg.c_puct);
    if (cfg.fpu_reduction) {
      out["fpu_reduction"] = as_config_number(*cfg.fpu_reduction);
    } else {
      out["fpu_reduction"] = py::none();
    }
    out["uct2_arrival_visits"] = cfg.uct2_arrival_visits;
    out["tail_solve_enabled"] = cfg.tail_solve_enabled;
    out["opponent_selection"] =
        cfg.opponent_selection == search::OpponentSelection::kFrozenPrior
            ? "prior" : "puct";
    out["leaf_batch_size"] = cfg.leaf_batch_size;
    return out;
  }, py::arg("game_id"), py::arg("name"));

  m.def("available_games", []() -> std::vector<std::string> {
    py::gil_scoped_release release;
    return GameRegistry::instance().game_ids();
  });

  // GameBundle::external_rules_switch: follow the game's other engine where
  // it departs from the rulebook, for every session in this process.
  m.def("set_external_rules", [](const std::string& game_id, bool enabled) {
    py::gil_scoped_release release;
    GameBundle bundle = GameRegistry::instance().create_game(game_id, 0);
    if (!bundle.external_rules_switch) {
      throw std::runtime_error("set_external_rules: game '" + game_id +
                               "' registers no external_rules_switch");
    }
    bundle.external_rules_switch(enabled);
  }, py::arg("game_id"), py::arg("enabled"));

  py::class_<GameSessionWrapper>(m, "GameSession")
      .def(py::init<const std::string&, std::uint64_t, const std::string&, bool,
                    const std::string&>(),
           py::arg("game_id"),
           py::arg("seed") = 0xC0FFEE,
           py::arg("model_path") = "",
           py::arg("use_filter") = false,
           py::arg("belief_model_path") = std::string(""))
      .def_property_readonly("is_terminal", &GameSessionWrapper::is_terminal)
      .def_property_readonly("is_turn_start", &GameSessionWrapper::is_turn_start)
      .def_property_readonly("current_player", &GameSessionWrapper::current_player)
      .def_property_readonly("winner", &GameSessionWrapper::winner)
      .def_property_readonly("winners", &GameSessionWrapper::winners)
      .def_property_readonly("losers", &GameSessionWrapper::losers)
      .def_property_readonly("num_players", &GameSessionWrapper::num_players)
      .def_property_readonly("game_id", &GameSessionWrapper::game_id)
      .def("state_hash_for_perspective",
           &GameSessionWrapper::state_hash_for_perspective,
           py::arg("player"))
      .def_property_readonly("step_count", &GameSessionWrapper::step_count)
      .def("get_state_dict", &GameSessionWrapper::get_state_dict)
      .def("get_public_state_dict", &GameSessionWrapper::get_public_state_dict,
           py::arg("viewer"),
           "State dict as seat `viewer` may see it: leaves that depend on "
           "slots hidden from the viewer are None.")
      .def("seat_sees_more_than", &GameSessionWrapper::seat_sees_more_than,
           py::arg("seat"), py::arg("viewer"))
      .def("_hidden_slots", &GameSessionWrapper::_hidden_slots, py::arg("viewer"))
      .def("_read_field", &GameSessionWrapper::_read_field,
           py::arg("name"), py::arg("idx"))
      .def("get_action_info", &GameSessionWrapper::get_action_info)
      .def("apply_public_snapshot", &GameSessionWrapper::apply_public_snapshot,
           py::arg("snapshot"), py::arg("receiver_seat"))
      .def("get_legal_actions", &GameSessionWrapper::get_legal_actions)
      .def("get_all_legal_actions", &GameSessionWrapper::get_all_legal_actions)
      .def("apply_action", &GameSessionWrapper::apply_action)
      .def("apply_action_with_trace", &GameSessionWrapper::apply_action_with_trace,
           py::arg("action"), py::arg("perspective"))
      .def("extract_initial_observation",
           &GameSessionWrapper::extract_initial_observation,
           py::arg("perspective"))
      .def("load_external_state", &GameSessionWrapper::load_external_state,
           py::arg("json"))
      .def("external_options", &GameSessionWrapper::external_options)
      .def("apply_observation", &GameSessionWrapper::apply_observation,
           py::arg("action"),
           py::arg("events") = py::list(),
           py::arg("public_snapshot") = py::dict())
      .def("apply_initial_observation", &GameSessionWrapper::apply_initial_observation,
           py::arg("perspective_player"), py::arg("initial_observation"))
      .def("_encode", &GameSessionWrapper::_encode, py::arg("perspective"),
           "Test hook: encoder features + legal mask of the session state.")
      .def("_force_write_field", &GameSessionWrapper::_force_write_field,
           py::arg("name"), py::arg("idx"), py::arg("value"),
           "Adapter-only escape hatch: write a single schema slot directly. "
           "Used by tests and external-judge adapters; not for production AI "
           "pipeline code.")
      .def("_debug_do_undo_roundtrip",
           &GameSessionWrapper::_debug_do_undo_roundtrip,
           py::arg("nested") = 2)
      .def("get_belief_snapshot", &GameSessionWrapper::get_belief_snapshot)
      .def("get_ai_action", &GameSessionWrapper::get_ai_action,
           py::arg("simulations") = 200,
           py::arg("temperature") = 0.0,
           py::arg("c_puct") = 1.4f,
           py::arg("cover_root_edges") = false,
           py::arg("opponent_selection") = std::string("puct"),
           py::arg("temperature_initial") = -1.0,
           py::arg("temperature_final") = -1.0,
           py::arg("temperature_decay_plies") = 0,
           py::arg("world_cap") = 0,
           py::arg("dirichlet_alpha") = 0.0,
           py::arg("dirichlet_epsilon") = 0.0,
           py::arg("dirichlet_on_first_n_plies") = 0,
           py::arg("top_k") = 0,
           py::arg("fpu_reduction") = py::none(),
           py::arg("uct2_arrival_visits").noconvert() = true)
      .def("get_heuristic_action", &GameSessionWrapper::get_heuristic_action,
           py::arg("selector"),
           py::arg("temperature") = 0.0)
      .def("configure_tail_solve", &GameSessionWrapper::configure_tail_solve,
           py::arg("enabled"),
           py::arg("depth_limit") = 10,
           py::arg("node_budget") = 200000LL,
           py::arg("auxiliary_weight") = 0.0f)
      .def("apply_ai_action", &GameSessionWrapper::apply_ai_action,
           py::arg("simulations") = 200,
           py::arg("temperature") = 0.0)
      .def("_debug_run_mcts_with_sim_traces",
           &GameSessionWrapper::_debug_run_mcts_with_sim_traces,
           py::arg("simulations") = 16)
      .def("_debug_run_mcts_with_edge_traces",
           &GameSessionWrapper::_debug_run_mcts_with_edge_traces,
           py::arg("simulations") = 16,
           py::arg("uct2_arrival_visits") = true,
           py::arg("leaf_batch_size") = 1);
}
