#pragma once

#include <any>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../core/belief_evaluator.h"
#include "../core/belief_feature_extractor.h"
#include "../core/belief_label_extractor.h"
#include "../core/belief_tracker.h"
#include "../core/feature_encoder.h"
#include "../core/game_interfaces.h"
#include "../core/game_registry.h"
#include "../core/heuristic.h"
#include "../search/net_mcts.h"
#include "../search/tail_solver.h"
#include "../search/temperature_schedule.h"
#include "../search/root_noise.h"

namespace board_ai::runtime {

struct SelfplaySample {
  int ply = 0;
  int player = 0;
  ActionId action_id = -1;
  std::vector<ActionId> policy_action_ids{};
  std::vector<int> policy_action_visits{};
  std::vector<float> features{};
  std::vector<float> legal_mask{};
  float z = 0.0f;
  std::vector<float> z_values{};
  bool tail_solved = false;
  // False for playout-cap fast moves: searched with fast_simulations and no
  // root noise, so the visit distribution is not a training target. Still
  // emitted so episode stats see every ply; the trainer drops them. Features
  // stay empty unless SelfplayConfig::encode_fast_samples.
  bool full_search = true;
  // MCTS simulations the selfplay search actually ran for this ply. 0 for
  // plies that needed no search (forced move, tail-solved) and for
  // heuristic-guided plies.
  int simulations = 0;
  // Heuristic-guided plies only: simulations the guiding heuristic ran
  // internally (HeuristicResult::simulations; its own network and budget).
  int heuristic_simulations = 0;
};

// Belief-net training sample (Coup belief net plan §2.2).
//
// One sample per ply per observer, emitted by selfplay_runner when both
// `belief_feature_extractor` and `belief_label_extractor` are wired
// through the runner. `features` is computed from the AI-side
// masked state + tracker (no truth read); `hand_counts` / `remaining` /
// `alive_per_opp` are computed from GT truth. Both halves are
// emit-side only — neither AI sessions nor wire snapshots ever consume
// a BeliefSample.
struct BeliefSample {
  int ply = 0;
  int observer = 0;
  std::vector<float> features{};                       // size feature_dim
  std::vector<std::vector<int>> hand_counts{};         // [N-1][K]
  std::vector<int> remaining{};                        // [K]
  std::vector<int> alive_per_opp{};                    // [N-1]
};

// Per-ply observation trace for API belief-equivalence tests. Only populated
// when run_selfplay_episode is called with trace_perspective >= 0 AND the
// game has registered a public_event_extractor.
//
// `public_snapshot`: populated iff the game's extractor fills it (==
// game registers public_state_applier). Empty map otherwise.
struct SelfplayObservationTrace {
  int ply = 0;
  int actor = 0;                           // player whose action this was
  ActionId action = -1;
  std::vector<std::pair<std::string, AnyMap>> events{};
  AnyMap public_snapshot{};
  std::map<std::string, std::any> belief_snapshot_after{};
};

struct SelfplayEpisodeResult {
  int winner = -1;
  bool draw = false;
  // Per-seat terminal z values, length == num_players. Same vector
  // assigned to every sample's z_values; lifted here so the binding
  // can expose it at episode level without iterating samples. Pipeline
  // converts to per-seat shares via (z+1)/N for the `pX=...` log column
  // and for eval/gating wins counting (winner-takes-all and
  // liarsdice 3+ loser-pays end up on the same scale: each seat's
  // share ∈ [0, 1], total over seats = 1).
  std::vector<float> terminal_z_values{};
  int total_plies = 0;
  std::vector<SelfplaySample> samples{};
  std::map<std::string, double> custom_stats{};
  std::unique_ptr<IGameState> final_state{};

  int tail_solve_attempts = 0;
  int tail_solve_completed = 0;
  int tail_solve_successes = 0;
  double tail_solve_total_ms = 0.0;

  bool trace_enabled = false;
  int trace_perspective = -1;
  std::map<std::string, std::any> initial_belief_snapshot{};
  AnyMap initial_observation{};
  std::vector<SelfplayObservationTrace> observation_trace{};

  // Belief-net training samples (Plan §2.2). Populated only when the
  // runner is given both a belief_feature_extractor and a
  // belief_label_extractor. Empty otherwise.
  std::vector<BeliefSample> belief_samples{};
  // K (label class count) copied from belief_label_extractor — pybind /
  // pipeline.py needs it to interpret hand_counts / remaining shape
  // without reaching back into the bundle.
  int belief_label_class_count = 0;
};

struct SelfplayConfig {
  int simulations = 200;
  // ISMCTS K outer worlds × M inner sims-per-world cap. 0 = legacy
  // K=simulations / M=1 (every sim its own world). >0 clamps K and
  // amortizes visits inside each fixed determinization. See
  // NetMctsConfig::world_cap.
  int world_cap = 0;
  float c_puct = 1.4f;
  // See NetMctsConfig::fpu_reduction. Unset = legacy Q=0 for unvisited edges.
  std::optional<float> fpu_reduction{};
  // See NetMctsConfig::uct2_arrival_visits. Applies to every seat's search
  // (pool opponents included); a guiding mcts_pv heuristic keeps its own
  // fixed setting (MctsPVHeuristic::search_config).
  bool uct2_arrival_visits = true;
  int max_depth = 128;
  float value_clip = 1.0f;
  double temperature = 1.0;
  search::TemperatureSchedule temperature_schedule{};
  double dirichlet_alpha = 0.0;
  double dirichlet_epsilon = 0.0;
  int dirichlet_on_first_n_plies = 0;
  int max_game_plies = 500;

  bool tail_solve_enabled = false;
  search::TailSolveConfig tail_solve_config{};

  double heuristic_guidance_ratio = 0.0;
  double heuristic_temperature = 0.0;
  double training_filter_ratio = 1.0;

  // See OpponentSelection in net_mcts.h. Default kPuct is current
  // behavior. kFrozenPrior turns descent on opponent nodes into a
  // multinomial sample from the policy head's prior, mitigating
  // strategy fusion in ISMCTS for hidden-info games.
  search::OpponentSelection opponent_selection = search::OpponentSelection::kPuct;

  // Top-k action selection: only the k most-visited actions participate
  // in the temperature-weighted sample. 0 = all, 1 = argmax. Default 0 —
  // matches Python mcts_profile._BASE_DICT["top_k"].
  int top_k = 0;

  // When true, validate DAG edges at every descent step. Slower but
  // catches hash/schema bugs early. Forwarded to NetMctsConfig.safe_mode.
  bool safe_mode = false;

  // Forwarded to NetMctsConfig.leaf_batch_size. 1 = unbatched search.
  int leaf_batch_size = 1;

  // Playout cap randomization (KataGo): each MCTS ply is a full search
  // (`simulations`, root noise, training sample) with probability
  // full_search_prob, otherwise a fast search with fast_simulations and no
  // root noise whose sample is marked full_search=false. 0 disables.
  int fast_simulations = 0;
  double full_search_prob = 1.0;
  // False: fast samples carry empty features / legal_mask (the trainer drops
  // them anyway; with long games and wide encoders they dominate the memory
  // of in-flight episodes). Episode stats then see a null features view.
  bool encode_fast_samples = true;
};

using GameAdjudicator = board_ai::GameAdjudicator;
using AuxiliaryScorer = board_ai::AuxiliaryScorer;
using TrainingActionFilter = board_ai::TrainingActionFilter;
using TailSolveTrigger = board_ai::TailSolveTrigger;

// Wraps another IGameRules, filtering legal_actions through a training
// filter. Forwards every other entry point to the inner rules. Because the
// framework wrappers around do_action_fast / do_action_deterministic /
// undo_action are non-virtual and bump state.step_count_ themselves, this
// class must NOT override them; instead it forwards through the *_impl
// hooks so the inner rules' impl runs exactly once and step_count_ stays
// in sync (we delegate to inner_'s public wrappers, which would double-bump
// if we then also bumped here — so we just call the impl side directly via
// inner_ as friend? No — IGameRules' public wrappers do the bookkeeping;
// our impls run after our own bump. Net: we forward by calling
// inner_.do_action_fast_impl etc. directly via the friend access).
class FilteredRulesWrapper final : public IGameRules {
 public:
  FilteredRulesWrapper(const IGameRules& inner, TrainingActionFilter filter)
      : inner_(inner), filter_(std::move(filter)) {}

  bool validate_action(const IGameState& state, ActionId action) const override {
    return inner_.validate_action(state, action);
  }

  std::vector<ActionId> legal_actions(const IGameState& state) const override {
    auto legal = inner_.legal_actions(state);
    if (filter_) {
      auto filtered = filter_(state, inner_, legal);
      if (!filtered.empty()) return filtered;
    }
    return legal;
  }

  ActionId public_action(const IGameState& state_before, ActionId action) const override {
    return inner_.public_action(state_before, action);
  }

 protected:
  // Our wrapper bumps step_count_ once (in IGameRules::do_action_fast),
  // then we delegate to the inner rules' impl directly so it runs without
  // a second bump. Same for deterministic / undo. Sibling-instance access
  // to protected impls goes through the static invoke_* helpers on
  // IGameRules.
  void do_action_fast_impl(IGameState& state, ActionId action,
                           std::mt19937_64& rng) const override {
    invoke_do_action_fast_impl(inner_, state, action, rng);
  }
  void undo_action_impl(IGameState& state, const UndoToken& token) const override {
    invoke_undo_action_impl(inner_, state, token);
  }
  UndoToken do_action_deterministic_impl(IGameState& state,
                                         ActionId action) const override {
    return invoke_do_action_deterministic_impl(inner_, state, action);
  }

 private:
  const IGameRules& inner_;
  TrainingActionFilter filter_;
};

using PublicEventExtractor = board_ai::PublicEventExtractor;
using PublicStateApplier = board_ai::PublicStateApplier;

// Per-seat policy/value evaluator lookup. Selfplay's opponent-pool path and
// arena both need this shape — same callback wired through both runners so
// MCTS at each ply queries the network registered for the acting seat.
using PolicyEvaluatorFactory = std::function<
    const search::IPolicyValueEvaluator&(int player_index)>;

SelfplayEpisodeResult run_selfplay_episode(
    IGameState& initial_state,
    const IGameRules& rules,
    const IStateValueModel& value_model,
    PolicyEvaluatorFactory evaluator_for_player,
    const SelfplayConfig& config,
    std::uint64_t episode_seed,
    // One tracker per player seat. For hidden-info games: size == num_players;
    // the runner inits each seat once at episode start, feeds every public
    // event to every tracker, and routes MCTS root to
    // per_perspective_trackers[current_player]. For games without hidden info:
    // empty vector; the runner skips tracker wiring entirely.
    std::vector<IBeliefTracker*> per_perspective_trackers = {},
    // Required per-seat session state (size == num_players, no nullptrs).
    // MCTS/encoder/heuristic/legal_actions read from per_seat_states[player]
    // — never from truth. Each seat's session state is advanced via the
    // public-event protocol after every truth do_action_fast: hidden-info
    // games run public_state_applier(seat, snapshot) +
    // tracker.observe_public_event(events); fully-public games (no
    // public_event_extractor registered) re-run do_action_fast(seat) on
    // each seat with its own step rng. The session's viz=0 slots are
    // never freshened — decision-side reads (hash kHiddenHashSentinel,
    // encoder masked-state placeholder, sim_tracker->randomize_unseen at
    // sim entry) make them structurally unreachable.
    std::vector<IGameState*> per_seat_states = {},
    PublicStateApplier public_state_applier = nullptr,
    const IFeatureEncoder* encoder = nullptr,
    const search::ITailSolver* tail_solver = nullptr,
    GameAdjudicator adjudicator = nullptr,
    AuxiliaryScorer auxiliary_scorer = nullptr,
    // Heuristic for the heuristic_guidance_ratio path. Looked up by name
    // from `bundle.heuristics` at the binding layer; null here disables
    // the guidance path. The heuristic implementation can be a simple
    // rule, a hand-written PV+MCTS wrapper, or an ONNX-backed PV+MCTS
    // — all expose the same IHeuristic interface.
    const IHeuristic* heuristic = nullptr,
    TrainingActionFilter training_action_filter = nullptr,
    TailSolveTrigger tail_solve_trigger = nullptr,
    EpisodeStatsExtractor episode_stats_extractor = nullptr,
    // Tracing hooks for AI API belief-equivalence tests. trace_perspective
    // >= 0 enables recording; -1 disables (default, zero overhead).
    // trace_belief_tracker must be a SEPARATE tracker instance dedicated to
    // the trace_perspective, kept decoupled from the per_perspective_trackers
    // so tracing output stays reproducible across refactors of the MCTS
    // routing. The caller is responsible for creating this instance via the
    // game's factory.
    int trace_perspective = -1,
    IBeliefTracker* trace_belief_tracker = nullptr,
    PublicEventExtractor public_event_extractor = nullptr,
    // Sim-only events-extractor (sim descent uses this to feed
    // sim_tracker — skips the public_snapshot serialization that
    // public_event_extractor performs for the wire/ply protocol).
    EventsOnlyExtractor events_only_extractor = nullptr,
    // Belief network plumbing. When belief_extractor + per-seat evaluator
    // are both non-null, MCTS root calls
    // `tracker.prepare_for_root(root, root_player, extractor, evaluator)`
    // once per decision; sims inherit the cached posterior via clone().
    // When either is null, the tracker's default `prepare_for_root` is a
    // no-op and `randomize_unseen` falls back to uniform sampling.
    const IBeliefFeatureExtractor* belief_extractor = nullptr,
    // Per-seat belief evaluators. Size == num_players (one per seat) or
    // empty (no belief evaluation). Individual entries may be nullptr (that
    // seat falls back to uniform sampling). When non-empty, MCTS for seat
    // `p` uses per_seat_belief_evaluators[p].
    std::vector<const IBeliefEvaluator*> per_seat_belief_evaluators = {},
    // Belief-net training sample emit (Plan §2.2). When both
    // belief_extractor and belief_label_extractor are non-null, the
    // runner emits one BeliefSample per ply per observer into
    // result.belief_samples. Features are taken from the per-seat AI
    // session (masked state + tracker); labels are read from truth on
    // the GT runner. Independent of belief_evaluator: sample emit can
    // run with no installed belief.onnx (early training).
    const IBeliefLabelExtractor* belief_label_extractor = nullptr);

}  // namespace board_ai::runtime
