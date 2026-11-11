#pragma once

#include <any>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "belief_evaluator.h"
#include "belief_feature_extractor.h"
#include "belief_label_extractor.h"
#include "belief_tracker.h"
#include "feature_encoder.h"
#include "heuristic.h"
#include "heuristic_registry.h"
#include "snapshot_io.h"
#include "game_interfaces.h"
#include "../search/net_mcts.h"
#include "../search/tail_solver.h"

namespace board_ai {

// AnyMap, PublicEvent, PublicEventTrace are defined in
// game_interfaces.h so IBeliefTracker can reference them without a
// circular include.

using StateSerializer = std::function<AnyMap(const IGameState& state)>;

using ActionDescriptor = std::function<AnyMap(ActionId action)>;

// HeuristicResult and IHeuristic live in heuristic.h. Games expose
// heuristics through `GameBundle::heuristics` (HeuristicRegistry) keyed
// by name (e.g. "rule_v1", "rule_mcts_v1"); selectors are driven by
// `games/<id>/config/heuristic.json`.

struct SelfplaySampleView {
  int ply;
  int player;
  ActionId action_id;
  // Pointer to the encoded feature vector for this ply, taken from the
  // perspective of `player`. Non-null whenever the runner had an encoder
  // wired (the standard selfplay path always does); games can use this to
  // read viz=1 slot encodings without having to walk state themselves.
  // Lifetime: borrowed from result.samples; valid only inside the
  // episode_stats_extractor callback.
  const std::vector<float>* features = nullptr;
};

using EpisodeStatsExtractor = std::function<
    std::map<std::string, double>(
        const IGameState& final_state,
        const IGameState& initial_state,
        const std::vector<SelfplaySampleView>& samples)>;

using GameAdjudicator = std::function<int(const IGameState& state)>;

// Per-seat game score of a finished (or ply-capped) episode, in the game's own
// units (Splendor: prestige points). Feeds the optional score-margin auxiliary
// training target; games without a meaningful score leave it unset.
using FinalScoreExtractor = std::function<std::vector<float>(const IGameState& state)>;

// depth_remaining is the tail-solver's remaining alpha-beta depth at the
// current node — game-side scorers can fold it in for "prefer faster wins"
// e.g. `depth_remaining + 0.1f * normalized_margin`.
using AuxiliaryScorer = std::function<float(const IGameState& state, int player, int depth_remaining)>;

using TrainingActionFilter = std::function<std::vector<ActionId>(
    const IGameState& state, const IGameRules& rules, const std::vector<ActionId>& legal)>;

using TailSolveTrigger = std::function<bool(const IGameState& state, int ply)>;

// Public-event protocol for the AI API (platform/ai_service):
//
// When the AI session is driven from observations, ground-truth's random
// outcomes (deck flips, hidden draws, revealed cards during challenges)
// must reach the AI's belief tracker as events. Each stochastic game
// registers a PublicEventExtractor — during self-play, it diffs
// state_before and state_after to produce the events an outside observer
// at perspective would have seen. Used to generate ground-truth traces
// for tests. The events go ONLY to the belief tracker (observer
// sessions do not run do_action_fast — public state is rebuilt from
// public_snapshot).
//
// Initial observation handshake (the perspective-specific facts the AI
// knows at game start, e.g. own starting hand) is NOT a per-game hook —
// it is walker-driven via viz::serialize_public_snapshot /
// viz::apply_public_snapshot, the same path the per-ply public_snapshot
// uses. Games only have to declare schema visibility correctly for
// game-start; the framework walker handles serialization in both
// directions.
//
// PublicEvent / PublicEventTrace are declared in game_interfaces.h (so
// belief_tracker.h can reference them without a circular include).

using PublicEventExtractor = std::function<PublicEventTrace(
    const IGameState& state_before,
    ActionId action,
    const IGameState& state_after,
    int perspective_player)>;

// Events-only flavor of the extractor — same diff logic as
// PublicEventExtractor, but skips the wholesale `viz::serialize_public_snapshot`
// walk over the schema. Consumed exclusively by the MCTS sim descent
// (engine/search/net_mcts.cpp): each sim step calls this to feed the
// cloned sim_tracker's `observe_public_event`. The sim never needs the
// `public_snapshot` half — sim_state is itself a GT-equivalent advanced
// by `do_action_fast`, so its viz is already correct and node-player
// masking is done via `make_masked_state(sim_state, schema, player)`,
// not via wire-protocol replacement.
//
// Tracker content is perspective-invariant (test_tracker_perspective_invariance),
// so the sim calls this with `perspective = root_player` once per step.
//
// Optional: when a game registers `public_event_extractor` but leaves
// `events_only_extractor` empty, the sim path falls back to the full
// extractor (correct, just pays the snapshot serialization cost).
using EventsOnlyExtractor = std::function<std::vector<PublicEvent>(
    const IGameState& state_before,
    ActionId action,
    const IGameState& state_after,
    int perspective_player)>;

// Message-driven public state: inverse of the public-fields-only
// serialization produced by public_event_extractor. Overwrites the
// observer session's state_ public fields from the truth snapshot at
// the end of apply_observation, so the observer's public state is
// rebuilt from the message stream — it never depends on whatever
// `do_action_fast(state_)` computed from its sampled hidden fields.
//
// Contract:
//   - Applier OVERWRITES every public field that state_hash_for_perspective
//     depends on. Fields absent from the snapshot are left unchanged
//     (but see `test_public_snapshot_round_trip` — every field that
//     state_hash_for_perspective reads must round-trip through snapshot+applier).
//   - Applier MUST NOT touch hidden fields. Those are left for
//     tracker.randomize_unseen to fill.
//
// Required for all hidden-info games. Fully-public games (tictactoe,
// quoridor) do not register one — there is no hidden for do_action_fast
// to read wrong, so the public side is already tamper-proof.
// receiver_seat: the perspective whose session this state belongs to.
// Needed by `viz::apply_public_snapshot` to overwrite the receiver's
// viz slice (`viz_[name][..., receiver_seat]`) byte-for-byte from the
// wire's `__viz__` payload. Pass -1 from contexts where the seat is
// unknown / N/A (fully-public games whose appliers don't consult it).
using PublicStateApplier = std::function<void(
    IGameState& state,
    const AnyMap& snapshot,
    int receiver_seat)>;

// Interop with another engine of the same game, so the AI can play on that engine's board.
// Optional; the state format and option keys are that engine's (see the game's API doc).
//   ExternalStateLoader: replace a truth state with one exported by that engine. AI sessions get
//     it through extract_initial_observation like any other truth state.
//   ExternalOptionLister: the pending decision's options consistent with the atoms already
//     chosen, as (that engine's option key, full atom sequence). An option may be listed once per
//     atom sequence.
//   ExternalRulesSwitch: follow that engine where it departs from the rulebook. Process-wide.
using ExternalStateLoader = std::function<void(IGameState& state, const std::string& json)>;
using ExternalOptionLister =
    std::function<std::vector<std::pair<std::string, std::vector<ActionId>>>(const IGameState& state)>;
using ExternalRulesSwitch = std::function<void(bool enabled)>;

struct GameBundle {
  std::unique_ptr<IGameState> state;
  std::unique_ptr<IGameRules> rules;
  std::unique_ptr<IStateValueModel> value_model;
  std::unique_ptr<IFeatureEncoder> encoder;
  std::unique_ptr<IBeliefTracker> belief_tracker;
  StateSerializer state_serializer;
  ActionDescriptor action_descriptor;
  // Per-game named pool of IHeuristic instances (rule_v1, rule_mcts_v1,
  // pv_<...>, etc.). heuristic.json's `selectors` (eval_vs_heuristic /
  // web) and train.json's `heuristic_guidance.name` reference entries
  // here; the framework looks up by name at call time.
  HeuristicRegistry heuristics;
  // Owns hand-written PV evaluators (mcts_pv flavor B) referenced by
  // entries in `heuristics`. Quoridor's `QuoridorRulePVEvaluator` lives
  // here; the corresponding `MctsPVHeuristic` borrows a raw pointer
  // back into this vector, with lifetime safe because the bundle owns
  // both. ONNX-based mcts_pv heuristics (flavor C) own their evaluator
  // internally and don't appear here.
  std::vector<std::unique_ptr<search::IPolicyValueEvaluator>>
      heuristic_pv_evaluators;
  // Encoders for ONNX-backed heuristics whose model expects a different
  // feature layout than `encoder` (e.g. an MLP guidance model inside a
  // token-encoder variant). Heuristics borrow raw pointers into here.
  std::vector<std::unique_ptr<IFeatureEncoder>> heuristic_encoders;
  std::unique_ptr<search::ITailSolver> tail_solver;
  // Set to true only after verifying that do_action_deterministic is
  // overridden and NEVER consumes hidden chance outcomes (drawing from
  // deck, flipping face-down, etc.). Required to combine tail_solver with
  // a belief_tracker. See create_game() for the full contract.
  bool stochastic_tail_solve_safe = false;
  TailSolveTrigger tail_solve_trigger;
  EpisodeStatsExtractor episode_stats_extractor;
  FinalScoreExtractor final_score_extractor;
  GameAdjudicator adjudicator;
  AuxiliaryScorer auxiliary_scorer;
  TrainingActionFilter training_action_filter;
  // Public-event protocol (see above). Optional — only needed for games
  // that register a belief_tracker AND want to be driveable through the
  // AI API with independent seeds.
  PublicEventExtractor public_event_extractor;
  // Sim-only events extractor (see EventsOnlyExtractor above). Optional;
  // when empty the sim falls back to public_event_extractor.
  EventsOnlyExtractor events_only_extractor;
  // Required for hidden-info games. See PublicStateApplier above.
  // Fully-public games (tictactoe, quoridor) leave this unset.
  PublicStateApplier public_state_applier;

  // External-engine interop (see ExternalStateLoader above). Optional.
  ExternalStateLoader external_state_loader;
  ExternalOptionLister external_option_lister;
  ExternalRulesSwitch external_rules_switch;

  // Belief network plumbing (Coup belief net plan). Optional — only
  // games whose `IBeliefTracker::prepare_for_root` consults a learned
  // posterior register both. The framework discovers the ONNX model
  // at runtime via `belief_model_path` (e.g.
  // `games/<id>/model/<id>_belief_<N>p.onnx`); the extractor produces
  // the input feature vector and declares the (N-1) × K logit count.
  // When either is empty, the tracker's default
  // `prepare_for_root` is a no-op and `randomize_unseen` falls back to
  // its uniform / multiset-derived sampling.
  std::unique_ptr<IBeliefFeatureExtractor> belief_feature_extractor;
  std::string belief_model_path;
  // GT-side label extractor for belief-net training samples (Plan §2.2).
  // Reads truth IGameState; never reachable from the AI session. Only
  // selfplay_runner consults it (emit path). Optional — only games that
  // emit BeliefSample register one.
  std::unique_ptr<IBeliefLabelExtractor> belief_label_extractor;

  std::string game_id;

  // One-shot helper for hidden-info games. Pass an events-only diff
  // (game-specific) and a schema provider; the helper auto-fills the
  // three wire-protocol fields:
  //
  //   events_only_extractor    = events_only_diff (consumed by sim)
  //   public_event_extractor   = events_only + viz::serialize_public_snapshot
  //                              (consumed by ply / wire / trace)
  //   public_state_applier     = viz::apply_public_snapshot
  //                              (consumed by receiver session)
  //
  // For games whose tracker is event-free (no_events_extractor), pass
  // viz::no_events_extractor as the diff.
  template <typename EventsOnlyFn, typename SchemaProviderFn>
  void install_event_protocol(EventsOnlyFn events_only,
                              SchemaProviderFn schema_provider) {
    events_only_extractor = events_only;
    public_event_extractor =
        viz::make_public_event_extractor_from_events_only(
            events_only, schema_provider);
    public_state_applier = viz::make_public_state_applier(schema_provider);
  }
};

using GameFactory = std::function<GameBundle(std::uint64_t seed)>;

class GameRegistry {
 public:
  static GameRegistry& instance() {
    static GameRegistry reg;
    return reg;
  }

  void register_game(const std::string& game_id, GameFactory factory) {
    factories_[game_id] = std::move(factory);
  }

  GameBundle create_game(const std::string& game_id, std::uint64_t seed) const {
    auto it = factories_.find(game_id);
    if (it == factories_.end()) {
      throw std::runtime_error("Unknown game: " + game_id);
    }
    GameBundle bundle = it->second(seed);
    // Invariant: if a game has both hidden information (belief_tracker) and a
    // tail solver, it MUST provide a deterministic action API. The tail solver
    // expands the game tree assuming each branch is deterministic; if the
    // game's do_action_fast draws from a hidden deck to advance turns, the
    // solver "peeks" at the true deck and produces a proven-win strategy that
    // is only correct against that specific realization. This silently
    // poisons training samples — it violates the AI-pipeline-independence
    // design principle.
    //
    // Games that hit this combination must override do_action_deterministic
    // AND flip stochastic_tail_solve_safe = true as an explicit
    // acknowledgment. The default is false — you have to opt in.
    if (bundle.belief_tracker && bundle.tail_solver &&
        !bundle.stochastic_tail_solve_safe) {
      throw std::runtime_error(
          "Game '" + game_id + "' registers both a belief_tracker and a "
          "tail_solver, but has not set stochastic_tail_solve_safe = true. "
          "This combination is unsafe unless the game's do_action_deterministic "
          "is overridden so tail-solver's alpha-beta search never consumes "
          "hidden chance outcomes. Override do_action_deterministic in your "
          "rules (it must advance turns without drawing hidden cards/tiles), "
          "then set stochastic_tail_solve_safe = true in the GameBundle.");
    }
    // Invariant: a hidden-info game (belief_tracker != null) MUST register the
    // public-event protocol — public_state_applier (receiver side: session
    // advances from the truth snapshot) and public_event_extractor (GT side:
    // packs events + public_snapshot per ply). Without public_state_applier,
    // advance_per_seat_states / apply_observation fall back to do_action_fast
    // on the session seat, which reads viz=0 hidden slots (unread bytes) and
    // draws from a divergent RNG — silently desyncing the session's public
    // projection from truth. Symptom: legal_actions(ai_view) on a wrong board
    // → illegal moves or empty legal sets, surfacing mid-training, not at
    // startup. See harmonies commit 632ebe2 for the failure mode this guards.
    if (bundle.belief_tracker && !bundle.public_state_applier) {
      throw std::runtime_error(
          "Game '" + game_id + "' registers a belief_tracker but no "
          "public_state_applier. Call install_event_protocol(...) in your "
          "register.cpp (or assign public_state_applier directly) so the AI "
          "session advances from the truth snapshot instead of replaying "
          "do_action_fast on a masked seat — replaying reads viz=0 hidden "
          "slots and silently desyncs, surfacing as illegal moves "
          "mid-training.");
    }
    if (bundle.belief_tracker && !bundle.public_event_extractor) {
      throw std::runtime_error(
          "Game '" + game_id + "' registers a belief_tracker but no "
          "public_event_extractor. Call install_event_protocol(...) in your "
          "register.cpp so the GT side packs events + public_snapshot per ply "
          "for the receiver session and its tracker.");
    }
    return bundle;
  }

  bool has_game(const std::string& game_id) const {
    return factories_.count(game_id) > 0;
  }

  std::vector<std::string> game_ids() const {
    std::vector<std::string> ids;
    ids.reserve(factories_.size());
    for (const auto& [id, _] : factories_) {
      ids.push_back(id);
    }
    return ids;
  }

 private:
  GameRegistry() = default;
  std::unordered_map<std::string, GameFactory> factories_;
};

struct GameRegistrar {
  GameRegistrar(const std::string& game_id, GameFactory factory) {
    GameRegistry::instance().register_game(game_id, std::move(factory));
  }
};

}  // namespace board_ai
