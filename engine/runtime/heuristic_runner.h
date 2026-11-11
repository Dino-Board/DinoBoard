#pragma once

#include <cstdint>
#include <cstddef>

#include "../core/game_registry.h"
#include "selfplay_runner.h"

namespace board_ai::runtime {

// Select an index from heuristic scores using softmax-with-temperature sampling.
// temperature <= 1e-6 means greedy argmax (first max on ties); >0 means
// exp(score/T) softmax then cumulative sample with rng_u01 in [0,1).
// Shared by selfplay (run_heuristic_episode) and eval benchmarks so both
// faces of the heuristic use the same selection rule.
std::size_t sample_heuristic_index(
    const std::vector<double>& scores,
    double temperature,
    double rng_u01);

// Single-heuristic shape: every seat plays the same IHeuristic. Used by
// selfplay heuristic-guidance and the eval_vs_heuristic benchmark.
SelfplayEpisodeResult run_heuristic_episode(
    IGameState& initial_state,
    const IGameRules& rules,
    const IStateValueModel& value_model,
    const IFeatureEncoder* encoder,
    const IHeuristic& heuristic,
    double temperature,
    int max_game_plies,
    std::uint64_t episode_seed,
    AuxiliaryScorer auxiliary_scorer = nullptr,
    GameAdjudicator adjudicator = nullptr,
    // Required per-seat session state (size == num_players, no nullptrs).
    // The heuristic / encoder / legal-action queries on the AI path read
    // from per_seat_states[player] — never from truth. Advanced after
    // every truth do_action_fast via the public-event protocol
    // (public_state_applier from snapshot + tracker.observe_public_event(
    // events) for hidden-info games; do_action_fast(seat) for fully-public
    // games). viz=0 slots are never freshened — see selfplay_runner.h for
    // rationale.
    std::vector<IBeliefTracker*> per_perspective_trackers = {},
    std::vector<IGameState*> per_seat_states = {},
    PublicStateApplier public_state_applier = nullptr,
    PublicEventExtractor public_event_extractor = nullptr);

// Per-seat heuristic shape: each seat runs its own IHeuristic. Used by
// platform/tools/eval_model.py to benchmark heuristic-vs-heuristic
// matches (e.g. lkf_v1 vs rule_v1 in 6 nimmt!). per_seat_heuristics and
// per_seat_temperatures must both have size == num_players.
SelfplayEpisodeResult run_heuristic_episode_per_seat(
    IGameState& initial_state,
    const IGameRules& rules,
    const IStateValueModel& value_model,
    const IFeatureEncoder* encoder,
    const std::vector<const IHeuristic*>& per_seat_heuristics,
    const std::vector<double>& per_seat_temperatures,
    int max_game_plies,
    std::uint64_t episode_seed,
    AuxiliaryScorer auxiliary_scorer = nullptr,
    GameAdjudicator adjudicator = nullptr,
    std::vector<IBeliefTracker*> per_perspective_trackers = {},
    std::vector<IGameState*> per_seat_states = {},
    PublicStateApplier public_state_applier = nullptr,
    PublicEventExtractor public_event_extractor = nullptr);

}  // namespace board_ai::runtime
