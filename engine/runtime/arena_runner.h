#pragma once

#include <cstdint>
#include <functional>
#include <memory>

#include "../core/game_interfaces.h"
#include "../search/net_mcts.h"
#include "selfplay_runner.h"

namespace board_ai::runtime {

struct ArenaPlayerConfig {
  int simulations = 200;
  // ISMCTS K outer worlds × M inner sims-per-world cap. 0 = legacy
  // K=simulations / M=1 (every sim its own world). >0 clamps K and
  // amortizes visits inside each fixed determinization. Per-player so
  // candidate and opponent can be independently configured. See
  // NetMctsConfig::world_cap.
  int world_cap = 0;
  float c_puct = 1.4f;
  // See NetMctsConfig::fpu_reduction. Unset = legacy Q=0 for unvisited edges.
  std::optional<float> fpu_reduction{};
  // See NetMctsConfig::uct2_arrival_visits. Per-player so the two UCT2
  // parent counts can be played against each other.
  bool uct2_arrival_visits = true;
  int max_depth = 128;
  float value_clip = 1.0f;
  double temperature = 0.0;
  bool tail_solve_enabled = false;
  search::TailSolveConfig tail_solve_config{};
  const search::ITailSolver* tail_solver = nullptr;
  TailSolveTrigger tail_solve_trigger = nullptr;
  // See OpponentSelection in net_mcts.h. Per-player so candidate and
  // opponent can be independently configured (A/B testing).
  search::OpponentSelection opponent_selection = search::OpponentSelection::kPuct;

  // See NetMctsConfig::top_k. 0 = no limit; matches Python defaults.
  int top_k = 0;

  // When true, validate DAG edges at every descent step. Slower but
  // catches hash/schema bugs early. Forwarded to NetMctsConfig.safe_mode.
  bool safe_mode = false;

  // Forwarded to NetMctsConfig.leaf_batch_size. 1 = unbatched search.
  int leaf_batch_size = 1;
};

struct ArenaPlyStats {
  bool tail_solved = false;
  float tail_solve_value = 0.0f;
};

struct ArenaMatchResult {
  int winner = -1;
  bool draw = false;
  // Per-seat terminal z values, length == num_players. Same role as
  // SelfplayEpisodeResult::terminal_z_values: lets the eval/gating
  // pipeline convert to (z+1)/N shares for unified counting under
  // multi-player loser-pays semantics. {} on draw or when num_players==0.
  std::vector<float> terminal_z_values;
  int total_plies = 0;
  std::vector<ActionId> action_history;
  std::vector<ArenaPlyStats> ply_stats;
};

// `PolicyEvaluatorFactory` lives in selfplay_runner.h (shared across selfplay's
// opponent-pool path and arena).

ArenaMatchResult run_arena_match(
    IGameState& initial_state,
    const IGameRules& rules,
    const IStateValueModel& value_model,
    PolicyEvaluatorFactory evaluator_for_player,
    const std::vector<ArenaPlayerConfig>& player_configs,
    int max_game_plies = 500,
    std::uint64_t match_seed = 0,
    IBeliefTracker* belief_tracker = nullptr,
    GameAdjudicator adjudicator = nullptr,
    // Tracker input adapter: needed when belief_tracker is set. The
    // tracker receives events from public_event_extractor and its
    // bootstrap state from `make_masked_state(state, schema, p)` —
    // state reads never pass through the tracker interface.
    PublicEventExtractor public_event_extractor = nullptr,
    // Per-seat trackers for hidden-info games. Size == num_players for
    // games with belief_tracker registered; empty for fully-public games.
    // When non-empty, each seat's tracker is init'd once at match start
    // and accumulates the full observation history exactly like selfplay.
    // MCTS root for the acting player reads from
    // per_perspective_trackers[player].
    std::vector<IBeliefTracker*> per_perspective_trackers = {},
    // Required per-seat session state (size == num_players, no nullptrs).
    // MCTS / encoder / legal_actions read from per_seat_states[player] —
    // never from truth. Advanced after every truth do_action_fast via the
    // public-event protocol: hidden-info games run public_state_applier(
    // seat, snapshot) + tracker.observe_public_event(events); fully-public
    // games re-run do_action_fast on each seat. viz=0 slots are never
    // freshened — see selfplay_runner.h for the structural-unreachability
    // argument.
    std::vector<IGameState*> per_seat_states = {},
    PublicStateApplier public_state_applier = nullptr,
    // Sim-only events-extractor — see selfplay_runner.h.
    EventsOnlyExtractor events_only_extractor = nullptr,
    // Belief network plumbing. belief_extractor must be non-null to activate;
    // nullptr → tracker prepare_for_root no-op + uniform sampling.
    const IBeliefFeatureExtractor* belief_extractor = nullptr,
    // Per-seat belief evaluators. Size == num_players (one per seat) or empty
    // (no belief evaluation). Individual entries may be nullptr (that seat
    // falls back to uniform sampling). When non-empty, MCTS for seat `p`
    // uses per_seat_belief_evaluators[p].
    std::vector<const IBeliefEvaluator*> per_seat_belief_evaluators = {});

}  // namespace board_ai::runtime
