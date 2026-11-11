#include "arena_runner.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <stdexcept>

#include "../core/masked_state.h"
#include "../core/rng_salt.h"

namespace board_ai::runtime {

ArenaMatchResult run_arena_match(
    IGameState& initial_state,
    const IGameRules& rules,
    const IStateValueModel& value_model,
    PolicyEvaluatorFactory evaluator_for_player,
    const std::vector<ArenaPlayerConfig>& player_configs,
    int max_game_plies,
    std::uint64_t match_seed,
    IBeliefTracker* belief_tracker,
    GameAdjudicator adjudicator,
    PublicEventExtractor public_event_extractor,
    std::vector<IBeliefTracker*> per_perspective_trackers,
    std::vector<IGameState*> per_seat_states,
    PublicStateApplier public_state_applier,
    EventsOnlyExtractor events_only_extractor,
    const IBeliefFeatureExtractor* belief_extractor,
    std::vector<const IBeliefEvaluator*> per_seat_belief_evaluators) {
  ArenaMatchResult result{};
  auto state = initial_state.clone_state();
  int ply = 0;

  // GT step rng for do_action_fast on truth state. Seeded directly from
  // match_seed so all GT runners (selfplay/arena/heuristic + API session)
  // sharing the same episode seed produce identical truth draws.
  std::mt19937_64 step_rng(match_seed);

  // Per-seat session state is mandatory (mirrors selfplay_runner).
  const int num_players = state->num_players();
  if (static_cast<int>(per_seat_states.size()) != num_players) {
    throw std::runtime_error(
        "run_arena_match: per_seat_states size != num_players");
  }
  for (int p = 0; p < num_players; ++p) {
    if (per_seat_states[p] == nullptr) {
      throw std::runtime_error(
          "run_arena_match: per_seat_states[p] must not be null");
    }
  }

  auto ai_view_for = [&](int seat) -> IGameState& {
    return *per_seat_states[seat];
  };

  // Init each seat's tracker once at match start. Every public event is
  // fed to every tracker in the main loop below.
  const bool use_per_perspective = !per_perspective_trackers.empty();
  if (use_per_perspective) {
    if (static_cast<int>(per_perspective_trackers.size()) != num_players) {
      throw std::runtime_error(
          "run_arena_match: per_perspective_trackers size != num_players");
    }
    for (int p = 0; p < num_players; ++p) {
      if (per_perspective_trackers[p]) {
        per_perspective_trackers[p]->init(*per_seat_states[p], p, AnyMap{});
      }
    }
  }

  // No episode-start freshen — see selfplay_runner.cpp for rationale.

  // Per-seat session-state advance: mirrors selfplay_runner's
  // advance_per_seat_states.
  auto advance_per_seat_states =
      [&](const IGameState& truth_before, const IGameState& truth_after,
          ActionId chosen, int /*actor*/) {
    for (int p = 0; p < num_players; ++p) {
      IGameState& seat = *per_seat_states[p];
      if (!public_event_extractor) {
        const std::uint64_t view_step_seed = board_ai::rng::derive_subseed(
            match_seed, "arena.view_step",
            static_cast<std::uint64_t>(ply) * 17ULL +
                static_cast<std::uint64_t>(p));
        std::mt19937_64 view_step_rng(view_step_seed);
        rules.do_action_fast(seat, chosen, view_step_rng);
        continue;
      }
      PublicEventTrace evt_p = public_event_extractor(
          truth_before, chosen, truth_after, p);
      seat.begin_step_for_session_observe();
      if (public_state_applier && !evt_p.public_snapshot.empty()) {
        public_state_applier(seat, evt_p.public_snapshot, p);
      }
    }
  };

  while (!state->is_terminal() && ply < max_game_plies) {
    const int player = state->current_player();
    IGameState& ai_view = ai_view_for(player);
    const auto legal = rules.legal_actions(ai_view);
    if (legal.empty()) break;

    const size_t cfg_idx = player_configs.empty()
        ? 0 : static_cast<size_t>(player) % player_configs.size();
    const ArenaPlayerConfig& pcfg = player_configs.empty()
        ? ArenaPlayerConfig{} : player_configs[cfg_idx];
    const search::IPolicyValueEvaluator& eval = evaluator_for_player(player);

    // MCTS tracker source: per-seat trackers when supplied; otherwise the
    // optional singular tracker (re-init each ply for callers that didn't
    // pre-allocate per-seat).
    IBeliefTracker* mcts_tracker = nullptr;
    if (use_per_perspective && player >= 0 &&
        player < static_cast<int>(per_perspective_trackers.size())) {
      mcts_tracker = per_perspective_trackers[player];
    } else if (belief_tracker) {
      belief_tracker->init(ai_view, player, AnyMap{});
      mcts_tracker = belief_tracker;
    }

    search::NetMctsConfig mcts_cfg{};
    mcts_cfg.simulations = pcfg.simulations;
    mcts_cfg.world_cap = pcfg.world_cap;
    mcts_cfg.c_puct = pcfg.c_puct;
    mcts_cfg.fpu_reduction = pcfg.fpu_reduction;
    mcts_cfg.uct2_arrival_visits = pcfg.uct2_arrival_visits;
    mcts_cfg.max_depth = pcfg.max_depth;
    mcts_cfg.value_clip = pcfg.value_clip;
    mcts_cfg.opponent_selection = pcfg.opponent_selection;
    mcts_cfg.safe_mode = pcfg.safe_mode;
    mcts_cfg.leaf_batch_size = pcfg.leaf_batch_size;
    if (mcts_tracker) {
      mcts_cfg.root_belief_tracker = mcts_tracker;
    }
    // Sim-tracker descent maintenance (SIM_TRACKER_DESCENT_PLAN). Uses
    // events-only extractor — see selfplay_runner.cpp for rationale.
    if (events_only_extractor) {
      mcts_cfg.events_only_extractor = events_only_extractor;
    }
    {
      const IBeliefEvaluator* seat_belief_eval =
          (!per_seat_belief_evaluators.empty() &&
           player >= 0 &&
           player < static_cast<int>(per_seat_belief_evaluators.size()))
              ? per_seat_belief_evaluators[player]
              : nullptr;
      if (belief_extractor && seat_belief_eval) {
        mcts_cfg.belief_extractor = belief_extractor;
        mcts_cfg.belief_evaluator = seat_belief_eval;
      }
    }

    if (pcfg.tail_solve_enabled) {
      if (!pcfg.tail_solver || !pcfg.tail_solve_trigger) {
        throw std::invalid_argument(
            "run_arena_match: tail_solve_enabled=true requires both a "
            "registered ITailSolver and a TailSolveTrigger.");
      }
      if (pcfg.tail_solve_trigger(ai_view, ply)) {
        mcts_cfg.tail_solve_enabled = true;
        mcts_cfg.tail_solve_config = pcfg.tail_solve_config;
        mcts_cfg.tail_solver = pcfg.tail_solver;
      }
    }

    search::NetMcts mcts(mcts_cfg);
    search::NetMctsStats stats{};
    const std::uint64_t mcts_seed = board_ai::rng::derive_subseed(
        match_seed, "arena.mcts", static_cast<std::uint64_t>(ply));
    mcts.search_root(ai_view, rules, value_model, eval, &stats, mcts_seed);

    const std::uint64_t action_seed = board_ai::rng::derive_subseed(
        match_seed, "arena.action_pick", static_cast<std::uint64_t>(ply));
    const ActionId chosen = search::select_action_from_visits(
        stats.root_actions, stats.root_action_visits, pcfg.temperature, action_seed,
        pcfg.top_k, legal[0]);

    result.action_history.push_back(chosen);
    result.ply_stats.push_back({stats.tail_solved, stats.tail_solve_value});
    std::unique_ptr<IGameState> state_before = state->clone_state();
    const ActionId observed = rules.public_action(*state_before, chosen);
    rules.do_action_fast(*state, chosen, step_rng);
    if (use_per_perspective) {
      const int num_players = static_cast<int>(per_perspective_trackers.size());
      for (int p = 0; p < num_players; ++p) {
        if (!per_perspective_trackers[p]) continue;
        PublicEventTrace evt_p;
        if (public_event_extractor) {
          evt_p = public_event_extractor(*state_before, chosen, *state, p);
        }
        per_perspective_trackers[p]->observe_public_event(
            player, observed, evt_p.events);
      }
    } else if (belief_tracker) {
      PublicEventTrace evt;
      if (public_event_extractor) {
        evt = public_event_extractor(*state_before, chosen, *state, player);
      }
      belief_tracker->observe_public_event(
          player, observed, evt.events);
    }
    advance_per_seat_states(*state_before, *state, chosen, player);
    ply += 1;
  }

  result.total_plies = ply;

  if (!state->is_terminal() && adjudicator) {
    const int adj_winner = adjudicator(*state);
    if (adj_winner < 0) {
      result.draw = true;
      result.winner = -1;
    } else {
      result.draw = false;
      result.winner = adj_winner;
    }
    result.terminal_z_values.assign(
        static_cast<size_t>(state->num_players()), -1.0f);
    return result;
  }

  if (state->is_terminal()) {
    const int num_players = state->num_players();
    result.terminal_z_values = value_model.terminal_values(*state);
    float best_val = -2.0f;
    int best_player = -1;
    for (int p = 0; p < num_players; ++p) {
      const float v = result.terminal_z_values[static_cast<size_t>(p)];
      if (v > best_val) {
        best_val = v;
        best_player = p;
      }
    }
    bool is_draw = true;
    for (int p = 0; p < num_players; ++p) {
      const float v = result.terminal_z_values[static_cast<size_t>(p)];
      if (std::abs(v - best_val) > 1e-6f) {
        is_draw = false;
        break;
      }
    }
    result.draw = is_draw;
    result.winner = is_draw ? -1 : best_player;
  } else {
    result.draw = true;
    result.winner = -1;
  }

  return result;
}

}  // namespace board_ai::runtime
