#include "heuristic_runner.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

#include "../core/masked_state.h"
#include "../core/rng_salt.h"

namespace board_ai::runtime {

std::size_t sample_heuristic_index(
    const std::vector<double>& scores,
    double temperature,
    double rng_u01) {
  if (scores.empty()) return 0;
  const double max_score = *std::max_element(scores.begin(), scores.end());
  if (temperature <= 1e-6) {
    // Uniform random over all argmax ties (contract documented in the
    // development guide; selfplay_runner does the same). Do NOT return the
    // first max_element — for games where many actions share the top score
    // (e.g. Quoridor, where every pawn-advance and every effective wall both
    // score +1), that degenerates to "always pick the first action in the
    // legal-action ordering" and makes the heuristic deterministic.
    std::size_t count = 0;
    for (std::size_t i = 0; i < scores.size(); ++i) {
      if (scores[i] >= max_score - 1e-9) ++count;
    }
    if (count <= 1) {
      auto it = std::max_element(scores.begin(), scores.end());
      return static_cast<std::size_t>(it - scores.begin());
    }
    std::size_t pick = static_cast<std::size_t>(rng_u01 * static_cast<double>(count));
    if (pick >= count) pick = count - 1;
    std::size_t seen = 0;
    for (std::size_t i = 0; i < scores.size(); ++i) {
      if (scores[i] >= max_score - 1e-9) {
        if (seen == pick) return i;
        ++seen;
      }
    }
    return scores.size() - 1;
  }
  const double inv_temp = 1.0 / temperature;
  std::vector<double> probs(scores.size());
  double sum_exp = 0.0;
  for (std::size_t i = 0; i < scores.size(); ++i) {
    probs[i] = std::exp((scores[i] - max_score) * inv_temp);
    sum_exp += probs[i];
  }
  if (sum_exp <= 0.0) return 0;
  const double r = rng_u01 * sum_exp;
  double cumulative = 0.0;
  for (std::size_t i = 0; i < probs.size(); ++i) {
    cumulative += probs[i];
    if (r <= cumulative) return i;
  }
  return probs.size() - 1;
}

SelfplayEpisodeResult run_heuristic_episode_per_seat(
    IGameState& initial_state,
    const IGameRules& rules,
    const IStateValueModel& value_model,
    const IFeatureEncoder* encoder,
    const std::vector<const IHeuristic*>& per_seat_heuristics,
    const std::vector<double>& per_seat_temperatures,
    int max_game_plies,
    std::uint64_t episode_seed,
    AuxiliaryScorer auxiliary_scorer,
    GameAdjudicator adjudicator,
    std::vector<IBeliefTracker*> per_perspective_trackers,
    std::vector<IGameState*> per_seat_states,
    PublicStateApplier public_state_applier,
    PublicEventExtractor public_event_extractor) {
  SelfplayEpisodeResult result{};
  const int num_players_init = initial_state.num_players();
  if (static_cast<int>(per_seat_heuristics.size()) != num_players_init) {
    throw std::runtime_error(
        "run_heuristic_episode_per_seat: per_seat_heuristics size != num_players");
  }
  if (static_cast<int>(per_seat_temperatures.size()) != num_players_init) {
    throw std::runtime_error(
        "run_heuristic_episode_per_seat: per_seat_temperatures size != num_players");
  }
  for (int p = 0; p < num_players_init; ++p) {
    if (per_seat_heuristics[p] == nullptr) {
      throw std::runtime_error(
          "run_heuristic_episode_per_seat: per_seat_heuristics[p] must not be null");
    }
  }

  auto state = initial_state.clone_state();
  int ply = 0;

  std::uint64_t rng_counter = 0;
  auto next_rng = [&rng_counter, episode_seed]() -> std::uint64_t {
    return board_ai::rng::derive_subseed(
        episode_seed, "heuristic.draw", rng_counter++);
  };

  // Step rng feeds do_action_fast hidden-info draws. Seeded directly from
  // episode_seed so all GT runners sharing the same seed advance truth
  // through identical draws.
  std::mt19937_64 step_rng(episode_seed);

  // Per-seat session state is mandatory.
  const int num_players = state->num_players();
  if (static_cast<int>(per_seat_states.size()) != num_players) {
    throw std::runtime_error(
        "run_heuristic_episode: per_seat_states size != num_players");
  }
  for (int p = 0; p < num_players; ++p) {
    if (per_seat_states[p] == nullptr) {
      throw std::runtime_error(
          "run_heuristic_episode: per_seat_states[p] must not be null");
    }
  }

  auto ai_view_for = [&](int seat) -> IGameState& {
    return *per_seat_states[seat];
  };

  const bool use_per_perspective = !per_perspective_trackers.empty();
  if (use_per_perspective) {
    if (static_cast<int>(per_perspective_trackers.size()) != num_players) {
      throw std::runtime_error(
          "run_heuristic_episode: per_perspective_trackers size != num_players");
    }
    for (int p = 0; p < num_players; ++p) {
      if (per_perspective_trackers[p]) {
        per_perspective_trackers[p]->init(*per_seat_states[p], p, AnyMap{});
      }
    }
  }

  // No episode-start freshen — see selfplay_runner.cpp for rationale.

  auto advance_per_seat_states =
      [&](const IGameState& truth_before, const IGameState& truth_after,
          ActionId chosen, int /*actor*/) {
    for (int p = 0; p < num_players; ++p) {
      IGameState& seat = *per_seat_states[p];
      if (!public_event_extractor) {
        const std::uint64_t view_step_seed = board_ai::rng::derive_subseed(
            episode_seed, "heuristic.view_step",
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
    auto legal = rules.legal_actions(ai_view);
    if (legal.empty()) break;

    std::vector<float> features;
    std::vector<float> legal_mask;
    if (encoder) {
      const IBeliefTracker* enc_tracker =
          (use_per_perspective && player >= 0 &&
           player < static_cast<int>(per_perspective_trackers.size()))
              ? per_perspective_trackers[player]
              : nullptr;
      encoder->encode(ai_view, player, enc_tracker, legal, &features, &legal_mask);
    }

    auto hr_masked = make_masked_state(ai_view, ai_view.schema_ref(), player);
    const IBeliefTracker* hr_tracker =
        (use_per_perspective && player >= 0 &&
         player < static_cast<int>(per_perspective_trackers.size()))
            ? per_perspective_trackers[player]
            : nullptr;
    const IHeuristic& seat_heuristic = *per_seat_heuristics[static_cast<size_t>(player)];
    const double seat_temperature = per_seat_temperatures[static_cast<size_t>(player)];
    HeuristicResult hr = seat_heuristic.pick(*hr_masked, hr_tracker, rules, next_rng());

    if (hr.scores.empty() || hr.actions.empty()) {
      throw std::runtime_error(
          "run_heuristic_episode: heuristic returned empty result at ply " +
          std::to_string(ply));
    }

    double rng_u01 = static_cast<double>(next_rng() & 0xFFFFFFFF) / 4294967296.0;
    size_t chosen_idx = sample_heuristic_index(hr.scores, seat_temperature, rng_u01);
    ActionId chosen = hr.actions[chosen_idx];

    // Convert scores to fake visits for pipeline compatibility
    std::vector<int> fake_visits(hr.scores.size());
    if (seat_temperature <= 1e-6) {
      double max_sc = *std::max_element(hr.scores.begin(), hr.scores.end());
      for (size_t i = 0; i < hr.scores.size(); ++i) {
        fake_visits[i] = (hr.scores[i] >= max_sc - 1e-9) ? 1 : 0;
      }
    } else {
      double max_sc = *std::max_element(hr.scores.begin(), hr.scores.end());
      double inv_t = 1.0 / seat_temperature;
      double sum_exp = 0.0;
      std::vector<double> probs(hr.scores.size());
      for (size_t i = 0; i < hr.scores.size(); ++i) {
        probs[i] = std::exp((hr.scores[i] - max_sc) * inv_t);
        sum_exp += probs[i];
      }
      for (size_t i = 0; i < probs.size(); ++i) {
        fake_visits[i] = static_cast<int>(probs[i] / sum_exp * 10000.0 + 0.5);
      }
    }

    SelfplaySample sample{};
    sample.ply = ply;
    sample.player = player;
    sample.action_id = chosen;
    sample.policy_action_ids = hr.actions;
    sample.policy_action_visits = fake_visits;
    sample.heuristic_simulations = hr.simulations;
    sample.features = features;
    sample.legal_mask = legal_mask;
    result.samples.push_back(std::move(sample));

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
    }
    advance_per_seat_states(*state_before, *state, chosen, player);
    ++ply;
  }

  result.total_plies = ply;
  result.final_state = state->clone_state();

  if (state->is_terminal()) {
    const int num_players = state->num_players();
    float best_val = -2.0f;
    int best_player = -1;
    bool is_draw = true;
    for (int p = 0; p < num_players; ++p) {
      const float v = value_model.terminal_value_for_player(*state, p);
      if (v > best_val) {
        best_val = v;
        best_player = p;
      }
    }
    for (int p = 0; p < num_players; ++p) {
      const float v = value_model.terminal_value_for_player(*state, p);
      if (std::abs(v - best_val) > 1e-6f) {
        is_draw = false;
        break;
      }
    }
    if (is_draw) {
      result.draw = true;
      result.winner = -1;
    } else {
      result.draw = false;
      result.winner = best_player;
    }

    auto terminal_vals = value_model.terminal_values(*state);
    // Episode-level copy, as in run_selfplay_episode: consumers label a
    // seat's result by its own terminal value (shared first place in 3-4
    // player games has no single `winner`).
    result.terminal_z_values = terminal_vals;
    for (auto& s : result.samples) {
      s.z_values = terminal_vals;
      s.z = result.draw ? 0.0f : ((s.player == result.winner) ? 1.0f : -1.0f);
    }
  } else if (adjudicator) {
    const int adj_winner = adjudicator(*state);
    if (adj_winner < 0) {
      result.draw = true;
      result.winner = -1;
    } else {
      result.draw = false;
      result.winner = adj_winner;
    }
    const int np = state->num_players();
    std::vector<float> adj_vals(np, 0.0f);
    if (!result.draw) {
      const float loser_val = -1.0f / static_cast<float>(np - 1);
      for (int p = 0; p < np; ++p) {
        adj_vals[p] = (p == result.winner) ? 1.0f : loser_val;
      }
    }
    assert(std::abs(std::accumulate(adj_vals.begin(), adj_vals.end(), 0.0f)) < 1e-4f);
    for (auto& s : result.samples) {
      s.z_values = adj_vals;
      s.z = result.draw ? 0.0f : ((s.player == result.winner) ? 1.0f : -1.0f);
    }
  } else {
    for (auto& s : result.samples) {
      s.z = 0.0f;
      s.z_values.assign(static_cast<size_t>(state->num_players()), 0.0f);
    }
  }

  return result;
}

SelfplayEpisodeResult run_heuristic_episode(
    IGameState& initial_state,
    const IGameRules& rules,
    const IStateValueModel& value_model,
    const IFeatureEncoder* encoder,
    const IHeuristic& heuristic,
    double temperature,
    int max_game_plies,
    std::uint64_t episode_seed,
    AuxiliaryScorer auxiliary_scorer,
    GameAdjudicator adjudicator,
    std::vector<IBeliefTracker*> per_perspective_trackers,
    std::vector<IGameState*> per_seat_states,
    PublicStateApplier public_state_applier,
    PublicEventExtractor public_event_extractor) {
  const int n = initial_state.num_players();
  std::vector<const IHeuristic*> per_seat_heuristics(static_cast<size_t>(n), &heuristic);
  std::vector<double> per_seat_temperatures(static_cast<size_t>(n), temperature);
  return run_heuristic_episode_per_seat(
      initial_state, rules, value_model, encoder,
      per_seat_heuristics, per_seat_temperatures,
      max_game_plies, episode_seed, std::move(auxiliary_scorer),
      std::move(adjudicator), std::move(per_perspective_trackers),
      std::move(per_seat_states), std::move(public_state_applier),
      std::move(public_event_extractor));
}

}  // namespace board_ai::runtime
