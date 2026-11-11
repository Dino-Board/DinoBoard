#include "quoridor_rule_heuristic.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "../quoridor_state.h"
#include "../quoridor_rules.h"

namespace board_ai::quoridor {

namespace {

// Probable wall mask. Adapted from quoridor-ai (gorisanson/quoridor-ai)
// game.js — `_probableNextWalls` + `setWallsBesidePawn` + leftmost/rightmost
// horizontal heuristic.
struct ProbableWallMasks {
  std::array<std::uint8_t, kWallSlots> h{};
  std::array<std::uint8_t, kWallSlots> v{};
};

void mark_walls_beside_pawn(ProbableWallMasks& mask, int row, int col) {
  auto seth = [&](int r, int c) {
    if (r < 0 || r >= kWallGrid || c < 0 || c >= kWallGrid) return;
    mask.h[static_cast<size_t>(wall_index(r, c))] = 1;
  };
  auto setv = [&](int r, int c) {
    if (r < 0 || r >= kWallGrid || c < 0 || c >= kWallGrid) return;
    mask.v[static_cast<size_t>(wall_index(r, c))] = 1;
  };
  if (row >= 1) {
    if (col >= 1) {
      seth(row - 1, col - 1); setv(row - 1, col - 1);
      if (col >= 2) seth(row - 1, col - 2);
    }
    if (col <= 7) {
      seth(row - 1, col); setv(row - 1, col);
      if (col <= 6) seth(row - 1, col + 1);
    }
    if (row >= 2) {
      if (col >= 1) setv(row - 2, col - 1);
      if (col <= 7) setv(row - 2, col);
    }
  }
  if (row <= 7) {
    if (col >= 1) {
      seth(row, col - 1); setv(row, col - 1);
      if (col >= 2) seth(row, col - 2);
    }
    if (col <= 7) {
      seth(row, col); setv(row, col);
      if (col <= 6) seth(row, col + 1);
    }
    if (row <= 6) {
      if (col >= 1) setv(row + 1, col - 1);
      if (col <= 7) setv(row + 1, col);
    }
  }
}

void mark_walls_near_existing(ProbableWallMasks& mask, const QuoridorState& s) {
  auto seth = [&](int r, int c) {
    if (r < 0 || r >= kWallGrid || c < 0 || c >= kWallGrid) return;
    mask.h[static_cast<size_t>(wall_index(r, c))] = 1;
  };
  auto setv = [&](int r, int c) {
    if (r < 0 || r >= kWallGrid || c < 0 || c >= kWallGrid) return;
    mask.v[static_cast<size_t>(wall_index(r, c))] = 1;
  };
  for (int r = 0; r < kWallGrid; ++r) {
    for (int c = 0; c < kWallGrid; ++c) {
      if (s.h_walls[static_cast<size_t>(wall_index(r, c))]) {
        if (r >= 1) setv(r - 1, c);
        if (r <= 6) setv(r + 1, c);
        if (c >= 1) {
          setv(r, c - 1);
          if (r >= 1) setv(r - 1, c - 1);
          if (r <= 6) setv(r + 1, c - 1);
          if (c >= 2) {
            seth(r, c - 2); setv(r, c - 2);
            if (c >= 3) seth(r, c - 3);
          }
        }
        if (c <= 6) {
          setv(r, c + 1);
          if (r >= 1) setv(r - 1, c + 1);
          if (r <= 6) setv(r + 1, c + 1);
          if (c <= 5) {
            seth(r, c + 2); setv(r, c + 2);
            if (c <= 4) seth(r, c + 3);
          }
        }
      }
      if (s.v_walls[static_cast<size_t>(wall_index(r, c))]) {
        if (c >= 1) seth(r, c - 1);
        if (c <= 6) seth(r, c + 1);
        if (r >= 1) {
          seth(r - 1, c);
          if (c >= 1) seth(r - 1, c - 1);
          if (c <= 6) seth(r - 1, c + 1);
          if (r >= 2) {
            setv(r - 2, c); seth(r - 2, c);
            if (r >= 3) setv(r - 3, c);
          }
        }
        if (r <= 6) {
          seth(r + 1, c);
          if (c >= 1) seth(r + 1, c - 1);
          if (c <= 6) seth(r + 1, c + 1);
          if (r <= 5) {
            setv(r + 2, c); seth(r + 2, c);
            if (r <= 4) setv(r + 3, c);
          }
        }
      }
    }
  }
}

ProbableWallMasks compute_probable_wall_masks(const QuoridorState& s) {
  ProbableWallMasks mask{};
  mark_walls_near_existing(mask, s);
  for (int r = 0; r < kWallGrid; ++r) {
    mask.h[static_cast<size_t>(wall_index(r, 0))] = 1;
    mask.h[static_cast<size_t>(wall_index(r, 7))] = 1;
  }
  const int me = s.current_player_;
  const int opp = 1 - me;
  mark_walls_beside_pawn(
      mask,
      s.pawn_row[static_cast<size_t>(opp)],
      s.pawn_col[static_cast<size_t>(opp)]);
  mark_walls_beside_pawn(
      mask,
      s.pawn_row[static_cast<size_t>(me)],
      s.pawn_col[static_cast<size_t>(me)]);
  return mask;
}

HeuristicResult quoridor_rule_score(
    IGameState& state, const IGameRules& rules) {
  auto legal = rules.legal_actions(state);
  const auto& s = checked_cast<QuoridorState>(state);
  const int me = s.current_player_;
  const int opp = 1 - me;

  const int d_me_before = QuoridorRules::shortest_path_distance(s, me);
  const int d_opp_before = QuoridorRules::shortest_path_distance(s, opp);
  const int my_walls = s.walls_remaining[static_cast<size_t>(me)];
  const int opp_walls = s.walls_remaining[static_cast<size_t>(opp)];

  const bool opp_no_walls = (opp_walls == 0);
  const ProbableWallMasks mask = compute_probable_wall_masks(s);

  HeuristicResult result;
  result.actions = legal;
  result.scores.resize(legal.size());

  constexpr double kNegInf = -1e9;

  for (size_t i = 0; i < legal.size(); ++i) {
    const ActionId a = legal[i];

    if (is_move_action(a)) {
      auto clone = state.clone_state();
      rules.do_action_deterministic(*clone, a);
      const auto& after = checked_cast<QuoridorState>(*clone);
      const int d_me = QuoridorRules::shortest_path_distance(after, me);
      const int d_opp = QuoridorRules::shortest_path_distance(after, opp);

      if (opp_no_walls) {
        if (d_me >= d_me_before) {
          result.scores[i] = kNegInf;
          continue;
        }
      }

      double score = static_cast<double>(d_opp - d_me);
      score += 0.05 * static_cast<double>(d_me_before - d_me);
      result.scores[i] = score;
      continue;
    }

    int wr, wc;
    bool is_h = is_hwall_action(a);
    if (is_h) { wr = decode_hwall_row(a); wc = decode_hwall_col(a); }
    else      { wr = decode_vwall_row(a); wc = decode_vwall_col(a); }

    const bool probable = is_h
        ? mask.h[static_cast<size_t>(wall_index(wr, wc))] != 0
        : mask.v[static_cast<size_t>(wall_index(wr, wc))] != 0;

    if (!probable) {
      result.scores[i] = kNegInf;
      continue;
    }

    if (s.move_count < 2) {
      result.scores[i] = kNegInf;
      continue;
    }

    auto clone = state.clone_state();
    rules.do_action_deterministic(*clone, a);
    const auto& after = checked_cast<QuoridorState>(*clone);
    const int d_me = QuoridorRules::shortest_path_distance(after, me);
    const int d_opp = QuoridorRules::shortest_path_distance(after, opp);
    const int d_me_delta = d_me - d_me_before;
    const int d_opp_delta = d_opp - d_opp_before;

    if (opp_no_walls && d_opp_delta <= 0) {
      result.scores[i] = kNegInf;
      continue;
    }

    double score = static_cast<double>(d_opp_delta - d_me_delta);
    if (my_walls <= 2) score -= 1.0;
    result.scores[i] = score;
  }

  bool any_finite = false;
  for (double v : result.scores) {
    if (v > kNegInf / 2.0) { any_finite = true; break; }
  }
  if (!any_finite) {
    for (size_t i = 0; i < legal.size(); ++i) {
      auto clone = state.clone_state();
      rules.do_action_deterministic(*clone, legal[i]);
      const auto& after = checked_cast<QuoridorState>(*clone);
      const int d_me = QuoridorRules::shortest_path_distance(after, me);
      const int d_opp = QuoridorRules::shortest_path_distance(after, opp);
      result.scores[i] = static_cast<double>(d_opp - d_me);
    }
  }

  return result;
}

}  // namespace

bool QuoridorRulePVEvaluator::evaluate(
    const IGameState& masked_state,
    int perspective_player,
    const IBeliefTracker* /*tracker*/,
    const std::vector<ActionId>& legal_actions,
    std::vector<float>* priors,
    std::vector<float>* values) const {
  if (legal_actions.empty()) return false;

  auto state_clone = masked_state.clone_state();
  HeuristicResult hr = quoridor_rule_score(*state_clone, *rules_);

  priors->assign(legal_actions.size(), 0.0f);
  std::vector<double> raw(legal_actions.size(), -1e9);
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    for (size_t j = 0; j < hr.actions.size(); ++j) {
      if (hr.actions[j] == legal_actions[i]) {
        raw[i] = hr.scores[j];
        break;
      }
    }
  }
  double max_score = raw[0];
  for (double v : raw) if (v > max_score) max_score = v;
  double sum_exp = 0.0;
  std::vector<double> probs(legal_actions.size());
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    probs[i] = std::exp(raw[i] - max_score);
    sum_exp += probs[i];
  }
  if (sum_exp <= 0.0) {
    for (size_t i = 0; i < legal_actions.size(); ++i) {
      (*priors)[i] = 1.0f / static_cast<float>(legal_actions.size());
    }
  } else {
    for (size_t i = 0; i < legal_actions.size(); ++i) {
      (*priors)[i] = static_cast<float>(probs[i] / sum_exp);
    }
  }

  const auto& qs = checked_cast<QuoridorState>(masked_state);
  const int np = qs.num_players();
  values->assign(static_cast<size_t>(np), 0.0f);
  const int d_persp = QuoridorRules::shortest_path_distance(qs, perspective_player);
  const int d_other = QuoridorRules::shortest_path_distance(qs, 1 - perspective_player);
  const float v_persp = std::tanh(static_cast<float>(d_other - d_persp) / 8.0f);
  (*values)[static_cast<size_t>(perspective_player)] = v_persp;
  (*values)[static_cast<size_t>(1 - perspective_player)] = -v_persp;
  return true;
}

}  // namespace board_ai::quoridor
