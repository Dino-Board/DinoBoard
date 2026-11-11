#include "azul_rule_heuristic.h"

#include <algorithm>
#include <cmath>

#include "../azul_state.h"

namespace board_ai::azul {

namespace {

constexpr int kFloorPenalties[7] = {1, 1, 2, 2, 2, 3, 3};

inline int floor_penalty(int floor_before, int added) {
  int total = 0;
  for (int i = 0; i < added; ++i) {
    int idx = std::min(floor_before + i, 6);
    total += kFloorPenalties[idx];
  }
  return total;
}

inline int wall_col(int row, int color_idx) {
  return (color_idx + row) % kColors;
}

template <typename PlayerT>
int wall_placement_score(const PlayerT& p, int row, int col) {
  auto filled = [&](int r, int c) -> bool {
    if (r < 0 || r >= kRows || c < 0 || c >= kColors) return false;
    return ((p.wall_mask[r] >> c) & 1U) != 0U;
  };
  int h = 1;
  for (int c = col - 1; c >= 0 && filled(row, c); --c) ++h;
  for (int c = col + 1; c < kColors && filled(row, c); ++c) ++h;
  int v = 1;
  for (int r = row - 1; r >= 0 && filled(r, col); --r) ++v;
  for (int r = row + 1; r < kRows && filled(r, col); ++r) ++v;
  if (h == 1 && v == 1) return 1;
  if (h > 1 && v > 1) return h + v;
  return std::max(h, v);
}

template <int NPlayers>
double score_action_on_state(const AzulState<NPlayers>& s, ActionId action) {
  using Cfg = AzulConfig<NPlayers>;

  const int source = static_cast<int>(action / (kColors * kTargetsPerColor));
  const int color = static_cast<int>((action / kTargetsPerColor) % kColors);
  const int target = static_cast<int>(action % kTargetsPerColor);
  const int actor = s.current_player_;
  const auto& me = s.players[actor];

  int available = 0;
  if (source == Cfg::kCenterSource) {
    available = s.center[color];
  } else if (source >= 0 && source < Cfg::kFactories) {
    available = s.factories[source][color];
  }
  if (available == 0) return -1e6;

  double score = 0.0;

  int floor_add = 0;
  if (source == Cfg::kCenterSource && s.first_player_token_in_center) {
    floor_add += 1;
    score += 0.3;
  }

  if (target == 5) {
    floor_add += available;
    score -= static_cast<double>(floor_penalty(me.floor_count, floor_add)) * 1.0;
    return score;
  }

  const int line_row = target;
  const int line_cap = line_row + 1;
  const int line_len = me.line_len[line_row];
  const int existing_color = me.line_color[line_row];

  const int wall_col_idx = wall_col(line_row, color);
  if (((me.wall_mask[line_row] >> wall_col_idx) & 1U) != 0U) {
    return -1e5;
  }
  if (existing_color >= 0 && existing_color != color) {
    return -1e5;
  }

  const int space_left = line_cap - line_len;
  const int placed = std::min(available, space_left);
  const int overflow = available - placed;

  if (placed + line_len == line_cap) {
    double wall_score = static_cast<double>(
        wall_placement_score(me, line_row, wall_col_idx));
    score += wall_score * 1.2;
    score += line_cap * 0.5;
  } else {
    double progress = static_cast<double>(placed + line_len) / line_cap;
    score += progress * 0.6;
  }

  if (overflow > 0) {
    floor_add += overflow;
  }
  score -= static_cast<double>(floor_penalty(me.floor_count, floor_add)) * 1.0;

  return score;
}

}  // namespace

template <int NPlayers>
HeuristicResult AzulRuleHeuristic<NPlayers>::pick(
    IGameState& masked_state,
    const IBeliefTracker* /*tracker*/,
    const IGameRules& rules,
    std::uint64_t /*rng_seed*/) const {
  auto& s = checked_cast<AzulState<NPlayers>>(masked_state);
  auto legal = rules.legal_actions(masked_state);
  HeuristicResult result;
  result.actions = legal;
  result.scores.reserve(legal.size());
  for (ActionId a : legal) {
    result.scores.push_back(score_action_on_state<NPlayers>(s, a));
  }
  return result;
}

template class AzulRuleHeuristic<2>;
template class AzulRuleHeuristic<3>;
template class AzulRuleHeuristic<4>;

}  // namespace board_ai::azul
