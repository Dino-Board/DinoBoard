#include "splendor_rule_heuristic.h"

#include <algorithm>

#include "../splendor_state.h"

namespace board_ai::splendor {

namespace {

template <int NPlayers>
int deficit_for_card(const SplendorData<NPlayers>& d, int player, const SplendorCard& card) {
  int total_short = 0;
  int gold = d.player_gems[player][5];
  for (int c = 0; c < kColorCount; ++c) {
    int raw = card.cost[c];
    int bonus = d.player_bonuses[player][c];
    int have = d.player_gems[player][c];
    int need = std::max(0, raw - bonus);
    int short_c = std::max(0, need - have);
    int gold_use = std::min(short_c, gold);
    short_c -= gold_use;
    gold -= gold_use;
    total_short += short_c;
  }
  return total_short;
}

template <int NPlayers>
int nearest_noble_missing(const SplendorData<NPlayers>& d, int player) {
  const auto& nobles = splendor_nobles();
  int best = 99;
  for (int i = 0; i < d.nobles_size; ++i) {
    int nid = d.nobles[i];
    if (nid < 0 || nid >= static_cast<int>(nobles.size())) continue;
    int missing = 0;
    for (int c = 0; c < kColorCount; ++c) {
      missing += std::max(0, static_cast<int>(nobles[nid][c]) - d.player_bonuses[player][c]);
    }
    best = std::min(best, missing);
  }
  return best;
}

template <int NPlayers>
double eval_position(const SplendorData<NPlayers>& d, int player) {
  if (d.terminal) {
    if (d.winner < 0) return 0.0;
    return d.winner == player ? 1000.0 : -1000.0;
  }
  double score = 0.0;
  for (int p = 0; p < NPlayers; ++p) {
    double sign = (p == player) ? 1.0 : -1.0 / std::max(1, NPlayers - 1);
    score += sign * static_cast<double>(d.player_points[p]) * 8.0;
  }
  for (int p = 0; p < NPlayers; ++p) {
    double sign = (p == player) ? 1.0 : -0.5 / std::max(1, NPlayers - 1);
    int bonus_total = 0;
    for (int c = 0; c < kColorCount; ++c) bonus_total += d.player_bonuses[p][c];
    score += sign * static_cast<double>(bonus_total) * 0.8;
  }
  int my_missing = nearest_noble_missing(d, player);
  score -= static_cast<double>(std::min(my_missing, 5)) * 0.6;
  const auto& cards = splendor_card_pool();
  for (int tier = 0; tier < 3; ++tier) {
    for (int slot = 0; slot < d.tableau_size[tier]; ++slot) {
      int cid = d.tableau[tier][slot];
      if (cid < 0 || cid >= static_cast<int>(cards.size())) continue;
      int deficit = deficit_for_card(d, player, cards[cid]);
      if (deficit == 0) {
        score += 0.8 + static_cast<double>(cards[cid].points) * 0.4;
      } else if (deficit <= 2) {
        score += 0.25;
      }
    }
  }
  for (int slot = 0; slot < d.reserved_size[player]; ++slot) {
    int cid = d.reserved[player][slot];
    if (cid < 0 || cid >= static_cast<int>(cards.size())) continue;
    int deficit = deficit_for_card(d, player, cards[cid]);
    score += static_cast<double>(cards[cid].points) * 0.6;
    score -= static_cast<double>(deficit) * 0.35;
  }
  for (int p = 0; p < NPlayers; ++p) {
    if (p == player) continue;
    for (int slot = 0; slot < d.reserved_size[p]; ++slot) {
      bool visible = d.reserved_visible[p][slot] != 0;
      if (visible) {
        int cid = d.reserved[p][slot];
        if (cid < 0 || cid >= static_cast<int>(cards.size())) continue;
        score -= static_cast<double>(cards[cid].points) * 0.5;
        int op_def = deficit_for_card(d, p, cards[cid]);
        if (op_def == 0) score -= 0.5;
      } else {
        score -= 0.15;
      }
    }
  }
  int my_gems = 0;
  for (int i = 0; i < 6; ++i) my_gems += d.player_gems[player][i];
  if (my_gems > 10) score -= static_cast<double>(my_gems - 10) * 0.8;
  return score;
}

}  // namespace

template <int NPlayers>
HeuristicResult SplendorRuleHeuristic<NPlayers>::pick(
    IGameState& masked_state,
    const IBeliefTracker* /*tracker*/,
    const IGameRules& rules,
    std::uint64_t /*rng_seed*/) const {
  auto& s = checked_cast<SplendorState<NPlayers>>(masked_state);
  auto legal = rules.legal_actions(masked_state);
  const int player = s.current_player();

  HeuristicResult result;
  result.actions = legal;
  result.scores.reserve(legal.size());

  for (ActionId a : legal) {
    auto tok = rules.do_action_deterministic(masked_state, a);
    double s_val = eval_position<NPlayers>(s.persistent.data(), player);
    rules.undo_action(masked_state, tok);
    result.scores.push_back(s_val);
  }
  return result;
}

template class SplendorRuleHeuristic<2>;
template class SplendorRuleHeuristic<3>;
template class SplendorRuleHeuristic<4>;

}  // namespace board_ai::splendor
