#include "jaipur_rule_heuristic.h"
#include "../jaipur_state.h"
#include "../jaipur_rules.h"
#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace board_ai::jaipur {

static int hand_total(const JaipurData& d, int p) {
  return std::accumulate(d.hand_count[p].begin(), d.hand_count[p].end(), 0);
}
static double top_token(const JaipurData& d, int g) {
  return d.value_tokens[g].empty() ? 0.0 : d.value_tokens[g].back();
}

// Expected value of one face-down bonus token of `tier` as `observer` knows
// it: the mean of the tier's full value multiset minus the tokens the
// observer itself drew (its own owner-only bonus_taken row). The pile's
// composition and the opponent's draws are hidden — never read them here.
// Initial-set mean if the pile is exhausted (or nothing is left to estimate).
double jaipur_bonus_expected(int tier, const JaipurData& d, int observer) {
  double init_sum = 0.0, sum = 0.0;
  int init_n = 0, n = 0;
  for (int v = 0; v < kBonusValues; ++v) {
    const int left = kBonusInitCount[tier][v] - d.bonus_taken[observer][tier][v];
    init_sum += kBonusInitCount[tier][v] * kBonusValue[tier][v];
    init_n += kBonusInitCount[tier][v];
    sum += left * kBonusValue[tier][v];
    n += left;
  }
  if (d.bonus_pile_size[tier] > 0 && n > 0) return sum / n;
  return init_sum / init_n;
}

static double bonus_expected(int tier, const JaipurData& d) {
  return jaipur_bonus_expected(tier, d, d.current_player);
}

double JaipurRuleHeuristic::score_sell(int g, int k, const JaipurData& d) const {
  double base = 0.0;
  const int n = static_cast<int>(d.value_tokens[g].size());
  for (int i = 0; i < std::min(k, n); ++i) base += d.value_tokens[g][n - 1 - i];
  if (k >= 5) base += bonus_expected(2, d);
  else if (k == 4) base += bonus_expected(1, d);
  else if (k == 3) base += bonus_expected(0, d);
  if (g >= 3) base *= 0.8;   // cheap goods worth less
  if (k < 3) base *= 0.6;    // small sales forgo bonus
  return base;
}

double JaipurRuleHeuristic::score_take(int g, const JaipurData& d) const {
  if (d.market_count[g] <= 0) return -1e9;
  double base = (6.0 - g);  // prefer expensive goods
  base += d.hand_count[d.current_player][g] * 1.0;  // synergy
  base += top_token(d, g) * 0.15;
  return base;
}

double JaipurRuleHeuristic::score_camels(const JaipurData& d) const {
  const int p = d.current_player;
  double sc = 2.0;
  if (d.camel_count[p] == 0) sc += 2.0;
  if (hand_total(d, p) <= 3) sc += 1.0;
  if (hand_total(d, 1 - p) >= kHandSize - 1) sc += 1.5;
  if (d.market_count[kCamel] >= 3 && d.camel_count[p] >= 3) sc -= 1.0;
  return sc;
}

// A trade is worth the take value of what it brings in minus what it pays:
// goods given are cheap when they are cheap types and break no set, camels
// cost a little, more if they hand the camel majority to the opponent.
double JaipurRuleHeuristic::score_trade(const JaipurTrade& t, const JaipurData& d) const {
  const int p = d.current_player;
  double gain = 0.0, cost = 0.0;
  for (int g = 0; g < kGoodTypes; ++g) {
    gain += t.take[g] * score_take(g, d);
    cost += t.give[g] * ((6.0 - g) * 0.6 + 0.3 * d.hand_count[p][g]);
  }
  cost += t.give[kCamel] * 1.0;
  const int opp_camels = d.camel_count[1 - p];
  if (d.camel_count[p] > opp_camels && d.camel_count[p] - t.give[kCamel] <= opp_camels) cost += 2.0;
  return 0.7 * (gain - cost);
}

HeuristicResult JaipurRuleHeuristic::pick(IGameState& masked_state,
                                           const IBeliefTracker* /*tracker*/,
                                           const IGameRules& rules,
                                           std::uint64_t /*rng_seed*/) const {
  auto legal = rules.legal_actions(masked_state);
  const auto& d = static_cast<const JaipurState&>(masked_state).data;
  HeuristicResult result;
  result.actions = legal;
  result.scores.assign(legal.size(), 0.0);

  // Trade atoms take the best score among the complete trades below them
  // (factored-actions skill: heuristics think in core options).
  constexpr double kUnset = -std::numeric_limits<double>::infinity();
  std::array<double, kActionSpace> trade_best;
  trade_best.fill(kUnset);
  for (const auto& t : jaipur_trades(d)) {
    const ActionId a = jaipur_trade_next_atom(d, t);
    trade_best[a] = std::max(trade_best[a], score_trade(t, d));
  }

  for (std::size_t i = 0; i < legal.size(); ++i) {
    const ActionId a = legal[i];
    double s;
    if (is_take_action(a)) s = score_take(a - kActTakeBase, d);
    else if (a == kActTakeCamels) s = score_camels(d);
    else if (is_sell_action(a)) s = score_sell(sell_type(a), sell_count(a), d);
    else {
      s = trade_best[a];
      if (s == kUnset)
        throw std::logic_error("JaipurRuleHeuristic: legal trade atom with no complete trade below it");
    }
    result.scores[i] = s;
  }
  return result;
}

}  // namespace board_ai::jaipur
