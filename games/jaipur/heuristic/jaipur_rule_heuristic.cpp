#include "jaipur_rule_heuristic.h"
#include "../jaipur_state.h"
#include "../jaipur_rules.h"
#include <algorithm>
#include <numeric>

namespace board_ai::jaipur {

static int hand_count(const JaipurData& d, int p, int g) { return d.hand_count[p][g]; }
static int hand_total(const JaipurData& d, int p) {
  return std::accumulate(d.hand_count[p].begin(), d.hand_count[p].end(), 0);
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

double JaipurRuleHeuristic::score_sell(int g, const JaipurData& d) const {
  int p = d.current_player;
  int count = hand_count(d, p, g);
  double base = 0.0;
  int n = static_cast<int>(d.value_tokens[g].size());
  for (int i = 0; i < std::min(count, n); ++i) {
    int idx = n - 1 - i;
    if (idx >= 0 && idx < n) base += d.value_tokens[g][idx];
  }
  if (count >= 5) base += bonus_expected(2, d);
  else if (count == 4) base += bonus_expected(1, d);
  else if (count == 3) base += bonus_expected(0, d);
  if (g >= 3) base *= 0.8;   // cheap goods worth less
  if (count < 3) base *= 0.6;  // small sales forgo bonus
  return base;
}

double JaipurRuleHeuristic::score_take(int g, const JaipurData& d) const {
  if (d.market_count[g] <= 0) return -1e9;
  double base = (6.0 - g);  // prefer expensive goods
  base += hand_count(d, d.current_player, g) * 1.0;  // synergy
  if (!d.value_tokens[g].empty()) base += d.value_tokens[g].back() * 0.15;
  return base;
}

// Feasibility of a take selection (mirrors rules' take_feasible).
static bool take_feasible(const JaipurData& d) {
  int p = d.current_player;
  int tc = std::accumulate(d.exchange_take_count.begin(),
                           d.exchange_take_count.end(), 0);
  if (tc < 2) return false;
  int take_bits = 0;
  for (int g = 0; g < kGoodTypes; ++g)
    if (d.exchange_take_count[g] > 0) take_bits |= (1 << g);
  int avail = 0;
  for (int g = 0; g < kGoodTypes; ++g)
    if (!(take_bits & (1 << g))) avail += d.hand_count[p][g];
  if (avail + d.camel_count[p] < tc) return false;
  int min_camels = std::max(0, tc - avail);
  if (hand_total(d, p) + min_camels > kHandSize) return false;
  return true;
}

HeuristicResult JaipurRuleHeuristic::pick(IGameState& masked_state,
                                           const IBeliefTracker* /*tracker*/,
                                           const IGameRules& rules,
                                           std::uint64_t rng_seed) const {
  (void)rng_seed;
  auto legal = rules.legal_actions(masked_state);
  const auto& d = static_cast<const JaipurState&>(masked_state).data;
  HeuristicResult result;
  result.actions = legal;
  result.scores.resize(legal.size(), 0.0);
  int p = d.current_player;
  int ph = d.exchange_phase;

  for (size_t i = 0; i < legal.size(); ++i) {
    ActionId a = legal[i];
    if (ph == kExchPhaseIdle) {
      if (a >= kActSellBase && a < kActSellBase + kGoodTypes) {
        result.scores[i] = score_sell(a - kActSellBase, d);
      } else if (a >= kActTakeBase && a < kActTakeBase + kGoodTypes) {
        result.scores[i] = score_take(a - kActTakeBase, d);
      } else if (a == kActTakeCamels) {
        double sc = 2.0;
        if (d.camel_count[p] == 0) sc += 2.0;
        if (hand_total(d, p) <= 3) sc += 1.0;
        int opp = 1 - p;
        if (hand_total(d, opp) >= kHandSize - 1) sc += 1.5;
        if (d.market_count[6] >= 3 && d.camel_count[p] >= 3) sc -= 1.0;
        result.scores[i] = sc;
      } else if (a == kActEnterExchange) {
        // Score by best feasible exchange's take value.
        double best = -1e9;
        // Quick scan: best pair of market goods (feasibility checked loosely).
        JaipurData trial = d;
        trial.exchange_take_count.fill(0);
        for (int g1 = 0; g1 < kGoodTypes; ++g1) {
          if (d.market_count[g1] <= 0) continue;
          for (int g2 = g1; g2 < kGoodTypes; ++g2) {
            if (g1 == g2 && d.market_count[g1] < 2) continue;
            if (d.market_count[g2] <= 0) continue;
            trial.exchange_take_count.fill(0);
            trial.exchange_take_count[g1] = (g1 == g2) ? 2 : 1;
            if (g1 != g2) trial.exchange_take_count[g2] = 1;
            if (!take_feasible(trial)) continue;
            double v = (6.0 - g1) + (6.0 - g2);
            if (v > best) best = v;
          }
        }
        result.scores[i] = (best > -1e8) ? best * 0.4 : -1.0;
      }
    } else if (ph == kExchPhaseMarket) {
      // mkt_done is always legal now (escape hatch for infeasible
      // selections — see jaipur_rules.cpp). Gate "commit now vs take more"
      // on FEASIBILITY of the current selection, not on legal membership.
      const bool feasible = take_feasible(d);
      if (a == kActExchMktDone) {
        result.scores[i] = feasible ? 100.0 : -1e9;
      } else if (a >= kActExchMktBase && a < kActExchMktBase + kGoodTypes) {
        int g = a - kActExchMktBase;
        result.scores[i] = feasible ? -1e9 : (10.0 + (6.0 - g) * 0.1);
      }
    } else if (ph == kExchPhaseCamel) {
      if (a == kActExchCamelBase + d.exchange_min_camels)
        result.scores[i] = 5.0;  // prefer fewest camels (min)
      else if (a >= kActExchCamelBase)
        result.scores[i] = -2.0;
    } else if (ph == kExchPhaseHand) {
      // Greedily fill toward (total - camel_count) given goods, cheapest first.
      int need = d.exchange_total - d.exchange_camel_count;
      int have = std::accumulate(d.exchange_give_count.begin(),
                                 d.exchange_give_count.end(), 0);
      if (a >= kActExchHandBase && a < kActExchHandBase + kGoodTypes) {
        int g = a - kActExchHandBase;
        double cheap = (6.0 - g) * 0.1;  // higher g (cheap) preferred
        result.scores[i] = (have < need ? 10.0 : -10.0) + cheap;
      }
    }
  }
  return result;
}

}  // namespace board_ai::jaipur
