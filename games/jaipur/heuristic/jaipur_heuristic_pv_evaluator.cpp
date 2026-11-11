#include "jaipur_heuristic_pv_evaluator.h"

#include <algorithm>
#include <cmath>

#include "../jaipur_state.h"
#include "../jaipur_rules.h"

namespace board_ai::jaipur {

// Expected bonus token value for a tier as `observer` knows it (see
// jaipur_bonus_expected): the values are face-down, so the observer estimates
// the next pile token and each opponent token by the mean of the residual
// pool it knows — never by the pile's real (hidden) composition.
double JaipurHeuristicPVEvaluator::bonus_expected(int tier, const JaipurData& d,
                                                  int observer) {
  return jaipur_bonus_expected(tier, d, observer);
}

bool JaipurHeuristicPVEvaluator::evaluate(
    const IGameState& masked_state,
    int perspective_player,
    const IBeliefTracker* tracker,
    const std::vector<ActionId>& legal_actions,
    std::vector<float>* priors,
    std::vector<float>* values) const {
  if (legal_actions.empty()) return false;

  // --- Policy prior: softmax over the rule heuristic's per-action scores. ---
  auto clone = masked_state.clone_state();
  HeuristicResult hr = rule_.pick(*clone, tracker, JaipurRules(), 1ULL);
  priors->assign(legal_actions.size(), 0.0f);
  std::vector<double> raw(legal_actions.size(), -1e9);
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    for (size_t j = 0; j < hr.actions.size(); ++j) {
      if (hr.actions[j] == legal_actions[i]) { raw[i] = hr.scores[j]; break; }
    }
  }
  double mx = *std::max_element(raw.begin(), raw.end());
  double sum_exp = 0.0;
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    double e = std::exp((raw[i] - mx) * 0.5);  // temperature 2 → softer prior
    (*priors)[i] = static_cast<float>(e);
    sum_exp += e;
  }
  if (sum_exp > 0.0) {
    for (size_t i = 0; i < legal_actions.size(); ++i)
      (*priors)[i] = static_cast<float>((*priors)[i] / sum_exp);
  } else {
    float u = 1.0f / static_cast<float>(legal_actions.size());
    for (size_t i = 0; i < legal_actions.size(); ++i) (*priors)[i] = u;
  }

  // --- Value: tanh(my_expected_score - opp_expected_score). ---
  // Expected final score includes:
  //   (a) rupees already banked (public),
  //   (b) bonus_total — owner-truth for self, count*pile_mean estimate for opp,
  //   (c) HAND potential: each good type g contributes the expected sell value
  //       (top tokens of pile g, plus the bonus tier if hand[g] >= 3/4/5),
  //   (d) CAMELS in stable: terminal +5 each (camel-bonus +5 winner-take-all,
  //       which we approximate as 5 to whichever side has more).
  // Without (c) and (d) the V signal sees "take into hand" as 0 reward, so
  // MCTS short-cuts to insta-selling tiny stacks instead of stockpiling.
  const auto& s = static_cast<const JaipurState&>(masked_state);
  const auto& d = s.data;
  values->assign(static_cast<size_t>(kPlayers), 0.0f);

  // (b) opponent bonus estimate
  double exp_bonus[kPlayers] = {0.0, 0.0};
  for (int p = 0; p < kPlayers; ++p) {
    for (int tier = 0; tier < 3; ++tier)
      exp_bonus[p] += d.bonus_count[p][tier] *
                      bonus_expected(tier, d, perspective_player);
  }

  // (c) hand potential per player. For good g with hand count n>=2, take the
  // top n tokens off the pile (face-up first); n>=3/4/5 also trigger a bonus
  // token of expected value. Solo cards (n==1) score full top-token (the
  // player can sell later, but with risk of pile depletion — discount lightly
  // by 0.7 to bias toward stockpiling).
  auto hand_value = [&](int p) -> double {
    double v = 0.0;
    for (int g = 0; g < kGoodTypes; ++g) {
      const int n = d.hand_count[p][g];
      if (n <= 0) continue;
      const auto& pile = d.value_tokens[g];
      // tokens are pushed cheapest-first then popped from back; back() is the
      // current face-up (highest remaining). For n hand cards we'd consume
      // the top n; sum those.
      const int take = std::min(n, static_cast<int>(pile.size()));
      for (int i = 0; i < take; ++i)
        v += pile[pile.size() - 1 - i];
      // discount under-stack hand (n==1) — selling alone is weak,
      // stockpiling toward 3+ is the real win
      if (n == 1) v *= 0.6;
      // bonus tier on actual sell of n>=3
      if (n >= 5)      v += bonus_expected(2, d, perspective_player);
      else if (n >= 4) v += bonus_expected(1, d, perspective_player);
      else if (n >= 3) v += bonus_expected(0, d, perspective_player);
    }
    return v;
  };

  // (d) camels: terminal +5 each guaranteed; +5 camel-bonus to side w/ more
  double camel_value[kPlayers];
  for (int p = 0; p < kPlayers; ++p)
    camel_value[p] = 5.0 * d.camel_count[p];
  if (d.camel_bonus_token) {
    if (d.camel_count[0] > d.camel_count[1])      camel_value[0] += 5.0;
    else if (d.camel_count[1] > d.camel_count[0]) camel_value[1] += 5.0;
  }

  for (int p = 0; p < kPlayers; ++p) {
    double my_bonus = (p == perspective_player) ? d.bonus_total[p]
                                                : exp_bonus[p];
    double my_total = d.rupees[p] + my_bonus + hand_value(p) + camel_value[p];
    double opp_best = -1e9;
    for (int o = 0; o < kPlayers; ++o) {
      if (o == p) continue;
      double opp = d.rupees[o] + exp_bonus[o] + hand_value(o) + camel_value[o];
      if (opp > opp_best) opp_best = opp;
    }
    opp_best = (opp_best < -1e8) ? 0.0 : opp_best;
    // Scale: typical winning gap is 5–15 pts; /20 keeps tanh in its useful
    // linear region for normal play.
    (*values)[static_cast<size_t>(p)] =
        std::tanh(static_cast<float>((my_total - opp_best) / 20.0f));
  }
  return true;
}

}  // namespace board_ai::jaipur
