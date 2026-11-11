#include "liarsdice_rule_heuristic.h"

#include <algorithm>
#include <array>
#include <cmath>

#include "../liarsdice_state.h"
#include "../liarsdice_rules.h"

namespace board_ai::liarsdice {

namespace {

inline double normal_cdf(double z) {
  return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
}

template <int NPlayers>
HeuristicResult heuristic_liarsdice_impl(
    IGameState& state, const IGameRules& rules) {
  using Cfg = LiarsdiceConfig<NPlayers>;
  const auto& s = checked_cast<LiarsdiceState<NPlayers>>(state);
  const auto& d = s.data;

  const int me = d.current_player_;
  std::array<int, kFaceCount + 1> my_count{};
  for (int f = 0; f < kFaceCount; ++f) {
    const int v = d.dice_count[static_cast<size_t>(me)][static_cast<size_t>(f)];
    my_count[static_cast<size_t>(f + 1)] = (v > 0) ? v : 0;
  }
  const int my_wild = my_count[1];

  const int hidden_dice = (NPlayers - 1) * kDicePerPlayer;
  const double p_match = 1.0 / 3.0;
  const double mu = hidden_dice * p_match;
  const double var = hidden_dice * p_match * (1.0 - p_match);
  const double sigma = std::sqrt(std::max(var, 1e-9));

  auto p_safe = [&](int X, int Y) -> double {
    const int own = my_wild + my_count[static_cast<size_t>(Y)];
    const int needed = X - own;
    if (needed <= 0) return 1.0;
    if (needed > hidden_dice) return 0.0;
    const double z = (static_cast<double>(needed) - 0.5 - mu) / sigma;
    return std::clamp(1.0 - normal_cdf(z), 0.0, 1.0);
  };

  auto legal = rules.legal_actions(state);
  HeuristicResult result;
  result.actions = legal;
  result.scores.assign(legal.size(), 0.0);

  const int cur_count = d.current_bid_count;
  const int cur_face = d.current_bid_face;
  const bool first_move = (cur_count == 0);

  for (size_t i = 0; i < legal.size(); ++i) {
    const ActionId a = legal[i];
    if (is_challenge(a, Cfg::kTotalDice)) {
      const double safe = p_safe(cur_count, cur_face);
      result.scores[i] = std::clamp(1.0 - safe, 0.0, 1.0);
      continue;
    }
    const int X = bid_count_from_action(a);
    const int Y = bid_face_from_action(a);
    const double safe = p_safe(X, Y);
    double s_raw = safe;
    if (!first_move) {
      const int x_jump = X - cur_count;
      if (x_jump > 1) s_raw *= std::pow(0.7, x_jump - 1);
    }
    result.scores[i] = std::max(s_raw, 1e-6);
  }
  return result;
}

}  // namespace

template <int NPlayers>
HeuristicResult LiarsdiceRuleHeuristic<NPlayers>::pick(
    IGameState& masked_state,
    const IBeliefTracker* /*tracker*/,
    const IGameRules& rules,
    std::uint64_t /*rng_seed*/) const {
  return heuristic_liarsdice_impl<NPlayers>(masked_state, rules);
}

template class LiarsdiceRuleHeuristic<2>;
template class LiarsdiceRuleHeuristic<3>;
template class LiarsdiceRuleHeuristic<4>;

}  // namespace board_ai::liarsdice
