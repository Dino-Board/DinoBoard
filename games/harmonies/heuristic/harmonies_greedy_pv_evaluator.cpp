#include "harmonies_greedy_pv_evaluator.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "harmonies_greedy_heuristic.h"
#include "../harmonies_rules.h"
#include "../harmonies_state.h"

namespace board_ai::harmonies {

template <int NPlayers, int Side>
bool HarmoniesGreedyPVEvaluator<NPlayers, Side>::evaluate(
    const IGameState& masked_state,
    int perspective_player,
    const IBeliefTracker* /*tracker*/,
    const std::vector<ActionId>& legal_actions,
    std::vector<float>* priors,
    std::vector<float>* values) const {
  if (legal_actions.empty()) return false;

  auto state_clone = masked_state.clone_state();
  HarmoniesGreedyHeuristic<NPlayers, Side> greedy;
  HeuristicResult hr = greedy.pick(*state_clone, nullptr, *rules_, 1ULL);

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
  double max_score = *std::max_element(raw.begin(), raw.end());
  double sum_exp = 0.0;
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    double e = std::exp(raw[i] - max_score);
    (*priors)[i] = static_cast<float>(e);
    sum_exp += e;
  }
  if (sum_exp > 0.0) {
    for (size_t i = 0; i < legal_actions.size(); ++i) {
      (*priors)[i] = static_cast<float>((*priors)[i] / sum_exp);
    }
  } else {
    float u = 1.0f / static_cast<float>(legal_actions.size());
    for (size_t i = 0; i < legal_actions.size(); ++i) (*priors)[i] = u;
  }

  const auto& s = checked_cast<HarmoniesState<NPlayers, Side>>(masked_state);
  values->assign(static_cast<size_t>(NPlayers), 0.0f);
  HarmoniesRules<NPlayers, Side>::recompute_landscape_scores(
      const_cast<HarmoniesState<NPlayers, Side>&>(s));

  for (int p = 0; p < NPlayers; ++p) {
    int my_total = s.animal_scores[p] + s.landscape_scores[p];
    int best_opp = 0;
    for (int o = 0; o < NPlayers; ++o) {
      if (o == p) continue;
      int opp_total = s.animal_scores[o] + s.landscape_scores[o];
      if (opp_total > best_opp) best_opp = opp_total;
    }
    (*values)[static_cast<size_t>(p)] =
        std::tanh(static_cast<float>(my_total - best_opp) / 30.0f);
  }
  return true;
}

#define HARMONIES_INST_HarmoniesGreedyPVEvaluator(N, S) template class HarmoniesGreedyPVEvaluator<N, S>;
HARMONIES_FOR_EACH_VARIANT(HARMONIES_INST_HarmoniesGreedyPVEvaluator)
#undef HARMONIES_INST_HarmoniesGreedyPVEvaluator

}  // namespace board_ai::harmonies
