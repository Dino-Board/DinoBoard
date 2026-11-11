#include "carcassonne_heuristic_pv_evaluator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "../carcassonne_rules.h"
#include "../carcassonne_state.h"

namespace board_ai::carcassonne {

CarcassonneHeuristicPVEvaluator::CarcassonneHeuristicPVEvaluator(
    CarcassonneHeuristicProfile profile)
    : profile_(profile), rule_("rule", profile) {}

bool CarcassonneHeuristicPVEvaluator::evaluate(
    const IGameState& masked_state,
    int perspective_player,
    const IBeliefTracker* tracker,
    const std::vector<ActionId>& legal_actions,
    std::vector<float>* priors,
    std::vector<float>* values) const {
  if (legal_actions.empty()) return false;
  if (priors == nullptr || values == nullptr) {
    throw std::invalid_argument("carcassonne pv evaluator output is null");
  }
  if (perspective_player < 0 || perspective_player >= kPlayers) {
    throw std::invalid_argument(
        "carcassonne pv evaluator perspective out of range");
  }
  const auto& s = checked_cast<CarcassonneState>(masked_state);
  const auto pv = carcassonne_profile_pv_settings(profile_);

  // ---- Prior: softmax of rule heuristic scores ----
  auto clone = masked_state.clone_state();
  const CarcassonneRules rules;
  const HeuristicResult hr = rule_.pick(*clone, tracker, rules, 1ULL);
  if (hr.actions.size() != hr.scores.size()) {
    throw std::runtime_error(
        "carcassonne pv evaluator heuristic action/score size mismatch");
  }

  std::unordered_map<ActionId, double> score_by_action;
  score_by_action.reserve(hr.actions.size());
  for (size_t i = 0; i < hr.actions.size(); ++i) {
    score_by_action.emplace(hr.actions[i], hr.scores[i]);
  }

  std::vector<double> raw(legal_actions.size(), 0.0);
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    const auto it = score_by_action.find(legal_actions[i]);
    if (it == score_by_action.end()) {
      throw std::runtime_error(
          "carcassonne pv evaluator missing legal action in heuristic scores");
    }
    raw[i] = it->second;
  }

  const double mx = *std::max_element(raw.begin(), raw.end());
  double sum_exp = 0.0;
  priors->assign(legal_actions.size(), 0.0f);
  for (size_t i = 0; i < raw.size(); ++i) {
    const double e = std::exp((raw[i] - mx) * pv.prior_temperature);
    if (!std::isfinite(e)) {
      throw std::runtime_error(
          "carcassonne pv evaluator produced non-finite prior");
    }
    (*priors)[i] = static_cast<float>(e);
    sum_exp += e;
  }
  if (!std::isfinite(sum_exp) || sum_exp <= 0.0) {
    throw std::runtime_error(
        "carcassonne pv evaluator produced zero prior mass");
  }
  for (float& p : *priors) {
    p = static_cast<float>(static_cast<double>(p) / sum_exp);
  }

  // ---- Value: 100% final-scoring projection ----
  // For non-terminal states, run the engine's apply_final_scoring on a
  // clone of the masked state and read the projected scores (in the meeple
  // stage, after scoring what the tile just placed completed). For
  // terminal states, use the live scoreboard (already final).
  std::array<int, kPlayers> projected;
  if (s.terminal) {
    projected = s.scores;
  } else {
    projected = CarcassonneRules::compute_final_scores(s);
  }

  values->assign(static_cast<size_t>(kPlayers), 0.0f);
  for (int p = 0; p < kPlayers; ++p) {
    const int opp = 1 - p;
    const double margin =
        static_cast<double>(projected[static_cast<size_t>(p)]) -
        static_cast<double>(projected[static_cast<size_t>(opp)]);
    (*values)[static_cast<size_t>(p)] =
        static_cast<float>(std::tanh(margin / pv.value_margin_scale));
  }
  return true;
}

}  // namespace board_ai::carcassonne
