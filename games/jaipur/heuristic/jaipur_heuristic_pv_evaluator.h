#pragma once
#include <vector>

#include "../../../engine/core/game_interfaces.h"
#include "../../../engine/search/net_mcts.h"
#include "jaipur_rule_heuristic.h"

namespace board_ai::jaipur {

// Heuristic policy/value evaluator for Jaipur — lets MCTS run WITHOUT a
// trained ONNX model by using the rule heuristic for both the policy prior
// and the value estimate.
//
//   P(action)  = softmax over the rule heuristic's per-action scores
//   V(player)  = tanh( (my_expected_score - opp_expected_score) / scale )
//
// Expected score = public rupees + bonus_count[tier] * expected_bonus_value
// (the bonus token VALUES are face-down, so the opponent's contribution is
// estimated by the mean of the residual value pool the observer knows; the
// observer's own bonus_total is known truth). Camel token (5) is public and included in rupees already.
//
// Used by the "rule_mcts" heuristic (a MctsPVHeuristic wrapping this
// evaluator) so the web can run a strong, tracker-sampling MCTS AI before
// any model is trained.
class JaipurHeuristicPVEvaluator final : public search::IPolicyValueEvaluator {
 public:
  explicit JaipurHeuristicPVEvaluator(std::string name = "jaipur_rule_pv")
      : name_(std::move(name)), rule_("rule") {}

  std::string name() const { return name_; }

  bool evaluate(const IGameState& masked_state,
                int perspective_player,
                const IBeliefTracker* tracker,
                const std::vector<ActionId>& legal_actions,
                std::vector<float>* priors,
                std::vector<float>* values) const override;

 private:
  std::string name_;
  // Reusable rule heuristic instance for prior scoring.
  mutable JaipurRuleHeuristic rule_;

  // Expected bonus value for a tier from `observer`'s knowledge (mean of
  // the residual pool, or the initial-set mean if the pile is exhausted).
  static double bonus_expected(int tier, const JaipurData& d, int observer);
};

}  // namespace board_ai::jaipur
