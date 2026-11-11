#pragma once

#include "../../../engine/core/game_interfaces.h"
#include "../../../engine/search/net_mcts.h"
#include "carcassonne_rule_heuristic.h"

namespace board_ai::carcassonne {

// Policy/value evaluator used by the "rule_mcts" heuristic. It lets
// Carcassonne run MCTS before a trained ONNX model exists:
//   P(action) = softmax(rule heuristic score)
//   V(player) = tanh(projected_score_margin / scale)
class CarcassonneHeuristicPVEvaluator final
    : public search::IPolicyValueEvaluator {
 public:
  explicit CarcassonneHeuristicPVEvaluator(
      CarcassonneHeuristicProfile profile =
          CarcassonneHeuristicProfile::kSharpPvSoft);

  bool evaluate(const IGameState& masked_state,
                int perspective_player,
                const IBeliefTracker* tracker,
                const std::vector<ActionId>& legal_actions,
                std::vector<float>* priors,
                std::vector<float>* values) const override;

 private:
  CarcassonneHeuristicProfile profile_;
  mutable CarcassonneRuleHeuristic rule_;
};

}  // namespace board_ai::carcassonne
