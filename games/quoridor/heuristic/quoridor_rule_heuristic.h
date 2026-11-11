#pragma once

#include <memory>
#include <string>

#include "../../../engine/core/heuristic.h"
#include "../../../engine/search/net_mcts.h"

namespace board_ai::quoridor {

// Hand-written policy/value evaluator using gorisanson-style scoring
// (softmax over (d_opp - d_me) with probable-wall mask):
// prior = softmax(rule scores, T=1), value = tanh((d_opp - d_me)/8).
// Owned by GameBundle::heuristic_pv_evaluators; borrowed by an
// MctsPVHeuristic registered as `rule_mcts_v1`.
class QuoridorRulePVEvaluator final
    : public board_ai::search::IPolicyValueEvaluator {
 public:
  bool evaluate(
      const IGameState& masked_state,
      int perspective_player,
      const IBeliefTracker* tracker,
      const std::vector<ActionId>& legal_actions,
      std::vector<float>* priors,
      std::vector<float>* values) const override;

  void set_rules(const IGameRules* r) { rules_ = r; }

 private:
  const IGameRules* rules_ = nullptr;
};

}  // namespace board_ai::quoridor
