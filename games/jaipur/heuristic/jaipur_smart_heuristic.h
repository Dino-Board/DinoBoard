#pragma once
#include <string>
#include "../../../engine/core/heuristic.h"
#include "../../../heuristic/mcts_pv_heuristic.h"
#include "jaipur_rule_heuristic.h"

namespace board_ai::jaipur {

// Composite heuristic: uses MCTS (rule_mcts) for the first decision of a turn
// where deep search matters (sell/take/trade choice), but falls back to the
// fast rule heuristic for the later atoms of a trade, where a full search per
// atom is slow and the best trade below the prefix is good enough.
class JaipurSmartHeuristic final : public IHeuristic {
 public:
  JaipurSmartHeuristic(std::string name,
                       const heuristic::MctsPVHeuristic* mcts_heuristic)
      : name_(std::move(name)), mcts_(mcts_heuristic), rule_("rule_fast") {}

  std::string name() const override { return name_; }

  HeuristicResult pick(IGameState& masked_state,
                       const IBeliefTracker* tracker,
                       const IGameRules& rules,
                       std::uint64_t rng_seed) const override;

 private:
  std::string name_;
  const heuristic::MctsPVHeuristic* mcts_;
  mutable JaipurRuleHeuristic rule_;
};

}  // namespace board_ai::jaipur
