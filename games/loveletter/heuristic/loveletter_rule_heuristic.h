#pragma once

#include <string>

#include "../../../engine/core/heuristic.h"

namespace board_ai::loveletter {

// Hand-scored A-flavor rule heuristic for Love Letter. Scores are per-action
// (no lookahead — Love Letter decisions are local). Tracker is accepted by
// signature but currently ignored; future versions can fold in claim history.
template <int NPlayers>
class LoveLetterRuleHeuristic final : public IHeuristic {
 public:
  explicit LoveLetterRuleHeuristic(std::string instance_name = "rule_v1")
      : name_(std::move(instance_name)) {}

  std::string name() const override { return name_; }

  HeuristicResult pick(IGameState& masked_state,
                       const IBeliefTracker* tracker,
                       const IGameRules& rules,
                       std::uint64_t rng_seed) const override;

 private:
  std::string name_;
};

}  // namespace board_ai::loveletter
