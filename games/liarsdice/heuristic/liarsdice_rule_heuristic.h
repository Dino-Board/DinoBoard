#pragma once

#include <string>

#include "../../../engine/core/heuristic.h"

namespace board_ai::liarsdice {

// Rule heuristic — estimates P(actual >= X for face Y) from own dice
// + uniform prior on opponents, then scores each legal action. Tracker
// is accepted by signature but not yet consumed (future work can fold
// in claim history).
template <int NPlayers>
class LiarsdiceRuleHeuristic final : public IHeuristic {
 public:
  explicit LiarsdiceRuleHeuristic(std::string instance_name = "rule_v1")
      : name_(std::move(instance_name)) {}

  std::string name() const override { return name_; }

  HeuristicResult pick(IGameState& masked_state,
                       const IBeliefTracker* tracker,
                       const IGameRules& rules,
                       std::uint64_t rng_seed) const override;

 private:
  std::string name_;
};

}  // namespace board_ai::liarsdice
