#pragma once

#include <string>

#include "../../../engine/core/heuristic.h"

namespace board_ai::sixnimmt {

// Rule heuristic — port of lyc_pbh_wzq.py's 6-tier strategy(). Tracker
// is accepted by signature but not yet consumed.
template <int NPlayers>
class SixNimmtRuleHeuristic final : public IHeuristic {
 public:
  explicit SixNimmtRuleHeuristic(std::string instance_name = "rule_v1")
      : name_(std::move(instance_name)) {}

  std::string name() const override { return name_; }

  HeuristicResult pick(IGameState& masked_state,
                       const IBeliefTracker* tracker,
                       const IGameRules& rules,
                       std::uint64_t rng_seed) const override;

 private:
  std::string name_;
};

}  // namespace board_ai::sixnimmt
