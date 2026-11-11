#pragma once

#include <string>

#include "../../../engine/core/heuristic.h"

namespace board_ai::splendor {

// Hand-scored A-flavor rule heuristic for Splendor. Reads only public +
// own-private state (gems, bonuses, visible reserved cards, board tableau,
// nobles). Templated on NPlayers; explicit instantiations for 2/3/4 in
// the .cpp.
template <int NPlayers>
class SplendorRuleHeuristic final : public IHeuristic {
 public:
  explicit SplendorRuleHeuristic(std::string instance_name = "rule_v1")
      : name_(std::move(instance_name)) {}

  std::string name() const override { return name_; }

  HeuristicResult pick(IGameState& masked_state,
                       const IBeliefTracker* tracker,
                       const IGameRules& rules,
                       std::uint64_t rng_seed) const override;

 private:
  std::string name_;
};

}  // namespace board_ai::splendor
