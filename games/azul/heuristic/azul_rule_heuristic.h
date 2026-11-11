#pragma once

#include <string>

#include "../../../engine/core/heuristic.h"

namespace board_ai::azul {

// Hand-scored A-flavor rule heuristic for Azul. Reads only public state
// (factories / center / players[*]) — Azul has no hidden info per
// perspective, so tracker is unused. Templated on NPlayers.
template <int NPlayers>
class AzulRuleHeuristic final : public IHeuristic {
 public:
  explicit AzulRuleHeuristic(std::string instance_name = "rule_v1")
      : name_(std::move(instance_name)) {}

  std::string name() const override { return name_; }

  HeuristicResult pick(IGameState& masked_state,
                       const IBeliefTracker* tracker,
                       const IGameRules& rules,
                       std::uint64_t rng_seed) const override;

 private:
  std::string name_;
};

}  // namespace board_ai::azul
