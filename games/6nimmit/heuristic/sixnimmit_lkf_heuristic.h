#pragma once

#include <string>

#include "../../../engine/core/heuristic.h"

namespace board_ai::sixnimmt {

// 2P-only port of games/6nimmit/参考/lkf_zhc_cyb2.cpp (CYB probability
// distribution + 2-3 ply minimax with chance node + Monte Carlo
// rollout estimate). For NPlayers != 2 the algorithm degenerates and
// pick() falls back to the rule_v1 strategy via internal delegation.
template <int NPlayers>
class SixNimmtLkfHeuristic final : public IHeuristic {
 public:
  explicit SixNimmtLkfHeuristic(std::string instance_name = "lkf_v1")
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
