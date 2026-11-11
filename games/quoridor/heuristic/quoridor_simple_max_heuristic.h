#pragma once

#include <string>

#include "../../../engine/core/heuristic.h"

namespace board_ai::quoridor {

// Simple-max heuristic: score every legal action by (d_opp - d_me) on the
// resulting position, no probable-wall mask, no walls-low penalty. Ties at
// the argmax are broken uniformly at random by the caller's
// `sample_heuristic_index` (T≈0 path) — every action that achieves the top
// score has equal probability of being picked.
class QuoridorSimpleMaxHeuristic final : public IHeuristic {
 public:
  explicit QuoridorSimpleMaxHeuristic(std::string instance_name = "simple_max")
      : name_(std::move(instance_name)) {}

  std::string name() const override { return name_; }

  HeuristicResult pick(IGameState& masked_state,
                       const IBeliefTracker* tracker,
                       const IGameRules& rules,
                       std::uint64_t rng_seed) const override;

 private:
  std::string name_;
};

}  // namespace board_ai::quoridor
