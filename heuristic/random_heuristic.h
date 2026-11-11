#pragma once

#include <string>

#include "../engine/core/heuristic.h"

namespace board_ai::heuristic {

// Uniform-over-legal-actions heuristic. Replaces the inline random pickers
// previously embedded in tictactoe_register.cpp / coup_register.cpp.
//
// `name()` defaults to "random" but can be overridden at construction so a
// game can register the same impl under multiple keys (e.g. "rule" for
// tictactoe to keep configs declarative).
class RandomHeuristic final : public IHeuristic {
 public:
  explicit RandomHeuristic(std::string instance_name = "random")
      : name_(std::move(instance_name)) {}

  std::string name() const override { return name_; }

  HeuristicResult pick(IGameState& masked_state,
                       const IBeliefTracker* tracker,
                       const IGameRules& rules,
                       std::uint64_t rng_seed) const override;

 private:
  std::string name_;
};

}  // namespace board_ai::heuristic
