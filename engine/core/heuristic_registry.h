#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "heuristic.h"

namespace board_ai {

// Per-game named pool of IHeuristic instances. Populated by each game's
// `<game>_register.cpp` and exposed via `GameBundle::heuristics`.
// `heuristic.json::named` lists the names that must be present; the
// two selectors (`eval_vs_heuristic`, `web`) in heuristic.json and the
// training-time `heuristic_guidance.name` in train.json reference names
// from this pool.
//
// Lookup by name throws on miss — config errors fail fast (no silent
// fallback). The framework's load-time validator
// (test_heuristic_registry.py) catches mismatches before training.
class HeuristicRegistry {
 public:
  HeuristicRegistry() = default;
  HeuristicRegistry(HeuristicRegistry&&) = default;
  HeuristicRegistry& operator=(HeuristicRegistry&&) = default;
  HeuristicRegistry(const HeuristicRegistry&) = delete;
  HeuristicRegistry& operator=(const HeuristicRegistry&) = delete;

  void register_named(const std::string& name, std::unique_ptr<IHeuristic> h);

  // Throws std::runtime_error on miss.
  const IHeuristic& get(const std::string& name) const;

  bool has(const std::string& name) const;

  std::vector<std::string> names() const;

  bool empty() const { return by_name_.empty(); }

 private:
  std::unordered_map<std::string, std::unique_ptr<IHeuristic>> by_name_;
};

}  // namespace board_ai
