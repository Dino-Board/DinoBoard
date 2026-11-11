#include "heuristic_registry.h"

#include <stdexcept>
#include <utility>

namespace board_ai {

void HeuristicRegistry::register_named(const std::string& name,
                                       std::unique_ptr<IHeuristic> h) {
  if (!h) {
    throw std::runtime_error(
        "HeuristicRegistry::register_named: null IHeuristic for '" + name + "'");
  }
  if (by_name_.count(name)) {
    throw std::runtime_error(
        "HeuristicRegistry::register_named: duplicate name '" + name + "'");
  }
  by_name_.emplace(name, std::move(h));
}

const IHeuristic& HeuristicRegistry::get(const std::string& name) const {
  auto it = by_name_.find(name);
  if (it == by_name_.end()) {
    throw std::runtime_error(
        "HeuristicRegistry::get: no heuristic registered as '" + name + "'");
  }
  return *it->second;
}

bool HeuristicRegistry::has(const std::string& name) const {
  return by_name_.count(name) > 0;
}

std::vector<std::string> HeuristicRegistry::names() const {
  std::vector<std::string> out;
  out.reserve(by_name_.size());
  for (const auto& [n, _] : by_name_) out.push_back(n);
  return out;
}

}  // namespace board_ai
