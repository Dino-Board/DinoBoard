#include "evaluator_pool.h"

#include <stdexcept>

namespace board_ai::infer {

search::IPolicyValueEvaluator* EvaluatorPool::get_or_create(
    const std::string& model_path,
    const IFeatureEncoder* encoder,
    const BatchedEvaluatorConfig& cfg) {
  if (model_path.empty()) {
    throw std::runtime_error(
        "EvaluatorPool::get_or_create: model_path must not be empty");
  }
  std::lock_guard<std::mutex> lk(mu_);
  auto it = by_path_.find(model_path);
  if (it != by_path_.end()) {
    return it->second.get();
  }
  auto ev = std::make_unique<BatchedPolicyValueEvaluator>(
      model_path, encoder, cfg);
  auto* raw = ev.get();
  by_path_.emplace(model_path, std::move(ev));
  return raw;
}

std::size_t EvaluatorPool::size() const {
  std::lock_guard<std::mutex> lk(mu_);
  return by_path_.size();
}

}  // namespace board_ai::infer
