#pragma once

// EvaluatorPool — owns one BatchedPolicyValueEvaluator per distinct
// `model_path` referenced during a selfplay/arena/eval step. Sampling workers
// route leaf eval requests by model_path; the pool ensures each independent
// network has its own queue + inference thread, and that switching steps
// (selfplay → arena → eval) reloads only the evaluators that actually changed.
//
// Lifecycle: caller (ConcurrentSelfplayRunner / arena_runner / eval_runner)
// constructs a pool at step entry, calls `get_or_create(...)` for each model
// it'll need, and destroys the pool at step exit. Destruction joins all
// inference threads (BatchedPolicyValueEvaluator's destructor handles this).
//
// Thread-safety: `get_or_create` is mutex-protected. Once an evaluator is
// returned, callers may use it concurrently (BatchedPolicyValueEvaluator
// itself is thread-safe — multiple sampling workers calling evaluate() in
// parallel is the intended use).

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

#include "batched_policy_value_evaluator.h"

namespace board_ai::infer {

class EvaluatorPool {
 public:
  EvaluatorPool() = default;
  ~EvaluatorPool() = default;

  EvaluatorPool(const EvaluatorPool&) = delete;
  EvaluatorPool& operator=(const EvaluatorPool&) = delete;

  // Returns a pointer to the BatchedPolicyValueEvaluator for `model_path`,
  // creating one on first access. The pool retains ownership; pointer is
  // valid until pool destruction.
  //
  // `encoder` and `cfg` are used only on first creation; subsequent calls
  // with the same model_path return the existing evaluator (encoder/cfg
  // ignored — this matches OnnxPolicyValueEvaluator's session-cache
  // semantics where a model is loaded once per process).
  //
  // Throws if model_path is empty or if BatchedPolicyValueEvaluator
  // construction fails.
  search::IPolicyValueEvaluator* get_or_create(
      const std::string& model_path,
      const IFeatureEncoder* encoder,
      const BatchedEvaluatorConfig& cfg);

  // For diagnostics / profile_metrics.
  std::size_t size() const;

 private:
  mutable std::mutex mu_;
  std::unordered_map<std::string, std::unique_ptr<BatchedPolicyValueEvaluator>> by_path_;
};

}  // namespace board_ai::infer
