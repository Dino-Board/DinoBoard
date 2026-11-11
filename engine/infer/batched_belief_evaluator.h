#pragma once

// BatchedBeliefEvaluator — leaf-batching wrapper for IBeliefEvaluator,
// mirrors BatchedPolicyValueEvaluator's design.
//
// Belief net is single-input single-output (features → logits), simpler
// than PV. Same async queue + cv + 1 inference thread. Used by sampling
// threads at MCTS root via IBeliefTracker::prepare_for_root, where many
// concurrent threads each request a single belief forward — batched here
// to amortize CPU SIMD or GPU launch overhead, just like PV.
//
// Owned by EvaluatorPool (or constructed directly). Phase 2 of
// PARALLEL_MCTS_VLOSS_PLAN.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../core/belief_evaluator.h"
#include "batched_policy_value_evaluator.h"  // reuses BatchedEvaluatorConfig

namespace board_ai::infer {

class BatchedBeliefEvaluator final : public IBeliefEvaluator {
 public:
  BatchedBeliefEvaluator(
      std::string model_path,
      BatchedEvaluatorConfig cfg = {});
  ~BatchedBeliefEvaluator();

  BatchedBeliefEvaluator(const BatchedBeliefEvaluator&) = delete;
  BatchedBeliefEvaluator& operator=(const BatchedBeliefEvaluator&) = delete;

  bool is_ready() const { return ready_; }
  const std::string& last_error() const { return last_error_; }
  const std::string& model_path() const { return model_path_; }

  bool evaluate(
      const std::vector<float>& features,
      std::vector<float>* logits) const override;

 private:
  struct EvalResult {
    std::vector<float> logits;
    std::exception_ptr error;
  };

  struct PendingRequest {
    std::vector<float> features;
    std::promise<EvalResult> promise;
    std::chrono::steady_clock::time_point enqueued_at;
  };

  void inference_loop();
  void run_batch(std::vector<std::unique_ptr<PendingRequest>>& requests);

  std::string model_path_;
  BatchedEvaluatorConfig cfg_{};
  bool ready_ = false;
  std::string last_error_{};

  mutable std::mutex mu_;
  mutable std::condition_variable cv_;
  std::deque<std::unique_ptr<PendingRequest>> queue_;
  std::atomic<bool> shutdown_{false};
  std::thread inference_thread_;

#if defined(BOARD_AI_WITH_ONNX) && BOARD_AI_WITH_ONNX
  struct Impl;
  Impl* impl_ = nullptr;
#endif
};

}  // namespace board_ai::infer
