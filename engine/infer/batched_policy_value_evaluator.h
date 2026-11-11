#pragma once

// BatchedPolicyValueEvaluator — leaf-batching wrapper around the same
// ONNX session machinery used by OnnxPolicyValueEvaluator.
//
// Public contract is synchronous (IPolicyValueEvaluator::evaluate): a caller
// thread invokes evaluate() and blocks until result is filled. Internally the
// caller encodes into thread-local buffers, pushes borrowed-pointer requests
// to a queue and waits once for the whole call; `num_inference_threads`
// dedicated threads drain the queue, pack `batch_size` requests into a single
// ONNX Run() with shape {N, feature_dim}, and copy the raw outputs back.
// Variable-length token encodings are zero-padded to the batch's
// padded_token_count instead.
// The inference threads are the serial point of selfplay, so they do no
// per-request allocation, no softmax and only one wake per caller call.
//
// Trigger rule: the inference thread fires Run() when EITHER (a) queue has
// >= batch_size pending requests, OR (b) the oldest request has been waiting
// >= batch_timeout_us. Whichever comes first.
//
// Designed for the Phase 1 selfplay path (PARALLEL_MCTS_VLOSS_PLAN.md):
// many sampling worker threads call evaluate() concurrently, this evaluator
// amortizes their leaf evaluations into batched forward passes. All callers
// for the same model_path share one BatchedPolicyValueEvaluator instance via
// EvaluatorPool (Step 2 of the plan). web/API path retains the synchronous
// OnnxPolicyValueEvaluator unchanged.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../core/feature_encoder.h"
#include "../search/net_mcts.h"

namespace board_ai::infer {

struct BatchedEvaluatorConfig {
  // Inference-thread Run() trigger: fire when queue size hits this OR oldest
  // request waited >= batch_timeout_us, whichever first.
  std::size_t batch_size = 32;
  // Empirical: 6nimmit M5 4-core scan showed 10–50us timeout dominates
  // 2ms by 2–3x. Larger timeout creates 2ms latency spikes when queue
  // doesn't fill, which dominates per-leaf time on small networks. Keep
  // tiny by default; tune larger only if profiling shows >50% of batches
  // are timeout-triggered AND inference is the dominant cost.
  std::size_t batch_timeout_us = 50;
  // Hard cap on enqueued+pending requests; throws if exceeded (defensive).
  std::size_t max_queue_size = 256;

  // ONNX session knobs (same semantics as OnnxEvaluatorConfig).
  int intra_op_num_threads = 2;
  int inter_op_num_threads = 1;

  // Inference threads draining the shared queue, each running its own
  // batches against its own ORT session. >1 overlaps
  // one batch's result dispatch (a futex wake per caller — expensive under
  // WSL2/VMs) with the next batch's Run.
  int num_inference_threads = 1;

  // ONNX execution providers tried in order. Recognized values:
  //   "coreml" — Apple CoreMLEP (M-series Mac), no extra build needed
  //              (brew onnxruntime 1.25+ ships coreml_provider_factory.h)
  //   "cuda"   — NVIDIA CUDAEP, requires onnxruntime built with CUDA
  //              support (not available in brew default build)
  //   "tensorrt" — NVIDIA TensorRT EP (GPU onnxruntime build + TensorRT
  //              libs on LD_LIBRARY_PATH, see scripts/gpu_env.sh). Shape
  //              profile 1..batch_size; engines cached in <model dir>/trt_cache.
  //              List "cuda" after it for ops TensorRT rejects.
  //   "cpu"    — fallback (default ORT CPU EP); always available
  // Empty vector defaults to ["cpu"] (CPU-only, Phase 1 behavior).
  // Each entry is appended via the matching ORT API; later entries
  // serve as fallback for ops the earlier provider can't handle.
  std::vector<std::string> providers{};

  // GPU device ids for the "cuda" / "tensorrt" providers. Inference thread i
  // runs on devices[i % devices.size()], so num_inference_threads must be a
  // multiple of devices.size() (every GPU gets the same number of sessions).
  // Sessions on different GPUs run in parallel; on one GPU they serialize.
  // Empty means device 0.
  std::vector<int> devices{};
};

class BatchedPolicyValueEvaluator final : public search::IPolicyValueEvaluator {
 public:
  BatchedPolicyValueEvaluator(
      std::string model_path,
      const IFeatureEncoder* encoder,
      BatchedEvaluatorConfig cfg = {});
  ~BatchedPolicyValueEvaluator();

  BatchedPolicyValueEvaluator(const BatchedPolicyValueEvaluator&) = delete;
  BatchedPolicyValueEvaluator& operator=(const BatchedPolicyValueEvaluator&) = delete;

  bool is_ready() const { return ready_; }
  const std::string& last_error() const { return last_error_; }
  const std::string& model_path() const { return model_path_; }

  bool evaluate(
      const IGameState& masked_state,
      int perspective_player,
      const IBeliefTracker* tracker,
      const std::vector<ActionId>& legal_actions,
      std::vector<float>* priors,
      std::vector<float>* values) const override;

  bool evaluate_with_features(
      const std::vector<float>& features,
      int num_players,
      int perspective_player,
      const std::vector<ActionId>& legal_actions,
      const std::vector<float>& legal_mask,
      std::vector<float>* priors,
      std::vector<float>* values) const override;

  // Encodes every item, enqueues them under one lock, then waits on all.
  void evaluate_batch(const std::vector<BatchItem>& items) const override;

 private:
  // Completion state shared by the requests of one submit_and_wait call.
  // Lives on the blocked caller's stack: one wake per call, not per leaf.
  struct Group {
    std::atomic<std::size_t> remaining{0};
    std::mutex mu;
    std::condition_variable cv;
    bool done = false;
    std::exception_ptr error;
  };

  // One leaf. Every pointer is borrowed from the caller, which stays blocked
  // until its group completes. The inference thread only copies raw network
  // outputs; softmax and seat rotation run on the caller's thread.
  struct Request {
    const float* features = nullptr;
    std::size_t num_features = 0;     // feature_dim, or n * token_dim for variable-length tokens
    float* logits_out = nullptr;      // action_space floats
    float* values_out = nullptr;      // num_values floats, perspective order
    int num_values = 0;
    Group* group = nullptr;
    std::chrono::steady_clock::time_point enqueued_at;
  };

  // Worker entry point — spins until shutdown_, draining queue_ and firing
  // run_batch on each collected batch.
  void inference_loop(int session_index);

  // Push `reqs` onto queue_ atomically (all or none), wake an inference
  // thread, block until every request of `group` is served, rethrow its
  // first error. Throws on shutdown or max_queue_size overflow.
  void submit_and_wait(std::vector<Request>& reqs, Group& group) const;

  // Pack `batch` into a single ONNX Run() and write each request's outputs.
  void run_batch(const std::vector<Request>& batch, int session_index);

  // Count `batch` against its groups (recording `error` if set) and wake
  // every group whose last request this was.
  static void finish(const std::vector<Request>& batch, std::exception_ptr error);

  std::string model_path_;
  const IFeatureEncoder* encoder_ = nullptr;
  BatchedEvaluatorConfig cfg_{};
  bool ready_ = false;
  std::string last_error_{};

  // queue + cv for inference thread coordination
  mutable std::mutex mu_;
  mutable std::condition_variable cv_;
  std::deque<Request> queue_;
  std::atomic<bool> shutdown_{false};
  std::vector<std::thread> inference_threads_;

#if defined(BOARD_AI_WITH_ONNX) && BOARD_AI_WITH_ONNX
  struct Impl;
  Impl* impl_ = nullptr;
#endif
};

}  // namespace board_ai::infer
