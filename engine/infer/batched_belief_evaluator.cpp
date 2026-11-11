#include "batched_belief_evaluator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(BOARD_AI_WITH_ONNX) && BOARD_AI_WITH_ONNX
#include <onnxruntime_c_api.h>
#if defined(__APPLE__)
#include <coreml_provider_factory.h>
#endif
#include <dlfcn.h>
#endif

namespace board_ai::infer {

namespace {

#if defined(BOARD_AI_WITH_ONNX) && BOARD_AI_WITH_ONNX
const OrtApi& ort_api_b() {
  static const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  return *api;
}

void throw_on_ort_error_b(OrtStatus* status) {
  if (!status) return;
  std::string msg = ort_api_b().GetErrorMessage(status);
  ort_api_b().ReleaseStatus(status);
  throw std::runtime_error(std::string("ONNX runtime error: ") + msg);
}

struct OrtEnvDeleterB {
  void operator()(OrtEnv* p) const noexcept { if (p) ort_api_b().ReleaseEnv(p); }
};
struct OrtSessionOptionsDeleterB {
  void operator()(OrtSessionOptions* p) const noexcept { if (p) ort_api_b().ReleaseSessionOptions(p); }
};
struct OrtSessionDeleterB {
  void operator()(OrtSession* p) const noexcept { if (p) ort_api_b().ReleaseSession(p); }
};
struct OrtMemoryInfoDeleterB {
  void operator()(OrtMemoryInfo* p) const noexcept { if (p) ort_api_b().ReleaseMemoryInfo(p); }
};
struct OrtValueDeleterB {
  void operator()(OrtValue* p) const noexcept { if (p) ort_api_b().ReleaseValue(p); }
};
struct OrtTensorShapeDeleterB {
  void operator()(OrtTensorTypeAndShapeInfo* p) const noexcept { if (p) ort_api_b().ReleaseTensorTypeAndShapeInfo(p); }
};

using OrtEnvPtrB = std::unique_ptr<OrtEnv, OrtEnvDeleterB>;
using OrtSessionOptionsPtrB = std::unique_ptr<OrtSessionOptions, OrtSessionOptionsDeleterB>;
using OrtSessionPtrB = std::unique_ptr<OrtSession, OrtSessionDeleterB>;
using OrtMemoryInfoPtrB = std::unique_ptr<OrtMemoryInfo, OrtMemoryInfoDeleterB>;
using OrtValuePtrB = std::unique_ptr<OrtValue, OrtValueDeleterB>;
using OrtTensorShapePtrB = std::unique_ptr<OrtTensorTypeAndShapeInfo, OrtTensorShapeDeleterB>;

std::string get_io_name_b(OrtSession* session, size_t index, OrtAllocator* allocator, bool input) {
  char* raw = nullptr;
  if (input) {
    throw_on_ort_error_b(ort_api_b().SessionGetInputName(session, index, allocator, &raw));
  } else {
    throw_on_ort_error_b(ort_api_b().SessionGetOutputName(session, index, allocator, &raw));
  }
  std::string s = raw ? raw : "";
  if (raw) throw_on_ort_error_b(ort_api_b().AllocatorFree(allocator, raw));
  return s;
}
#endif

}  // namespace

#if defined(BOARD_AI_WITH_ONNX) && BOARD_AI_WITH_ONNX
struct BatchedBeliefEvaluator::Impl {
  OrtEnvPtrB env{};
  OrtSessionOptionsPtrB opts{};
  OrtSessionPtrB session{};
  std::string input_name;
  std::string logits_output_name;
};
#endif

BatchedBeliefEvaluator::BatchedBeliefEvaluator(
    std::string model_path,
    BatchedEvaluatorConfig cfg)
    : model_path_(std::move(model_path)), cfg_(cfg), ready_(false) {
  if (cfg_.batch_size == 0) {
    throw std::runtime_error("BatchedBeliefEvaluator: batch_size must be > 0");
  }
  if (cfg_.max_queue_size < cfg_.batch_size) {
    throw std::runtime_error(
        "BatchedBeliefEvaluator: max_queue_size < batch_size");
  }

#if defined(BOARD_AI_WITH_ONNX) && BOARD_AI_WITH_ONNX
  try {
    auto impl = std::make_unique<Impl>();
    OrtEnv* raw_env = nullptr;
    throw_on_ort_error_b(ort_api_b().CreateEnv(
        ORT_LOGGING_LEVEL_WARNING, "dino_batched_belief", &raw_env));
    impl->env.reset(raw_env);

    OrtSessionOptions* raw_opts = nullptr;
    throw_on_ort_error_b(ort_api_b().CreateSessionOptions(&raw_opts));
    impl->opts.reset(raw_opts);
    throw_on_ort_error_b(ort_api_b().SetIntraOpNumThreads(
        impl->opts.get(), std::max(1, cfg_.intra_op_num_threads)));
    throw_on_ort_error_b(ort_api_b().SetInterOpNumThreads(
        impl->opts.get(), std::max(1, cfg_.inter_op_num_threads)));
    throw_on_ort_error_b(ort_api_b().SetSessionGraphOptimizationLevel(
        impl->opts.get(), GraphOptimizationLevel::ORT_ENABLE_EXTENDED));

    // Same EP append logic as BatchedPolicyValueEvaluator.
    for (const auto& provider : cfg_.providers) {
      if (provider == "cpu") continue;
#if defined(__APPLE__)
      if (provider == "coreml") {
        const uint32_t coreml_flags = COREML_FLAG_USE_NONE;
        throw_on_ort_error_b(
            OrtSessionOptionsAppendExecutionProvider_CoreML(
                impl->opts.get(), coreml_flags));
        continue;
      }
#endif
      if (provider == "cuda") {
        using CudaAppendFn = OrtStatus* (*)(OrtSessionOptions*, int);
        CudaAppendFn cuda_append = reinterpret_cast<CudaAppendFn>(
            dlsym(RTLD_DEFAULT,
                  "OrtSessionOptionsAppendExecutionProvider_CUDA"));
        if (!cuda_append) {
          throw std::runtime_error(
              "BatchedBeliefEvaluator: cuda provider requested but "
              "onnxruntime build does not export CUDA EP");
        }
        throw_on_ort_error_b(cuda_append(impl->opts.get(), /*device_id=*/0));
        continue;
      }
      throw std::runtime_error(
          "BatchedBeliefEvaluator: unknown execution provider '" + provider + "'");
    }

    OrtSession* raw_session = nullptr;
#if defined(_WIN32)
    std::wstring model_path_w(model_path_.begin(), model_path_.end());
    throw_on_ort_error_b(ort_api_b().CreateSession(
        impl->env.get(), model_path_w.c_str(), impl->opts.get(), &raw_session));
#else
    throw_on_ort_error_b(ort_api_b().CreateSession(
        impl->env.get(), model_path_.c_str(), impl->opts.get(), &raw_session));
#endif
    impl->session.reset(raw_session);

    OrtAllocator* allocator = nullptr;
    throw_on_ort_error_b(ort_api_b().GetAllocatorWithDefaultOptions(&allocator));
    impl->input_name = get_io_name_b(impl->session.get(), 0, allocator, true);
    impl->logits_output_name = get_io_name_b(impl->session.get(), 0, allocator, false);

    impl_ = impl.release();
    ready_ = true;
    inference_thread_ = std::thread(&BatchedBeliefEvaluator::inference_loop, this);
  } catch (const std::exception& e) {
    last_error_ = e.what();
    ready_ = false;
    throw std::runtime_error(
        std::string("BatchedBeliefEvaluator: failed to load model '") +
        model_path_ + "': " + e.what());
  }
#else
  ready_ = false;
  last_error_ = "onnx runtime not enabled at build time";
  throw std::runtime_error(
      "BatchedBeliefEvaluator: ONNX runtime not enabled at build time");
#endif
}

BatchedBeliefEvaluator::~BatchedBeliefEvaluator() {
  shutdown_.store(true, std::memory_order_release);
  cv_.notify_all();
  if (inference_thread_.joinable()) {
    inference_thread_.join();
  }
  {
    std::unique_lock<std::mutex> lk(mu_);
    while (!queue_.empty()) {
      auto req = std::move(queue_.front());
      queue_.pop_front();
      EvalResult r{};
      r.error = std::make_exception_ptr(std::runtime_error(
          "BatchedBeliefEvaluator: shutdown before request served"));
      req->promise.set_value(std::move(r));
    }
  }
#if defined(BOARD_AI_WITH_ONNX) && BOARD_AI_WITH_ONNX
  delete impl_;
  impl_ = nullptr;
#endif
}

bool BatchedBeliefEvaluator::evaluate(
    const std::vector<float>& features,
    std::vector<float>* logits) const {
  if (!logits) {
    throw std::runtime_error("BatchedBeliefEvaluator::evaluate: out ptr null");
  }
  if (!ready_) {
    throw std::runtime_error("BatchedBeliefEvaluator::evaluate: not ready");
  }
  if (shutdown_.load(std::memory_order_acquire)) {
    throw std::runtime_error("BatchedBeliefEvaluator::evaluate: shut down");
  }

  auto req = std::make_unique<PendingRequest>();
  req->features = features;
  req->enqueued_at = std::chrono::steady_clock::now();
  auto future = req->promise.get_future();

  auto* self = const_cast<BatchedBeliefEvaluator*>(this);
  {
    std::unique_lock<std::mutex> lk(self->mu_);
    if (self->queue_.size() >= cfg_.max_queue_size) {
      throw std::runtime_error(
          "BatchedBeliefEvaluator: queue overflow (max_queue_size=" +
          std::to_string(cfg_.max_queue_size) + ")");
    }
    self->queue_.push_back(std::move(req));
  }
  self->cv_.notify_one();

  EvalResult result = future.get();
  if (result.error) {
    std::rethrow_exception(result.error);
  }
  *logits = std::move(result.logits);
  return true;
}

void BatchedBeliefEvaluator::inference_loop() {
  while (!shutdown_.load(std::memory_order_acquire)) {
    std::vector<std::unique_ptr<PendingRequest>> batch;
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [this] {
        return shutdown_.load(std::memory_order_acquire) || !queue_.empty();
      });
      if (shutdown_.load(std::memory_order_acquire) && queue_.empty()) break;
      const auto deadline = queue_.front()->enqueued_at +
          std::chrono::microseconds(cfg_.batch_timeout_us);
      cv_.wait_until(lk, deadline, [this] {
        return shutdown_.load(std::memory_order_acquire) ||
               queue_.size() >= cfg_.batch_size;
      });
      const std::size_t take = std::min(queue_.size(), cfg_.batch_size);
      batch.reserve(take);
      for (std::size_t i = 0; i < take; ++i) {
        batch.push_back(std::move(queue_.front()));
        queue_.pop_front();
      }
    }
    if (batch.empty()) continue;
    try {
      run_batch(batch);
    } catch (...) {
      auto err = std::current_exception();
      for (auto& req : batch) {
        EvalResult r{};
        r.error = err;
        req->promise.set_value(std::move(r));
      }
    }
  }
}

#if defined(BOARD_AI_WITH_ONNX) && BOARD_AI_WITH_ONNX

void BatchedBeliefEvaluator::run_batch(
    std::vector<std::unique_ptr<PendingRequest>>& requests) {
  if (requests.empty()) return;

  const std::size_t feat_dim = requests.front()->features.size();
  for (const auto& r : requests) {
    if (r->features.size() != feat_dim) {
      throw std::runtime_error(
          "BatchedBeliefEvaluator: feature length mismatch in batch");
    }
  }

  const std::size_t batch_n = requests.size();
  std::vector<float> input_buf(batch_n * feat_dim);
  for (std::size_t i = 0; i < batch_n; ++i) {
    std::memcpy(input_buf.data() + i * feat_dim,
                requests[i]->features.data(),
                feat_dim * sizeof(float));
  }

  std::array<int64_t, 2> shape = {
      static_cast<int64_t>(batch_n), static_cast<int64_t>(feat_dim)};

  OrtMemoryInfo* raw_meminfo = nullptr;
  throw_on_ort_error_b(ort_api_b().CreateCpuMemoryInfo(
      OrtArenaAllocator, OrtMemTypeDefault, &raw_meminfo));
  OrtMemoryInfoPtrB meminfo(raw_meminfo);

  OrtValue* raw_input = nullptr;
  throw_on_ort_error_b(ort_api_b().CreateTensorWithDataAsOrtValue(
      meminfo.get(),
      input_buf.data(),
      input_buf.size() * sizeof(float),
      shape.data(),
      shape.size(),
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      &raw_input));
  OrtValuePtrB input(raw_input);

  const char* in_names[] = {impl_->input_name.c_str()};
  const char* out_names[] = {impl_->logits_output_name.c_str()};
  const OrtValue* input_values[] = {input.get()};
  OrtValue* raw_outputs[1] = {nullptr};
  throw_on_ort_error_b(ort_api_b().Run(
      impl_->session.get(), nullptr, in_names, input_values, 1,
      out_names, 1, raw_outputs));
  OrtValuePtrB logits_output(raw_outputs[0]);

  OrtTensorTypeAndShapeInfo* raw_lshape = nullptr;
  throw_on_ort_error_b(ort_api_b().GetTensorTypeAndShape(
      logits_output.get(), &raw_lshape));
  OrtTensorShapePtrB lshape(raw_lshape);
  size_t ldims = 0;
  throw_on_ort_error_b(ort_api_b().GetDimensionsCount(lshape.get(), &ldims));
  if (ldims < 2) {
    throw std::runtime_error("BatchedBeliefEvaluator: logits dims < 2");
  }
  std::vector<int64_t> l_dim_arr(ldims, 0);
  throw_on_ort_error_b(ort_api_b().GetDimensions(
      lshape.get(), l_dim_arr.data(), l_dim_arr.size()));
  if (l_dim_arr[0] != static_cast<int64_t>(batch_n)) {
    throw std::runtime_error("BatchedBeliefEvaluator: logits batch dim mismatch");
  }
  const int64_t logit_len = l_dim_arr.back();
  if (logit_len <= 0) {
    throw std::runtime_error("BatchedBeliefEvaluator: invalid logit length");
  }
  void* logits_raw = nullptr;
  throw_on_ort_error_b(ort_api_b().GetTensorMutableData(
      logits_output.get(), &logits_raw));
  const float* logits_ptr = static_cast<const float*>(logits_raw);

  for (std::size_t i = 0; i < batch_n; ++i) {
    auto& req = *requests[i];
    EvalResult res{};
    try {
      res.logits.assign(static_cast<size_t>(logit_len), 0.0f);
      const float* this_logits = logits_ptr + i * logit_len;
      std::copy(this_logits, this_logits + logit_len, res.logits.begin());
      // Sanity: NaN check
      for (auto v : res.logits) {
        if (!std::isfinite(v)) {
          throw std::runtime_error("BatchedBeliefEvaluator: non-finite logit");
        }
      }
    } catch (...) {
      res.error = std::current_exception();
    }
    req.promise.set_value(std::move(res));
  }
}

#else

void BatchedBeliefEvaluator::run_batch(
    std::vector<std::unique_ptr<PendingRequest>>& requests) {
  for (auto& req : requests) {
    EvalResult r{};
    r.error = std::make_exception_ptr(std::runtime_error(
        "BatchedBeliefEvaluator: ONNX runtime not enabled at build time"));
    req->promise.set_value(std::move(r));
  }
}

#endif

}  // namespace board_ai::infer
