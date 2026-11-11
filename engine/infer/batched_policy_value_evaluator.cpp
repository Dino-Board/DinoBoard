#include "batched_policy_value_evaluator.h"

#include <algorithm>
#include <array>
#include <chrono>
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
const OrtApi& ort_api_v2() {
  static const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  return *api;
}

void throw_on_ort_error_v2(OrtStatus* status) {
  if (!status) return;
  std::string msg = ort_api_v2().GetErrorMessage(status);
  ort_api_v2().ReleaseStatus(status);
  throw std::runtime_error(std::string("ONNX runtime error: ") + msg);
}

struct OrtEnvDeleter2 {
  void operator()(OrtEnv* p) const noexcept { if (p) ort_api_v2().ReleaseEnv(p); }
};
struct OrtSessionOptionsDeleter2 {
  void operator()(OrtSessionOptions* p) const noexcept { if (p) ort_api_v2().ReleaseSessionOptions(p); }
};
struct OrtSessionDeleter2 {
  void operator()(OrtSession* p) const noexcept { if (p) ort_api_v2().ReleaseSession(p); }
};
struct OrtMemoryInfoDeleter2 {
  void operator()(OrtMemoryInfo* p) const noexcept { if (p) ort_api_v2().ReleaseMemoryInfo(p); }
};
struct OrtValueDeleter2 {
  void operator()(OrtValue* p) const noexcept { if (p) ort_api_v2().ReleaseValue(p); }
};
struct OrtTensorShapeDeleter2 {
  void operator()(OrtTensorTypeAndShapeInfo* p) const noexcept { if (p) ort_api_v2().ReleaseTensorTypeAndShapeInfo(p); }
};

using OrtEnvPtr2 = std::unique_ptr<OrtEnv, OrtEnvDeleter2>;
using OrtSessionOptionsPtr2 = std::unique_ptr<OrtSessionOptions, OrtSessionOptionsDeleter2>;
using OrtSessionPtr2 = std::unique_ptr<OrtSession, OrtSessionDeleter2>;
using OrtMemoryInfoPtr2 = std::unique_ptr<OrtMemoryInfo, OrtMemoryInfoDeleter2>;
using OrtValuePtr2 = std::unique_ptr<OrtValue, OrtValueDeleter2>;
using OrtTensorShapePtr2 = std::unique_ptr<OrtTensorTypeAndShapeInfo, OrtTensorShapeDeleter2>;

std::string get_io_name(OrtSession* session, size_t index, OrtAllocator* allocator, bool input) {
  char* raw = nullptr;
  if (input) {
    throw_on_ort_error_v2(ort_api_v2().SessionGetInputName(session, index, allocator, &raw));
  } else {
    throw_on_ort_error_v2(ort_api_v2().SessionGetOutputName(session, index, allocator, &raw));
  }
  std::string s = raw ? raw : "";
  if (raw) throw_on_ort_error_v2(ort_api_v2().AllocatorFree(allocator, raw));
  return s;
}

OrtSession* create_session(OrtEnv* env, const std::string& model_path,
                           const OrtSessionOptions* opts) {
  OrtSession* raw = nullptr;
#if defined(_WIN32)
  std::wstring w(model_path.begin(), model_path.end());
  throw_on_ort_error_v2(ort_api_v2().CreateSession(env, w.c_str(), opts, &raw));
#else
  throw_on_ort_error_v2(ort_api_v2().CreateSession(env, model_path.c_str(), opts, &raw));
#endif
  return raw;
}

// TensorRT builds one engine per shape profile, and rebuilds (tens of
// seconds) whenever a batch falls outside it, so the profile must span every
// batch the queue can produce: 1..max_batch. Engines and kernel timings are
// cached next to the model so the per-thread sessions share one build.
void append_tensorrt(OrtEnv* env, OrtSessionOptions* opts,
                     const std::string& model_path, std::size_t max_batch, int device) {
  OrtSessionOptions* raw_probe_opts = nullptr;
  throw_on_ort_error_v2(ort_api_v2().CreateSessionOptions(&raw_probe_opts));
  OrtSessionOptionsPtr2 probe_opts(raw_probe_opts);
  OrtSessionPtr2 probe(create_session(env, model_path, probe_opts.get()));
  OrtAllocator* allocator = nullptr;
  throw_on_ort_error_v2(ort_api_v2().GetAllocatorWithDefaultOptions(&allocator));
  const std::string input = get_io_name(probe.get(), 0, allocator, true);
  OrtTypeInfo* type_info = nullptr;
  throw_on_ort_error_v2(ort_api_v2().SessionGetInputTypeInfo(probe.get(), 0, &type_info));
  const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
  std::vector<int64_t> dims;
  try {
    throw_on_ort_error_v2(ort_api_v2().CastTypeInfoToTensorInfo(type_info, &tensor_info));
    size_t ndim = 0;
    throw_on_ort_error_v2(ort_api_v2().GetDimensionsCount(tensor_info, &ndim));
    dims.resize(ndim);
    throw_on_ort_error_v2(ort_api_v2().GetDimensions(tensor_info, dims.data(), ndim));
  } catch (...) {
    ort_api_v2().ReleaseTypeInfo(type_info);
    throw;
  }
  ort_api_v2().ReleaseTypeInfo(type_info);
  if (dims.size() != 2 || dims[1] <= 0) {
    throw std::runtime_error(
        "tensorrt provider: expected input '" + input + "' of shape [batch, features] with a static "
        "feature count (variable-length token models are not supported)");
  }
  const std::string feat = std::to_string(dims[1]);
  const std::string min_shape = input + ":1x" + feat;
  const std::string max_shape = input + ":" + std::to_string(max_batch) + "x" + feat;
  const auto slash = model_path.find_last_of('/');
  const std::string cache_dir =
      (slash == std::string::npos ? std::string(".") : model_path.substr(0, slash)) + "/trt_cache";

  const std::array<const char*, 8> keys = {
      "device_id", "trt_engine_cache_enable", "trt_engine_cache_path",
      "trt_timing_cache_enable", "trt_timing_cache_path",
      "trt_profile_min_shapes", "trt_profile_opt_shapes", "trt_profile_max_shapes"};
  const std::string device_id = std::to_string(device);
  const std::array<const char*, 8> values = {
      device_id.c_str(), "1", cache_dir.c_str(), "1", cache_dir.c_str(),
      min_shape.c_str(), max_shape.c_str(), max_shape.c_str()};
  OrtTensorRTProviderOptionsV2* trt = nullptr;
  throw_on_ort_error_v2(ort_api_v2().CreateTensorRTProviderOptions(&trt));
  try {
    throw_on_ort_error_v2(ort_api_v2().UpdateTensorRTProviderOptions(
        trt, keys.data(), values.data(), keys.size()));
    throw_on_ort_error_v2(ort_api_v2().SessionOptionsAppendExecutionProvider_TensorRT_V2(opts, trt));
  } catch (...) {
    ort_api_v2().ReleaseTensorRTProviderOptions(trt);
    throw;
  }
  ort_api_v2().ReleaseTensorRTProviderOptions(trt);
}

// Session options for one device: thread counts, graph optimization and the
// execution providers in declared order. CPU EP is always available as a
// fallback (whether or not we explicitly ask for it), so a non-cpu provider
// that fails registration just means later EPs (or CPU) take its nodes.
// Unknown provider names hard-error so users catch typos instead of
// silently falling back.
OrtSessionOptionsPtr2 make_session_options(OrtEnv* env, const BatchedEvaluatorConfig& cfg,
                                           const std::string& model_path, int device) {
  OrtSessionOptions* raw_opts = nullptr;
  throw_on_ort_error_v2(ort_api_v2().CreateSessionOptions(&raw_opts));
  OrtSessionOptionsPtr2 opts(raw_opts);
  throw_on_ort_error_v2(ort_api_v2().SetIntraOpNumThreads(
      opts.get(), std::max(1, cfg.intra_op_num_threads)));
  throw_on_ort_error_v2(ort_api_v2().SetInterOpNumThreads(
      opts.get(), std::max(1, cfg.inter_op_num_threads)));
  throw_on_ort_error_v2(ort_api_v2().SetSessionGraphOptimizationLevel(
      opts.get(), GraphOptimizationLevel::ORT_ENABLE_EXTENDED));
  for (const auto& provider : cfg.providers) {
    if (provider == "cpu") {
      // ORT's default CPU provider is always registered last; explicit
      // request is a no-op (intentional — keeps the API uniform).
      continue;
    }
#if defined(__APPLE__)
    if (provider == "coreml") {
      // Default flags: let CoreML pick CPU+ANE+GPU; no static-shape
      // requirement (our nets aren't always static after batched
      // inputs of different sizes).
      const uint32_t coreml_flags = COREML_FLAG_USE_NONE;
      throw_on_ort_error_v2(
          OrtSessionOptionsAppendExecutionProvider_CoreML(opts.get(), coreml_flags));
      continue;
    }
#endif
    if (provider == "cuda") {
      // Look up CUDA EP symbol at runtime via dlsym so Apple/CPU-only
      // onnxruntime builds (which don't export this symbol) still load
      // and link the .so. Function exists only on GPU builds.
      using CudaAppendFn = OrtStatus* (*)(OrtSessionOptions*, int);
      CudaAppendFn cuda_append = reinterpret_cast<CudaAppendFn>(
          dlsym(RTLD_DEFAULT, "OrtSessionOptionsAppendExecutionProvider_CUDA"));
      if (!cuda_append) {
        throw std::runtime_error(
            "BatchedPolicyValueEvaluator: cuda provider requested but "
            "onnxruntime build does not export CUDA EP (install the "
            "GPU build of onnxruntime)");
      }
      throw_on_ort_error_v2(cuda_append(opts.get(), device));
      continue;
    }
    if (provider == "tensorrt") {
      append_tensorrt(env, opts.get(), model_path, cfg.batch_size, device);
      continue;
    }
    throw std::runtime_error(
        "BatchedPolicyValueEvaluator: unknown execution provider '" +
        provider + "' (recognized: 'cpu', 'cuda', 'tensorrt'"
#if defined(__APPLE__)
        ", 'coreml'"
#endif
        ")");
  }
  return opts;
}
#endif

// Same softmax-with-mask as OnnxPolicyValueEvaluator. Inlined here to avoid
// pulling that whole TU as a dependency. Kept bit-equivalent.
void masked_softmax(
    const float* logits,
    std::size_t num_logits,
    const std::vector<float>& mask,
    const std::vector<ActionId>& legal_actions,
    std::vector<float>* out) {
  out->assign(legal_actions.size(), 0.0f);
  if (legal_actions.empty()) {
    throw std::runtime_error("BatchedEvaluator masked_softmax: legal_actions empty");
  }
  if (num_logits == 0) {
    throw std::runtime_error("BatchedEvaluator masked_softmax: logits empty");
  }
  if (mask.empty()) {
    throw std::runtime_error("BatchedEvaluator masked_softmax: legal mask empty");
  }
  float max_logit = -std::numeric_limits<float>::infinity();
  for (ActionId a : legal_actions) {
    const int idx = static_cast<int>(a);
    if (idx < 0 || idx >= static_cast<int>(num_logits)) {
      throw std::runtime_error(
          "BatchedEvaluator masked_softmax: action " + std::to_string(a) +
          " outside policy logits length " + std::to_string(num_logits));
    }
    if (idx >= static_cast<int>(mask.size())) {
      throw std::runtime_error(
          "BatchedEvaluator masked_softmax: action " + std::to_string(a) +
          " outside legal mask length " + std::to_string(mask.size()));
    }
    if (mask[idx] <= 0.0f) {
      throw std::runtime_error(
          "BatchedEvaluator masked_softmax: action " + std::to_string(a) +
          " is masked out by encoder");
    }
    if (logits[idx] > max_logit) max_logit = logits[idx];
  }
  float denom = 0.0f;
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    const int idx = static_cast<int>(legal_actions[i]);
    const float e = std::exp(logits[idx] - max_logit);
    (*out)[i] = e;
    denom += e;
  }
  if (denom <= 0.0f) {
    throw std::runtime_error("BatchedEvaluator masked_softmax: zero denominator");
  }
  for (auto& p : *out) p /= denom;
}

// Caller-side post-processing of one leaf's raw outputs: rotate the
// perspective-relative values back to absolute seats (clipped to [-1, 1])
// and softmax the logits over the legal actions.
void finalize_leaf(
    const float* logits, std::size_t num_logits,
    const float* raw_values, int num_players, int perspective_player,
    const std::vector<float>& legal_mask,
    const std::vector<ActionId>& legal_actions,
    std::vector<float>* priors, std::vector<float>* values) {
  values->assign(static_cast<std::size_t>(num_players), 0.0f);
  for (int k = 0; k < num_players; ++k) {
    if (!std::isfinite(raw_values[k])) {
      throw std::runtime_error(
          "BatchedPolicyValueEvaluator: non-finite value at idx " + std::to_string(k));
    }
    const int abs_player = (perspective_player + k) % num_players;
    (*values)[static_cast<std::size_t>(abs_player)] =
        std::max(-1.0f, std::min(1.0f, raw_values[k]));
  }
  masked_softmax(logits, num_logits, legal_mask, legal_actions, priors);
}

}  // namespace

#if defined(BOARD_AI_WITH_ONNX) && BOARD_AI_WITH_ONNX
struct BatchedPolicyValueEvaluator::Impl {
  OrtEnvPtr2 env{};
  // One per device; session i uses opts[i % opts.size()].
  std::vector<OrtSessionOptionsPtr2> opts;
  // One per inference thread: concurrent Run() on a shared CUDA-EP session
  // returns corrupted outputs for some graphs (seen with the token
  // transformer), so sessions are never shared across threads.
  std::vector<OrtSessionPtr2> sessions;
  // One packed-input buffer per inference thread, reused across batches.
  std::vector<std::vector<float>> input_bufs;
  OrtMemoryInfoPtr2 cpu_meminfo{};
  std::string input_name;
  std::string policy_output_name;
  std::string value_output_name;
};
#endif

BatchedPolicyValueEvaluator::BatchedPolicyValueEvaluator(
    std::string model_path,
    const IFeatureEncoder* encoder,
    BatchedEvaluatorConfig cfg)
    : model_path_(std::move(model_path)), encoder_(encoder), cfg_(cfg), ready_(false) {
  if (!encoder_) {
    last_error_ = "encoder is null";
    throw std::runtime_error("BatchedPolicyValueEvaluator: encoder is null");
  }
  if (cfg_.batch_size == 0) {
    throw std::runtime_error("BatchedPolicyValueEvaluator: batch_size must be > 0");
  }
  if (cfg_.max_queue_size < cfg_.batch_size) {
    throw std::runtime_error(
        "BatchedPolicyValueEvaluator: max_queue_size < batch_size");
  }
  if (cfg_.num_inference_threads < 1) {
    throw std::runtime_error(
        "BatchedPolicyValueEvaluator: num_inference_threads must be >= 1");
  }

  const std::vector<int> devices = cfg_.devices.empty() ? std::vector<int>{0} : cfg_.devices;
  for (std::size_t i = 0; i < devices.size(); ++i) {
    if (devices[i] < 0) {
      throw std::runtime_error("BatchedPolicyValueEvaluator: device ids must be >= 0");
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (devices[j] == devices[i]) {
        throw std::runtime_error("BatchedPolicyValueEvaluator: duplicate device id " +
                                 std::to_string(devices[i]));
      }
    }
  }
  if (cfg_.num_inference_threads % static_cast<int>(devices.size()) != 0) {
    throw std::runtime_error(
        "BatchedPolicyValueEvaluator: num_inference_threads (" +
        std::to_string(cfg_.num_inference_threads) + ") must be a multiple of the " +
        std::to_string(devices.size()) + " devices");
  }
  if (!cfg_.devices.empty()) {
    bool gpu = false;
    for (const auto& p : cfg_.providers) gpu = gpu || p == "cuda" || p == "tensorrt";
    if (!gpu) {
      throw std::runtime_error(
          "BatchedPolicyValueEvaluator: devices needs a 'cuda' or 'tensorrt' provider");
    }
  }

#if defined(BOARD_AI_WITH_ONNX) && BOARD_AI_WITH_ONNX
  try {
    auto impl = std::make_unique<Impl>();
    OrtEnv* raw_env = nullptr;
    throw_on_ort_error_v2(ort_api_v2().CreateEnv(
        ORT_LOGGING_LEVEL_WARNING, "dino_batched_eval", &raw_env));
    impl->env.reset(raw_env);

    for (const int device : devices) {
      impl->opts.push_back(make_session_options(impl->env.get(), cfg_, model_path_, device));
    }
    const int n_inf = cfg_.num_inference_threads;
    for (int i = 0; i < n_inf; ++i) {
      impl->sessions.emplace_back(create_session(
          impl->env.get(), model_path_, impl->opts[static_cast<std::size_t>(i) % impl->opts.size()].get()));
    }

    OrtAllocator* allocator = nullptr;
    throw_on_ort_error_v2(ort_api_v2().GetAllocatorWithDefaultOptions(&allocator));
    OrtSession* first = impl->sessions.front().get();
    impl->input_name = get_io_name(first, 0, allocator, true);
    impl->policy_output_name = get_io_name(first, 0, allocator, false);
    impl->value_output_name = get_io_name(first, 1, allocator, false);
    impl->input_bufs.resize(static_cast<std::size_t>(n_inf));
    OrtMemoryInfo* raw_meminfo = nullptr;
    throw_on_ort_error_v2(ort_api_v2().CreateCpuMemoryInfo(
        OrtArenaAllocator, OrtMemTypeDefault, &raw_meminfo));
    impl->cpu_meminfo.reset(raw_meminfo);

    impl_ = impl.release();
    ready_ = true;
    for (int i = 0; i < n_inf; ++i) {
      inference_threads_.emplace_back(&BatchedPolicyValueEvaluator::inference_loop, this, i);
    }
  } catch (const std::exception& e) {
    last_error_ = e.what();
    ready_ = false;
    throw std::runtime_error(
        std::string("BatchedPolicyValueEvaluator: failed to load model '") +
        model_path_ + "': " + e.what());
  }
#else
  ready_ = false;
  last_error_ = "onnx runtime not enabled at build time";
  throw std::runtime_error(
      "BatchedPolicyValueEvaluator: ONNX runtime not enabled at build time");
#endif
}

BatchedPolicyValueEvaluator::~BatchedPolicyValueEvaluator() {
  shutdown_.store(true, std::memory_order_release);
  cv_.notify_all();
  for (auto& th : inference_threads_) {
    if (th.joinable()) th.join();
  }
  // Drain any remaining requests with errors, in case shutdown raced.
  {
    std::unique_lock<std::mutex> lk(mu_);
    const std::vector<Request> rest(queue_.begin(), queue_.end());
    queue_.clear();
    finish(rest, std::make_exception_ptr(std::runtime_error(
        "BatchedPolicyValueEvaluator: shutdown before request served")));
  }
#if defined(BOARD_AI_WITH_ONNX) && BOARD_AI_WITH_ONNX
  delete impl_;
  impl_ = nullptr;
#endif
}

bool BatchedPolicyValueEvaluator::evaluate(
    const IGameState& masked_state,
    int perspective_player,
    const IBeliefTracker* tracker,
    const std::vector<ActionId>& legal_actions,
    std::vector<float>* priors,
    std::vector<float>* values) const {
  if (!priors || !values) {
    throw std::runtime_error(
        "BatchedPolicyValueEvaluator::evaluate: priors or values pointer is null");
  }
  if (!encoder_) {
    throw std::runtime_error("BatchedPolicyValueEvaluator::evaluate: encoder is null");
  }
  const int num_players = masked_state.num_players();

  std::vector<float> features;
  std::vector<float> legal_mask;
  if (!encoder_->encode_with_masked(
          masked_state, perspective_player, tracker,
          legal_actions, &features, &legal_mask)) {
    throw std::runtime_error(
        "BatchedPolicyValueEvaluator::evaluate: encoder.encode_with_masked() failed");
  }

  return evaluate_with_features(
      features, num_players, perspective_player,
      legal_actions, legal_mask, priors, values);
}

bool BatchedPolicyValueEvaluator::evaluate_with_features(
    const std::vector<float>& features,
    int num_players,
    int perspective_player,
    const std::vector<ActionId>& legal_actions,
    const std::vector<float>& legal_mask,
    std::vector<float>* priors,
    std::vector<float>* values) const {
  if (!priors || !values) {
    throw std::runtime_error(
        "BatchedPolicyValueEvaluator::evaluate_with_features: out ptr null");
  }
  if (!encoder_->valid_feature_length(features.size())) {
    throw std::runtime_error(
        "BatchedPolicyValueEvaluator::evaluate_with_features: feature length " +
        std::to_string(features.size()) + " is not a whole encoding (feature_dim " +
        std::to_string(encoder_->feature_dim()) + ")");
  }
  const std::size_t action_space = static_cast<std::size_t>(encoder_->action_space());
  thread_local std::vector<float> logits;
  thread_local std::vector<float> raw_values;
  thread_local std::vector<Request> reqs;
  logits.resize(action_space);
  raw_values.resize(static_cast<std::size_t>(num_players));
  reqs.clear();
  Request r{};
  r.features = features.data();
  r.num_features = features.size();
  r.logits_out = logits.data();
  r.values_out = raw_values.data();
  r.num_values = num_players;
  Group group;
  r.group = &group;
  reqs.push_back(r);
  group.remaining.store(1, std::memory_order_relaxed);
  submit_and_wait(reqs, group);
  finalize_leaf(logits.data(), action_space, raw_values.data(), num_players,
                perspective_player, legal_mask, legal_actions, priors, values);
  return true;
}

void BatchedPolicyValueEvaluator::evaluate_batch(
    const std::vector<BatchItem>& items) const {
  if (items.empty()) return;
  if (!encoder_) {
    throw std::runtime_error("BatchedPolicyValueEvaluator::evaluate_batch: encoder is null");
  }
  const std::size_t n = items.size();
  const std::size_t action_space = static_cast<std::size_t>(encoder_->action_space());
  // Reused across calls on this thread so a search allocates nothing per leaf.
  thread_local std::vector<std::vector<float>> features;
  thread_local std::vector<std::vector<float>> masks;
  thread_local std::vector<float> logits;
  thread_local std::vector<float> raw_values;
  thread_local std::vector<std::size_t> value_offsets;
  thread_local std::vector<Request> reqs;
  if (features.size() < n) {
    features.resize(n);
    masks.resize(n);
  }
  value_offsets.resize(n);
  std::size_t total_values = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto& it = items[i];
    if (!it.masked_state || !it.legal_actions || !it.priors || !it.values) {
      throw std::runtime_error(
          "BatchedPolicyValueEvaluator::evaluate_batch: null pointer in batch item");
    }
    if (!encoder_->encode_with_masked(
            *it.masked_state, it.perspective_player, it.tracker,
            *it.legal_actions, &features[i], &masks[i])) {
      throw std::runtime_error(
          "BatchedPolicyValueEvaluator::evaluate_batch: encoder.encode_with_masked() failed");
    }
    value_offsets[i] = total_values;
    total_values += static_cast<std::size_t>(it.masked_state->num_players());
  }
  logits.resize(n * action_space);
  raw_values.resize(total_values);
  Group group;
  reqs.clear();
  for (std::size_t i = 0; i < n; ++i) {
    Request r{};
    r.features = features[i].data();
    r.num_features = features[i].size();
    r.logits_out = logits.data() + i * action_space;
    r.values_out = raw_values.data() + value_offsets[i];
    r.num_values = items[i].masked_state->num_players();
    r.group = &group;
    reqs.push_back(r);
  }
  group.remaining.store(n, std::memory_order_relaxed);
  submit_and_wait(reqs, group);
  for (std::size_t i = 0; i < n; ++i) {
    const auto& it = items[i];
    finalize_leaf(logits.data() + i * action_space, action_space,
                  raw_values.data() + value_offsets[i], it.masked_state->num_players(),
                  it.perspective_player, masks[i], *it.legal_actions,
                  it.priors, it.values);
  }
}

void BatchedPolicyValueEvaluator::submit_and_wait(
    std::vector<Request>& reqs, Group& group) const {
  if (!ready_) {
    throw std::runtime_error("BatchedPolicyValueEvaluator: not ready");
  }
  if (shutdown_.load(std::memory_order_acquire)) {
    throw std::runtime_error("BatchedPolicyValueEvaluator: evaluator shut down");
  }
  const auto now = std::chrono::steady_clock::now();
  // Cast away const because we own the queue mutation. The const on
  // evaluate*() is a legacy interface contract — callers treat the evaluator
  // as logically const (no externally-visible state mutation).
  auto* self = const_cast<BatchedPolicyValueEvaluator*>(this);
  {
    std::unique_lock<std::mutex> lk(self->mu_);
    if (self->queue_.size() + reqs.size() > cfg_.max_queue_size) {
      throw std::runtime_error(
          "BatchedPolicyValueEvaluator: queue overflow (max_queue_size=" +
          std::to_string(cfg_.max_queue_size) + ", pending=" +
          std::to_string(self->queue_.size()) + ", adding=" +
          std::to_string(reqs.size()) + ")");
    }
    for (auto& r : reqs) {
      r.enqueued_at = now;
      self->queue_.push_back(r);
    }
  }
  self->cv_.notify_one();
  {
    std::unique_lock<std::mutex> lk(group.mu);
    group.cv.wait(lk, [&group] { return group.done; });
  }
  if (group.error) std::rethrow_exception(group.error);
}

void BatchedPolicyValueEvaluator::finish(
    const std::vector<Request>& batch, std::exception_ptr error) {
  // Requests of one call are contiguous in the queue, so count runs of the
  // same group instead of touching the shared counter once per leaf.
  std::size_t i = 0;
  while (i < batch.size()) {
    Group* g = batch[i].group;
    std::size_t j = i + 1;
    while (j < batch.size() && batch[j].group == g) ++j;
    const std::size_t k = j - i;
    if (error) {
      std::lock_guard<std::mutex> lk(g->mu);
      if (!g->error) g->error = error;
    }
    if (g->remaining.fetch_sub(k, std::memory_order_acq_rel) == k) {
      // Notify under the lock: once `done` is visible the caller may return
      // and destroy `g`, so nothing may touch it after this scope.
      std::lock_guard<std::mutex> lk(g->mu);
      g->done = true;
      g->cv.notify_one();
    }
    i = j;
  }
}

void BatchedPolicyValueEvaluator::inference_loop(int session_index) {
  std::vector<Request> batch;
  batch.reserve(cfg_.batch_size);
  while (!shutdown_.load(std::memory_order_acquire)) {
    batch.clear();
    {
      std::unique_lock<std::mutex> lk(mu_);

      // Wait until queue not empty (or shutdown).
      cv_.wait(lk, [this] {
        return shutdown_.load(std::memory_order_acquire) || !queue_.empty();
      });
      if (shutdown_.load(std::memory_order_acquire) && queue_.empty()) break;

      // Now compute the timeout deadline based on oldest request's
      // enqueued_at, and wait for batch_size or deadline (whichever first).
      const auto deadline = queue_.front().enqueued_at +
          std::chrono::microseconds(cfg_.batch_timeout_us);
      cv_.wait_until(lk, deadline, [this] {
        return shutdown_.load(std::memory_order_acquire) ||
               queue_.size() >= cfg_.batch_size;
      });

      // Drain up to batch_size; stop early if shutdown so destructor's
      // drain-and-error path can run (we leave the rest in queue_).
      const std::size_t take = std::min(queue_.size(), cfg_.batch_size);
      batch.assign(queue_.begin(), queue_.begin() + static_cast<std::ptrdiff_t>(take));
      queue_.erase(queue_.begin(), queue_.begin() + static_cast<std::ptrdiff_t>(take));
    }

    if (batch.empty()) continue;

    // Never let an exception escape inference_loop and silently kill the
    // thread: a failed batch is reported to every group it touches.
    std::exception_ptr error;
    try {
      run_batch(batch, session_index);
    } catch (...) {
      error = std::current_exception();
    }
    finish(batch, error);
  }
}

#if defined(BOARD_AI_WITH_ONNX) && BOARD_AI_WITH_ONNX

void BatchedPolicyValueEvaluator::run_batch(
    const std::vector<Request>& batch, int session_index) {
  if (batch.empty()) return;
  const std::size_t action_space = static_cast<std::size_t>(encoder_->action_space());
  const std::size_t batch_n = batch.size();

  std::size_t feat_dim = static_cast<std::size_t>(encoder_->feature_dim());
  if (encoder_->variable_tokens()) {
    const auto token_dim = static_cast<std::size_t>(encoder_->token_layout()->token_dim);
    std::size_t longest = 0;
    for (const Request& r : batch) longest = std::max(longest, r.num_features / token_dim);
    feat_dim = static_cast<std::size_t>(padded_token_count(static_cast<int>(longest))) * token_dim;
  }
  std::vector<float>& input_buf = impl_->input_bufs[static_cast<std::size_t>(session_index)];
  input_buf.assign(batch_n * feat_dim, 0.0f);
  for (std::size_t i = 0; i < batch_n; ++i) {
    std::memcpy(input_buf.data() + i * feat_dim, batch[i].features, batch[i].num_features * sizeof(float));
  }

  std::array<int64_t, 2> shape = {
      static_cast<int64_t>(batch_n), static_cast<int64_t>(feat_dim)};
  OrtValue* raw_input = nullptr;
  throw_on_ort_error_v2(ort_api_v2().CreateTensorWithDataAsOrtValue(
      impl_->cpu_meminfo.get(),
      input_buf.data(),
      input_buf.size() * sizeof(float),
      shape.data(),
      shape.size(),
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      &raw_input));
  OrtValuePtr2 input(raw_input);

  const char* in_names[] = {impl_->input_name.c_str()};
  const char* out_names[] = {
      impl_->policy_output_name.c_str(),
      impl_->value_output_name.c_str()};
  const OrtValue* input_values[] = {input.get()};
  OrtValue* raw_outputs[2] = {nullptr, nullptr};
  throw_on_ort_error_v2(ort_api_v2().Run(
      impl_->sessions[static_cast<std::size_t>(session_index)].get(), nullptr, in_names, input_values, 1,
      out_names, 2, raw_outputs));
  OrtValuePtr2 policy_output(raw_outputs[0]);
  OrtValuePtr2 value_output(raw_outputs[1]);

  const auto output_dims = [](OrtValue* value) {
    OrtTensorTypeAndShapeInfo* raw_shape = nullptr;
    throw_on_ort_error_v2(ort_api_v2().GetTensorTypeAndShape(value, &raw_shape));
    OrtTensorShapePtr2 shape_info(raw_shape);
    size_t ndims = 0;
    throw_on_ort_error_v2(ort_api_v2().GetDimensionsCount(shape_info.get(), &ndims));
    std::vector<int64_t> dims(ndims, 0);
    if (ndims > 0) {
      throw_on_ort_error_v2(ort_api_v2().GetDimensions(shape_info.get(), dims.data(), ndims));
    }
    return dims;
  };
  const std::vector<int64_t> p_dims = output_dims(policy_output.get());
  if (p_dims.size() != 2 || p_dims[0] != static_cast<int64_t>(batch_n) ||
      p_dims[1] != static_cast<int64_t>(action_space)) {
    throw std::runtime_error(
        "BatchedPolicyValueEvaluator: policy output shape mismatch (expected [" +
        std::to_string(batch_n) + ", " + std::to_string(action_space) + "])");
  }
  const std::vector<int64_t> v_dims = output_dims(value_output.get());
  if (v_dims.size() != 2 || v_dims[0] != static_cast<int64_t>(batch_n) || v_dims[1] <= 0) {
    throw std::runtime_error("BatchedPolicyValueEvaluator: value output shape mismatch");
  }
  const std::size_t value_len = static_cast<std::size_t>(v_dims[1]);

  void* policy_raw = nullptr;
  throw_on_ort_error_v2(ort_api_v2().GetTensorMutableData(policy_output.get(), &policy_raw));
  const float* policy_ptr = static_cast<const float*>(policy_raw);
  void* value_raw = nullptr;
  throw_on_ort_error_v2(ort_api_v2().GetTensorMutableData(value_output.get(), &value_raw));
  const float* value_ptr = static_cast<const float*>(value_raw);

  for (std::size_t i = 0; i < batch_n; ++i) {
    const Request& r = batch[i];
    if (static_cast<std::size_t>(r.num_values) > value_len) {
      throw std::runtime_error(
          "BatchedPolicyValueEvaluator: value_len " + std::to_string(value_len) +
          " < num_players " + std::to_string(r.num_values));
    }
    std::memcpy(r.logits_out, policy_ptr + i * action_space, action_space * sizeof(float));
    std::memcpy(r.values_out, value_ptr + i * value_len,
                static_cast<std::size_t>(r.num_values) * sizeof(float));
  }
}

#else

void BatchedPolicyValueEvaluator::run_batch(
    const std::vector<Request>& /*batch*/, int /*session_index*/) {
  throw std::runtime_error(
      "BatchedPolicyValueEvaluator: ONNX runtime not enabled at build time");
}

#endif

}  // namespace board_ai::infer
