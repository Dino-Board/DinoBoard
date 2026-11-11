#include "onnx_belief_evaluator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unordered_map>

#if defined(BOARD_AI_WITH_ONNX) && BOARD_AI_WITH_ONNX
#include <onnxruntime_c_api.h>
#endif

namespace board_ai::infer {

#if defined(BOARD_AI_WITH_ONNX) && BOARD_AI_WITH_ONNX
namespace {
const OrtApi& ort_api() noexcept {
  static const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  return *api;
}

void throw_on_ort_error(OrtStatus* status) {
  if (!status) return;
  const char* message = ort_api().GetErrorMessage(status);
  std::string error = message ? message : "onnxruntime error";
  ort_api().ReleaseStatus(status);
  throw std::runtime_error(error);
}

struct OrtEnvDeleter {
  void operator()(OrtEnv* ptr) const noexcept { if (ptr) ort_api().ReleaseEnv(ptr); }
};
struct OrtSessionOptionsDeleter {
  void operator()(OrtSessionOptions* ptr) const noexcept { if (ptr) ort_api().ReleaseSessionOptions(ptr); }
};
struct OrtSessionDeleter {
  void operator()(OrtSession* ptr) const noexcept { if (ptr) ort_api().ReleaseSession(ptr); }
};
struct OrtMemoryInfoDeleter {
  void operator()(OrtMemoryInfo* ptr) const noexcept { if (ptr) ort_api().ReleaseMemoryInfo(ptr); }
};
struct OrtValueDeleter {
  void operator()(OrtValue* ptr) const noexcept { if (ptr) ort_api().ReleaseValue(ptr); }
};
struct OrtTensorShapeDeleter {
  void operator()(OrtTensorTypeAndShapeInfo* ptr) const noexcept { if (ptr) ort_api().ReleaseTensorTypeAndShapeInfo(ptr); }
};

using OrtEnvPtr = std::unique_ptr<OrtEnv, OrtEnvDeleter>;
using OrtSessionOptionsPtr = std::unique_ptr<OrtSessionOptions, OrtSessionOptionsDeleter>;
using OrtSessionPtr = std::unique_ptr<OrtSession, OrtSessionDeleter>;
using OrtMemoryInfoPtr = std::unique_ptr<OrtMemoryInfo, OrtMemoryInfoDeleter>;
using OrtValuePtr = std::unique_ptr<OrtValue, OrtValueDeleter>;
using OrtTensorShapePtr = std::unique_ptr<OrtTensorTypeAndShapeInfo, OrtTensorShapeDeleter>;

std::string get_session_name(OrtSession* session, size_t index, OrtAllocator* allocator, bool input) {
  char* raw_name = nullptr;
  if (input) {
    throw_on_ort_error(ort_api().SessionGetInputName(session, index, allocator, &raw_name));
  } else {
    throw_on_ort_error(ort_api().SessionGetOutputName(session, index, allocator, &raw_name));
  }
  std::string name = raw_name ? raw_name : "";
  if (raw_name) {
    throw_on_ort_error(ort_api().AllocatorFree(allocator, raw_name));
  }
  return name;
}

struct CachedBeliefSessionBundle {
  OrtEnvPtr env{};
  OrtSessionOptionsPtr opts{};
  OrtSessionPtr session{};
  std::string input_name;
  std::string output_name;
};
}  // namespace

struct OnnxBeliefEvaluator::Impl {
  std::shared_ptr<CachedBeliefSessionBundle> bundle;
};

namespace {
struct OnnxBeliefSessionCache {
  std::mutex mu;
  std::unordered_map<std::string, std::shared_ptr<CachedBeliefSessionBundle>> map;
};
OnnxBeliefSessionCache& belief_session_cache() {
  static auto* cache = new OnnxBeliefSessionCache();
  return *cache;
}
constexpr std::size_t kMaxCachedBeliefSessions = 8;

std::string make_belief_cache_key(const std::string& model_path, const OnnxEvaluatorConfig& cfg) {
  std::string fingerprint = "missing";
  struct stat st{};
  if (::stat(model_path.c_str(), &st) == 0) {
    fingerprint = std::to_string(static_cast<unsigned long long>(st.st_size)) + "@" +
        std::to_string(static_cast<long long>(st.st_mtime));
  }
  return model_path + "|fp=" + fingerprint + "|intra=" + std::to_string(std::max(1, cfg.intra_threads)) +
      "|inter=" + std::to_string(std::max(1, cfg.inter_threads));
}
}  // namespace
#endif

OnnxBeliefEvaluator::OnnxBeliefEvaluator(
    std::string model_path, OnnxEvaluatorConfig cfg)
    : model_path_(std::move(model_path)), cfg_(cfg), ready_(false) {
#if defined(BOARD_AI_WITH_ONNX) && BOARD_AI_WITH_ONNX
  try {
    const std::string cache_key = make_belief_cache_key(model_path_, cfg_);
    std::shared_ptr<CachedBeliefSessionBundle> bundle;
    {
      auto& cache = belief_session_cache();
      std::lock_guard<std::mutex> lk(cache.mu);
      auto it = cache.map.find(cache_key);
      if (it != cache.map.end()) {
        bundle = it->second;
      }
      if (!bundle) {
        bundle = std::make_shared<CachedBeliefSessionBundle>();
        OrtEnv* raw_env = nullptr;
        throw_on_ort_error(ort_api().CreateEnv(ORT_LOGGING_LEVEL_WARNING, "dino_onnx_belief", &raw_env));
        bundle->env.reset(raw_env);

        OrtSessionOptions* raw_opts = nullptr;
        throw_on_ort_error(ort_api().CreateSessionOptions(&raw_opts));
        bundle->opts.reset(raw_opts);
        throw_on_ort_error(ort_api().SetIntraOpNumThreads(bundle->opts.get(), std::max(1, cfg_.intra_threads)));
        throw_on_ort_error(ort_api().SetInterOpNumThreads(bundle->opts.get(), std::max(1, cfg_.inter_threads)));
        throw_on_ort_error(
            ort_api().SetSessionGraphOptimizationLevel(bundle->opts.get(), GraphOptimizationLevel::ORT_ENABLE_EXTENDED));

        OrtSession* raw_session = nullptr;
#if defined(_WIN32)
        std::wstring model_path_w(model_path_.begin(), model_path_.end());
        throw_on_ort_error(ort_api().CreateSession(bundle->env.get(), model_path_w.c_str(), bundle->opts.get(), &raw_session));
#else
        throw_on_ort_error(ort_api().CreateSession(bundle->env.get(), model_path_.c_str(), bundle->opts.get(), &raw_session));
#endif
        bundle->session.reset(raw_session);

        OrtAllocator* allocator = nullptr;
        throw_on_ort_error(ort_api().GetAllocatorWithDefaultOptions(&allocator));
        bundle->input_name = get_session_name(bundle->session.get(), 0, allocator, true);
        bundle->output_name = get_session_name(bundle->session.get(), 0, allocator, false);
        cache.map[cache_key] = bundle;
        while (cache.map.size() > kMaxCachedBeliefSessions) {
          auto erase_it = cache.map.begin();
          if (erase_it == cache.map.end()) break;
          if (erase_it->first == cache_key && cache.map.size() > 1) {
            ++erase_it;
            if (erase_it == cache.map.end()) {
              erase_it = cache.map.begin();
            }
          }
          cache.map.erase(erase_it);
        }
      }
    }
    impl_ = new Impl();
    impl_->bundle = std::move(bundle);
    ready_ = true;
  } catch (const std::exception& e) {
    last_error_ = e.what();
    ready_ = false;
    throw std::runtime_error(
        std::string("OnnxBeliefEvaluator: failed to load model '") +
        model_path_ + "': " + e.what());
  }
#else
  (void)model_path_;
  (void)cfg_;
  ready_ = false;
  last_error_ = "onnx runtime not enabled at build time";
  throw std::runtime_error(
      "OnnxBeliefEvaluator: ONNX runtime not enabled at build time. "
      "Rebuild with BOARD_AI_WITH_ONNX=1 BOARD_AI_ONNXRUNTIME_ROOT=/path/to/onnxruntime");
#endif
}

OnnxBeliefEvaluator::~OnnxBeliefEvaluator() {
#if defined(BOARD_AI_WITH_ONNX) && BOARD_AI_WITH_ONNX
  delete impl_;
  impl_ = nullptr;
#endif
}

bool OnnxBeliefEvaluator::evaluate(
    const std::vector<float>& features,
    std::vector<float>* logits) const {
  if (!logits) {
    throw std::runtime_error("OnnxBeliefEvaluator::evaluate: logits pointer is null");
  }
  if (features.empty()) {
    throw std::runtime_error("OnnxBeliefEvaluator::evaluate: features is empty");
  }

#if defined(BOARD_AI_WITH_ONNX) && BOARD_AI_WITH_ONNX
  if (!ready_ || !impl_) {
    throw std::runtime_error("OnnxBeliefEvaluator::evaluate: model not loaded (ready=" +
        std::to_string(ready_) + ", last_error=" + last_error_ + ")");
  }

  try {
    std::array<int64_t, 2> shape = {1, static_cast<int64_t>(features.size())};
    OrtMemoryInfo* raw_memory_info = nullptr;
    throw_on_ort_error(ort_api().CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &raw_memory_info));
    OrtMemoryInfoPtr memory_info(raw_memory_info);

    OrtValue* raw_input = nullptr;
    throw_on_ort_error(ort_api().CreateTensorWithDataAsOrtValue(
        memory_info.get(),
        const_cast<float*>(features.data()),
        features.size() * sizeof(float),
        shape.data(),
        shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &raw_input));
    OrtValuePtr input(raw_input);

    const char* in_names[] = {impl_->bundle->input_name.c_str()};
    const char* out_names[] = {impl_->bundle->output_name.c_str()};
    const OrtValue* input_values[] = {input.get()};
    OrtValue* raw_outputs[1] = {nullptr};
    throw_on_ort_error(ort_api().Run(
        impl_->bundle->session.get(), nullptr, in_names, input_values, 1, out_names, 1, raw_outputs));
    OrtValuePtr output(raw_outputs[0]);

    OrtTensorTypeAndShapeInfo* raw_shape = nullptr;
    throw_on_ort_error(ort_api().GetTensorTypeAndShape(output.get(), &raw_shape));
    OrtTensorShapePtr out_shape(raw_shape);
    size_t dim_count = 0;
    throw_on_ort_error(ort_api().GetDimensionsCount(out_shape.get(), &dim_count));
    if (dim_count == 0) {
      throw std::runtime_error("OnnxBeliefEvaluator::evaluate: output has 0 dimensions");
    }
    std::vector<int64_t> dims(dim_count, 0);
    throw_on_ort_error(ort_api().GetDimensions(out_shape.get(), dims.data(), dims.size()));
    const int64_t out_len = dims.back();
    if (out_len <= 0 || out_len > static_cast<int64_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("OnnxBeliefEvaluator::evaluate: invalid output length: " +
          std::to_string(out_len));
    }
    void* out_raw = nullptr;
    throw_on_ort_error(ort_api().GetTensorMutableData(output.get(), &out_raw));
    const float* out_ptr = static_cast<const float*>(out_raw);
    logits->assign(out_ptr, out_ptr + out_len);
    for (float v : *logits) {
      if (!std::isfinite(v)) {
        throw std::runtime_error("OnnxBeliefEvaluator::evaluate: non-finite logit");
      }
    }
    return true;
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("OnnxBeliefEvaluator::evaluate: ") + e.what());
  }
#else
  throw std::runtime_error("OnnxBeliefEvaluator::evaluate: ONNX runtime not enabled at build time");
#endif
}

}  // namespace board_ai::infer
