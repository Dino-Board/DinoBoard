#pragma once

#include <string>
#include <vector>

#include "../core/belief_evaluator.h"
#include "onnx_policy_value_evaluator.h"

namespace board_ai::infer {

// ONNX-backed implementation of IBeliefEvaluator. Single-input, single-
// output: input is a flat float feature vector (produced by the game's
// IBeliefFeatureExtractor), output is a flat (N-1) × K logit tensor.
// An autoregressive model has the inputs "features" [1, F] and "prefix"
// [rows, P] instead and is run with evaluate_steps.
//
// The belief net is independent of the policy/value net — separate ONNX
// session, separate cache slot. Logits are returned uncalibrated; the
// tracker applies the per-game softmax temperature.
class OnnxBeliefEvaluator final : public IBeliefEvaluator {
 public:
  OnnxBeliefEvaluator(
      std::string model_path,
      OnnxEvaluatorConfig cfg = {});
  ~OnnxBeliefEvaluator();

  OnnxBeliefEvaluator(const OnnxBeliefEvaluator&) = delete;
  OnnxBeliefEvaluator& operator=(const OnnxBeliefEvaluator&) = delete;

  bool is_ready() const { return ready_; }
  const std::string& last_error() const { return last_error_; }

  bool evaluate(
      const std::vector<float>& features,
      std::vector<float>* logits) const override;
  void evaluate_steps(
      const std::vector<float>& features,
      const std::vector<float>& prefixes,
      int rows,
      std::vector<float>* logits) const override;

 private:
  // One session run. `prefixes` null: the single-input model on one row of
  // features; else the autoregressive model on `rows` prefixes.
  void run(const std::vector<float>& features, const std::vector<float>* prefixes, int rows,
           std::vector<float>* logits) const;

  std::string model_path_;
  OnnxEvaluatorConfig cfg_{};
  bool ready_ = false;
  std::string last_error_{};

#if defined(BOARD_AI_WITH_ONNX) && BOARD_AI_WITH_ONNX
  struct Impl;
  Impl* impl_ = nullptr;
#endif
};

}  // namespace board_ai::infer
