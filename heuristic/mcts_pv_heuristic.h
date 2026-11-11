#pragma once

#include <memory>
#include <string>

#include "../engine/core/heuristic.h"
#include "../engine/search/net_mcts.h"

namespace board_ai {
class IFeatureEncoder;
}

namespace board_ai::infer {
class OnnxPolicyValueEvaluator;
}

namespace board_ai::heuristic {

// PV-MCTS heuristic. Wraps a `search::IPolicyValueEvaluator` plus a
// fresh `NetMcts` per call, returning visit counts as scores.
//
// Two construction paths:
//   - make_from_evaluator: borrows an evaluator owned elsewhere
//     (typically `GameBundle::heuristic_pv_evaluators`). Used by
//     hand-written PV heuristics like Quoridor's rule-based PV.
//   - make_from_onnx:      owns an OnnxPolicyValueEvaluator built from
//     model_path. Used by ONNX-backed PV heuristics (e.g. Splendor's
//     `pv_512x4`).
//
// `pick(...)` builds a NetMctsConfig{simulations, c_puct, ...},
// runs `NetMcts::search_root(masked_state, rules, value_model,
// evaluator)`, and returns `(root_actions, root_action_visits)` as
// `(actions, scores)`. The masked_state's value_model dependency is
// resolved by reading `masked_state.value_model_ref()` — wait, no:
// IGameState doesn't expose one. Instead we require the caller to
// supply a value_model via the heuristic's stored back-pointer to
// the GameBundle's `IStateValueModel`. Concretely the construction
// path also takes a `value_model` pointer.
class MctsPVHeuristic final : public IHeuristic {
 public:
  // Flavor B: borrowed evaluator. evaluator + value_model must outlive
  // this heuristic instance (true when both live in the GameBundle that
  // also owns this heuristic).
  static std::unique_ptr<IHeuristic> make_from_evaluator(
      std::string name,
      const search::IPolicyValueEvaluator* evaluator,
      const IStateValueModel* value_model,
      int simulations,
      float c_puct,
      int world_cap = 0,
      int leaf_batch_size = 1);

  // Flavor C: ONNX-owned. The heuristic owns the OnnxPolicyValueEvaluator;
  // encoder + value_model must outlive this heuristic (always the case —
  // bundle owns them).
  static std::unique_ptr<IHeuristic> make_from_onnx(
      std::string name,
      const std::string& model_path,
      const IFeatureEncoder* encoder,
      const IStateValueModel* value_model,
      int simulations,
      float c_puct,
      int world_cap = 0,
      int leaf_batch_size = 1);

  ~MctsPVHeuristic() override;

  std::string name() const override { return name_; }

  HeuristicResult pick(IGameState& masked_state,
                       const IBeliefTracker* tracker,
                       const IGameRules& rules,
                       std::uint64_t rng_seed) const override;

  // The search configuration pick() runs with (the per-call tracker aside).
  // Single source for pick() and for tools that report how strong this
  // heuristic is as a benchmark opponent (dinoboard_engine.heuristic_settings).
  // Knobs heuristic.json does not carry are fixed here, not inherited from
  // the caller: notably uct2_arrival_visits stays false (see the .cpp).
  search::NetMctsConfig search_config() const;

  // Hot-swap the evaluator (e.g. selfplay step entry replaces the
  // bundled per-game OnnxPolicyValueEvaluator with a shared
  // BatchedPolicyValueEvaluator from EvaluatorPool so all sampling
  // threads' guidance MCTS calls batch together). Caller owns the
  // replacement evaluator; it must outlive this heuristic. The
  // previously-owned `owned_evaluator_` (if any) is dropped to release
  // the old per-bundle ONNX session.
  //
  // Used only by the concurrent selfplay/eval entry; legacy multi-process
  // path leaves the bundled evaluator in place.
  void replace_evaluator(const search::IPolicyValueEvaluator* evaluator);

  // Lookup hook so the concurrent runner can identify which model_path
  // this heuristic was originally built against (so EvaluatorPool can
  // dedup-load it and inject the shared pointer).
  const std::string& model_path_hint() const { return model_path_hint_; }
  // Encoder the ONNX model was built against (flavor C only; null for
  // flavor B). A replacement evaluator for `model_path_hint()` must use
  // this encoder, not the bundle's, since they can differ in layout.
  const IFeatureEncoder* encoder_hint() const { return encoder_hint_; }
  // Concurrent callers size the shared evaluator queue from this: each
  // thread in pick() can have this many requests in flight.
  int leaf_batch_size() const { return leaf_batch_size_; }

 private:
  MctsPVHeuristic(std::string name,
                  const search::IPolicyValueEvaluator* evaluator,
                  std::unique_ptr<infer::OnnxPolicyValueEvaluator> owned,
                  const IStateValueModel* value_model,
                  int simulations,
                  float c_puct,
                  int world_cap,
                  int leaf_batch_size);

  std::string name_;
  // Always non-null. Either points into `owned_evaluator_` (flavor C) or
  // to a borrowed evaluator owned by the GameBundle (flavor B). Mutable
  // so `replace_evaluator()` can swap it without breaking the public
  // const-correct API of pick().
  mutable const search::IPolicyValueEvaluator* evaluator_;
  std::unique_ptr<infer::OnnxPolicyValueEvaluator> owned_evaluator_;
  const IStateValueModel* value_model_;
  int simulations_;
  float c_puct_;
  int world_cap_;
  int leaf_batch_size_;
  // Original ONNX path that built `owned_evaluator_` (flavor C only;
  // empty for flavor B). Lets the concurrent runner identify which
  // batched evaluator to inject.
  std::string model_path_hint_;
  const IFeatureEncoder* encoder_hint_ = nullptr;
};

}  // namespace board_ai::heuristic
