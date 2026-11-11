#pragma once

#include <array>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "../../engine/core/belief_evaluator.h"
#include "../../engine/core/belief_feature_extractor.h"
#include "../../engine/core/belief_label_extractor.h"
#include "../../engine/core/belief_tracker.h"
#include "../../engine/core/feature_encoder.h"
#include "../../engine/core/game_interfaces.h"
#include "liarsdice_state.h"
#include "outcome_index.h"

namespace board_ai::liarsdice {

// Main feature encoder for policy/value network. See PLAN §1.6.
//
// Layout (per perspective):
//   own_dice_count[6]            (raw counts, 0..5)
//   current_bid_count            (scalar dense, 0..5N; 0 = no bid)
//   current_bid_face one-hot[6]  ({no-bid, 2, 3, 4, 5, 6})
//   per-seat bid history[N][5][6]
//     seat ordering: slot[k] = bids_by_player_[(perspective + k) % N]
//     within seat: 5 bids in REVERSE order (slot 0 = most recent),
//       each 6-d = X dense (1) + Y one-hot 5 (face 2..6).
//     low slots padded with all-0 6-d vector ("no bid").
template <int NPlayers>
class LiarsdiceFeatureEncoder final : public IFeatureEncoder {
 public:
  using Cfg = LiarsdiceConfig<NPlayers>;
  int action_space() const override { return Cfg::kActionSpace; }
  int feature_dim() const override { return Cfg::kFeatureDim; }

  void encode_features(
      const IGameState& masked_state,
      int perspective_player,
      const IBeliefTracker* tracker,
      std::vector<float>* out) const override;
};

// Belief tracker. Maintains the per-seat bid history (public action log)
// and caches a per-decision belief posterior over each opp's joint
// (n_1..n_6) outcome — reconstructed at root from the belief net's
// per-face marginal head (∏_f p(n_f|f), renormalized on the 252
// enumerated Σ=5 multisets). Perspective-invariant: every seat receives
// the same public bid stream and produces byte-equal tracker content
// (tested by test_tracker_perspective_invariance).
template <int NPlayers>
class LiarsdiceBeliefTracker final : public IBeliefTracker {
 public:
  using Cfg = LiarsdiceConfig<NPlayers>;

  void init(IGameState& state, int perspective,
            const AnyMap& payload) override;
  void observe_public_event(
      int actor,
      ActionId action,
      const std::vector<PublicEvent>& events) override;
  void randomize_unseen(IGameState& state, int observer,
                        std::mt19937_64& rng) const override;
  void prepare_for_root(
      const IGameState& root_state,
      int root_player,
      const IBeliefFeatureExtractor* extractor,
      const IBeliefEvaluator* evaluator) override;
  std::unique_ptr<IBeliefTracker> clone() const override {
    return std::make_unique<LiarsdiceBeliefTracker<NPlayers>>(*this);
  }
  AnyMap serialize() const override;

  // Read-only accessors for encoder / tests.
  const std::vector<std::pair<int, int>>& bids_for(int seat) const {
    return bids_by_player_[static_cast<size_t>(seat)];
  }
  int perspective() const { return perspective_; }

 private:
  int perspective_ = -1;

  // Per-seat full bid log (public). bids_by_player_[seat][i] = (X, Y).
  // Encoder reads only the last K=5 entries per seat in reverse order.
  std::array<std::vector<std::pair<int, int>>, NPlayers> bids_by_player_{};

  // Cached per-opp joint posterior over the 252 enumerated Σ=5 outcomes.
  // pi_outcome_[i] is a length-252 simplex, computed in prepare_for_root
  // by combining the belief net's 6 per-face marginals
  // (∏_f p(n_f) renormalized). randomize_unseen reads this. clone()
  // copies by value so each sim_tracker carries the cached posterior.
  std::array<std::array<float, kOutcomeCount>, NPlayers - 1> pi_outcome_{};
  bool pi_valid_ = false;
  int pi_observer_ = -1;
};

// Belief-net feature extractor. PLAN §1.5.4 / §1.6.
// Input = per-seat last-5 bid slice [N][5][6] only. NO own_dice and NO
// current_bid (the current bid is always covered by some seat's most-
// recent bid slot).
template <int NPlayers>
class LiarsdiceBeliefFeatureExtractor final : public IBeliefFeatureExtractor {
 public:
  using Cfg = LiarsdiceConfig<NPlayers>;
  static constexpr int kFeatureDim = Cfg::kBeliefFeatureDim;
  // Per-opp output: kFaceCount × (kDicePerPlayer + 1) = 6 × 6 = 36.
  // For each face f, a (kDicePerPlayer + 1)-way categorical over n_f ∈
  // [0, 5]. Joint 252-way distribution is reconstructed at sample time
  // by ∏_f p(n_f) over the 252 enumerated Σ=5 multisets.
  static constexpr int kCountBins = kDicePerPlayer + 1;
  static constexpr int kPerOppLogits = kFaceCount * kCountBins;  // 36
  static constexpr int kLogitCount = (NPlayers - 1) * kPerOppLogits;

  int feature_dim() const override { return kFeatureDim; }
  int output_logit_count() const override { return kLogitCount; }

  void extract(
      const IGameState& masked_state,
      int perspective_player,
      const IBeliefTracker* tracker,
      std::vector<float>* out) const override;
};

// Belief-net label extractor. GT-side: reads truth dice_count for each
// opponent and emits a per-face one-hot block. The framework's label
// channel only carries (hand_counts, remaining, alive_per_opp); we
// repurpose `hand_counts[i]` as a flat length-36 vector for opp i —
// 6 concatenated 6-way one-hots, slot (f * 6 + n_f) = 1. Trainer
// reshapes back to (6, 6) and runs per-face CE × 6.
// `remaining` and `alive_per_opp` are unused here — the framework
// still requires their shapes.
template <int NPlayers>
class LiarsdiceBeliefLabelExtractor final : public IBeliefLabelExtractor {
 public:
  // Flat per-face one-hot length: 6 faces * 6 count bins = 36.
  static constexpr int kCountBins = kDicePerPlayer + 1;
  static constexpr int kClassCount = kFaceCount * kCountBins;  // 36

  int label_class_count() const override { return kClassCount; }

  void extract(
      const IGameState& truth_state,
      int observer,
      std::vector<std::vector<int>>* out_hand_counts,
      std::vector<int>* out_remaining,
      std::vector<int>* out_alive) const override;
};

extern template class LiarsdiceFeatureEncoder<2>;
extern template class LiarsdiceFeatureEncoder<3>;
extern template class LiarsdiceFeatureEncoder<4>;
extern template class LiarsdiceBeliefTracker<2>;
extern template class LiarsdiceBeliefTracker<3>;
extern template class LiarsdiceBeliefTracker<4>;
extern template class LiarsdiceBeliefFeatureExtractor<2>;
extern template class LiarsdiceBeliefFeatureExtractor<3>;
extern template class LiarsdiceBeliefFeatureExtractor<4>;
extern template class LiarsdiceBeliefLabelExtractor<2>;
extern template class LiarsdiceBeliefLabelExtractor<3>;
extern template class LiarsdiceBeliefLabelExtractor<4>;

}  // namespace board_ai::liarsdice
