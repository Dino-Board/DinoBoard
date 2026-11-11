#pragma once

#include <array>
#include <memory>
#include <random>
#include <vector>

#include "../../engine/core/belief_evaluator.h"
#include "../../engine/core/belief_feature_extractor.h"
#include "../../engine/core/belief_label_extractor.h"
#include "../../engine/core/belief_tracker.h"
#include "../../engine/core/feature_encoder.h"
#include "../../engine/core/game_interfaces.h"
#include "6nimmit_state.h"

namespace board_ai::sixnimmt {

template <int NPlayers>
class SixNimmtFeatureEncoder final : public IFeatureEncoder {
 public:
  using Cfg = SixNimmtConfig<NPlayers>;
  SixNimmtFeatureEncoder() = default;
  int action_space() const override { return kActionSpace; }
  int feature_dim() const override { return Cfg::kFeatureDim; }

  void encode_features(
      const IGameState& masked_state,
      int perspective_player,
      const IBeliefTracker* tracker,
      std::vector<float>* out) const override;
};

// Belief tracker. Maintains per-player public play history (`played_by_`)
// from the events stream — every `card_played` event flips the
// corresponding bit. Selfplay / web / API all run the same wholesale
// public_snapshot protocol on `state` (table / bullhead_pile / committed /
// pending_commits-after-seal); the tracker only adds the per-card
// "who played this" channel that the public snapshot doesn't directly
// expose (bullhead_pile is keyed by eater, not original player).
//
// `randomize_unseen` derives the unseen pool as
//   {1..104} \ own_hand \ ⋃_p played_by_[p]
// and either fills opp hands uniformly (no belief net) or via the
// belief-net posterior (Plackett-Luce sampling, future Step 3).
//
// init() bootstraps the perspective's own hand into state via the
// "own_hand" payload key — the API path needs this because its session
// was seeded independently from truth. Selfplay / arena seed sessions
// from the same RNG, so init() with empty payload is a no-op there
// (own hand already present).
template <int NPlayers>
class SixNimmtBeliefTracker final : public IBeliefTracker {
 public:
  using Cfg = SixNimmtConfig<NPlayers>;

  void init(IGameState& state, int perspective,
            const AnyMap& payload) override;
  AnyMap pack_init_payload(const IGameState& gt_state,
                           int perspective) const override;
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
    return std::make_unique<SixNimmtBeliefTracker<NPlayers>>(*this);
  }
  AnyMap serialize() const override;

  // Read-only accessor for the belief-net feature extractor.
  // played_by_[p][c-1] == 1 iff the public event stream has reported
  // player p playing card c this game.
  bool played_by(int p, int card_1based) const {
    if (p < 0 || p >= NPlayers) return false;
    if (card_1based < 1 || card_1based > kNumCards) return false;
    return played_by_[static_cast<size_t>(p)]
                     [static_cast<size_t>(card_1based - 1)] != 0;
  }

 private:
  // Public play history. Updated by `observe_public_event` from
  // `card_played` events; cloned by value for sim trackers.
  std::array<std::array<std::int8_t, kNumCards>, NPlayers> played_by_{};

  // Cached per-opp marginal posterior over `1..104`. Dimension
  // (NPlayers - 1) × kNumCards. belief_probs_[i][c-1] = P(opp_i holds
  // card c) under the belief-net's sigmoid output (PLAN §3.4). Computed
  // by `prepare_for_root`; consumed by `randomize_unseen`'s
  // Plackett-Luce sampler. clone() copies by value so each sim_tracker
  // carries the cached posterior.
  std::array<std::array<float, kNumCards>, (NPlayers > 1 ? NPlayers - 1 : 1)>
      belief_probs_{};
  bool pi_valid_ = false;
  int pi_observer_ = -1;

};

// Belief-net feature extractor. PLAN §3.1.
// Input layout (per perspective):
//   [A] unseen_pool         : 104 — `1..104 \ own_hand \ ⋃_p played_by_
//                                    \ table \ bullhead piles \ visible
//                                    pending` (never a public card)
//   [B] own_played          : 104 — `played_by_[perspective]`
//   [C] opp_played_union    : 104 — `⋃_{p != perspective} played_by_[p]`
//   [D] table dense         :  12 — 4 row-tail card_ids + 4 row lengths +
//                                    4 row heads totals (normalized)
//   [E] bullhead_counts     :   N — per-player accumulated heads,
//                                    rotated own-first, normalized by 30
//   [F] meta                :   1 + N — small_phase_active + small_player
//                                       one-hot rotated own-first
//   = 312 + 12 + N + 1 + N = 325 + 2N
// Per-opp output: 104 sigmoid logits (binary "opp holds card c" probability).
// `output_logit_count = (NPlayers - 1) * 104`.
template <int NPlayers>
class SixNimmtBeliefFeatureExtractor final : public IBeliefFeatureExtractor {
 public:
  using Cfg = SixNimmtConfig<NPlayers>;
  static constexpr int kFeatureDim = 325 + 2 * NPlayers;
  static constexpr int kPerOppLogits = kNumCards;  // 104
  static constexpr int kLogitCount = (NPlayers - 1) * kPerOppLogits;

  int feature_dim() const override { return kFeatureDim; }
  int output_logit_count() const override { return kLogitCount; }

  void extract(
      const IGameState& masked_state,
      int perspective_player,
      const IBeliefTracker* tracker,
      std::vector<float>* out) const override;
};

// Belief-net label extractor. PLAN §3.2.
// GT-side: per opp emits a 104-dim multi-hot of opp's CURRENT held cards
// (hand[opp] ∪ {pending_commits[opp] if non-zero}). The framework's label
// channel `hand_counts[opp_idx]` is repurposed as the multi-hot vector;
// `remaining` and `alive_per_opp` are filled with placeholder/all-ones to
// satisfy the framework's shape contract (training uses BCE-with-logits
// directly on `hand_counts`).
template <int NPlayers>
class SixNimmtBeliefLabelExtractor final : public IBeliefLabelExtractor {
 public:
  static constexpr int kClassCount = kNumCards;  // 104

  int label_class_count() const override { return kClassCount; }

  void extract(
      const IGameState& truth_state,
      int observer,
      std::vector<std::vector<int>>* out_hand_counts,
      std::vector<int>* out_remaining,
      std::vector<int>* out_alive) const override;
};

extern template class SixNimmtFeatureEncoder<2>;
extern template class SixNimmtFeatureEncoder<3>;
extern template class SixNimmtFeatureEncoder<4>;
extern template class SixNimmtBeliefTracker<2>;
extern template class SixNimmtBeliefTracker<3>;
extern template class SixNimmtBeliefTracker<4>;
extern template class SixNimmtBeliefFeatureExtractor<2>;
extern template class SixNimmtBeliefFeatureExtractor<3>;
extern template class SixNimmtBeliefFeatureExtractor<4>;
extern template class SixNimmtBeliefLabelExtractor<2>;
extern template class SixNimmtBeliefLabelExtractor<3>;
extern template class SixNimmtBeliefLabelExtractor<4>;

}  // namespace board_ai::sixnimmt
