#pragma once

#include <array>
#include <memory>
#include <random>
#include <vector>

#include "../../engine/core/belief_feature_extractor.h"
#include "../../engine/core/belief_tracker.h"
#include "../../engine/core/feature_encoder.h"
#include "../../engine/core/game_interfaces.h"
#include "loveletter_state.h"

namespace board_ai::loveletter {

template <int NPlayers>
class LoveLetterFeatureEncoder final : public IFeatureEncoder {
 public:
  using Cfg = LoveLetterConfig<NPlayers>;
  LoveLetterFeatureEncoder() = default;
  int action_space() const override { return kActionSpace; }
  int feature_dim() const override { return Cfg::kFeatureDim; }

  void encode_features(
      const IGameState& state,
      int perspective_player,
      const IBeliefTracker* tracker,
      std::vector<float>* out) const override;
};

// LL belief tracker is perspective-agnostic and stateless after §G.
// All per-perspective knowledge lives in state.viz_["hand"] /
// state.viz_["drawn_card"] (rules are the sole writer, see
// loveletter_rules.cpp). The tracker exists only to provide
// `randomize_unseen`, which fills viz=0 slots from the public deck-
// multiset implied by state's discard_piles + face_up_removed +
// observer's viz=1 hand/drawn cards. There are no private fields.
template <int NPlayers>
class LoveLetterBeliefTracker final : public IBeliefTracker {
 public:
  using Cfg = LoveLetterConfig<NPlayers>;

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
  std::unique_ptr<IBeliefTracker> clone() const override {
    return std::make_unique<LoveLetterBeliefTracker<NPlayers>>(*this);
  }
  AnyMap serialize() const override;
};

// Minimal Phase-1 carrier extractor for the belief network pipeline.
// Inputs: per-player public discard counts (N × 8) + own viz=1 hand
// one-hot (8) + alive flags (N). Output: (N-1) × kCardTypes opp-role
// logits. The actual LL `randomize_unseen` is uniform-multiset and
// does not consult the resulting pi posterior — this extractor exists
// to exercise the engineering pipeline (extract → ONNX → cache pi)
// before Coup needs it. The structural barrier is identical to
// IFeatureEncoder: reads a perspective-masked state only, viz=0 slots
// arrive as kPlaceholder.
template <int NPlayers>
class LoveLetterBeliefFeatureExtractor final : public IBeliefFeatureExtractor {
 public:
  using Cfg = LoveLetterConfig<NPlayers>;
  static constexpr int kFeatureDim = NPlayers * kCardTypes + kCardTypes + NPlayers;
  static constexpr int kLogitCount = (NPlayers - 1) * kCardTypes;

  int feature_dim() const override { return kFeatureDim; }
  int output_logit_count() const override { return kLogitCount; }

  void extract(
      const IGameState& masked_state,
      int perspective_player,
      const IBeliefTracker* tracker,
      std::vector<float>* out) const override;
};

extern template class LoveLetterFeatureEncoder<2>;
extern template class LoveLetterFeatureEncoder<3>;
extern template class LoveLetterFeatureEncoder<4>;
extern template class LoveLetterBeliefTracker<2>;
extern template class LoveLetterBeliefTracker<3>;
extern template class LoveLetterBeliefTracker<4>;
extern template class LoveLetterBeliefFeatureExtractor<2>;
extern template class LoveLetterBeliefFeatureExtractor<3>;
extern template class LoveLetterBeliefFeatureExtractor<4>;

}  // namespace board_ai::loveletter
