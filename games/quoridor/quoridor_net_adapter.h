#pragma once

#include <vector>

#include "../../engine/core/feature_encoder.h"
#include "../../engine/core/game_interfaces.h"
#include "quoridor_state.h"

namespace board_ai::quoridor {

constexpr int kFeatureDim = 295;

// Quoridor is fully observable.
class QuoridorFeatureEncoder final : public IFeatureEncoder {
 public:
  int action_space() const override { return kActionSpace; }
  int feature_dim() const override { return kFeatureDim; }

  void encode_features(
      const IGameState& state,
      int perspective_player,
      const IBeliefTracker* tracker,
      std::vector<float>* out) const override;
};

}  // namespace board_ai::quoridor
