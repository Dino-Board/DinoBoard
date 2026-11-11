#pragma once

#include <vector>

#include "../../engine/core/feature_encoder.h"
#include "../../engine/core/game_interfaces.h"
#include "tictactoe_state.h"

namespace board_ai::tictactoe {

constexpr int kActionSpace = 9;
constexpr int kFeatureDim = 28;

// TicTacToe is fully observable — no perspective-private fields.
class TicTacToeFeatureEncoder final : public IFeatureEncoder {
 public:
  int action_space() const override { return kActionSpace; }
  int feature_dim() const override { return kFeatureDim; }

  void encode_features(
      const IGameState& state,
      int perspective_player,
      const IBeliefTracker* tracker,
      std::vector<float>* out) const override;
};

}  // namespace board_ai::tictactoe
