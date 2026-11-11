#pragma once

#include "../../engine/core/feature_encoder.h"
#include "carcassonne_state.h"

namespace board_ai::carcassonne {

class CarcassonneFeatureEncoder final : public IFeatureEncoder {
 public:
  int action_space() const override { return kActionSpace; }
  int feature_dim() const override { return kFeatureDim; }

  void encode_features(const IGameState& masked_state, int perspective_player,
                       const IBeliefTracker* tracker,
                       std::vector<float>* out) const override;
};

// Token encoder for the transformer network (`carcassonne_tok`); see
// TokenLayout. Variable-length: only the tokens present, in the order
//   global | current tile | placed tiles | frontier cells
// each with its type and board position (x = column, y = row) for the
// network's relative position bias (TokenLayout::token_positions).
// Placed-tile and frontier tokens are anchored at their board cell (anchor
// id = cell + 1) and listed in cell order. The current tile is anchored at
// kBoardCells + 1 so its meeple heads bind by anchor, and positioned at the
// tile just placed during the meeple stage. Absolute coordinate features
// are zeroed: only relative positions count.
// City / road / farm connectivity is precomputed: every token that
// touches a feature carries that whole feature's size, open edges, shields
// and per-player meeple counts. In the placement stage the current-tile
// token spells out the tile's geometry and every frontier token says, per
// legal rotation, what placing the tile there would do (computed by
// CarcassonneRules::preview_placement).
// Policy: the four rotation heads on a frontier token score placing the
// current tile there (via anchor_action); the meeple heads on the
// current-tile token score the 13 meeple positions; "no meeple" is global.
// Tokens are first written into fixed slots (kNumTokens x kTokenDim, absent
// ones all-zero), then the valid ones are compacted into the output.
class CarcassonneTokenEncoder final : public IFeatureEncoder {
 public:
  enum TokenType { kGlobalToken = 0, kCurrentToken, kTileToken, kFrontierToken, kTokenTypeCount };
  // Heads [0, 4): place at rotation r. Heads [4, 17): meeple position p.
  static constexpr int kMeepleHeadBase = 4;
  static constexpr int kPointerHeadCount = kMeepleHeadBase + kMeeplePositions;

  static constexpr int kMaxPlacedTiles = kDeckSlots + 1;
  // A polyomino of n cells has at most 2n + 2 empty 4-neighbours.
  static constexpr int kMaxFrontierCells = 2 * kMaxPlacedTiles + 2;

  static constexpr int kGlobalFeatures = 37;
  static constexpr int kMeepleSlotFeatures = 6;
  static constexpr int kCurrentFeatures = 9 + kMeeplePositions * kMeepleSlotFeatures;
  static constexpr int kTileFeatures = 77;
  static constexpr int kFrontierFeatures = 85;
  static constexpr int kTokenFeatures = kCurrentFeatures;
  static constexpr int kTokenDim = 3 + kTokenFeatures;

  static constexpr int kCurrentSlot = 1;
  static constexpr int kTileBase = 2;
  static constexpr int kFrontierBase = kTileBase + kMaxPlacedTiles;
  static constexpr int kNumTokens = kFrontierBase + kMaxFrontierCells;

  static constexpr int kVarHeader = 7;
  static constexpr int kVarTokenDim = kVarHeader + kTokenFeatures;
  static constexpr int kCurrentAnchor = kBoardCells + 1;

  CarcassonneTokenEncoder();

  int action_space() const override { return kActionSpace; }
  int feature_dim() const override { return 0; }
  const TokenLayout* token_layout() const override { return &layout_; }

  void encode_features(const IGameState& masked_state, int perspective_player,
                       const IBeliefTracker* tracker,
                       std::vector<float>* out) const override;

 private:
  void encode_slots(const IGameState& masked_state, int perspective_player, std::vector<float>* out) const;
  void compact(const IGameState& masked_state, const float* slots, std::vector<float>* out) const;

  TokenLayout layout_;
};

}  // namespace board_ai::carcassonne
