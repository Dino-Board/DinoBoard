#pragma once

#include <memory>
#include <random>
#include <vector>

#include "azul_state.h"
#include "../../engine/core/belief_tracker.h"
#include "../../engine/core/feature_encoder.h"
#include "../../engine/core/game_interfaces.h"

namespace board_ai::azul {

// Azul has fully-public state — every game-facing field is all_public,
// including bag and box_lid (declared in schema as per-color counts). No
// per-player hidden information; no belief tracker is registered
// (decision-side determinization at MCTS sim entry has nothing to fill).
template <int NPlayers>
class AzulFeatureEncoder final : public IFeatureEncoder {
 public:
  using Cfg = AzulConfig<NPlayers>;
  int action_space() const override { return Cfg::kActionSpace; }
  int feature_dim() const override { return Cfg::kFeatureDim; }

  void encode_features(
      const IGameState& state,
      int perspective_player,
      const IBeliefTracker* tracker,
      std::vector<float>* out) const override;
};

// Token encoder for the set-style (transformer) network; see TokenLayout.
// Same rules and action ids as AzulFeatureEncoder. Slot order (player-indexed
// blocks in perspective-relative order):
//   global(1) | players(P) | pattern rows(5P, row-major per player) |
//   piles((F + 1) * 5: source-major, the centre last)
// A pile is all tiles of one colour on one factory or in the centre, i.e.
// what one action takes; pointer head t on it scores "take this pile to
// target t" (pattern line 0..4, floor). No token says which factory it is
// (a pile only lists the other tiles lying on its factory), so a network
// without positional embeddings gives the same policy and value however the
// factories are ordered.
template <int NPlayers>
class AzulTokenEncoder final : public IFeatureEncoder {
 public:
  using Cfg = AzulConfig<NPlayers>;

  enum TokenType { kGlobalToken = 0, kPlayerToken, kRowToken, kPileToken, kTokenTypeCount };

  static constexpr int kSeatSlots = 4;
  static constexpr int kGlobalFeatures = 16;
  static constexpr int kPlayerFeatures = 28;
  static constexpr int kRowFeatures = 33;
  static constexpr int kPileFeatures = 41;
  static constexpr int kTokenDim = 2 + kPileFeatures;

  static constexpr int kPlayerBase = 1;
  static constexpr int kRowBase = kPlayerBase + Cfg::kPlayers;
  static constexpr int kPileBase = kRowBase + Cfg::kPlayers * kRows;
  static constexpr int kNumTokens = kPileBase + (Cfg::kFactories + 1) * kColors;
  static constexpr int kFeatureDim = kNumTokens * kTokenDim;

  AzulTokenEncoder();

  int action_space() const override { return Cfg::kActionSpace; }
  int feature_dim() const override { return kFeatureDim; }
  const TokenLayout* token_layout() const override { return &layout_; }

  void encode_features(
      const IGameState& state,
      int perspective_player,
      const IBeliefTracker* tracker,
      std::vector<float>* out) const override;

 private:
  TokenLayout layout_;
};

extern template class AzulFeatureEncoder<2>;
extern template class AzulFeatureEncoder<3>;
extern template class AzulFeatureEncoder<4>;
extern template class AzulTokenEncoder<2>;
extern template class AzulTokenEncoder<3>;
extern template class AzulTokenEncoder<4>;

}  // namespace board_ai::azul
