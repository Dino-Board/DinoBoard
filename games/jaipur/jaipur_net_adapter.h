#pragma once
#include <vector>
#include "../../engine/core/feature_encoder.h"
#include "../../engine/core/game_interfaces.h"
#include "../../engine/core/game_registry.h"
#include "jaipur_state.h"
#include "jaipur_rules.h"

namespace board_ai::jaipur {

// Token encoder for the set-style (transformer) network; see TokenLayout.
// Goods cards of one type are interchangeable, so tokens stand for card
// TYPES, not cards:
//   0..5  one token per goods type (entity id 1 + g)
//   6     camels (entity 7)
//   7..9  bonus-token tiers 3 / 4 / 5+ (entity 8..10)
//   10    global (entity 11)
// Everything is perspective-relative ("me" = perspective, "opp" = the other
// seat). The six goods tokens share one projection and differ only by their
// entity embedding, so what the network learns about one type (when a set
// is worth selling, when to grab a card) transfers to the others.
//
// Every action is scored by a pointer head on the token it concerns: take /
// sell k / trade-take k / trade-give k on the goods token of its type, take
// camels and give k camels on the camel token. Sale size, trade size and
// type are thus factored into (token, head) instead of 114 unrelated logits.
class JaipurTokenEncoder final : public IFeatureEncoder {
 public:
  enum TokenType { kGoodToken = 0, kCamelToken, kBonusToken, kGlobalToken, kTokenTypeCount };

  static constexpr int kGoodBase = 0;
  static constexpr int kCamelSlot = kGoodTypes;
  static constexpr int kBonusBase = kCamelSlot + 1;
  static constexpr int kGlobalSlot = kBonusBase + kBonusTiers;
  static constexpr int kNumTokens = kGlobalSlot + 1;  // 11

  static constexpr int kGoodFeatures = 20;
  static constexpr int kCamelFeatures = 8;
  static constexpr int kBonusFeatures = 6;
  static constexpr int kGlobalFeatures = 20;
  static constexpr int kTokenDim = 2 + kGoodFeatures;
  static constexpr int kFeatureDim = kNumTokens * kTokenDim;  // 242

  // Pointer heads: take, sell k (k = 1..7), trade-take k, trade-give k (1..5).
  static constexpr int kTakeHead = 0;
  static constexpr int kSellHead = 1;                         // + (k - 1)
  static constexpr int kTradeTakeHead = kSellHead + kMaxSell;  // + (k - 1)
  static constexpr int kTradeGiveHead = kTradeTakeHead + kMaxTrade;
  static constexpr int kPointerHeadCount = kTradeGiveHead + kMaxTrade;  // 18

  JaipurTokenEncoder();

  int action_space() const override { return kActionSpace; }
  int feature_dim() const override { return kFeatureDim; }
  const TokenLayout* token_layout() const override { return &layout_; }

  void encode_features(const IGameState& state, int perspective_player,
                       const IBeliefTracker* tracker,
                       std::vector<float>* out) const override;

 private:
  TokenLayout layout_;
};

StateSerializer make_jaipur_serializer();
ActionDescriptor make_jaipur_descriptor();

}  // namespace board_ai::jaipur
