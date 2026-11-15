#pragma once

#include <string>
#include <vector>

#include "../../engine/core/feature_encoder.h"
#include "6nimmit_state.h"

namespace board_ai::sixnimmt {

// ---- Token encoder (transformer network), all player counts ----
//
// Variable-length tokens with positions (TokenLayout::variable_tokens,
// token_positions); header [valid, entity, anchor, type, has position, x, y].
// "Me" = the perspective (the player to decide when the policy is queried).
//
// No token says which of the 104 cards it is. A card is its number and its
// bullheads, and what counts is where it stands relative to the row ends and
// to the cards that can still turn up. "Unseen" = every card that has not
// been shown and is not in my hand: the other players' hands, their
// face-down cards and the rest of the deck.
//   global     round, what is being decided, how many unseen cards there are
//              and what share of them the other players hold
//   player     one per player (me first, then the others in seat order):
//              bullheads taken, and the margin they make so far
//   row   x4   its end card, length, bullheads, the unseen / my / face-up
//              cards that would land on it, and, when a row is being chosen,
//              how the rest of the round plays out if this one is taken
//   card       one per card in my hand and per face-up played card still to
//              be placed: its number, bullheads, the row it would go to, and
//              what lies between that row's end and the card
// Position of a row or card token: x = unseen cards below its (end) card,
// y = row ends below it, counting a row's own end. So for a card and a row
// dy = 0 means "this card goes on this row" and dx is the number of unseen
// cards that could still get in between; the network sees only these
// offsets (relative position bias), never a card's place on the 1..104 line.
// Pointer heads (2): `play` on a hand card plays it, `take` on a row takes
// it. Every action is bound by anchor; the network must run with
// `anchor_embedding: false`, or the anchor id would be a card id again.
// The layout is the same for two, three and four players. The feature list
// is the tables below, exported by layout_json() and documented in
// docs/games/6nimmit_api.md.

#define SIXNIMMT_GLOBAL_FEATURES(X)                                                       \
  X(round, 1) X(choosing_row, 1) X(to_move, 1) X(unseen, 1) X(opp_cards, 1) X(density, 1) \
  X(low_unseen, 1) X(low_expected, 1) X(min_row_heads, 1)

#define SIXNIMMT_PLAYER_FEATURES(X)                                                       \
  X(seat, 4) X(hand_heads, 1) X(margin, 1) X(shown, 1) X(chooser, 1)

#define SIXNIMMT_ROW_FEATURES(X)                                                          \
  X(tail, 1) X(length, 5) X(space, 1) X(heads, 1) X(cheapest, 1) X(order, 4)              \
  X(reach_unseen, 1) X(reach_expected, 1) X(reach_heads, 1) X(reach_mine, 1)              \
  X(reach_shown, 1) X(if_taken_mine, 1) X(if_taken_others, 1)

#define SIXNIMMT_CARD_FEATURES(X)                                                         \
  X(in_hand, 1) X(my_shown, 1) X(opp_shown, 1) X(owner, 4) X(resolving, 1) X(value, 1)    \
  X(heads, 1) X(hand_rank, 1) X(percentile, 1) X(below_expected, 1) X(small, 1)           \
  X(small_cost, 1) X(gap_safe, 1) X(gap_unseen, 1) X(gap_expected, 1) X(gap_mine, 1)      \
  X(gap_shown, 1) X(row_length, 5) X(row_space, 1) X(row_heads, 1) X(sixth, 1)            \
  X(cost_now, 1)

#define SIXNIMMT_FEATURE_ENUM(name, width) name, name##_last = name + (width)-1,
namespace sgf { enum : int { SIXNIMMT_GLOBAL_FEATURES(SIXNIMMT_FEATURE_ENUM) kCount }; }
namespace spf { enum : int { SIXNIMMT_PLAYER_FEATURES(SIXNIMMT_FEATURE_ENUM) kCount }; }
namespace srf { enum : int { SIXNIMMT_ROW_FEATURES(SIXNIMMT_FEATURE_ENUM) kCount }; }
namespace scf { enum : int { SIXNIMMT_CARD_FEATURES(SIXNIMMT_FEATURE_ENUM) kCount }; }
#undef SIXNIMMT_FEATURE_ENUM

template <int NPlayers>
class SixNimmtTokenEncoder final : public IFeatureEncoder {
 public:
  enum TokenType { kGlobalToken, kPlayerToken, kRowToken, kCardToken, kTokenTypeCount };
  enum PointerHead { kPlayHead, kTakeHead, kNumHeads };

  static constexpr int kHeader = 7;
  static constexpr int kFeatures = scf::kCount;  // the widest token type
  static constexpr int kTokenDim = kHeader + kFeatures;

  // Anchor ids are 1 + index; the bases below are indices.
  static constexpr int kAnchorCard = 0;                      // + card - 1 (hand cards only)
  static constexpr int kAnchorRow = kAnchorCard + kNumCards;  // + row
  static constexpr int kNumAnchors = kAnchorRow + kNumRows;

  SixNimmtTokenEncoder();

  int action_space() const override { return kActionSpace; }
  int feature_dim() const override { return 0; }
  const TokenLayout* token_layout() const override { return &layout_; }

  void encode_features(const IGameState& masked_state, int perspective_player,
                       const IBeliefTracker* tracker, std::vector<float>* out) const override;

  // The layout as JSON: header, per token type its features (name, offset,
  // width), pointer heads and anchor ranges. For tests and documentation.
  static std::string layout_json();

 private:
  TokenLayout layout_;
};

extern template class SixNimmtTokenEncoder<2>;
extern template class SixNimmtTokenEncoder<3>;
extern template class SixNimmtTokenEncoder<4>;

}  // namespace board_ai::sixnimmt
