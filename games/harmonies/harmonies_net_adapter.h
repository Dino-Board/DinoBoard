#pragma once

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "../../engine/core/belief_tracker.h"
#include "../../engine/core/feature_encoder.h"
#include "../../engine/core/game_interfaces.h"
#include "harmonies_state.h"

namespace board_ai::harmonies {

// Feature layout: per-player public slice (rotated so perspective is first)
// + global state. All values are public (no hidden info).
template <int NPlayers, int Side>
struct HarmoniesFeatureSpec {
  static constexpr int kHexes = Board<Side>::kHexes;
  static constexpr int kTowerTypeBuckets = kTowerTypes + 1;  // includes "empty"
  static constexpr int kPerPlayer =
      kHexes * kTowerTypeBuckets +     // tower one-hot
      kHexes +                          // has_animal flags
      kHandSize * (kAnimalCards + 1) +  // held_cards one-hot (0 = empty slot)
      kHandSize +                       // cube_remaining / cube_initial ratio
      kHandSize +                       // cube_initial / max
      1 + 1 + 1;                        // space, animal_score, landscape_score
  static constexpr int kGlobal =
      kSlots * kColors +                // central_board_counts (normalized)
      kColors +                         // selected_tokens
      kSlots * (kAnimalCards + 1) +     // animal_market one-hot
      kColors +                         // pouch_counts (normalized)
      3 +                               // turn_started/took_card/end_triggered
      4;                                // current_player one-hot (max 4)
  static constexpr int kFeatureDim = NPlayers * kPerPlayer + kGlobal;
};

// Hex-indexed actions under board symmetry g: place token / place cube move
// with their hex, everything else stays. Shared by both encoders.
template <int Side>
std::vector<int> board_symmetry_action_map(int g);

template <int NPlayers, int Side>
class HarmoniesFeatureEncoder final : public IFeatureEncoder {
 public:
  int action_space() const override { return Board<Side>::kActionSpace; }
  int feature_dim() const override {
    return HarmoniesFeatureSpec<NPlayers, Side>::kFeatureDim;
  }
  void encode_features(const IGameState& state,
                       int perspective_player,
                       const IBeliefTracker* tracker,
                       std::vector<float>* out) const override;
  // The board's symmetry group (harmonies_board_data.h) acting on the hex
  // blocks of every player slice.
  std::vector<EncoderSymmetry> symmetries() const override;
};

#define HARMONIES_EXTERN_HarmoniesFeatureEncoder(N, S) extern template class HarmoniesFeatureEncoder<N, S>;
HARMONIES_FOR_EACH_VARIANT(HARMONIES_EXTERN_HarmoniesFeatureEncoder)
#undef HARMONIES_EXTERN_HarmoniesFeatureEncoder

template <int NPlayers, int Side>
class HarmoniesBeliefTracker final : public IBeliefTracker {
 public:
  void init(IGameState& state, int perspective,
            const AnyMap& payload) override;
  void observe_public_event(
      int actor, ActionId action,
      const std::vector<PublicEvent>& events) override;
  void randomize_unseen(IGameState& state, int observer,
                        std::mt19937_64& rng) const override;
  std::unique_ptr<IBeliefTracker> clone() const override;
  AnyMap serialize() const override;
};

#define HARMONIES_EXTERN_HarmoniesBeliefTracker(N, S) extern template class HarmoniesBeliefTracker<N, S>;
HARMONIES_FOR_EACH_VARIANT(HARMONIES_EXTERN_HarmoniesBeliefTracker)
#undef HARMONIES_EXTERN_HarmoniesBeliefTracker

// ---- Token encoder (`harmonies_tok`, side B) ----
//
// Variable-length tokens with positions (TokenLayout::variable_tokens,
// token_positions); header [valid, entity, anchor, type, has position, q, r].
// "Me" = the perspective (the player to move when the policy is queried).
//   global     turn phase, my token pool, pouch, deck, end-of-game proximity
//   player     one per player (me first, then the others in seat order):
//              scores and their landscape breakdown, islands, spaces left
//   slot  x5   one per central-board space: its tokens and what they could do
//              on my board (best landscape gain, tokens advancing / completing
//              a held card)
//   card       one per market card and per held card of every player (entity
//              = card id): its printed pattern (colour needs, habitat colour,
//              points) and how far the holder's board is from a cube (market
//              cards: my board, plus the closest opponent)
//   cell  x25  my board, positioned at the hex's axial (q, r): the stack, the
//              landscape group it belongs to, per colour whether it can go
//              here and the landscape points that would add, and per held
//              card (by rank of card id) whether the cell is in a best partial
//              match of the pattern, a missing piece of it, a legal cube spot,
//              and per colour how much placing it here brings the card closer
//   opp_cell   every opponent cell, without a position (relative position
//              bias only ever relates cells of my board): the same landscape
//              facts plus its neighbours' top colours by direction
//   cube       one per legal cube placement (hex, card), positioned at the
//              hex (entity = card id): what placing it scores and what it
//              blocks for my other cards
// Pointer heads (6): a cell's head c places colour c there; head 0 of a slot
// selects it, of a market card takes it, of a cube token places that cube.
// Ending the turn is the global head. The full feature list is the tables
// below, exported by layout_json() and documented in docs/games/harmonies_api.md.

#define HARMONIES_GLOBAL_FEATURES(X)                                                      \
  X(turn_started, 1) X(took_card, 1) X(end_triggered, 1) X(my_turn, 1) X(pool, 6)         \
  X(pool_total, 1) X(pouch, 6) X(pouch_total, 1) X(deck, 1) X(market_cards, 1)            \
  X(refills_left, 1) X(min_spaces, 1)

#define HARMONIES_PLAYER_FEATURES(X)                                                      \
  X(seat, 4) X(to_move, 1) X(first_player, 1) X(animal_score, 1) X(landscape_score, 1)    \
  X(total_score, 1) X(margin, 1) X(trees, 1) X(mountains, 1) X(fields, 1) X(buildings, 1) \
  X(water, 1) X(islands, 1) X(spaces, 1) X(cubes_placed, 1) X(cards_held, 1)              \
  X(card_points_open, 1)

#define HARMONIES_SLOT_FEATURES(X)                                                        \
  X(tokens, 6) X(total, 1) X(best_landscape, 1) X(advancing, 1) X(completing, 1)

#define HARMONIES_CARD_FEATURES(X)                                                        \
  X(market, 1) X(my_hand, 1) X(opp_hand, 1) X(holder_seat, 4) X(rank, 4)                  \
  X(pattern_size, 1) X(need, 6) X(habitat, 6) X(cubes_total, 1) X(cubes_left, 1)          \
  X(points_now, 1) X(next_points, 1) X(full_points, 1) X(missing, 1) X(no_fit, 1)         \
  X(ready, 1) X(opp_missing, 1)

#define HARMONIES_CELL_FEATURES(X)                                                        \
  X(tower, 13) X(height, 1) X(animal, 1) X(points, 1) X(field_group, 1) X(island, 1)      \
  X(island_split, 1) X(nb_colours, 1) X(nb_empty, 1) X(edge, 1) X(can_place, 6)           \
  X(legal, 6) X(delta, 6) X(card_part, 4) X(card_missing, 4) X(card_ready, 4)             \
  X(card_advance, 24)

#define HARMONIES_OPP_CELL_FEATURES(X)                                                    \
  X(owner_seat, 3) X(tower, 13) X(height, 1) X(animal, 1) X(points, 1) X(field_group, 1)  \
  X(island, 1) X(island_split, 1) X(nb_colours, 1) X(nb_empty, 1) X(edge, 1) X(delta, 6)  \
  X(neighbour, 48)

#define HARMONIES_CUBE_FEATURES(X)                                                        \
  X(rank, 4) X(points, 1) X(completes, 1) X(cubes_left, 1) X(blocks, 1)

#define HARMONIES_FEATURE_ENUM(name, width) name, name##_last = name + (width)-1,
namespace hgf { enum : int { HARMONIES_GLOBAL_FEATURES(HARMONIES_FEATURE_ENUM) kCount }; }
namespace hpf { enum : int { HARMONIES_PLAYER_FEATURES(HARMONIES_FEATURE_ENUM) kCount }; }
namespace hsf { enum : int { HARMONIES_SLOT_FEATURES(HARMONIES_FEATURE_ENUM) kCount }; }
namespace hkf { enum : int { HARMONIES_CARD_FEATURES(HARMONIES_FEATURE_ENUM) kCount }; }
namespace hcf { enum : int { HARMONIES_CELL_FEATURES(HARMONIES_FEATURE_ENUM) kCount }; }
namespace hof { enum : int { HARMONIES_OPP_CELL_FEATURES(HARMONIES_FEATURE_ENUM) kCount }; }
namespace hqf { enum : int { HARMONIES_CUBE_FEATURES(HARMONIES_FEATURE_ENUM) kCount }; }
#undef HARMONIES_FEATURE_ENUM

template <int NPlayers, int Side>
class HarmoniesTokenEncoder final : public IFeatureEncoder {
  static_assert(Side == kSideB, "the token encoder's water features follow side B (islands)");

 public:
  using Geo = Board<Side>;
  static constexpr int kHexes = Geo::kHexes;

  enum TokenType {
    kGlobalToken, kPlayerToken, kSlotToken, kCardToken, kCellToken, kOppCellToken, kCubeToken,
    kTokenTypeCount
  };

  static constexpr int kHeader = 7;
  static constexpr int kFeatures = hof::kCount;  // the widest token type
  static constexpr int kTokenDim = kHeader + kFeatures;
  static constexpr int kNumHeads = kColors;
  static constexpr int kNumEntities = 1 + kAnimalCards;  // card id

  // Anchor ids are 1 + index; the bases below are indices.
  static constexpr int kAnchorCell = 0;                           // + hex
  static constexpr int kAnchorSlot = kAnchorCell + kHexes;        // + slot
  static constexpr int kAnchorCard = kAnchorSlot + kSlots;        // + card id - 1 (market)
  static constexpr int kAnchorCube = kAnchorCard + kAnimalCards;  // + hex * 32 + card id - 1
  static constexpr int kNumAnchors = kAnchorCube + kHexes * kAnimalCards;

  HarmoniesTokenEncoder();

  int action_space() const override { return Geo::kActionSpace; }
  int feature_dim() const override { return 0; }
  const TokenLayout* token_layout() const override { return &layout_; }
  std::vector<EncoderSymmetry> symmetries() const override;

  void encode_features(const IGameState& masked_state, int perspective_player,
                       const IBeliefTracker* tracker, std::vector<float>* out) const override;

  // The layout as JSON: header, per token type its features (name, offset,
  // width), entity and anchor ranges. For tests and documentation.
  static std::string layout_json();

 private:
  TokenLayout layout_;
};

extern template class HarmoniesTokenEncoder<2, kSideB>;
extern template class HarmoniesTokenEncoder<3, kSideB>;
extern template class HarmoniesTokenEncoder<4, kSideB>;

}  // namespace board_ai::harmonies
