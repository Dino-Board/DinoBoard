#pragma once

#include <any>
#include <array>
#include <cstdint>
#include <vector>

#include "../../engine/core/game_interfaces.h"
#include "../../engine/core/visibility_schema.h"
#include "harmonies_board_data.h"

namespace board_ai::harmonies {

// ---- Board geometry ----
//
// Two personal-board sides (rules p.2, p.8), generated into
// harmonies_board_data.h by scripts/gen_card_data.py: flat-top hexes in
// columns, hex ids column-major.
//   Side A: columns 5-4-5-4-5 (23 hexes), water scores the longest river.
//   Side B: columns 4-3-4-3-4-3-4 (25 hexes), water scores islands.
constexpr int kSlots = 5;            // central-board token slots
constexpr int kColors = 6;           // Blue/Gray/Brown/Green/Yellow/Red (1..6, 0=empty)
constexpr int kAnimalCards = 32;
constexpr int kHandSize = 4;
constexpr int kTowerTypes = 13;      // 0=empty, 1..6=single, 7=房2, 8=木木, 9=树2, 10=山2, 11=树3, 12=山3
constexpr int kMaxTokensPerSlot = 3;

// Token color ids (token-color domain, 1..6). 0 is "empty" / unused.
constexpr std::int8_t kColorEmpty  = 0;
constexpr std::int8_t kColorBlue   = 1;
constexpr std::int8_t kColorGray   = 2;
constexpr std::int8_t kColorBrown  = 3;
constexpr std::int8_t kColorGreen  = 4;
constexpr std::int8_t kColorYellow = 5;
constexpr std::int8_t kColorRed    = 6;

// Initial token counts in the pouch (sum=120). See rules p.2.
constexpr std::array<int, kColors> kInitialPouchCounts = {23, 23, 21, 19, 19, 15};

// ---- Action segments (the first two do not depend on the board) ----
constexpr int kActSelectSlotBase = 0;        // 0..4
constexpr int kActTakeCardBase   = 5;        // 5..36 (card_id 1..32)
constexpr int kActPlaceTokenBase = 37;       // + hex * 6 + color - 1

// A board side: its generated geometry plus the action layout that follows
// from its hex count. Side A: place cube 175..910, end turn 911, 912 actions;
// side B: place cube 187..986, end turn 987, 988 actions.
template <int Side>
struct Board : BoardData<Side> {
  using Data = BoardData<Side>;
  static constexpr int kHexes = Data::kHexes;
  static constexpr bool kIslands = Side == kSideB;
  static constexpr int kPlaceTokenCount = kHexes * kColors;
  static constexpr int kPlaceCubeCount = kHexes * kAnimalCards;
  static constexpr int kActPlaceCubeBase = kActPlaceTokenBase + kPlaceTokenCount;
  static constexpr int kActEndTurn = kActPlaceCubeBase + kPlaceCubeCount;
  static constexpr int kActionSpace = kActEndTurn + 1;

  static int axial_to_hex(int q, int r) {
    const int qi = q + Data::kAxialQOffset;
    const int ri = r + Data::kAxialROffset;
    if (qi < 0 || qi >= Data::kAxialWidth || ri < 0 || ri >= Data::kAxialHeight) return -1;
    return Data::kAxialToHex[static_cast<std::size_t>(qi * Data::kAxialHeight + ri)];
  }
  static int neighbor(int hex, int dir) {
    return Data::kNeighbors[static_cast<std::size_t>(hex)][static_cast<std::size_t>(dir)];
  }
};

// ---- Tower transition table ----
// kTowerTransition[tower_type][color-1] -> new tower_type, or -1 if illegal.
// Indexed by color in 1..6 (subtract 1 for table lookup).
extern const std::array<std::array<std::int8_t, kColors>, kTowerTypes> kTowerTransition;

// Top color of each tower_type (0=empty, 1=Blue..6=Red).
constexpr std::array<int, kTowerTypes> kTowerTopColor = {0, 1, 2, 3, 4, 5, 6, 6, 3, 4, 2, 4, 2};

template <int NPlayers, int Side>
struct HarmoniesConfig {
  static_assert(NPlayers >= 2 && NPlayers <= 4);
  static_assert(Side == kSideA || Side == kSideB);
  static constexpr int kPlayers = NPlayers;
  static constexpr int kActionSpaceVal = Board<Side>::kActionSpace;
};

template <int NPlayers, int Side>
class HarmoniesState final : public CloneableState<HarmoniesState<NPlayers, Side>> {
 public:
  using Cfg = HarmoniesConfig<NPlayers, Side>;
  using Geo = Board<Side>;
  static constexpr int kHexes = Geo::kHexes;

  HarmoniesState();

  static const viz::VisibilitySchema& schema();
  const viz::VisibilitySchema& schema_ref() const override { return schema(); }

  // ---- Public scalars / small arrays ----
  int current_player_ = 0;
  int game_first_player_ = 0;
  int winner_ = -1;
  bool terminal = false;
  bool end_triggered = false;

  // Turn-state machine.
  bool turn_started = false;          // true after SELECT_SLOT, until END_TURN
  bool took_card_this_turn = false;
  std::array<std::int8_t, kColors> selected_tokens{};  // 6-color count for current player

  // Central board: 5 slots × 6 colors (count of tokens of each color in slot).
  std::array<std::array<int, kColors>, kSlots> central_board_counts{};

  // Animal market: 5 slots showing card_id (1..32; 0=empty slot).
  std::array<std::int8_t, kSlots> animal_market{};

  // Animal deck: 32 cards, 1=still in draw pile, 0=consumed/in market/in hand.
  // Hidden from AI (all_hidden visibility). Draw is random via rng.
  std::array<std::int8_t, kAnimalCards> animal_deck{};
  int animal_deck_count = 0;  // public count of cards remaining in deck

  // Completed animal cards (all cubes placed; the card left its owner's
  // hand and lies next to the board). Public: 1 = completed. Without it a
  // completed card is indistinguishable from an undrawn one, and
  // randomize_unseen would put it back into the sampled deck.
  std::array<std::int8_t, kAnimalCards> completed_cards{};

  // Pouch counts per color.
  std::array<int, kColors> pouch_counts{};

  // Per-player public state (no hidden info in Harmonies).
  std::array<std::array<std::int8_t, kHexes>, NPlayers> tower_type{};
  std::array<std::array<std::int8_t, kHexes>, NPlayers> has_animal{};
  std::array<std::array<std::int8_t, kHandSize>, NPlayers> held_cards{};
  std::array<std::array<std::int8_t, kHandSize>, NPlayers> cube_remaining{};
  std::array<std::array<std::int8_t, kHandSize>, NPlayers> cube_initial{};
  std::array<std::int8_t, NPlayers> space_remaining{};
  std::array<int, NPlayers> animal_scores{};
  std::array<int, NPlayers> landscape_scores{};

  // ---- IGameState overrides ----
  StateHash64 state_hash() const override;
  void hash_field_slot(Hasher& h, viz::FieldId fid,
                       const std::vector<int>& idx) const override;
  void hash_field_bulk(Hasher& h, viz::FieldId fid,
                       const viz::VizTensor& v,
                       int perspective) const override;
  std::any read_field_slot(viz::FieldId fid,
                           const std::vector<int>& idx) const override;
  void write_field_slot(viz::FieldId fid,
                        const std::vector<int>& idx,
                        const std::any& value) override;

  int current_player() const override { return current_player_; }
  int first_player() const override { return game_first_player_; }
  bool is_terminal() const override { return terminal; }
  int num_players() const override { return NPlayers; }
  int winner() const override { return winner_; }
  // Rulebook ranking: most points; tie -> most animal cubes placed; still
  // tied -> shared victory. first_place() is every player tied for first on
  // that key (rules use it to set winner_, which is -1 unless it is one
  // player). On a shared victory winners() is that whole group and losers()
  // everyone else, so in 3-4 player games the players outside the tie still
  // lose (zero-sum: +1/nw each winner, -1/nl each loser); a tie between ALL
  // players — and every 2-player tie — stays a plain draw: both lists empty,
  // all-zero.
  std::vector<int> first_place() const;
  std::vector<int> winners() const override;
  std::vector<int> losers() const override;

  void reset_with_seed(std::uint64_t seed) override;

  // Helper used by rules + tests: number of token layers currently on a hex.
  // Returns 0 for tower_type==0, 1 for single-layer (1..6), 2 for tower
  // types 7/8/9/10, 3 for tower types 11/12.
  static int tower_height(int tower_type);
};

#define HARMONIES_FOR_EACH_VARIANT(X) \
  X(2, kSideA) X(3, kSideA) X(4, kSideA) X(2, kSideB) X(3, kSideB) X(4, kSideB)

#define HARMONIES_EXTERN_STATE(N, S) extern template class HarmoniesState<N, S>;
HARMONIES_FOR_EACH_VARIANT(HARMONIES_EXTERN_STATE)
#undef HARMONIES_EXTERN_STATE

}  // namespace board_ai::harmonies
