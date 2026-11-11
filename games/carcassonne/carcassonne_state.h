#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "../../engine/core/game_interfaces.h"
#include "../../engine/core/visibility_schema.h"
#include "carcassonne_tiles.h"

namespace board_ai::carcassonne {

constexpr int kPlayers = 2;
constexpr int kBoardRows = 20;
constexpr int kBoardCols = 20;
constexpr int kBoardCells = kBoardRows * kBoardCols;
constexpr int kPlaceActionCount = kBoardCells * 4;
constexpr int kMeeplePassAction = kPlaceActionCount;
constexpr int kMeepleSideActionBase = kMeeplePassAction + 1;
constexpr int kMeepleFarmActionBase = kMeepleSideActionBase + kSides;
constexpr int kMeepleCenterAction = kMeepleFarmActionBase + kFarmSides;
constexpr int kActionSpace = kMeepleCenterAction + 1;
constexpr int kDeckSlots = 71;
constexpr int kHistorySlots = kDeckSlots + 1;
constexpr int kMeeplesPerPlayer = 7;
constexpr int kMeepleNoPos = -1;
constexpr int kMeepleCenterPos = kSides + kFarmSides;
constexpr int kMeeplePositions = kMeepleCenterPos + 1;
constexpr int kStagePlaceTile = 0;
constexpr int kStageMeeple = 1;
constexpr int kHistoryFeatureDim =
    kTileTypes + 4 + 2 + kMeeplePositions;
constexpr int kFeatureDim = kHistorySlots * kHistoryFeatureDim + 4 + kTileTypes;
constexpr int kNoTile = -1;
constexpr int kNoOwner = -1;

struct CarcassonneState final : public CloneableState<CarcassonneState> {
  int current_player_ = 0;
  std::int8_t stage = kStagePlaceTile;
  int winner_ = -1;
  bool terminal = false;
  bool shared_victory = false;
  int plies = 0;
  std::int8_t current_tile = kNoTile;
  std::int8_t deck_count = 0;
  std::int8_t last_row = -1;
  std::int8_t last_col = -1;
  std::int8_t last_rotation = 0;
  int last_score_delta = 0;
  std::array<int, kPlayers> scores{0, 0};
  std::array<std::int8_t, kPlayers> meeples_remaining{};
  std::array<std::int8_t, kBoardCells> board_tile{};
  std::array<std::int8_t, kBoardCells> board_rotation{};
  std::array<std::int8_t, kBoardCells> board_meeple_owner{};
  std::array<std::int8_t, kBoardCells> board_meeple_pos{};
  std::array<int, kBoardCells> final_meeple_scores{};
  std::array<int, kPlayers> final_score_deltas{0, 0};
  // Bag is a count-only multiset (game-rules/bag-draw recipe).
  // bag_counts[t] = remaining tiles of type t in the bag. The order in
  // which a future draw will resolve is a property of the RNG, not state —
  // do NOT add a pre-shuffled tile_deck array. See bag-draw.md for why.
  std::array<int, kTileTypes> bag_counts{};
  std::array<std::int8_t, kTileTypes> discarded_tile_counts{};
  std::array<std::int8_t, kHistorySlots> history_tile{};
  std::array<std::int8_t, kHistorySlots> history_rotation{};
  std::array<std::int8_t, kHistorySlots> history_row{};
  std::array<std::int8_t, kHistorySlots> history_col{};

  CarcassonneState();

  static const viz::VisibilitySchema& schema();
  void reset_with_seed(std::uint64_t seed) override;

  // Weighted-by-count draw of one tile type from bag_counts. Decrements
  // the chosen type's count and deck_count, returns the tile type.
  // Returns kNoTile if the bag is empty. The rng is the caller's
  // (do_action_fast's `rng` arg in the rules path) — state holds no rng.
  int draw_one_tile_from_bag(std::mt19937_64& rng);

  StateHash64 state_hash() const override;
  void hash_field_slot(Hasher& h, viz::FieldId fid,
                       const std::vector<int>& idx) const override;
  std::any read_field_slot(viz::FieldId fid,
                           const std::vector<int>& idx) const override;
  void write_field_slot(viz::FieldId fid, const std::vector<int>& idx,
                        const std::any& value) override;
  void mask_field_slot(viz::FieldId fid,
                       const std::vector<int>& idx) override;
  const viz::VisibilitySchema& schema_ref() const override { return schema(); }

  int current_player() const override { return current_player_; }
  int first_player() const override { return 0; }
  bool is_terminal() const override { return terminal; }
  bool is_turn_start() const override { return stage == kStagePlaceTile; }
  int num_players() const override { return kPlayers; }
  int winner() const override { return winner_; }
  std::vector<int> winners() const override;
  std::vector<int> losers() const override;

  // Lowest rotation that shows `tile` the same way (see rotation_period);
  // values that are not a tile / rotation are returned unchanged.
  static std::int8_t canonical_rotation(int tile, int rotation);
  // Store the rules' own forms of the rotation and meeple position at `cell`.
  void canonicalize_cell(int cell);

  int index(int row, int col) const { return row * kBoardCols + col; }
  bool in_bounds(int row, int col) const {
    return row >= 0 && row < kBoardRows && col >= 0 && col < kBoardCols;
  }
};

}  // namespace board_ai::carcassonne
