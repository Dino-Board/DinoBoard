#include "carcassonne_state.h"

#include <algorithm>
#include <any>
#include <cstddef>
#include <numeric>
#include <random>
#include <stdexcept>

#include "../../engine/core/masked_state.h"
#include "../../engine/core/schema_hash.h"
#include "../../engine/core/viz_runtime.h"

namespace board_ai::carcassonne {

namespace {

enum : viz::FieldId {
  kCurrentPlayer = 0,
  kStage,
  kWinner,
  kTerminal,
  kSharedVictory,
  kPlies,
  kCurrentTile,
  kDeckCount,
  kLastRow,
  kLastCol,
  kLastRotation,
  kLastScoreDelta,
  kScores,
  kMeeplesRemaining,
  kBoardTile,
  kBoardRotation,
  kBoardMeepleOwner,
  kBoardMeeplePos,
  kFinalMeepleScores,
  kFinalScoreDeltas,
  kBagCounts,
  kDiscardedTileCounts,
  kHistoryTile,
  kHistoryRotation,
  kHistoryRow,
  kHistoryCol,
};

}  // namespace

const viz::VisibilitySchema& CarcassonneState::schema() {
  static const viz::VisibilitySchema s = []() {
    viz::VisibilitySchema schema;
    schema.n_players = kPlayers;
    viz::declare_field(schema, "current_player", viz::all_public({}, kPlayers));
    viz::declare_field(schema, "stage", viz::all_public({}, kPlayers));
    viz::declare_field(schema, "winner", viz::all_public({}, kPlayers));
    viz::declare_field(schema, "terminal", viz::all_public({}, kPlayers));
    viz::declare_field(schema, "shared_victory", viz::all_public({}, kPlayers));
    viz::declare_field(schema, "plies", viz::all_public({}, kPlayers));
    viz::declare_field(schema, "current_tile", viz::all_public({}, kPlayers));
    viz::declare_field(schema, "deck_count", viz::all_public({}, kPlayers));
    viz::declare_field(schema, "last_row", viz::all_public({}, kPlayers));
    viz::declare_field(schema, "last_col", viz::all_public({}, kPlayers));
    viz::declare_field(schema, "last_rotation", viz::all_public({}, kPlayers));
    viz::declare_field(schema, "last_score_delta", viz::all_public({}, kPlayers));
    viz::declare_field(schema, "scores", viz::all_public({kPlayers}, kPlayers));
    viz::declare_field(schema, "meeples_remaining",
                       viz::all_public({kPlayers}, kPlayers));
    viz::declare_field(schema, "board_tile",
                       viz::all_public({kBoardCells}, kPlayers));
    viz::declare_field(schema, "board_rotation",
                       viz::all_public({kBoardCells}, kPlayers));
    viz::declare_field(schema, "board_meeple_owner",
                       viz::all_public({kBoardCells}, kPlayers));
    viz::declare_field(schema, "board_meeple_pos",
                       viz::all_public({kBoardCells}, kPlayers));
    viz::declare_field(schema, "final_meeple_scores",
                       viz::all_public({kBoardCells}, kPlayers));
    viz::declare_field(schema, "final_score_deltas",
                       viz::all_public({kPlayers}, kPlayers));
    // bag_counts: per-type remaining count. Public — derivable from
    // initial - board - current_tile - discarded. See bag-draw.md.
    viz::declare_field(schema, "bag_counts",
                       viz::all_public({kTileTypes}, kPlayers));
    viz::declare_field(schema, "discarded_tile_counts",
                       viz::all_public({kTileTypes}, kPlayers));
    viz::declare_field(schema, "history_tile",
                       viz::all_public({kHistorySlots}, kPlayers));
    viz::declare_field(schema, "history_rotation",
                       viz::all_public({kHistorySlots}, kPlayers));
    viz::declare_field(schema, "history_row",
                       viz::all_public({kHistorySlots}, kPlayers));
    viz::declare_field(schema, "history_col",
                       viz::all_public({kHistorySlots}, kPlayers));
    return schema;
  }();
  return s;
}

CarcassonneState::CarcassonneState() {
  reset_with_seed(0xCA4CA550u);
}

void CarcassonneState::reset_with_seed(std::uint64_t seed) {
  IGameState::reset_step_count_base();
  current_player_ = 0;
  stage = static_cast<std::int8_t>(kStagePlaceTile);
  winner_ = -1;
  terminal = false;
  shared_victory = false;
  plies = 0;
  current_tile = kNoTile;
  deck_count = 0;
  last_row = -1;
  last_col = -1;
  last_rotation = 0;
  last_score_delta = 0;
  scores = {0, 0};
  meeples_remaining.fill(static_cast<std::int8_t>(kMeeplesPerPlayer));
  board_tile.fill(static_cast<std::int8_t>(kNoTile));
  board_rotation.fill(0);
  board_meeple_owner.fill(static_cast<std::int8_t>(kNoOwner));
  board_meeple_pos.fill(static_cast<std::int8_t>(kMeepleNoPos));
  final_meeple_scores.fill(0);
  final_score_deltas = {0, 0};
  bag_counts.fill(0);
  discarded_tile_counts.fill(0);
  history_tile.fill(static_cast<std::int8_t>(kNoTile));
  history_rotation.fill(0);
  history_row.fill(-1);
  history_col.fill(-1);

  const int start_type = tile_type_from_code('D');
  const int start_row = kBoardRows / 2;
  const int start_col = kBoardCols / 2;
  const int start_idx = index(start_row, start_col);
  board_tile[static_cast<size_t>(start_idx)] =
      static_cast<std::int8_t>(start_type);
  board_rotation[static_cast<size_t>(start_idx)] = 0;
  history_tile[0] = static_cast<std::int8_t>(start_type);
  history_rotation[0] = 0;
  history_row[0] = static_cast<std::int8_t>(start_row);
  history_col[0] = static_cast<std::int8_t>(start_col);

  int total = 0;
  for (int tile = 0; tile < kTileTypes; ++tile) {
    int count = tile_def(tile).initial_count;
    if (tile == start_type) count -= 1;
    bag_counts[static_cast<size_t>(tile)] = count;
    total += count;
  }
  if (total != kDeckSlots) {
    throw std::runtime_error("carcassonne initial deck size mismatch");
  }
  deck_count = static_cast<std::int8_t>(total);
  viz::init_viz(*this, schema());

  // Initial draw: a one-shot rng from the seed (rng is not stored on
  // state; subsequent draws use do_action_fast's rng arg). The first
  // tile drawn is always placeable next to start tile D — every tile
  // type has at least one rotation that matches one of D's N/E/S/W
  // edges — so we don't need the placement-check loop here, only the
  // weighted sample. Discards loop lives in CarcassonneRules::draw_next_tile.
  std::mt19937_64 init_rng(seed);
  const int drawn = draw_one_tile_from_bag(init_rng);
  if (drawn == kNoTile) {
    throw std::runtime_error("carcassonne reset: bag empty on initial draw");
  }
  current_tile = static_cast<std::int8_t>(drawn);
}

int CarcassonneState::draw_one_tile_from_bag(std::mt19937_64& rng) {
  int total = 0;
  for (int c : bag_counts) total += c;
  if (total == 0) return kNoTile;
  // Weighted uniform sample over remaining tiles (Azul-style).
  int pick = static_cast<int>(rng() % static_cast<std::uint64_t>(total));
  for (int t = 0; t < kTileTypes; ++t) {
    const int c = bag_counts[static_cast<size_t>(t)];
    if (pick < c) {
      bag_counts[static_cast<size_t>(t)] = c - 1;
      deck_count = static_cast<std::int8_t>(deck_count - 1);
      return t;
    }
    pick -= c;
  }
  throw std::runtime_error("carcassonne draw_one_tile_from_bag: unreachable");
}

StateHash64 CarcassonneState::state_hash() const {
  // Delegate to the schema-driven hash so visibility is the single
  // source of truth. All non-bag fields are all_public; bag_counts is
  // also all_public (counts are publicly derivable). No hidden slots
  // remain on this state — every field that participates in DAG
  // identity is visible to every player. Matches maskmen / coup /
  // liarsdice / jaipur. See bag-draw.md for why a hand-written hash
  // mixing a hidden ordered draw container would corrupt TT keys.
  return state_hash_for_perspective(current_player_);
}

void CarcassonneState::hash_field_slot(
    Hasher& h, viz::FieldId fid,
    const std::vector<int>& idx) const {
  switch (fid) {
    case kCurrentPlayer: h.add(current_player_); return;
    case kStage: h.add(stage); return;
    case kWinner: h.add(winner_ + 1); return;
    case kTerminal: h.add(terminal ? 1 : 0); return;
    case kSharedVictory: h.add(shared_victory ? 1 : 0); return;
    case kPlies: h.add(plies); return;
    case kCurrentTile: h.add(current_tile + 1); return;
    case kDeckCount: h.add(deck_count); return;
    case kLastRow: h.add(last_row + 1); return;
    case kLastCol: h.add(last_col + 1); return;
    case kLastRotation: h.add(last_rotation); return;
    case kLastScoreDelta: h.add(last_score_delta + 4096); return;
    case kScores: h.add(scores[static_cast<size_t>(idx[0])] + 4096); return;
    case kMeeplesRemaining:
      h.add(meeples_remaining[static_cast<size_t>(idx[0])] + 1); return;
    case kBoardTile:
      h.add(board_tile[static_cast<size_t>(idx[0])] + 2); return;
    case kBoardRotation:
      h.add(board_rotation[static_cast<size_t>(idx[0])] + 1); return;
    case kBoardMeepleOwner:
      h.add(board_meeple_owner[static_cast<size_t>(idx[0])] + 2); return;
    case kBoardMeeplePos:
      h.add(board_meeple_pos[static_cast<size_t>(idx[0])] + 2); return;
    case kFinalMeepleScores:
      h.add(final_meeple_scores[static_cast<size_t>(idx[0])] + 4096); return;
    case kFinalScoreDeltas:
      h.add(final_score_deltas[static_cast<size_t>(idx[0])] + 4096); return;
    case kBagCounts:
      h.add(bag_counts[static_cast<size_t>(idx[0])]); return;
    case kDiscardedTileCounts:
      h.add(discarded_tile_counts[static_cast<size_t>(idx[0])] + 1); return;
    case kHistoryTile:
      h.add(history_tile[static_cast<size_t>(idx[0])] + 2); return;
    case kHistoryRotation:
      h.add(history_rotation[static_cast<size_t>(idx[0])] + 1); return;
    case kHistoryRow:
      h.add(history_row[static_cast<size_t>(idx[0])] + 1); return;
    case kHistoryCol:
      h.add(history_col[static_cast<size_t>(idx[0])] + 1); return;
  }
}

std::any CarcassonneState::read_field_slot(
    viz::FieldId fid, const std::vector<int>& idx) const {
  switch (fid) {
    case kCurrentPlayer: return std::any(current_player_);
    case kStage: return std::any(static_cast<int>(stage));
    case kWinner: return std::any(winner_);
    case kTerminal: return std::any(terminal);
    case kSharedVictory: return std::any(shared_victory);
    case kPlies: return std::any(plies);
    case kCurrentTile: return std::any(static_cast<int>(current_tile));
    case kDeckCount: return std::any(static_cast<int>(deck_count));
    case kLastRow: return std::any(static_cast<int>(last_row));
    case kLastCol: return std::any(static_cast<int>(last_col));
    case kLastRotation: return std::any(static_cast<int>(last_rotation));
    case kLastScoreDelta: return std::any(last_score_delta);
    case kScores:
      return std::any(scores[static_cast<size_t>(idx[0])]);
    case kMeeplesRemaining:
      return std::any(static_cast<int>(
          meeples_remaining[static_cast<size_t>(idx[0])]));
    case kBoardTile:
      return std::any(static_cast<int>(
          board_tile[static_cast<size_t>(idx[0])]));
    case kBoardRotation:
      return std::any(static_cast<int>(
          board_rotation[static_cast<size_t>(idx[0])]));
    case kBoardMeepleOwner:
      return std::any(static_cast<int>(
          board_meeple_owner[static_cast<size_t>(idx[0])]));
    case kBoardMeeplePos:
      return std::any(static_cast<int>(
          board_meeple_pos[static_cast<size_t>(idx[0])]));
    case kFinalMeepleScores:
      return std::any(final_meeple_scores[static_cast<size_t>(idx[0])]);
    case kFinalScoreDeltas:
      return std::any(final_score_deltas[static_cast<size_t>(idx[0])]);
    case kBagCounts:
      return std::any(bag_counts[static_cast<size_t>(idx[0])]);
    case kDiscardedTileCounts:
      return std::any(static_cast<int>(
          discarded_tile_counts[static_cast<size_t>(idx[0])]));
    case kHistoryTile:
      return std::any(static_cast<int>(
          history_tile[static_cast<size_t>(idx[0])]));
    case kHistoryRotation:
      return std::any(static_cast<int>(
          history_rotation[static_cast<size_t>(idx[0])]));
    case kHistoryRow:
      return std::any(static_cast<int>(
          history_row[static_cast<size_t>(idx[0])]));
    case kHistoryCol:
      return std::any(static_cast<int>(
          history_col[static_cast<size_t>(idx[0])]));
  }
  throw std::runtime_error("carcassonne read_field_slot unknown field");
}

// A snapshot built by another program (GameSession.apply_initial_observation)
// may show a symmetric tile in any of its equal rotations, and a meeple on
// any edge / half-edge of its city, road or farm. The rules read both
// correctly, but the hash and the encoders compare stored values, so the
// forms the rules themselves produce are stored instead: equal positions
// then are equal states. This relies on the order the snapshot walker
// writes in (fields by name, AnyMap being a std::map): a cell's rotation and
// meeple position are in place when its tile arrives, a history slot's
// rotation when its tile arrives, the board and last_rotation when last_row
// arrives. Writes in another order are stored as given.
void CarcassonneState::write_field_slot(
    viz::FieldId fid, const std::vector<int>& idx, const std::any& value) {
  auto as_int = [&]() -> int {
    if (value.type() == typeid(int)) return std::any_cast<int>(value);
    if (value.type() == typeid(bool)) return std::any_cast<bool>(value) ? 1 : 0;
    throw std::runtime_error("carcassonne field value has unsupported type");
  };
  auto as_bool = [&]() -> bool {
    if (value.type() == typeid(bool)) return std::any_cast<bool>(value);
    if (value.type() == typeid(int)) return std::any_cast<int>(value) != 0;
    throw std::runtime_error("carcassonne bool field value has unsupported type");
  };
  switch (fid) {
    case kCurrentPlayer: current_player_ = as_int(); return;
    case kStage: stage = static_cast<std::int8_t>(as_int()); return;
    case kWinner: winner_ = as_int(); return;
    case kTerminal: terminal = as_bool(); return;
    case kSharedVictory: shared_victory = as_bool(); return;
    case kPlies: plies = as_int(); return;
    case kCurrentTile: current_tile = static_cast<std::int8_t>(as_int()); return;
    case kDeckCount: deck_count = static_cast<std::int8_t>(as_int()); return;
    case kLastRow:
      last_row = static_cast<std::int8_t>(as_int());
      if (in_bounds(last_row, last_col)) {
        last_rotation = canonical_rotation(
            board_tile[static_cast<size_t>(index(last_row, last_col))], last_rotation);
      }
      return;
    case kLastCol: last_col = static_cast<std::int8_t>(as_int()); return;
    case kLastRotation: last_rotation = static_cast<std::int8_t>(as_int()); return;
    case kLastScoreDelta: last_score_delta = as_int(); return;
    case kScores:
      scores[static_cast<size_t>(idx[0])] = as_int(); return;
    case kMeeplesRemaining:
      meeples_remaining[static_cast<size_t>(idx[0])] =
          static_cast<std::int8_t>(as_int()); return;
    case kBoardTile:
      board_tile[static_cast<size_t>(idx[0])] =
          static_cast<std::int8_t>(as_int());
      canonicalize_cell(idx[0]);
      return;
    case kBoardRotation:
      board_rotation[static_cast<size_t>(idx[0])] =
          static_cast<std::int8_t>(as_int()); return;
    case kBoardMeepleOwner:
      board_meeple_owner[static_cast<size_t>(idx[0])] =
          static_cast<std::int8_t>(as_int()); return;
    case kBoardMeeplePos:
      board_meeple_pos[static_cast<size_t>(idx[0])] =
          static_cast<std::int8_t>(as_int()); return;
    case kFinalMeepleScores:
      final_meeple_scores[static_cast<size_t>(idx[0])] = as_int(); return;
    case kFinalScoreDeltas:
      final_score_deltas[static_cast<size_t>(idx[0])] = as_int(); return;
    case kBagCounts:
      bag_counts[static_cast<size_t>(idx[0])] = as_int(); return;
    case kDiscardedTileCounts:
      discarded_tile_counts[static_cast<size_t>(idx[0])] =
          static_cast<std::int8_t>(as_int()); return;
    case kHistoryTile: {
      const auto slot = static_cast<size_t>(idx[0]);
      history_tile[slot] = static_cast<std::int8_t>(as_int());
      history_rotation[slot] = canonical_rotation(history_tile[slot], history_rotation[slot]);
      return;
    }
    case kHistoryRotation:
      history_rotation[static_cast<size_t>(idx[0])] =
          static_cast<std::int8_t>(as_int()); return;
    case kHistoryRow:
      history_row[static_cast<size_t>(idx[0])] =
          static_cast<std::int8_t>(as_int()); return;
    case kHistoryCol:
      history_col[static_cast<size_t>(idx[0])] =
          static_cast<std::int8_t>(as_int()); return;
  }
  throw std::runtime_error("carcassonne write_field_slot unknown field");
}

std::int8_t CarcassonneState::canonical_rotation(int tile, int rotation) {
  if (tile < 0 || tile >= kTileTypes || rotation < 0 || rotation >= 4) {
    return static_cast<std::int8_t>(rotation);
  }
  return static_cast<std::int8_t>(rotation % rotation_period(tile));
}

void CarcassonneState::canonicalize_cell(int cell) {
  const auto c = static_cast<size_t>(cell);
  const int tile = board_tile[c];
  board_rotation[c] = canonical_rotation(tile, board_rotation[c]);
  const int rotation = board_rotation[c];
  if (tile < 0 || tile >= kTileTypes || rotation < 0 || rotation >= 4) return;
  board_meeple_pos[c] = static_cast<std::int8_t>(
      canonical_meeple_position(tile, rotation, board_meeple_pos[c]));
}

void CarcassonneState::mask_field_slot(
    viz::FieldId /*fid*/, const std::vector<int>& /*idx*/) {
  // No hidden fields on this state — bag is now count-only and public.
  // (kTileDeck removed; would have masked the pre-shuffled order.)
}

std::vector<int> CarcassonneState::winners() const {
  if (!terminal) return {};
  int best = *std::max_element(scores.begin(), scores.end());
  std::vector<int> out;
  for (int p = 0; p < kPlayers; ++p) {
    if (scores[static_cast<size_t>(p)] == best) out.push_back(p);
  }
  return out;
}

std::vector<int> CarcassonneState::losers() const {
  if (!terminal) return {};
  int best = *std::max_element(scores.begin(), scores.end());
  std::vector<int> out;
  for (int p = 0; p < kPlayers; ++p) {
    if (scores[static_cast<size_t>(p)] != best) out.push_back(p);
  }
  return out;
}

}  // namespace board_ai::carcassonne
