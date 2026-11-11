#include "quoridor_state.h"

#include "../../engine/core/viz_runtime.h"

namespace board_ai::quoridor {

const viz::VisibilitySchema& QuoridorState::schema() {
  // Built once on first use. Quoridor is fully public — pawn positions,
  // wall placements and remaining-wall counts are all visible to every
  // viewer. Same all_public-only template as tictactoe.
  static const viz::VisibilitySchema s = []() {
    viz::VisibilitySchema schema;
    schema.n_players = kPlayers;
    viz::declare_field(schema, "current_player",
                       viz::all_public({}, kPlayers));
    viz::declare_field(schema, "winner", viz::all_public({}, kPlayers));
    viz::declare_field(schema, "terminal", viz::all_public({}, kPlayers));
    viz::declare_field(schema, "move_count", viz::all_public({}, kPlayers));
    viz::declare_field(schema, "scores",
                       viz::all_public({kPlayers}, kPlayers));
    viz::declare_field(schema, "pawn_row",
                       viz::all_public({kPlayers}, kPlayers));
    viz::declare_field(schema, "pawn_col",
                       viz::all_public({kPlayers}, kPlayers));
    viz::declare_field(schema, "walls_remaining",
                       viz::all_public({kPlayers}, kPlayers));
    viz::declare_field(schema, "h_walls",
                       viz::all_public({kWallSlots}, kPlayers));
    viz::declare_field(schema, "v_walls",
                       viz::all_public({kWallSlots}, kPlayers));
    return schema;
  }();
  return s;
}

QuoridorState::QuoridorState() {
  reset_with_seed(0xC0FFEE1234ULL);
}

void QuoridorState::reset_with_seed(std::uint64_t seed) {
  IGameState::reset_step_count_base();
  (void)seed;  // quoridor is deterministic; seed unused.
  current_player_ = 0;
  winner_ = -1;
  terminal = false;
  move_count = 0;
  scores = {0, 0};
  pawn_row = {0, static_cast<std::int8_t>(kBoardSize - 1)};
  pawn_col = {4, 4};
  walls_remaining = {kMaxWallsPerPlayer, kMaxWallsPerPlayer};
  h_walls.fill(0);
  v_walls.fill(0);
  undo_stack.clear();
  viz::init_viz(*this, schema());
}

StateHash64 QuoridorState::state_hash() const {
  std::size_t h = 0;

  hash_combine(h,static_cast<std::size_t>(current_player_));
  hash_combine(h,static_cast<std::size_t>(winner_ + 1));
  hash_combine(h,static_cast<std::size_t>(terminal ? 1 : 0));
  hash_combine(h,static_cast<std::size_t>(move_count));
  hash_combine(h,static_cast<std::size_t>(scores[0] + 2));
  hash_combine(h,static_cast<std::size_t>(scores[1] + 2));
  for (int p = 0; p < kPlayers; ++p) {
    hash_combine(h,static_cast<std::size_t>(pawn_row[static_cast<size_t>(p)] + 1));
    hash_combine(h,static_cast<std::size_t>(pawn_col[static_cast<size_t>(p)] + 1));
    hash_combine(h,static_cast<std::size_t>(walls_remaining[static_cast<size_t>(p)] + 1));
  }
  for (std::uint8_t w : h_walls) hash_combine(h,static_cast<std::size_t>(w));
  for (std::uint8_t w : v_walls) hash_combine(h,static_cast<std::size_t>(w));
  return static_cast<StateHash64>(h);
}

namespace {
enum : viz::FieldId {
  kCurrentPlayer = 0,
  kWinner,
  kTerminal,
  kMoveCount,
  kScores,
  kPawnRow,
  kPawnCol,
  kWallsRemaining,
  kHWalls,
  kVWalls,
};
}  // namespace

void QuoridorState::hash_field_slot(
    Hasher& h, viz::FieldId fid,
    const std::vector<int>& idx) const {
  switch (fid) {
    case kCurrentPlayer:   h.add(current_player_); return;
    case kWinner:          h.add(winner_ + 1); return;
    case kTerminal:        h.add(terminal ? 1 : 0); return;
    case kMoveCount:       h.add(move_count); return;
    case kScores:
      h.add(scores[static_cast<size_t>(idx[0])] + 2); return;
    case kPawnRow:
      h.add(static_cast<int>(pawn_row[static_cast<size_t>(idx[0])]) + 1); return;
    case kPawnCol:
      h.add(static_cast<int>(pawn_col[static_cast<size_t>(idx[0])]) + 1); return;
    case kWallsRemaining:
      h.add(static_cast<int>(walls_remaining[static_cast<size_t>(idx[0])]) + 1); return;
    case kHWalls:
      h.add(static_cast<int>(h_walls[static_cast<size_t>(idx[0])])); return;
    case kVWalls:
      h.add(static_cast<int>(v_walls[static_cast<size_t>(idx[0])])); return;
  }
}

}  // namespace board_ai::quoridor
