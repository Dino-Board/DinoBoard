#include "tictactoe_state.h"

#include "../../engine/core/viz_runtime.h"

namespace board_ai::tictactoe {

const viz::VisibilitySchema& TicTacToeState::schema() {
  // Built once on first use, then returned by reference. Same object for
  // all tictactoe states across the process — schema is per-game-template,
  // not per-state.
  static const viz::VisibilitySchema s = []() {
    viz::VisibilitySchema schema;
    schema.n_players = kPlayers;
    // Scalars use shape {} (rank 0); 1D arrays use shape {len}.
    viz::declare_field(schema, "current_player",
                       viz::all_public({}, kPlayers));
    viz::declare_field(schema, "winner", viz::all_public({}, kPlayers));
    viz::declare_field(schema, "terminal", viz::all_public({}, kPlayers));
    viz::declare_field(schema, "move_count", viz::all_public({}, kPlayers));
    viz::declare_field(schema, "scores",
                       viz::all_public({kPlayers}, kPlayers));
    viz::declare_field(schema, "board",
                       viz::all_public({kBoardSize}, kPlayers));
    return schema;
  }();
  return s;
}

TicTacToeState::TicTacToeState() {
  reset_with_seed(0xC0FFEEu);
}

void TicTacToeState::reset_with_seed(std::uint64_t seed) {
  IGameState::reset_step_count_base();
  (void)seed;  // tictactoe is deterministic; seed unused.
  current_player_ = 0;
  winner_ = -1;
  terminal = false;
  move_count = 0;
  scores = {0, 0};
  board.fill(static_cast<std::int8_t>(kEmptyCell));
  undo_stack.clear();
  viz::init_viz(*this, schema());
}

StateHash64 TicTacToeState::state_hash() const {
  std::size_t h = 0;
  hash_combine(h, static_cast<std::size_t>(current_player_));
  hash_combine(h, static_cast<std::size_t>(winner_ + 1));
  hash_combine(h, static_cast<std::size_t>(terminal ? 1 : 0));
  hash_combine(h, static_cast<std::size_t>(move_count));
  hash_combine(h, static_cast<std::size_t>(scores[0] + 2));
  hash_combine(h, static_cast<std::size_t>(scores[1] + 2));
  for (std::int8_t c : board) {
    hash_combine(h, static_cast<std::size_t>(c + 2));
  }
  return static_cast<StateHash64>(h);
}

// Field IDs match the schema declaration order in schema(). Walker
// hands us a FieldId; we switch on it instead of doing a string
// compare per slot.
namespace {
enum : viz::FieldId {
  kCurrentPlayer = 0,
  kWinner,
  kTerminal,
  kMoveCount,
  kScores,
  kBoard,
};
}  // namespace

void TicTacToeState::hash_field_slot(
    Hasher& h, viz::FieldId fid,
    const std::vector<int>& idx) const {
  switch (fid) {
    case kCurrentPlayer: h.add(current_player_); return;
    case kWinner:        h.add(winner_ + 1); return;
    case kTerminal:      h.add(terminal ? 1 : 0); return;
    case kMoveCount:     h.add(move_count); return;
    case kScores:
      h.add(scores[static_cast<size_t>(idx[0])] + 2); return;
    case kBoard:
      h.add(static_cast<int>(board[static_cast<size_t>(idx[0])]) + 2); return;
  }
}

}  // namespace board_ai::tictactoe
