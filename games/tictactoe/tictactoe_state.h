#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "../../engine/core/game_interfaces.h"
#include "../../engine/core/visibility_schema.h"

namespace board_ai::tictactoe {

constexpr int kBoardSize = 9;
constexpr int kPlayers = 2;
constexpr int kEmptyCell = -1;

struct UndoRecord {
  int action = -1;
  int prev_player = 0;
  int prev_winner = -1;
  bool prev_terminal = false;
  int prev_move_count = 0;
  std::array<int, kPlayers> prev_scores{};
};

struct TicTacToeState final : public CloneableState<TicTacToeState> {
  int current_player_ = 0;
  int winner_ = -1;
  bool terminal = false;
  int move_count = 0;
  std::array<int, kPlayers> scores{0, 0};
  std::array<std::int8_t, kBoardSize> board{};
  std::vector<UndoRecord> undo_stack{};

  TicTacToeState();

  // Visibility schema. Tictactoe is fully public, so every field
  // declares all_public viz. Keeps shape contracts in one place for
  // framework consumers (snapshot extractor, hash walker, encoder
  // masker). step_count is framework-managed and out of scope.
  static const viz::VisibilitySchema& schema();

  StateHash64 state_hash() const override;
  void hash_field_slot(Hasher& h, viz::FieldId fid,
                       const std::vector<int>& idx) const override;
  const viz::VisibilitySchema& schema_ref() const override { return schema(); }
  int current_player() const override { return current_player_; }
  bool is_terminal() const override { return terminal; }
  int num_players() const override { return kPlayers; }
  int winner() const override { return winner_; }


  void reset_with_seed(std::uint64_t seed) override;
};

}  // namespace board_ai::tictactoe
