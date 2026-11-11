#pragma once

#include <vector>

#include "../../engine/core/game_interfaces.h"
#include "tictactoe_state.h"

namespace board_ai::tictactoe {

class TicTacToeRules final : public IGameRules {
 public:
  bool validate_action(const IGameState& state, ActionId action) const override;
  std::vector<ActionId> legal_actions(const IGameState& state) const override;

 protected:
  void do_action_fast_impl(IGameState& state, ActionId action,
                           std::mt19937_64& rng) const override;
  void undo_action_impl(IGameState& state, const UndoToken& token) const override;

 private:
  static int evaluate_winner(const TicTacToeState& state);
};

}  // namespace board_ai::tictactoe
