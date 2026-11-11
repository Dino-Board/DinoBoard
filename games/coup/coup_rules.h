#pragma once

#include "../../engine/core/game_interfaces.h"
#include "coup_state.h"

namespace board_ai::coup {

template <int NPlayers>
class CoupRules final : public IGameRules {
 public:
  bool validate_action(const IGameState& state, ActionId action) const override;
  std::vector<ActionId> legal_actions(const IGameState& state) const override;
  // An exchange return names the character put back into the court deck,
  // which only the exchanging player may know: every observer sees
  // kHiddenReturnAction instead.
  ActionId public_action(const IGameState& state_before,
                         ActionId action) const override;

 protected:
  void do_action_fast_impl(IGameState& state, ActionId action,
                           std::mt19937_64& rng) const override;
  void undo_action_impl(IGameState& state, const UndoToken& token) const override;
};

extern template class CoupRules<2>;
extern template class CoupRules<3>;
extern template class CoupRules<4>;

}  // namespace board_ai::coup
