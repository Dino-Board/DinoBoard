#pragma once

#include "../../engine/core/game_interfaces.h"
#include "maskmen_state.h"

namespace board_ai::maskmen {

template <int NPlayers>
class MaskmenRules final : public IGameRules {
 public:
  bool validate_action(const IGameState& state, ActionId action) const override;
  std::vector<ActionId> legal_actions(const IGameState& state) const override;

 protected:
  void do_action_fast_impl(IGameState& state, ActionId action,
                           std::mt19937_64& rng) const override;
  void undo_action_impl(IGameState& state, const UndoToken& token) const override;
};

extern template class MaskmenRules<2>;
extern template class MaskmenRules<3>;
extern template class MaskmenRules<4>;

}  // namespace board_ai::maskmen
