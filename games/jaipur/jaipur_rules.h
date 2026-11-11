#pragma once
#include <vector>
#include "../../engine/core/game_interfaces.h"
#include "jaipur_state.h"

namespace board_ai::jaipur {

class JaipurRules final : public IGameRules {
 public:
  JaipurRules() = default;

  bool validate_action(const IGameState& state, ActionId action) const override;
  std::vector<ActionId> legal_actions(const IGameState& state) const override;

 protected:
  void do_action_fast_impl(IGameState& state, ActionId action,
                           std::mt19937_64& rng) const override;
  UndoToken do_action_deterministic_impl(IGameState& state, ActionId action) const override;
  void undo_action_impl(IGameState& state, const UndoToken& token) const override;
};

}  // namespace board_ai::jaipur
