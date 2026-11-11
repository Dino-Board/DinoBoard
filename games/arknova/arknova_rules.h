#pragma once

// Ark Nova rules over the factored action space (arknova_actions.h). An action is one atom; the
// core option is applied once the prefix spells it out, and decisions with a single possible
// continuation are applied without asking. The tail solver is not supported: deterministic
// application and undo throw.

#include <random>
#include <vector>

#include "../../engine/core/game_interfaces.h"
#include "arknova_state.h"

namespace board_ai::arknova {

class ArknovaRules final : public IGameRules {
 public:
  bool validate_action(const IGameState& state, ActionId action) const override;
  std::vector<ActionId> legal_actions(const IGameState& state) const override;
  // A card tucked by Pouch stays hidden, so its hand_discard atom is observed as atom::kHiddenHandCard.
  ActionId public_action(const IGameState& state_before, ActionId action) const override;

 protected:
  void do_action_fast_impl(IGameState& state, ActionId action, std::mt19937_64& rng) const override;
  UndoToken do_action_deterministic_impl(IGameState& state, ActionId action) const override;
  void undo_action_impl(IGameState& state, const UndoToken& token) const override;
};

// After the core changed: absorb it into the adapter fields (ArknovaState::absorb_core) and show
// a hand entry to the opponent exactly while it holds a card that entered the hand publicly.
void absorb_step(ArknovaState& st);

// Applies decisions that need no choice (a single possible continuation from the prefix) until
// a real choice or the end of the game.
void advance_forced(ArknovaState& st, std::mt19937_64& rng);

// Appends atom `a` to the prefix and applies the option once it is complete.
void apply_atom(ArknovaState& st, ActionId a, std::mt19937_64& rng);

}  // namespace board_ai::arknova
