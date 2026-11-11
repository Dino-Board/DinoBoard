#pragma once

#include <vector>

#include "../../engine/core/game_interfaces.h"
#include "loveletter_state.h"

namespace board_ai::loveletter {

template <int NPlayers>
class LoveLetterRules final : public IGameRules {
 public:
  using Cfg = LoveLetterConfig<NPlayers>;

  bool validate_action(const IGameState& state, ActionId action) const override;
  std::vector<ActionId> legal_actions(const IGameState& state) const override;

  // Apply the start-of-game viz reveals: the seat starting as
  // current_player has just drawn the top card and physically holds it.
  // Mirrors what advance_turn does after every subsequent draw. Lives
  // in rules so I1 (rules are sole viz writer) holds — called once
  // from `reset_with_seed` (truth) and from `apply_initial_observation`
  // (per-seat session) after init_viz seeds schema base.
  static void reveal_starting_draw(IGameState& state, int starting_player);
  // Reveal `drawn_card` only to the named viewer (used by API session
  // restore: when perspective is the starting current_player, ship the
  // truth card to them but leave it hidden to other potential viewers).
  static void reveal_starting_draw_to(IGameState& state, int viewer);

 protected:
  void do_action_fast_impl(IGameState& state, ActionId action,
                           std::mt19937_64& rng) const override;
  UndoToken do_action_deterministic_impl(IGameState& state,
                                         ActionId action) const override;
  void undo_action_impl(IGameState& state, const UndoToken& token) const override;
};

extern template class LoveLetterRules<2>;
extern template class LoveLetterRules<3>;
extern template class LoveLetterRules<4>;

}  // namespace board_ai::loveletter
