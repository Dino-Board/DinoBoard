#pragma once

#include <vector>

#include "../../engine/core/game_interfaces.h"
#include "6nimmit_state.h"

namespace board_ai::sixnimmt {

// Rules of one hand (see 6nimmit_state.h).
//
// A round: every player plays one card face down (one ply per seat, in seat
// order; a card stays hidden from the other seats until the last one is
// played). With the last card all are turned face up and placed in ascending
// order inside the same do_action_fast call:
//   - a card goes to the row whose end is the highest one below it;
//   - as the sixth card of a row it takes the five (their bullheads go to
//     its owner) and starts the row anew;
//   - a card below every row end pauses the round: its owner chooses a row
//     to take (one extra ply), then placement continues. This can only be
//     the lowest card of the round, so it happens at most once a round and
//     before any other card has been placed.
// The game ends after the tenth round. Nothing is random after the deal.
//
// No undo, no tail solver.
template <int NPlayers>
class SixNimmtRules final : public IGameRules {
 public:
  using Cfg = SixNimmtConfig<NPlayers>;

  bool validate_action(const IGameState& state, ActionId action) const override;
  std::vector<ActionId> legal_actions(const IGameState& state) const override;
  // A card that leaves the round open is played face down: the other seats
  // observe kHiddenCommitAction instead of the card.
  ActionId public_action(const IGameState& state_before, ActionId action) const override;

 protected:
  void do_action_fast_impl(IGameState& state, ActionId action, std::mt19937_64& rng) const override;
  void undo_action_impl(IGameState& state, const UndoToken& token) const override;
};

extern template class SixNimmtRules<2>;
extern template class SixNimmtRules<3>;
extern template class SixNimmtRules<4>;

}  // namespace board_ai::sixnimmt
