#pragma once

#include "../../engine/core/game_interfaces.h"
#include "liarsdice_state.h"

namespace board_ai::liarsdice {

template <int NPlayers>
class LiarsdiceRules final : public IGameRules {
 public:
  bool validate_action(const IGameState& state, ActionId action) const override;
  std::vector<ActionId> legal_actions(const IGameState& state) const override;

 protected:
  void do_action_fast_impl(IGameState& state, ActionId action,
                           std::mt19937_64& rng) const override;
  void undo_action_impl(IGameState& state, const UndoToken& token) const override;
};

extern template class LiarsdiceRules<2>;
extern template class LiarsdiceRules<3>;
extern template class LiarsdiceRules<4>;

// Decode helpers shared with encoder/tracker so they don't redefine the
// (X, Y) layout. Keep these in sync with `LiarsdiceConfig::kBidCount`.
inline bool is_challenge(ActionId a, int total_dice) {
  return a == total_dice * 5;
}
inline int bid_count_from_action(ActionId a) { return a / 5 + 1; }
inline int bid_face_from_action(ActionId a) { return a % 5 + 2; }
inline ActionId encode_bid(int x, int y) {
  // x ∈ 1..5N, y ∈ 2..6
  return static_cast<ActionId>((x - 1) * 5 + (y - 2));
}

}  // namespace board_ai::liarsdice
