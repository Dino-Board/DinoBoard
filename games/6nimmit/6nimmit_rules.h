#pragma once

#include <vector>

#include "../../engine/core/game_interfaces.h"
#include "6nimmit_state.h"

namespace board_ai::sixnimmt {

// 6 nimmt! rules. Two action kinds (§1.4):
//   COMMIT_CARD       (action_id ∈ [0, 103], card = action_id + 1)
//   SMALL_CHOOSE_ROW  (action_id ∈ [104, 107], row = action_id - 104)
//
// Commit phase serializes simultaneous commits seat-by-seat. After the
// N-th commit, rules flip every pending_commits slot's viz to all-
// public (the simultaneous reveal) and run the deterministic post-
// processing loop *inside the same `do_action_fast` call*, consuming
// pending entries in ascending-card order:
//   - regular insert (find largest row tail < card) → no decision
//   - row would become 6 → eat first 5 into bullhead, new card heads → no decision
//   - card smaller than all 4 row tails → SMALL pause: small_phase_active=1,
//     small_player=p, current_player=p, break out of the loop and return.
// Subsequent SMALL_CHOOSE_ROW consumes the small_player's pending slot
// and resumes the loop.
//
// Tail solver is NOT supported (§3.4); `do_action_deterministic` /
// `undo_action` are intentionally not implemented — the inherited default
// would forward to `do_action_fast_impl`, but we override `undo_action_impl`
// to throw fail-loud so misuse does not silently produce broken state.
template <int NPlayers>
class SixNimmtRules final : public IGameRules {
 public:
  using Cfg = SixNimmtConfig<NPlayers>;

  bool validate_action(const IGameState& state, ActionId action) const override;
  std::vector<ActionId> legal_actions(const IGameState& state) const override;
  // A commit that leaves the round open is played face down: observers
  // see kHiddenCommitAction, not the card id. The round-completing commit
  // (revealed at once) and SMALL_CHOOSE_ROW are public and pass through.
  ActionId public_action(const IGameState& state_before,
                         ActionId action) const override;

 protected:
  void do_action_fast_impl(IGameState& state, ActionId action,
                           std::mt19937_64& rng) const override;
  void undo_action_impl(IGameState& state, const UndoToken& token) const override;
};

extern template class SixNimmtRules<2>;
extern template class SixNimmtRules<3>;
extern template class SixNimmtRules<4>;

}  // namespace board_ai::sixnimmt
