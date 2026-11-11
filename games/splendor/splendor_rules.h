#pragma once

#include <vector>

#include "../../engine/core/game_interfaces.h"
#include "splendor_state.h"

namespace board_ai::splendor {

template <int NPlayers>
class SplendorRules final : public IGameRules {
 public:
  using Cfg = SplendorConfig<NPlayers>;

  SplendorRules() = default;

  bool validate_action(const IGameState& state, ActionId action) const override;
  std::vector<ActionId> legal_actions(const IGameState& state) const override;

  static bool is_terminal_data(const SplendorData<NPlayers>& data);
  static std::vector<ActionId> legal_actions_data(const SplendorData<NPlayers>& d);
  // Caller-owned rng feeds tableau refills / random draws. Deterministic
  // path passes a throwaway rng since forced_draw_override freezes draws.
  static SplendorData<NPlayers> apply_action_copy(
      const SplendorData<NPlayers>& src, ActionId action,
      std::mt19937_64& rng);

 protected:
  void do_action_fast_impl(IGameState& state, ActionId action,
                           std::mt19937_64& rng) const override;
  UndoToken do_action_deterministic_impl(IGameState& state, ActionId action) const override;
  void undo_action_impl(IGameState& state, const UndoToken& token) const override;
};

extern template class SplendorRules<2>;
extern template class SplendorRules<3>;
extern template class SplendorRules<4>;

}  // namespace board_ai::splendor
