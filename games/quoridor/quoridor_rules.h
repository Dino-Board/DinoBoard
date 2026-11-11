#pragma once

#include <array>
#include <vector>

#include "../../engine/core/game_interfaces.h"
#include "quoridor_state.h"

namespace board_ai::quoridor {

class QuoridorRules final : public IGameRules {
 public:
  bool validate_action(const IGameState& state, ActionId action) const override;
  std::vector<ActionId> legal_actions(const IGameState& state) const override;

  static bool has_path_to_goal(const QuoridorState& state, int player);
  static int shortest_path_distance(const QuoridorState& state, int player);
  static void compute_distance_map(const QuoridorState& state, int player,
                                   std::array<int, kCellCount>* out);

 protected:
  void do_action_fast_impl(IGameState& state, ActionId action,
                           std::mt19937_64& rng) const override;
  UndoToken do_action_deterministic_impl(IGameState& state,
                                         ActionId action) const override;
  void undo_action_impl(IGameState& state, const UndoToken& token) const override;

 private:
  // Pure action application — no undo bookkeeping, no rng (Quoridor is
  // fully deterministic). Shared between the fast and deterministic
  // entries; only the deterministic side wraps it with undo_stack push.
  static void apply_action_no_undo(QuoridorState& s, ActionId action);
};

}  // namespace board_ai::quoridor
