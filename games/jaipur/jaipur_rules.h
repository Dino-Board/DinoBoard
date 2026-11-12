#pragma once
#include <array>
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

// A complete trade in count form: market goods taken, hand goods / camels
// (index kCamel) given.
struct JaipurTrade {
  std::array<int, kGoodTypes> take{};
  std::array<int, kGoodTypes + 1> give{};
};

// Every complete trade the mover can still finish from the current trade
// prefix (all legal trades when the prefix is empty). Reads the mover's hand.
std::vector<JaipurTrade> jaipur_trades(const JaipurData& d);
// The atom that continues the current prefix towards `t` (one of `t`'s atoms
// not yet chosen). `t` must extend the prefix.
ActionId jaipur_trade_next_atom(const JaipurData& d, const JaipurTrade& t);
// Whether the trade prefix is empty.
bool jaipur_trade_idle(const JaipurData& d);

}  // namespace board_ai::jaipur
