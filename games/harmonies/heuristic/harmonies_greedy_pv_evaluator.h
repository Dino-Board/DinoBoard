#pragma once

#include <string>

#include "../../../engine/core/game_interfaces.h"
#include "../../../engine/search/net_mcts.h"
#include "../harmonies_state.h"

namespace board_ai::harmonies {

template <int NPlayers, int Side>
class HarmoniesGreedyPVEvaluator final
    : public board_ai::search::IPolicyValueEvaluator {
 public:
  void set_rules(const IGameRules* r) { rules_ = r; }

  bool evaluate(
      const IGameState& masked_state,
      int perspective_player,
      const IBeliefTracker* tracker,
      const std::vector<ActionId>& legal_actions,
      std::vector<float>* priors,
      std::vector<float>* values) const override;

 private:
  const IGameRules* rules_ = nullptr;
};

#define HARMONIES_EXTERN_HarmoniesGreedyPVEvaluator(N, S) extern template class HarmoniesGreedyPVEvaluator<N, S>;
HARMONIES_FOR_EACH_VARIANT(HARMONIES_EXTERN_HarmoniesGreedyPVEvaluator)
#undef HARMONIES_EXTERN_HarmoniesGreedyPVEvaluator

}  // namespace board_ai::harmonies
