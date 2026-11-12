#include "jaipur_smart_heuristic.h"
#include "../jaipur_rules.h"
#include "../jaipur_state.h"

namespace board_ai::jaipur {

HeuristicResult JaipurSmartHeuristic::pick(IGameState& masked_state,
                                            const IBeliefTracker* tracker,
                                            const IGameRules& rules,
                                            std::uint64_t rng_seed) const {
  const auto& d = static_cast<const JaipurState&>(masked_state).data;
  // Turn start → MCTS (deep search for sell/take/trade choice).
  // Later trade atoms → fast rule heuristic (best trade below the prefix).
  if (jaipur_trade_idle(d) && mcts_ != nullptr) {
    return mcts_->pick(masked_state, tracker, rules, rng_seed);
  }
  return rule_.pick(masked_state, tracker, rules, rng_seed);
}

}  // namespace board_ai::jaipur
