#include "random_heuristic.h"

namespace board_ai::heuristic {

HeuristicResult RandomHeuristic::pick(IGameState& masked_state,
                                      const IBeliefTracker* /*tracker*/,
                                      const IGameRules& rules,
                                      std::uint64_t /*rng_seed*/) const {
  HeuristicResult out;
  out.actions = rules.legal_actions(masked_state);
  out.scores.assign(out.actions.size(), 1.0);
  return out;
}

}  // namespace board_ai::heuristic
