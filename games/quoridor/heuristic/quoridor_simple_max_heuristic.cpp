#include "quoridor_simple_max_heuristic.h"

#include "../quoridor_state.h"
#include "../quoridor_rules.h"

namespace board_ai::quoridor {

HeuristicResult QuoridorSimpleMaxHeuristic::pick(
    IGameState& masked_state,
    const IBeliefTracker* /*tracker*/,
    const IGameRules& rules,
    std::uint64_t /*rng_seed*/) const {
  HeuristicResult result;
  result.actions = rules.legal_actions(masked_state);
  result.scores.assign(result.actions.size(), 0.0);
  if (result.actions.empty()) return result;

  const auto& s = checked_cast<QuoridorState>(masked_state);
  const int me = s.current_player_;
  const int opp = 1 - me;

  for (size_t i = 0; i < result.actions.size(); ++i) {
    auto clone = masked_state.clone_state();
    rules.do_action_deterministic(*clone, result.actions[i]);
    const auto& after = checked_cast<QuoridorState>(*clone);
    const int d_me = QuoridorRules::shortest_path_distance(after, me);
    const int d_opp = QuoridorRules::shortest_path_distance(after, opp);
    result.scores[i] = static_cast<double>(d_opp - d_me);
  }
  return result;
}

}  // namespace board_ai::quoridor
