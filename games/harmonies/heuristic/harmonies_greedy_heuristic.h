#pragma once

#include <string>

#include "../../../engine/core/heuristic.h"
#include "../harmonies_state.h"

namespace board_ai::harmonies {

// Greedy "build terrain to land animals" heuristic.
//
// Strategy:
//   1. If any PLACE_CUBE is legal -> take it (largest immediate animal_score
//      gain). Completing a card always wins over keeping it.
//   2. Otherwise, if turn_started=false: pick the SELECT_SLOT whose drawn
//      tokens we can best use right now (heuristic match with currently-held
//      cards and best tower-building opportunities).
//   3. Otherwise, if turn_started AND pool non-empty: pick the PLACE_TOKEN
//      that brings the board closer to a held card's pattern (highest "score
//      delta to completing some card"); break ties with landscape value.
//   4. Otherwise, if hand has fewer than 4 cards and TAKE_CARD legal:
//      take the card whose pattern best fits the current board / tokens.
//   5. Otherwise (pool empty, turn_started): END_TURN.
//
// All scores are computed by simulating the action on a cloned state and
// re-evaluating a static board-quality function.
template <int NPlayers, int Side>
class HarmoniesGreedyHeuristic final : public IHeuristic {
 public:
  explicit HarmoniesGreedyHeuristic(std::string instance_name = "greedy")
      : name_(std::move(instance_name)) {}

  std::string name() const override { return name_; }

  HeuristicResult pick(IGameState& masked_state,
                       const IBeliefTracker* tracker,
                       const IGameRules& rules,
                       std::uint64_t rng_seed) const override;

 private:
  std::string name_;
};

#define HARMONIES_EXTERN_HarmoniesGreedyHeuristic(N, S) extern template class HarmoniesGreedyHeuristic<N, S>;
HARMONIES_FOR_EACH_VARIANT(HARMONIES_EXTERN_HarmoniesGreedyHeuristic)
#undef HARMONIES_EXTERN_HarmoniesGreedyHeuristic

}  // namespace board_ai::harmonies
