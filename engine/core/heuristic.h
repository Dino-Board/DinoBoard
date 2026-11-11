#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "game_interfaces.h"

namespace board_ai {

class IBeliefTracker;
class IGameRules;
class IGameState;

// HeuristicResult is the contract every IHeuristic returns. `actions`
// and `scores` must have the same length, with `scores[i]` being the
// preference for `actions[i]`. The selection rule (argmax-with-tie or
// softmax(T)) is applied by the caller via runtime::sample_heuristic_index,
// not by the heuristic itself.
struct HeuristicResult {
  std::vector<ActionId> actions;
  std::vector<double> scores;
  // MCTS simulations the heuristic ran to produce this result; 0 for
  // heuristics that do not search. Reporting only (throughput metrics) —
  // callers must not base the move choice on it.
  int simulations = 0;
};

// Heuristic interface.
//
// Signature parity with IFeatureEncoder: takes a perspective-masked
// state (viz=0 slots = kPlaceholder, structurally identical to what the
// encoder sees) plus an optional tracker carrying public-derived state.
// `tracker` may be nullptr for fully-public games (TicTacToe, Quoridor,
// Azul) — implementations that need tracker info must check for null
// and fall back to public-state-only reasoning.
//
// `masked_state` is non-const because some heuristics use
// `do_action_deterministic + undo_action` for shallow lookahead on the
// clone. The clone is fresh per call (created by the caller via
// make_masked_state); mutations don't escape.
//
// `rng_seed` lets the heuristic be deterministic given the (state, seed)
// pair — each call gets a fresh seed derived from the episode seed.
class IHeuristic {
 public:
  virtual ~IHeuristic() = default;
  // Stable identifier for logging/diagnostics. Should match the key in
  // heuristic.json::named under which the instance is registered.
  virtual std::string name() const = 0;

  virtual HeuristicResult pick(IGameState& masked_state,
                               const IBeliefTracker* tracker,
                               const IGameRules& rules,
                               std::uint64_t rng_seed) const = 0;
};

}  // namespace board_ai
