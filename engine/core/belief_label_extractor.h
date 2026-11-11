#pragma once

#include <vector>

#include "game_interfaces.h"

namespace board_ai {

// Belief-network label extractor (Coup belief net plan §2.2).
//
// GT-side counterpart to IBeliefFeatureExtractor. Reads the **truth**
// IGameState and emits the raw counts the training loop turns into the
// label distribution. Runs only on the selfplay GT runner — never on
// the AI session, never inside MCTS, never on the wire. The feature
// path (masked state + tracker) and the label path (truth) are
// physically separated; the AI session has no access to this
// extractor at all (training samples are emit→file→train, exactly
// like the existing pv samples).
//
// Output layout per call (sized to (observer, N-1, K)):
//   - hand_counts[i][R] = number of unrevealed R-cards in opp i's hand,
//     where opp i = opp_to_player(observer, i, NPlayers).
//   - remaining[R]      = total_R - observer's own unrevealed R - publicly
//                         revealed R (across all seats). Pre-sample full
//                         unseen pool; shared across all opp rows.
//   - alive_per_opp[i]  = 1 iff opp i still has at least one unrevealed
//                         influence; 0 otherwise (used as row mask in
//                         loss).
//
// The framework does not interpret these values; it just plumbs them
// from selfplay_runner into the SelfplaySample so the Python training
// loop can compute targets per Plan §2.2's q/remaining formula. Per-game
// implementations decide the K class space and what "card" / "role"
// means; the framework only fixes the (N-1, K) shape and "alive_per_opp"
// semantics.
class IBeliefLabelExtractor {
 public:
  virtual ~IBeliefLabelExtractor() = default;

  // K class count (Coup: 5, LL: 8). Used by the training loop / sample
  // serializer to interpret hand_counts / remaining shape.
  virtual int label_class_count() const = 0;

  // Fill the three label tensors. `out_hand_counts` is sized
  // (NPlayers - 1, K); `out_remaining` is sized K; `out_alive` is sized
  // (NPlayers - 1).
  virtual void extract(
      const IGameState& truth_state,
      int observer,
      std::vector<std::vector<int>>* out_hand_counts,
      std::vector<int>* out_remaining,
      std::vector<int>* out_alive) const = 0;
};

}  // namespace board_ai
