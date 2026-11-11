#pragma once

#include <vector>

#include "../search/net_mcts.h"

namespace board_ai::infer {

// Wraps two underlying policy/value evaluators so that during a single
// MCTS search rooted at `latest_seat`, opponent decision nodes mix the
// two networks: opponent network supplies the prior (policy expansion),
// latest network supplies the leaf value.
//
// Used by pool-selfplay tasks for the latest seat only. Self/root nodes
// (perspective == latest_seat) forward entirely to `latest_eval`,
// matching the unwrapped behavior. Opponent nodes
// (perspective != latest_seat) call BOTH evaluators on the same masked
// state, take priors from `opponent_eval`, and take values from
// `latest_eval`.
//
// Lifetime: stores raw pointers; both evaluators must outlive this
// wrapper. Constructed once per pool episode (or batch) by the binding;
// destroyed when its evaluators are.
class SplitPolicyValueEvaluator final : public search::IPolicyValueEvaluator {
 public:
  SplitPolicyValueEvaluator(
      const search::IPolicyValueEvaluator* latest_eval,
      const search::IPolicyValueEvaluator* opponent_eval,
      int latest_seat);

  bool evaluate(
      const IGameState& masked_state,
      int perspective_player,
      const IBeliefTracker* tracker,
      const std::vector<ActionId>& legal_actions,
      std::vector<float>* priors,
      std::vector<float>* values) const override;

  bool evaluate_with_features(
      const std::vector<float>& features,
      int num_players,
      int perspective_player,
      const std::vector<ActionId>& legal_actions,
      const std::vector<float>& legal_mask,
      std::vector<float>* priors,
      std::vector<float>* values) const override;

 private:
  const search::IPolicyValueEvaluator* latest_eval_;
  const search::IPolicyValueEvaluator* opponent_eval_;
  int latest_seat_;
};

}  // namespace board_ai::infer
