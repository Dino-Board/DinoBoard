#include "split_policy_value_evaluator.h"

#include <stdexcept>

namespace board_ai::infer {

SplitPolicyValueEvaluator::SplitPolicyValueEvaluator(
    const search::IPolicyValueEvaluator* latest_eval,
    const search::IPolicyValueEvaluator* opponent_eval,
    int latest_seat)
    : latest_eval_(latest_eval),
      opponent_eval_(opponent_eval),
      latest_seat_(latest_seat) {
  if (!latest_eval_) {
    throw std::invalid_argument(
        "SplitPolicyValueEvaluator: latest_eval must not be null");
  }
  if (!opponent_eval_) {
    throw std::invalid_argument(
        "SplitPolicyValueEvaluator: opponent_eval must not be null");
  }
  if (latest_seat_ < 0) {
    throw std::invalid_argument(
        "SplitPolicyValueEvaluator: latest_seat must be >= 0");
  }
}

bool SplitPolicyValueEvaluator::evaluate(
    const IGameState& masked_state,
    int perspective_player,
    const IBeliefTracker* tracker,
    const std::vector<ActionId>& legal_actions,
    std::vector<float>* priors,
    std::vector<float>* values) const {
  if (perspective_player == latest_seat_) {
    return latest_eval_->evaluate(
        masked_state, perspective_player, tracker, legal_actions,
        priors, values);
  }

  // Opponent node: priors from opponent network, values from latest.
  // Both are evaluated on the same masked state — the encoder is
  // perspective-driven, so each evaluator masks/encodes for
  // `perspective_player` (the opponent seat) on its own ORT session.
  std::vector<float> opp_values_unused;
  if (!opponent_eval_->evaluate(
          masked_state, perspective_player, tracker, legal_actions,
          priors, &opp_values_unused)) {
    return false;
  }

  std::vector<float> latest_priors_unused;
  if (!latest_eval_->evaluate(
          masked_state, perspective_player, tracker, legal_actions,
          &latest_priors_unused, values)) {
    return false;
  }
  return true;
}

bool SplitPolicyValueEvaluator::evaluate_with_features(
    const std::vector<float>& features,
    int num_players,
    int perspective_player,
    const std::vector<ActionId>& legal_actions,
    const std::vector<float>& legal_mask,
    std::vector<float>* priors,
    std::vector<float>* values) const {
  if (perspective_player == latest_seat_) {
    return latest_eval_->evaluate_with_features(
        features, num_players, perspective_player, legal_actions,
        legal_mask, priors, values);
  }

  // Opponent perspective: priors from opponent network, values from
  // latest. Mirrors the per-node split semantics in `evaluate(...)`.
  std::vector<float> opp_values_unused;
  if (!opponent_eval_->evaluate_with_features(
          features, num_players, perspective_player, legal_actions,
          legal_mask, priors, &opp_values_unused)) {
    return false;
  }
  std::vector<float> latest_priors_unused;
  if (!latest_eval_->evaluate_with_features(
          features, num_players, perspective_player, legal_actions,
          legal_mask, &latest_priors_unused, values)) {
    return false;
  }
  return true;
}

}  // namespace board_ai::infer
