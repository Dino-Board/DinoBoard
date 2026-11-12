#pragma once
#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include "../../engine/core/belief_tracker.h"
#include "../../engine/core/game_interfaces.h"
#include "jaipur_state.h"

namespace board_ai::jaipur {

// Belief tracker for Jaipur.
//
// Hidden info:
//   1. The opponent's hand *contents* (which good types they hold). Hand
//      *size* and herd are public.
//   2. The opponent's collected bonus-token *values* (face-down).
//
// Per player p and goods type g the tracker keeps known_[p][g], a LOWER
// BOUND on the type-g cards p holds, from public goods flow only:
//   take k of g (single take or trade)   known += k
//   sell / give k of g                   known = max(known, k) - k
// (a sale or a give of k proves p held at least k). The cards of p's hand
// beyond Σ known are unknown cards of the opening hand; randomize_unseen
// draws them uniformly from the unseen pool. When Σ known equals the public
// hand size the hand is known exactly.
//
// Bonus values: the tracker keeps no bonus memory. Which values the observer
// itself drew is perspective-private knowledge and lives on state
// (bonus_taken, owner-only); randomize_unseen reads it from there to build
// the residual value multiset for the opponent's tokens and the piles.
//
// The draw pile has no order (jaipur_state.h): randomize_unseen only fills
// its hidden per-type composition, consistent with the sampled hand.
class JaipurBeliefTracker final : public IBeliefTracker {
 public:
  void init(IGameState& state, int perspective, const AnyMap& payload) override;
  AnyMap pack_init_payload(const IGameState& gt_state,
                           int perspective) const override;
  void observe_public_event(int actor, ActionId action,
                            const std::vector<PublicEvent>& events) override;
  void randomize_unseen(IGameState& state, int observer,
                        std::mt19937_64& rng) const override;
  std::unique_ptr<IBeliefTracker> clone() const override {
    return std::make_unique<JaipurBeliefTracker>(*this);
  }
  AnyMap serialize() const override;

  int known(int player, int good) const { return known_[player][good]; }

 private:
  void sample_world(JaipurState& s, int observer, std::mt19937_64& rng) const;

  // [player][good_type] lower bound of held cards (perspective-agnostic).
  std::array<std::array<int, kGoodTypes>, kPlayers> known_{};
};

}  // namespace board_ai::jaipur
