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
// Per-good-type "net inflow" bookkeeping (the core invariant the user asked
// for). For each player p and goods type g the tracker maintains:
//   sold_[p][g]  — has p ever sold type g?
//   net_in_[p][g] — net goods of type g that p has taken IN (take +1;
//                   exchange += take_count - give_count, clamped ≥ 0; sell → 0).
//
// Invariants:
//   - If sold_[p][g] is true: net_in_[p][g] EXACTLY equals the number of
//     type-g cards p currently holds. Selling zeros the count AND erases the
//     unknown starting-hand contribution, so every take after that is the
//     whole truth. This is the "strong sell signal".
//   - If sold_[p][g] is false: net_in_[p][g] is a LOWER BOUND on p's held
//     type-g count — the surplus over net_in is p's (unknown) starting hand
//     of type g. That surplus is sampled uniformly from the still-unsold
//     types' unseen pool.
//
// The opponent's hand is reconstructed at sample time: sold types are filled
// exactly from net_in; the residual (hand_size − Σ net_in over sold types,
// distributed across unsold types) is uniform-sampled from the unsold-types
// unseen pool. Late game (many types sold) the hand is near-fully determined.
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

 private:
  // Per-player net-inflow bookkeeping (perspective-agnostic: both players
  // tracked; the observer samples its opponent's row). [player][good_type].
  std::array<std::array<bool, kGoodTypes>, kPlayers> sold_{};
  std::array<std::array<int, kGoodTypes>, kPlayers> net_in_{};
};

}  // namespace board_ai::jaipur
