#pragma once

#include <memory>
#include <random>
#include <vector>

#include "../../engine/core/belief_tracker.h"
#include "../../engine/core/game_interfaces.h"
#include "6nimmit_state.h"

namespace board_ai::sixnimmt {

// The network adapter of this game is the token encoder
// (6nimmit_token_encoder.h). This file holds the other half of imperfect
// information: what a search assumes about the cards it cannot see.
//
// Belief tracker without memory and without a belief network. Everything
// known about the hidden cards is on the state: `used` (every card shown),
// my hand, the face-up played cards. The other
// players hold cards drawn uniformly from the rest, which is exact for the
// deal and ignores only what their earlier plays suggest.
//
// randomize_unseen(state, observer):
//   - a face-down card of another player counts as not played yet: its slot
//     is cleared and the search plays that card out of the sampled hand;
//   - every other player's hand is redrawn from the unseen cards: as many
//     as that player holds (10 minus the rounds finished, one fewer while
//     their card of this round lies face up);
//   - the deck is whatever is left, and is stored nowhere: a search that
//     runs past the end of the hand deals the next one from it.
template <int NPlayers>
class SixNimmtBeliefTracker final : public IBeliefTracker {
 public:
  // payload "own_hand": the perspective's cards, for a session that was not
  // seeded like the truth state (AI API / web).
  void init(IGameState& state, int perspective, const AnyMap& payload) override;
  AnyMap pack_init_payload(const IGameState& gt_state, int perspective) const override;
  void observe_public_event(int actor, ActionId action, const std::vector<PublicEvent>& events) override;
  void randomize_unseen(IGameState& state, int observer, std::mt19937_64& rng) const override;
  std::unique_ptr<IBeliefTracker> clone() const override {
    return std::make_unique<SixNimmtBeliefTracker<NPlayers>>(*this);
  }
};

extern template class SixNimmtBeliefTracker<2>;
extern template class SixNimmtBeliefTracker<3>;
extern template class SixNimmtBeliefTracker<4>;

}  // namespace board_ai::sixnimmt
