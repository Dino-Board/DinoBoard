#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "../../engine/core/belief_evaluator.h"
#include "../../engine/core/belief_feature_extractor.h"
#include "../../engine/core/belief_label_extractor.h"
#include "../../engine/core/belief_tracker.h"
#include "../../engine/core/game_interfaces.h"
#include "6nimmit_state.h"

namespace board_ai::sixnimmt {

// The network adapter of this game is the token encoder
// (6nimmit_token_encoder.h). This file holds the other half of imperfect
// information: what a search assumes about the cards it cannot see.
//
// Everything known for certain about the hidden cards is on the state:
// `used` (every card shown), my hand, the face-up played cards. Without a
// belief network the other players hold cards drawn uniformly from the
// rest, which is exact for the deal and ignores what their earlier plays
// suggest. With one (config/belief.json, type "autoregressive") their hands
// are sampled from it card by card, lowest first
// (engine/core/belief_sequence.h): prepare_for_root draws kBeliefSamples
// joint hands for the position and every world of that search uses one of
// them.
//
// randomize_unseen(state, observer):
//   - a face-down card of another player counts as not played yet: its slot
//     is cleared and the search plays that card out of the sampled hand;
//   - every other player's hand is redrawn from the unseen cards: as many
//     as that player holds (10 minus the rounds finished, one fewer while
//     their card of this round lies face up);
//   - the deck is whatever is left, and is stored nowhere: a search that
//     runs past the end of the hand deals the next one from it.

// What `observer` cannot place, read from the public slots and the ones the
// viz tensors show to the observer (truth, world and masked states alike).
template <int NPlayers>
struct HiddenCards {
  std::array<bool, kNumCards + 1> seen{};  // seen[c]: the observer knows where card c is
  std::array<bool, NPlayers> face_up{};    // a played card of this seat lies where the observer sees it
  std::array<int, NPlayers> hidden{};      // cards this seat holds that the observer cannot see (0 for itself)
};
template <int NPlayers>
HiddenCards<NPlayers> hidden_cards(const SixNimmtState<NPlayers>& state, int observer);

// Hands the belief net is asked for at a search root. A world picks one of
// them at random, so more samples than the search has worlds (mcts.json
// world_cap) would not be used.
constexpr int kBeliefSamples = 128;

template <int NPlayers>
class SixNimmtBeliefTracker final : public IBeliefTracker {
 public:
  // One finished round as the players saw it when they chose: the row ends
  // and lengths before any card of the round was placed, and each seat's card.
  struct Round {
    std::array<std::int8_t, kNumRows> ends{};
    std::array<std::int8_t, kNumRows> lens{};
    std::array<std::int8_t, NPlayers> cards{};
  };

  // payload "own_hand": the perspective's cards, for a session that was not
  // seeded like the truth state (observation sessions / web).
  void init(IGameState& state, int perspective, const AnyMap& payload) override;
  AnyMap pack_init_payload(const IGameState& gt_state, int perspective) const override;
  // Remembers the rounds from the `card_played` events (public, the same for
  // every seat). Only the belief features read them.
  void observe_public_event(int actor, ActionId action, const std::vector<PublicEvent>& events) override;
  void randomize_unseen(IGameState& state, int observer, std::mt19937_64& rng) const override;
  void prepare_for_root(const IGameState& root_state, int root_player, const IBeliefFeatureExtractor* extractor,
                        const IBeliefEvaluator* evaluator) override;
  std::unique_ptr<IBeliefTracker> clone() const override {
    return std::make_unique<SixNimmtBeliefTracker<NPlayers>>(*this);
  }
  // "rounds": per round [4 ends, 4 lens, one card per seat], flattened.
  AnyMap serialize() const override;

  const std::vector<Round>& rounds() const { return rounds_; }

 private:
  std::vector<Round> rounds_;
  // The belief net's hands ([sample][other seat in turn order after the
  // observer][cards ascending]), the observer they were drawn for and that
  // observer's hash of the root position: only worlds of that very position
  // use them. Shared: the search clones the tracker for every world.
  std::shared_ptr<const std::vector<std::vector<std::vector<int>>>> samples_;
  int samples_observer_ = -1;
  StateHash64 samples_state_ = 0;
};

// Belief features (the position as `perspective` knows it; the other seats in
// turn order after it): the cards it cannot place, its hand, per other seat
// the cards that seat played and how often each card lay between a row end
// and the higher card that seat put on that row; then every finished round
// (row ends and lengths ascending by end, each seat's card), the hidden hand
// sizes, bullheads and the rows now. The net's input adds the prefix of
// belief_sequence.h; its output is the next card of the hand being written.
template <int NPlayers>
class SixNimmtBeliefFeatureExtractor final : public IBeliefFeatureExtractor {
 public:
  static constexpr int kOthers = NPlayers - 1;
  static constexpr int kRoundDim = 2 * kNumRows + NPlayers;
  static constexpr int kScalars = 1 + 2 * kOthers + NPlayers + 2 * kNumRows;
  static constexpr int kFeatureDim = 2 * kNumCards + 2 * kOthers * kNumCards + kHandSize * kRoundDim + kScalars;
  int feature_dim() const override { return kFeatureDim; }
  int output_logit_count() const override { return kNumCards; }
  BeliefSequenceLayout sequence_layout() const override { return {kNumCards, kOthers}; }
  void extract(const IGameState& masked_state, int perspective_player, const IBeliefTracker* tracker,
               std::vector<float>* out) const override;
};

// Labels: hand_counts[i][c-1] = 1 for the cards the i-th seat after the
// observer holds that the observer cannot see (its hand, and its card of this
// round while that lies face down); remaining[c-1] = 1 for every card the
// observer cannot place (those hands and the deck).
template <int NPlayers>
class SixNimmtBeliefLabelExtractor final : public IBeliefLabelExtractor {
 public:
  int label_class_count() const override { return kNumCards; }
  void extract(const IGameState& truth_state, int observer, std::vector<std::vector<int>>* out_hand_counts,
               std::vector<int>* out_remaining, std::vector<int>* out_alive) const override;
};

extern template class SixNimmtBeliefTracker<2>;
extern template class SixNimmtBeliefTracker<3>;
extern template class SixNimmtBeliefTracker<4>;
extern template class SixNimmtBeliefFeatureExtractor<2>;
extern template class SixNimmtBeliefFeatureExtractor<3>;
extern template class SixNimmtBeliefFeatureExtractor<4>;
extern template class SixNimmtBeliefLabelExtractor<2>;
extern template class SixNimmtBeliefLabelExtractor<3>;
extern template class SixNimmtBeliefLabelExtractor<4>;

}  // namespace board_ai::sixnimmt
