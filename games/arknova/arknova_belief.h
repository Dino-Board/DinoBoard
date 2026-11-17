#pragma once

// Hidden-information side of the Ark Nova adapter: belief tracker (determinization), belief-net
// feature extractor and label extractor.
//
// The tracker is stateless: everything it needs is public state plus the observer's visible
// slots. Unseen cards split into two pools by deck:
//   regular  (animals, sponsors, projects): the opponent's hidden hand slots, tucked cards and
//            Perception/Scavenging draws, then the deck;
//   endgame: the opponent's final cards and Resistance draws, then the final deck (the observer's
//            own returned endgame cards stay at its bottom).
// The counts are public, so a pool whose size does not match its slots means a card location is
// missing from the derivation; that throws.
//
// Belief net (belief.json type "autoregressive", engine/core/belief_sequence.h): the opponent's
// hidden hand cards from the regular pool, lowest CardIdx first. prepare_for_root draws
// kBeliefSamples hands for the position and every world of that search uses one of them;
// everything else (tucked cards, private draws, the decks) is uniform.

#include <array>
#include <memory>
#include <random>
#include <vector>

#include "../../engine/core/belief_evaluator.h"
#include "../../engine/core/belief_feature_extractor.h"
#include "../../engine/core/belief_label_extractor.h"
#include "../../engine/core/belief_tracker.h"
#include "arknova_state.h"

namespace board_ai::arknova {

constexpr int kBeliefClasses = kCardSlots;

// Hands the belief net is asked for at a search root. A world picks one of them at random, so
// more samples than the search has worlds (mcts.json world_cap) would not be used.
constexpr int kBeliefSamples = 128;

// Per CardIdx, 1 when the observer knows where the card is: the public zones (discard pile,
// display, projects, zoos, Hunter reveals) and the hidden-field slots visible to the observer.
// `by_viz`: visibility from the viz tensors (truth or world states); false for a state masked
// for the observer, where hidden slots hold kHiddenCard. Shared by the tracker, the belief
// features and the token encoder's unseen-card summary.
std::vector<std::uint8_t> known_cards(const ArknovaState& st, int observer, bool by_viz);

// The opponent's hand as the observer can place it, from the viz tensors (truth or world states).
struct HiddenHand {
  std::vector<int> regular;  // unseen regular cards, not named, not maybe-tucked: the hand's pool
  std::vector<int> endgame;  // unseen endgame cards
  std::vector<int> named;    // hand cards the mover named in the public prefix
  std::vector<int> maybe;    // maybe-tucked cards, not named: in the hand or tucked
  int hidden = 0;            // hand cards neither visible nor named (the maybe ones among them)
};
HiddenHand hidden_hand(const ArknovaState& st, int observer);

class ArknovaBeliefTracker final : public IBeliefTracker {
 public:
  // One world's opponent hand from the belief net: the maybe-tucked cards placed in the hand, and
  // the cards drawn from the regular pool for the rest (ascending).
  struct HandSample {
    std::vector<int> maybe_in_hand;
    std::vector<int> drawn;
  };

  void init(IGameState& state, int perspective, const AnyMap& payload) override;
  AnyMap pack_init_payload(const IGameState& gt_state, int perspective) const override;
  void observe_public_event(int, ActionId, const std::vector<PublicEvent>&) override {}
  void randomize_unseen(IGameState& state, int observer, std::mt19937_64& rng) const override;
  void prepare_for_root(const IGameState& root_state, int root_player, const IBeliefFeatureExtractor* extractor,
                        const IBeliefEvaluator* evaluator) override;
  std::unique_ptr<IBeliefTracker> clone() const override { return std::make_unique<ArknovaBeliefTracker>(*this); }

 private:
  void sample_world(ArknovaState& st, int observer, const HandSample* sample, std::mt19937_64& rng) const;

  // The belief net's hands, the observer they were drawn for and that observer's hash of the
  // root position: only worlds of that very position use them. Shared: the search clones the
  // tracker for every world.
  std::shared_ptr<const std::vector<HandSample>> samples_;
  int samples_observer_ = -1;
  StateHash64 samples_state_ = 0;
};

// Input: unseen regular pool, the opponent's known hand cards, its maybe-tucked cards (in the hand
// or tucked), its zoo, the discard pile (kBeliefClasses each), then opponent / table scalars. The
// net's input adds the prefix of belief_sequence.h; its output is the next card of the hand.
class ArknovaBeliefFeatureExtractor final : public IBeliefFeatureExtractor {
 public:
  static constexpr int kScalars = 18;
  static constexpr int kFeatureDim = 5 * kBeliefClasses + kScalars;
  int feature_dim() const override { return kFeatureDim; }
  int output_logit_count() const override { return kBeliefClasses; }
  BeliefSequenceLayout sequence_layout() const override { return {kBeliefClasses, kPlayers - 1}; }
  void extract(const IGameState& masked_state, int perspective_player, const IBeliefTracker* tracker,
               std::vector<float>* out) const override;
};

// Labels: hand_counts[0][c] = 1 for the opponent's hand cards in HiddenHand::regular (hidden from
// the observer, not named, not maybe-tucked); remaining[c] = 1 for HiddenHand::regular.
class ArknovaBeliefLabelExtractor final : public IBeliefLabelExtractor {
 public:
  int label_class_count() const override { return kBeliefClasses; }
  void extract(const IGameState& truth_state, int observer, std::vector<std::vector<int>>* out_hand_counts,
               std::vector<int>* out_remaining, std::vector<int>* out_alive) const override;
};

}  // namespace board_ai::arknova
