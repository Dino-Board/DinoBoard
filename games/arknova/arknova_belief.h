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
// Belief net (sigmoid per card, as in 6nimmit): P(the opponent holds card c in a hidden hand
// slot). With a posterior installed by prepare_for_root, hidden hand slots are filled by
// weighted sampling without replacement; everything else is uniform.

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

// Per CardIdx, 1 when the observer knows where the card is: the public zones (discard pile,
// display, projects, zoos, Hunter reveals) and the hidden-field slots visible to the observer.
// `by_viz`: visibility from the viz tensors (truth or world states); false for a state masked
// for the observer, where hidden slots hold kHiddenCard. Shared by the tracker, the belief
// features and the token encoder's unseen-card summary.
std::vector<std::uint8_t> known_cards(const ArknovaState& st, int observer, bool by_viz);

class ArknovaBeliefTracker final : public IBeliefTracker {
 public:
  void init(IGameState& state, int perspective, const AnyMap& payload) override;
  AnyMap pack_init_payload(const IGameState& gt_state, int perspective) const override;
  void observe_public_event(int, ActionId, const std::vector<PublicEvent>&) override {}
  void randomize_unseen(IGameState& state, int observer, std::mt19937_64& rng) const override;
  void prepare_for_root(const IGameState& root_state, int root_player, const IBeliefFeatureExtractor* extractor,
                        const IBeliefEvaluator* evaluator) override;
  std::unique_ptr<IBeliefTracker> clone() const override { return std::make_unique<ArknovaBeliefTracker>(*this); }

 private:
  void sample_world(ArknovaState& st, int observer, bool weighted, std::mt19937_64& rng) const;

  // The belief net's posterior, the observer it was computed for and that observer's hash of the
  // root position: it is only used to sample worlds of that very position.
  std::array<float, kBeliefClasses> hand_probs_{};
  int probs_observer_ = -1;
  StateHash64 probs_state_ = 0;
};

// Input: unseen regular pool, the opponent's known hand cards, its maybe-tucked cards (in the hand
// or tucked), its zoo, the discard pile (kBeliefClasses each), then opponent / table scalars.
class ArknovaBeliefFeatureExtractor final : public IBeliefFeatureExtractor {
 public:
  static constexpr int kScalars = 18;
  static constexpr int kFeatureDim = 5 * kBeliefClasses + kScalars;
  int feature_dim() const override { return kFeatureDim; }
  int output_logit_count() const override { return (kPlayers - 1) * kBeliefClasses; }
  void extract(const IGameState& masked_state, int perspective_player, const IBeliefTracker* tracker,
               std::vector<float>* out) const override;
};

// Label: multi-hot of the cards in the opponent's hand slots hidden from the observer.
class ArknovaBeliefLabelExtractor final : public IBeliefLabelExtractor {
 public:
  int label_class_count() const override { return kBeliefClasses; }
  void extract(const IGameState& truth_state, int observer, std::vector<std::vector<int>>* out_hand_counts,
               std::vector<int>* out_remaining, std::vector<int>* out_alive) const override;
};

}  // namespace board_ai::arknova
