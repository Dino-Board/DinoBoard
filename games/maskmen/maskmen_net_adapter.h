#pragma once

#include <array>
#include <memory>
#include <random>
#include <vector>

#include "../../engine/core/belief_evaluator.h"
#include "../../engine/core/belief_feature_extractor.h"
#include "../../engine/core/belief_label_extractor.h"
#include "../../engine/core/belief_tracker.h"
#include "../../engine/core/feature_encoder.h"
#include "../../engine/core/game_interfaces.h"
#include "maskmen_state.h"

namespace board_ai::maskmen {

struct MaskmenPlay {
  int actor;
  int type;
  int size;
};

// Maskmen feature dim is per-NPlayers because per-seat blocks scale with N.
// Layout (encode_features):
//   own hand counts (normalized by kStartingHand)        : 6
//   last_play_type one-hot                                : 6
//   last_play=none bit                                    : 1
//   last_play_size / kMaxPlaySize                         : 1
//   is_first_play_of_round                                : 1
//   strength_above closure flat                           : 36
//   per-seat (in perspective-rotated order):              : 9 * N
//     hand_total / kStartingHand                          : 1
//     passed_this_round                                   : 1
//     is_current_player                                   : 1
//     belief expected per-type counts (own = own hand)    : 6
//   current round chain (reverse order, 6 slots)         : 6 * (8+N)
//   history chains (reverse order, 8 rounds × 6 slots)   : 48 * (8+N)
//   current_round_idx                                     : 1
//
// Sum: 51 + 9N + 54*(8+N) + 1 → 2p=610, 3p=673, 4p=736.
template <int NPlayers>
struct MaskmenFeatureLayout {
  static constexpr int kPublicDim = 6 + 6 + 1 + 1 + 1 + 36;  // 51
  static constexpr int kPerSeatDim = 1 + 1 + 1 + 6;          // 9
  static constexpr int kChainPerPlay = 6 + 1 + NPlayers + 1;  // 8+N
  static constexpr int kMaxChainLen = 6;
  static constexpr int kMaxHistoryRounds = 8;
  static constexpr int kCurrentChainDim = kMaxChainLen * kChainPerPlay;
  static constexpr int kHistoryChainDim =
      kMaxHistoryRounds * kMaxChainLen * kChainPerPlay;
  static constexpr int kChainDim =
      kCurrentChainDim + kHistoryChainDim + 1;  // +1 for round_idx
  static constexpr int kFeatureDim =
      kPublicDim + kPerSeatDim * NPlayers + kChainDim;
};

template <int NPlayers>
class MaskmenFeatureEncoder final : public IFeatureEncoder {
 public:
  using Layout = MaskmenFeatureLayout<NPlayers>;
  int action_space() const override { return kActionSpace; }
  int feature_dim() const override { return Layout::kFeatureDim; }

  void encode_features(
      const IGameState& masked_state,
      int perspective_player,
      const IBeliefTracker* tracker,
      std::vector<float>* out) const override;
};

// Belief tracker. Maintains the public per-type count of cards already
// played (`played_per_type_`), so the unseen pool size per type is
// derivable as kCardsPerType - own[t] - played_per_type_[t]. Also caches
// belief-net expected per-type counts E[i][t] for each opp under the
// observer's perspective; randomize_unseen runs IPF to match (row sum =
// opp_total, column sum = pool_remaining), then weighted-urn samples a
// concrete multiset per opp.
template <int NPlayers>
class MaskmenBeliefTracker final : public IBeliefTracker {
 public:
  void init(IGameState& state, int perspective,
            const AnyMap& payload) override;
  void observe_public_event(
      int actor,
      ActionId action,
      const std::vector<PublicEvent>& events) override;
  void randomize_unseen(IGameState& state, int observer,
                        std::mt19937_64& rng) const override;
  void prepare_for_root(
      const IGameState& root_state,
      int root_player,
      const IBeliefFeatureExtractor* extractor,
      const IBeliefEvaluator* evaluator) override;
  std::unique_ptr<IBeliefTracker> clone() const override {
    return std::make_unique<MaskmenBeliefTracker<NPlayers>>(*this);
  }
  AnyMap serialize() const override;

  // Read-only accessors for encoder / tests.
  int played_per_type(int t) const {
    return played_per_type_[static_cast<size_t>(t)];
  }
  int hand_total_for(int seat) const {
    return hand_total_[static_cast<size_t>(seat)];
  }
  int passed_for(int seat) const {
    return passed_this_round_[static_cast<size_t>(seat)];
  }
  bool pi_valid() const { return pi_valid_; }
  int pi_observer() const { return pi_observer_; }
  // Belief expected per-type count for opp_idx (observer-rotated, 0..N-2)
  // and type t. observer maps opp_idx → seat by
  //   seat = (observer + 1 + opp_idx) % NPlayers.
  float pi_expected(int opp_idx, int t) const {
    return pi_expected_[static_cast<size_t>(opp_idx)]
                      [static_cast<size_t>(t)];
  }
  const std::vector<MaskmenPlay>& current_round_plays() const {
    return current_round_plays_;
  }
  const std::vector<std::vector<MaskmenPlay>>& history_plays() const {
    return history_plays_;
  }

 private:
  int perspective_ = -1;

  // Public: number of cards of each type that have been played overall.
  // Updated on every non-pass observe_public_event. Bounded by
  // kCardsPerType per slot.
  std::array<std::int8_t, kTypeCount> played_per_type_{};
  // Public: per-seat remaining hand total (kStartingHand minus everything
  // they have played). Updated on every non-pass observe_public_event.
  std::array<std::int8_t, NPlayers> hand_total_{};
  // Public: per-seat passed-this-round bit, kept in sync with state's
  // round flow. observe_public_event resets it on round-end transitions
  // (signaled by the round_end public event we emit from the rules).
  std::array<std::int8_t, NPlayers> passed_this_round_{};

  // Cached belief-net expected per-type count per opp (post-IPF).
  // pi_expected_[i][t]; row = opp index relative to pi_observer_.
  std::array<std::array<float, kTypeCount>, NPlayers - 1> pi_expected_{};
  bool pi_valid_ = false;
  int pi_observer_ = -1;

  std::vector<MaskmenPlay> current_round_plays_;
  std::vector<std::vector<MaskmenPlay>> history_plays_;
};

// Belief-net feature extractor.
//   Public-only inputs the belief net needs to predict opp hand
//   distributions:
//     own hand counts / kStartingHand                       : 6
//     last_play_type one-hot                                : 6
//     last_play=none bit                                    : 1
//     last_play_size / kMaxPlaySize                         : 1
//     is_first_play_of_round                                : 1
//     strength_above closure flat                           : 36
//     played_per_type / kCardsPerType                       : 6
//     per-opp (in observer-rotated order, i = 0..N-2):      : 3 * (N-1)
//       hand_total / kStartingHand                          : 1
//       passed_this_round                                   : 1
//       is_current_player                                   : 1
//
//   Output (logits): (N-1) opp × kTypeCount expected counts.
//   The belief network is trained as a regression head (Huber / MSE)
//   against ground-truth per-type hand counts; the framework still calls
//   it "logits" structurally but no softmax happens — extractor cache is
//   the raw scalar output.
template <int NPlayers>
class MaskmenBeliefFeatureExtractor final : public IBeliefFeatureExtractor {
 public:
  static constexpr int kGlobalDim = 6 + 6 + 1 + 1 + 1 + 36 + 6;  // 57
  static constexpr int kPerOppDim = 3;
  static constexpr int kBeliefChainPerPlay = 6 + 1 + NPlayers + 1;  // 8+N
  static constexpr int kBeliefMaxChainLen = 6;
  static constexpr int kBeliefMaxHistoryRounds = 8;
  static constexpr int kBeliefChainDim =
      kBeliefMaxChainLen * kBeliefChainPerPlay +
      kBeliefMaxHistoryRounds * kBeliefMaxChainLen * kBeliefChainPerPlay + 1;
  static constexpr int kFeatureDim =
      kGlobalDim + kPerOppDim * (NPlayers - 1) + kBeliefChainDim;
  // Per opp: kTypeCount expected counts. NOT logits — the head is a
  // regression head; the trainer reads the raw values as predicted means.
  static constexpr int kPerOppOutput = kTypeCount;
  static constexpr int kLogitCount = (NPlayers - 1) * kPerOppOutput;

  int feature_dim() const override { return kFeatureDim; }
  int output_logit_count() const override { return kLogitCount; }

  void extract(
      const IGameState& masked_state,
      int perspective_player,
      const IBeliefTracker* tracker,
      std::vector<float>* out) const override;
};

// Belief-net label extractor. Per-opp ground truth hand_count[t].
// hand_counts[i] is a length-kTypeCount integer vector (raw count, NOT
// one-hot — trainer uses MSE / Huber on the regression head).
// remaining[t] = kCardsPerType - publicly_played[t] - own_hand[t]
// alive_per_opp[i] = 1 if opp i still has cards else 0.
template <int NPlayers>
class MaskmenBeliefLabelExtractor final : public IBeliefLabelExtractor {
 public:
  int label_class_count() const override { return kTypeCount; }

  void extract(
      const IGameState& truth_state,
      int observer,
      std::vector<std::vector<int>>* out_hand_counts,
      std::vector<int>* out_remaining,
      std::vector<int>* out_alive) const override;
};

extern template class MaskmenFeatureEncoder<2>;
extern template class MaskmenFeatureEncoder<3>;
extern template class MaskmenFeatureEncoder<4>;
extern template class MaskmenBeliefTracker<2>;
extern template class MaskmenBeliefTracker<3>;
extern template class MaskmenBeliefTracker<4>;
extern template class MaskmenBeliefFeatureExtractor<2>;
extern template class MaskmenBeliefFeatureExtractor<3>;
extern template class MaskmenBeliefFeatureExtractor<4>;
extern template class MaskmenBeliefLabelExtractor<2>;
extern template class MaskmenBeliefLabelExtractor<3>;
extern template class MaskmenBeliefLabelExtractor<4>;

}  // namespace board_ai::maskmen
