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
#include "coup_state.h"

namespace board_ai::coup {

template <int NPlayers>
class CoupFeatureEncoder final : public IFeatureEncoder {
 public:
  using Cfg = CoupConfig<NPlayers>;
  int action_space() const override { return kActionSpace; }
  int feature_dim() const override { return Cfg::kFeatureDim; }

  void encode_features(
      const IGameState& masked_state,
      int perspective_player,
      const IBeliefTracker* tracker,
      std::vector<float>* out) const override;
};

// Heuristic belief tracker for Coup. Maintains a per-opponent role-signal
// count derived from public claim/challenge/reveal/exchange history and uses
// it to bias `randomize_unseen` away from uniform sampling. Motivation:
// under uniform sampling, MCTS converges to a degenerate "never challenge,
// never bluff" equilibrium on bluff-heavy games like Coup.
template <int NPlayers>
class CoupBeliefTracker final : public IBeliefTracker {
 public:
  using Cfg = CoupConfig<NPlayers>;
  CoupBeliefTracker() { last_revealed_role_.fill(-1); }
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
    return std::make_unique<CoupBeliefTracker<NPlayers>>(*this);
  }
  AnyMap serialize() const override;

  // Cached per-opp posterior from prepare_for_root.
  // pi_hand_[i] is a length-20 vector: indices [0..14] are the two-card
  // multiset categorical (alive=2 branch), indices [15..19] are the
  // single-card categorical (alive=1 branch). Each branch is independently
  // softmax'd along its own axis (sum within branch == 1).
  // pi_valid_ is false when no belief net has been installed for this
  // decision; randomize_unseen falls back to per-slot hand-craft sampling.
  bool pi_valid() const { return pi_valid_; }
  float pi_hand(int opp_idx, int idx) const {
    return pi_hand_[opp_idx][idx];
  }
  int pi_observer() const { return pi_observer_; }

  // Two-card multiset enumeration (Plan §5.1). Index 0..14 is the
  // canonical (R_a, R_b) order with R_a ≤ R_b in role-id order
  // (D=0, As=1, Cap=2, Amb=3, Con=4). need_2card_[m][R] = number of R
  // copies in multiset m (0/1/2). multiset_id_2card_(a,b) returns m for
  // any (a,b) regardless of order. Both are static — populated once from
  // the canonical enumeration.
  static constexpr int kTwoCardMultisets = 15;
  static constexpr int kBeliefBranch2 = 0;
  static constexpr int kBeliefBranch2Size = 15;
  static constexpr int kBeliefBranch1 = 15;
  static constexpr int kBeliefBranch1Size = kCharacterCount;  // 5
  static constexpr int kBeliefHandDim =
      kBeliefBranch2Size + kBeliefBranch1Size;  // 20
  static const std::array<std::array<int, kCharacterCount>,
                          kTwoCardMultisets>& need_2card();
  // Lookup table: id_2card[a][b] for any 0 ≤ a, b < kCharacterCount.
  // Symmetric: id_2card[a][b] == id_2card[b][a].
  static int multiset_id_2card(int a, int b);

  // For tests / encoder: expose pending-claim state.
  int pending_claimer() const { return pending_claimer_; }
  int pending_claim_role() const { return pending_claim_role_; }
  bool pending_challenged() const { return pending_challenged_; }

  // v0.2 raw-event accumulators. Read by CoupBeliefFeatureExtractor (which
  // is the only consumer that needs the pre/post layering).
  enum class ReshuffleKind : std::int8_t {
    kNone = 0,
    kExchange = 1,
    kRevealTruthful = 2,
  };
  int pre_claim_count(int player, int role) const {
    return pre_claim_counts_[player][role];
  }
  int post_claim_count(int player, int role) const {
    return post_claim_counts_[player][role];
  }
  int pre_challenge_initiated(int player, int role) const {
    return pre_challenge_initiated_[player][role];
  }
  int post_challenge_initiated(int player, int role) const {
    return post_challenge_initiated_[player][role];
  }
  ReshuffleKind last_reshuffle_kind(int player) const {
    return last_reshuffle_kind_[player];
  }
  int last_revealed_role(int player) const {
    return last_revealed_role_[player];
  }
  // Number of successful Exchange (Ambassador) actions this player has
  // completed — i.e. exchange_complete events emitted with player==p.
  // Read by CoupFeatureEncoder as a per-player feature.
  int exchange_complete_count(int player) const {
    if (player < 0 || player >= NPlayers) return 0;
    return exchange_complete_counts_[player];
  }

 private:
  // Reshuffle promotion: pre += post; post = 0. Called from
  // observe_public_event on RevealTruthful / Exchange events.
  void promote_post_to_pre(int player) {
    if (player < 0 || player >= NPlayers) return;
    for (int r = 0; r < kCharacterCount; ++r) {
      pre_claim_counts_[player][r] += post_claim_counts_[player][r];
      post_claim_counts_[player][r] = 0;
      pre_challenge_initiated_[player][r] +=
          post_challenge_initiated_[player][r];
      post_challenge_initiated_[player][r] = 0;
    }
  }

  int perspective_player_ = -1;

  // v0.2 raw-event accumulators (per opponent × role). On any reshuffle
  // event (Exchange complete / claim_resolved_truthful), we promote
  // post → pre and zero post for that opp; the pair therefore encodes
  // "evidence carried over a reshuffle" vs "evidence accrued since".
  std::array<std::array<int, kCharacterCount>, NPlayers> pre_claim_counts_{};
  std::array<std::array<int, kCharacterCount>, NPlayers> post_claim_counts_{};
  std::array<std::array<int, kCharacterCount>, NPlayers>
      pre_challenge_initiated_{};
  std::array<std::array<int, kCharacterCount>, NPlayers>
      post_challenge_initiated_{};
  std::array<ReshuffleKind, NPlayers> last_reshuffle_kind_{};
  // -1 when last_reshuffle_kind != kRevealTruthful, else 0..kCharacterCount-1.
  // Default-constructed to all -1 in the ctor body so a freshly-bundled
  // (un-init()'d) tracker already reports "no reveal yet" rather than
  // role=Duke. (Cannot use lambda NSDMI here — `kCharacterCount` is not
  // visible inside the inline initializer of a template member.)
  std::array<std::int8_t, NPlayers> last_revealed_role_{};
  // Successful Exchange completions per player. Bumped on every
  // `exchange_complete` event; reset in init().
  std::array<int, NPlayers> exchange_complete_counts_{};

  // Cached belief-net posterior — multiset categorical (Plan §5).
  // pi_hand_[i] is length 20: [0..14] is a softmax over two-card
  // multisets (alive=2 branch), [15..19] is a softmax over single roles
  // (alive=1 branch). Each branch is independently normalized; the two
  // branches are NOT a single 20-way distribution. randomize_unseen
  // selects the branch per opponent based on that opp's current alive
  // count, then categorically samples within the selected branch under
  // the deck feasibility mask. Filled by prepare_for_root, read in sim.
  // clone() copies these by value, so each sim's sim_tracker carries
  // the cached posterior — extractor + evaluator run only once per
  // decision at the root (Plan §2.1).
  std::array<std::array<float, kBeliefHandDim>, NPlayers - 1> pi_hand_{};
  bool pi_valid_ = false;
  int pi_observer_ = -1;

  // Multi-step claim resolution requires tracking pending state across calls:
  //   stage t  : claimer plays Tax         -> pending_claimer=P, claim_role=Duke
  //   stage t+1: opponent plays Challenge  -> pending_challenger=Q
  //   stage t+2: claimer plays RevealSlotN -> extractor emits card_revealed,
  //              compare revealed role vs pending_claim_role, resolve signals.
  //   stage t+1': opponent plays Allow     -> no challenge, claim_unchallenged
  int pending_claimer_ = -1;
  int pending_claim_role_ = -1;
  bool pending_challenged_ = false;
};

// Belief-net feature extractor (Plan §1.3).
//
// Output layout per call (length kFeatureDim):
//
//   global block (6):
//     remaining[R] for R in {Duke,Assassin,Captain,Ambassador,Contessa}  (5)
//     ply_count / 200                                                    (1)
//
//   per-opp block (28 × (N-1)), in opp_to_player(observer, i, N) order:
//     pre_claim_counts[R]            / 4  clamp [0,1]                    (5)
//     post_claim_counts[R]           / 4  clamp [0,1]                    (5)
//     pre_challenge_initiated[R]     / 4  clamp [0,1]                    (5)
//     post_challenge_initiated[R]    / 4  clamp [0,1]                    (5)
//     last_reshuffle_kind one-hot ∈ {None, Exchange, RevealTruthful}     (3)
//     last_revealed_role one-hot (zero unless kind==RevealTruthful)      (5)
//
// `remaining[R]` is observer-derived: total 3 cards per role, minus any
// `revealed=true` slot of that role (publicly revealed across all
// perspectives), minus the observer's own unrevealed influence slots of
// that role (viz=1 to observer). Other players' unrevealed slots are
// viz=0 → kPlaceholder; do not contribute.
template <int NPlayers>
class CoupBeliefFeatureExtractor final : public IBeliefFeatureExtractor {
 public:
  using Cfg = CoupConfig<NPlayers>;
  static constexpr int kPerOppDim = 28;
  static constexpr int kGlobalDim = 6;
  static constexpr int kFeatureDim = kGlobalDim + kPerOppDim * (NPlayers - 1);
  // Output: per opp [15-way two-card multiset | 5-way single-card] = 20.
  // Two independent softmax branches selected at sample time by alive count.
  static constexpr int kLogitCount =
      (NPlayers - 1) * CoupBeliefTracker<NPlayers>::kBeliefHandDim;

  int feature_dim() const override { return kFeatureDim; }
  int output_logit_count() const override { return kLogitCount; }

  void extract(
      const IGameState& masked_state,
      int perspective_player,
      const IBeliefTracker* tracker,
      std::vector<float>* out) const override;
};

extern template class CoupFeatureEncoder<2>;
extern template class CoupFeatureEncoder<3>;
extern template class CoupFeatureEncoder<4>;
extern template class CoupBeliefTracker<2>;
extern template class CoupBeliefTracker<3>;
extern template class CoupBeliefTracker<4>;
extern template class CoupBeliefFeatureExtractor<2>;
extern template class CoupBeliefFeatureExtractor<3>;
extern template class CoupBeliefFeatureExtractor<4>;

// Belief-net label extractor (multiset variant, Plan §7). GT-side
// counterpart to CoupBeliefFeatureExtractor — reads truth CoupState and
// emits the per-observer (hand_counts, remaining, alive_per_opp) tuple
// consumed by the Python training loop. Never reachable from the AI
// session.
//
// Shape per call (all fields plumb to BeliefSample):
//   hand_counts[i][R] = count of unrevealed R-cards in opp i's hand,
//                       opp i = (observer + 1 + i) mod NPlayers.
//                       Sums to alive_per_opp[i] (==1 or ==2 for
//                       multiset training).
//   remaining[R]      = kCardsPerCharacter(3) − own unrevealed R
//                       − publicly revealed R across all seats.
//   alive_per_opp[i]  = exact count of unrevealed slots opp i still
//                       has (0/1/2). The Python trainer uses this to
//                       pick the multiset branch (alive==2 → 15-way
//                       two-card head, alive==1 → 5-way single-card
//                       head) and to skip dead rows in the loss; the
//                       multiset id itself is derived from hand_counts
//                       inside the trainer.
template <int NPlayers>
class CoupBeliefLabelExtractor final : public IBeliefLabelExtractor {
 public:
  int label_class_count() const override { return kCharacterCount; }

  void extract(
      const IGameState& truth_state,
      int observer,
      std::vector<std::vector<int>>* out_hand_counts,
      std::vector<int>* out_remaining,
      std::vector<int>* out_alive) const override;
};

extern template class CoupBeliefLabelExtractor<2>;
extern template class CoupBeliefLabelExtractor<3>;
extern template class CoupBeliefLabelExtractor<4>;

}  // namespace board_ai::coup
