#pragma once

#include <any>
#include <array>
#include <cstdint>
#include <vector>

#include "../../engine/core/game_interfaces.h"
#include "../../engine/core/visibility_schema.h"

namespace board_ai::coup {

using CharId = std::int8_t;
constexpr CharId kDuke = 0;
constexpr CharId kAssassin = 1;
constexpr CharId kCaptain = 2;
constexpr CharId kAmbassador = 3;
constexpr CharId kContessa = 4;
constexpr int kCharacterCount = 5;
constexpr int kCardsPerCharacter = 3;
constexpr int kTotalCards = 15;
constexpr int kStartingCoins = 2;
constexpr int kCoupCost = 7;
constexpr int kAssassinateCost = 3;
constexpr int kForceCoupThreshold = 10;
constexpr int kInfluencePerPlayer = 2;
constexpr int kExchangeDrawSlots = 2;

constexpr int kIncomeAction = 0;
constexpr int kForeignAidAction = 1;
constexpr int kCoupOffset = 2;
constexpr int kCoupCount = 4;
constexpr int kTaxAction = 6;
constexpr int kAssassinateOffset = 7;
constexpr int kAssassinateCount = 4;
constexpr int kStealOffset = 11;
constexpr int kStealCount = 4;
constexpr int kExchangeAction = 15;

constexpr int kChallengeAction = 16;
constexpr int kAllowAction = 17;

constexpr int kBlockDukeAction = 18;
constexpr int kBlockContessaAction = 19;
constexpr int kBlockAmbassadorAction = 20;
constexpr int kBlockCaptainAction = 21;
constexpr int kAllowNoBlockAction = 22;

constexpr int kRevealSlot0 = 23;
constexpr int kRevealSlot1 = 24;

constexpr int kLoseSlot0 = 25;
constexpr int kLoseSlot1 = 26;

constexpr int kReturnDuke = 27;
constexpr int kReturnAssassin = 28;
constexpr int kReturnCaptain = 29;
constexpr int kReturnAmbassador = 30;
constexpr int kReturnContessa = 31;

constexpr int kActionSpace = 32;

// What every seat observes in place of an exchange-return action
// (CoupRules::public_action): which character went back to the court deck
// is private to the exchanging player. Outside the action space, so it is
// never legal and never a policy target.
constexpr int kHiddenReturnAction = kActionSpace;

enum class CoupStage : std::int8_t {
  kDeclareAction = 0,
  kChallengeAction = 1,
  kResolveChallengeAction = 2,
  kChooseLoseInfluence = 3,
  kCounterAction = 4,
  kChallengeCounter = 5,
  kResolveChallengeCounter = 6,
  kChooseLoseInfluenceCounter = 7,
  kExchangeReturn1 = 8,
  kExchangeReturn2 = 9,
  kLoseInfluenceFromAction = 10,
};
constexpr int kStageCount = 11;

template <int NPlayers>
struct CoupConfig {
  static_assert(NPlayers >= 2 && NPlayers <= 4);
  static constexpr int kPlayers = NPlayers;
  // Belief-net plan §16. Per-player block (47/player):
  //   alive(1) + coins/12(1) + 2 x {revealed?(1)+char-OH(5)}(12) +
  //   role flags active/target/blocker/challenger(4) +
  //   pre_claim_counts/4(5) + post_claim_counts/4(5) +
  //   pre_challenge_initiated/4(5) + post_challenge_initiated/4(5) +
  //   last_reshuffle_kind OH None/Exchange/RevealTruthful(3) +
  //   last_revealed_role OH(5) +
  //   exchange_complete_count/4 (clamp [0,1])(1) = 47.
  static constexpr int kPerPlayerFeatures = 47;
  // Global block (47 + N):
  //   stage OH(11) + declared_action_type OH(7) + claimed_character OH(5) +
  //   block_character OH(5) + pending_claimer relative OH(N) +
  //   pending_challenged(1) + ply/200(1) + remaining[R](5) +
  //   2 x {occupied?(1)+OH(5)} exchange_drawn(12) = 47 + N.
  static constexpr int kGlobalFeatures = 47 + NPlayers;
  static constexpr int kFeatureDim = kPerPlayerFeatures * NPlayers + kGlobalFeatures;
};

template <int NPlayers>
struct CoupData {
  using Cfg = CoupConfig<NPlayers>;

  int current_player = 0;
  std::int8_t first_player = 0;
  int winner = -1;
  bool terminal = false;
  int ply = 0;
  CoupStage stage = CoupStage::kDeclareAction;

  std::array<std::array<CharId, kInfluencePerPlayer>, Cfg::kPlayers> influence{};
  std::array<std::array<bool, kInfluencePerPlayer>, Cfg::kPlayers> revealed{};
  std::array<int, Cfg::kPlayers> coins{};
  std::array<bool, Cfg::kPlayers> alive{};

  // Court deck as fixed-shape multiset count + size scalar. Multiset
  // shape lets the schema walker hash/serialize/randomize without
  // variable-length escape hatches; size is the public count exposed
  // to all viewers.
  std::array<std::int8_t, kCharacterCount> deck_count{};
  std::int8_t deck_size = 0;

  int active_player = 0;
  ActionId declared_action = -1;
  int action_target = -1;
  CharId claimed_character = -1;

  int challenger = -1;
  int challenge_loser = -1;
  bool action_challenged = false;
  bool action_challenge_succeeded = false;

  int blocker = -1;
  CharId block_character = -1;
  bool counter_challenged = false;
  bool counter_challenge_succeeded = false;

  int challenge_check_index = 0;

  // Exchange-drawn buffer. Sentinel -1 means "no card here". Rules
  // call viz::reveal_slot_to(active_player) when drawing and
  // viz::reset_to_base on return; the schema base is all_hidden.
  std::array<CharId, kExchangeDrawSlots> exchange_drawn{-1, -1};
  int exchange_held_count = 0;

  // What each seat knows FOR CERTAIN about the court deck, maintained by
  // rules. Private to the seat (owner-only viz): it mostly stems from
  // cards the seat itself returned during an Exchange.
  //   deck_known[p][c]       = copies of character c that seat p knows to
  //                            lie in the deck or in a slot flagged below.
  //   deck_known_taint[p][q][s] = 1 iff influence slot (q, s) received a
  //                            card from the deck, unseen by p, while p
  //                            held such knowledge — a known card may
  //                            have moved there.
  // Invariant (for the real state and every world sampled for p):
  //   deck_known[p] ⊆ court deck ⊎ contents of p's flagged slots.
  // `randomize_unseen` enforces it, so a world never deals an opponent a
  // card the observer knows is in the deck.
  std::array<std::array<std::int8_t, kCharacterCount>, Cfg::kPlayers>
      deck_known{};
  std::array<std::array<std::array<std::int8_t, kInfluencePerPlayer>,
                        Cfg::kPlayers>,
             Cfg::kPlayers>
      deck_known_taint{};
};

template <int NPlayers>
struct CoupState final : public CloneableState<CoupState<NPlayers>> {
  using Cfg = CoupConfig<NPlayers>;
  CoupData<NPlayers> data;

  CoupState();

  // Visibility schema. Coup's hidden info partitions into:
  //   - all_public scalars/arrays: stage/coins/alive/revealed/etc.,
  //     and the multiset COUNT view (deck_size). Public to every
  //     perspective at all times.
  //   - owner_only_first_axis: influence[N][2]. base viz[p, *, p]=1.
  //     When a card is revealed by a challenge or lose-influence,
  //     rules call viz::reveal_slot(state, "influence", {p, s}) so
  //     every perspective sees the truth (not gated on a side flag).
  //   - all_hidden: exchange_drawn[2], deck_count[5]. Active player
  //     gets reveal_slot_to during exchange; deck_count is filled by
  //     the tracker's randomize_unseen for sim entries.
  //   - owner_only_first_axis (never revealed): deck_known[N][5],
  //     deck_known_taint[N][N][2] — each seat's certain knowledge of
  //     the court deck (see CoupData).
  static const viz::VisibilitySchema& schema();

  void reset_with_seed(std::uint64_t seed) override;
  StateHash64 state_hash() const override;
  void hash_field_slot(Hasher& h, viz::FieldId fid,
                       const std::vector<int>& idx) const override;
  void hash_field_bulk(Hasher& h, viz::FieldId fid,
                       const viz::VizTensor& v,
                       int perspective) const override;
  std::any read_field_slot(viz::FieldId fid,
                           const std::vector<int>& idx) const override;
  void write_field_slot(viz::FieldId fid,
                        const std::vector<int>& idx,
                        const std::any& value) override;
  void mask_field_slot(viz::FieldId fid,
                       const std::vector<int>& idx) override;
  const viz::VisibilitySchema& schema_ref() const override { return schema(); }
  int current_player() const override;
  int first_player() const override;
  bool is_terminal() const override;
  int num_players() const override { return Cfg::kPlayers; }
  int winner() const override;
  bool is_turn_start() const override;
};

extern template struct CoupData<2>;
extern template struct CoupData<3>;
extern template struct CoupData<4>;
extern template struct CoupState<2>;
extern template struct CoupState<3>;
extern template struct CoupState<4>;

}  // namespace board_ai::coup
