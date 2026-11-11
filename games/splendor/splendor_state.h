#pragma once

#include <any>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <string>

#include "../../engine/core/game_interfaces.h"
#include "../../engine/core/visibility_schema.h"

namespace board_ai::splendor {

constexpr int kColorCount = 5;
constexpr int kTokenTypes = 6;
constexpr int kTargetPoints = 15;
constexpr int kMaxPlies = 160;

// Placeholder for masked card-id slots (`reserved`, `decks`). Card-id
// fields are int16, so the framework's kPlaceholderInt32 would truncate
// to 0 — a real card id.
constexpr std::int16_t kHiddenCardId = std::numeric_limits<std::int16_t>::min();

template <int NPlayers>
struct SplendorConfig {
  static_assert(NPlayers >= 2 && NPlayers <= 4);
  static constexpr int kPlayers = NPlayers;
  static constexpr int kGemCount = NPlayers == 2 ? 4 : (NPlayers == 3 ? 5 : 7);
  static constexpr int kGoldCount = 5;
  static constexpr int kNobleCount = NPlayers + 1;

  static constexpr int kBuyFaceupOffset = 0;
  static constexpr int kBuyFaceupCount = 12;
  static constexpr int kReserveFaceupOffset = kBuyFaceupOffset + kBuyFaceupCount;
  static constexpr int kReserveFaceupCount = 12;
  static constexpr int kReserveDeckOffset = kReserveFaceupOffset + kReserveFaceupCount;
  static constexpr int kReserveDeckCount = 3;
  static constexpr int kBuyReservedOffset = kReserveDeckOffset + kReserveDeckCount;
  static constexpr int kBuyReservedCount = 3;
  static constexpr int kTakeThreeOffset = kBuyReservedOffset + kBuyReservedCount;
  static constexpr int kTakeThreeCount = 10;
  static constexpr int kTakeTwoDifferentOffset = kTakeThreeOffset + kTakeThreeCount;
  static constexpr int kTakeTwoDifferentCount = 10;
  static constexpr int kTakeOneOffset = kTakeTwoDifferentOffset + kTakeTwoDifferentCount;
  static constexpr int kTakeOneCount = 5;
  static constexpr int kTakeTwoSameOffset = kTakeOneOffset + kTakeOneCount;
  static constexpr int kTakeTwoSameCount = 5;
  static constexpr int kChooseNobleOffset = kTakeTwoSameOffset + kTakeTwoSameCount;
  static constexpr int kChooseNobleCount = kNobleCount;
  static constexpr int kReturnTokenOffset = kChooseNobleOffset + kChooseNobleCount;
  static constexpr int kReturnTokenCount = 6;
  static constexpr int kPassAction = kReturnTokenOffset + kReturnTokenCount;
  static constexpr int kActionSpace = kPassAction + 1;

  static constexpr int kPerPlayerFeatures = 15;  // gems(6)+bonuses(5)+points(1)+reserved(1)+cards(1)+nobles(1)
  static constexpr int kPerPlayerReserves = 3 * 13;  // 3 slots * 13 features per card
  static constexpr int kFeatureDim = 6 + kPerPlayerFeatures * kPlayers +
      kNobleCount * 6 + 12 * 13 + kPerPlayerReserves * kPlayers + 7;
};

enum class SplendorTurnStage : std::int8_t {
  kNormal = 0,
  kReturnTokens = 1,
  kChooseNoble = 2,
};

struct SplendorCard {
  std::int8_t tier = 1;
  std::int8_t bonus = 0;
  std::int8_t points = 0;
  std::array<std::int8_t, kColorCount> cost{};
};

template <int NPlayers>
struct SplendorData {
  using Cfg = SplendorConfig<NPlayers>;
  int current_player = 0;
  std::int8_t first_player = 0;
  int plies = 0;
  int final_round_remaining = -1;
  std::int8_t stage = static_cast<std::int8_t>(SplendorTurnStage::kNormal);
  int pending_returns = 0;
  std::array<std::int8_t, Cfg::kNobleCount> pending_noble_slots{};
  std::int8_t pending_nobles_size = 0;
  std::int16_t forced_draw_override = -1;
  int winner = -1;
  bool terminal = false;
  bool shared_victory = false;
  std::array<int, Cfg::kPlayers> scores{};

  std::array<std::int8_t, kTokenTypes> bank{};
  std::array<std::array<std::int8_t, kTokenTypes>, Cfg::kPlayers> player_gems{};
  std::array<std::array<std::int8_t, kColorCount>, Cfg::kPlayers> player_bonuses{};
  std::array<std::int16_t, Cfg::kPlayers> player_points{};
  std::array<std::int16_t, Cfg::kPlayers> player_cards_count{};
  std::array<std::int16_t, Cfg::kPlayers> player_nobles_count{};

  std::array<std::array<std::int16_t, 3>, Cfg::kPlayers> reserved{};
  std::array<std::array<std::int8_t, 3>, Cfg::kPlayers> reserved_visible{};
  // Per-reserve-slot tier label (0/1/2 for tiers 1/2/3, -1 for empty).
  // Public to all observers — even when the card id itself is hidden,
  // every player sees which deck a blind reserve was drawn from. This
  // is what randomize_unseen needs to consume the right tier from the
  // unseen pool; reading tier off a stale `reserved` cid in a sim
  // produces inconsistent deck sizes across sims and lets the same DAG
  // node disagree on whether reserve-from-deck-tier-X is legal.
  std::array<std::array<std::int8_t, 3>, Cfg::kPlayers> reserved_tier{};
  std::array<std::int8_t, Cfg::kPlayers> reserved_size{};

  std::array<std::vector<std::int16_t>, 3> decks{};
  // First-class public per-tier deck size. Maintained by rules at every
  // draw / refill / reserve site so observers never depend on the
  // size of `decks[t]` (whose contents are hidden) to reconstruct this
  // hash value. read/write_field_slot for `deck_sizes` address
  // d.deck_sizes[t] directly.
  std::array<std::int16_t, 3> deck_sizes{};
  std::array<std::array<std::int16_t, 4>, 3> tableau{};
  std::array<std::int8_t, 3> tableau_size{};

  std::array<std::int16_t, Cfg::kNobleCount> nobles{};
  std::int8_t nobles_size = 0;
};

template <int NPlayers>
struct SplendorPersistentNode {
  std::shared_ptr<const SplendorPersistentNode<NPlayers>> parent{};
  ActionId action_from_parent = -1;
  mutable std::shared_ptr<const SplendorData<NPlayers>> materialized{};
};

template <int NPlayers>
class SplendorPersistentState {
 public:
  SplendorPersistentState() = default;
  explicit SplendorPersistentState(std::shared_ptr<const SplendorPersistentNode<NPlayers>> node);

  static SplendorPersistentState root_from_state(std::mt19937_64& rng);

  bool valid() const { return static_cast<bool>(node_); }
  const SplendorData<NPlayers>& data() const;
  SplendorPersistentState advance(ActionId action, std::mt19937_64& rng) const;
  StateHash64 state_hash() const;

 private:
  std::shared_ptr<const SplendorPersistentNode<NPlayers>> node_{};
};

// Undo frame bundling persistent state with a snapshot of viz_ taken
// before do_action_fast applies any reveal_slot / reset_to_base. Rules
// mutate state.viz_, so undo_action must restore both pieces atomically.
template <int NPlayers>
struct SplendorUndoFrame {
  SplendorPersistentState<NPlayers> persistent{};
  std::vector<viz::VizTensor> viz_snapshot{};
};

template <int NPlayers>
struct SplendorState final : public CloneableState<SplendorState<NPlayers>> {
  using Cfg = SplendorConfig<NPlayers>;
  SplendorPersistentState<NPlayers> persistent{};
  std::vector<SplendorUndoFrame<NPlayers>> undo_stack{};

  SplendorState();

  // Visibility schema. Splendor partitions:
  //   - all_public: bank, tableau, tableau_size, nobles, scores, bonuses,
  //     points, card/noble counts, reserved_size, reserved_visible (the
  //     "is this reserve face-up?" public flag), stage / pending_*, etc.
  //   - owner_only_first_axis on reserved[N][3]: base = blind. Rules'
  //     do_action_fast calls viz::reveal_slot when a reserve becomes
  //     face-up (Reserve-from-tableau path), reset_to_base on the slot
  //     when it's bought/discarded.
  //
  // Decks (variable-length per-tier vectors): contents hidden, size public.
  // The size is exposed as schema field `deck_sizes` (all_public, shape [3]),
  // which read/write_field_slot project to / from `decks[t].resize(...)`.
  // Contents are not declared as a schema slot — `randomize_unseen` resamples
  // them from belief.
  static const viz::VisibilitySchema& schema();

  void reset_with_seed(std::uint64_t seed) override;

  StateHash64 state_hash() const override;
  void hash_field_slot(Hasher& h, viz::FieldId fid,
                       const std::vector<int>& idx) const override;
  void hash_field_bulk(Hasher& h, viz::FieldId fid,
                       const viz::VizTensor& v,
                       int perspective) const override;
  // Walker-driven snapshot wire I/O — typed read/write of one schema
  // slot. Mirrors hash_field_slot dispatch shape.
  std::any read_field_slot(viz::FieldId fid,
                           const std::vector<int>& idx) const override;
  void write_field_slot(viz::FieldId fid,
                        const std::vector<int>& idx,
                        const std::any& value) override;
  // mask_field_slot writes kPlaceholder into the matching slot. Splendor
  // only has one hidden field — `reserved` (face-down id) — so the
  // dispatcher is one branch.
  void mask_field_slot(viz::FieldId fid,
                       const std::vector<int>& idx) override;
  // COW persistent-aware override. Detaches a writable copy of
  // SplendorData before running the default walker body; otherwise
  // mask_field_slot would write through `persistent.data()` const ref
  // and undefined-behave.
  void mask_all_hidden_slots(
      const viz::VisibilitySchema& schema, int perspective) override;
  const viz::VisibilitySchema& schema_ref() const override { return schema(); }
  int current_player() const override;
  int first_player() const override;
  bool is_terminal() const override;
  bool is_turn_start() const override;
  int num_players() const override { return Cfg::kPlayers; }
  int winner() const override;
  // winner() stays -1 whenever there is no single winner. On a shared
  // victory (still tied after the fewest-cards tie-break) winners() is every
  // player tied for first and losers() everyone else, so with 3-4 players
  // the ones outside the tie still lose (zero-sum: +1/nw each winner,
  // -1/nl each loser). A tie between ALL players — and every 2-player tie —
  // stays a plain draw: both lists empty, all-zero.
  std::vector<int> winners() const override;
  std::vector<int> losers() const override;
};

const std::vector<SplendorCard>& splendor_card_pool();
const std::array<std::array<std::int8_t, kColorCount>, 12>& splendor_nobles();

extern template struct SplendorData<2>;
extern template struct SplendorData<3>;
extern template struct SplendorData<4>;
extern template class SplendorPersistentState<2>;
extern template class SplendorPersistentState<3>;
extern template class SplendorPersistentState<4>;
extern template struct SplendorState<2>;
extern template struct SplendorState<3>;
extern template struct SplendorState<4>;

}  // namespace board_ai::splendor
