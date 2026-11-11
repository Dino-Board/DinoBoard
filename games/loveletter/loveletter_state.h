#pragma once

#include <any>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../../engine/core/game_interfaces.h"
#include "../../engine/core/visibility_schema.h"

namespace board_ai::loveletter {

constexpr std::int8_t kGuard = 1;
constexpr std::int8_t kPriest = 2;
constexpr std::int8_t kBaron = 3;
constexpr std::int8_t kHandmaid = 4;
constexpr std::int8_t kPrince = 5;
constexpr std::int8_t kKing = 6;
constexpr std::int8_t kCountess = 7;
constexpr std::int8_t kPrincess = 8;

constexpr int kCardTypes = 8;
constexpr int kTotalCards = 16;
constexpr std::array<int, 9> kCardCounts = {0, 5, 2, 2, 2, 2, 1, 1, 1};

constexpr int kGuardOffset = 0;
constexpr int kGuardCount = 28;       // target(0..3) * 7 + (guess-2)
constexpr int kPriestOffset = 28;
constexpr int kPriestCount = 4;
constexpr int kBaronOffset = 32;
constexpr int kBaronCount = 4;
constexpr int kHandmaidAction = 36;
constexpr int kPrinceOffset = 37;
constexpr int kPrinceCount = 4;       // includes self-target
constexpr int kKingOffset = 41;
constexpr int kKingCount = 4;
constexpr int kCountessAction = 45;
constexpr int kPrincessAction = 46;
constexpr int kActionSpace = 47;

template <int NPlayers>
struct LoveLetterConfig {
  static_assert(NPlayers >= 2 && NPlayers <= 4);
  static constexpr int kPlayers = NPlayers;
  // Per-player features: alive, protected, current_player, hand_exposed
  // (4; hand_exposed is reduced from viz["hand"][p,*] at encode time —
  // not a stored field) + discard counts (8) + discard size (1) +
  // hand one-hot (8) + drawn_card one-hot (8) = 29.
  static constexpr int kPerPlayerFeatures = 29;
  static constexpr int kGlobalFeatures = 12;
  static constexpr int kFeatureDim = kPerPlayerFeatures * NPlayers + kGlobalFeatures;
  static constexpr int kFaceUpRemoved = NPlayers == 2 ? 3 : 0;
};

template <int NPlayers>
struct LoveLetterData {
  using Cfg = LoveLetterConfig<NPlayers>;

  int current_player = 0;
  std::int8_t first_player = 0;
  int winner = -1;
  bool terminal = false;
  int ply = 0;

  std::array<std::int8_t, Cfg::kPlayers> hand{};
  std::int8_t drawn_card = 0;

  std::array<std::int8_t, Cfg::kPlayers> alive{};
  std::array<std::int8_t, Cfg::kPlayers> protected_flags{};

  // Multiset bookkeeping. Card values are 1..kCardTypes; index 0 unused.
  // Engine never needs ordering — discards are summed for tie-break,
  // deck is "draw-from-multiset" (handled by rules with rng), face-up
  // is initial-draw-only and read by encoder/heuristic as a count.
  std::array<std::int8_t, kCardTypes + 1> deck_count{};
  // First-class public total. Maintained by rules at every draw/refill
  // site so observers never depend on summing the (hidden) deck_count
  // multiset to reconstruct the deck size. Treated as an independent
  // schema slot (declared all_public); hash/read/write_field_slot all
  // address d.deck_size directly.
  std::int8_t deck_size = 0;
  std::int8_t set_aside_card = 0;
  std::array<std::array<std::int8_t, kCardTypes + 1>, Cfg::kPlayers> discard_count{};
  std::array<std::int8_t, kCardTypes + 1> face_up_count{};
};

template <int NPlayers>
struct LoveLetterUndoFrame {
  LoveLetterData<NPlayers> data;
  std::vector<viz::VizTensor> viz_snapshot;
};

template <int NPlayers>
struct LoveLetterState final : public CloneableState<LoveLetterState<NPlayers>> {
  using Cfg = LoveLetterConfig<NPlayers>;
  LoveLetterData<NPlayers> data;
  std::vector<LoveLetterUndoFrame<NPlayers>> undo_stack;

  LoveLetterState();

  // Visibility schema. LoveLetter is the most subtle game we ship;
  // the schema captures the static visibility layout. Dynamic reveals
  // (Priest peek, Baron compare, end-of-round flips) are wired by
  // do_action_fast via reveal_slot_to / reveal_slot. Partition:
  //
  //   - all_public scalars: current_player, first_player, winner,
  //     terminal, ply.
  //   - all_public 1D: alive[N], protected_flags[N].
  //   - owner_only_first_axis: hand[N]. Each player sees only their
  //     own hand card. Reveals (Priest peek / Baron showdown / King swap)
  //     mutate viz["hand"] directly via reveal_slot_to / reveal_slot;
  //     "is hand[p] exposed to anyone other than p" is reduced from
  //     viz["hand"][p, *] at encode/snapshot time — no parallel cache.
  //   - all_hidden: drawn_card (rules will reveal_slot_to(current_player)
  //     on draw, reset_to_base on play; the field is briefly held only
  //     by the active player). set_aside_card is permanently hidden
  //     (never revealed to anyone — it's the card removed from the
  //     bottom of the deck at game start).
  //
  // Multiset count arrays (all schema-declared, fixed shape):
  //   - deck_count[kCardTypes+1]: per-type remaining count in the
  //     hidden deck. The size is public; per-type contents are NOT
  //     declared all_public — randomize_unseen fills from the
  //     tracker's information set.
  //   - discard_count[N, kCardTypes+1]: per-player public discard
  //     multiset. Order is purely visual; UI reconstructs from the
  //     action stream.
  //   - face_up_count[kCardTypes+1]: 2p-only, public, set once at
  //     game start.
  //
  // Higher-order belief reasoning (e.g. tracking "what does opp infer
  // from my last Guard guess?") is an MCTS-tracker concern, not a
  // schema concern: the schema only models direct first-order
  // visibility ("who sees what right now"). The tracker can still
  // condition on the public action history independent of viz.
  static const viz::VisibilitySchema& schema();

  void reset_with_seed(std::uint64_t seed) override;
  // Re-seed `viz_` from the visibility schema base without touching
  // the data payload. Called by the registrar on the API/session
  // path after `apply_initial_observation` mutates `data` directly,
  // so any stale reveals carried over from `reset_with_seed` are
  // cleared before the per-perspective starting reveals are re-applied
  // through rules.
  void reseed_viz();

  StateHash64 state_hash() const override;
  void hash_field_slot(Hasher& h, viz::FieldId fid,
                       const std::vector<int>& idx) const override;
  void mask_field_slot(viz::FieldId fid,
                       const std::vector<int>& idx) override;
  std::any read_field_slot(viz::FieldId fid,
                           const std::vector<int>& idx) const override;
  void write_field_slot(viz::FieldId fid,
                        const std::vector<int>& idx,
                        const std::any& value) override;
  const viz::VisibilitySchema& schema_ref() const override { return schema(); }
  int current_player() const override;
  int first_player() const override;
  bool is_terminal() const override;
  int num_players() const override { return Cfg::kPlayers; }
  int winner() const override;
  // Showdown tie (same highest card AND same discard total): winner() is
  // -1, but only the tied players share the win — winners() returns all
  // of them and losers() everyone else, so the value model pays +1/nw
  // and -1/nl instead of 0 to a player who plainly lost. If every player
  // is tied the game is a draw (all values 0).
  std::vector<int> winners() const override;
  std::vector<int> losers() const override;
};

extern template struct LoveLetterData<2>;
extern template struct LoveLetterData<3>;
extern template struct LoveLetterData<4>;
extern template struct LoveLetterState<2>;
extern template struct LoveLetterState<3>;
extern template struct LoveLetterState<4>;

}  // namespace board_ai::loveletter
