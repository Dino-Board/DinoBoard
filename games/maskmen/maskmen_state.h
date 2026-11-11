#pragma once

#include <any>
#include <array>
#include <cstdint>
#include <vector>

#include "../../engine/core/game_interfaces.h"
#include "../../engine/core/visibility_schema.h"

namespace board_ai::maskmen {

// 6 wrestler types. Cards per type = 10 (kCardsPerType * kTypeCount = 60).
constexpr int kTypeCount = 6;
constexpr int kCardsPerType = 10;
constexpr int kTotalCards = kTypeCount * kCardsPerType;

// Each player starts with kStartingHand cards (deal 15 per player, the
// surplus cards are discarded for N<4 — see PLAN §1.1).
constexpr int kStartingHand = 15;
constexpr int kMaxPlaySize = 3;

// Action layout: 6 play-type actions + 1 pass = 7. The size of the play
// is *fully determined* by current round state:
//   - round opener  → size 1
//   - known-stronger (A2) → size = last_play_size (same)
//   - unknown relation (A1) → size = last_play_size + 1
// legal_actions filters per-type by whether the seat holds that many
// cards; the action only encodes which wrestler type to play.
constexpr int kPlayActionCount = kTypeCount;          // 6
constexpr int kPassAction = kPlayActionCount;         // 6
constexpr int kActionSpace = kPlayActionCount + 1;    // 7

inline int play_type_from_action(ActionId a) { return a; }
inline ActionId encode_play(int type) {
  return static_cast<ActionId>(type);
}
inline bool is_pass(ActionId a) { return a == kPassAction; }

template <int NPlayers>
struct MaskmenConfig {
  static_assert(NPlayers >= 2 && NPlayers <= 4);
  static constexpr int kPlayers = NPlayers;
  static constexpr int kActionSpace = board_ai::maskmen::kActionSpace;
};

template <int NPlayers>
struct MaskmenData {
  // Whose turn it is to act.
  int current_player_ = 0;

  // Termination.
  bool terminal = false;
  int winner = -1;

  // Round-state scalars. last_play_type = -1 and last_play_size = 0 mean
  // the round has just started (no "previous play" to compare against);
  // is_first_play_of_round captures the same fact more explicitly.
  std::int8_t last_play_type = -1;
  std::int8_t last_play_size = 0;
  std::int8_t is_first_play_of_round = 1;
  // Whoever just played most recently (becomes the next round's host
  // when the current round ends). -1 only at game start, before any play.
  int last_player_to_play = -1;

  // Per-player "have I passed in the current round?" bit.
  std::array<std::int8_t, NPlayers> passed_this_round{};

  // Strength partial order, kept as its transitive closure.
  // strength_above[a][b] == 1  ⟺  type a is known strictly stronger
  // than type b. Diagonal must stay 0.
  std::array<std::array<std::int8_t, kTypeCount>, kTypeCount>
      strength_above{};

  // Per-player per-type card counts. Owner-only visibility — only the
  // owning seat sees its own row; the rest read placeholder.
  std::array<std::array<std::int8_t, kTypeCount>, NPlayers> hand_count{};

  // Per-player number of cards in hand (= row sum of hand_count). Public:
  // the card backs are on the table.
  std::array<std::int8_t, NPlayers> hand_size{};
};

template <int NPlayers>
struct MaskmenState final : public CloneableState<MaskmenState<NPlayers>> {
  using Cfg = MaskmenConfig<NPlayers>;
  MaskmenData<NPlayers> data;

  MaskmenState();

  static const viz::VisibilitySchema& schema();

  void reset_with_seed(std::uint64_t seed) override;
  StateHash64 state_hash() const override;
  void hash_field_slot(Hasher& h, viz::FieldId fid,
                       const std::vector<int>& idx) const override;
  std::any read_field_slot(viz::FieldId fid,
                           const std::vector<int>& idx) const override;
  void write_field_slot(viz::FieldId fid,
                        const std::vector<int>& idx,
                        const std::any& value) override;
  void mask_field_slot(viz::FieldId fid,
                       const std::vector<int>& idx) override;
  const viz::VisibilitySchema& schema_ref() const override { return schema(); }
  int current_player() const override;
  int first_player() const override { return 0; }
  bool is_terminal() const override;
  int num_players() const override { return Cfg::kPlayers; }
  int winner() const override;
  bool is_turn_start() const override { return true; }
};

extern template struct MaskmenData<2>;
extern template struct MaskmenData<3>;
extern template struct MaskmenData<4>;
extern template struct MaskmenState<2>;
extern template struct MaskmenState<3>;
extern template struct MaskmenState<4>;

}  // namespace board_ai::maskmen
