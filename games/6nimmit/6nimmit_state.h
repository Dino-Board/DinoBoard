#pragma once

#include <algorithm>
#include <any>
#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include "../../engine/core/game_interfaces.h"
#include "../../engine/core/visibility_schema.h"

namespace board_ai::sixnimmt {

// 6 nimmt!, one hand, for two to four players.
//
// Four row starters on the table, ten cards each from all 104, ten rounds
// of simultaneous play; the game ends with the hand. `used` holds every card
// shown so far; the cards the other players can hold are the ones neither
// used, nor in my hand, nor lying face up.
//
// A match to 66 points (Board Game Arena) shuffles all 104 cards before
// every hand, so nothing but the score carries over from one hand to the
// next: it is played one hand at a time, each a game of its own. The value
// of a hand is its bullhead margin (SixNimmtMarginValueModel), not who won
// it.
constexpr int kNumCards = 104;
constexpr int kNumRows = 4;
constexpr int kRowCapacity = 5;  // the sixth card takes the row
constexpr int kHandSize = 10;    // cards per player = rounds
constexpr int kMaxPlayers = 4;

// Bullheads of margin that count as a whole win or loss (value +-1). Between
// the heuristics a hand's margin is beyond 25 in 1-3.5% of the hands
// (2-4 players).
constexpr float kMarginScale = 25.0f;

// Action space, shared by all player counts:
//   [0, 103]    play card (card = action + 1)
//   [104, 107]  take row (row = action - 104), for a card below every row end
constexpr int kCommitOffset = 0;
constexpr int kSmallChooseRowOffset = kNumCards;
constexpr int kActionSpace = kNumCards + kNumRows;  // 108

// What the other seats observe in place of a card played face down
// (SixNimmtRules::public_action). Outside the action space.
constexpr int kHiddenCommitAction = kActionSpace;

// Bullheads on a card: 55 has 7, other multiples of 11 have 5, multiples of
// 10 have 3, other multiples of 5 have 2, the rest 1.
inline int head_value(int card) {
  if (card < 1 || card > kNumCards) return 0;
  int h = 0;
  if (card % 5 == 0) h += 2;
  if (card % 10 == 0) h += 1;
  if (card % 11 == 0) h += 5;
  return h == 0 ? 1 : h;
}

template <int NPlayers>
struct SixNimmtConfig {
  static_assert(NPlayers >= 2 && NPlayers <= kMaxPlayers);
  static constexpr int kPlayers = NPlayers;
};

template <int NPlayers>
struct SixNimmtData {
  using Cfg = SixNimmtConfig<NPlayers>;

  // Seat to act: the commit cursor, see current_player() below and
  // .claude/skills/game-rules/simultaneous.md.
  int current_player = 0;
  bool terminal = false;
  int winner = -1;
  int ply = 0;

  int current_round = 0;  // rounds finished, 0..10
  bool small_phase_active = false;  // a player is choosing a row to take
  int small_player = -1;
  int apply_cursor = 0;  // face-up cards of this round already placed

  // Bullheads taken so far.
  std::array<int, Cfg::kPlayers> hand_heads{};

  // 1 = this seat still has to play a card (commit phase) or is the one
  // choosing a row. Public.
  std::array<std::int8_t, Cfg::kPlayers> acts_mask{};

  // Rows: card numbers, first card first, 0 = empty.
  std::array<std::array<std::int8_t, kRowCapacity>, kNumRows> table{};

  // used[c-1] = 1: card c has been shown (as a row starter or a played
  // card). Public. The deck is the cards that are neither used nor in a hand
  // nor lying face down.
  std::array<std::int8_t, kNumCards> used{};

  // hand[p][c-1] = 1: player p holds card c. Owner only.
  std::array<std::array<std::int8_t, kNumCards>, Cfg::kPlayers> hand{};

  // The card each player has played this round, 0 = none (not played yet, or
  // already placed). Hidden from the other seats while it lies face down;
  // turned face up for everyone once all have played, and cleared one by one
  // as the cards are placed.
  std::array<std::int8_t, Cfg::kPlayers> pending_commits{};
};

// Deal from all 104 cards: ten cards each and four row starters.
template <int NPlayers>
void deal_hand(SixNimmtData<NPlayers>& d, std::mt19937_64& rng);

template <int NPlayers>
struct SixNimmtState final : public CloneableState<SixNimmtState<NPlayers>> {
  using Cfg = SixNimmtConfig<NPlayers>;
  SixNimmtData<NPlayers> data;

  SixNimmtState();

  static const viz::VisibilitySchema& schema();

  void reset_with_seed(std::uint64_t seed) override;

  StateHash64 state_hash() const override;
  void hash_field_slot(Hasher& h, viz::FieldId fid, const std::vector<int>& idx) const override;
  void hash_field_bulk(Hasher& h, viz::FieldId fid, const viz::VizTensor& v, int perspective) const override;
  void mask_field_slot(viz::FieldId fid, const std::vector<int>& idx) override;
  std::any read_field_slot(viz::FieldId fid, const std::vector<int>& idx) const override;
  void write_field_slot(viz::FieldId fid, const std::vector<int>& idx, const std::any& value) override;
  const viz::VisibilitySchema& schema_ref() const override { return schema(); }

  // The seat to act. The players play their cards "at the same time": the
  // public cursor names the seat that plays next; a seat whose face-down
  // card this state cannot see (stale 0) counts as not having played yet,
  // which is what lets a search play the opponents' cards out itself.
  int current_player() const override {
    const auto& d = data;
    const int cur = d.current_player;
    if (cur >= 0 && cur < Cfg::kPlayers && d.acts_mask[static_cast<std::size_t>(cur)] &&
        d.pending_commits[static_cast<std::size_t>(cur)] == 0) {
      return cur;
    }
    for (int p = 0; p < Cfg::kPlayers; ++p) {
      if (d.acts_mask[static_cast<std::size_t>(p)] && d.pending_commits[static_cast<std::size_t>(p)] == 0) {
        return p;
      }
    }
    return d.small_player;
  }
  bool is_terminal() const override { return data.terminal; }
  int num_players() const override { return Cfg::kPlayers; }
  // The only player with the fewest bullheads, -1 if that is shared.
  int winner() const override { return data.winner; }

  // The players with the most bullheads lose, everybody else wins; nobody
  // does when all are level.
  std::vector<int> losers() const override {
    std::vector<int> out;
    if (!data.terminal) return out;
    int low = data.hand_heads[0], high = data.hand_heads[0];
    for (int p = 1; p < Cfg::kPlayers; ++p) {
      low = std::min(low, data.hand_heads[static_cast<std::size_t>(p)]);
      high = std::max(high, data.hand_heads[static_cast<std::size_t>(p)]);
    }
    if (low == high) return out;
    for (int p = 0; p < Cfg::kPlayers; ++p) {
      if (data.hand_heads[static_cast<std::size_t>(p)] == high) out.push_back(p);
    }
    return out;
  }
  std::vector<int> winners() const override {
    std::vector<int> out;
    const std::vector<int> lost = losers();
    if (lost.empty()) return out;
    for (int p = 0; p < Cfg::kPlayers; ++p) {
      bool is_loser = false;
      for (int x : lost) is_loser = is_loser || x == p;
      if (!is_loser) out.push_back(p);
    }
    return out;
  }
};

// The bullhead margin of a finished hand: the other players' average minus
// mine, in units of kMarginScale, clipped to [-1, 1]. Zero-sum unless
// clipped.
template <int NPlayers>
float margin_value(const SixNimmtData<NPlayers>& d, int player) {
  int total = 0;
  for (int p = 0; p < NPlayers; ++p) total += d.hand_heads[static_cast<std::size_t>(p)];
  const int mine = d.hand_heads[static_cast<std::size_t>(player)];
  const float others = static_cast<float>(total - mine) / static_cast<float>(NPlayers - 1);
  return std::clamp((others - static_cast<float>(mine)) / kMarginScale, -1.0f, 1.0f);
}

template <int NPlayers>
class SixNimmtMarginValueModel final : public IStateValueModel {
 public:
  float terminal_value_for_player(const IGameState& state, int player) const override {
    if (!state.is_terminal()) return 0.0f;
    return margin_value(checked_cast<SixNimmtState<NPlayers>>(state).data, player);
  }
};

extern template struct SixNimmtData<2>;
extern template struct SixNimmtData<3>;
extern template struct SixNimmtData<4>;
extern template struct SixNimmtState<2>;
extern template struct SixNimmtState<3>;
extern template struct SixNimmtState<4>;

}  // namespace board_ai::sixnimmt
