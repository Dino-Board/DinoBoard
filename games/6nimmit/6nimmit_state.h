#pragma once

#include <any>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../../engine/core/game_interfaces.h"
#include "../../engine/core/visibility_schema.h"

namespace board_ai::sixnimmt {

// Game constants — DinoBoard 6 nimmt! ships ONE rule version only:
// 104-card full deck (1..104), 10 cards per hand, 4 row heads on table,
// row capacity 5 (6th card triggers "eat the row"), 10 rounds total.
// No Tactics/Logic/Professional/Tournament/long-game variants.
constexpr int kNumCards = 104;
constexpr int kNumRows = 4;
constexpr int kRowCapacity = 5;     // 6th card triggers eat-the-row
constexpr int kNumRounds = 10;
constexpr int kHandSize = 10;

// Action space (global, shared across all variants):
//   action_id ∈ [0, 103]   COMMIT_CARD   (card_number = action_id + 1)
//   action_id ∈ [104, 107] SMALL_CHOOSE_ROW (row = action_id - 104)
constexpr int kCommitOffset = 0;
constexpr int kSmallChooseRowOffset = kNumCards;
constexpr int kActionSpace = kNumCards + kNumRows;  // 108

// What every seat observes in place of a COMMIT_CARD action that leaves
// the round open (SixNimmtRules::public_action): the committed card stays
// face down until all seats have committed, so its id must not travel
// with the action. Outside the action space — never legal, never a
// policy target. The card becomes public through the `card_played` event
// and the table / pending_commits snapshot once it is revealed.
constexpr int kHiddenCommitAction = kActionSpace;

// Bullhead (head) value per card (1-indexed). The judge formula
// (`%5→2 / %10→1 / %11→5 / default 1`) is reproduced exactly so
// our scoring matches `games/6nimmit/参考/main.cpp::compute_heads()`.
inline int head_value(int card) {
  if (card < 1 || card > kNumCards) return 0;
  int h = 0;
  if (card % 5 == 0) h += 2;
  if (card % 10 == 0) h += 1;
  if (card % 11 == 0) h += 5;
  if (h == 0) h = 1;
  return h;
}

template <int NPlayers>
struct SixNimmtConfig {
  static_assert(NPlayers >= 2 && NPlayers <= 4);
  static constexpr int kPlayers = NPlayers;
  // Encoder feature dim breakdown (§1.5):
  //   own_hand          : 104
  //   table dense       : 12  (4 row-tail card_ids + 4 row lengths + 4 row heads)
  //   bullhead counts   : N   (per-player accumulated heads, normalized)
  //   meta              : 10 (current_round one-hot) + 1 (small_phase_active)
  //                      + N (small_player one-hot)
  //   pending per seat  : N * 105   (1 hidden flag + 104 one-hot per seat,
  //                                  rotated own-first; viz=0 → flag=1, all
  //                                  one-hot=0; viz=1 → flag=0, one-hot of
  //                                  the revealed card. Owner sees own
  //                                  pending pre-seal; everyone sees all
  //                                  pendings post-seal during apply phase.)
  //   = 104 + 12 + N + 10 + 1 + N + N*105
  //   = 127 + N*107
  // 2p → 341, 3p → 448, 4p → 555.
  // Note: bullhead pile one-hot (N*104) used to live here; dropped because
  // which specific cards landed in which pile is irrelevant to value
  // (head_value is determined by card_id alone) and irrelevant to belief
  // (the tracker's played_by_ already records "this card is out of play").
  // Counts are kept because they ARE the score: value head needs them.
  static constexpr int kFeatureDim = 127 + NPlayers * 107;
};

template <int NPlayers>
struct SixNimmtData {
  using Cfg = SixNimmtConfig<NPlayers>;

  // Public scalars. `current_player` is the round-internal scan cursor:
  // points at the seat that should act next. Rules advance it on every
  // commit (and on phase transitions). Combined with `acts_mask` and the
  // `pending_commits` placeholder it drives the two-stage current_player()
  // scan that gives 6 nimmt!'s simultaneous-action semantics without any
  // framework-level change. See .claude/skills/game-rules/simultaneous.md
  // ("The three slots").
  int current_player = 0;
  bool terminal = false;
  int winner = -1;
  int ply = 0;

  // Round state.
  int current_round = 0;
  bool small_phase_active = false;
  int small_player = -1;
  int apply_cursor = 0;

  // Hand[p][c-1] = 1 iff player p holds card c (1..104). owner_only viz.
  std::array<std::array<std::int8_t, kNumCards>, Cfg::kPlayers> hand{};

  // Table: 4 rows × up-to-5 slots. Cell value = card number (1..104), 0 = empty.
  std::array<std::array<std::int8_t, kRowCapacity>, kNumRows> table{};

  // bullhead_pile[p][c-1] = 1 iff card c is in player p's eaten pile. all_public.
  std::array<std::array<std::int8_t, kNumCards>, Cfg::kPlayers> bullhead_pile{};

  // Per-player commit-phase slots. owner_only viz at base.
  // After all `acts_mask`-active seats commit, rules call viz::reveal_slot
  // on each pending_commits[p] (all viewers) — the simultaneous reveal is
  // the viz flip itself, no separate apply_queue field. apply_cursor walks
  // pending_commits in ascending-card order. SMALL pause: cursor freezes
  // on a pending slot whose card is smaller than every row tail.
  //
  // Placeholder convention: 0 = "this seat has not committed yet". A
  // perspective whose viz=0 to opp's pending slot reads stale 0 in raw
  // session state — naturally indistinguishable from "opp not yet
  // committed", which is exactly what makes the sim unwind opp's commit
  // as a decision node (.claude/skills/game-rules/simultaneous.md,
  // "The sentinel convention").
  std::array<std::int8_t, Cfg::kPlayers> pending_commits{};

  // `acts_mask[p]` = "this seat must commit during the current phase".
  // all_public, phase-constant (rules write it once at phase boundaries:
  // commit phase = all-true; SMALL/apply phase = single 1 at small_player;
  // terminal = all-false). Combined with `current_player` cursor and
  // `pending_commits` placeholder gives the simultaneous-action scan.
  std::array<std::int8_t, Cfg::kPlayers> acts_mask{};

  // Internal per-seat "this seat has committed this round" flag. NOT a
  // schema slot — does not enter hash/snapshot/wire. Maintained by rules
  // on the GT path (used by the web state serializer / heuristics that
  // run on truth state). On a session state where rules never run
  // do_action_fast (snapshot-driven path), this field stays at reset
  // value (0) — that's fine, no consumer reads session-side `committed`.
  std::array<std::int8_t, Cfg::kPlayers> committed{};
};

template <int NPlayers>
struct SixNimmtState final : public CloneableState<SixNimmtState<NPlayers>> {
  using Cfg = SixNimmtConfig<NPlayers>;
  SixNimmtData<NPlayers> data;

  SixNimmtState();

  // Schema declares 13 fields:
  //   public scalars : current_player (cursor), terminal, winner, ply,
  //                    current_round, small_phase_active, small_player,
  //                    apply_cursor
  //   public 1D/2D   : acts_mask[N], table[4][5], bullhead_pile[N][104]
  //   private 1D     : hand[N][104]        owner_only_first_axis
  //                    pending_commits[N]  owner_only_first_axis at base;
  //                                        rules flip to all-public on
  //                                        seal (simultaneous reveal)
  static const viz::VisibilitySchema& schema();

  void reset_with_seed(std::uint64_t seed) override;

  StateHash64 state_hash() const override;
  void hash_field_slot(Hasher& h, viz::FieldId fid,
                       const std::vector<int>& idx) const override;
  // Per-field bulk hash override. The framework default body walks
  // every slot of `v` and dispatches per-slot to `hash_field_slot`;
  // for 6 nimmt!'s big fields (bullhead_pile, hand, table) we run a
  // tight inlined loop over the underlying std::array, removing the
  // per-slot virtual + switch + idx-vector construction. Behavior is
  // bit-equal: same (fid + idx[]) structural mix + same value mix.
  void hash_field_bulk(Hasher& h, viz::FieldId fid,
                       const viz::VizTensor& v,
                       int perspective) const override;
  void mask_field_slot(viz::FieldId fid,
                       const std::vector<int>& idx) override;
  std::any read_field_slot(viz::FieldId fid,
                           const std::vector<int>& idx) const override;
  void write_field_slot(viz::FieldId fid,
                        const std::vector<int>& idx,
                        const std::any& value) override;
  const viz::VisibilitySchema& schema_ref() const override { return schema(); }
  // Two-stage scan with cursor priority — see
  // .claude/skills/game-rules/simultaneous.md ("current_player() override").
  //   1. If `current_player` is still a placeholder in the active range,
  //      return it (this lets a session whose owner_only opp pending is
  //      stale 0 still align with GT's actual actor via the public
  //      cursor).
  //   2. Else scan the first acts_mask=1 ∧ pending=0 seat (sim naturally
  //      walks into opp decision nodes after self commits).
  //   3. Fallback to small_player for the brief window between commit
  //      seal and acts_mask phase rewrite.
  int current_player() const override {
    const auto& d = data;
    int cur = d.current_player;
    if (cur >= 0 && cur < Cfg::kPlayers
        && d.acts_mask[static_cast<std::size_t>(cur)]
        && d.pending_commits[static_cast<std::size_t>(cur)] == 0) {
      return cur;
    }
    for (int p = 0; p < Cfg::kPlayers; ++p) {
      if (d.acts_mask[static_cast<std::size_t>(p)]
          && d.pending_commits[static_cast<std::size_t>(p)] == 0) {
        return p;
      }
    }
    return d.small_player;
  }
  bool is_terminal() const override { return data.terminal; }
  int num_players() const override { return Cfg::kPlayers; }
  int winner() const override { return data.winner; }

  std::vector<int> losers() const override {
    if (!data.terminal) return {};
    int worst = -1;
    for (int p = 0; p < Cfg::kPlayers; ++p) {
      int s = 0;
      for (int c = 1; c <= kNumCards; ++c) {
        if (data.bullhead_pile[static_cast<size_t>(p)]
                               [static_cast<size_t>(c - 1)])
          s += head_value(c);
      }
      if (s > worst) worst = s;
    }
    if (worst <= 0) return {};
    std::vector<int> result;
    for (int p = 0; p < Cfg::kPlayers; ++p) {
      int s = 0;
      for (int c = 1; c <= kNumCards; ++c) {
        if (data.bullhead_pile[static_cast<size_t>(p)]
                               [static_cast<size_t>(c - 1)])
          s += head_value(c);
      }
      if (s == worst) result.push_back(p);
    }
    if (static_cast<int>(result.size()) == Cfg::kPlayers) return {};
    return result;
  }

  std::vector<int> winners() const override {
    if (!data.terminal) return {};
    auto lo = losers();
    if (lo.empty()) return {};
    std::vector<int> result;
    for (int p = 0; p < Cfg::kPlayers; ++p) {
      bool is_loser = false;
      for (int x : lo) { if (x == p) { is_loser = true; break; } }
      if (!is_loser) result.push_back(p);
    }
    return result;
  }
};

extern template struct SixNimmtData<2>;
extern template struct SixNimmtData<3>;
extern template struct SixNimmtData<4>;
extern template struct SixNimmtState<2>;
extern template struct SixNimmtState<3>;
extern template struct SixNimmtState<4>;

}  // namespace board_ai::sixnimmt
