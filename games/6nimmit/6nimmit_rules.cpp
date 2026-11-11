#include "6nimmit_rules.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "../../engine/core/viz_runtime.h"

namespace board_ai::sixnimmt {

namespace {

template <int NPlayers>
int row_length(const SixNimmtData<NPlayers>& d, int row) {
  int n = 0;
  for (int s = 0; s < kRowCapacity; ++s) {
    if (d.table[static_cast<size_t>(row)][static_cast<size_t>(s)] != 0) ++n;
    else break;
  }
  return n;
}

template <int NPlayers>
int row_tail(const SixNimmtData<NPlayers>& d, int row) {
  int len = row_length<NPlayers>(d, row);
  if (len == 0) return 0;
  return d.table[static_cast<size_t>(row)][static_cast<size_t>(len - 1)];
}

// Cursor-advancement helper: scan from `from` for the next seat that is
// in the active commit set (acts_mask=1) AND still a placeholder
// (pending_commits=0). Drives `current_player_` between commits in the
// simultaneous-action three-slot pattern (see
// .claude/skills/game-rules/simultaneous.md "current_player() override").
template <int NPlayers>
int next_uncommitted_player(const SixNimmtData<NPlayers>& d, int from) {
  for (int i = 0; i < NPlayers; ++i) {
    int p = (from + i) % NPlayers;
    if (d.acts_mask[static_cast<size_t>(p)]
        && d.pending_commits[static_cast<size_t>(p)] == 0) {
      return p;
    }
  }
  return -1;
}

template <int NPlayers>
bool all_acts_committed(const SixNimmtData<NPlayers>& d) {
  for (int p = 0; p < NPlayers; ++p) {
    if (d.acts_mask[static_cast<size_t>(p)]
        && d.pending_commits[static_cast<size_t>(p)] == 0) {
      return false;
    }
  }
  return true;
}

template <int NPlayers>
int count_pending(const SixNimmtData<NPlayers>& d) {
  int n = 0;
  for (int p = 0; p < NPlayers; ++p) {
    if (d.pending_commits[static_cast<size_t>(p)] != 0) ++n;
  }
  return n;
}

template <int NPlayers>
void compute_winner(SixNimmtData<NPlayers>& d) {
  // Lowest bullhead count wins. Tie → winner = -1.
  std::array<int, NPlayers> heads{};
  for (int p = 0; p < NPlayers; ++p) {
    int s = 0;
    for (int c = 1; c <= kNumCards; ++c) {
      if (d.bullhead_pile[static_cast<size_t>(p)]
                          [static_cast<size_t>(c - 1)]) {
        s += head_value(c);
      }
    }
    heads[static_cast<size_t>(p)] = s;
  }
  int best = heads[0];
  for (int p = 1; p < NPlayers; ++p) {
    if (heads[static_cast<size_t>(p)] < best) best = heads[static_cast<size_t>(p)];
  }
  int count_best = 0;
  int best_player = -1;
  for (int p = 0; p < NPlayers; ++p) {
    if (heads[static_cast<size_t>(p)] == best) {
      ++count_best;
      best_player = p;
    }
  }
  d.winner = (count_best == 1) ? best_player : -1;
}

// Find the player whose pending_commits holds the smallest non-zero
// card. Returns -1 if no pending card remains. After seal, the apply
// order is "ascending by card" — derived each step by re-scanning the
// (small) pending_commits array.
template <int NPlayers>
int next_pending_owner(const SixNimmtData<NPlayers>& d) {
  int best_p = -1;
  int best_card = kNumCards + 1;
  for (int p = 0; p < NPlayers; ++p) {
    int c = d.pending_commits[static_cast<size_t>(p)];
    if (c >= 1 && c <= kNumCards && c < best_card) {
      best_card = c;
      best_p = p;
    }
  }
  return best_p;
}

// Apply pending cards in ascending card order. May pause on SMALL: sets
// small_phase_active / small_player / current_player and returns without
// consuming the pending slot. Caller (do_action_fast_impl) is
// responsible for ply / round bookkeeping after this returns.
template <int NPlayers>
void run_apply_loop(IGameState& state, SixNimmtData<NPlayers>& d) {
  // Apply consumes pending entries one at a time; each consumed slot
  // gets pending_commits[p] cleared back to 0. Loop ends when there are
  // no more populated pending slots — equivalently, no slot still has
  // pending != 0.
  while (count_pending<NPlayers>(d) > 0) {
    int player = next_pending_owner<NPlayers>(d);
    if (player < 0) break;  // defensive — should be impossible
    int card = d.pending_commits[static_cast<size_t>(player)];

    // Find largest row tail strictly less than card.
    int target_row = -1;
    int best_tail = -1;
    for (int r = 0; r < kNumRows; ++r) {
      int t = row_tail<NPlayers>(d, r);
      if (t > 0 && t < card && t > best_tail) {
        best_tail = t;
        target_row = r;
      }
    }

    if (target_row < 0) {
      // SMALL: card smaller than all four row tails. Pause; the
      // pending slot stays populated + viz=public so the small_player
      // (and observers) still see it. The three-slot scan needs
      // acts_mask to be single-1 at the small_player so stage-2 finds
      // them as the unique actor; pending_commits[small_player] != 0
      // would otherwise filter them out — for SMALL_CHOOSE_ROW the
      // current_player() override has to fall through to stage 3
      // (small_player fallback). Keep pending populated so apply_loop
      // can still consume the slot after SMALL_CHOOSE_ROW resolves.
      d.small_phase_active = true;
      d.small_player = player;
      d.current_player = player;
      d.acts_mask.fill(0);
      d.acts_mask[static_cast<size_t>(player)] = 1;
      return;
    }

    int len = row_length<NPlayers>(d, target_row);
    if (len >= kRowCapacity) {
      // Eat first 5 into bullhead_pile, new card heads the cleared row.
      for (int s = 0; s < kRowCapacity; ++s) {
        int c = d.table[static_cast<size_t>(target_row)][static_cast<size_t>(s)];
        if (c >= 1 && c <= kNumCards) {
          d.bullhead_pile[static_cast<size_t>(player)]
                          [static_cast<size_t>(c - 1)] = 1;
        }
        d.table[static_cast<size_t>(target_row)][static_cast<size_t>(s)] = 0;
      }
      d.table[static_cast<size_t>(target_row)][0] =
          static_cast<std::int8_t>(card);
    } else {
      d.table[static_cast<size_t>(target_row)][static_cast<size_t>(len)] =
          static_cast<std::int8_t>(card);
    }

    // Consume the pending slot: clear card, drop viz back to base.
    d.pending_commits[static_cast<size_t>(player)] = 0;
    viz::reset_to_base(state, "pending_commits",
                       SixNimmtState<NPlayers>::schema(), {player});
    d.apply_cursor++;
  }
}

template <int NPlayers>
void finish_round_or_advance(IGameState& state, SixNimmtData<NPlayers>& d) {
  // Caller has consumed all pending entries. Reset round state.
  d.committed.fill(0);
  d.apply_cursor = 0;
  d.small_phase_active = false;
  d.small_player = -1;

  // pending_commits should already be 0 + viz back to base after the
  // apply loop walked every slot. Defensive reset matches LoveLetter's
  // pattern in case any future code path leaves a slot dirty.
  d.pending_commits.fill(0);
  for (int p = 0; p < NPlayers; ++p) {
    viz::reset_to_base(state, "pending_commits",
                       SixNimmtState<NPlayers>::schema(), {p});
  }

  d.current_round++;
  if (d.current_round >= kNumRounds) {
    d.terminal = true;
    compute_winner<NPlayers>(d);
    return;
  }

  // Next round: simultaneous commits start. Reset the three slots:
  // acts_mask all 1 (every seat must commit), pending_commits all 0
  // (placeholders), current_player = 0 (cursor anchor; first acts seat).
  d.acts_mask.fill(1);
  d.current_player = 0;
}

// Reveal pending_commits to all viewers — the simultaneous-action
// "翻明" turning point in §1.3. Cards stay in pending_commits; the viz
// flip is the only state change the public observers see. The apply
// loop subsequently consumes pending slots in ascending-card order.
template <int NPlayers>
void seal_pending_into_queue(IGameState& state, SixNimmtData<NPlayers>& d) {
  d.apply_cursor = 0;
  for (int p = 0; p < NPlayers; ++p) {
    viz::reveal_slot(state, "pending_commits", {p});
  }
}

}  // namespace

template <int NPlayers>
bool SixNimmtRules<NPlayers>::validate_action(
    const IGameState& state, ActionId action) const {
  const auto& s = checked_cast<SixNimmtState<NPlayers>>(state);
  if (s.data.terminal) return false;
  auto legal = legal_actions(state);
  return std::find(legal.begin(), legal.end(), action) != legal.end();
}

template <int NPlayers>
std::vector<ActionId> SixNimmtRules<NPlayers>::legal_actions(
    const IGameState& state) const {
  const auto& s = checked_cast<SixNimmtState<NPlayers>>(state);
  const auto& d = s.data;
  if (d.terminal) return {};

  std::vector<ActionId> actions;
  if (d.small_phase_active) {
    actions.reserve(kNumRows);
    for (int r = 0; r < kNumRows; ++r) {
      actions.push_back(kSmallChooseRowOffset + r);
    }
    return actions;
  }

  // Commit phase: legal = cards in current_player's hand.
  const int me = d.current_player;
  actions.reserve(kHandSize);
  for (int c = 1; c <= kNumCards; ++c) {
    if (d.hand[static_cast<size_t>(me)][static_cast<size_t>(c - 1)]) {
      actions.push_back(kCommitOffset + (c - 1));
    }
  }
  return actions;
}

template <int NPlayers>
ActionId SixNimmtRules<NPlayers>::public_action(
    const IGameState& state_before, ActionId action) const {
  if (action < kCommitOffset || action >= kCommitOffset + kNumCards) {
    return action;
  }
  // A commit stays face down as long as some other seat still has to
  // commit. The commit that completes the round is turned face up, with
  // all the others, inside the same transition — nothing left to hide.
  // (Whether a seat has committed is public: its face-down card lies on
  // the table and the cursor serves seats in a fixed order.)
  const auto& d = checked_cast<SixNimmtState<NPlayers>>(state_before).data;
  const int me = d.current_player;
  for (int p = 0; p < NPlayers; ++p) {
    if (p == me) continue;
    if (d.acts_mask[static_cast<size_t>(p)] &&
        d.pending_commits[static_cast<size_t>(p)] == 0) {
      return kHiddenCommitAction;
    }
  }
  return action;
}

template <int NPlayers>
void SixNimmtRules<NPlayers>::do_action_fast_impl(
    IGameState& state, ActionId action, std::mt19937_64& /*rng*/) const {
  auto& s = checked_cast<SixNimmtState<NPlayers>>(state);
  auto& d = s.data;
  d.ply++;

  if (d.small_phase_active) {
    // SMALL_CHOOSE_ROW: small_player picks row to eat.
    if (action < kSmallChooseRowOffset ||
        action >= kSmallChooseRowOffset + kNumRows) {
      throw std::invalid_argument("6nimmit: expected SMALL_CHOOSE_ROW");
    }
    int row = action - kSmallChooseRowOffset;
    int small_player = d.small_player;
    int card = d.pending_commits[static_cast<size_t>(small_player)];

    // Eat the chosen row into small_player's bullhead_pile.
    for (int s_idx = 0; s_idx < kRowCapacity; ++s_idx) {
      int c = d.table[static_cast<size_t>(row)][static_cast<size_t>(s_idx)];
      if (c >= 1 && c <= kNumCards) {
        d.bullhead_pile[static_cast<size_t>(small_player)]
                        [static_cast<size_t>(c - 1)] = 1;
      }
      d.table[static_cast<size_t>(row)][static_cast<size_t>(s_idx)] = 0;
    }
    // Place the small player's card as the new row head.
    d.table[static_cast<size_t>(row)][0] = static_cast<std::int8_t>(card);

    // Consume the small_player's pending slot: clear card, drop viz
    // back to base.
    d.pending_commits[static_cast<size_t>(small_player)] = 0;
    viz::reset_to_base(state, "pending_commits",
                       SixNimmtState<NPlayers>::schema(), {small_player});
    d.apply_cursor++;
    d.small_phase_active = false;
    d.small_player = -1;

    // Resume apply loop. May SMALL again or finish the round.
    run_apply_loop<NPlayers>(state, d);
    if (!d.small_phase_active && count_pending<NPlayers>(d) == 0) {
      finish_round_or_advance<NPlayers>(state, d);
    } else if (!d.small_phase_active) {
      // Defensive: loop returned without finishing without SMALL — should
      // be impossible by run_apply_loop's contract.
      throw std::logic_error("6nimmit: apply loop returned in unknown state");
    }
    return;
  }

  // Commit phase: action is a card_id.
  if (action < kCommitOffset || action >= kCommitOffset + kNumCards) {
    throw std::invalid_argument("6nimmit: expected COMMIT_CARD");
  }
  int card = action - kCommitOffset + 1;  // 1..104
  int me = d.current_player;

  // Remove card from hand. Owner_only_first_axis viz on hand keeps
  // this change owner-only-visible automatically (base viz unchanged).
  d.hand[static_cast<size_t>(me)][static_cast<size_t>(card - 1)] = 0;

  // Place into pending_commits (owner_only_first_axis: viz already
  // owner-only by base). pending_commits[me] != 0 = "this seat has
  // committed this round" — single source of truth in the new
  // three-slot pattern. data.committed[] is kept as an internal
  // non-schema mirror for the web/heuristic GT-side serializer (does
  // NOT enter hash/snapshot/wire — see 6nimmit_state.h).
  d.pending_commits[static_cast<size_t>(me)] = static_cast<std::int8_t>(card);
  d.committed[static_cast<size_t>(me)] = 1;

  if (!all_acts_committed<NPlayers>(d)) {
    // Round still gathering commits: advance the public cursor to the
    // next uncommitted seat (acts_mask=1 ∧ pending=0). On a session
    // whose owner-only opp pending reads stale 0, the public cursor
    // (along with acts_mask) is what carries "GT has advanced past p0
    // commit" across the snapshot wire — see §1.4 stage 1.
    int nxt = next_uncommitted_player<NPlayers>(d, (me + 1) % NPlayers);
    d.current_player = nxt;
    return;
  }

  // All acts_mask=1 seats have non-zero pending: rules flip
  // pending_commits viz to all-public (the simultaneous reveal in §1.3)
  // and run the deterministic apply loop, consuming pending slots in
  // ascending-card order.
  seal_pending_into_queue<NPlayers>(state, d);
  run_apply_loop<NPlayers>(state, d);

  if (!d.small_phase_active && count_pending<NPlayers>(d) == 0) {
    finish_round_or_advance<NPlayers>(state, d);
  }
  // If small_phase_active: paused waiting for SMALL_CHOOSE_ROW; no
  // round finish until that resumes the loop.
}

template <int NPlayers>
void SixNimmtRules<NPlayers>::undo_action_impl(
    IGameState& /*state*/, const UndoToken& /*token*/) const {
  // §3.4: tail solver not supported. do_action_deterministic + undo not
  // wired; throwing fail-loud guards against accidental use.
  throw std::logic_error(
      "6nimmit: undo_action not supported (tail solver disabled, §3.4)");
}

template class SixNimmtRules<2>;
template class SixNimmtRules<3>;
template class SixNimmtRules<4>;

}  // namespace board_ai::sixnimmt
