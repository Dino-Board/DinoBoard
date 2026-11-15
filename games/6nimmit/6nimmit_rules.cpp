#include "6nimmit_rules.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "../../engine/core/viz_runtime.h"

namespace board_ai::sixnimmt {

namespace {

std::size_t at(int i) { return static_cast<std::size_t>(i); }

bool is_card(int c) { return c >= 1 && c <= kNumCards; }

template <int N>
int row_length(const SixNimmtData<N>& d, int row) {
  int n = 0;
  while (n < kRowCapacity && d.table[at(row)][at(n)] != 0) ++n;
  return n;
}

template <int N>
int row_tail(const SixNimmtData<N>& d, int row) {
  const int len = row_length(d, row);
  return len == 0 ? 0 : d.table[at(row)][at(len - 1)];
}

// The next seat from `from` on (wrapping) that still has to play, -1 if all have.
template <int N>
int next_to_play(const SixNimmtData<N>& d, int from) {
  for (int i = 0; i < N; ++i) {
    const int p = (from + i) % N;
    if (d.acts_mask[at(p)] && d.pending_commits[at(p)] == 0) return p;
  }
  return -1;
}

// The seat whose face-up card is placed next (the lowest), -1 if none is left.
template <int N>
int next_pending_owner(const SixNimmtData<N>& d) {
  int best = -1;
  for (int p = 0; p < N; ++p) {
    const int c = d.pending_commits[at(p)];
    if (is_card(c) && (best < 0 || c < d.pending_commits[at(best)])) best = p;
  }
  return best;
}

// `player` takes the cards of `row`; `card` starts the row anew.
template <int N>
void take_row(SixNimmtData<N>& d, int player, int row, int card) {
  int heads = 0;
  for (int s = 0; s < kRowCapacity; ++s) {
    heads += head_value(d.table[at(row)][at(s)]);
    d.table[at(row)][at(s)] = 0;
  }
  d.hand_heads[at(player)] += heads;
  d.table[at(row)][0] = static_cast<std::int8_t>(card);
}

// The face-up card of `player` has been placed. The slot stays face up:
// that it is empty again is public, and a session that only receives
// observations is sent the visible slots only. (Hidden again here, such a
// session would keep the card it last saw in the slot and take the seat
// for one that has already played in the next round.)
template <int N>
void consume_pending(SixNimmtData<N>& d, int player) {
  d.pending_commits[at(player)] = 0;
  d.apply_cursor++;
}

// Place the face-up cards in ascending order. Stops early, with
// small_phase_active set, at a card that is below every row end.
template <int N>
void run_apply_loop(SixNimmtData<N>& d) {
  for (int player = next_pending_owner(d); player >= 0; player = next_pending_owner(d)) {
    const int card = d.pending_commits[at(player)];
    int target = -1;
    for (int r = 0; r < kNumRows; ++r) {
      const int t = row_tail(d, r);
      if (t > 0 && t < card && (target < 0 || t > row_tail(d, target))) target = r;
    }
    if (target < 0) {
      d.small_phase_active = true;
      d.small_player = player;
      d.current_player = player;
      d.acts_mask.fill(0);
      d.acts_mask[at(player)] = 1;
      return;
    }
    const int len = row_length(d, target);
    if (len >= kRowCapacity) {
      take_row(d, player, target, card);
    } else {
      d.table[at(target)][at(len)] = static_cast<std::int8_t>(card);
    }
    consume_pending(d, player);
  }
}

// The hand is over: the only player with the fewest bullheads wins it.
template <int N>
void end_hand(SixNimmtData<N>& d) {
  d.terminal = true;
  const int best = *std::min_element(d.hand_heads.begin(), d.hand_heads.end());
  d.winner = -1;
  for (int p = 0; p < N; ++p) {
    if (d.hand_heads[at(p)] != best) continue;
    if (d.winner >= 0) {
      d.winner = -1;
      return;
    }
    d.winner = p;
  }
}

// All cards of the round have been placed.
template <int N>
void finish_round(SixNimmtData<N>& d) {
  d.apply_cursor = 0;
  d.small_phase_active = false;
  d.small_player = -1;
  d.pending_commits.fill(0);  // already empty; the slots stay face up, see consume_pending
  d.acts_mask.fill(1);
  d.current_player = 0;

  d.current_round++;
  if (d.current_round == kHandSize) end_hand(d);
}

}  // namespace

template <int NPlayers>
bool SixNimmtRules<NPlayers>::validate_action(const IGameState& state, ActionId action) const {
  const auto legal = legal_actions(state);
  return std::find(legal.begin(), legal.end(), action) != legal.end();
}

template <int NPlayers>
std::vector<ActionId> SixNimmtRules<NPlayers>::legal_actions(const IGameState& state) const {
  const auto& d = checked_cast<SixNimmtState<NPlayers>>(state).data;
  std::vector<ActionId> actions;
  if (d.terminal) return actions;
  if (d.small_phase_active) {
    for (int r = 0; r < kNumRows; ++r) actions.push_back(kSmallChooseRowOffset + r);
    return actions;
  }
  const int me = d.current_player;
  for (int c = 1; c <= kNumCards; ++c) {
    if (d.hand[at(me)][at(c - 1)] == 1) actions.push_back(kCommitOffset + (c - 1));
  }
  return actions;
}

template <int NPlayers>
ActionId SixNimmtRules<NPlayers>::public_action(const IGameState& state_before, ActionId action) const {
  if (action < kCommitOffset || action >= kCommitOffset + kNumCards) return action;
  // The card that completes the round is turned face up, with the others,
  // inside the same transition: nothing left to hide.
  const auto& d = checked_cast<SixNimmtState<NPlayers>>(state_before).data;
  for (int p = 0; p < NPlayers; ++p) {
    if (p != d.current_player && d.acts_mask[at(p)] && d.pending_commits[at(p)] == 0) return kHiddenCommitAction;
  }
  return action;
}

template <int NPlayers>
void SixNimmtRules<NPlayers>::do_action_fast_impl(IGameState& state, ActionId action,
                                                  std::mt19937_64& /*rng*/) const {
  auto& d = checked_cast<SixNimmtState<NPlayers>>(state).data;
  if (d.terminal) throw std::invalid_argument("6nimmit: the game is over");
  d.ply++;

  if (d.small_phase_active) {
    if (action < kSmallChooseRowOffset || action >= kSmallChooseRowOffset + kNumRows) {
      throw std::invalid_argument("6nimmit: expected a row to take");
    }
    const int player = d.small_player;
    take_row(d, player, action - kSmallChooseRowOffset, d.pending_commits[at(player)]);
    consume_pending(d, player);
    d.small_phase_active = false;
    d.small_player = -1;
  } else {
    if (action < kCommitOffset || action >= kCommitOffset + kNumCards) {
      throw std::invalid_argument("6nimmit: expected a card to play");
    }
    const int card = action - kCommitOffset + 1;
    const int me = d.current_player;
    if (d.hand[at(me)][at(card - 1)] != 1) throw std::invalid_argument("6nimmit: card is not in the hand");
    d.hand[at(me)][at(card - 1)] = 0;
    d.pending_commits[at(me)] = static_cast<std::int8_t>(card);
    // Played face down: only its owner sees the slot until all have played.
    viz::reset_to_base(state, "pending_commits", SixNimmtState<NPlayers>::schema(), {me});
    const int next = next_to_play(d, (me + 1) % NPlayers);
    if (next >= 0) {
      d.current_player = next;  // public cursor: this seat plays next
      return;
    }
    // All cards are turned face up: from here on they are public and out of
    // play.
    d.apply_cursor = 0;
    for (int p = 0; p < NPlayers; ++p) {
      viz::reveal_slot(state, "pending_commits", {p});
      d.used[at(d.pending_commits[at(p)] - 1)] = 1;
    }
  }

  run_apply_loop(d);
  if (!d.small_phase_active) finish_round(d);
}

template <int NPlayers>
void SixNimmtRules<NPlayers>::undo_action_impl(IGameState& /*state*/, const UndoToken& /*token*/) const {
  throw std::logic_error("6nimmit: undo_action is not supported");
}

template class SixNimmtRules<2>;
template class SixNimmtRules<3>;
template class SixNimmtRules<4>;

}  // namespace board_ai::sixnimmt
