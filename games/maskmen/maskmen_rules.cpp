#include "maskmen_rules.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "../../engine/core/viz_runtime.h"

namespace board_ai::maskmen {

namespace {

// Expand the transitive closure to include the new edge (u above v).
// Pre: strength_above[v][u] == 0 (legal_actions guards this). If
// strength_above[u][v] is already 1, this is a no-op.
//
// Three-pass Warshall over the 6x6 matrix:
//   1. anything already above u is now above v;
//   2. v is now above anything that v was already above (the new edge
//      itself);
//   3. anything above u is above anything below v (cross-product).
template <int NPlayers>
void add_strength_edge(MaskmenData<NPlayers>& d, int u, int v) {
  if (u == v) {
    throw std::logic_error(
        "MaskmenRules::add_strength_edge: u == v (self loop)");
  }
  if (d.strength_above[static_cast<size_t>(v)][static_cast<size_t>(u)]) {
    throw std::logic_error(
        "MaskmenRules::add_strength_edge: contradicts existing "
        "strength relation (v already above u)");
  }
  if (d.strength_above[static_cast<size_t>(u)][static_cast<size_t>(v)]) {
    return;
  }
  d.strength_above[static_cast<size_t>(u)][static_cast<size_t>(v)] = 1;

  // Snapshot the columns we need before the cross-product overwrites
  // them. (a above u) and (v above b) both come from the matrix as it
  // was when we entered, plus the brand-new (u,v) edge we just wrote.
  std::array<std::int8_t, kTypeCount> above_u{};
  std::array<std::int8_t, kTypeCount> below_v{};
  for (int a = 0; a < kTypeCount; ++a) {
    above_u[static_cast<size_t>(a)] =
        d.strength_above[static_cast<size_t>(a)][static_cast<size_t>(u)];
  }
  for (int b = 0; b < kTypeCount; ++b) {
    below_v[static_cast<size_t>(b)] =
        d.strength_above[static_cast<size_t>(v)][static_cast<size_t>(b)];
  }

  // Pass 1: a above u → a above v.
  for (int a = 0; a < kTypeCount; ++a) {
    if (above_u[static_cast<size_t>(a)]) {
      d.strength_above[static_cast<size_t>(a)][static_cast<size_t>(v)] = 1;
    }
  }
  // Pass 2: u above b for every b that v is already above.
  for (int b = 0; b < kTypeCount; ++b) {
    if (below_v[static_cast<size_t>(b)]) {
      d.strength_above[static_cast<size_t>(u)][static_cast<size_t>(b)] = 1;
    }
  }
  // Pass 3: cross product — a above v above b ⇒ a above b.
  for (int a = 0; a < kTypeCount; ++a) {
    if (!above_u[static_cast<size_t>(a)]) continue;
    for (int b = 0; b < kTypeCount; ++b) {
      if (below_v[static_cast<size_t>(b)]) {
        d.strength_above[static_cast<size_t>(a)][static_cast<size_t>(b)] = 1;
      }
    }
  }

  // Diagonal must stay 0. (If we ever drove it to 1 we'd hit the
  // "v above u" guard on the next add — but check for cheap sanity.)
  for (int i = 0; i < kTypeCount; ++i) {
    if (d.strength_above[static_cast<size_t>(i)][static_cast<size_t>(i)]) {
      throw std::logic_error(
          "MaskmenRules::add_strength_edge: closure produced self-loop "
          "(implies the partial-order rules introduced a cycle)");
    }
  }
}

// Returns true iff seat `p` has at least one card of any type left.
template <int NPlayers>
bool seat_has_any_card(const MaskmenData<NPlayers>& d, int p) {
  for (int t = 0; t < kTypeCount; ++t) {
    if (d.hand_count[static_cast<size_t>(p)][static_cast<size_t>(t)] > 0) {
      return true;
    }
  }
  return false;
}

// Walk seats clockwise from start_inclusive looking for the next one
// that hasn't passed in the current round AND still has cards. Returns
// the seat index, or -1 if no such seat exists.
template <int NPlayers>
int next_active_seat(const MaskmenData<NPlayers>& d, int start_inclusive) {
  for (int k = 0; k < NPlayers; ++k) {
    const int p = (start_inclusive + k) % NPlayers;
    if (d.passed_this_round[static_cast<size_t>(p)]) continue;
    if (!seat_has_any_card(d, p)) continue;
    return p;
  }
  return -1;
}

// How many seats are still active in the current round (haven't passed
// AND still have cards). Round ends once this drops to <= 1.
template <int NPlayers>
int count_active_seats(const MaskmenData<NPlayers>& d) {
  int n = 0;
  for (int p = 0; p < NPlayers; ++p) {
    if (d.passed_this_round[static_cast<size_t>(p)]) continue;
    if (!seat_has_any_card(d, p)) continue;
    ++n;
  }
  return n;
}

}  // namespace

template <int NPlayers>
bool MaskmenRules<NPlayers>::validate_action(
    const IGameState& state, ActionId action) const {
  auto legal = legal_actions(state);
  for (ActionId a : legal) {
    if (a == action) return true;
  }
  return false;
}

template <int NPlayers>
std::vector<ActionId> MaskmenRules<NPlayers>::legal_actions(
    const IGameState& state) const {
  const auto& s = checked_cast<MaskmenState<NPlayers>>(state);
  const auto& d = s.data;
  if (d.terminal) return {};

  std::vector<ActionId> out;
  out.reserve(kActionSpace);

  const int cur = d.current_player_;
  const int last_t = d.last_play_type;
  const int last_s = d.last_play_size;
  const bool first_play = (d.is_first_play_of_round != 0);

  if (first_play) {
    // Host opens the round — must play exactly 1 card of any type they
    // actually hold (PLAN §1.4 rule 2; user round 4 round-1 clarification).
    for (int t = 0; t < kTypeCount; ++t) {
      if (d.hand_count[static_cast<size_t>(cur)][static_cast<size_t>(t)] >= 1) {
        out.push_back(encode_play(t));
      }
    }
    // Host opens by playing — pass is not a legal opener (otherwise the
    // game can deadlock with everyone passing at round-start).
  } else {
    for (int t = 0; t < kTypeCount; ++t) {
      const int hand = d.hand_count[static_cast<size_t>(cur)]
                                   [static_cast<size_t>(t)];
      if (hand <= 0) continue;
      if (t == last_t) continue;  // same type → no relation can be built
      if (d.strength_above[static_cast<size_t>(last_t)]
                          [static_cast<size_t>(t)]) {
        // Known weaker than last play — illegal.
        continue;
      }
      if (d.strength_above[static_cast<size_t>(t)]
                          [static_cast<size_t>(last_t)]) {
        // (A2) Known stronger — must match size exactly.
        if (hand >= last_s) out.push_back(encode_play(t));
      } else {
        // (A1) Unknown relation — must play last_s + 1, capped at kMaxPlaySize.
        const int need = last_s + 1;
        if (need <= kMaxPlaySize && hand >= need) {
          out.push_back(encode_play(t));
        }
      }
    }
    // Pass is always legal mid-round (host has already opened).
    out.push_back(static_cast<ActionId>(kPassAction));
  }
  return out;
}

template <int NPlayers>
void MaskmenRules<NPlayers>::do_action_fast_impl(
    IGameState& state, ActionId action, std::mt19937_64& /*rng*/) const {
  auto& s = checked_cast<MaskmenState<NPlayers>>(state);
  auto& d = s.data;
  const int cur = d.current_player_;

  if (is_pass(action)) {
    if (d.is_first_play_of_round) {
      throw std::logic_error(
          "MaskmenRules::do_action_fast: host cannot pass the first "
          "play of a round (legal_actions guards this)");
    }
    d.passed_this_round[static_cast<size_t>(cur)] = 1;
  } else {
    const int t = play_type_from_action(action);

    // Derive the size implied by current round state.
    //   first play of round → 1
    //   known stronger (A2) → last_play_size
    //   unknown (A1)        → last_play_size + 1
    // legal_actions has already filtered by feasibility (hand size, no
    // known-weaker, no same-type, no >kMaxPlaySize cap); this branch is
    // strictly derivation, not validation.
    int sz;
    if (d.is_first_play_of_round) {
      sz = 1;
    } else {
      const int v = d.last_play_type;
      if (t == v) {
        throw std::logic_error(
            "MaskmenRules::do_action_fast: cannot play same type as "
            "previous play");
      }
      if (d.strength_above[static_cast<size_t>(v)][static_cast<size_t>(t)]) {
        throw std::logic_error(
            "MaskmenRules::do_action_fast: cannot play known-weaker "
            "type against current line");
      }
      const bool known_above =
          d.strength_above[static_cast<size_t>(t)][static_cast<size_t>(v)] != 0;
      if (known_above) {
        sz = d.last_play_size;          // A2
      } else {
        sz = d.last_play_size + 1;      // A1
        if (sz > kMaxPlaySize) {
          throw std::logic_error(
              "MaskmenRules::do_action_fast: A1 size > kMaxPlaySize "
              "(legal_actions should have filtered this)");
        }
        add_strength_edge<NPlayers>(d, t, v);
      }
    }

    if (d.hand_count[static_cast<size_t>(cur)][static_cast<size_t>(t)] < sz) {
      throw std::logic_error(
          "MaskmenRules::do_action_fast: not enough cards in hand for "
          "play");
    }

    d.hand_count[static_cast<size_t>(cur)][static_cast<size_t>(t)] -=
        static_cast<std::int8_t>(sz);
    d.hand_size[static_cast<size_t>(cur)] -= static_cast<std::int8_t>(sz);
    d.last_play_type = static_cast<std::int8_t>(t);
    d.last_play_size = static_cast<std::int8_t>(sz);
    d.is_first_play_of_round = 0;
    d.last_player_to_play = cur;

    // Game over: first seat to clear its hand wins. Winner-takes-all.
    if (!seat_has_any_card(d, cur)) {
      d.terminal = true;
      d.winner = cur;
      return;
    }
  }

  // Round transition: round ends as soon as <= 1 seat is still active.
  // The last seat to have actually played becomes the next host.
  if (count_active_seats(d) <= 1) {
    for (auto& p : d.passed_this_round) p = 0;
    d.last_play_type = -1;
    d.last_play_size = 0;
    d.is_first_play_of_round = 1;
    if (d.last_player_to_play >= 0 &&
        seat_has_any_card(d, d.last_player_to_play)) {
      d.current_player_ = d.last_player_to_play;
    } else {
      // The last player to play just emptied their hand (handled above
      // as game-over), or — degenerate — no one played at all this
      // round. The latter cannot happen because legal_actions forces
      // the host to play, but be defensive and pick the next seat with
      // cards to keep the game advancing.
      const int fallback = next_active_seat<NPlayers>(d, 0);
      if (fallback < 0) {
        throw std::logic_error(
            "MaskmenRules::do_action_fast: round ended with no seat "
            "able to start the next round");
      }
      d.current_player_ = fallback;
    }
    return;
  }

  // Otherwise advance to the next active seat clockwise.
  const int nxt = next_active_seat<NPlayers>(d, (cur + 1) % NPlayers);
  if (nxt < 0) {
    throw std::logic_error(
        "MaskmenRules::do_action_fast: no active seat after action "
        "(should have triggered round end)");
  }
  d.current_player_ = nxt;
}

template <int NPlayers>
void MaskmenRules<NPlayers>::undo_action_impl(
    IGameState& /*state*/, const UndoToken& /*token*/) const {
  // No tail solver / undo path for maskmen — selfplay/sim only use
  // do_action_fast. Reaching here is a contract violation.
  throw std::logic_error(
      "MaskmenRules::undo_action_impl: maskmen does not support undo");
}

template class MaskmenRules<2>;
template class MaskmenRules<3>;
template class MaskmenRules<4>;

}  // namespace board_ai::maskmen
