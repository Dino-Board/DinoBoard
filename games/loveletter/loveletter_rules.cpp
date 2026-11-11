#include "loveletter_rules.h"

#include <algorithm>
#include <cstdint>
#include <utility>

#include "../../engine/core/viz_runtime.h"

namespace board_ai::loveletter {

namespace {

struct DecodedAction {
  std::int8_t card = 0;
  int target = -1;
  std::int8_t guess = 0;
};

DecodedAction decode_action(ActionId action) {
  DecodedAction r;
  if (action >= kGuardOffset && action < kGuardOffset + kGuardCount) {
    r.card = kGuard;
    int idx = action - kGuardOffset;
    r.target = idx / 7;
    r.guess = static_cast<std::int8_t>(idx % 7 + 2);
  } else if (action >= kPriestOffset && action < kPriestOffset + kPriestCount) {
    r.card = kPriest;
    r.target = action - kPriestOffset;
  } else if (action >= kBaronOffset && action < kBaronOffset + kBaronCount) {
    r.card = kBaron;
    r.target = action - kBaronOffset;
  } else if (action == kHandmaidAction) {
    r.card = kHandmaid;
  } else if (action >= kPrinceOffset && action < kPrinceOffset + kPrinceCount) {
    r.card = kPrince;
    r.target = action - kPrinceOffset;
  } else if (action >= kKingOffset && action < kKingOffset + kKingCount) {
    r.card = kKing;
    r.target = action - kKingOffset;
  } else if (action == kCountessAction) {
    r.card = kCountess;
  } else if (action == kPrincessAction) {
    r.card = kPrincess;
  }
  return r;
}

template <int NPlayers>
bool has_valid_target(const LoveLetterData<NPlayers>& d, int me) {
  for (int p = 0; p < NPlayers; ++p) {
    if (p == me) continue;
    if (d.alive[p] && !d.protected_flags[p]) return true;
  }
  return false;
}

template <int NPlayers>
void add_guard_actions(const LoveLetterData<NPlayers>& d, int me,
                       std::vector<ActionId>& actions) {
  bool found = false;
  for (int t = 0; t < NPlayers; ++t) {
    if (t == me || !d.alive[t] || d.protected_flags[t]) continue;
    for (int g = 2; g <= 8; ++g) {
      actions.push_back(kGuardOffset + t * 7 + (g - 2));
    }
    found = true;
  }
  if (!found) {
    actions.push_back(kGuardOffset + me * 7);
  }
}

template <int NPlayers>
void add_targeted_actions(const LoveLetterData<NPlayers>& d, int me,
                          int offset, bool allow_self,
                          std::vector<ActionId>& actions) {
  bool found = false;
  for (int t = 0; t < NPlayers; ++t) {
    if (!allow_self && t == me) continue;
    if (!d.alive[t] || (t != me && d.protected_flags[t])) continue;
    actions.push_back(offset + t);
    found = true;
  }
  if (!found) {
    actions.push_back(offset + me);
  }
}

template <int NPlayers>
void add_actions_for_card(const LoveLetterData<NPlayers>& d, int me,
                          std::int8_t card, std::vector<ActionId>& actions) {
  switch (card) {
    case kGuard:
      add_guard_actions(d, me, actions);
      break;
    case kPriest:
      add_targeted_actions(d, me, kPriestOffset, false, actions);
      break;
    case kBaron:
      add_targeted_actions(d, me, kBaronOffset, false, actions);
      break;
    case kHandmaid:
      actions.push_back(kHandmaidAction);
      break;
    case kPrince:
      add_targeted_actions(d, me, kPrinceOffset, true, actions);
      break;
    case kKing:
      add_targeted_actions(d, me, kKingOffset, false, actions);
      break;
    case kCountess:
      actions.push_back(kCountessAction);
      break;
    case kPrincess:
      actions.push_back(kPrincessAction);
      break;
    default:
      break;
  }
}

template <int NPlayers>
void eliminate_player(IGameState& state, LoveLetterData<NPlayers>& d, int player) {
  d.alive[player] = 0;
  d.protected_flags[player] = 0;
  if (d.hand[player] != 0) {
    d.discard_count[static_cast<size_t>(player)]
                   [static_cast<size_t>(d.hand[player])]++;
    d.hand[player] = 0;
  }
  // Hand contents now 0; drop any in-round reveals so viz returns to
  // owner_only_first_axis base (owner sees self, no other viewer sees).
  viz::reset_to_base(state, "hand",
                     LoveLetterState<NPlayers>::schema(), {player});
}

template <int NPlayers>
int deck_total(const LoveLetterData<NPlayers>& d) {
  int n = 0;
  for (int i = 1; i <= kCardTypes; ++i) n += d.deck_count[static_cast<size_t>(i)];
  return n;
}

template <int NPlayers>
std::int8_t draw_from_deck_local(LoveLetterData<NPlayers>& d, std::mt19937_64& rng) {
  int total = deck_total<NPlayers>(d);
  if (total <= 0) return 0;
  std::uint64_t r = rng();
  int pick = static_cast<int>(r % static_cast<std::uint64_t>(total));
  for (int i = 1; i <= kCardTypes; ++i) {
    int n = d.deck_count[static_cast<size_t>(i)];
    if (pick < n) {
      d.deck_count[static_cast<size_t>(i)] = static_cast<std::int8_t>(n - 1);
      // Sole runtime draw site: keep first-class deck_size in sync
      // with the multiset.
      if (d.deck_size > 0) d.deck_size = static_cast<std::int8_t>(d.deck_size - 1);
      return static_cast<std::int8_t>(i);
    }
    pick -= n;
  }
  return 0;
}

template <int NPlayers>
int count_alive(const LoveLetterData<NPlayers>& d) {
  int n = 0;
  for (int p = 0; p < NPlayers; ++p) {
    if (d.alive[p]) ++n;
  }
  return n;
}

template <int NPlayers>
int sole_survivor(const LoveLetterData<NPlayers>& d) {
  int survivor = -1;
  for (int p = 0; p < NPlayers; ++p) {
    if (d.alive[p]) {
      if (survivor >= 0) return -1;
      survivor = p;
    }
  }
  return survivor;
}

template <int NPlayers>
void check_end_game(LoveLetterData<NPlayers>& d) {
  int surv = sole_survivor(d);
  if (surv >= 0) {
    d.terminal = true;
    d.winner = surv;
    return;
  }

  if (deck_total<NPlayers>(d) == 0) {
    int best_card = -1;
    int best_player = -1;
    bool tie = false;
    auto pile_sum = [&](int p) {
      int s = 0;
      for (int c = 1; c <= kCardTypes; ++c) {
        s += c * d.discard_count[static_cast<size_t>(p)][static_cast<size_t>(c)];
      }
      return s;
    };
    for (int p = 0; p < NPlayers; ++p) {
      if (!d.alive[p]) continue;
      int card = d.hand[p];
      if (card > best_card) {
        best_card = card;
        best_player = p;
        tie = false;
      } else if (card == best_card) {
        int sum_p = pile_sum(p);
        int sum_best = pile_sum(best_player);
        if (sum_p > sum_best) {
          best_player = p;
          tie = false;
        } else if (sum_p == sum_best) {
          tie = true;
        }
      }
    }
    d.terminal = true;
    d.winner = tie ? -1 : best_player;
  }
}

template <int NPlayers>
int next_alive_player(const LoveLetterData<NPlayers>& d, int from) {
  for (int i = 1; i <= NPlayers; ++i) {
    int p = (from + i) % NPlayers;
    if (d.alive[p]) return p;
  }
  return from;
}

// Round ended with the deck empty: "all players still in the round reveal
// their hands". Makes the showdown — and thus winners()/losers() on a
// tie — readable from any seat's view, not just from truth.
template <int NPlayers>
void reveal_showdown_hands(IGameState& state,
                           const LoveLetterData<NPlayers>& d) {
  if (!d.terminal || count_alive(d) < 2) return;
  for (int p = 0; p < NPlayers; ++p) {
    if (d.alive[p]) viz::reveal_slot(state, "hand", {p});
  }
}

template <int NPlayers>
void advance_turn(IGameState& state, LoveLetterData<NPlayers>& d,
                  std::mt19937_64& rng) {
  check_end_game(d);
  if (d.terminal) {
    reveal_showdown_hands(state, d);
    return;
  }

  int next = next_alive_player(d, d.current_player);
  d.current_player = next;

  d.protected_flags[static_cast<size_t>(next)] = 0;

  if (deck_total<NPlayers>(d) == 0) {
    check_end_game(d);
    reveal_showdown_hands(state, d);
    return;
  }

  d.drawn_card = draw_from_deck_local<NPlayers>(d, rng);
  // drawn_card scalar holds the new card. Base viz is all_hidden; reveal
  // only to the new current_player.
  viz::reset_to_base(state, "drawn_card",
                     LoveLetterState<NPlayers>::schema(), {});
  viz::reveal_slot_to(state, "drawn_card", {}, next);
}

}  // namespace

template <int NPlayers>
bool LoveLetterRules<NPlayers>::validate_action(const IGameState& state, ActionId action) const {
  const auto& s = checked_cast<LoveLetterState<NPlayers>>(state);
  if (s.data.terminal) return false;
  auto legal = legal_actions(state);
  return std::find(legal.begin(), legal.end(), action) != legal.end();
}

template <int NPlayers>
std::vector<ActionId> LoveLetterRules<NPlayers>::legal_actions(const IGameState& state) const {
  const auto& s = checked_cast<LoveLetterState<NPlayers>>(state);
  const auto& d = s.data;
  if (d.terminal) return {};

  const int me = d.current_player;
  const std::int8_t h = d.hand[me];
  const std::int8_t dc = d.drawn_card;

  bool must_countess = false;
  if ((h == kCountess && (dc == kKing || dc == kPrince)) ||
      (dc == kCountess && (h == kKing || h == kPrince))) {
    must_countess = true;
  }

  if (must_countess) {
    return {kCountessAction};
  }

  std::vector<ActionId> actions;
  actions.reserve(16);

  add_actions_for_card(d, me, h, actions);

  if (dc != h) {
    add_actions_for_card(d, me, dc, actions);
  }

  return actions;
}

template <int NPlayers>
void LoveLetterRules<NPlayers>::do_action_fast_impl(IGameState& state, ActionId action,
                                                    std::mt19937_64& rng) const {
  auto& s = checked_cast<LoveLetterState<NPlayers>>(state);
  auto& d = s.data;

  d.ply++;

  const DecodedAction act = decode_action(action);
  const int me = d.current_player;

  std::int8_t played_card = act.card;
  bool played_from_hand = (d.hand[me] == played_card);
  if (played_from_hand) {
    d.hand[me] = d.drawn_card;
    // hand[me] contents changed (now holds the drawn card). Drop any
    // prior in-round reveals on hand[me] (King swaps, Priest peeks)
    // since they were tied to the OLD cid.
    viz::reset_to_base(state, "hand",
                       LoveLetterState<NPlayers>::schema(), {me});
  }
  d.drawn_card = 0;
  // drawn_card consumed → reset to base (all_hidden).
  viz::reset_to_base(state, "drawn_card",
                     LoveLetterState<NPlayers>::schema(), {});
  d.discard_count[static_cast<size_t>(me)]
                 [static_cast<size_t>(played_card)]++;

  // played_from_hand → hand[me] contents changed above (viz reset to
  // owner-only base via viz::reset_to_base on line 322). Anyone who had
  // viz=1 on the OLD cid no longer sees the new card → reduces correctly
  // to "exposed = no" without a parallel cache flag.

  int target = act.target;
  bool target_is_self = (target == me);
  bool target_protected = (target >= 0 && target < NPlayers &&
                           (d.protected_flags[target] || !d.alive[target]));
  bool no_effect = target_is_self && played_card != kPrince && played_card != kHandmaid &&
                   played_card != kCountess && played_card != kPrincess;

  if (!no_effect && !target_protected) {
    switch (played_card) {
      case kGuard:
        if (target >= 0 && target < NPlayers && d.alive[target] &&
            d.hand[target] == act.guess) {
          eliminate_player(state, d, target);
        }
        break;

      case kPriest:
        if (target >= 0 && target < NPlayers && d.alive[target]) {
          // Targeted private reveal: only actor learns target's hand cid.
          viz::reveal_slot_to(state, "hand", {target}, me);
        }
        break;

      case kBaron:
        if (target >= 0 && target < NPlayers && d.alive[target]) {
          int my_card = d.hand[me];
          int their_card = d.hand[target];
          if (my_card < their_card) {
            // Actor (me) loses. Loser's hand becomes fully public; loser
            // also learns winner's hand (last-second info before
            // elimination — keeps selfplay encoder's pre-death view
            // accurate).
            viz::reveal_slot(state, "hand", {me});
            viz::reveal_slot_to(state, "hand", {target}, me);
            eliminate_player(state, d, me);
          } else if (their_card < my_card) {
            // Target loses.
            viz::reveal_slot(state, "hand", {target});
            viz::reveal_slot_to(state, "hand", {me}, target);
            eliminate_player(state, d, target);
          } else {
            // Tie: both learn the other's hand privately. Third
            // perspective sees only the public "tie" event and remains
            // viz=0 on both hands.
            viz::reveal_slot_to(state, "hand", {me}, target);
            viz::reveal_slot_to(state, "hand", {target}, me);
          }
        }
        break;

      case kHandmaid:
        d.protected_flags[me] = 1;
        break;

      case kPrince:
        if (target >= 0 && target < NPlayers && d.alive[target]) {
          std::int8_t discarded = d.hand[target];
          d.discard_count[static_cast<size_t>(target)]
                         [static_cast<size_t>(discarded)]++;
          if (discarded == kPrincess) {
            d.hand[target] = 0;
            eliminate_player(state, d, target);
          } else {
            // hand[target] contents are about to change; drop prior
            // reveals before redraw.
            viz::reset_to_base(state, "hand",
                               LoveLetterState<NPlayers>::schema(), {target});
            if (deck_total<NPlayers>(d) > 0) {
              d.hand[target] = draw_from_deck_local<NPlayers>(d, rng);
            } else {
              d.hand[target] = d.set_aside_card;
              d.set_aside_card = 0;
            }
          }
        }
        break;

      case kKing:
        if (target >= 0 && target < NPlayers && d.alive[target]) {
          std::swap(d.hand[me], d.hand[target]);
          // viz follows content: each owner now sees their NEW card
          // (the other player's old hand). The old viewer's reveal row
          // travels with the cid, so observers who already saw a cid
          // continue seeing it after the swap.
          viz::swap_slot_owned(state, "hand", me, target);
        }
        break;

      case kCountess:
        break;

      case kPrincess:
        eliminate_player(state, d, me);
        break;

      default:
        break;
    }
  }

  advance_turn(state, d, rng);
}

template <int NPlayers>
UndoToken LoveLetterRules<NPlayers>::do_action_deterministic_impl(
    IGameState& state, ActionId action) const {
  auto& s = checked_cast<LoveLetterState<NPlayers>>(state);
  UndoToken token{};
  token.undo_depth = static_cast<std::uint32_t>(s.undo_stack.size());
  LoveLetterUndoFrame<NPlayers> frame;
  frame.data = s.data;
  frame.viz_snapshot = s.viz_;
  s.undo_stack.push_back(std::move(frame));

  // Love Letter does not currently register a tail solver. This path exists
  // only to keep the generic deterministic/undo contract locally correct.
  std::mt19937_64 unused_rng(0);
  do_action_fast_impl(state, action, unused_rng);
  return token;
}

template <int NPlayers>
void LoveLetterRules<NPlayers>::undo_action_impl(
    IGameState& state, const UndoToken& /*token*/) const {
  auto& s = checked_cast<LoveLetterState<NPlayers>>(state);
  if (s.undo_stack.empty()) return;
  auto frame = std::move(s.undo_stack.back());
  s.undo_stack.pop_back();
  s.data = std::move(frame.data);
  s.viz_ = std::move(frame.viz_snapshot);
}

template <int NPlayers>
void LoveLetterRules<NPlayers>::reveal_starting_draw(
    IGameState& state, int starting_player) {
  viz::reveal_slot_to(state, "drawn_card", {}, starting_player);
}

template <int NPlayers>
void LoveLetterRules<NPlayers>::reveal_starting_draw_to(
    IGameState& state, int viewer) {
  viz::reveal_slot_to(state, "drawn_card", {}, viewer);
}

template class LoveLetterRules<2>;
template class LoveLetterRules<3>;
template class LoveLetterRules<4>;

}  // namespace board_ai::loveletter
