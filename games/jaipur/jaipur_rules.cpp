#include "jaipur_rules.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace board_ai::jaipur {

static const JaipurData& cast_data(const IGameState& state) {
  return static_cast<const JaipurState&>(state).data;
}
static JaipurData& cast_data_mut(IGameState& state) {
  return static_cast<JaipurState&>(state).data;
}

// ---- Count helpers ----
static int hand_total(const JaipurData& d, int p) {
  return std::accumulate(d.hand_count[p].begin(), d.hand_count[p].end(), 0);
}
static int market_goods_total(const JaipurData& d) {
  int n = 0;
  for (int g = 0; g < kGoodTypes; ++g) n += d.market_count[g];
  return n;
}
static int take_count_total(const JaipurData& d) {
  return std::accumulate(d.exchange_take_count.begin(),
                         d.exchange_take_count.end(), 0);
}
static int give_count_total(const JaipurData& d) {
  return std::accumulate(d.exchange_give_count.begin(),
                         d.exchange_give_count.end(), 0);
}

// ---- Terminal / scoring ----
// Full score = public rupees (goods + camel token) + private bonus total.
// Tie-break per rulebook: richer total → most bonus tokens → most goods tokens.
static int count_bonus_tokens_taken(const JaipurData& d, int p) {
  return d.bonus_count[p][0] + d.bonus_count[p][1] + d.bonus_count[p][2];
}
static int count_goods_tokens_taken(const JaipurData& d, int p) {
  return d.goods_tokens_taken[p];
}
static int decide_winner(const JaipurData& d) {
  auto score = [&](int p) { return d.rupees[p] + d.bonus_total[p]; };
  if (score(0) != score(1)) return score(0) > score(1) ? 0 : 1;
  int b0 = count_bonus_tokens_taken(d, 0), b1 = count_bonus_tokens_taken(d, 1);
  if (b0 != b1) return b0 > b1 ? 0 : 1;
  int g0 = count_goods_tokens_taken(d, 0), g1 = count_goods_tokens_taken(d, 1);
  if (g0 != g1) return g0 > g1 ? 0 : 1;
  return -1;
}

static void check_terminal(JaipurData& d) {
  if (d.terminal) return;
  // Rulebook: the round ends when the market cannot be refilled, not when the
  // deck merely runs out. Refills happen after every take, so a short market
  // with an empty deck means a refill just failed; a full market with an
  // empty deck keeps the round going (sells and exchanges need no refill).
  const bool deck_empty = (d.deck_pos >= kTotalDeckSize);
  const bool refill_failed =
      deck_empty && market_goods_total(d) + d.market_count[6] < 5;
  int exhausted = 0;
  for (int g = 0; g < kGoodTypes; ++g)
    if (d.value_tokens[g].empty()) ++exhausted;
  d.piles_exhausted = exhausted;
  // Only piles_exhausted/refill_failed are real terminals (rulebook).
  // The ply_cap (d.plies >= kMaxPlies) is NOT a real terminal — it routes
  // through the framework adjudicator path (jaipur_register installs one
  // that always returns -1) which assigns z=-1 to BOTH players. That's a
  // hard penalty for stalling: a network that drives the game to truncation
  // sees -1 just like a real loss, so "wait for the other side to commit"
  // is no longer a free equilibrium. plies still lives in schema (kPlies)
  // so MCTS DAG distinguishes plies and won't merge across the timeline.
  if (exhausted >= 3 || refill_failed) {
    d.terminal = true;
    if (d.camel_bonus_token) {
      if (d.camel_count[0] > d.camel_count[1]) d.rupees[0] += 5;
      else if (d.camel_count[1] > d.camel_count[0]) d.rupees[1] += 5;
      d.camel_bonus_token = false;
    }
    d.winner = decide_winner(d);
  }
}

// Refill the market to 5 cards from the draw pile. Each card is a weighted
// draw from deck_count with the caller's rng (draw_deck_card, jaipur_state):
// the pile has no stored order, so nothing can know the next card early.
static void refill_market(JaipurData& d, std::mt19937_64& rng) {
  int total = market_goods_total(d) + d.market_count[6];
  while (total < 5) {
    const int c = draw_deck_card(d, rng);
    if (c < 0) break;
    d.market_count[c]++;
    ++total;
  }
}

static void next_turn(JaipurData& d) {
  d.plies++;
  d.current_player = (d.current_player + 1) % kPlayers;
  check_terminal(d);
}

// ================================================================
//  Sell
// ================================================================
static int min_sell_count(int g) { return (g <= 2) ? 2 : 1; }  // dia/gold/silv ≥2

// Draw one face-down bonus token of `tier` for player p: weighted by the
// values left in the pile, caller's rng. Same no-stored-order rule as the
// draw pile; throws on a masked / stale pile instead of inventing a value.
static void draw_bonus_token(JaipurData& d, int p, int tier,
                             std::mt19937_64& rng) {
  int total = 0;
  for (int c : d.bonus_pile[tier]) {
    if (c < 0) {
      throw std::runtime_error(
          "jaipur: draw from a masked bonus pile (randomize_unseen was not "
          "applied to this state)");
    }
    total += c;
  }
  if (total != d.bonus_pile_size[tier] || total <= 0) {
    throw std::runtime_error(
        "jaipur: bonus_pile disagrees with the public pile size; hidden "
        "slots are stale — randomize_unseen was not applied");
  }
  int pick = static_cast<int>(rng() % static_cast<std::uint64_t>(total));
  for (int v = 0; v < kBonusValues; ++v) {
    if (pick < d.bonus_pile[tier][v]) {
      --d.bonus_pile[tier][v];
      --d.bonus_pile_size[tier];
      ++d.bonus_taken[p][tier][v];
      d.bonus_total[p] += kBonusValue[tier][v];
      ++d.bonus_count[p][tier];
      return;
    }
    pick -= d.bonus_pile[tier][v];
  }
}

static void apply_sell(JaipurData& d, int g, std::mt19937_64& rng) {
  int p = d.current_player;
  int count = d.hand_count[p][g];
  d.hand_count[p][g] = 0;
  d.discard_count[g] += count;  // sold cards go face up on the discard pile
  for (int i = 0; i < count; ++i) {
    if (!d.value_tokens[g].empty()) {
      d.rupees[p] += d.value_tokens[g].back();
      d.value_tokens[g].pop_back();
      ++d.goods_tokens_taken[p];
    }
  }
  const int tier = (count >= 5) ? 2 : (count == 4) ? 1 : (count == 3) ? 0 : -1;
  if (tier >= 0 && d.bonus_pile_size[tier] > 0) draw_bonus_token(d, p, tier, rng);
}

// ================================================================
//  Take one good / take all camels
// ================================================================
static void apply_take_good(JaipurData& d, int g, std::mt19937_64& rng) {
  int p = d.current_player;
  d.market_count[g]--;
  d.hand_count[p][g]++;
  refill_market(d, rng);
}
static void apply_take_camels(JaipurData& d, std::mt19937_64& rng) {
  int p = d.current_player;
  d.camel_count[p] += d.market_count[6];
  d.market_count[6] = 0;
  refill_market(d, rng);
}

// ================================================================
//  Exchange feasibility & bounds (count form)
// ================================================================
// Goods types present in the take selection.
static int take_types_bitset(const JaipurData& d) {
  int b = 0;
  for (int g = 0; g < kGoodTypes; ++g)
    if (d.exchange_take_count[g] > 0) b |= (1 << g);
  return b;
}
// Hand goods whose type is NOT in the take selection — eligible to surrender.
static int available_non_overlap(const JaipurData& d, int take_bits) {
  int p = d.current_player;
  int avail = 0;
  for (int g = 0; g < kGoodTypes; ++g)
    if (!(take_bits & (1 << g))) avail += d.hand_count[p][g];
  return avail;
}
// Is the current take selection feasible to complete? Need ≥2 taken, enough
// non-overlapping hand goods + camels to cover, and hand limit at min camels.
static bool take_feasible(const JaipurData& d) {
  int p = d.current_player;
  int tc = take_count_total(d);
  if (tc < 2) return false;
  int take_bits = take_types_bitset(d);
  int avail = available_non_overlap(d, take_bits);
  if (avail + d.camel_count[p] < tc) return false;
  int min_camels = std::max(0, tc - avail);
  if (hand_total(d, p) + min_camels > kHandSize) return false;
  return true;
}
// Resource-only feasibility check (no rulebook tc≥2 constraint). Returns
// true iff the GIVE side can be satisfied for a hypothetical state — used
// as a building block by take_g_safe. User-derived (审稿): use as many
// camels as possible (minimum hand-good demand), then check hand covers
// the residual.
static bool give_resources_ok(int new_tc, int new_take_bits,
                               const JaipurData& d) {
  int p = d.current_player;
  if (new_tc > kMaxExchange) return false;
  int avail = 0;
  for (int gg = 0; gg < kGoodTypes; ++gg)
    if (!(new_take_bits & (1 << gg))) avail += d.hand_count[p][gg];
  int K_max = std::min({d.camel_count[p],
                        kHandSize - hand_total(d, p),
                        new_tc});
  if (K_max < 0) K_max = 0;
  int need_from_hand = new_tc - K_max;
  if (need_from_hand < 0) need_from_hand = 0;
  return need_from_hand <= avail;
}

// "Is it safe to take one more of type g?" — guarantees that after taking
// g, the player can complete the exchange (reach a feasible mkt_done state
// without escape). Two conditions:
//   1. Resources after the take cover the give side (give_resources_ok).
//   2. The resulting tc reaches ≥2 — either directly (new_tc ≥ 2) or via
//      at least one further safe take (recursive 1-level: only needed
//      when new_tc == 1, since tc ≥ 2 + resources_ok ⇒ take_feasible).
static bool take_g_safe(const JaipurData& d, int g) {
  if (d.exchange_take_count[g] + 1 > d.market_count[g]) return false;
  int new_tc = take_count_total(d) + 1;
  int new_take_bits = take_types_bitset(d) | (1 << g);
  if (!give_resources_ok(new_tc, new_take_bits, d)) return false;
  if (new_tc >= 2) return true;  // resources + tc≥2 ⇒ commit-feasible
  // new_tc == 1: must still have at least one further safe take so we can
  // reach tc=2. Probe each candidate g' in the post-take state.
  JaipurData trial = d;
  trial.exchange_take_count[g] += 1;
  for (int gp = 0; gp < kGoodTypes; ++gp) {
    if (trial.exchange_take_count[gp] >= trial.market_count[gp]) continue;
    int next_tc = new_tc + 1;  // = 2
    int next_take_bits = new_take_bits | (1 << gp);
    if (give_resources_ok(next_tc, next_take_bits, trial)) return true;
  }
  return false;
}

static int compute_min_camels(const JaipurData& d) {
  int tc = take_count_total(d);
  int avail = available_non_overlap(d, take_types_bitset(d));
  return std::min(std::max(0, tc - avail), d.camel_count[d.current_player]);
}
static int compute_max_camels(const JaipurData& d) {
  int p = d.current_player;
  int tc = take_count_total(d);
  int by_count = std::min(tc, d.camel_count[p]);
  int by_hand = kHandSize - hand_total(d, p);
  return std::min({by_count, by_hand, tc});
}

static void reset_exchange_state(JaipurData& d) {
  d.exchange_phase = kExchPhaseIdle;
  d.exchange_take_count.fill(0);
  d.exchange_give_count.fill(0);
  d.exchange_camel_count = 0;
  d.exchange_max_camels = 0;
  d.exchange_min_camels = 0;
  d.exchange_total = 0;
}

// Apply the fully-built exchange (count form).
static void apply_exchange(JaipurData& d) {
  int p = d.current_player;
  int camels_give = d.exchange_camel_count;
  // Take selected market goods into hand.
  for (int g = 0; g < kGoodTypes; ++g) {
    int n = d.exchange_take_count[g];
    d.market_count[g] -= n;
    d.hand_count[p][g] += n;
  }
  // Give selected hand goods to market.
  for (int g = 0; g < kGoodTypes; ++g) {
    int n = d.exchange_give_count[g];
    d.hand_count[p][g] -= n;
    d.market_count[g] += n;
  }
  // Give camels to market.
  d.market_count[6] += camels_give;
  d.camel_count[p] -= camels_give;
  // No refill — exchange keeps the market at the same size (give == take).
}

// ================================================================
//  validate_action (phase-aware)
// ================================================================
bool JaipurRules::validate_action(const IGameState& state, ActionId action) const {
  const auto& d = cast_data(state);
  if (d.terminal) return false;
  int ph = d.exchange_phase;

  if (ph == kExchPhaseIdle) {
    if (action >= kActSellBase && action < kActSellBase + kGoodTypes) {
      int g = action - kActSellBase;
      return d.hand_count[d.current_player][g] >= min_sell_count(g);
    }
    if (action >= kActTakeBase && action < kActTakeBase + kGoodTypes) {
      int g = action - kActTakeBase;
      return d.market_count[g] > 0 && hand_total(d, d.current_player) < kHandSize;
    }
    if (action == kActTakeCamels) return d.market_count[6] > 0;
    if (action == kActEnterExchange) {
      // Legal iff phase=1 will have ≥1 legal take. Probe by trying each
      // candidate first-take g with take_g_safe (the same forward-progress
      // condition used in phase=1 legal_actions). If any g works as a
      // starting move, the whole phase chain is guaranteed deadlock-free.
      JaipurData trial = d;
      trial.exchange_take_count.fill(0);
      for (int g = 0; g < kGoodTypes; ++g) {
        if (trial.market_count[g] > 0 && take_g_safe(trial, g)) return true;
      }
      return false;
    }
    return false;
  }

  if (ph == kExchPhaseMarket) {
    if (action >= kActExchMktBase && action < kActExchMktBase + kGoodTypes) {
      int g = action - kActExchMktBase;
      if (d.exchange_take_count[g] >= d.market_count[g]) return false;
      if (take_count_total(d) >= kMaxExchange) return false;
      return take_g_safe(d, g);  // forward-progress guard
    }
    if (action == kActExchMktDone) return take_feasible(d);
    return false;
  }

  if (ph == kExchPhaseCamel) {
    if (action >= kActExchCamelBase && action < kActExchCamelBase + 8) {
      int k = action - kActExchCamelBase;
      return k >= d.exchange_min_camels && k <= d.exchange_max_camels;
    }
    return false;
  }

  if (ph == kExchPhaseHand) {
    if (action >= kActExchHandBase && action < kActExchHandBase + kGoodTypes) {
      int g = action - kActExchHandBase;
      int take_bits = take_types_bitset(d);
      if (take_bits & (1 << g)) return false;  // can't give a type you're taking
      if (d.exchange_give_count[g] >= d.hand_count[d.current_player][g]) return false;
      return true;
    }
    return false;
  }
  return false;
}

// ================================================================
//  legal_actions (phase-aware) — same graph as do_action
// ================================================================
std::vector<ActionId> JaipurRules::legal_actions(const IGameState& state) const {
  const auto& d = cast_data(state);
  std::vector<ActionId> r;
  if (d.terminal) return r;
  int ph = d.exchange_phase;

  if (ph == kExchPhaseIdle) {
    int p = d.current_player;
    for (int g = 0; g < kGoodTypes; ++g)
      if (d.hand_count[p][g] >= min_sell_count(g)) r.push_back(kActSellBase + g);
    for (int g = 0; g < kGoodTypes; ++g)
      if (d.market_count[g] > 0 && hand_total(d, p) < kHandSize)
        r.push_back(kActTakeBase + g);
    if (d.market_count[6] > 0) r.push_back(kActTakeCamels);
    if (validate_action(state, kActEnterExchange)) r.push_back(kActEnterExchange);
    return r;
  }

  if (ph == kExchPhaseMarket) {
    // A take_g is legal ONLY if the resulting selection still has a path
    // forward (i.e. some camel/hand combo can satisfy the give side once
    // we mkt_done). take_g_safe is the user-derived sufficient check:
    // pretend we take g, use max camels, see if hand can cover residual.
    // This guarantees legal_actions never empties out mid-phase — no
    // deadlock, no escape needed.
    for (int g = 0; g < kGoodTypes; ++g) {
      if (d.exchange_take_count[g] < d.market_count[g] &&
          take_count_total(d) < kMaxExchange &&
          take_g_safe(d, g))
        r.push_back(kActExchMktBase + g);
    }
    // mkt_done strictly iff feasible. Enter_exchange's pre-check + take_g
    // gating jointly ensure that some take→take_g_safe path to feasible
    // always exists, so this is safe (legal won't empty out).
    if (take_feasible(d)) r.push_back(kActExchMktDone);
    return r;
  }

  if (ph == kExchPhaseCamel) {
    for (int k = d.exchange_min_camels; k <= d.exchange_max_camels; ++k)
      r.push_back(kActExchCamelBase + k);
    return r;
  }

  if (ph == kExchPhaseHand) {
    int take_bits = take_types_bitset(d);
    int p = d.current_player;
    for (int g = 0; g < kGoodTypes; ++g) {
      if ((take_bits & (1 << g))) continue;
      if (d.exchange_give_count[g] < d.hand_count[p][g])
        r.push_back(kActExchHandBase + g);
    }
    return r;
  }
  return r;
}

// ================================================================
//  do_action_fast_impl (phase-aware)
// ================================================================
void JaipurRules::do_action_fast_impl(IGameState& state, ActionId action,
                                       std::mt19937_64& rng) const {
  auto& d = cast_data_mut(state);
  if (d.terminal) return;
  int ph = d.exchange_phase;

  if (ph == kExchPhaseIdle) {
    if (action >= kActSellBase && action < kActSellBase + kGoodTypes) {
      apply_sell(d, action - kActSellBase, rng);
    } else if (action >= kActTakeBase && action < kActTakeBase + kGoodTypes) {
      apply_take_good(d, action - kActTakeBase, rng);
    } else if (action == kActTakeCamels) {
      apply_take_camels(d, rng);
    } else if (action == kActEnterExchange) {
      reset_exchange_state(d);
      d.exchange_phase = kExchPhaseMarket;
      return;  // no turn advance
    }
  } else if (ph == kExchPhaseMarket) {
    if (action >= kActExchMktBase && action < kActExchMktBase + kGoodTypes) {
      d.exchange_take_count[action - kActExchMktBase]++;
      return;
    }
    if (action == kActExchMktDone) {
      // legal_actions/validate gate mkt_done strictly by take_feasible —
      // the take_g_safe gating on take_g ensures phase=1 always has a path
      // to a feasible selection, so an infeasible mkt_done is a contract
      // violation. No escape needed; treat as no-op rather than corrupting.
      if (!take_feasible(d)) return;
      d.exchange_total = take_count_total(d);
      d.exchange_max_camels = compute_max_camels(d);
      d.exchange_min_camels = compute_min_camels(d);
      d.exchange_camel_count = d.exchange_min_camels;  // start at lower bound
      d.exchange_phase = kExchPhaseCamel;
      // If camels already cover the full take, commit now (no hand goods needed).
      if (d.exchange_camel_count == d.exchange_total) {
        d.exchange_give_count.fill(0);
        apply_exchange(d);
        reset_exchange_state(d);
        next_turn(d);
      }
      return;
    }
  } else if (ph == kExchPhaseCamel) {
    if (action >= kActExchCamelBase && action < kActExchCamelBase + 8) {
      // Picking the camel count is a single decision. Set K, then either
      // commit immediately (K covers the whole take, no hand goods needed) or
      // advance to the hand phase to pick the remaining goods to give.
      d.exchange_camel_count = action - kActExchCamelBase;
      if (d.exchange_camel_count == d.exchange_total) {
        d.exchange_give_count.fill(0);
        apply_exchange(d);
        reset_exchange_state(d);
        next_turn(d);
      } else {
        d.exchange_phase = kExchPhaseHand;
      }
      return;
    }
  } else if (ph == kExchPhaseHand) {
    if (action >= kActExchHandBase && action < kActExchHandBase + kGoodTypes) {
      d.exchange_give_count[action - kActExchHandBase]++;
      int given = give_count_total(d) + d.exchange_camel_count;
      if (given == d.exchange_total) {
        apply_exchange(d);
        reset_exchange_state(d);
        next_turn(d);
        return;
      }
      return;
    }
  }

  if (d.exchange_phase == kExchPhaseIdle)
    next_turn(d);
}

UndoToken JaipurRules::do_action_deterministic_impl(IGameState& state, ActionId action) const {
  auto& s = static_cast<JaipurState&>(state);
  s.undo_records.push_back({action, s.data});
  std::mt19937_64 dummy(0);
  do_action_fast_impl(state, action, dummy);
  return {};
}

void JaipurRules::undo_action_impl(IGameState& state, const UndoToken&) const {
  auto& s = static_cast<JaipurState&>(state);
  if (s.undo_records.empty()) return;
  s.data = s.undo_records.back().snapshot;
  s.undo_records.pop_back();
}

}  // namespace board_ai::jaipur
