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
template <std::size_t N>
static int sum_of(const std::array<int, N>& a) {
  return std::accumulate(a.begin(), a.end(), 0);
}
// Highest index holding a non-zero count, -1 if none.
template <std::size_t N>
static int last_nonzero(const std::array<int, N>& a) {
  for (int i = static_cast<int>(N) - 1; i >= 0; --i)
    if (a[i] != 0) return i;
  return -1;
}
static int hand_total(const JaipurData& d, int p) { return sum_of(d.hand_count[p]); }

// ---- Terminal / scoring ----
// Full score = public rupees (goods + camel token) + private bonus total.
// Tie-break per rulebook: richer total → most bonus tokens → most goods tokens.
static int decide_winner(const JaipurData& d) {
  auto score = [&](int p) { return d.rupees[p] + d.bonus_total[p]; };
  if (score(0) != score(1)) return score(0) > score(1) ? 0 : 1;
  const int b0 = sum_of(d.bonus_count[0]), b1 = sum_of(d.bonus_count[1]);
  if (b0 != b1) return b0 > b1 ? 0 : 1;
  const int g0 = d.goods_tokens_taken[0], g1 = d.goods_tokens_taken[1];
  if (g0 != g1) return g0 > g1 ? 0 : 1;
  return -1;
}

static void check_terminal(JaipurData& d) {
  if (d.terminal) return;
  // Rulebook: the round ends when the market cannot be refilled, not when the
  // deck merely runs out. Refills happen after every take, so a short market
  // with an empty deck means a refill just failed; a full market with an
  // empty deck keeps the round going (sells and trades need no refill).
  const bool deck_empty = (d.deck_pos >= kTotalDeckSize);
  const bool refill_failed = deck_empty && sum_of(d.market_count) < kMarketSize;
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
  int total = sum_of(d.market_count);
  while (total < kMarketSize) {
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

static void apply_sell(JaipurData& d, int g, int k, std::mt19937_64& rng) {
  const int p = d.current_player;
  d.hand_count[p][g] -= k;
  d.discard_count[g] += k;  // sold cards go face up on the discard pile
  for (int i = 0; i < k; ++i) {
    if (!d.value_tokens[g].empty()) {
      d.rupees[p] += d.value_tokens[g].back();
      d.value_tokens[g].pop_back();
      ++d.goods_tokens_taken[p];
    }
  }
  const int tier = (k >= 5) ? 2 : (k == 4) ? 1 : (k == 3) ? 0 : -1;
  if (tier >= 0 && d.bonus_pile_size[tier] > 0) draw_bonus_token(d, p, tier, rng);
}

// ================================================================
//  Take one good / take all camels
// ================================================================
static void apply_take_good(JaipurData& d, int g, std::mt19937_64& rng) {
  const int p = d.current_player;
  d.market_count[g]--;
  d.hand_count[p][g]++;
  refill_market(d, rng);
}
static void apply_take_camels(JaipurData& d, std::mt19937_64& rng) {
  const int p = d.current_player;
  d.camel_count[p] += d.market_count[kCamel];
  d.market_count[kCamel] = 0;
  refill_market(d, rng);
}

// ================================================================
//  Trades (factored: ascending take atoms, then ascending give atoms)
// ================================================================
namespace {
struct Prefix {
  std::array<int, kGoodTypes> take{};
  std::array<int, kGoodTypes + 1> give{};
};
Prefix prefix_of(const JaipurData& d) { return {d.trade_take, d.trade_give}; }
}  // namespace

// Most camels a trade can pay with: the herd, and the hand limit (each camel
// given leaves one more good in the hand after the swap).
static int camel_cap(const JaipurData& d, int p) {
  return std::max(0, std::min(d.camel_count[p], kHandSize - hand_total(d, p)));
}

// Cards p can still give from the give types after `after` (hand goods of a
// type not taken, then camels).
static int give_capacity(const JaipurData& d, int p,
                         const std::array<int, kGoodTypes>& take, int after) {
  int cap = 0;
  for (int x = after + 1; x < kGoodTypes; ++x)
    if (take[x] == 0) cap += d.hand_count[p][x];
  if (after < kCamel) cap += camel_cap(d, p);
  return cap;
}

// A take set can be paid for: 2..5 goods, enough cards of other types.
static bool take_set_payable(const JaipurData& d, int p,
                             const std::array<int, kGoodTypes>& take) {
  const int total = sum_of(take);
  return total >= 2 && total <= kMaxTrade && give_capacity(d, p, take, -1) >= total;
}

// Taking k of type g next keeps the trade completable. A single taken card
// needs one more take type above g: adding takes only makes paying harder,
// so one extra card is the cheapest completion to probe.
static bool take_atom_ok(const JaipurData& d, const Prefix& pre, int g, int k) {
  if (sum_of(pre.give) > 0 || g <= last_nonzero(pre.take)) return false;
  if (k < 1 || k > d.market_count[g]) return false;
  const int total = sum_of(pre.take) + k;
  if (total > kMaxTrade) return false;
  const int p = d.current_player;
  auto take = pre.take;
  take[g] += k;
  if (total >= 2) return take_set_payable(d, p, take);
  for (int g2 = g + 1; g2 < kGoodTypes; ++g2) {
    if (d.market_count[g2] == 0) continue;
    auto more = take;
    ++more[g2];
    if (take_set_payable(d, p, more)) return true;
  }
  return false;
}

// Giving k of type x next (x = kCamel: camels) leaves a payable remainder.
static bool give_atom_ok(const JaipurData& d, const Prefix& pre, int x, int k) {
  const int taken = sum_of(pre.take);
  if (taken < 2 || x <= last_nonzero(pre.give)) return false;
  const int owed = taken - sum_of(pre.give);
  if (k < 1 || k > owed) return false;
  const int p = d.current_player;
  if (x < kGoodTypes) {
    if (pre.take[x] > 0 || k > d.hand_count[p][x]) return false;
  } else if (k > camel_cap(d, p)) {
    return false;
  }
  const int rest = owed - k;
  return rest == 0 || give_capacity(d, p, pre.take, x) >= rest;
}

static void push_trade_atoms(const JaipurData& d, const Prefix& pre,
                             std::vector<ActionId>& out) {
  for (int g = 0; g < kGoodTypes; ++g)
    for (int k = 1; k <= kMaxTrade; ++k)
      if (take_atom_ok(d, pre, g, k)) out.push_back(trade_take_action(g, k));
  for (int x = 0; x <= kCamel; ++x)
    for (int k = 1; k <= kMaxTrade; ++k)
      if (give_atom_ok(d, pre, x, k)) out.push_back(trade_give_action(x, k));
}

static void apply_trade(JaipurData& d) {
  const int p = d.current_player;
  for (int g = 0; g < kGoodTypes; ++g) {
    d.market_count[g] += d.trade_give[g] - d.trade_take[g];
    d.hand_count[p][g] += d.trade_take[g] - d.trade_give[g];
  }
  d.market_count[kCamel] += d.trade_give[kCamel];
  d.camel_count[p] -= d.trade_give[kCamel];
  d.trade_take.fill(0);
  d.trade_give.fill(0);
  // No refill: a trade keeps the market at five cards (give == take).
}

bool jaipur_trade_idle(const JaipurData& d) { return sum_of(d.trade_take) == 0; }

std::vector<JaipurTrade> jaipur_trades(const JaipurData& d) {
  std::vector<JaipurTrade> out;
  std::vector<ActionId> atoms;
  // Depth-first over the trade atoms: every legal atom keeps the trade
  // completable, so each branch ends in a complete trade.
  auto walk = [&](auto&& self, const Prefix& pre) -> void {
    const int taken = sum_of(pre.take);
    if (taken >= 2 && sum_of(pre.give) == taken) {
      out.push_back({pre.take, pre.give});
      return;
    }
    atoms.clear();
    push_trade_atoms(d, pre, atoms);
    const std::vector<ActionId> next = atoms;
    for (ActionId a : next) {
      Prefix child = pre;
      if (is_trade_take_action(a)) child.take[trade_take_type(a)] += trade_take_count(a);
      else child.give[trade_give_type(a)] += trade_give_count(a);
      self(self, child);
    }
  };
  walk(walk, prefix_of(d));
  return out;
}

ActionId jaipur_trade_next_atom(const JaipurData& d, const JaipurTrade& t) {
  const Prefix pre = prefix_of(d);
  const int last_take = last_nonzero(pre.take);
  for (int g = 0; g <= last_take; ++g)
    if (t.take[g] != pre.take[g])
      throw std::logic_error("jaipur_trade_next_atom: trade does not extend the prefix");
  for (int x = 0; x <= kCamel; ++x)
    if (pre.give[x] != 0 && t.give[x] != pre.give[x])
      throw std::logic_error("jaipur_trade_next_atom: trade does not extend the prefix");
  if (sum_of(pre.give) == 0)
    for (int g = last_take + 1; g < kGoodTypes; ++g)
      if (t.take[g] > 0) return trade_take_action(g, t.take[g]);
  for (int x = last_nonzero(pre.give) + 1; x <= kCamel; ++x)
    if (t.give[x] > 0) return trade_give_action(x, t.give[x]);
  throw std::logic_error("jaipur_trade_next_atom: trade is already complete");
}

// ================================================================
//  validate_action / legal_actions (same predicates)
// ================================================================
bool JaipurRules::validate_action(const IGameState& state, ActionId action) const {
  const auto& d = cast_data(state);
  if (d.terminal || action < 0 || action >= kActionSpace) return false;
  const int p = d.current_player;
  const Prefix pre = prefix_of(d);
  if (is_trade_take_action(action))
    return take_atom_ok(d, pre, trade_take_type(action), trade_take_count(action));
  if (is_trade_give_action(action))
    return give_atom_ok(d, pre, trade_give_type(action), trade_give_count(action));
  if (!jaipur_trade_idle(d)) return false;
  if (is_take_action(action))
    return d.market_count[action - kActTakeBase] > 0 && hand_total(d, p) < kHandSize;
  if (action == kActTakeCamels) return d.market_count[kCamel] > 0;
  const int g = sell_type(action), k = sell_count(action);
  return k >= min_sell_count(g) && k <= d.hand_count[p][g];
}

std::vector<ActionId> JaipurRules::legal_actions(const IGameState& state) const {
  const auto& d = cast_data(state);
  std::vector<ActionId> r;
  if (d.terminal) return r;
  const int p = d.current_player;
  if (jaipur_trade_idle(d)) {
    if (hand_total(d, p) < kHandSize)
      for (int g = 0; g < kGoodTypes; ++g)
        if (d.market_count[g] > 0) r.push_back(kActTakeBase + g);
    if (d.market_count[kCamel] > 0) r.push_back(kActTakeCamels);
    for (int g = 0; g < kGoodTypes; ++g)
      for (int k = min_sell_count(g); k <= d.hand_count[p][g]; ++k)
        r.push_back(sell_action(g, k));
  }
  push_trade_atoms(d, prefix_of(d), r);
  return r;
}

// ================================================================
//  do_action_fast_impl
// ================================================================
void JaipurRules::do_action_fast_impl(IGameState& state, ActionId action,
                                       std::mt19937_64& rng) const {
  auto& d = cast_data_mut(state);
  if (d.terminal) return;
  if (is_trade_take_action(action)) {
    d.trade_take[trade_take_type(action)] += trade_take_count(action);
    return;
  }
  if (is_trade_give_action(action)) {
    d.trade_give[trade_give_type(action)] += trade_give_count(action);
    if (sum_of(d.trade_give) < sum_of(d.trade_take)) return;
    apply_trade(d);
  } else if (is_take_action(action)) {
    apply_take_good(d, action - kActTakeBase, rng);
  } else if (action == kActTakeCamels) {
    apply_take_camels(d, rng);
  } else if (is_sell_action(action)) {
    apply_sell(d, sell_type(action), sell_count(action), rng);
  } else {
    throw std::invalid_argument("jaipur: action id out of range");
  }
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
