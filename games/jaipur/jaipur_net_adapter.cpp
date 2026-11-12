#include "jaipur_net_adapter.h"

#include <algorithm>
#include <any>
#include <numeric>
#include <stdexcept>
#include <string>

#include "../../engine/core/masked_state.h"  // kPlaceholderInt8
#include "jaipur_belief_tracker.h"
#include "heuristic/jaipur_rule_heuristic.h"

namespace board_ai::jaipur {

namespace {
// Value of the top `k` goods tokens of type g (fewer if the pile runs out).
int top_tokens_value(const JaipurData& d, int g, int k) {
  const auto& pile = d.value_tokens[g];
  const int n = std::min(k, static_cast<int>(pile.size()));
  int v = 0;
  for (int i = 0; i < n; ++i) v += pile[pile.size() - 1 - i];
  return v;
}
int sale_tier(int k) { return k >= 5 ? 2 : k == 4 ? 1 : k == 3 ? 0 : -1; }
float clip_unit(float v) { return std::max(-1.0f, std::min(1.0f, v)); }
}  // namespace

JaipurTokenEncoder::JaipurTokenEncoder() {
  layout_.num_tokens = kNumTokens;
  layout_.token_dim = kTokenDim;
  layout_.num_token_types = kTokenTypeCount;
  layout_.num_entity_ids = 1 + kNumTokens;
  layout_.num_pointer_heads = kPointerHeadCount;
  layout_.token_types.assign(kNumTokens, kGoodToken);
  layout_.token_types[kCamelSlot] = kCamelToken;
  for (int t = 0; t < kBonusTiers; ++t) layout_.token_types[kBonusBase + t] = kBonusToken;
  layout_.token_types[kGlobalSlot] = kGlobalToken;

  layout_.action_token.assign(kActionSpace, -1);
  layout_.action_head.assign(kActionSpace, -1);
  auto bind = [this](ActionId action, int token, int head) {
    layout_.action_token[static_cast<std::size_t>(action)] = token;
    layout_.action_head[static_cast<std::size_t>(action)] = head;
  };
  for (int g = 0; g < kGoodTypes; ++g) {
    bind(kActTakeBase + g, kGoodBase + g, kTakeHead);
    for (int k = 1; k <= kMaxSell; ++k) bind(sell_action(g, k), kGoodBase + g, kSellHead + k - 1);
    for (int k = 1; k <= kMaxTrade; ++k)
      bind(trade_take_action(g, k), kGoodBase + g, kTradeTakeHead + k - 1);
  }
  bind(kActTakeCamels, kCamelSlot, kTakeHead);
  for (int x = 0; x <= kCamel; ++x)
    for (int k = 1; k <= kMaxTrade; ++k)
      bind(trade_give_action(x, k), x == kCamel ? kCamelSlot : kGoodBase + x, kTradeGiveHead + k - 1);
}

void JaipurTokenEncoder::encode_features(const IGameState& state,
    int perspective_player, const IBeliefTracker* tracker,
    std::vector<float>* out) const {
  const auto* s = dynamic_cast<const JaipurState*>(&state);
  if (!s) throw std::runtime_error("JaipurTokenEncoder: state is not a JaipurState");
  if (!out) throw std::runtime_error("JaipurTokenEncoder: out is null");
  if (perspective_player < 0 || perspective_player >= kPlayers)
    throw std::runtime_error("JaipurTokenEncoder: perspective_player out of range");
  const JaipurBeliefTracker* jt = nullptr;
  if (tracker) {
    jt = dynamic_cast<const JaipurBeliefTracker*>(tracker);
    if (!jt) throw std::runtime_error("JaipurTokenEncoder: tracker is not a JaipurBeliefTracker");
  }
  const auto& d = s->data;
  const int me = perspective_player;
  const int opp = 1 - me;

  // Public / own counts only: the opponent's hand row and bonus values are
  // placeholders in the masked state and are never read.
  const int my_hand = std::accumulate(d.hand_count[me].begin(), d.hand_count[me].end(), 0);
  int hands_total = d.deck_pos;
  for (int c : d.market_count) hands_total -= c;
  for (int p = 0; p < kPlayers; ++p) hands_total -= d.camel_count[p];
  for (int c : d.discard_count) hands_total -= c;
  const int opp_hand = hands_total - my_hand;
  std::array<int, kGoodTypes> opp_known{};
  int known_total = 0;
  for (int g = 0; g < kGoodTypes; ++g) {
    opp_known[g] = jt ? jt->known(opp, g) : 0;
    known_total += opp_known[g];
  }
  const int opp_unknown = std::max(0, opp_hand - known_total);
  double bonus_mean[kBonusTiers];
  for (int t = 0; t < kBonusTiers; ++t) bonus_mean[t] = jaipur_bonus_expected(t, d, me);
  double opp_bonus = 0.0;
  for (int t = 0; t < kBonusTiers; ++t) opp_bonus += d.bonus_count[opp][t] * bonus_mean[t];
  const int take_total = std::accumulate(d.trade_take.begin(), d.trade_take.end(), 0);
  const int give_total = std::accumulate(d.trade_give.begin(), d.trade_give.end(), 0);

  const std::size_t base = out->size();
  out->resize(base + static_cast<std::size_t>(kFeatureDim), 0.0f);
  auto token = [&](int i, int entity) {
    float* t = out->data() + base + static_cast<std::size_t>(i) * kTokenDim;
    t[0] = 1.0f;
    t[1] = static_cast<float>(entity);
    return t + 2;
  };

  for (int g = 0; g < kGoodTypes; ++g) {
    float* f = token(kGoodBase + g, 1 + kGoodBase + g);
    const int mine = d.hand_count[me][g];
    const int unseen = kGoodCardCounts[g] - d.market_count[g] - mine - d.discard_count[g];
    const int pile = static_cast<int>(d.value_tokens[g].size());
    f[0] = d.market_count[g] / 5.0f;
    f[1] = mine / 7.0f;
    f[2] = opp_known[g] / 7.0f;
    f[3] = unseen / 10.0f;
    f[4] = std::min(unseen, opp_known[g] + opp_unknown) / 7.0f;
    f[5] = d.discard_count[g] / 10.0f;
    f[6] = kGoodCardCounts[g] / 10.0f;
    f[7] = pile / 9.0f;
    for (int k = 1; k <= kMaxSell; ++k) f[7 + k] = top_tokens_value(d, g, k) / 30.0f;
    f[15] = min_sell_count(g) > 1 ? 1.0f : 0.0f;
    f[16] = d.trade_take[g] / 5.0f;
    f[17] = d.trade_give[g] / 5.0f;
    if (mine >= min_sell_count(g)) {
      const int tier = sale_tier(mine);
      const double bonus = tier >= 0 && d.bonus_pile_size[tier] > 0 ? bonus_mean[tier] : 0.0;
      f[18] = static_cast<float>((top_tokens_value(d, g, mine) + bonus) / 40.0);
    }
    f[19] = top_tokens_value(d, g, opp_known[g]) / 40.0f;
  }

  {
    float* f = token(kCamelSlot, 1 + kCamelSlot);
    const int deck_camels = kCamelCardCount - d.market_count[kCamel] - d.camel_count[0] - d.camel_count[1];
    f[0] = d.market_count[kCamel] / 5.0f;
    f[1] = d.camel_count[me] / 11.0f;
    f[2] = d.camel_count[opp] / 11.0f;
    f[3] = deck_camels / 11.0f;
    f[4] = (d.camel_count[me] - d.camel_count[opp]) / 11.0f;
    f[5] = d.camel_bonus_token ? 1.0f : 0.0f;
    f[6] = d.trade_give[kCamel] / 5.0f;
    f[7] = std::max(0, std::min(d.camel_count[me], kHandSize - my_hand)) / 7.0f;
  }

  for (int t = 0; t < kBonusTiers; ++t) {
    float* f = token(kBonusBase + t, 1 + kBonusBase + t);
    int init = 0, my_value = 0;
    for (int v = 0; v < kBonusValues; ++v) {
      init += kBonusInitCount[t][v];
      my_value += d.bonus_taken[me][t][v] * kBonusValue[t][v];
    }
    f[0] = static_cast<float>(d.bonus_pile_size[t]) / init;
    f[1] = static_cast<float>(bonus_mean[t] / 10.0);
    f[2] = d.bonus_count[me][t] / 5.0f;
    f[3] = d.bonus_count[opp][t] / 5.0f;
    f[4] = my_value / 30.0f;
    f[5] = static_cast<float>(d.bonus_count[opp][t] * bonus_mean[t] / 30.0);
  }

  {
    float* f = token(kGlobalSlot, 1 + kGlobalSlot);
    const int my_bonus_n = d.bonus_count[me][0] + d.bonus_count[me][1] + d.bonus_count[me][2];
    const int opp_bonus_n = d.bonus_count[opp][0] + d.bonus_count[opp][1] + d.bonus_count[opp][2];
    f[0] = d.rupees[me] / 100.0f;
    f[1] = d.rupees[opp] / 100.0f;
    f[2] = d.bonus_total[me] / 60.0f;
    f[3] = static_cast<float>(opp_bonus / 60.0);
    f[4] = clip_unit(static_cast<float>((d.rupees[me] + d.bonus_total[me] - d.rupees[opp] - opp_bonus) / 50.0));
    f[5] = my_hand / 7.0f;
    f[6] = opp_hand / 7.0f;
    f[7] = (kTotalDeckSize - d.deck_pos) / 55.0f;
    f[8] = d.piles_exhausted / 3.0f;
    f[9] = std::min(d.plies, kMaxPlies) / static_cast<float>(kMaxPlies);
    f[10] = d.goods_tokens_taken[me] / 20.0f;
    f[11] = d.goods_tokens_taken[opp] / 20.0f;
    f[12] = my_bonus_n / 10.0f;
    f[13] = opp_bonus_n / 10.0f;
    f[14] = take_total > 0 && give_total == 0 ? 1.0f : 0.0f;
    f[15] = give_total > 0 ? 1.0f : 0.0f;
    f[16] = take_total / 5.0f;
    f[17] = (take_total - give_total) / 5.0f;
    f[18] = d.current_player == me ? 1.0f : 0.0f;
    f[19] = opp_unknown / 7.0f;
  }
}

StateSerializer make_jaipur_serializer() {
  return [](const IGameState& state) -> AnyMap {
    const auto& d = static_cast<const JaipurState&>(state).data;
    AnyMap m;
    m["current_player"] = std::any(d.current_player);
    m["is_terminal"] = std::any(d.terminal);
    m["winner"] = std::any(d.winner);
    m["plies"] = std::any(d.plies);
    // Market as a flat count list [g0..g5, camels].
    std::vector<int> mkt(d.market_count.begin(), d.market_count.end());
    m["market_count"] = std::any(mkt);
    // Goods in both hands together, from public counts only: every card
    // drawn from the deck is in the market, a herd, the discard pile or a
    // hand. hand_size_<p> subtracts the OTHER seat's hand, so for a viewer
    // the opponent's hand size depends only on slots that viewer sees.
    int hands_total = d.deck_pos;
    for (int c : d.market_count) hands_total -= c;
    for (int p = 0; p < kPlayers; ++p) hands_total -= d.camel_count[p];
    for (int c : d.discard_count) hands_total -= c;
    for (int p = 0; p < kPlayers; ++p) {
      int other = 0;
      for (int c : d.hand_count[1 - p]) other += c;
      m["hand_size_" + std::to_string(p)] = std::any(hands_total - other);
    }
    for (int p = 0; p < kPlayers; ++p) {
      std::vector<int> hc(d.hand_count[p].begin(), d.hand_count[p].end());
      m["hand_count_" + std::to_string(p)] = std::any(hc);
      m["camel_count_" + std::to_string(p)] = std::any(d.camel_count[p]);
      m["rupees_" + std::to_string(p)] = std::any(d.rupees[p]);
      std::vector<int> bc = {d.bonus_count[p][0], d.bonus_count[p][1],
                             d.bonus_count[p][2]};
      m["bonus_count_" + std::to_string(p)] = std::any(bc);
      m["bonus_total_" + std::to_string(p)] = std::any(d.bonus_total[p]);
      // Per-tier, per-value breakdown of the bonus tokens p drew (flat
      // [tier*3 + value_index]; owner-only on the wire).
      std::vector<int> bt;
      for (int t = 0; t < kBonusTiers; ++t)
        for (int v = 0; v < kBonusValues; ++v) bt.push_back(d.bonus_taken[p][t][v]);
      m["bonus_taken_" + std::to_string(p)] = std::any(bt);
      m["goods_tokens_taken_" + std::to_string(p)] = std::any(d.goods_tokens_taken[p]);
    }
    for (int g = 0; g < kGoodTypes; ++g) {
      m["price_remaining_" + std::to_string(g)] =
          std::any(static_cast<int>(d.value_tokens[g].size()));
      int top = d.value_tokens[g].empty() ? 0 : d.value_tokens[g].back();
      m["current_price_" + std::to_string(g)] = std::any(top);
    }
    m["bonus_remaining_3"] = std::any(d.bonus_pile_size[0]);
    m["bonus_remaining_4"] = std::any(d.bonus_pile_size[1]);
    m["bonus_remaining_5"] = std::any(d.bonus_pile_size[2]);
    // Hidden compositions (truth-side dump, like hand_count_<opp>): the
    // face-down bonus piles (flat [tier*3 + value_index]) and the draw pile
    // ([g0..g5, camels]). Neither has an order.
    std::vector<int> bp;
    for (int t = 0; t < kBonusTiers; ++t)
      for (int v = 0; v < kBonusValues; ++v) bp.push_back(d.bonus_pile[t][v]);
    m["bonus_pile"] = std::any(bp);
    std::vector<int> dk(d.deck_count.begin(), d.deck_count.end());
    m["deck_count"] = std::any(dk);
    std::vector<int> dc(d.discard_count.begin(), d.discard_count.end());
    m["discard_count"] = std::any(dc);
    m["piles_exhausted"] = std::any(d.piles_exhausted);
    m["deck_remaining"] = std::any(kTotalDeckSize - d.deck_pos);
    m["camel_bonus_token"] = std::any(d.camel_bonus_token);
    std::vector<int> tt(d.trade_take.begin(), d.trade_take.end());
    std::vector<int> tg(d.trade_give.begin(), d.trade_give.end());
    m["trade_take"] = std::any(tt);
    m["trade_give"] = std::any(tg);
    return m;
  };
}

ActionDescriptor make_jaipur_descriptor() {
  return [](ActionId action) -> AnyMap {
    AnyMap m;
    m["action_id"] = std::any(static_cast<int>(action));
    auto set = [&](const char* type, int good, int count) {
      m["type"] = std::any(std::string(type));
      if (good >= 0) m["good_type"] = std::any(good);
      if (count > 0) m["count"] = std::any(count);
    };
    if (is_take_action(action)) set("take", action - kActTakeBase, 1);
    else if (action == kActTakeCamels) set("take_camels", -1, 0);
    else if (is_sell_action(action)) set("sell", sell_type(action), sell_count(action));
    else if (is_trade_take_action(action))
      set("trade_take", trade_take_type(action), trade_take_count(action));
    else if (is_trade_give_action(action))
      set("trade_give", trade_give_type(action), trade_give_count(action));
    return m;
  };
}

}  // namespace board_ai::jaipur
