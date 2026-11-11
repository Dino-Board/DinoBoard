#include "jaipur_net_adapter.h"
#include "../../engine/core/masked_state.h"  // kPlaceholderInt8

#include <any>

namespace board_ai::jaipur {

// Perspective-rotated count encoding. See plan §3.3 for block layout.
// 74 dims total. Self is always written before opp so the network's
// weights see the same layout regardless of which seat the perspective is.
void JaipurFeatureEncoder::encode_features(const IGameState& state,
    int perspective_player, const IBeliefTracker* tracker,
    std::vector<float>* out) const {
  const auto* s = dynamic_cast<const JaipurState*>(&state);
  if (!s || !out) return;
  const auto& d = s->data;
  const int me = perspective_player;
  const int opp = 1 - me;

  auto push_hand = [&](int pp) {
    // Owner-only viz: own row is truth, opponent row is kPlaceholderInt8.
    for (int g = 0; g < kGoodTypes; ++g) {
      int c = d.hand_count[pp][g];
      out->push_back(c == kPlaceholderInt8 ? 0.0f
                                           : static_cast<float>(c) / 7.0f);
    }
  };

  // --- Block 1: market (7) ---
  for (int i = 0; i < kGoodTypes + 1; ++i)
    out->push_back(static_cast<float>(d.market_count[i]) / 5.0f);

  // --- Block 2: hand_count, self then opp (12) ---
  push_hand(me);
  push_hand(opp);

  // --- Block 3: camels, self then opp (2) ---
  out->push_back(static_cast<float>(d.camel_count[me]) / 11.0f);
  out->push_back(static_cast<float>(d.camel_count[opp]) / 11.0f);

  // --- Block 4: rupees, self then opp (2) ---
  out->push_back(static_cast<float>(d.rupees[me]) / 100.0f);
  out->push_back(static_cast<float>(d.rupees[opp]) / 100.0f);

  // --- Block 5: value-token pile sizes per good (6) ---
  for (int g = 0; g < kGoodTypes; ++g)
    out->push_back(static_cast<float>(d.value_tokens[g].size()) / 10.0f);

  // --- Block 6: bonus pile sizes per tier (3) ---
  out->push_back(static_cast<float>(d.bonus_pile_size[0]) / 7.0f);
  out->push_back(static_cast<float>(d.bonus_pile_size[1]) / 6.0f);
  out->push_back(static_cast<float>(d.bonus_pile_size[2]) / 5.0f);

  // --- Block 7: misc public (3) ---
  // Dropped: camel_bonus_token (≈ constant 1 until terminal — no signal),
  //          current_player==me (framework handles perspective; no signal).
  // plies/kMaxPlies tells V how close we are to the rule-induced terminal
  // (rupees tiebreak fires at d.plies == kMaxPlies). Without this the
  // network can't tell "ply 50, lots of game left" from "ply 199, rupees
  // tiebreak imminent" — and a winning lead in rupees should weight V
  // very differently in those two cases.
  out->push_back(static_cast<float>(d.piles_exhausted) / 6.0f);
  out->push_back(static_cast<float>(kTotalDeckSize - d.deck_pos) / 55.0f);
  out->push_back(static_cast<float>(d.plies) / static_cast<float>(kMaxPlies));

  // --- Block 8: exchange scratch (16) ---
  // Dropped: exchange_total (== sum(take_count[6]); pure redundant).
  out->push_back(static_cast<float>(d.exchange_phase) / 3.0f);
  for (int g = 0; g < kGoodTypes; ++g)
    out->push_back(static_cast<float>(d.exchange_take_count[g]) / 5.0f);
  for (int g = 0; g < kGoodTypes; ++g)
    out->push_back(static_cast<float>(d.exchange_give_count[g]) / 5.0f);
  out->push_back(static_cast<float>(d.exchange_camel_count) / 7.0f);
  out->push_back(static_cast<float>(d.exchange_max_camels) / 7.0f);
  out->push_back(static_cast<float>(d.exchange_min_camels) / 7.0f);

  // --- Block 9: tracker summary (24) ---
  // Order: sold[me], sold[opp], net_in[me], net_in[opp] (each 6 dims).
  // tracker.serialize() returns the perspective-agnostic flat layout
  // [sold[0][0..5], sold[1][0..5], net_in[0][0..5], net_in[1][0..5]] —
  // we rotate to put `me` first. If no tracker (snapshot-only sessions
  // that bypass tracker), emit zeros.
  std::array<int, 12> sold_flat{}, net_flat{};
  if (tracker) {
    AnyMap m = tracker->serialize();
    auto it_sold = m.find("sold");
    auto it_net = m.find("net_in");
    if (it_sold != m.end() && it_sold->second.type() == typeid(std::vector<int>)) {
      const auto& v = std::any_cast<const std::vector<int>&>(it_sold->second);
      for (size_t i = 0; i < v.size() && i < sold_flat.size(); ++i)
        sold_flat[i] = v[i];
    }
    if (it_net != m.end() && it_net->second.type() == typeid(std::vector<int>)) {
      const auto& v = std::any_cast<const std::vector<int>&>(it_net->second);
      for (size_t i = 0; i < v.size() && i < net_flat.size(); ++i)
        net_flat[i] = v[i];
    }
  }
  // sold[me] then sold[opp]
  for (int g = 0; g < kGoodTypes; ++g)
    out->push_back(sold_flat[me * kGoodTypes + g] ? 1.0f : 0.0f);
  for (int g = 0; g < kGoodTypes; ++g)
    out->push_back(sold_flat[opp * kGoodTypes + g] ? 1.0f : 0.0f);
  // net_in[me] then net_in[opp], clamp to [0,7] then ÷7
  for (int g = 0; g < kGoodTypes; ++g) {
    int v = net_flat[me * kGoodTypes + g];
    if (v < 0) v = 0; if (v > 7) v = 7;
    out->push_back(static_cast<float>(v) / 7.0f);
  }
  for (int g = 0; g < kGoodTypes; ++g) {
    int v = net_flat[opp * kGoodTypes + g];
    if (v < 0) v = 0; if (v > 7) v = 7;
    out->push_back(static_cast<float>(v) / 7.0f);
  }
}

StateSerializer make_jaipur_serializer() {
  return [](const IGameState& state) -> AnyMap {
    const auto& d = static_cast<const JaipurState&>(state).data;
    AnyMap m;
    m["current_player"] = std::any(d.current_player);
    m["is_terminal"] = std::any(d.terminal);
    m["winner"] = std::any(d.winner);
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
    m["exch_phase"] = std::any(d.exchange_phase);
    std::vector<int> tk(d.exchange_take_count.begin(), d.exchange_take_count.end());
    std::vector<int> gv(d.exchange_give_count.begin(), d.exchange_give_count.end());
    m["exch_take_count"] = std::any(tk);
    m["exch_give_count"] = std::any(gv);
    m["exch_camel_count"] = std::any(d.exchange_camel_count);
    m["exch_max_camels"] = std::any(d.exchange_max_camels);
    m["exch_min_camels"] = std::any(d.exchange_min_camels);
    m["exch_total"] = std::any(d.exchange_total);
    return m;
  };
}

ActionDescriptor make_jaipur_descriptor() {
  static const char* kGoodNames[kGoodTypes] = {
      "diamond", "gold", "silver", "spice", "cloth", "leather"};
  return [](ActionId action) -> AnyMap {
    AnyMap m;
    m["action_id"] = std::any(static_cast<int>(action));
    auto type_str = [&](const char* s) { m["type"] = std::any(std::string(s)); };
    if (action >= kActSellBase && action < kActSellBase + kGoodTypes) {
      type_str("sell"); m["good_type"] = std::any(action - kActSellBase);
    } else if (action >= kActTakeBase && action < kActTakeBase + kGoodTypes) {
      type_str("take"); m["good_type"] = std::any(action - kActTakeBase);
    } else if (action == kActTakeCamels) {
      type_str("take_camels");
    } else if (action == kActEnterExchange) {
      type_str("enter_exchange");
    } else if (action >= kActExchMktBase && action < kActExchMktBase + kGoodTypes) {
      type_str("exch_take"); m["good_type"] = std::any(action - kActExchMktBase);
    } else if (action == kActExchMktDone) {
      type_str("exch_mkt_done");
    } else if (action >= kActExchCamelBase && action < kActExchCamelBase + 8) {
      type_str("exch_camel"); m["camels"] = std::any(action - kActExchCamelBase);
    } else if (action >= kActExchHandBase && action < kActExchHandBase + kGoodTypes) {
      type_str("exch_give"); m["good_type"] = std::any(action - kActExchHandBase);
    }
    return m;
  };
}

}  // namespace board_ai::jaipur
