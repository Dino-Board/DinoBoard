#include "jaipur_belief_tracker.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

namespace board_ai::jaipur {

namespace {
constexpr int kMemorySlots = kPlayers * kGoodTypes;

[[noreturn]] void inconsistent(const std::string& what) {
  throw std::runtime_error(
      "JaipurBeliefTracker::randomize_unseen: " + what +
      " — the observer's public state / tracker memory is inconsistent "
      "(wrong perspective, or events were not streamed / restored)");
}
}  // namespace

void JaipurBeliefTracker::init(IGameState& state, int perspective,
                                const AnyMap& payload) {
  auto& s = static_cast<JaipurState&>(state);
  auto& d = s.data;

  // Own hand (private bootstrap the public snapshot cannot carry). Payload is
  // a per-type count vector.
  auto it_hand = payload.find("own_hand");
  if (it_hand != payload.end() && it_hand->second.type() == typeid(std::vector<int>)) {
    const auto& own = std::any_cast<const std::vector<int>&>(it_hand->second);
    d.hand_count[perspective].fill(0);
    for (int g = 0; g < kGoodTypes && g < static_cast<int>(own.size()); ++g)
      d.hand_count[perspective][g] = own[g];
  }

  sold_ = {};
  net_in_ = {};

  // Mid-game bootstrap (belief_tracker.h contract): the payload carries the
  // complete goods-flow memory, so a session attached at ply N holds the
  // same belief as one that streamed every action from ply 0. sold / net_in
  // cannot be re-derived from the current public state.
  auto it_sold = payload.find("sold");
  auto it_net = payload.find("net_in");
  if (it_sold != payload.end() || it_net != payload.end()) {
    if (it_sold == payload.end() || it_net == payload.end()) {
      throw std::invalid_argument(
          "JaipurBeliefTracker::init: payload must carry both 'sold' and 'net_in'");
    }
    const auto sold = any_int_list(it_sold->second, "JaipurBeliefTracker::init sold");
    const auto net = any_int_list(it_net->second, "JaipurBeliefTracker::init net_in");
    if (static_cast<int>(sold.size()) != kMemorySlots ||
        static_cast<int>(net.size()) != kMemorySlots) {
      throw std::invalid_argument(
          "JaipurBeliefTracker::init: 'sold' / 'net_in' must hold " +
          std::to_string(kMemorySlots) + " ints");
    }
    for (int p = 0; p < kPlayers; ++p)
      for (int g = 0; g < kGoodTypes; ++g) {
        sold_[p][g] = sold[p * kGoodTypes + g] != 0;
        net_in_[p][g] = net[p * kGoodTypes + g];
      }
  }
}

AnyMap JaipurBeliefTracker::pack_init_payload(const IGameState& gt_state,
                                              int perspective) const {
  const auto& s = static_cast<const JaipurState&>(gt_state);
  const auto& d = s.data;
  AnyMap payload;
  std::vector<int> own(d.hand_count[perspective].begin(),
                       d.hand_count[perspective].end());
  payload["own_hand"] = std::any(own);
  // Complete tracker memory (public, perspective-agnostic): see init().
  std::vector<int> sold_flat, net_flat;
  for (int p = 0; p < kPlayers; ++p)
    for (int g = 0; g < kGoodTypes; ++g) {
      sold_flat.push_back(sold_[p][g] ? 1 : 0);
      net_flat.push_back(net_in_[p][g]);
    }
  payload["sold"] = std::any(sold_flat);
  payload["net_in"] = std::any(net_flat);
  return payload;
}

void JaipurBeliefTracker::observe_public_event(
    int actor, ActionId /*action*/,
    const std::vector<PublicEvent>& events) {
  auto player_of = [&](const AnyMap& p) -> int {
    auto it = p.find("player");
    return (it != p.end() && it->second.type() == typeid(int))
               ? std::any_cast<int>(it->second) : -1;
  };
  for (const auto& [kind, payload] : events) {
    if (kind == "goods_taken") {
      // Public: player took one goods card of type g from the market.
      int p = player_of(payload);
      auto it = payload.find("good_type");
      if (p >= 0 && p < kPlayers && it != payload.end() &&
          it->second.type() == typeid(int)) {
        int g = std::any_cast<int>(it->second);
        if (g >= 0 && g < kGoodTypes) net_in_[p][g] += 1;
      }
    } else if (kind == "goods_sold") {
      // Public: player sold all of type g → strong signal. Mark sold and
      // zero net_in: from now net_in exactly tracks their held count.
      int p = player_of(payload);
      auto it = payload.find("good_type");
      if (p >= 0 && p < kPlayers && it != payload.end() &&
          it->second.type() == typeid(int)) {
        int g = std::any_cast<int>(it->second);
        if (g >= 0 && g < kGoodTypes) { sold_[p][g] = true; net_in_[p][g] = 0; }
      }
    } else if (kind == "exchange_resolved") {
      // Public: player's exchange completed. Per type, net_in += (take_count
      // - give_count), then clamp ≥ 0 (a negative means they gave more than
      // they took of that type — only possible from starting-hand cards,
      // which the lower-bound invariant already absorbs).
      int p = player_of(payload);
      if (p < 0 || p >= kPlayers) continue;
      auto take_it = payload.find("take_types");
      auto give_it = payload.find("give_types");
      if (take_it == payload.end() || give_it == payload.end()) continue;
      // any_int_list: over the wire an EMPTY list (an exchange paid with
      // camels only has no give_types) arrives as vector<any>, not
      // vector<int>; skipping it would silently drop the whole exchange.
      const auto takes = any_int_list(
          take_it->second, "JaipurBeliefTracker exchange_resolved take_types");
      const auto gives = any_int_list(
          give_it->second, "JaipurBeliefTracker exchange_resolved give_types");
      int delta[kGoodTypes] = {0};
      for (int g : takes) if (g >= 0 && g < kGoodTypes) ++delta[g];
      for (int g : gives) if (g >= 0 && g < kGoodTypes) --delta[g];
      for (int g = 0; g < kGoodTypes; ++g) {
        net_in_[p][g] += delta[g];
        if (net_in_[p][g] < 0) net_in_[p][g] = 0;
      }
    }
  }
  (void)actor;
}

// Fills EVERY slot hidden from `observer` — opponent hand, draw-pile
// composition, opponent bonus values, bonus-pile composition — from what the
// observer knows, and touches no public slot. All inputs are exact public
// counts (plus the observer's own private slots), so the sampled world
// conserves every card and token by construction; an input that does not add
// up is reported, not clamped.
void JaipurBeliefTracker::randomize_unseen(IGameState& state, int observer,
                                            std::mt19937_64& rng) const {
  auto& s = static_cast<JaipurState&>(state);
  auto& d = s.data;
  if (observer < 0 || observer >= kPlayers) return;
  int opp = 1 - observer;

  // --- Unseen goods: every goods card that is not face up in the market,
  // in the observer's hand, or on the (public) discard pile. They are split
  // between the opponent's hand and the draw pile.
  std::array<int, kGoodTypes> unseen{};
  int unseen_goods = 0;
  for (int g = 0; g < kGoodTypes; ++g) {
    unseen[g] = kGoodCardCounts[g] - d.market_count[g] -
                d.hand_count[observer][g] - d.discard_count[g];
    if (unseen[g] < 0) inconsistent("negative unseen count for a goods type");
    unseen_goods += unseen[g];
  }
  // Camels are fully public (market + both herds), so the pile's camel count
  // and hence the opponent's hand size are known exactly.
  const int deck_size = kTotalDeckSize - d.deck_pos;
  const int camels_in_deck = kCamelCardCount - d.market_count[kGoodTypes] -
                             d.camel_count[0] - d.camel_count[1];
  const int deck_goods = deck_size - camels_in_deck;
  const int opp_hand_goods = unseen_goods - deck_goods;
  if (deck_size < 0 || camels_in_deck < 0 || deck_goods < 0 ||
      opp_hand_goods < 0 || opp_hand_goods > kHandSize) {
    inconsistent("public card counts do not add up");
  }

  // --- Opponent hand contents, via the net-inflow bookkeeping ---
  // For each type g:
  //   - if sold_[opp][g]: opponent holds EXACTLY net_in_[opp][g] of type g.
  //   - else: opponent holds AT LEAST net_in_[opp][g]; the surplus is unknown
  //     starting-hand, sampled from the unsold-types unseen pool.
  d.hand_count[opp].fill(0);
  int placed = 0;
  for (int g = 0; g < kGoodTypes; ++g) {
    const int n = net_in_[opp][g];
    if (n < 0 || n > unseen[g]) inconsistent("net_in exceeds the unseen pool");
    d.hand_count[opp][g] = n;
    placed += n;
  }
  int residual = opp_hand_goods - placed;
  if (residual < 0) inconsistent("net_in exceeds the opponent's hand size");

  if (residual > 0) {
    // Unsold-types unseen pool: per unsold type g, the unseen cards not
    // already placed in the opponent's hand.
    std::vector<std::int8_t> start_pool;
    for (int g = 0; g < kGoodTypes; ++g) {
      if (sold_[opp][g]) continue;
      for (int k = 0; k < unseen[g] - net_in_[opp][g]; ++k)
        start_pool.push_back(static_cast<std::int8_t>(g));
    }
    if (static_cast<int>(start_pool.size()) < residual) {
      inconsistent("not enough unseen cards for the opponent's hand");
    }
    std::shuffle(start_pool.begin(), start_pool.end(), rng);
    for (int k = 0; k < residual; ++k) d.hand_count[opp][start_pool[k]]++;
  }

  // --- Draw pile (hidden composition): exactly the unseen cards the sampled
  // opponent hand does not hold. Sizes add up to the public deck size by
  // construction; sim-time draws sample from these counts with the sim rng,
  // so no sim can follow the true draw sequence.
  for (int g = 0; g < kGoodTypes; ++g)
    d.deck_count[g] = unseen[g] - d.hand_count[opp][g];
  d.deck_count[kGoodTypes] = camels_in_deck;

  // --- Bonus tokens (face down). Per tier the observer knows the full value
  // multiset and its own draws; the rest is split between the opponent's
  // bonus_count[opp][tier] tokens and the pile (public size). Sample that
  // split uniformly — the pile keeps its true size and no value is used twice.
  d.bonus_total[opp] = 0;
  for (int tier = 0; tier < kBonusTiers; ++tier) {
    std::vector<int> pool;
    for (int v = 0; v < kBonusValues; ++v) {
      const int left = kBonusInitCount[tier][v] - d.bonus_taken[observer][tier][v];
      if (left < 0) inconsistent("observer holds more bonus tokens than exist");
      for (int k = 0; k < left; ++k) pool.push_back(v);
    }
    const int need = d.bonus_count[opp][tier];
    if (need < 0 ||
        static_cast<int>(pool.size()) - need != d.bonus_pile_size[tier]) {
      inconsistent("bonus token counts do not add up");
    }
    std::shuffle(pool.begin(), pool.end(), rng);
    d.bonus_taken[opp][tier].fill(0);
    d.bonus_pile[tier].fill(0);
    for (int i = 0; i < static_cast<int>(pool.size()); ++i) {
      const int v = pool[i];
      if (i < need) {
        ++d.bonus_taken[opp][tier][v];
        d.bonus_total[opp] += kBonusValue[tier][v];
      } else {
        ++d.bonus_pile[tier][v];
      }
    }
  }
}

AnyMap JaipurBeliefTracker::serialize() const {
  // Expose the perspective-AGNOSTIC net-inflow bookkeeping (it is derived
  // purely from public goods-flow events, so two trackers fed the same event
  // stream produce equal output — BUG-017 safe).
  AnyMap m;
  std::vector<int> sold_flat, net_flat;
  for (int p = 0; p < kPlayers; ++p)
    for (int g = 0; g < kGoodTypes; ++g) {
      sold_flat.push_back(sold_[p][g] ? 1 : 0);
      net_flat.push_back(net_in_[p][g]);
    }
  m["sold"] = std::any(sold_flat);
  m["net_in"] = std::any(net_flat);
  return m;
}

}  // namespace board_ai::jaipur
