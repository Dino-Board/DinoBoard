#include "jaipur_belief_tracker.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

#include "jaipur_rules.h"

namespace board_ai::jaipur {

namespace {
constexpr int kMemorySlots = kPlayers * kGoodTypes;

[[noreturn]] void inconsistent(const std::string& what) {
  throw std::runtime_error(
      "JaipurBeliefTracker::randomize_unseen: " + what +
      " — the observer's public state / tracker memory is inconsistent "
      "(wrong perspective, or events were not streamed / restored)");
}

int int_field(const AnyMap& m, const char* key, const char* event) {
  auto it = m.find(key);
  if (it == m.end() || it->second.type() != typeid(int)) {
    throw std::invalid_argument(std::string("JaipurBeliefTracker: event '") + event +
                                "' needs an int '" + key + "'");
  }
  return std::any_cast<int>(it->second);
}
}  // namespace

void JaipurBeliefTracker::init(IGameState& state, int perspective,
                                const AnyMap& payload) {
  auto& d = static_cast<JaipurState&>(state).data;

  // Own hand (private bootstrap the public snapshot cannot carry). Payload is
  // a per-type count vector.
  auto it_hand = payload.find("own_hand");
  if (it_hand != payload.end()) {
    const auto own = any_int_list(it_hand->second, "JaipurBeliefTracker::init own_hand");
    if (static_cast<int>(own.size()) != kGoodTypes) {
      throw std::invalid_argument("JaipurBeliefTracker::init: 'own_hand' must hold 6 ints");
    }
    for (int g = 0; g < kGoodTypes; ++g) d.hand_count[perspective][g] = own[g];
  }

  known_ = {};
  // Mid-game bootstrap (belief_tracker.h contract): the payload carries the
  // complete goods-flow memory, so a session attached at ply N holds the
  // same belief as one that streamed every action from ply 0. It cannot be
  // re-derived from the current public state.
  auto it_known = payload.find("known");
  if (it_known != payload.end()) {
    const auto known = any_int_list(it_known->second, "JaipurBeliefTracker::init known");
    if (static_cast<int>(known.size()) != kMemorySlots) {
      throw std::invalid_argument(
          "JaipurBeliefTracker::init: 'known' must hold " +
          std::to_string(kMemorySlots) + " ints");
    }
    for (int p = 0; p < kPlayers; ++p)
      for (int g = 0; g < kGoodTypes; ++g) {
        const int v = known[p * kGoodTypes + g];
        if (v < 0 || v > kHandSize)
          throw std::invalid_argument("JaipurBeliefTracker::init: 'known' entry out of range");
        known_[p][g] = v;
      }
  }
}

AnyMap JaipurBeliefTracker::pack_init_payload(const IGameState& gt_state,
                                              int perspective) const {
  const auto& d = static_cast<const JaipurState&>(gt_state).data;
  AnyMap payload;
  std::vector<int> own(d.hand_count[perspective].begin(),
                       d.hand_count[perspective].end());
  payload["own_hand"] = std::any(own);
  // Complete tracker memory (public, perspective-agnostic): see init().
  payload["known"] = serialize().at("known");
  return payload;
}

void JaipurBeliefTracker::observe_public_event(
    int /*actor*/, ActionId /*action*/,
    const std::vector<PublicEvent>& events) {
  auto player_of = [](const AnyMap& m, const char* event) {
    const int p = int_field(m, "player", event);
    if (p < 0 || p >= kPlayers)
      throw std::invalid_argument("JaipurBeliefTracker: event player out of range");
    return p;
  };
  auto spend = [](int& known, int k) { known = std::max(known, k) - k; };
  for (const auto& [kind, payload] : events) {
    if (kind == "goods_taken") {
      const int p = player_of(payload, "goods_taken");
      known_[p][int_field(payload, "good_type", "goods_taken")] +=
          int_field(payload, "count", "goods_taken");
    } else if (kind == "goods_sold") {
      const int p = player_of(payload, "goods_sold");
      spend(known_[p][int_field(payload, "good_type", "goods_sold")],
            int_field(payload, "count", "goods_sold"));
    } else if (kind == "trade_resolved") {
      const int p = player_of(payload, "trade_resolved");
      auto take_it = payload.find("take");
      auto give_it = payload.find("give");
      if (take_it == payload.end() || give_it == payload.end())
        throw std::invalid_argument("JaipurBeliefTracker: trade_resolved needs 'take' and 'give'");
      const auto take = any_int_list(take_it->second, "JaipurBeliefTracker trade_resolved take");
      const auto give = any_int_list(give_it->second, "JaipurBeliefTracker trade_resolved give");
      if (static_cast<int>(take.size()) != kGoodTypes ||
          static_cast<int>(give.size()) != kGoodTypes + 1)
        throw std::invalid_argument("JaipurBeliefTracker: trade_resolved count vectors have the wrong size");
      for (int g = 0; g < kGoodTypes; ++g) {
        spend(known_[p][g], give[g]);
        known_[p][g] += take[g];
      }
    }
  }
}

// Fills EVERY slot hidden from `observer` — opponent hand, draw-pile
// composition, opponent bonus values, bonus-pile composition — from what the
// observer knows, and touches no public slot. While the opponent is in the
// middle of a trade, its public prefix must stay completable in the sampled
// world (factored-actions skill): goods it already gave are placed in its
// hand, and worlds where the trade cannot be finished are redrawn.
void JaipurBeliefTracker::randomize_unseen(IGameState& state, int observer,
                                            std::mt19937_64& rng) const {
  auto& s = static_cast<JaipurState&>(state);
  if (observer < 0 || observer >= kPlayers) return;
  const bool opp_mid_trade =
      s.data.current_player != observer && !jaipur_trade_idle(s.data) && !s.data.terminal;
  constexpr int kMaxTries = 1000;
  for (int t = 0; t < kMaxTries; ++t) {
    sample_world(s, observer, rng);
    if (!opp_mid_trade || !JaipurRules().legal_actions(s).empty()) return;
  }
  inconsistent("no sampled world lets the opponent finish its chosen trade atoms");
}

void JaipurBeliefTracker::sample_world(JaipurState& s, int observer,
                                       std::mt19937_64& rng) const {
  auto& d = s.data;
  const int opp = 1 - observer;

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
  const int camels_in_deck = kCamelCardCount - d.market_count[kCamel] -
                             d.camel_count[0] - d.camel_count[1];
  const int deck_goods = deck_size - camels_in_deck;
  const int opp_hand_goods = unseen_goods - deck_goods;
  if (deck_size < 0 || camels_in_deck < 0 || deck_goods < 0 ||
      opp_hand_goods < 0 || opp_hand_goods > kHandSize) {
    inconsistent("public card counts do not add up");
  }

  // --- Opponent hand: the known lower bounds (raised by goods it already
  // gave in a pending trade), the rest drawn uniformly from the unseen pool.
  const bool opp_moves = d.current_player == opp;
  d.hand_count[opp].fill(0);
  int placed = 0;
  for (int g = 0; g < kGoodTypes; ++g) {
    int n = known_[opp][g];
    if (opp_moves) n = std::max(n, d.trade_give[g]);
    if (n > unseen[g]) inconsistent("known cards exceed the unseen pool");
    d.hand_count[opp][g] = n;
    placed += n;
  }
  const int residual = opp_hand_goods - placed;
  if (residual < 0) inconsistent("known cards exceed the opponent's hand size");
  if (residual > 0) {
    std::vector<std::int8_t> pool;
    for (int g = 0; g < kGoodTypes; ++g)
      for (int k = d.hand_count[opp][g]; k < unseen[g]; ++k)
        pool.push_back(static_cast<std::int8_t>(g));
    std::shuffle(pool.begin(), pool.end(), rng);
    for (int k = 0; k < residual; ++k) d.hand_count[opp][pool[k]]++;
  }

  // --- Draw pile (hidden composition): exactly the unseen cards the sampled
  // opponent hand does not hold. Sizes add up to the public deck size by
  // construction; sim-time draws sample from these counts with the sim rng,
  // so no sim can follow the true draw sequence.
  for (int g = 0; g < kGoodTypes; ++g)
    d.deck_count[g] = unseen[g] - d.hand_count[opp][g];
  d.deck_count[kCamel] = camels_in_deck;

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
  // Perspective-AGNOSTIC: derived purely from public goods-flow events, so
  // two trackers fed the same event stream produce equal output (BUG-017).
  std::vector<int> flat;
  for (int p = 0; p < kPlayers; ++p)
    for (int g = 0; g < kGoodTypes; ++g) flat.push_back(known_[p][g]);
  AnyMap m;
  m["known"] = std::any(flat);
  return m;
}

}  // namespace board_ai::jaipur
