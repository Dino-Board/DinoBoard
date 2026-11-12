#include "jaipur_state.h"
#include <algorithm>
#include <random>
#include <stdexcept>
#include <string>
#include "../../engine/core/masked_state.h"
#include "../../engine/core/rng_salt.h"
#include "../../engine/core/viz_runtime.h"

namespace board_ai::jaipur {

namespace {
enum : viz::FieldId {
  kCurrentPlayer = 0,
  kWinner,
  kTerminal,
  kPlies,          // all_public {} — see jaipur_state.h kMaxPlies note
  kMarketCount,    // all_public {7}  (6 goods + camels)
  kDeckPos,
  kHandCount,      // owner_only_first_axis {kPlayers, kGoodTypes}
  kCamelCount,     // all_public {kPlayers}
  kRupees,
  kValueTokens,    // all_public {kGoodTypes} (remaining pile size)
  kBonusTokens3,
  kBonusTokens4,
  kBonusTokens5,
  kCamelBonusToken,
  kPilesExhausted,
  kBonusCount,     // all_public {kPlayers, 3}
  kBonusTotal,     // owner_only_first_axis {kPlayers}
  kTradeTake,      // all_public {kGoodTypes}
  kTradeGive,      // all_public {kGoodTypes + 1}
  kGoodsTokensTaken,  // all_public {kPlayers}
  kDiscardCount,   // all_public {kGoodTypes}
  kDeckCount,      // all_hidden {kGoodTypes + 1}
  kBonusPile,      // all_hidden {kBonusTiers, kBonusValues}
  kBonusTaken,     // owner_only_first_axis {kPlayers, kBonusTiers, kBonusValues}
  kFieldCount
};
}  // namespace

const viz::VisibilitySchema& JaipurState::schema() {
  static const viz::VisibilitySchema s = []() {
    viz::VisibilitySchema schema;
    schema.n_players = kPlayers;
    viz::declare_field(schema, "current_player",  viz::all_public({}, kPlayers));
    viz::declare_field(schema, "winner",           viz::all_public({}, kPlayers));
    viz::declare_field(schema, "terminal",         viz::all_public({}, kPlayers));
    viz::declare_field(schema, "plies",            viz::all_public({}, kPlayers));
    // Market as a count vector (unordered multiset). Count form so equivalent
    // markets hash identically → DAG reuse.
    viz::declare_field(schema, "market_count",
                       viz::all_public({kGoodTypes + 1}, kPlayers));
    viz::declare_field(schema, "deck_pos",         viz::all_public({}, kPlayers));
    // Hand contents are owner-only (opponent hand is the core hidden info).
    viz::declare_field(
        schema, "hand_count",
        viz::owner_only_first_axis({kPlayers, kGoodTypes}, kPlayers));
    viz::declare_field(schema, "camel_count",      viz::all_public({kPlayers}, kPlayers));
    viz::declare_field(schema, "rupees",           viz::all_public({kPlayers}, kPlayers));
    viz::declare_field(schema, "value_tokens",     viz::all_public({kGoodTypes}, kPlayers));
    viz::declare_field(schema, "bonus_tokens_3",   viz::all_public({}, kPlayers));
    viz::declare_field(schema, "bonus_tokens_4",   viz::all_public({}, kPlayers));
    viz::declare_field(schema, "bonus_tokens_5",   viz::all_public({}, kPlayers));
    viz::declare_field(schema, "camel_bonus_token",viz::all_public({}, kPlayers));
    viz::declare_field(schema, "piles_exhausted",  viz::all_public({}, kPlayers));
    viz::declare_field(schema, "bonus_count",
                       viz::all_public({kPlayers, 3}, kPlayers));
    viz::declare_field(
        schema, "bonus_total",
        viz::owner_only_first_axis({kPlayers}, kPlayers));
    // Trade prefix: the mover's chosen trade atoms (factored-actions skill).
    viz::declare_field(schema, "trade_take",
                       viz::all_public({kGoodTypes}, kPlayers));
    viz::declare_field(schema, "trade_give",
                       viz::all_public({kGoodTypes + 1}, kPlayers));
    viz::declare_field(schema, "goods_tokens_taken",
                       viz::all_public({kPlayers}, kPlayers));
    // Sold goods cards lie face up on the discard pile.
    viz::declare_field(schema, "discard_count",
                       viz::all_public({kGoodTypes}, kPlayers));
    // Draw-pile composition: hidden from both players (size is public via
    // deck_pos). randomize_unseen fills it together with the opponent hand.
    viz::declare_field(schema, "deck_count",
                       viz::all_hidden({kGoodTypes + 1}, kPlayers));
    // Face-down bonus piles: per-value composition hidden from both players
    // (sizes are public via bonus_tokens_3/4/5).
    viz::declare_field(schema, "bonus_pile",
                       viz::all_hidden({kBonusTiers, kBonusValues}, kPlayers));
    // Which bonus values each player drew: owner-only (like bonus_total).
    viz::declare_field(
        schema, "bonus_taken",
        viz::owner_only_first_axis({kPlayers, kBonusTiers, kBonusValues},
                                   kPlayers));
    return schema;
  }();
  return s;
}

JaipurState::JaipurState() { viz::init_viz(*this, schema()); }

int draw_deck_card(JaipurData& d, std::mt19937_64& rng) {
  int total = 0;
  for (int c : d.deck_count) {
    if (c < 0) {
      throw std::runtime_error(
          "jaipur: draw from a masked draw pile (deck_count holds a "
          "placeholder; randomize_unseen was not applied to this state)");
    }
    total += c;
  }
  if (total != kTotalDeckSize - d.deck_pos) {
    throw std::runtime_error(
        "jaipur: deck_count (" + std::to_string(total) +
        " cards) disagrees with the public deck size (" +
        std::to_string(kTotalDeckSize - d.deck_pos) +
        "); hidden slots are stale — randomize_unseen was not applied");
  }
  if (total == 0) return -1;
  // Weighted pick over the remaining cards (same shape as Azul's bag draw).
  int pick = static_cast<int>(rng() % static_cast<std::uint64_t>(total));
  for (int i = 0; i <= kGoodTypes; ++i) {
    if (pick < d.deck_count[i]) {
      --d.deck_count[i];
      ++d.deck_pos;
      return i;
    }
    pick -= d.deck_count[i];
  }
  return -1;  // unreachable
}

void JaipurState::reset_with_seed(std::uint64_t seed) {
  IGameState::reset_step_count_base();
  // The deal uses its own salted stream: runners seed their step rng (which
  // drives every later draw) with this same seed, and the two must not
  // replay the same numbers.
  std::mt19937_64 rng(board_ai::rng::derive_subseed(seed, "jaipur.deal"));
  data = JaipurData{};
  undo_records.clear();

  // Draw pile as counts: 44 goods + 11 camels, no stored order.
  for (int g = 0; g < kGoodTypes; ++g) data.deck_count[g] = kGoodCardCounts[g];
  data.deck_count[kGoodTypes] = kCamelCardCount;
  data.deck_pos = 0;

  // Market: 3 camels are placed face up first (rulebook setup), then 2
  // random cards.
  data.market_count.fill(0);
  data.deck_count[kGoodTypes] -= 3;
  data.market_count[kGoodTypes] = 3;
  data.deck_pos = 3;
  for (int i = 0; i < 2; ++i) {
    const int c = draw_deck_card(data, rng);
    data.market_count[c]++;
  }

  // Deal 5 cards to each player; camels go to the herd.
  for (int p = 0; p < kPlayers; ++p) data.hand_count[p].fill(0);
  for (int i = 0; i < kInitialDealPerPlayer; ++i) {
    for (int p = 0; p < kPlayers; ++p) {
      const int c = draw_deck_card(data, rng);
      if (c == kGoodTypes) data.camel_count[p]++;
      else data.hand_count[p][c]++;
    }
  }

  data.value_tokens = {{
    {5,5,5,7,7}, {5,5,5,6,6}, {5,5,5,5,5},
    {1,1,2,2,3,3,5}, {1,1,2,2,3,3,5}, {1,1,1,1,1,1,2,3,4},
  }};
  // Bonus piles as per-value counts; which value a sale draws is decided by
  // the caller's rng at that moment (apply_sell).
  for (int t = 0; t < kBonusTiers; ++t) {
    data.bonus_pile_size[t] = 0;
    for (int v = 0; v < kBonusValues; ++v) {
      data.bonus_pile[t][v] = kBonusInitCount[t][v];
      data.bonus_pile_size[t] += kBonusInitCount[t][v];
    }
  }
  viz::init_viz(*this, schema());
}

StateHash64 JaipurState::state_hash() const {
  Hasher h;
  for (int p = 0; p < kPlayers; ++p) h.add(state_hash_for_perspective(p));
  return h.finalize();
}

void JaipurState::hash_field_slot(Hasher& h, viz::FieldId fid,
                                   const std::vector<int>& idx) const {
  switch (fid) {
    case kCurrentPlayer:  h.add(data.current_player); return;
    case kWinner:         h.add(data.winner + 1); return;
    case kTerminal:       h.add(data.terminal ? 1 : 0); return;
    case kPlies:          h.add(data.plies); return;
    case kMarketCount:    h.add(data.market_count[idx[0]]); return;
    case kDeckPos:        h.add(data.deck_pos); return;
    case kHandCount:      h.add(data.hand_count[idx[0]][idx[1]]); return;
    case kCamelCount:     h.add(data.camel_count[idx[0]]); return;
    case kRupees:         h.add(data.rupees[idx[0]]); return;
    case kValueTokens:    h.add(static_cast<int>(data.value_tokens[idx[0]].size())); return;
    case kBonusTokens3:   h.add(data.bonus_pile_size[0]); return;
    case kBonusTokens4:   h.add(data.bonus_pile_size[1]); return;
    case kBonusTokens5:   h.add(data.bonus_pile_size[2]); return;
    case kCamelBonusToken:h.add(data.camel_bonus_token ? 1 : 0); return;
    case kPilesExhausted: h.add(data.piles_exhausted); return;
    case kBonusCount:     h.add(data.bonus_count[idx[0]][idx[1]]); return;
    case kBonusTotal:     h.add(data.bonus_total[idx[0]]); return;
    case kTradeTake:      h.add(data.trade_take[idx[0]]); return;
    case kTradeGive:      h.add(data.trade_give[idx[0]]); return;
    case kGoodsTokensTaken: h.add(data.goods_tokens_taken[idx[0]]); return;
    case kDiscardCount:   h.add(data.discard_count[idx[0]]); return;
    case kDeckCount:      h.add(data.deck_count[idx[0]]); return;
    case kBonusPile:      h.add(data.bonus_pile[idx[0]][idx[1]]); return;
    case kBonusTaken:     h.add(data.bonus_taken[idx[0]][idx[1]][idx[2]]); return;
  }
}

std::any JaipurState::read_field_slot(viz::FieldId fid,
                                       const std::vector<int>& idx) const {
  switch (fid) {
    case kCurrentPlayer:  return std::any(data.current_player);
    case kWinner:         return std::any(data.winner);
    case kTerminal:       return std::any(static_cast<int>(data.terminal ? 1 : 0));
    case kPlies:          return std::any(data.plies);
    case kMarketCount:    return std::any(data.market_count[idx[0]]);
    case kDeckPos:        return std::any(data.deck_pos);
    case kHandCount:      return std::any(data.hand_count[idx[0]][idx[1]]);
    case kCamelCount:     return std::any(data.camel_count[idx[0]]);
    case kRupees:         return std::any(data.rupees[idx[0]]);
    case kValueTokens:    return std::any(static_cast<int>(data.value_tokens[idx[0]].size()));
    case kBonusTokens3:   return std::any(data.bonus_pile_size[0]);
    case kBonusTokens4:   return std::any(data.bonus_pile_size[1]);
    case kBonusTokens5:   return std::any(data.bonus_pile_size[2]);
    case kCamelBonusToken:return std::any(static_cast<int>(data.camel_bonus_token ? 1 : 0));
    case kPilesExhausted: return std::any(data.piles_exhausted);
    case kBonusCount:     return std::any(data.bonus_count[idx[0]][idx[1]]);
    case kBonusTotal:     return std::any(data.bonus_total[idx[0]]);
    case kTradeTake:      return std::any(data.trade_take[idx[0]]);
    case kTradeGive:      return std::any(data.trade_give[idx[0]]);
    case kGoodsTokensTaken: return std::any(data.goods_tokens_taken[idx[0]]);
    case kDiscardCount:   return std::any(data.discard_count[idx[0]]);
    case kDeckCount:      return std::any(data.deck_count[idx[0]]);
    case kBonusPile:      return std::any(data.bonus_pile[idx[0]][idx[1]]);
    case kBonusTaken:     return std::any(data.bonus_taken[idx[0]][idx[1]][idx[2]]);
  }
  return {};
}

void JaipurState::write_field_slot(viz::FieldId fid,
                                    const std::vector<int>& idx,
                                    const std::any& value) {
  auto v = std::any_cast<int>(value);
  switch (fid) {
    case kCurrentPlayer:  data.current_player = v; return;
    case kWinner:         data.winner = v; return;
    case kTerminal:       data.terminal = (v != 0); return;
    case kPlies:          data.plies = v; return;
    case kMarketCount:    data.market_count[idx[0]] = v; return;
    case kDeckPos:        data.deck_pos = v; return;
    case kHandCount:      data.hand_count[idx[0]][idx[1]] = v; return;
    case kCamelCount:     data.camel_count[idx[0]] = v; return;
    case kRupees:         data.rupees[idx[0]] = v; return;
    case kValueTokens: {
      data.value_tokens[idx[0]].resize(static_cast<std::size_t>(std::max(0, v)), 0);
      return;
    }
    case kBonusTokens3:   data.bonus_pile_size[0] = v; return;
    case kBonusTokens4:   data.bonus_pile_size[1] = v; return;
    case kBonusTokens5:   data.bonus_pile_size[2] = v; return;
    case kCamelBonusToken: data.camel_bonus_token = (v != 0); return;
    case kPilesExhausted: data.piles_exhausted = v; return;
    case kBonusCount:     data.bonus_count[idx[0]][idx[1]] = v; return;
    case kBonusTotal:     data.bonus_total[idx[0]] = v; return;
    case kTradeTake:      data.trade_take[idx[0]] = v; return;
    case kTradeGive:      data.trade_give[idx[0]] = v; return;
    case kGoodsTokensTaken: data.goods_tokens_taken[idx[0]] = v; return;
    case kDiscardCount:   data.discard_count[idx[0]] = v; return;
    case kDeckCount:      data.deck_count[idx[0]] = v; return;
    case kBonusPile:      data.bonus_pile[idx[0]][idx[1]] = v; return;
    case kBonusTaken:     data.bonus_taken[idx[0]][idx[1]][idx[2]] = v; return;
    default: return;
  }
}

void JaipurState::mask_field_slot(viz::FieldId fid, const std::vector<int>& idx) {
  // Owner-only fields: write the placeholder sentinel into hidden (opponent)
  // slots so the encoder reads "unknown".
  switch (fid) {
    case kHandCount:   data.hand_count[idx[0]][idx[1]] = kPlaceholderInt8; return;
    case kBonusTotal:  data.bonus_total[idx[0]] = kPlaceholderInt8; return;
    case kDeckCount:   data.deck_count[idx[0]] = kPlaceholderInt8; return;
    case kBonusPile:   data.bonus_pile[idx[0]][idx[1]] = kPlaceholderInt8; return;
    case kBonusTaken:
      data.bonus_taken[idx[0]][idx[1]][idx[2]] = kPlaceholderInt8; return;
    default: return;
  }
}

void JaipurState::mask_all_hidden_slots(const viz::VisibilitySchema& schema,
                                        int perspective) {
  IGameState::mask_all_hidden_slots(schema, perspective);
  // Every hidden value of JaipurData is a schema slot (wiped above). The
  // only other storage that can carry hidden bytes is the undo stack,
  // whose records are full unmasked copies of earlier states.
  undo_records.clear();
}

}  // namespace board_ai::jaipur
