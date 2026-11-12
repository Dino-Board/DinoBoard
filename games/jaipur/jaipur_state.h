#pragma once
#include <any>
#include <array>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>
#include <string>

#include "../../engine/core/game_interfaces.h"
#include "../../engine/core/visibility_schema.h"

namespace board_ai::jaipur {

constexpr int kGoodTypes = 6;       // diamond, gold, silver, spice, cloth, leather
constexpr int kPlayers = 2;
constexpr int kHandSize = 7;        // max goods in hand (camels don't count)
// Mirrors game.json::max_game_plies; only used to normalise the encoder's
// `plies` feature. The rules do NOT terminate on it: an episode that reaches
// the runner's ply cap goes through the adjudicator registered in
// jaipur_register.cpp (z = -1 for both players, the anti-stall signal).
// Plies must be a schema slot (kPlies) so it enters the per-perspective hash
// — otherwise the DAG would merge states across plies, rule-induced cycles
// would form, and MCTS sims would stall (observed: 600k → 170k sims/s after
// the network learned to stall).
constexpr int kMaxPlies = 200;

constexpr int kGoodCardCounts[kGoodTypes] = {6, 6, 6, 8, 8, 10};
constexpr int kCamelCardCount = 11;
constexpr int kTotalDeckSize = 55;
constexpr int kInitialDealPerPlayer = 5;
constexpr int kMarketSize = 5;
constexpr int kMaxTrade = 5;         // at most 5 cards swapped (market size)
constexpr int kMaxSell = kHandSize;  // a sale holds at most a full hand
constexpr int kCamel = kGoodTypes;   // market / trade index of camels

// Face-down bonus tokens (rulebook): per tier (3 / 4 / 5+ cards sold) the
// distinct face values and how many tokens of each value exist.
constexpr int kBonusTiers = 3;
constexpr int kBonusValues = 3;
constexpr int kBonusValue[kBonusTiers][kBonusValues] = {
    {1, 2, 3}, {4, 5, 6}, {8, 9, 10}};
constexpr int kBonusInitCount[kBonusTiers][kBonusValues] = {
    {2, 3, 2}, {2, 2, 2}, {2, 1, 2}};

// ---- Action encoding: one turn = one BGA request ----
//   0..5     take one good of type g from the market
//   6        take all camels
//   7..48    sell k cards of type g: 7 + g*7 + (k-1), k = 1..7 (partial
//            sales allowed; diamond / gold / silver need k >= 2)
//   49..78   trade, take k market goods of type g: 49 + g*5 + (k-1)
//   79..113  trade, give k cards of type x: 79 + x*5 + (k-1), x = 0..5 goods
//            from hand, x = 6 camels from the herd
// A trade is a factored action (factored-actions skill): take atoms in
// strictly ascending g, then give atoms in strictly ascending x; it resolves
// on the give atom that makes the given count equal the taken count. The
// chosen atoms are public state (trade_take / trade_give) and in the hash.
// Every legal atom keeps the trade completable, so a started trade never
// dead-ends and needs no cancel.
constexpr int kActTakeBase       = 0;
constexpr int kActTakeCamels     = 6;
constexpr int kActSellBase       = 7;
constexpr int kActTradeTakeBase  = kActSellBase + kGoodTypes * kMaxSell;           // 49
constexpr int kActTradeGiveBase  = kActTradeTakeBase + kGoodTypes * kMaxTrade;     // 79
constexpr int kActionSpace       = kActTradeGiveBase + (kGoodTypes + 1) * kMaxTrade;  // 114

constexpr ActionId sell_action(int g, int k) { return kActSellBase + g * kMaxSell + (k - 1); }
constexpr ActionId trade_take_action(int g, int k) { return kActTradeTakeBase + g * kMaxTrade + (k - 1); }
constexpr ActionId trade_give_action(int x, int k) { return kActTradeGiveBase + x * kMaxTrade + (k - 1); }
constexpr bool is_take_action(ActionId a) { return a >= kActTakeBase && a < kActTakeBase + kGoodTypes; }
constexpr bool is_sell_action(ActionId a) { return a >= kActSellBase && a < kActTradeTakeBase; }
constexpr bool is_trade_take_action(ActionId a) { return a >= kActTradeTakeBase && a < kActTradeGiveBase; }
constexpr bool is_trade_give_action(ActionId a) { return a >= kActTradeGiveBase && a < kActionSpace; }
// (type, count) of a sell / trade-take / trade-give atom.
constexpr int sell_type(ActionId a) { return (a - kActSellBase) / kMaxSell; }
constexpr int sell_count(ActionId a) { return (a - kActSellBase) % kMaxSell + 1; }
constexpr int trade_take_type(ActionId a) { return (a - kActTradeTakeBase) / kMaxTrade; }
constexpr int trade_take_count(ActionId a) { return (a - kActTradeTakeBase) % kMaxTrade + 1; }
constexpr int trade_give_type(ActionId a) { return (a - kActTradeGiveBase) / kMaxTrade; }
constexpr int trade_give_count(ActionId a) { return (a - kActTradeGiveBase) % kMaxTrade + 1; }

// Diamond / gold / silver sell in sets of at least two.
constexpr int min_sell_count(int g) { return g <= 2 ? 2 : 1; }

struct JaipurData {
  int current_player = 0;
  int plies = 0;
  int winner = -1;
  bool terminal = false;

  // Market as a COUNT vector (unordered multiset): market_count[g] = number of
  // good-type-g cards face up; market_count[6] = number of camels. Count form
  // so equivalent markets (same multiset, different layout) hash identically
  // → maximal DAG reuse in MCTS.
  std::array<int, kGoodTypes + 1> market_count{};   // [0..5]=goods, [6]=camels
  // Draw pile. There is NO stored order: a draw is a weighted sample from
  // `deck_count` with the caller's rng at the moment it happens (bag-draw
  // recipe), so no state — truth, session or sim — can hold "the next card".
  // The composition is hidden (it is the complement of the opponent's hand in
  // the unseen pool); the size is public via `deck_pos`.
  std::array<int, kGoodTypes + 1> deck_count{};     // HIDDEN: [0..5]=goods, [6]=camels
  int deck_pos = 0;  // PUBLIC: cards drawn so far; size = kTotalDeckSize - deck_pos
  // PUBLIC: goods cards sold (discarded face up), per type. Not derivable
  // from the value-token piles: a sale may hold more cards than tokens left.
  std::array<int, kGoodTypes> discard_count{};

  // Hands as COUNT vectors: hand_count[p][g] = number of good-type-g cards in
  // player p's hand. Unordered → equivalent hands hash identically. Camels
  // are separate (camel_count), not in hand_count.
  std::array<std::array<int, kGoodTypes>, kPlayers> hand_count{};
  std::array<int, kPlayers> camel_count{};
  std::array<int, kPlayers> rupees{};      // PUBLIC: goods-token value + camel token

  std::array<std::vector<int8_t>, kGoodTypes> value_tokens;
  // Face-down bonus piles, per tier (0=3-kind, 1=4-kind, 2=5-kind). Like the
  // deck there is no stored order: the size is PUBLIC, the per-value
  // composition is HIDDEN from both players (each knows only its own draws),
  // and a draw is a weighted sample with the caller's rng.
  std::array<int, kBonusTiers> bonus_pile_size{};
  std::array<std::array<int, kBonusValues>, kBonusTiers> bonus_pile{};
  bool camel_bonus_token = true;

  // Bonus tokens: bonus_count[p][tier] PUBLIC; bonus_total[p] and the
  // per-value breakdown bonus_taken[p][tier][v] owner-only.
  std::array<std::array<int, 3>, kPlayers> bonus_count{};
  std::array<int, kPlayers> bonus_total{};
  std::array<std::array<std::array<int, kBonusValues>, kBonusTiers>, kPlayers>
      bonus_taken{};
  // Goods tokens each player has collected (third tie-break).
  std::array<int, kPlayers> goods_tokens_taken{};

  int piles_exhausted = 0;

  // Trade prefix (public): atoms chosen so far this turn, in count form.
  // Market and hand only change when the trade resolves.
  std::array<int, kGoodTypes> trade_take{};      // market goods to take
  std::array<int, kGoodTypes + 1> trade_give{};  // hand goods / camels to give
};

// Draw one card from the draw pile: weighted by `deck_count`, consuming the
// caller's rng. Returns the market index of the card ([0..5] goods, 6 camel)
// or -1 if the pile is empty. Throws if `deck_count` is masked or does not
// add up to the public pile size — i.e. on any state whose hidden slots were
// not produced by the rules (truth) or by randomize_unseen (sim world).
int draw_deck_card(JaipurData& d, std::mt19937_64& rng);

struct JaipurUndoRecord { ActionId action; JaipurData snapshot; };

struct JaipurState final : public CloneableState<JaipurState> {
  JaipurData data;
  std::vector<JaipurUndoRecord> undo_records;

  JaipurState();

  static const viz::VisibilitySchema& schema();

  void reset_with_seed(std::uint64_t seed) override;

  StateHash64 state_hash() const override;
  void hash_field_slot(Hasher& h, viz::FieldId fid,
                       const std::vector<int>& idx) const override;
  std::any read_field_slot(viz::FieldId fid,
                           const std::vector<int>& idx) const override;
  void write_field_slot(viz::FieldId fid,
                        const std::vector<int>& idx,
                        const std::any& value) override;
  void mask_field_slot(viz::FieldId fid,
                       const std::vector<int>& idx) override;
  void mask_all_hidden_slots(const viz::VisibilitySchema& schema,
                             int perspective) override;
  const viz::VisibilitySchema& schema_ref() const override { return schema(); }
  int current_player() const override { return data.current_player; }
  int first_player() const override { return 0; }
  bool is_terminal() const override { return data.terminal; }
  int num_players() const override { return kPlayers; }
  int winner() const override { return data.winner; }
};

}  // namespace board_ai::jaipur
