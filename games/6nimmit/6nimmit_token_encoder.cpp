#include "6nimmit_token_encoder.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "../../engine/core/game_interfaces.h"

namespace board_ai::sixnimmt {

namespace {

// Scales of the count features (see docs/games/6nimmit_api.md). The first
// three are also caps: beyond them a range is simply "far" / "crowded".
constexpr float kUnseenScale = 20.0f;      // unseen cards in a range
constexpr float kExpectedScale = 5.0f;     // of them, the share the other players hold
constexpr float kUnseenHeadsScale = 40.0f; // bullheads on the unseen cards of a range
constexpr float kRowHeadsScale = 10.0f;    // bullheads lying in a row, or taken in one round
constexpr float kHandHeadsScale = 30.0f;   // bullheads taken in one hand

struct FeatureDesc {
  const char* name;
  int offset;
  int width;
};
#define SIXNIMMT_DESC_TABLE(table, ns, FEATURES) const FeatureDesc table[] = {FEATURES(SIXNIMMT_DESC_##ns)};
#define SIXNIMMT_DESC_sgf(name, width) {#name, sgf::name, width},
#define SIXNIMMT_DESC_spf(name, width) {#name, spf::name, width},
#define SIXNIMMT_DESC_srf(name, width) {#name, srf::name, width},
#define SIXNIMMT_DESC_scf(name, width) {#name, scf::name, width},
SIXNIMMT_DESC_TABLE(kGlobalDesc, sgf, SIXNIMMT_GLOBAL_FEATURES)
SIXNIMMT_DESC_TABLE(kPlayerDesc, spf, SIXNIMMT_PLAYER_FEATURES)
SIXNIMMT_DESC_TABLE(kRowDesc, srf, SIXNIMMT_ROW_FEATURES)
SIXNIMMT_DESC_TABLE(kCardDesc, scf, SIXNIMMT_CARD_FEATURES)
#undef SIXNIMMT_DESC_TABLE

template <std::size_t N>
void append_type(std::string& out, const char* name, const FeatureDesc (&table)[N]) {
  out += std::string("{\"name\":\"") + name + "\",\"features\":[";
  for (std::size_t i = 0; i < N; ++i) {
    if (i) out += ',';
    out += std::string("[\"") + table[i].name + "\"," + std::to_string(table[i].offset) + "," +
           std::to_string(table[i].width) + "]";
  }
  out += "]}";
}

struct Row {
  int index = 0;  // table row
  int tail = 0;   // its end card
  int length = 0;
  int heads = 0;
};

// Counts over the cards strictly below a number: below[v] covers 1..v-1.
using Below = std::array<int, kNumCards + 2>;
int between(const Below& below, int lo, int hi) {  // cards in the open range (lo, hi)
  return hi > lo ? below[static_cast<std::size_t>(hi)] - below[static_cast<std::size_t>(lo + 1)] : 0;
}

}  // namespace

template <int NPlayers>
SixNimmtTokenEncoder<NPlayers>::SixNimmtTokenEncoder() {
  static_assert(sgf::kCount <= kFeatures && spf::kCount <= kFeatures && srf::kCount <= kFeatures);
  static_assert(srf::length_last - srf::length + 1 == kRowCapacity && srf::order_last - srf::order + 1 == kNumRows);
  static_assert(NPlayers <= spf::seat_last - spf::seat + 1 && NPlayers <= scf::owner_last - scf::owner + 1);
  layout_.variable_tokens = true;
  layout_.token_positions = true;
  layout_.token_dim = kTokenDim;
  layout_.num_token_types = kTokenTypeCount;
  layout_.num_entity_ids = 1;  // no token names its card
  layout_.num_pointer_heads = kNumHeads;
  layout_.action_token.assign(static_cast<std::size_t>(kActionSpace), -1);
  layout_.action_head.assign(static_cast<std::size_t>(kActionSpace), -1);
  layout_.num_anchor_ids = kNumAnchors;
  layout_.anchor_action.assign(static_cast<std::size_t>(kNumAnchors * kNumHeads), -1);
  auto bind = [&](int anchor_index, int head, int action) {
    layout_.anchor_action[static_cast<std::size_t>(anchor_index * kNumHeads + head)] = action;
  };
  for (int c = 0; c < kNumCards; ++c) bind(kAnchorCard + c, kPlayHead, kCommitOffset + c);
  for (int r = 0; r < kNumRows; ++r) bind(kAnchorRow + r, kTakeHead, kSmallChooseRowOffset + r);
}

template <int NPlayers>
std::string SixNimmtTokenEncoder<NPlayers>::layout_json() {
  std::string out = "{\"token_dim\":" + std::to_string(kTokenDim) + ",\"header\":" + std::to_string(kHeader) +
                    ",\"header_fields\":[\"valid\",\"entity\",\"anchor\",\"type\",\"has_position\",\"x\",\"y\"]," +
                    "\"heads\":[\"play\",\"take\"],\"types\":[";
  append_type(out, "global", kGlobalDesc);
  out += ',';
  append_type(out, "player", kPlayerDesc);
  out += ',';
  append_type(out, "row", kRowDesc);
  out += ',';
  append_type(out, "card", kCardDesc);
  out += "],\"anchors\":[[\"card\"," + std::to_string(1 + kAnchorCard) + "," + std::to_string(kNumCards) +
         "],[\"row\"," + std::to_string(1 + kAnchorRow) + "," + std::to_string(kNumRows) + "]]}";
  return out;
}

template <int NPlayers>
void SixNimmtTokenEncoder<NPlayers>::encode_features(const IGameState& masked_state, int perspective_player,
                                                     const IBeliefTracker* /*tracker*/,
                                                     std::vector<float>* out) const {
  const auto& d = checked_cast<SixNimmtState<NPlayers>>(masked_state).data;
  if (perspective_player < 0 || perspective_player >= NPlayers) {
    throw std::invalid_argument("SixNimmtTokenEncoder: perspective out of range");
  }
  if (out == nullptr) throw std::invalid_argument("SixNimmtTokenEncoder: output is null");
  const int me = perspective_player;
  auto idx = [](int i) { return static_cast<std::size_t>(i); };

  // ---- What I can see ----
  std::array<bool, kNumCards + 2> seen{}, in_hand{}, is_shown{};
  // Seen = shown (the table and the cards taken), my hand, and
  // the played cards I can see: my own from the moment I play it, the
  // others' once all are face up.
  std::vector<int> hand;  // ascending
  for (int c = 1; c <= kNumCards; ++c) {
    if (d.used[idx(c - 1)] == 1) seen[idx(c)] = true;
    if (d.hand[idx(me)][idx(c - 1)] == 1) {
      hand.push_back(c);
      in_hand[idx(c)] = seen[idx(c)] = true;
    }
  }
  std::vector<Row> rows;  // in table order
  for (int r = 0; r < kNumRows; ++r) {
    Row row;
    row.index = r;
    for (int i = 0; i < kRowCapacity; ++i) {
      const int c = d.table[idx(r)][idx(i)];
      if (c < 1 || c > kNumCards) break;
      row.tail = c;
      row.heads += head_value(c);
      ++row.length;
    }
    if (row.length > 0) rows.push_back(row);
  }
  // A played card I can see and that is not on the table yet. Hidden slots
  // hold a placeholder outside 1..104.
  std::array<int, NPlayers> shown{};
  for (int p = 0; p < NPlayers; ++p) {
    const int c = d.pending_commits[idx(p)];
    if (c >= 1 && c <= kNumCards) {
      shown[idx(p)] = c;
      is_shown[idx(c)] = seen[idx(c)] = true;
    }
  }

  Below unseen_below{}, unseen_heads_below{}, mine_below{}, shown_below{};
  for (int c = 1; c <= kNumCards; ++c) {
    unseen_below[idx(c + 1)] = unseen_below[idx(c)] + (seen[idx(c)] ? 0 : 1);
    unseen_heads_below[idx(c + 1)] = unseen_heads_below[idx(c)] + (seen[idx(c)] ? 0 : head_value(c));
    mine_below[idx(c + 1)] = mine_below[idx(c)] + (in_hand[idx(c)] ? 1 : 0);
    shown_below[idx(c + 1)] = shown_below[idx(c)] + (is_shown[idx(c)] ? 1 : 0);
  }
  const int unseen = unseen_below[idx(kNumCards + 1)];
  const bool choosing_row = d.small_phase_active && !d.terminal;
  // Cards the other players hold face down: their hands, plus the card each
  // has (or will have) played this round until those are turned face up.
  const int opp_cards =
      (NPlayers - 1) * std::max(0, kHandSize - d.current_round - (choosing_row ? 1 : 0));
  const float density = unseen > 0 ? static_cast<float>(opp_cards) / static_cast<float>(unseen) : 0.0f;
  auto unseen_feature = [](int n) { return std::min(static_cast<float>(n), kUnseenScale) / kUnseenScale; };
  auto expected_feature = [&](int n) {
    return std::min(static_cast<float>(n) * density, kExpectedScale) / kExpectedScale;
  };

  std::vector<Row> by_tail = rows;  // ascending end card
  std::sort(by_tail.begin(), by_tail.end(), [](const Row& a, const Row& b) { return a.tail < b.tail; });
  const int num_rows = static_cast<int>(by_tail.size());
  auto rows_below = [&](int card) {  // row ends below this card
    int n = 0;
    while (n < num_rows && by_tail[idx(n)].tail < card) ++n;
    return n;
  };
  int min_row_heads = 0;
  for (int i = 0; i < num_rows; ++i) {
    if (i == 0 || by_tail[idx(i)].heads < min_row_heads) min_row_heads = by_tail[idx(i)].heads;
  }
  const bool to_move = !d.terminal && (choosing_row ? d.small_player == me
                                                    : d.acts_mask[idx(me)] != 0 && d.pending_commits[idx(me)] == 0);

  auto token = [&](int type, int anchor_id) {
    const std::size_t base = out->size();
    out->resize(base + static_cast<std::size_t>(kTokenDim), 0.0f);
    float* t = out->data() + base;
    t[0] = 1.0f;
    t[2] = static_cast<float>(anchor_id);
    t[3] = static_cast<float>(type);
    return base;
  };
  auto at = [&](std::size_t base) { return out->data() + base + kHeader; };
  auto position = [&](std::size_t base, int card, int row_class) {
    float* t = out->data() + base;
    t[4] = 1.0f;
    t[5] = static_cast<float>(unseen_below[idx(card)]);
    t[6] = static_cast<float>(row_class);
  };

  // ---- global ----
  {
    float* f = at(token(kGlobalToken, 0));
    f[sgf::round] = static_cast<float>(d.current_round) / static_cast<float>(kHandSize);
    f[sgf::choosing_row] = choosing_row ? 1.0f : 0.0f;
    f[sgf::to_move] = to_move ? 1.0f : 0.0f;
    f[sgf::unseen] = static_cast<float>(unseen) / static_cast<float>(kNumCards);
    f[sgf::opp_cards] = static_cast<float>(opp_cards) / static_cast<float>((NPlayers - 1) * kHandSize);
    f[sgf::density] = density;
    const int low = num_rows > 0 ? unseen_below[idx(by_tail[0].tail)] : 0;  // unseen cards below every row end
    f[sgf::low_unseen] = unseen_feature(low);
    f[sgf::low_expected] = expected_feature(low);
    f[sgf::min_row_heads] = static_cast<float>(min_row_heads) / kRowHeadsScale;
  }

  // ---- players, me first ----
  for (int offset = 0; offset < NPlayers; ++offset) {
    const int p = (me + offset) % NPlayers;
    float* f = at(token(kPlayerToken, 0));
    f[spf::seat + offset] = 1.0f;
    f[spf::hand_heads] = std::min(static_cast<float>(d.hand_heads[idx(p)]), kHandHeadsScale) / kHandHeadsScale;
    f[spf::margin] = margin_value(d, p);  // the value if the game ended now
    f[spf::shown] = shown[idx(p)] != 0 ? 1.0f : 0.0f;
    f[spf::chooser] = choosing_row && d.small_player == p ? 1.0f : 0.0f;
  }

  std::vector<std::pair<int, int>> face_up;  // (card, owner), ascending
  for (int p = 0; p < NPlayers; ++p) {
    if (shown[idx(p)] != 0) face_up.emplace_back(shown[idx(p)], p);
  }
  std::sort(face_up.begin(), face_up.end());
  // The card waiting for its owner to choose a row (0 = no row is being chosen).
  const int chooser_card =
      choosing_row && d.small_player >= 0 && d.small_player < NPlayers ? shown[idx(d.small_player)] : 0;

  // ---- rows, in table order ----
  for (const Row& row : rows) {
    const int order = rows_below(row.tail);
    const int next = order + 1 < num_rows ? by_tail[idx(order + 1)].tail : kNumCards + 1;
    const std::size_t base = token(kRowToken, 1 + kAnchorRow + row.index);
    position(base, row.tail, order + 1);
    float* f = at(base);
    f[srf::tail] = static_cast<float>(row.tail) / static_cast<float>(kNumCards);
    f[srf::length + row.length - 1] = 1.0f;
    f[srf::space] = static_cast<float>(kRowCapacity - row.length) / static_cast<float>(kRowCapacity - 1);
    f[srf::heads] = static_cast<float>(row.heads) / kRowHeadsScale;
    f[srf::cheapest] = row.heads == min_row_heads ? 1.0f : 0.0f;
    f[srf::order + order] = 1.0f;
    // The cards that would land on this row: above its end, below the next.
    const int reach = between(unseen_below, row.tail, next);
    f[srf::reach_unseen] = unseen_feature(reach);
    f[srf::reach_expected] = expected_feature(reach);
    f[srf::reach_heads] =
        std::min(static_cast<float>(between(unseen_heads_below, row.tail, next)), kUnseenHeadsScale) / kUnseenHeadsScale;
    f[srf::reach_mine] = static_cast<float>(between(mine_below, row.tail, next)) / 5.0f;
    f[srf::reach_shown] = static_cast<float>(between(shown_below, row.tail, next)) / 3.0f;
    if (chooser_card != 0) {
      // The rest of the round is forced once the chooser takes a row: their
      // card starts it anew and the other face-up cards follow in ascending
      // order (none can be below every row end any more).
      std::vector<Row> table = rows;
      std::array<int, NPlayers> gain{};
      for (Row& t : table) {
        if (t.index != row.index) continue;
        gain[idx(d.small_player)] += t.heads;
        t = Row{t.index, chooser_card, 1, head_value(chooser_card)};
      }
      for (const auto& [card, owner] : face_up) {
        if (owner == d.small_player) continue;
        Row* target = nullptr;
        for (Row& t : table) {
          if (t.tail < card && (target == nullptr || t.tail > target->tail)) target = &t;
        }
        if (target == nullptr) continue;  // cannot happen: the chooser's card is lower
        if (target->length == kRowCapacity) {
          gain[idx(owner)] += target->heads;
          *target = Row{target->index, card, 1, head_value(card)};
        } else {
          target->tail = card;
          target->heads += head_value(card);
          ++target->length;
        }
      }
      int others = 0;
      for (int p = 0; p < NPlayers; ++p) {
        if (p != me) others += gain[idx(p)];
      }
      f[srf::if_taken_mine] = static_cast<float>(gain[idx(me)]) / kRowHeadsScale;
      f[srf::if_taken_others] = static_cast<float>(others) / kRowHeadsScale;
    }
  }

  // ---- cards: my hand, then the face-up played cards, each ascending ----
  auto card_token = [&](int card, int owner, bool held, int hand_rank) {
    const int row_class = rows_below(card);
    const std::size_t base = token(kCardToken, held ? 1 + kAnchorCard + card - 1 : 0);
    position(base, card, row_class);
    float* f = at(base);
    const int offset = (owner - me + NPlayers) % NPlayers;
    f[scf::in_hand] = held ? 1.0f : 0.0f;
    f[scf::my_shown] = !held && owner == me ? 1.0f : 0.0f;
    f[scf::opp_shown] = !held && owner != me ? 1.0f : 0.0f;
    f[scf::owner + offset] = 1.0f;
    f[scf::resolving] = !held && choosing_row && d.small_player == owner ? 1.0f : 0.0f;
    f[scf::value] = static_cast<float>(card) / static_cast<float>(kNumCards);
    f[scf::heads] = static_cast<float>(head_value(card)) / 7.0f;
    f[scf::hand_rank] = static_cast<float>(hand_rank) / static_cast<float>(kHandSize - 1);
    const int below = unseen_below[idx(card)];
    f[scf::percentile] = unseen > 0 ? static_cast<float>(below) / static_cast<float>(unseen) : 0.0f;
    f[scf::below_expected] = expected_feature(below);
    if (row_class == 0) {  // below every row end: its owner takes a row of their choice
      f[scf::small] = 1.0f;
      f[scf::small_cost] = static_cast<float>(min_row_heads) / kRowHeadsScale;
      f[scf::cost_now] = f[scf::small_cost];
      return;
    }
    const Row& row = by_tail[idx(row_class - 1)];
    const int gap = between(unseen_below, row.tail, card);
    f[scf::gap_safe] = gap == 0 ? 1.0f : 0.0f;
    f[scf::gap_unseen] = unseen_feature(gap);
    f[scf::gap_expected] = expected_feature(gap);
    f[scf::gap_mine] = static_cast<float>(between(mine_below, row.tail, card)) / 5.0f;
    f[scf::gap_shown] = static_cast<float>(between(shown_below, row.tail, card)) / 3.0f;
    f[scf::row_length + row.length - 1] = 1.0f;
    f[scf::row_space] = static_cast<float>(kRowCapacity - row.length) / static_cast<float>(kRowCapacity - 1);
    f[scf::row_heads] = static_cast<float>(row.heads) / kRowHeadsScale;
    if (row.length == kRowCapacity) {  // placed now, it is the sixth card and takes the row
      f[scf::sixth] = 1.0f;
      f[scf::cost_now] = f[scf::row_heads];
    }
  };
  for (std::size_t i = 0; i < hand.size(); ++i) card_token(hand[i], me, true, static_cast<int>(i));
  for (const auto& [card, owner] : face_up) card_token(card, owner, false, 0);
}

template class SixNimmtTokenEncoder<2>;
template class SixNimmtTokenEncoder<3>;
template class SixNimmtTokenEncoder<4>;

}  // namespace board_ai::sixnimmt
