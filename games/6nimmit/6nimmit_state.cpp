#include "6nimmit_state.h"

#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "../../engine/core/masked_state.h"
#include "../../engine/core/schema_hash.h"
#include "../../engine/core/viz_runtime.h"

namespace board_ai::sixnimmt {

namespace {

// Field ids = declaration order in schema().
enum : viz::FieldId {
  kCurrentPlayer = 0,
  kTerminal,
  kWinner,
  kPly,
  kCurrentRound,
  kSmallPhaseActive,
  kSmallPlayer,
  kApplyCursor,
  kHandHeads,
  kActsMask,
  kTable,
  kUsed,
  kHand,
  kPendingCommits,
};

std::size_t at(int i) { return static_cast<std::size_t>(i); }

}  // namespace

template <int NPlayers>
void deal_hand(SixNimmtData<NPlayers>& d, std::mt19937_64& rng) {
  d.used.fill(0);
  std::vector<int> deck;
  deck.reserve(at(kNumCards));
  for (int c = 1; c <= kNumCards; ++c) deck.push_back(c);
  // One uniform draw per card (no shuffled order is ever stored).
  auto draw = [&]() {
    std::uniform_int_distribution<std::size_t> pick(0, deck.size() - 1);
    const std::size_t i = pick(rng);
    const int c = deck[i];
    deck[i] = deck.back();
    deck.pop_back();
    return c;
  };
  for (int p = 0; p < NPlayers; ++p) {
    d.hand[at(p)].fill(0);
    for (int i = 0; i < kHandSize; ++i) d.hand[at(p)][at(draw() - 1)] = 1;
  }
  for (int r = 0; r < kNumRows; ++r) {
    d.table[at(r)].fill(0);
    const int c = draw();
    d.table[at(r)][0] = static_cast<std::int8_t>(c);
    d.used[at(c - 1)] = 1;
  }
}

template <int NPlayers>
const viz::VisibilitySchema& SixNimmtState<NPlayers>::schema() {
  static const viz::VisibilitySchema s = []() {
    viz::VisibilitySchema schema;
    schema.n_players = NPlayers;
    for (const char* name : {"current_player", "terminal", "winner", "ply", "current_round",
                             "small_phase_active", "small_player", "apply_cursor"}) {
      viz::declare_field(schema, name, viz::all_public({}, NPlayers));
    }
    viz::declare_field(schema, "hand_heads", viz::all_public({NPlayers}, NPlayers));
    viz::declare_field(schema, "acts_mask", viz::all_public({NPlayers}, NPlayers));
    viz::declare_field(schema, "table", viz::all_public({kNumRows, kRowCapacity}, NPlayers));
    viz::declare_field(schema, "used", viz::all_public({kNumCards}, NPlayers));
    viz::declare_field(schema, "hand", viz::owner_only_first_axis({NPlayers, kNumCards}, NPlayers));
    // Owner only while a card lies face down in the slot (and before the
    // first card is played). The rules turn the slots face up when all
    // have played, leave them so while they are emptied, and hide a slot
    // again when its owner plays the next card. kAllViewers: the pattern
    // changes during play; every viewer's column must follow.
    viz::declare_field(schema, "pending_commits", viz::owner_only_first_axis({NPlayers}, NPlayers),
                       viz::VizSync::kAllViewers);
    return schema;
  }();
  return s;
}

template <int NPlayers>
SixNimmtState<NPlayers>::SixNimmtState() {
  reset_with_seed(0xC0FFEEu);
}

template <int NPlayers>
void SixNimmtState<NPlayers>::reset_with_seed(std::uint64_t seed) {
  IGameState::reset_step_count_base();
  data = SixNimmtData<NPlayers>{};
  data.acts_mask.fill(1);
  std::mt19937_64 rng(seed);
  deal_hand<NPlayers>(data, rng);
  viz::init_viz(*this, schema());
}

// Legacy whole-state hash (callers outside the schema walker).
template <int NPlayers>
StateHash64 SixNimmtState<NPlayers>::state_hash() const {
  const auto& d = data;
  std::size_t h = 0;
  hash_combine(h, static_cast<std::size_t>(d.current_player + 3));
  hash_combine(h, static_cast<std::size_t>(d.ply + 7));
  hash_combine(h, static_cast<std::size_t>(d.current_round + 11));
  hash_combine(h, static_cast<std::size_t>(d.small_phase_active ? 1 : 0));
  hash_combine(h, static_cast<std::size_t>(d.small_player + 17));
  hash_combine(h, static_cast<std::size_t>(d.apply_cursor + 19));
  hash_combine(h, static_cast<std::size_t>(d.terminal ? 1 : 0));
  hash_combine(h, static_cast<std::size_t>(d.winner + 23));
  for (int p = 0; p < NPlayers; ++p) {
    hash_combine(h, static_cast<std::size_t>(d.hand_heads[at(p)] + 28));
    hash_combine(h, static_cast<std::size_t>(d.acts_mask[at(p)] + 29));
    hash_combine(h, static_cast<std::size_t>(d.pending_commits[at(p)] + 41));
    if (p == d.current_player) {
      for (int c = 0; c < kNumCards; ++c) hash_combine(h, static_cast<std::size_t>(d.hand[at(p)][at(c)] + 43 + c));
    }
  }
  for (int c = 0; c < kNumCards; ++c) hash_combine(h, static_cast<std::size_t>(d.used[at(c)] + 47 + c));
  for (int r = 0; r < kNumRows; ++r) {
    for (int s = 0; s < kRowCapacity; ++s) hash_combine(h, static_cast<std::size_t>(d.table[at(r)][at(s)] + 53 + r));
  }
  return static_cast<StateHash64>(h);
}

// Tight loops for the big fields; bit-equal to the per-slot default:
// (fid, idx...) then the value, or the hidden sentinel.
template <int NPlayers>
void SixNimmtState<NPlayers>::hash_field_bulk(Hasher& h, viz::FieldId fid, const viz::VizTensor& v,
                                              int perspective) const {
  const auto& d = data;
  switch (fid) {
    case kUsed:
      for (int c = 0; c < kNumCards; ++c) {
        h.add(static_cast<std::size_t>(fid));
        h.add(c);
        h.add(d.used[at(c)] + 47);
      }
      return;
    case kHand:
      for (int p = 0; p < NPlayers; ++p) {
        for (int c = 0; c < kNumCards; ++c) {
          h.add(static_cast<std::size_t>(fid));
          h.add(p);
          h.add(c);
          if (p == perspective) {
            h.add(d.hand[at(p)][at(c)] + 59);
          } else {
            h.combine(kHiddenHashSentinel);
          }
        }
      }
      return;
    case kTable:
      for (int r = 0; r < kNumRows; ++r) {
        for (int s = 0; s < kRowCapacity; ++s) {
          h.add(static_cast<std::size_t>(fid));
          h.add(r);
          h.add(s);
          h.add(d.table[at(r)][at(s)] + 53);
        }
      }
      return;
    default:
      IGameState::hash_field_bulk(h, fid, v, perspective);
      return;
  }
}

template <int NPlayers>
void SixNimmtState<NPlayers>::hash_field_slot(Hasher& h, viz::FieldId fid, const std::vector<int>& idx) const {
  const auto& d = data;
  switch (fid) {
    case kCurrentPlayer:    h.add(d.current_player + 3); return;
    case kTerminal:         h.add(d.terminal ? 1 : 0); return;
    case kWinner:           h.add(d.winner + 23); return;
    case kPly:              h.add(d.ply + 7); return;
    case kCurrentRound:     h.add(d.current_round + 11); return;
    case kSmallPhaseActive: h.add(d.small_phase_active ? 1 : 0); return;
    case kSmallPlayer:      h.add(d.small_player + 17); return;
    case kApplyCursor:      h.add(d.apply_cursor + 19); return;
    case kHandHeads:        h.add(d.hand_heads[at(idx[0])] + 28); return;
    case kActsMask:         h.add(d.acts_mask[at(idx[0])] + 29); return;
    case kTable:            h.add(d.table[at(idx[0])][at(idx[1])] + 53); return;
    case kUsed:             h.add(d.used[at(idx[0])] + 47); return;
    case kHand:             h.add(d.hand[at(idx[0])][at(idx[1])] + 59); return;
    case kPendingCommits:   h.add(d.pending_commits[at(idx[0])] + 61); return;
  }
}

template <int NPlayers>
std::any SixNimmtState<NPlayers>::read_field_slot(viz::FieldId fid, const std::vector<int>& idx) const {
  const auto& d = data;
  switch (fid) {
    case kCurrentPlayer:    return std::any(static_cast<int>(d.current_player));
    case kTerminal:         return std::any(static_cast<bool>(d.terminal));
    case kWinner:           return std::any(static_cast<int>(d.winner));
    case kPly:              return std::any(static_cast<int>(d.ply));
    case kCurrentRound:     return std::any(static_cast<int>(d.current_round));
    case kSmallPhaseActive: return std::any(static_cast<bool>(d.small_phase_active));
    case kSmallPlayer:      return std::any(static_cast<int>(d.small_player));
    case kApplyCursor:      return std::any(static_cast<int>(d.apply_cursor));
    case kHandHeads:        return std::any(static_cast<int>(d.hand_heads[at(idx[0])]));
    case kActsMask:         return std::any(static_cast<int>(d.acts_mask[at(idx[0])]));
    case kTable:            return std::any(static_cast<int>(d.table[at(idx[0])][at(idx[1])]));
    case kUsed:             return std::any(static_cast<int>(d.used[at(idx[0])]));
    case kHand:             return std::any(static_cast<int>(d.hand[at(idx[0])][at(idx[1])]));
    case kPendingCommits:   return std::any(static_cast<int>(d.pending_commits[at(idx[0])]));
  }
  return {};
}

template <int NPlayers>
void SixNimmtState<NPlayers>::write_field_slot(viz::FieldId fid, const std::vector<int>& idx,
                                               const std::any& value) {
  auto& d = data;
  auto as_int = [&]() -> int {
    if (value.type() == typeid(int)) return std::any_cast<int>(value);
    if (value.type() == typeid(bool)) return std::any_cast<bool>(value) ? 1 : 0;
    return 0;
  };
  auto as_bool = [&]() { return as_int() != 0; };
  auto as_int8 = [&]() { return static_cast<std::int8_t>(as_int()); };
  switch (fid) {
    case kCurrentPlayer:    d.current_player = as_int(); return;
    case kTerminal:         d.terminal = as_bool(); return;
    case kWinner:           d.winner = as_int(); return;
    case kPly:              d.ply = as_int(); return;
    case kCurrentRound:     d.current_round = as_int(); return;
    case kSmallPhaseActive: d.small_phase_active = as_bool(); return;
    case kSmallPlayer:      d.small_player = as_int(); return;
    case kApplyCursor:      d.apply_cursor = as_int(); return;
    case kHandHeads:        d.hand_heads[at(idx[0])] = as_int(); return;
    case kActsMask:         d.acts_mask[at(idx[0])] = as_int8(); return;
    case kTable:            d.table[at(idx[0])][at(idx[1])] = as_int8(); return;
    case kUsed:             d.used[at(idx[0])] = as_int8(); return;
    case kHand:             d.hand[at(idx[0])][at(idx[1])] = as_int8(); return;
    case kPendingCommits:   d.pending_commits[at(idx[0])] = as_int8(); return;
  }
}

template <int NPlayers>
void SixNimmtState<NPlayers>::mask_field_slot(viz::FieldId fid, const std::vector<int>& idx) {
  switch (fid) {
    case kHand:
      data.hand[at(idx[0])][at(idx[1])] = kPlaceholderInt8;
      return;
    case kPendingCommits:
      data.pending_commits[at(idx[0])] = kPlaceholderInt8;
      return;
    default:
      return;
  }
}

template void deal_hand<2>(SixNimmtData<2>&, std::mt19937_64&);
template void deal_hand<3>(SixNimmtData<3>&, std::mt19937_64&);
template void deal_hand<4>(SixNimmtData<4>&, std::mt19937_64&);
template struct SixNimmtData<2>;
template struct SixNimmtData<3>;
template struct SixNimmtData<4>;
template struct SixNimmtState<2>;
template struct SixNimmtState<3>;
template struct SixNimmtState<4>;

}  // namespace board_ai::sixnimmt
