#include "6nimmit_state.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "../../engine/core/masked_state.h"
#include "../../engine/core/schema_hash.h"
#include "../../engine/core/viz_runtime.h"

namespace board_ai::sixnimmt {

template <int NPlayers>
const viz::VisibilitySchema& SixNimmtState<NPlayers>::schema() {
  static const viz::VisibilitySchema s = []() {
    viz::VisibilitySchema schema;
    schema.n_players = Cfg::kPlayers;

    // ---- public scalars ----
    viz::declare_field(schema, "current_player",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "terminal",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "winner",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "ply", viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "current_round",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "small_phase_active",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "small_player",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "apply_cursor",
                       viz::all_public({}, Cfg::kPlayers));

    // ---- public per-player 1D ----
    // acts_mask[p]: 1 iff player p must commit during current phase.
    // all_public, phase-constant. Combined with `current_player_` cursor
    // and `pending_commits` placeholder gives the simultaneous-action
    // scan (see .claude/skills/game-rules/simultaneous.md "The three slots").
    viz::declare_field(schema, "acts_mask",
                       viz::all_public({Cfg::kPlayers}, Cfg::kPlayers));

    // ---- public 2D ----
    // table[4][5]: row × slot, value = card number (0 empty).
    viz::declare_field(
        schema, "table",
        viz::all_public({kNumRows, kRowCapacity}, Cfg::kPlayers));
    // bullhead_pile[N][104]: per-player public eaten pile bitmap.
    viz::declare_field(
        schema, "bullhead_pile",
        viz::all_public({Cfg::kPlayers, kNumCards}, Cfg::kPlayers));

    // ---- private ----
    // hand[N][104]: each player sees only own row (owner_only_first_axis).
    viz::declare_field(
        schema, "hand",
        viz::owner_only_first_axis({Cfg::kPlayers, kNumCards}, Cfg::kPlayers));
    // pending_commits[N]: face-down committed card per player.
    // owner_only_first_axis — owner sees own pending; others viz=0.
    // Slot meaningful only when committed[p] == 1 (§1.2).
    // kAllViewers: the pattern only changes at the public seal (all
    // face up) and as revealed cards are applied in ascending order.
    viz::declare_field(
        schema, "pending_commits",
        viz::owner_only_first_axis({Cfg::kPlayers}, Cfg::kPlayers),
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
  auto& d = data;
  d.current_player = 0;
  d.terminal = false;
  d.winner = -1;
  d.ply = 0;
  d.current_round = 0;
  d.small_phase_active = false;
  d.small_player = -1;
  d.apply_cursor = 0;

  for (int p = 0; p < Cfg::kPlayers; ++p) {
    d.hand[static_cast<size_t>(p)].fill(0);
    d.bullhead_pile[static_cast<size_t>(p)].fill(0);
  }
  for (int r = 0; r < kNumRows; ++r) {
    d.table[static_cast<size_t>(r)].fill(0);
  }
  d.pending_commits.fill(0);
  d.committed.fill(0);
  // Initial commit phase: all seats must commit.
  d.acts_mask.fill(1);

  // Deal: pick-and-remove from a 104-card pool — N*10 hand cards + 4
  // row heads. The remaining 104 - N*10 - 4 cards stay out of play (never
  // enter any state field). Per §1.6 the unseen pool to a perspective is
  // {1..104} \ table \ bullhead_pile \ own_hand — the leftover deck
  // contributes to that pool implicitly because it never appears.
  // Per golden standard §2.3 / I3 we don't `std::shuffle` (banned in
  // *_state.cpp); instead we draw indices via uniform_int_distribution
  // which expresses deck-as-multiset randomness one draw at a time.
  std::vector<int> deck(static_cast<size_t>(kNumCards));
  std::iota(deck.begin(), deck.end(), 1);
  std::mt19937_64 rng(seed);
  auto draw_one = [&]() -> int {
    std::uniform_int_distribution<std::size_t> pick(0, deck.size() - 1);
    std::size_t i = pick(rng);
    int c = deck[i];
    deck[i] = deck.back();
    deck.pop_back();
    return c;
  };

  for (int p = 0; p < Cfg::kPlayers; ++p) {
    for (int i = 0; i < kHandSize; ++i) {
      const int card = draw_one();
      d.hand[static_cast<size_t>(p)][static_cast<size_t>(card - 1)] = 1;
    }
  }
  for (int r = 0; r < kNumRows; ++r) {
    const int card = draw_one();
    d.table[static_cast<size_t>(r)][0] = static_cast<std::int8_t>(card);
  }

  viz::init_viz(*this, schema());
}

namespace {

// Hand-rolled fallback hash (legacy state_hash() for non-ISMCTS callers).
// Mirrors what the schema walker produces for current_player's view minus
// the framework's structural (field_pos, idx) salt.
template <int NPlayers>
StateHash64 hash_full(const SixNimmtData<NPlayers>& d) {
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
    hash_combine(h, static_cast<std::size_t>(d.acts_mask[p] + 29));
    // pending_commits viz is owner-only at base, but rules flip it to
    // all-viewers after seal_pending_into_queue. The legacy fallback
    // hash here is a coarse approximation; real ISMCTS hashing flows
    // through hash_field_slot/bulk + viz, where this is exact.
    hash_combine(h, static_cast<std::size_t>(d.pending_commits[p] + 41));
    if (p == d.current_player) {
      for (int c = 0; c < kNumCards; ++c) {
        hash_combine(h, static_cast<std::size_t>(d.hand[p][c] + 43 + c));
      }
    }
    for (int c = 0; c < kNumCards; ++c) {
      hash_combine(h, static_cast<std::size_t>(d.bullhead_pile[p][c] + 47 + c));
    }
  }
  for (int r = 0; r < kNumRows; ++r) {
    for (int s = 0; s < kRowCapacity; ++s) {
      hash_combine(h, static_cast<std::size_t>(d.table[r][s] + 53 + r));
    }
  }
  return static_cast<StateHash64>(h);
}

}  // namespace

template <int NPlayers>
StateHash64 SixNimmtState<NPlayers>::state_hash() const {
  return hash_full<NPlayers>(data);
}

// Field IDs match schema declaration order in schema(). Walker hands
// us a FieldId; switch dispatch eliminates the per-slot string-compare
// chain that was the dominant cost on 6 nimmt!'s 453-slot schema.
namespace {
enum : viz::FieldId {
  kCurrentPlayer = 0,
  kTerminal,
  kWinner,
  kPly,
  kCurrentRound,
  kSmallPhaseActive,
  kSmallPlayer,
  kApplyCursor,
  kActsMask,
  kTable,
  kBullheadPile,
  kHand,
  kPendingCommits,
};
}  // namespace

// Per-field bulk hash override for 6 nimmt!'s big fields. The
// framework default in `mask_state_impl.cpp` walks slot by slot and
// dispatches back here per slot; the override below replaces that with
// a tight inlined loop over the underlying std::array. Smaller fields
// (scalars, the per-player count arrays) keep the framework default
// because the savings don't justify the boilerplate.
template <int NPlayers>
void SixNimmtState<NPlayers>::hash_field_bulk(
    Hasher& h, viz::FieldId fid,
    const viz::VizTensor& v, int perspective) const {
  const auto& d = data;
  switch (fid) {
    case kBullheadPile: {
      // all_public — every slot visible to every perspective. Mix
      // (fid, p, c) then value as `hash_field_slot` would.
      for (int p = 0; p < Cfg::kPlayers; ++p) {
        for (int c = 0; c < kNumCards; ++c) {
          h.add(static_cast<std::size_t>(fid));
          h.add(p);
          h.add(c);
          h.add(d.bullhead_pile[static_cast<size_t>(p)]
                                [static_cast<size_t>(c)] + 47);
        }
      }
      return;
    }
    case kHand: {
      // owner_only_first_axis — owner == perspective sees its full
      // row; other rows are hidden (kHiddenHashSentinel) for this
      // perspective. The owner's row is contiguous in d.hand.
      for (int p = 0; p < Cfg::kPlayers; ++p) {
        const bool visible = (p == perspective);
        if (visible) {
          for (int c = 0; c < kNumCards; ++c) {
            h.add(static_cast<std::size_t>(fid));
            h.add(p);
            h.add(c);
            h.add(d.hand[static_cast<size_t>(p)]
                         [static_cast<size_t>(c)] + 59);
          }
        } else {
          for (int c = 0; c < kNumCards; ++c) {
            h.add(static_cast<std::size_t>(fid));
            h.add(p);
            h.add(c);
            h.combine(kHiddenHashSentinel);
          }
        }
      }
      return;
    }
    case kTable: {
      for (int r = 0; r < kNumRows; ++r) {
        for (int s = 0; s < kRowCapacity; ++s) {
          h.add(static_cast<std::size_t>(fid));
          h.add(r);
          h.add(s);
          h.add(d.table[static_cast<size_t>(r)]
                        [static_cast<size_t>(s)] + 53);
        }
      }
      return;
    }
    default:
      // Fall back to the framework default for fields we don't
      // override — preserves behavior on every other field.
      IGameState::hash_field_bulk(h, fid, v, perspective);
      return;
  }
}

template <int NPlayers>
void SixNimmtState<NPlayers>::hash_field_slot(
    Hasher& h, viz::FieldId fid,
    const std::vector<int>& idx) const {
  const auto& d = data;
  switch (fid) {
    case kCurrentPlayer:          h.add(d.current_player + 3); return;
    case kTerminal:               h.add(d.terminal ? 1 : 0); return;
    case kWinner:                 h.add(d.winner + 23); return;
    case kPly:                    h.add(d.ply + 7); return;
    case kCurrentRound:           h.add(d.current_round + 11); return;
    case kSmallPhaseActive:       h.add(d.small_phase_active ? 1 : 0); return;
    case kSmallPlayer:            h.add(d.small_player + 17); return;
    case kApplyCursor:            h.add(d.apply_cursor + 19); return;
    case kActsMask:
      h.add(d.acts_mask[static_cast<size_t>(idx[0])] + 29); return;
    case kTable:
      h.add(d.table[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] + 53);
      return;
    case kBullheadPile:
      h.add(d.bullhead_pile[static_cast<size_t>(idx[0])]
                            [static_cast<size_t>(idx[1])] + 47);
      return;
    case kHand:
      h.add(d.hand[static_cast<size_t>(idx[0])]
                   [static_cast<size_t>(idx[1])] + 59);
      return;
    case kPendingCommits:
      h.add(d.pending_commits[static_cast<size_t>(idx[0])] + 61);
      return;
  }
}

template <int NPlayers>
std::any SixNimmtState<NPlayers>::read_field_slot(
    viz::FieldId fid, const std::vector<int>& idx) const {
  const auto& d = data;
  switch (fid) {
    case kCurrentPlayer:          return std::any(static_cast<int>(d.current_player));
    case kTerminal:               return std::any(static_cast<bool>(d.terminal));
    case kWinner:                 return std::any(static_cast<int>(d.winner));
    case kPly:                    return std::any(static_cast<int>(d.ply));
    case kCurrentRound:           return std::any(static_cast<int>(d.current_round));
    case kSmallPhaseActive:       return std::any(static_cast<bool>(d.small_phase_active));
    case kSmallPlayer:            return std::any(static_cast<int>(d.small_player));
    case kApplyCursor:            return std::any(static_cast<int>(d.apply_cursor));
    case kActsMask:
      return std::any(static_cast<int>(d.acts_mask[static_cast<size_t>(idx[0])]));
    case kTable:
      return std::any(static_cast<int>(
          d.table[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])]));
    case kBullheadPile:
      return std::any(static_cast<int>(
          d.bullhead_pile[static_cast<size_t>(idx[0])]
                         [static_cast<size_t>(idx[1])]));
    case kHand:
      return std::any(static_cast<int>(
          d.hand[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])]));
    case kPendingCommits:
      return std::any(static_cast<int>(
          d.pending_commits[static_cast<size_t>(idx[0])]));
  }
  return {};
}

template <int NPlayers>
void SixNimmtState<NPlayers>::write_field_slot(
    viz::FieldId fid, const std::vector<int>& idx,
    const std::any& value) {
  auto& d = data;
  auto as_int = [&]() -> int {
    if (value.type() == typeid(int)) return std::any_cast<int>(value);
    if (value.type() == typeid(bool)) return std::any_cast<bool>(value) ? 1 : 0;
    return 0;
  };
  auto as_bool = [&]() -> bool {
    if (value.type() == typeid(bool)) return std::any_cast<bool>(value);
    if (value.type() == typeid(int)) return std::any_cast<int>(value) != 0;
    return false;
  };
  switch (fid) {
    case kCurrentPlayer:          d.current_player = as_int(); return;
    case kTerminal:               d.terminal = as_bool(); return;
    case kWinner:                 d.winner = as_int(); return;
    case kPly:                    d.ply = as_int(); return;
    case kCurrentRound:           d.current_round = as_int(); return;
    case kSmallPhaseActive:       d.small_phase_active = as_bool(); return;
    case kSmallPlayer:            d.small_player = as_int(); return;
    case kApplyCursor:            d.apply_cursor = as_int(); return;
    case kActsMask:
      d.acts_mask[static_cast<size_t>(idx[0])] =
          static_cast<std::int8_t>(as_int());
      return;
    case kTable:
      d.table[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] =
          static_cast<std::int8_t>(as_int());
      return;
    case kBullheadPile:
      d.bullhead_pile[static_cast<size_t>(idx[0])]
                     [static_cast<size_t>(idx[1])] =
          static_cast<std::int8_t>(as_int());
      return;
    case kHand:
      d.hand[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] =
          static_cast<std::int8_t>(as_int());
      return;
    case kPendingCommits:
      d.pending_commits[static_cast<size_t>(idx[0])] =
          static_cast<std::int8_t>(as_int());
      return;
  }
}

template <int NPlayers>
void SixNimmtState<NPlayers>::mask_field_slot(
    viz::FieldId fid, const std::vector<int>& idx) {
  // Only schema-declared fields with viz != all_public ever reach here.
  // For 6 nimmt!: hand[N][104] and pending_commits[N].
  switch (fid) {
    case kHand:
      data.hand[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] =
          kPlaceholderInt8;
      return;
    case kPendingCommits:
      data.pending_commits[static_cast<size_t>(idx[0])] = kPlaceholderInt8;
      return;
    default:
      return;
  }
}

template struct SixNimmtData<2>;
template struct SixNimmtData<3>;
template struct SixNimmtData<4>;
template struct SixNimmtState<2>;
template struct SixNimmtState<3>;
template struct SixNimmtState<4>;

}  // namespace board_ai::sixnimmt
