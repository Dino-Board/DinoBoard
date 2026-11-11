#include "loveletter_state.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "../../engine/core/masked_state.h"
#include "../../engine/core/schema_hash.h"
#include "../../engine/core/viz_runtime.h"

namespace board_ai::loveletter {

template <int NPlayers>
const viz::VisibilitySchema& LoveLetterState<NPlayers>::schema() {
  static const viz::VisibilitySchema s = []() {
    viz::VisibilitySchema schema;
    schema.n_players = Cfg::kPlayers;

    // ---- public scalars ----
    viz::declare_field(schema, "current_player",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "first_player",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "winner", viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "terminal", viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "ply", viz::all_public({}, Cfg::kPlayers));

    // ---- public per-player 1D ----
    viz::declare_field(schema, "alive",
                       viz::all_public({Cfg::kPlayers}, Cfg::kPlayers));
    viz::declare_field(schema, "protected_flags",
                       viz::all_public({Cfg::kPlayers}, Cfg::kPlayers));

    // ---- private (owner-only) ----
    // hand[p]: each seat sees only their own hand card. Reveals
    // (Priest peek / Baron / King swap / Princess play) mutate this
    // slot's viz directly via viz::reveal_slot_to / reveal_slot;
    // "hand[p] is exposed to anyone other than p" is reduced from
    // viz["hand"][p, *] at encode/snapshot time — no parallel cache.
    //
    // Who has seen whose hand is common knowledge only with 2 players
    // (kAllViewers there, so a seat's session knows when its own hand
    // has been peeked). With 3+ players a peek by q on p survives p's
    // next play iff p played the drawn card rather than the peeked one —
    // a third seat cannot tell, so shipping it q's bit would leak a fact
    // about p's hand. Those variants keep the receiver's column only.
    viz::declare_field(
        schema, "hand",
        viz::owner_only_first_axis({Cfg::kPlayers}, Cfg::kPlayers),
        NPlayers == 2 ? viz::VizSync::kAllViewers
                      : viz::VizSync::kReceiverOnly);

    // ---- hidden ----
    // drawn_card: scalar holding the card just drawn at top of turn.
    // Held only by current_player. Base hidden; rules will
    // reveal_slot_to(current_player) on draw and reset_to_base on
    // play. Note: drawn_card is also implicitly cleared on Prince
    // discard (drawn-card-as-target), so reset_to_base must be paired
    // with the hand-write that consumes it. kAllViewers: who holds the
    // drawn card (the player to move) is public.
    viz::declare_field(schema, "drawn_card",
                       viz::all_hidden({}, Cfg::kPlayers),
                       viz::VizSync::kAllViewers);
    // set_aside_card: removed from the bottom of the deck at game
    // start, NEVER revealed to anyone. Permanently hidden.
    viz::declare_field(schema, "set_aside_card",
                       viz::all_hidden({}, Cfg::kPlayers));

    // ---- multiset counts (all_public, fixed shape) ----
    // discard_count[N, kCardTypes+1]: per-player public discard
    // multiset. Walker visits every slot; observer reconstructs
    // visual order from the action stream client-side.
    viz::declare_field(
        schema, "discard_count",
        viz::all_public({Cfg::kPlayers, kCardTypes + 1}, Cfg::kPlayers));
    // face_up_count[kCardTypes+1]: 2p-only public count of cards
    // burned face-up at game start. Other variants leave it all-zero.
    viz::declare_field(
        schema, "face_up_count",
        viz::all_public({kCardTypes + 1}, Cfg::kPlayers));
    // deck_count[kCardTypes+1]: per-type counts in the hidden deck.
    // Total (sum) is public — but per-type counts are NOT something
    // every observer can see in the actual game (an observer only
    // knows total minus what they've witnessed), so we declare the
    // total via a separate `deck_size` scalar and treat
    // `deck_count` itself as all_hidden — randomize_unseen fills
    // it from the tracker's information set on each sim.
    viz::declare_field(
        schema, "deck_count",
        viz::all_hidden({kCardTypes + 1}, Cfg::kPlayers));
    viz::declare_field(schema, "deck_size",
                       viz::all_public({}, Cfg::kPlayers));

    return schema;
  }();
  return s;
}

namespace {

// Draw one card from a count-array deck using the supplied rng,
// weighted by remaining count per type. Returns 0 if the deck is
// empty. Decrements the chosen entry. Functionally equivalent to
// "shuffle once + pop_back" — both produce a uniformly distributed
// permutation, the count-array form just integrates the lazy rng.
template <std::size_t Size>
std::int8_t draw_from_count(std::array<std::int8_t, Size>& count,
                            std::mt19937_64& rng) {
  int total = 0;
  for (std::size_t i = 1; i < Size; ++i) total += count[i];
  if (total <= 0) return 0;
  std::uint64_t r = rng();
  int pick = static_cast<int>(r % static_cast<std::uint64_t>(total));
  for (std::size_t i = 1; i < Size; ++i) {
    int n = count[i];
    if (pick < n) {
      count[i] = static_cast<std::int8_t>(n - 1);
      return static_cast<std::int8_t>(i);
    }
    pick -= n;
  }
  return 0;
}

template <int NPlayers>
int deck_total(const LoveLetterData<NPlayers>& d) {
  int n = 0;
  for (int i = 1; i <= kCardTypes; ++i) n += d.deck_count[static_cast<size_t>(i)];
  return n;
}

}  // namespace

template <int NPlayers>
LoveLetterState<NPlayers>::LoveLetterState() {
  reset_with_seed(0xC0FFEEu);
}

template <int NPlayers>
void LoveLetterState<NPlayers>::reset_with_seed(std::uint64_t seed) {
  IGameState::reset_step_count_base();
  auto& d = data;
  d.current_player = 0;
  d.first_player = 0;
  d.winner = -1;
  d.terminal = false;
  d.ply = 0;
  d.hand.fill(0);
  d.drawn_card = 0;
  d.alive.fill(1);
  d.protected_flags.fill(0);
  d.set_aside_card = 0;
  d.face_up_count.fill(0);
  for (int p = 0; p < Cfg::kPlayers; ++p) {
    d.discard_count[static_cast<size_t>(p)].fill(0);
  }
  undo_stack.clear();

  d.deck_count.fill(0);
  for (int card = 1; card <= kCardTypes; ++card) {
    d.deck_count[static_cast<size_t>(card)] =
        static_cast<std::int8_t>(kCardCounts[static_cast<size_t>(card)]);
  }

  // One-shot rng for the initial draws. RNG is not stored on
  // state — caller of do_action_fast supplies its own rng for any
  // subsequent randomness.
  std::mt19937_64 rng(seed);

  d.set_aside_card = draw_from_count(d.deck_count, rng);

  if constexpr (NPlayers == 2) {
    for (int i = 0; i < 3; ++i) {
      std::int8_t c = draw_from_count(d.deck_count, rng);
      d.face_up_count[static_cast<size_t>(c)]++;
    }
  }

  for (int p = 0; p < Cfg::kPlayers; ++p) {
    d.hand[static_cast<size_t>(p)] = draw_from_count(d.deck_count, rng);
  }

  d.drawn_card = draw_from_count(d.deck_count, rng);

  // Initial deck_size: sum after all start-of-game draws (set_aside,
  // 2p face-up trio, opening hands, current_player's drawn_card).
  // Rules update d.deck_size on subsequent draws.
  int total = 0;
  for (int c = 1; c <= kCardTypes; ++c) {
    total += d.deck_count[static_cast<size_t>(c)];
  }
  d.deck_size = static_cast<std::int8_t>(total);

  viz::init_viz(*this, schema());
  // Start-of-game viz reveals (drawn_card → current_player) live in
  // rules.cpp per I1 (rules are sole viz writer); they're invoked
  // by the registrar's make_loveletter wrapper after this returns.
}

template <int NPlayers>
void LoveLetterState<NPlayers>::reseed_viz() {
  viz::init_viz(*this, schema());
}

template <int NPlayers>
StateHash64 LoveLetterState<NPlayers>::state_hash() const {
  // Hand-rolled fallback for callers that don't go through the
  // perspective-aware framework path. Mirrors what the schema walker
  // produces for current_player's view, minus the framework's
  // structural (field_pos, idx) salt — close enough for parity tests
  // that don't compare bit-for-bit with the framework hash.
  const auto& d = data;
  std::size_t h = 0;
  hash_combine(h, static_cast<std::size_t>(d.current_player + 3));
  hash_combine(h, static_cast<std::size_t>(d.first_player + 5));
  hash_combine(h, static_cast<std::size_t>(d.ply + 7));
  hash_combine(h, static_cast<std::size_t>(d.winner + 11));
  hash_combine(h, static_cast<std::size_t>(d.terminal ? 1 : 0));

  for (int p = 0; p < Cfg::kPlayers; ++p) {
    hash_combine(h, static_cast<std::size_t>(d.alive[p] + 13));
    hash_combine(h, static_cast<std::size_t>(d.protected_flags[p] + 17));
    if (p == d.current_player) {
      hash_combine(h, static_cast<std::size_t>(d.hand[p] + 19));
    } else {
      hash_combine(h, static_cast<std::size_t>(0 + 19));
    }
    for (int c = 1; c <= kCardTypes; ++c) {
      hash_combine(h, static_cast<std::size_t>(
          d.discard_count[static_cast<size_t>(p)][static_cast<size_t>(c)] + 23 + c));
    }
  }

  if (d.drawn_card != 0) {
    hash_combine(h, static_cast<std::size_t>(d.drawn_card + 31));
  }

  for (int c = 1; c <= kCardTypes; ++c) {
    hash_combine(h, static_cast<std::size_t>(
        d.face_up_count[static_cast<size_t>(c)] + 47 + c));
  }

  int deck_size = 0;
  for (int c = 1; c <= kCardTypes; ++c) {
    deck_size += d.deck_count[static_cast<size_t>(c)];
  }
  hash_combine(h, static_cast<std::size_t>(deck_size + 37));

  return static_cast<StateHash64>(h);
}

// Field IDs match schema declaration order in schema().
namespace {
enum : viz::FieldId {
  kCurrentPlayer = 0,
  kFirstPlayer,
  kWinner,
  kTerminal,
  kPly,
  kAlive,
  kProtectedFlags,
  kHand,
  kDrawnCard,
  kSetAsideCard,
  kDiscardCount,
  kFaceUpCount,
  kDeckCount,
  kDeckSize,
};
}  // namespace

template <int NPlayers>
void LoveLetterState<NPlayers>::hash_field_slot(
    Hasher& h, viz::FieldId fid,
    const std::vector<int>& idx) const {
  const auto& d = data;
  switch (fid) {
    case kCurrentPlayer:   h.add(d.current_player + 3); return;
    case kFirstPlayer:     h.add(d.first_player + 5); return;
    case kWinner:          h.add(d.winner + 11); return;
    case kTerminal:        h.add(d.terminal ? 1 : 0); return;
    case kPly:             h.add(d.ply + 7); return;
    case kAlive:
      h.add(d.alive[static_cast<size_t>(idx[0])] + 13); return;
    case kProtectedFlags:
      h.add(d.protected_flags[static_cast<size_t>(idx[0])] + 17); return;
    // Owner-only with dynamic reveals (Priest peek / Baron showdown /
    // King swap). Framework already mixed (field_pos, idx[]); here only
    // the value. BUG-037 defended at framework level.
    case kHand:
      h.add(d.hand[static_cast<size_t>(idx[0])] + 19); return;
    // drawn_card: base all_hidden but reveal_slot_to(current_player) on draw.
    case kDrawnCard:       h.add(d.drawn_card + 31); return;
    // set_aside_card: never revealed; walker always emits sentinel — no-op.
    case kSetAsideCard:    return;
    case kDiscardCount:
      h.add(d.discard_count[static_cast<size_t>(idx[0])]
                            [static_cast<size_t>(idx[1])] + 23);
      return;
    case kFaceUpCount:
      h.add(d.face_up_count[static_cast<size_t>(idx[0])] + 47); return;
    // deck_count is all_hidden — walker emits sentinel; never reaches here.
    case kDeckCount:       return;
    // First-class deck_size — do NOT sum d.deck_count: deck_count is
    // hidden, summing forces observers to re-randomize the multiset
    // every ply just to recover this hash value (the bug Plan A surfaced).
    case kDeckSize:        h.add(d.deck_size + 37); return;
  }
}

template <int NPlayers>
std::any LoveLetterState<NPlayers>::read_field_slot(
    viz::FieldId fid, const std::vector<int>& idx) const {
  const auto& d = data;
  switch (fid) {
    case kCurrentPlayer:   return std::any(static_cast<int>(d.current_player));
    case kFirstPlayer:     return std::any(static_cast<int>(d.first_player));
    case kWinner:          return std::any(static_cast<int>(d.winner));
    case kTerminal:        return std::any(static_cast<bool>(d.terminal));
    case kPly:             return std::any(static_cast<int>(d.ply));
    case kAlive:
      return std::any(static_cast<int>(d.alive[static_cast<size_t>(idx[0])]));
    case kProtectedFlags:
      return std::any(static_cast<int>(d.protected_flags[static_cast<size_t>(idx[0])]));
    case kHand:
      return std::any(static_cast<int>(d.hand[static_cast<size_t>(idx[0])]));
    case kDrawnCard:       return std::any(static_cast<int>(d.drawn_card));
    case kSetAsideCard:    return std::any(static_cast<int>(d.set_aside_card));
    case kDiscardCount:
      return std::any(static_cast<int>(
          d.discard_count[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])]));
    case kFaceUpCount:
      return std::any(static_cast<int>(
          d.face_up_count[static_cast<size_t>(idx[0])]));
    case kDeckCount:
      return std::any(static_cast<int>(
          d.deck_count[static_cast<size_t>(idx[0])]));
    case kDeckSize:        return std::any(static_cast<int>(d.deck_size));
  }
  return {};
}

template <int NPlayers>
void LoveLetterState<NPlayers>::write_field_slot(
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
    case kCurrentPlayer:   d.current_player = as_int(); return;
    case kFirstPlayer:     d.first_player = static_cast<std::int8_t>(as_int()); return;
    case kWinner:          d.winner = as_int(); return;
    case kTerminal:        d.terminal = as_bool(); return;
    case kPly:             d.ply = as_int(); return;
    case kAlive:
      d.alive[static_cast<size_t>(idx[0])] = static_cast<std::int8_t>(as_int());
      return;
    case kProtectedFlags:
      d.protected_flags[static_cast<size_t>(idx[0])] = static_cast<std::int8_t>(as_int());
      return;
    case kHand:
      d.hand[static_cast<size_t>(idx[0])] = static_cast<std::int8_t>(as_int());
      return;
    case kDrawnCard:
      d.drawn_card = static_cast<std::int8_t>(as_int()); return;
    case kSetAsideCard:
      d.set_aside_card = static_cast<std::int8_t>(as_int()); return;
    case kDiscardCount:
      d.discard_count[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] =
          static_cast<std::int8_t>(as_int());
      return;
    case kFaceUpCount:
      d.face_up_count[static_cast<size_t>(idx[0])] =
          static_cast<std::int8_t>(as_int());
      return;
    case kDeckCount:
      d.deck_count[static_cast<size_t>(idx[0])] =
          static_cast<std::int8_t>(as_int());
      return;
    case kDeckSize:
      d.deck_size = static_cast<std::int8_t>(as_int()); return;
  }
}

template <int NPlayers>
void LoveLetterState<NPlayers>::mask_field_slot(
    viz::FieldId fid, const std::vector<int>& idx) {
  // Walker reaches here only for slots whose viz can be 0 for some
  // perspective: hand (owner_only), drawn_card / set_aside_card
  // (all_hidden), deck_count (all_hidden).
  switch (fid) {
    case kHand:
      data.hand[static_cast<size_t>(idx[0])] = kPlaceholderInt8;
      return;
    case kDrawnCard:
      data.drawn_card = kPlaceholderInt8;
      return;
    case kSetAsideCard:
      data.set_aside_card = kPlaceholderInt8;
      return;
    case kDeckCount:
      data.deck_count[static_cast<size_t>(idx[0])] = kPlaceholderInt8;
      return;
    default:
      return;
  }
}

template <int NPlayers>
int LoveLetterState<NPlayers>::current_player() const {
  return data.current_player;
}

template <int NPlayers>
int LoveLetterState<NPlayers>::first_player() const {
  return data.first_player;
}

template <int NPlayers>
bool LoveLetterState<NPlayers>::is_terminal() const {
  return data.terminal;
}

template <int NPlayers>
int LoveLetterState<NPlayers>::winner() const {
  return data.winner;
}

template <int NPlayers>
std::vector<int> LoveLetterState<NPlayers>::winners() const {
  const auto& d = data;
  if (!d.terminal) return {};
  if (d.winner >= 0) return {d.winner};
  // winner == -1 only arises from a showdown tie; hands are face up by
  // then (rules reveal them), so this reads public data on any view.
  auto pile_sum = [&](int p) {
    int s = 0;
    for (int c = 1; c <= kCardTypes; ++c) {
      s += c * d.discard_count[static_cast<size_t>(p)][static_cast<size_t>(c)];
    }
    return s;
  };
  int best_card = -1;
  int best_sum = -1;
  for (int p = 0; p < NPlayers; ++p) {
    if (!d.alive[p]) continue;
    const int card = d.hand[p];
    const int sum = pile_sum(p);
    if (card > best_card || (card == best_card && sum > best_sum)) {
      best_card = card;
      best_sum = sum;
    }
  }
  std::vector<int> result;
  for (int p = 0; p < NPlayers; ++p) {
    if (d.alive[p] && d.hand[p] == best_card && pile_sum(p) == best_sum) {
      result.push_back(p);
    }
  }
  return result;
}

template <int NPlayers>
std::vector<int> LoveLetterState<NPlayers>::losers() const {
  if (!data.terminal) return {};
  const std::vector<int> won = winners();
  std::vector<int> result;
  for (int p = 0; p < NPlayers; ++p) {
    if (std::find(won.begin(), won.end(), p) == won.end()) result.push_back(p);
  }
  return result;
}

template struct LoveLetterData<2>;
template struct LoveLetterData<3>;
template struct LoveLetterData<4>;
template struct LoveLetterState<2>;
template struct LoveLetterState<3>;
template struct LoveLetterState<4>;

}  // namespace board_ai::loveletter
