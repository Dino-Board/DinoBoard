#include "coup_state.h"

#include <cstdint>
#include <random>
#include <stdexcept>

#include "../../engine/core/masked_state.h"
#include "../../engine/core/schema_hash.h"
#include "../../engine/core/viz_runtime.h"

namespace board_ai::coup {

namespace {

// Draw one card from a count-array deck, weighted by remaining count
// per character. Returns -1 if the deck is empty. Decrements both the
// chosen entry and `deck_size`. Mirrors LL's `draw_from_count`.
template <std::size_t Size>
CharId draw_from_count(std::array<std::int8_t, Size>& count,
                       std::int8_t& deck_size,
                       std::mt19937_64& rng) {
  int total = 0;
  for (std::size_t i = 0; i < Size; ++i) total += count[i];
  if (total <= 0) return -1;
  std::uint64_t r = rng();
  int pick = static_cast<int>(r % static_cast<std::uint64_t>(total));
  for (std::size_t i = 0; i < Size; ++i) {
    int n = count[i];
    if (pick < n) {
      count[i] = static_cast<std::int8_t>(n - 1);
      deck_size = static_cast<std::int8_t>(deck_size - 1);
      return static_cast<CharId>(i);
    }
    pick -= n;
  }
  return -1;
}

}  // namespace

template <int NPlayers>
const viz::VisibilitySchema& CoupState<NPlayers>::schema() {
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
    viz::declare_field(schema, "stage", viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "active_player",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "declared_action",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "action_target",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "claimed_character",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "challenger",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "challenge_loser",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "action_challenged",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "action_challenge_succeeded",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "blocker", viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "block_character",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "counter_challenged",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "counter_challenge_succeeded",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "challenge_check_index",
                       viz::all_public({}, Cfg::kPlayers));
    viz::declare_field(schema, "exchange_held_count",
                       viz::all_public({}, Cfg::kPlayers));
    // Public scalar count of court deck size — analogous to LL's
    // deck_size. Per-character contents (deck_count) are all_hidden;
    // the size is what every observer can see in the physical game.
    viz::declare_field(schema, "deck_size",
                       viz::all_public({}, Cfg::kPlayers));

    // ---- public per-player 1D ----
    viz::declare_field(schema, "coins",
                       viz::all_public({Cfg::kPlayers}, Cfg::kPlayers));
    viz::declare_field(schema, "alive",
                       viz::all_public({Cfg::kPlayers}, Cfg::kPlayers));
    // revealed[p][s]: public face-up flag per influence slot. all_public
    // by definition — it's the "is this card now face-up?" public bit.
    viz::declare_field(
        schema, "revealed",
        viz::all_public({Cfg::kPlayers, kInfluencePerPlayer}, Cfg::kPlayers));

    // ---- private per-player ----
    // influence[N][2]: face-down cards. Owner-only base; rules call
    // viz::reveal_slot(state, "influence", {p, s}) when a card flips
    // public (challenge / coup / assassinate / lose-influence). Walker
    // dispatches by viz directly — downstream consumers don't gate on
    // revealed[].
    // kAllViewers: the pattern only ever changes by a public flip
    // (lose influence) or a reveal-and-redraw inside one action, so every
    // seat can derive it — a seat's search must know that the others see
    // its face-up cards.
    viz::declare_field(
        schema, "influence",
        viz::owner_only_first_axis({Cfg::kPlayers, kInfluencePerPlayer},
                                   Cfg::kPlayers),
        viz::VizSync::kAllViewers);

    // ---- hidden ----
    // exchange_drawn[2]: two cards drawn from court_deck during the
    // Exchange action, visible only to active_player. Base is hidden;
    // rules reveal_slot_to(active_player) on draw and reset_to_base on
    // return-to-deck. Stays kReceiverOnly: WHICH slot is reset first
    // tells whether the first returned card was a drawn one — private
    // to the exchanging player.
    viz::declare_field(schema, "exchange_drawn",
                       viz::all_hidden({kExchangeDrawSlots}, Cfg::kPlayers));
    // deck_count[kCharacterCount]: per-character counts in the hidden
    // deck. The size is public (deck_size), per-character contents are
    // not — randomize_unseen fills from the tracker's information set
    // on each sim entry.
    viz::declare_field(
        schema, "deck_count",
        viz::all_hidden({kCharacterCount}, Cfg::kPlayers));

    // ---- private per-player, never revealed ----
    // Each seat's certain knowledge of the court deck (CoupData). Rules
    // maintain it; randomize_unseen reads the observer's row to keep
    // sampled worlds consistent and clears the other rows.
    viz::declare_field(
        schema, "deck_known",
        viz::owner_only_first_axis({Cfg::kPlayers, kCharacterCount},
                                   Cfg::kPlayers));
    viz::declare_field(
        schema, "deck_known_taint",
        viz::owner_only_first_axis(
            {Cfg::kPlayers, Cfg::kPlayers, kInfluencePerPlayer},
            Cfg::kPlayers));

    return schema;
  }();
  return s;
}

template <int NPlayers>
CoupState<NPlayers>::CoupState() = default;

template <int NPlayers>
void CoupState<NPlayers>::reset_with_seed(std::uint64_t seed) {
  IGameState::reset_step_count_base();
  data = CoupData<NPlayers>{};

  for (CharId c = 0; c < kCharacterCount; ++c) {
    data.deck_count[static_cast<size_t>(c)] =
        static_cast<std::int8_t>(kCardsPerCharacter);
  }
  data.deck_size = static_cast<std::int8_t>(kTotalCards);

  // One-shot rng for opening 2-cards-per-player deal. RNG is not stored
  // on state — caller of do_action_fast supplies its own rng for any
  // subsequent randomness (challenge redraws / exchange).
  {
    std::mt19937_64 rng(seed);
    for (int p = 0; p < NPlayers; ++p) {
      data.influence[p][0] = draw_from_count(data.deck_count, data.deck_size, rng);
      data.influence[p][1] = draw_from_count(data.deck_count, data.deck_size, rng);
    }
  }
  for (int p = 0; p < NPlayers; ++p) {
    data.revealed[p] = {false, false};
    data.coins[p] = kStartingCoins;
    data.alive[p] = true;
  }

  // Sentinel -1 = "no card here" for exchange_drawn.
  data.exchange_drawn = {-1, -1};
  data.exchange_held_count = 0;

  data.current_player = 0;
  data.first_player = 0;
  data.active_player = 0;
  data.stage = CoupStage::kDeclareAction;

  viz::init_viz(*this, schema());
}

template <int NPlayers>
StateHash64 CoupState<NPlayers>::state_hash() const {
  // Single canonical path: framework's perspective-aware hash for the
  // current player's information set. Mirrors LL after BUG-037 cleanup.
  return this->state_hash_for_perspective(data.current_player);
}

// Field IDs match schema declaration order in schema().
namespace {
enum : viz::FieldId {
  kCurrentPlayer = 0,
  kFirstPlayer,
  kWinner,
  kTerminal,
  kPly,
  kStage,
  kActivePlayer,
  kDeclaredAction,
  kActionTarget,
  kClaimedCharacter,
  kChallenger,
  kChallengeLoser,
  kActionChallenged,
  kActionChallengeSucceeded,
  kBlocker,
  kBlockCharacter,
  kCounterChallenged,
  kCounterChallengeSucceeded,
  kChallengeCheckIndex,
  kExchangeHeldCount,
  kDeckSize,
  kCoins,
  kAlive,
  kRevealed,
  kInfluence,
  kExchangeDrawn,
  kDeckCount,
  kDeckKnown,
  kDeckKnownTaint,
};
}  // namespace

// Per-field bulk hash override for Coup's array fields. Coup has hidden
// info (influence: owner_only_first_axis, exchange_drawn / deck_count:
// all_hidden), so viz checks matter for those fields. The override
// eliminates per-slot virtual dispatch + idx vector construction.
template <int NPlayers>
void CoupState<NPlayers>::hash_field_bulk(
    Hasher& h, viz::FieldId fid,
    const viz::VizTensor& v, int perspective) const {
  const auto& d = data;
  switch (fid) {
    case kRevealed: {
      // 2D: [kPlayers][kInfluencePerPlayer], all_public.
      for (int p = 0; p < Cfg::kPlayers; ++p) {
        for (int s = 0; s < kInfluencePerPlayer; ++s) {
          h.add(static_cast<std::size_t>(fid));
          h.add(p);
          h.add(s);
          h.add(d.revealed[static_cast<size_t>(p)]
                           [static_cast<size_t>(s)] ? 1 : 0);
        }
      }
      return;
    }
    case kInfluence: {
      // 2D: [kPlayers][kInfluencePerPlayer], owner_only_first_axis with
      // dynamic reveals. viz shape: [kPlayers, 2, kPlayers].
      for (int p = 0; p < Cfg::kPlayers; ++p) {
        for (int s = 0; s < kInfluencePerPlayer; ++s) {
          h.add(static_cast<std::size_t>(fid));
          h.add(p);
          h.add(s);
          const std::size_t flat =
              (static_cast<std::size_t>(p) * kInfluencePerPlayer +
               static_cast<std::size_t>(s)) *
                  static_cast<std::size_t>(Cfg::kPlayers) +
              static_cast<std::size_t>(perspective);
          if (v.data[flat]) {
            h.add(d.influence[static_cast<size_t>(p)]
                             [static_cast<size_t>(s)] + 100);
          } else {
            h.combine(kHiddenHashSentinel);
          }
        }
      }
      return;
    }
    case kExchangeDrawn: {
      // 1D: [kExchangeDrawSlots], all_hidden base but rules may
      // reveal_slot_to(active_player). viz shape: [2, kPlayers].
      for (int i = 0; i < kExchangeDrawSlots; ++i) {
        h.add(static_cast<std::size_t>(fid));
        h.add(i);
        const std::size_t flat =
            static_cast<std::size_t>(i) *
                static_cast<std::size_t>(Cfg::kPlayers) +
            static_cast<std::size_t>(perspective);
        if (v.data[flat]) {
          h.add(d.exchange_drawn[static_cast<size_t>(i)] + 200);
        } else {
          h.combine(kHiddenHashSentinel);
        }
      }
      return;
    }
    case kDeckCount: {
      // 1D: [kCharacterCount], all_hidden base. viz shape: [5, kPlayers].
      for (int c = 0; c < kCharacterCount; ++c) {
        h.add(static_cast<std::size_t>(fid));
        h.add(c);
        const std::size_t flat =
            static_cast<std::size_t>(c) *
                static_cast<std::size_t>(Cfg::kPlayers) +
            static_cast<std::size_t>(perspective);
        if (v.data[flat]) {
          h.add(d.deck_count[static_cast<size_t>(c)] + 50);
        } else {
          h.combine(kHiddenHashSentinel);
        }
      }
      return;
    }
    case kCoins: {
      // 1D: [kPlayers], all_public.
      for (int p = 0; p < Cfg::kPlayers; ++p) {
        h.add(static_cast<std::size_t>(fid));
        h.add(p);
        h.add(d.coins[static_cast<size_t>(p)]);
      }
      return;
    }
    case kAlive: {
      // 1D: [kPlayers], all_public.
      for (int p = 0; p < Cfg::kPlayers; ++p) {
        h.add(static_cast<std::size_t>(fid));
        h.add(p);
        h.add(d.alive[static_cast<size_t>(p)] ? 1 : 0);
      }
      return;
    }
    default:
      IGameState::hash_field_bulk(h, fid, v, perspective);
      return;
  }
}

template <int NPlayers>
void CoupState<NPlayers>::hash_field_slot(
    Hasher& h, viz::FieldId fid,
    const std::vector<int>& idx) const {
  const auto& d = data;
  switch (fid) {
    case kCurrentPlayer:               h.add(d.current_player); return;
    case kFirstPlayer:                 h.add(static_cast<int>(d.first_player)); return;
    case kWinner:                      h.add(d.winner + 1); return;
    case kTerminal:                    h.add(d.terminal ? 1 : 0); return;
    case kPly:                         h.add(d.ply); return;
    case kStage:                       h.add(static_cast<int>(d.stage)); return;
    case kActivePlayer:                h.add(d.active_player); return;
    case kDeclaredAction:              h.add(d.declared_action + 1); return;
    case kActionTarget:                h.add(d.action_target + 1); return;
    case kClaimedCharacter:            h.add(d.claimed_character + 1); return;
    case kChallenger:                  h.add(d.challenger + 1); return;
    case kChallengeLoser:              h.add(d.challenge_loser + 1); return;
    case kActionChallenged:            h.add(d.action_challenged ? 1 : 0); return;
    case kActionChallengeSucceeded:    h.add(d.action_challenge_succeeded ? 1 : 0); return;
    case kBlocker:                     h.add(d.blocker + 1); return;
    case kBlockCharacter:              h.add(d.block_character + 1); return;
    case kCounterChallenged:           h.add(d.counter_challenged ? 1 : 0); return;
    case kCounterChallengeSucceeded:   h.add(d.counter_challenge_succeeded ? 1 : 0); return;
    case kChallengeCheckIndex:         h.add(d.challenge_check_index); return;
    case kExchangeHeldCount:           h.add(d.exchange_held_count); return;
    case kDeckSize:                    h.add(d.deck_size + 37); return;
    case kCoins:
      h.add(d.coins[static_cast<size_t>(idx[0])]); return;
    case kAlive:
      h.add(d.alive[static_cast<size_t>(idx[0])] ? 1 : 0); return;
    case kRevealed:
      h.add(d.revealed[static_cast<size_t>(idx[0])]
                      [static_cast<size_t>(idx[1])] ? 1 : 0);
      return;
    // Owner-only with dynamic reveals (rules call viz::reveal_slot on
    // lose-influence / challenge-loss). Walker decides visibility.
    case kInfluence:
      h.add(d.influence[static_cast<size_t>(idx[0])]
                       [static_cast<size_t>(idx[1])] + 100);
      return;
    // exchange_drawn: schema-base all_hidden, but rules
    // reveal_slot_to(active_player) on draw.
    case kExchangeDrawn:
      h.add(d.exchange_drawn[static_cast<size_t>(idx[0])] + 200);
      return;
    // deck_count: all_hidden; randomize_unseen fills before sim hashing.
    case kDeckCount:
      h.add(d.deck_count[static_cast<size_t>(idx[0])] + 50);
      return;
    case kDeckKnown:
      h.add(d.deck_known[static_cast<size_t>(idx[0])]
                        [static_cast<size_t>(idx[1])] + 300);
      return;
    case kDeckKnownTaint:
      h.add(d.deck_known_taint[static_cast<size_t>(idx[0])]
                              [static_cast<size_t>(idx[1])]
                              [static_cast<size_t>(idx[2])] + 400);
      return;
  }
}

template <int NPlayers>
std::any CoupState<NPlayers>::read_field_slot(
    viz::FieldId fid, const std::vector<int>& idx) const {
  const auto& d = data;
  switch (fid) {
    case kCurrentPlayer:               return std::any(static_cast<int>(d.current_player));
    case kFirstPlayer:                 return std::any(static_cast<int>(d.first_player));
    case kWinner:                      return std::any(static_cast<int>(d.winner));
    case kTerminal:                    return std::any(static_cast<bool>(d.terminal));
    case kPly:                         return std::any(static_cast<int>(d.ply));
    case kStage:                       return std::any(static_cast<int>(d.stage));
    case kActivePlayer:                return std::any(static_cast<int>(d.active_player));
    case kDeclaredAction:              return std::any(static_cast<int>(d.declared_action));
    case kActionTarget:                return std::any(static_cast<int>(d.action_target));
    case kClaimedCharacter:            return std::any(static_cast<int>(d.claimed_character));
    case kChallenger:                  return std::any(static_cast<int>(d.challenger));
    case kChallengeLoser:              return std::any(static_cast<int>(d.challenge_loser));
    case kActionChallenged:            return std::any(static_cast<bool>(d.action_challenged));
    case kActionChallengeSucceeded:    return std::any(static_cast<bool>(d.action_challenge_succeeded));
    case kBlocker:                     return std::any(static_cast<int>(d.blocker));
    case kBlockCharacter:              return std::any(static_cast<int>(d.block_character));
    case kCounterChallenged:           return std::any(static_cast<bool>(d.counter_challenged));
    case kCounterChallengeSucceeded:   return std::any(static_cast<bool>(d.counter_challenge_succeeded));
    case kChallengeCheckIndex:         return std::any(static_cast<int>(d.challenge_check_index));
    case kExchangeHeldCount:           return std::any(static_cast<int>(d.exchange_held_count));
    case kDeckSize:                    return std::any(static_cast<int>(d.deck_size));
    case kCoins:
      return std::any(static_cast<int>(d.coins[static_cast<size_t>(idx[0])]));
    case kAlive:
      return std::any(static_cast<bool>(d.alive[static_cast<size_t>(idx[0])]));
    case kRevealed:
      return std::any(static_cast<bool>(
          d.revealed[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])]));
    case kInfluence:
      return std::any(static_cast<int>(
          d.influence[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])]));
    case kExchangeDrawn:
      return std::any(static_cast<int>(
          d.exchange_drawn[static_cast<size_t>(idx[0])]));
    case kDeckCount:
      return std::any(static_cast<int>(
          d.deck_count[static_cast<size_t>(idx[0])]));
    case kDeckKnown:
      return std::any(static_cast<int>(
          d.deck_known[static_cast<size_t>(idx[0])]
                      [static_cast<size_t>(idx[1])]));
    case kDeckKnownTaint:
      return std::any(static_cast<int>(
          d.deck_known_taint[static_cast<size_t>(idx[0])]
                            [static_cast<size_t>(idx[1])]
                            [static_cast<size_t>(idx[2])]));
  }
  return {};
}

template <int NPlayers>
void CoupState<NPlayers>::write_field_slot(
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
    case kCurrentPlayer:               d.current_player = as_int(); return;
    case kFirstPlayer:                 d.first_player = static_cast<std::int8_t>(as_int()); return;
    case kWinner:                      d.winner = as_int(); return;
    case kTerminal:                    d.terminal = as_bool(); return;
    case kPly:                         d.ply = as_int(); return;
    case kStage:                       d.stage = static_cast<CoupStage>(as_int()); return;
    case kActivePlayer:                d.active_player = as_int(); return;
    case kDeclaredAction:              d.declared_action = as_int(); return;
    case kActionTarget:                d.action_target = as_int(); return;
    case kClaimedCharacter:            d.claimed_character = static_cast<CharId>(as_int()); return;
    case kChallenger:                  d.challenger = as_int(); return;
    case kChallengeLoser:              d.challenge_loser = as_int(); return;
    case kActionChallenged:            d.action_challenged = as_bool(); return;
    case kActionChallengeSucceeded:    d.action_challenge_succeeded = as_bool(); return;
    case kBlocker:                     d.blocker = as_int(); return;
    case kBlockCharacter:              d.block_character = static_cast<CharId>(as_int()); return;
    case kCounterChallenged:           d.counter_challenged = as_bool(); return;
    case kCounterChallengeSucceeded:   d.counter_challenge_succeeded = as_bool(); return;
    case kChallengeCheckIndex:         d.challenge_check_index = as_int(); return;
    case kExchangeHeldCount:           d.exchange_held_count = as_int(); return;
    case kDeckSize:                    d.deck_size = static_cast<std::int8_t>(as_int()); return;
    case kCoins:
      d.coins[static_cast<size_t>(idx[0])] = as_int(); return;
    case kAlive:
      d.alive[static_cast<size_t>(idx[0])] = as_bool(); return;
    case kRevealed:
      d.revealed[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] = as_bool();
      return;
    case kInfluence:
      d.influence[static_cast<size_t>(idx[0])][static_cast<size_t>(idx[1])] =
          static_cast<CharId>(as_int());
      return;
    case kExchangeDrawn:
      d.exchange_drawn[static_cast<size_t>(idx[0])] =
          static_cast<CharId>(as_int());
      return;
    case kDeckCount:
      d.deck_count[static_cast<size_t>(idx[0])] =
          static_cast<std::int8_t>(as_int());
      return;
    case kDeckKnown:
      d.deck_known[static_cast<size_t>(idx[0])]
                  [static_cast<size_t>(idx[1])] =
          static_cast<std::int8_t>(as_int());
      return;
    case kDeckKnownTaint:
      d.deck_known_taint[static_cast<size_t>(idx[0])]
                        [static_cast<size_t>(idx[1])]
                        [static_cast<size_t>(idx[2])] =
          static_cast<std::int8_t>(as_int());
      return;
  }
}

template <int NPlayers>
void CoupState<NPlayers>::mask_field_slot(
    viz::FieldId fid, const std::vector<int>& idx) {
  // Walker only reaches here for slots whose viz is 0 for the
  // perspective: influence (owner_only), exchange_drawn / deck_count
  // (all_hidden).
  switch (fid) {
    case kInfluence:
      data.influence[static_cast<size_t>(idx[0])]
                    [static_cast<size_t>(idx[1])] = kPlaceholderInt8;
      return;
    case kExchangeDrawn:
      data.exchange_drawn[static_cast<size_t>(idx[0])] = kPlaceholderInt8;
      return;
    case kDeckCount:
      data.deck_count[static_cast<size_t>(idx[0])] = kPlaceholderInt8;
      return;
    case kDeckKnown:
      data.deck_known[static_cast<size_t>(idx[0])]
                     [static_cast<size_t>(idx[1])] = kPlaceholderInt8;
      return;
    case kDeckKnownTaint:
      data.deck_known_taint[static_cast<size_t>(idx[0])]
                           [static_cast<size_t>(idx[1])]
                           [static_cast<size_t>(idx[2])] = kPlaceholderInt8;
      return;
    default:
      return;
  }
}

template <int NPlayers>
int CoupState<NPlayers>::current_player() const {
  return data.current_player;
}

template <int NPlayers>
int CoupState<NPlayers>::first_player() const {
  return data.first_player;
}

template <int NPlayers>
bool CoupState<NPlayers>::is_terminal() const {
  return data.terminal;
}

template <int NPlayers>
int CoupState<NPlayers>::winner() const {
  return data.winner;
}

template <int NPlayers>
bool CoupState<NPlayers>::is_turn_start() const {
  return data.stage == CoupStage::kDeclareAction;
}

template struct CoupData<2>;
template struct CoupData<3>;
template struct CoupData<4>;
template struct CoupState<2>;
template struct CoupState<3>;
template struct CoupState<4>;

}  // namespace board_ai::coup
