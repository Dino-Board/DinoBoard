#include "harmonies_state.h"

#include "harmonies_rules.h"

#include <algorithm>
#include <numeric>
#include <random>

#include "../../engine/core/schema_hash.h"
#include "../../engine/core/viz_runtime.h"

namespace board_ai::harmonies {

// ---- Tower transition table ----
// rows: tower_type 0..12. cols: color-1 (0=Blue ... 5=Red).
// Encodes "what tower do I become if I add `color` to my top".
// Layer-2 stacking restriction (Red can only sit at layer 0 or 1) is
// already baked in: any "+Red" entry that would land at layer 2 is -1.
const std::array<std::array<std::int8_t, kColors>, kTowerTypes> kTowerTransition = {{
    // Blue Gray Brown Green Yellow Red
    {{ 1,   2,   3,   4,    5,    6 }},   // 0 = empty
    {{-1,  -1,  -1,  -1,   -1,   -1 }},   // 1 = single Blue (water): no stacking
    {{-1,  10,  -1,  -1,   -1,    7 }},   // 2 = single Gray (mountain1): +Gray->mountain2, +Red->house
    {{-1,  -1,   8,   9,   -1,    7 }},   // 3 = single Brown (wood): +Brown->wood2, +Green->tree2, +Red->house
    {{-1,  -1,  -1,  -1,   -1,   -1 }},   // 4 = single Green (tree1): top is leaf
    {{-1,  -1,  -1,  -1,   -1,   -1 }},   // 5 = single Yellow (field): no stacking
    {{-1,  -1,  -1,  -1,   -1,    7 }},   // 6 = single Red (house1): +Red layer1 -> house2 (stays type 7)
    {{-1,  -1,  -1,  -1,   -1,   -1 }},   // 7 = house2: full
    {{-1,  -1,  -1,  11,   -1,   -1 }},   // 8 = wood2: +Green -> tree3
    {{-1,  -1,  -1,  -1,   -1,   -1 }},   // 9 = tree2: top is leaf
    {{-1,  12,  -1,  -1,   -1,   -1 }},   // 10 = mountain2: +Gray -> mountain3
    {{-1,  -1,  -1,  -1,   -1,   -1 }},   // 11 = tree3: full
    {{-1,  -1,  -1,  -1,   -1,   -1 }},   // 12 = mountain3: full
}};

// ---- Schema ----

template <int NPlayers, int Side>
const viz::VisibilitySchema& HarmoniesState<NPlayers, Side>::schema() {
  static const viz::VisibilitySchema s = []() {
    viz::VisibilitySchema schema;
    schema.n_players = NPlayers;

    // Order MUST match FieldId enum below.
    viz::declare_field(schema, "current_player",        viz::all_public({}, NPlayers));
    viz::declare_field(schema, "game_first_player",     viz::all_public({}, NPlayers));
    viz::declare_field(schema, "winner",                viz::all_public({}, NPlayers));
    viz::declare_field(schema, "terminal",              viz::all_public({}, NPlayers));
    viz::declare_field(schema, "end_triggered",         viz::all_public({}, NPlayers));
    viz::declare_field(schema, "turn_started",          viz::all_public({}, NPlayers));
    viz::declare_field(schema, "took_card_this_turn",   viz::all_public({}, NPlayers));
    viz::declare_field(schema, "selected_tokens",       viz::all_public({kColors}, NPlayers));
    viz::declare_field(schema, "central_board_counts",  viz::all_public({kSlots, kColors}, NPlayers));
    viz::declare_field(schema, "animal_market",         viz::all_public({kSlots}, NPlayers));
    viz::declare_field(schema, "animal_deck",           viz::all_hidden({kAnimalCards}, NPlayers));
    viz::declare_field(schema, "animal_deck_count",     viz::all_public({}, NPlayers));
    viz::declare_field(schema, "pouch_counts",          viz::all_public({kColors}, NPlayers));
    viz::declare_field(schema, "tower_type",            viz::all_public({NPlayers, kHexes}, NPlayers));
    viz::declare_field(schema, "has_animal",            viz::all_public({NPlayers, kHexes}, NPlayers));
    viz::declare_field(schema, "held_cards",            viz::all_public({NPlayers, kHandSize}, NPlayers));
    viz::declare_field(schema, "cube_remaining",        viz::all_public({NPlayers, kHandSize}, NPlayers));
    viz::declare_field(schema, "cube_initial",          viz::all_public({NPlayers, kHandSize}, NPlayers));
    viz::declare_field(schema, "space_remaining",       viz::all_public({NPlayers}, NPlayers));
    viz::declare_field(schema, "animal_scores",         viz::all_public({NPlayers}, NPlayers));
    viz::declare_field(schema, "landscape_scores",      viz::all_public({NPlayers}, NPlayers));
    viz::declare_field(schema, "completed_cards",       viz::all_public({kAnimalCards}, NPlayers));
    return schema;
  }();
  return s;
}

namespace {
enum : viz::FieldId {
  kFidCurrentPlayer = 0,
  kFidGameFirstPlayer,
  kFidWinner,
  kFidTerminal,
  kFidEndTriggered,
  kFidTurnStarted,
  kFidTookCardThisTurn,
  kFidSelectedTokens,
  kFidCentralBoardCounts,
  kFidAnimalMarket,
  kFidAnimalDeck,
  kFidAnimalDeckCount,
  kFidPouchCounts,
  kFidTowerType,
  kFidHasAnimal,
  kFidHeldCards,
  kFidCubeRemaining,
  kFidCubeInitial,
  kFidSpaceRemaining,
  kFidAnimalScores,
  kFidLandscapeScores,
  kFidCompletedCards,
};
}  // namespace

template <int NPlayers, int Side>
HarmoniesState<NPlayers, Side>::HarmoniesState() {
  reset_with_seed(0xC0FFEEu);
}

template <int NPlayers, int Side>
int HarmoniesState<NPlayers, Side>::tower_height(int tt) {
  switch (tt) {
    case 0: return 0;
    case 1: case 2: case 3: case 4: case 5: case 6: return 1;
    case 7: case 8: case 9: case 10: return 2;
    case 11: case 12: return 3;
  }
  return 0;
}

template <int NPlayers, int Side>
void HarmoniesState<NPlayers, Side>::reset_with_seed(std::uint64_t seed) {
  IGameState::reset_step_count_base();
  std::mt19937_64 rng(seed);

  current_player_ = 0;
  game_first_player_ = 0;
  winner_ = -1;
  terminal = false;
  end_triggered = false;
  turn_started = false;
  took_card_this_turn = false;
  selected_tokens.fill(0);

  // Boards.
  for (int p = 0; p < NPlayers; ++p) {
    tower_type[p].fill(0);
    has_animal[p].fill(0);
    held_cards[p].fill(0);
    cube_remaining[p].fill(0);
    cube_initial[p].fill(0);
    space_remaining[p] = static_cast<std::int8_t>(kHexes);
    animal_scores[p] = 0;
    // Not 0 on side B: an empty board is one island.
    landscape_scores[p] = HarmoniesRules<NPlayers, Side>::landscape_breakdown(tower_type[p]).total();
  }

  // Pouch.
  for (int c = 0; c < kColors; ++c) pouch_counts[c] = kInitialPouchCounts[c];

  // Animal deck: Fisher-Yates permutation of 1..32 driven by `rng`. We
  // avoid std::shuffle (banned by framework lint §2.3) and do the swaps
  // inline against the caller-derived rng.
  std::array<int, kAnimalCards> deck_order{};
  std::iota(deck_order.begin(), deck_order.end(), 1);  // ids 1..32
  for (int i = kAnimalCards - 1; i > 0; --i) {
    const int j = static_cast<int>(rng() % static_cast<std::uint64_t>(i + 1));
    std::swap(deck_order[i], deck_order[j]);
  }
  animal_deck.fill(0);
  completed_cards.fill(0);
  for (int s = 0; s < kSlots; ++s) {
    animal_market[s] = static_cast<std::int8_t>(deck_order[s]);
  }
  for (int i = kSlots; i < kAnimalCards; ++i) {
    animal_deck[deck_order[i] - 1] = 1;
  }
  animal_deck_count = kAnimalCards - kSlots;

  // Central board: each of 5 slots gets 3 tokens drawn weighted by pouch.
  central_board_counts = {};
  for (int s = 0; s < kSlots; ++s) {
    for (int t = 0; t < kMaxTokensPerSlot; ++t) {
      int total = 0;
      for (int c = 0; c < kColors; ++c) total += pouch_counts[c];
      if (total == 0) break;
      int pick = static_cast<int>(rng() % static_cast<std::uint64_t>(total));
      for (int c = 0; c < kColors; ++c) {
        if (pick < pouch_counts[c]) {
          --pouch_counts[c];
          ++central_board_counts[s][c];
          break;
        }
        pick -= pouch_counts[c];
      }
    }
  }

  viz::init_viz(*this, schema());
}

// ---- Final ranking ----
template <int NPlayers, int Side>
std::vector<int> HarmoniesState<NPlayers, Side>::first_place() const {
  // Key: total points, then animal cubes placed (rulebook tie-break).
  auto cubes_of = [&](int p) {
    int n = 0;
    for (auto v : has_animal[p]) n += v ? 1 : 0;
    return n;
  };
  int best_total = -1;
  int best_cubes = -1;
  for (int p = 0; p < NPlayers; ++p) {
    const int total = animal_scores[p] + landscape_scores[p];
    const int cubes = cubes_of(p);
    if (total > best_total || (total == best_total && cubes > best_cubes)) {
      best_total = total;
      best_cubes = cubes;
    }
  }
  std::vector<int> out;
  for (int p = 0; p < NPlayers; ++p) {
    if (animal_scores[p] + landscape_scores[p] == best_total &&
        cubes_of(p) == best_cubes) {
      out.push_back(p);
    }
  }
  return out;
}

template <int NPlayers, int Side>
std::vector<int> HarmoniesState<NPlayers, Side>::winners() const {
  if (!terminal) return {};
  std::vector<int> w = first_place();
  if (static_cast<int>(w.size()) == NPlayers) return {};  // everyone tied: draw
  return w;
}

template <int NPlayers, int Side>
std::vector<int> HarmoniesState<NPlayers, Side>::losers() const {
  const std::vector<int> w = winners();
  if (w.empty()) return {};
  std::vector<int> out;
  for (int p = 0; p < NPlayers; ++p) {
    if (std::find(w.begin(), w.end(), p) == w.end()) out.push_back(p);
  }
  return out;
}

// ---- state_hash (legacy / non-ISMCTS path) ----
template <int NPlayers, int Side>
StateHash64 HarmoniesState<NPlayers, Side>::state_hash() const {
  std::size_t h = 0;
  hash_combine(h, static_cast<std::size_t>(current_player_));
  hash_combine(h, static_cast<std::size_t>(game_first_player_));
  hash_combine(h, static_cast<std::size_t>(winner_ + 1));
  hash_combine(h, static_cast<std::size_t>(terminal ? 1 : 0));
  hash_combine(h, static_cast<std::size_t>(end_triggered ? 1 : 0));
  hash_combine(h, static_cast<std::size_t>(turn_started ? 1 : 0));
  hash_combine(h, static_cast<std::size_t>(took_card_this_turn ? 1 : 0));
  for (auto v : selected_tokens) hash_combine(h, static_cast<std::size_t>(v));
  for (const auto& slot : central_board_counts)
    for (int v : slot) hash_combine(h, static_cast<std::size_t>(v));
  for (auto v : animal_market) hash_combine(h, static_cast<std::size_t>(v));
  for (auto v : animal_deck) hash_combine(h, static_cast<std::size_t>(v));
  hash_combine(h, static_cast<std::size_t>(animal_deck_count));
  for (auto v : completed_cards) hash_combine(h, static_cast<std::size_t>(v));
  for (int v : pouch_counts) hash_combine(h, static_cast<std::size_t>(v));
  for (int p = 0; p < NPlayers; ++p) {
    for (auto v : tower_type[p]) hash_combine(h, static_cast<std::size_t>(v));
    for (auto v : has_animal[p]) hash_combine(h, static_cast<std::size_t>(v));
    for (auto v : held_cards[p]) hash_combine(h, static_cast<std::size_t>(v));
    for (auto v : cube_remaining[p]) hash_combine(h, static_cast<std::size_t>(v));
    for (auto v : cube_initial[p]) hash_combine(h, static_cast<std::size_t>(v));
    hash_combine(h, static_cast<std::size_t>(space_remaining[p]));
    hash_combine(h, static_cast<std::size_t>(animal_scores[p]));
    hash_combine(h, static_cast<std::size_t>(landscape_scores[p]));
  }
  return static_cast<StateHash64>(h);
}

template <int NPlayers, int Side>
void HarmoniesState<NPlayers, Side>::hash_field_slot(
    Hasher& h, viz::FieldId fid,
    const std::vector<int>& idx) const {
  switch (fid) {
    case kFidCurrentPlayer:      h.add(current_player_); return;
    case kFidGameFirstPlayer:    h.add(game_first_player_); return;
    case kFidWinner:             h.add(winner_ + 1); return;
    case kFidTerminal:           h.add(terminal ? 1 : 0); return;
    case kFidEndTriggered:       h.add(end_triggered ? 1 : 0); return;
    case kFidTurnStarted:        h.add(turn_started ? 1 : 0); return;
    case kFidTookCardThisTurn:   h.add(took_card_this_turn ? 1 : 0); return;
    case kFidSelectedTokens:
      h.add(static_cast<int>(selected_tokens[idx[0]])); return;
    case kFidCentralBoardCounts:
      h.add(central_board_counts[idx[0]][idx[1]]); return;
    case kFidAnimalMarket:
      h.add(static_cast<int>(animal_market[idx[0]])); return;
    case kFidAnimalDeck:
      h.add(static_cast<int>(animal_deck[idx[0]])); return;
    case kFidAnimalDeckCount:
      h.add(animal_deck_count); return;
    case kFidPouchCounts:
      h.add(pouch_counts[idx[0]]); return;
    case kFidTowerType:
      h.add(static_cast<int>(tower_type[idx[0]][idx[1]])); return;
    case kFidHasAnimal:
      h.add(static_cast<int>(has_animal[idx[0]][idx[1]])); return;
    case kFidHeldCards:
      h.add(static_cast<int>(held_cards[idx[0]][idx[1]])); return;
    case kFidCubeRemaining:
      h.add(static_cast<int>(cube_remaining[idx[0]][idx[1]])); return;
    case kFidCubeInitial:
      h.add(static_cast<int>(cube_initial[idx[0]][idx[1]])); return;
    case kFidSpaceRemaining:
      h.add(static_cast<int>(space_remaining[idx[0]])); return;
    case kFidAnimalScores:
      h.add(animal_scores[idx[0]]); return;
    case kFidLandscapeScores:
      h.add(landscape_scores[idx[0]]); return;
    case kFidCompletedCards:
      h.add(static_cast<int>(completed_cards[idx[0]])); return;
  }
}

template <int NPlayers, int Side>
void HarmoniesState<NPlayers, Side>::hash_field_bulk(
    Hasher& h, viz::FieldId fid,
    const viz::VizTensor& v, int perspective) const {
  // All Harmonies fields are all_public — no viz check needed. Bulk-hash
  // the bigger arrays to skip per-slot virtual dispatch + idx vector.
  switch (fid) {
    case kFidTowerType: {
      for (int p = 0; p < NPlayers; ++p) {
        for (int i = 0; i < kHexes; ++i) {
          h.add(static_cast<std::size_t>(fid));
          h.add(p);
          h.add(i);
          h.add(static_cast<int>(tower_type[p][i]));
        }
      }
      return;
    }
    case kFidHasAnimal: {
      for (int p = 0; p < NPlayers; ++p) {
        for (int i = 0; i < kHexes; ++i) {
          h.add(static_cast<std::size_t>(fid));
          h.add(p);
          h.add(i);
          h.add(static_cast<int>(has_animal[p][i]));
        }
      }
      return;
    }
    case kFidCentralBoardCounts: {
      for (int s = 0; s < kSlots; ++s) {
        for (int c = 0; c < kColors; ++c) {
          h.add(static_cast<std::size_t>(fid));
          h.add(s);
          h.add(c);
          h.add(central_board_counts[s][c]);
        }
      }
      return;
    }
    default:
      IGameState::hash_field_bulk(h, fid, v, perspective);
      return;
  }
}

template <int NPlayers, int Side>
std::any HarmoniesState<NPlayers, Side>::read_field_slot(
    viz::FieldId fid, const std::vector<int>& idx) const {
  switch (fid) {
    case kFidCurrentPlayer:    return std::any(current_player_);
    case kFidGameFirstPlayer:  return std::any(game_first_player_);
    case kFidWinner:           return std::any(winner_);
    case kFidTerminal:         return std::any(terminal);
    case kFidEndTriggered:     return std::any(end_triggered);
    case kFidTurnStarted:      return std::any(turn_started);
    case kFidTookCardThisTurn: return std::any(took_card_this_turn);
    case kFidSelectedTokens:   return std::any(static_cast<int>(selected_tokens[idx[0]]));
    case kFidCentralBoardCounts: return std::any(central_board_counts[idx[0]][idx[1]]);
    case kFidAnimalMarket:     return std::any(static_cast<int>(animal_market[idx[0]]));
    case kFidAnimalDeck:       return std::any(static_cast<int>(animal_deck[idx[0]]));
    case kFidAnimalDeckCount:  return std::any(animal_deck_count);
    case kFidPouchCounts:      return std::any(pouch_counts[idx[0]]);
    case kFidTowerType:        return std::any(static_cast<int>(tower_type[idx[0]][idx[1]]));
    case kFidHasAnimal:        return std::any(static_cast<int>(has_animal[idx[0]][idx[1]]));
    case kFidHeldCards:        return std::any(static_cast<int>(held_cards[idx[0]][idx[1]]));
    case kFidCubeRemaining:    return std::any(static_cast<int>(cube_remaining[idx[0]][idx[1]]));
    case kFidCubeInitial:      return std::any(static_cast<int>(cube_initial[idx[0]][idx[1]]));
    case kFidSpaceRemaining:   return std::any(static_cast<int>(space_remaining[idx[0]]));
    case kFidAnimalScores:     return std::any(animal_scores[idx[0]]);
    case kFidLandscapeScores:  return std::any(landscape_scores[idx[0]]);
    case kFidCompletedCards:   return std::any(static_cast<int>(completed_cards[idx[0]]));
  }
  return {};
}

template <int NPlayers, int Side>
void HarmoniesState<NPlayers, Side>::write_field_slot(
    viz::FieldId fid, const std::vector<int>& idx, const std::any& value) {
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
    case kFidCurrentPlayer:    current_player_ = as_int(); return;
    case kFidGameFirstPlayer:  game_first_player_ = as_int(); return;
    case kFidWinner:           winner_ = as_int(); return;
    case kFidTerminal:         terminal = as_bool(); return;
    case kFidEndTriggered:     end_triggered = as_bool(); return;
    case kFidTurnStarted:      turn_started = as_bool(); return;
    case kFidTookCardThisTurn: took_card_this_turn = as_bool(); return;
    case kFidSelectedTokens:
      selected_tokens[idx[0]] = static_cast<std::int8_t>(as_int()); return;
    case kFidCentralBoardCounts:
      central_board_counts[idx[0]][idx[1]] = as_int(); return;
    case kFidAnimalMarket:
      animal_market[idx[0]] = static_cast<std::int8_t>(as_int()); return;
    case kFidAnimalDeck:
      animal_deck[idx[0]] = static_cast<std::int8_t>(as_int()); return;
    case kFidAnimalDeckCount:
      animal_deck_count = as_int(); return;
    case kFidPouchCounts:
      pouch_counts[idx[0]] = as_int(); return;
    case kFidTowerType:
      tower_type[idx[0]][idx[1]] = static_cast<std::int8_t>(as_int()); return;
    case kFidHasAnimal:
      has_animal[idx[0]][idx[1]] = static_cast<std::int8_t>(as_int()); return;
    case kFidHeldCards:
      held_cards[idx[0]][idx[1]] = static_cast<std::int8_t>(as_int()); return;
    case kFidCubeRemaining:
      cube_remaining[idx[0]][idx[1]] = static_cast<std::int8_t>(as_int()); return;
    case kFidCubeInitial:
      cube_initial[idx[0]][idx[1]] = static_cast<std::int8_t>(as_int()); return;
    case kFidSpaceRemaining:
      space_remaining[idx[0]] = static_cast<std::int8_t>(as_int()); return;
    case kFidAnimalScores:
      animal_scores[idx[0]] = as_int(); return;
    case kFidLandscapeScores:
      landscape_scores[idx[0]] = as_int(); return;
    case kFidCompletedCards:
      completed_cards[idx[0]] = static_cast<std::int8_t>(as_int()); return;
  }
}

#define HARMONIES_INST_HarmoniesState(N, S) template class HarmoniesState<N, S>;
HARMONIES_FOR_EACH_VARIANT(HARMONIES_INST_HarmoniesState)
#undef HARMONIES_INST_HarmoniesState

}  // namespace board_ai::harmonies
