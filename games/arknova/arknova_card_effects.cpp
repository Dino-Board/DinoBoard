// Card effects: sponsor rules (sponsor_rules.py), sponsor passives / immediate / income / endgame
// effects (sponsor_effects.py), the map facts they score (mapinfo.py), and the endgame cards
// (final_scoring.py). Effects are dispatched by printed card id.

#include <algorithm>

#include "arknova_internal.h"

namespace board_ai::arknova {

// ---- card tokens ------------------------------------------------------------------------

CardTokens* find_card_tokens(Player& pl, CardIdx c) {
  for (int i = 0; i < pl.n_card_tokens; ++i) {
    if (pl.card_tokens[i].card == c) return &pl.card_tokens[i];
  }
  return nullptr;
}

const CardTokens* find_card_tokens(const Player& pl, CardIdx c) { return find_card_tokens(const_cast<Player&>(pl), c); }

int card_tokens_get(const Player& pl, CardIdx c, int fallback) {
  const CardTokens* t = find_card_tokens(pl, c);
  return t ? t->n : fallback;
}

void card_tokens_set(Player& pl, CardIdx c, int value) {
  CardTokens* t = find_card_tokens(pl, c);
  if (!t) {
    if (pl.n_card_tokens >= kMaxCardTokens) engine_error("too many card token entries");
    t = &pl.card_tokens[pl.n_card_tokens++];
    t->card = c;
  }
  t->n = static_cast<std::int8_t>(value);
}

int card_tokens_setdefault(Player& pl, CardIdx c, int value) {
  if (const CardTokens* t = find_card_tokens(pl, c)) return t->n;
  card_tokens_set(pl, c, value);
  return value;
}

// ---- conditions ---------------------------------------------------------------------------

int unmet_requirements(const Player& pl, CardIdx c) {
  const CardDef& d = card(c);
  IconCounts have = icons(pl);
  int unmet = 0;
  for (int r = 0; r < kNumReqs; ++r) {
    int n = d.req[r];
    if (!n) continue;
    bool ok;
    if (r == kReqAnimalsII) {
      ok = pl.is_upgraded(kAnimals);
    } else if (r == kReqSponsorsII) {
      ok = pl.is_upgraded(kSponsors);
    } else if (r == kReqPartnerZoo) {
      int owned = 0;
      for (int i = 0; i < pl.n_partner_zoos; ++i) owned += !d.continents || (d.continents >> pl.partner_zoos[i] & 1);
      ok = owned >= (d.continents ? 1 : n);
    } else if (r == kReqReputation) {
      ok = pl.reputation >= d.min_reputation;
    } else if (r == kReqAppeal) {
      ok = pl.appeal <= d.max_appeal;
    } else {
      ok = have[r] >= n;
    }
    unmet += !ok;
  }
  return unmet;
}

// ---- sponsor rules ------------------------------------------------------------------------

namespace sr {
namespace {
constexpr int kVeterinarianProjectValue = 4;
constexpr int kBreedingTokens = 2;
constexpr int kOkapiTokens = 3;
constexpr int kExpertSmallDiscount = 3, kExpertLargeDiscount = 4;
constexpr int kBreeding[] = {215, 218};
}  // namespace

bool has(const Player& pl, int printed_id) { return pl.in_zoo(card_by_id(printed_id)); }

int task_value(const Player& pl, int task) {
  int value = kTaskValue[task];
  if (task == kTaskProject && has(pl, 203)) value = std::min(value, kVeterinarianProjectValue);
  return value;
}

std::vector<CardIdx> breeding_cards(const Player& pl) {
  std::vector<CardIdx> out;
  for (int id : kBreeding) {
    CardIdx c = card_by_id(id);
    if (pl.in_zoo(c) && card_tokens_get(pl, c) > 0) out.push_back(c);
  }
  return out;
}

void spend_breeding_token(Player& pl, CardIdx c) {
  int n = card_tokens_get(pl, c);
  if (n <= 0) engine_error("no breeding token left");
  card_tokens_set(pl, c, n - 1);
}

bool multi_support_release(const Player& pl) { return has(pl, 224); }

void release_bonus(State& s, int p) {
  if (has(s.players[p], 224)) gain(s, p, EffectKind::kEffConservation, 1);
}

std::vector<CardIdx> releasable(const Player& pl, CardIdx project, int slot_index) {
  const CardDef& pj = card(project);
  const ProjectSlot& slot = pj.slots[slot_index];
  int lo = slot.size_min >= 0 ? slot.size_min : 0;
  int hi = slot.size_max >= 0 ? slot.size_max : 99;
  std::vector<CardIdx> found;
  for (int i = 0; i < pl.n_zoo_cards; ++i) {
    CardIdx c = pl.zoo_cards[i];
    if (!is_animal(c) || !project_icon_is_plain(pj.proj_icon) || !card(c).tags[project_icon(pj.proj_icon)]) continue;
    if (card(c).size >= lo && card(c).size <= hi) found.push_back(c);
  }
  return found;
}

int waza_kind(const Player& pl) {
  CardIdx w = card_by_id(227);
  if (!pl.in_zoo(w)) return 0;
  int k = card_tokens_get(pl, w);
  return k == kWazaSmall || k == kWazaLarge ? k : 0;
}

bool animal_allowed(const Player& pl, CardIdx c) {
  int kind = waza_kind(pl);
  if (kind == kWazaSmall) return !is_large(c);
  if (kind == kWazaLarge) return !is_small(c);
  return true;
}

int animal_cost_reduction(const Player& pl, CardIdx c) {
  int reduction = 0;
  if (is_small(c) && has(pl, 229)) reduction += kExpertSmallDiscount;
  if (is_large(c) && has(pl, 230)) reduction += kExpertLargeDiscount;
  return reduction;
}

bool requirements_ok(const Player& pl, CardIdx c) {
  int unmet = unmet_requirements(pl, c);
  if (unmet == 0) return true;
  int allowance = has(pl, 263) && is_large(c) ? 1 : 0;
  return unmet <= allowance;
}

int sponsor_level(const State&, int, CardIdx c) { return std::max<int>(0, card(c).level); }

void on_animal_played(State& s, int p, CardIdx c) {
  int kind = waza_kind(s.players[p]);
  if (kind == kWazaSmall && is_small(c)) gain(s, p, EffectKind::kEffAppeal, 2);
  if (kind == kWazaLarge && is_large(c)) gain(s, p, EffectKind::kEffAppeal, 4);
}

bool has_small_program(const Player& pl) { return has(pl, 228); }
bool ignores_terrain(const Player& pl) { return has(pl, 219); }
bool has_engineer(const Player& pl) { return has(pl, 217); }

CardIdx reveal_until(State& s, int p, bool (*test)(CardIdx), Chance& ch) {
  std::vector<CardIdx> revealed;
  CardIdx found = kNoCard;
  bool reshuffled = false;
  while (true) {
    if (s.deck.empty()) {
      if (reshuffled || s.discard.empty()) break;
      s.deck.swap(s.discard);
      s.discard.clear();
      ch.shuffle(s.deck);
      reshuffled = true;
    }
    CardIdx c = s.deck.back();
    s.deck.pop_back();
    if (test(c)) {
      found = c;
      break;
    }
    revealed.push_back(c);
  }
  s.discard.insert(s.discard.end(), revealed.begin(), revealed.end());
  if (found != kNoCard) to_hand(s, p, found, true);
  return found;
}

}  // namespace sr

// ---- map facts (mapinfo.py) ------------------------------------------------------------------

namespace {

bool next_to_building(const Zoo& zoo, int cell) {
  for (int n : map_cell(cell).nbr) {
    if (n >= 0 && zoo.cell_owner[n] != kNoBuilding) return true;
  }
  return false;
}

// (connected, unconnected) counts of water or rock spaces.
std::pair<int, int> connected_terrain(const Zoo& zoo, Terrain t) {
  int total = 0, connected = 0;
  for (int c = 0; c < kMapCells; ++c) {
    if (map_cell(c).terrain != t) continue;
    ++total;
    connected += next_to_building(zoo, c);
  }
  return {connected, total - connected};
}

std::vector<int> empty_spaces(const Zoo& zoo) {
  std::vector<int> out;
  for (int c = 0; c < kMapCells; ++c) {
    if (map_cell(c).terrain == Terrain::kPlain && zoo.cell_owner[c] == kNoBuilding) out.push_back(c);
  }
  return out;
}

int connected_empty_border(const Zoo& zoo) {
  int n = 0;
  for (int c : empty_spaces(zoo)) n += map_cell(c).border && next_to_building(zoo, c);
  return n;
}

bool all_border_connected(const Zoo& zoo) {
  for (int c : empty_spaces(zoo)) {
    if (map_cell(c).border && !next_to_building(zoo, c)) return false;
  }
  return true;
}

std::pair<int, int> bonus_spaces(const Zoo& zoo) {
  int total = 0, connected = 0;
  for (int c : empty_spaces(zoo)) {
    if (map_cell(c).bonus == kPBNone) continue;
    ++total;
    connected += next_to_building(zoo, c);
  }
  return {connected, total - connected};
}

// ---- sponsor effects ---------------------------------------------------------------------------

int pid(CardIdx c) { return card(c).id; }

int distinct(const State& s, int p, bool continents, bool categories) {
  IconCounts have = icons(s.players[p]);
  int n = 0;
  if (categories) {
    for (int k = 0; k < kNumAnimalCategories; ++k) n += have[k] > 0;
  }
  if (continents) {
    for (int k = 0; k < kNumContinents; ++k) n += have[kContinentIcon0 + k] > 0;
  }
  return n;
}

void give_others_money(State& s, int p, int conservation) {
  for (int q = 0; q < kPlayers; ++q) {
    if (q != p) gain(s, q, EffectKind::kEffMoney, 2 * conservation);
  }
}

int own_icon(int id, EffectKind* kind, int* amount) {
  *amount = 1;
  switch (id) {
    case 202: *kind = EffectKind::kEffReputation; return kIcon_science;
    case 204: *kind = EffectKind::kEffConservation; return kIcon_science;
    case 243: *kind = EffectKind::kEffAppeal; *amount = 2; return kIcon_herbivore;
    case 244: *kind = EffectKind::kEffAppeal; *amount = 2; return kIcon_bird;
    case 245: *kind = EffectKind::kEffAppeal; *amount = 2; return kIcon_water;
    case 246: *kind = EffectKind::kEffAppeal; *amount = 2; return kIcon_rock;
    case 247: *kind = EffectKind::kEffAppeal; *amount = 2; return kIcon_primate;
    case 248: *kind = EffectKind::kEffXtoken; return kIcon_primate;
    default: return -1;
  }
}

int any_icon(int id, EffectKind* kind, int* amount) {
  switch (id) {
    case 208: *kind = EffectKind::kEffMoney; *amount = 2; return kIcon_science;
    case 236: *kind = EffectKind::kEffMoney; *amount = 3; return kIcon_primate;
    case 237: *kind = EffectKind::kEffMoney; *amount = 3; return kIcon_reptile;
    case 238: *kind = EffectKind::kEffMoney; *amount = 3; return kIcon_bird;
    case 239: *kind = EffectKind::kEffMoney; *amount = 3; return kIcon_predator;
    case 240: *kind = EffectKind::kEffMoney; *amount = 3; return kIcon_herbivore;
    case 251: *kind = EffectKind::kEffAppeal; *amount = 2; return kIcon_bear;
    default: return -1;
  }
}

// The icon of the other icon triggers (free buildings, Pouch, Clever, card picks, Sun Bathing,
// Hunter, Okapi Stable).
int effect_icon(int id) {
  switch (id) {
    case 210: return kIcon_americas;
    case 211: return kIcon_europe;
    case 213: return kIcon_asia;
    case 212: return kIcon_australia;
    case 214: return kIcon_africa;
    case 249: return kIcon_bird;
    case 250: return kIcon_reptile;
    case 252: return kIcon_predator;
    case 253: return kIcon_herbivore;
    default: return -1;
  }
}

int category_card_icon(int id) {
  switch (id) {
    case 231: return kIcon_primate;
    case 232: return kIcon_reptile;
    case 233: return kIcon_bird;
    case 234: return kIcon_predator;
    case 235: return kIcon_herbivore;
    default: return -1;
  }
}

bool okapi_playable(const State& s, int p) {
  const Player& pl = s.players[p];
  for (CardIdx c : pl.hand_cards()) {
    if (can_play_sponsor(s, p, c) && pl.money >= sr::sponsor_level(s, p, c)) return true;
  }
  return false;
}

void push_free_build(State& s, int p, BuildingKind kind, int size) {
  int fi = push(s, FrameKind::free_build, p);
  Frame& f = frame(s, fi);
  f.d[fd::free_build_n] = 1;
  f.d[fd::free_build_kinds] = static_cast<std::int16_t>(kind);
  f.d[fd::free_build_kinds + 1] = static_cast<std::int16_t>(size);
}

void free_build_trigger(State& s, int owner, int count, BuildingKind kind, int size) {
  const Player& pl = s.players[owner];
  for (int i = 0; i < count; ++i) {
    PlacementQuery q;
    q.kind = kind;
    q.size = size;
    q.build2 = pl.is_upgraded(kBuild);
    q.any_terrain = pl.in_zoo(card_by_id(219));
    if (!pl.zoo.legal_placements(q).empty()) push_free_build(s, owner, kind, size);
  }
}

}  // namespace

bool has_icon_trigger(CardIdx c) {
  switch (pid(c)) {
    case 202: case 204: case 208: case 210: case 211: case 212: case 213: case 214: case 236: case 237: case 238:
    case 239: case 240: case 243: case 244: case 245: case 246: case 247: case 248: case 249: case 250: case 251:
    case 252: case 253: case 262:
      return true;
    default:
      return false;
  }
}

void icon_trigger(State& s, CardIdx trigger, int owner, int p, const IconCounts& added, const std::vector<int>& new_icons,
                  Chance& ch) {
  int id = pid(trigger);
  EffectKind kind{};
  int amount = 0;
  if (int icon = any_icon(id, &kind, &amount); icon >= 0) {
    gain(s, owner, kind, amount * added[icon]);
    return;
  }
  if (p != owner) return;
  if (int icon = own_icon(id, &kind, &amount); icon >= 0) {
    gain(s, owner, kind, amount * added[icon]);
    return;
  }
  const int icon = effect_icon(id);
  switch (id) {
    case 210: free_build_trigger(s, owner, added[icon], BuildingKind::kKiosk, 1); return;
    case 211: free_build_trigger(s, owner, added[icon], BuildingKind::kEnclosure, 1); return;
    case 213: free_build_trigger(s, owner, added[icon], BuildingKind::kPavilion, 1); return;
    case 212:
      for (int i = 0; i < added[icon]; ++i) {
        if (s.players[owner].hand_size) {
          int fi = push(s, FrameKind::ab_pouch, owner);
          frame(s, fi).d[fd::left] = 1;
        }
      }
      return;
    case 214:
      for (int i = 0; i < added[icon]; ++i) push(s, FrameKind::clever, owner);
      return;
    case 249:
      for (int i = 0; i < added[icon]; ++i) {
        std::vector<CardIdx> drawn;
        for (int k = 0; k < 2; ++k) drawn.push_back(draw_from_deck(s, ch));
        offer_pick(s, owner, drawn, 1);
      }
      return;
    case 250:
      for (int i = 0; i < added[icon]; ++i) {
        if (s.players[owner].hand_size) {
          int fi = push(s, FrameKind::ab_sun, owner);
          frame(s, fi).d[fd::left] = 2;
        }
      }
      return;
    case 252:
      for (int i = 0; i < added[icon]; ++i) hunter(s, owner, icons(s.players[owner])[kIcon_predator], ch);
      return;
    case 262: {
      int n = 0;
      for (int icon : new_icons) n += icon < kNumAnimalCategories || (icon >= kContinentIcon0 && icon < kContinentIcon0 + kNumContinents);
      gain(s, owner, EffectKind::kEffAppeal, n);
      gain(s, owner, EffectKind::kEffMoney, 2 * n);
      return;
    }
    case 253: {
      if (!added[icon]) return;
      Player& me = s.players[owner];
      if (card_tokens_setdefault(me, trigger, sr::kOkapiTokens) > 0 && okapi_playable(s, owner)) {
        push(s, FrameKind::okapi_stable, owner);
      }
      return;
    }
    default:
      engine_error("icon trigger not implemented");
  }
}

IconTriggerSpec icon_trigger_spec(CardIdx c) {
  IconTriggerSpec t;
  if (!has_icon_trigger(c)) return t;
  const int id = pid(c);
  if ((t.icon = any_icon(id, &t.kind, &t.amount)) >= 0) {
    t.any_zoo = true;
  } else if ((t.icon = own_icon(id, &t.kind, &t.amount)) < 0) {
    t.icon = effect_icon(id);
    t.amount = 0;
  }
  return t;
}

bool has_immediate(CardIdx c) {
  switch (pid(c)) {
    case 201: case 207: case 208: case 215: case 216: case 218: case 219: case 222: case 227: case 231: case 232:
    case 233: case 234: case 235: case 241: case 242: case 253: case 254: case 258: case 259: case 260: case 262:
    case 263: case 264:
      return true;
    default:
      return false;
  }
}

void immediate_effect(State& s, CardIdx c, int p) {
  Player& pl = s.players[p];
  int id = pid(c);
  if (int icon = category_card_icon(id); icon >= 0) {
    gain(s, p, EffectKind::kEffAppeal, icons(pl)[icon]);
    return;
  }
  switch (id) {
    case 201: case 254: gain(s, p, EffectKind::kEffCard, 1); return;
    case 216: gain(s, p, EffectKind::kEffWorker, 1); return;
    case 207: {
      int n = distinct(s, p, true, true) / 2;
      give_others_money(s, p, n);
      gain(s, p, EffectKind::kEffConservation, n);
      return;
    }
    case 222: {
      int n = std::min<int>(3, icons(pl)[kIcon_science]);
      give_others_money(s, p, n);
      gain(s, p, EffectKind::kEffConservation, n);
      return;
    }
    case 208: gain(s, p, EffectKind::kEffAppeal, icons(pl)[kIcon_science]); return;
    case 241: gain(s, p, EffectKind::kEffAppeal, icons(pl)[kIcon_water]); return;
    case 242: gain(s, p, EffectKind::kEffAppeal, 3 * (icons(pl)[kIcon_rock] / 2)); return;
    case 258: gain(s, p, EffectKind::kEffAppeal, connected_terrain(pl.zoo, Terrain::kWater).first); return;
    case 259: gain(s, p, EffectKind::kEffAppeal, connected_terrain(pl.zoo, Terrain::kRock).first); return;
    case 260: gain(s, p, EffectKind::kEffAppeal, connected_empty_border(pl.zoo)); return;
    case 264: gain(s, p, EffectKind::kEffAppeal, bonus_spaces(pl.zoo).first); return;
    case 262: gain(s, p, EffectKind::kEffMoney, 2 * distinct(s, p, true, true)); return;
    case 215: case 218:
      for (int b : sr::kBreeding) {
        if (sr::has(pl, b)) card_tokens_setdefault(pl, card_by_id(b), sr::kBreedingTokens);
      }
      return;
    case 219: {
      IconCounts have = icons(pl);
      gain(s, p, EffectKind::kEffMoney, 2 * (have[kIcon_water] + have[kIcon_rock]));
      return;
    }
    case 263: push_free_build(s, p, BuildingKind::kEnclosure, 5); return;
    case 227: push(s, FrameKind::waza_kind, p); return;
    case 253: card_tokens_setdefault(pl, c, sr::kOkapiTokens); return;
    default: engine_error("immediate effect not implemented");
  }
}

bool has_income(CardIdx c) {
  switch (pid(c)) {
    case 201: case 206: case 209: case 220: case 231: case 232: case 233: case 234: case 235: case 257:
      return true;
    default:
      return false;
  }
}

void income_effect(State& s, CardIdx c, int p) {
  const Player& pl = s.players[p];
  int id = pid(c);
  if (int icon = category_card_icon(id); icon >= 0) {
    int n = icons(pl)[icon];
    gain(s, p, EffectKind::kEffMoney, n >= 5 ? 9 : n >= 3 ? 6 : n >= 1 ? 3 : 0);
    return;
  }
  switch (id) {
    case 201: gain(s, p, EffectKind::kEffCard, 1); return;
    case 206: gain(s, p, EffectKind::kEffConservation, 1); return;
    case 209: gain(s, p, EffectKind::kEffXtoken, 1); return;
    case 220: gain(s, p, EffectKind::kEffMoney, 3); return;
    case 257: {
      const Zoo& zoo = pl.zoo;
      int entrance = -1;
      for (int b = 0; b < zoo.n_buildings; ++b) {
        if (zoo.buildings[b].name == c) {
          entrance = b;
          break;
        }
      }
      if (entrance < 0) return;
      int n = 0;
      for (int a : zoo.adjacent_buildings(entrance)) {
        const Building& o = zoo.buildings[a];
        n += !(o.kind == BuildingKind::kEnclosure && !o.occupied);
      }
      gain(s, p, EffectKind::kEffMoney, 2 * n);
      return;
    }
    default: engine_error("income effect not implemented");
  }
}

// ---- build hooks -------------------------------------------------------------------------------

void after_build_hooks(State& s, int p, int building) {
  Player& pl = s.players[p];
  const Building& b = pl.zoo.buildings[building];
  for (auto [id, terrain] : {std::pair<int, Terrain>{241, Terrain::kWater}, {242, Terrain::kRock}}) {
    if (!sr::has(pl, id)) continue;
    int n = 0;
    for (int c = 0; c < kMapCells; ++c) {
      if (!(b.cells >> c & 1)) continue;
      bool near = false;
      for (int nb : map_cell(c).nbr) near |= nb >= 0 && map_cell(nb).terrain == terrain;
      n += near;
    }
    gain(s, p, EffectKind::kEffMoney, n);
  }
  after_special_built(s, p, building);
  if (sr::has(pl, 221)) {
    CellMask cells = pl.zoo.buildings[building].cells;
    for (int c = 0; c < kMapCells; ++c) {
      if ((cells >> c & 1) && map_cell(c).bonus != kPBNone && map_cell(c).border) {
        gain(s, p, placement_bonus_effect(map_cell(c).bonus));
      }
    }
  }
}

// ---- endgame -----------------------------------------------------------------------------------

namespace {

constexpr int kMaxEndgamePoints = 4;

void award(State& s, int p, int points) {
  if (points > 0) score(s, p, EffectKind::kEffConservation, std::min(points, kMaxEndgamePoints));
}

void threshold(State& s, int p, CardIdx c, int count) {
  const CardDef& d = card(c);
  int points = 0;
  for (int i = 0; i < d.n_scoring; ++i) {
    if (count >= d.scoring[i].threshold) points = d.scoring[i].conservation;
  }
  award(s, p, std::min(points, kMaxEndgamePoints));
}

void score_steps(State& s, int p, int have, std::initializer_list<std::pair<int, int>> steps) {
  for (auto [need, c] : steps) {
    if (have >= need) {
      score(s, p, EffectKind::kEffConservation, c);
      return;
    }
  }
}

}  // namespace

bool has_endgame(CardIdx c) {
  int id = pid(c);
  if (id >= 1 && id <= 11) return true;
  switch (id) {
    case 201: case 208: case 215: case 218: case 225: case 226: case 241: case 242: case 243: case 244: case 245:
    case 246: case 247: case 251: case 257: case 258: case 259: case 260: case 261: case 264:
      return true;
    default:
      return false;
  }
}

void endgame_effect(State& s, CardIdx c, int p) {
  const Player& pl = s.players[p];
  IconCounts have = icons(pl);
  int id = pid(c);
  auto count_cards = [&](bool (*test)(CardIdx)) {
    int n = 0;
    for (int i = 0; i < pl.n_zoo_cards; ++i) n += test(pl.zoo_cards[i]);
    return n;
  };
  EffectKind kind{};
  int amount = 0;
  switch (id) {
    case 1: threshold(s, p, c, count_cards(is_large)); return;
    case 2: threshold(s, p, c, count_cards(is_small)); return;
    case 3: threshold(s, p, c, have[kIcon_science]); return;
    case 4: {
      int n = (connected_terrain(pl.zoo, Terrain::kWater).second == 0) + (connected_terrain(pl.zoo, Terrain::kRock).second == 0) +
              all_border_connected(pl.zoo) + pl.zoo.is_fully_covered();
      award(s, p, n);
      return;
    }
    case 5: threshold(s, p, c, pl.projects_supported()); return;
    case 6: threshold(s, p, c, static_cast<int>(empty_spaces(pl.zoo).size())); return;
    case 7: threshold(s, p, c, pl.reputation); return;
    case 8: threshold(s, p, c, count_cards(is_sponsor)); return;
    case 9: {
      IconCounts theirs = icons(s.players[(p + kPlayers - 1) % kPlayers]);
      int more = 0;
      for (int k = 0; k < kNumAnimalCategories; ++k) more += have[k] > theirs[k];
      award(s, p, std::min(more, kMaxEndgamePoints));
      return;
    }
    case 10: threshold(s, p, c, have[kIcon_rock]); return;
    case 11: threshold(s, p, c, have[kIcon_water]); return;
    case 201: score_steps(s, p, have[kIcon_science], {{6, 2}, {3, 1}}); return;
    case 251: score_steps(s, p, have[kIcon_bear], {{6, 2}, {3, 1}}); return;
    case 243: case 244: case 245: case 246: case 247:
      score_steps(s, p, have[own_icon(id, &kind, &amount)], {{6, 1}});
      return;
    case 208: case 261:
      if (distinct(s, p, false, true) >= 5) score(s, p, EffectKind::kEffConservation, 1);
      return;
    case 225: case 226:
      if (distinct(s, p, true, false) >= 5) score(s, p, EffectKind::kEffConservation, 1);
      return;
    case 215: case 218:
      if (pl.projects_supported() >= 5) score(s, p, EffectKind::kEffConservation, 1);
      return;
    case 241:
      if (connected_terrain(pl.zoo, Terrain::kWater).second == 0) score(s, p, EffectKind::kEffConservation, 1);
      return;
    case 242:
      if (connected_terrain(pl.zoo, Terrain::kRock).second == 0) score(s, p, EffectKind::kEffConservation, 1);
      return;
    case 257:
      if (pl.zoo.is_fully_covered()) score(s, p, EffectKind::kEffAppeal, 5);
      return;
    case 258: score(s, p, EffectKind::kEffConservation, connected_terrain(pl.zoo, Terrain::kWater).second / 2); return;
    case 259: score(s, p, EffectKind::kEffConservation, connected_terrain(pl.zoo, Terrain::kRock).second / 2); return;
    case 260: score(s, p, EffectKind::kEffConservation, static_cast<int>(empty_spaces(pl.zoo).size()) / 6); return;
    case 264: score(s, p, EffectKind::kEffConservation, bonus_spaces(pl.zoo).second / 2); return;
    default: engine_error("endgame effect not implemented");
  }
}

// ---- frames: Waza Special Assignment, Okapi Stable ------------------------------------------------

namespace {

void waza_options(State&, int, std::vector<Option>& out) {
  out.push_back(opt(Head::kind, kWazaSmall));
  out.push_back(opt(Head::kind, kWazaLarge));
}

void waza_apply(State& s, int fi, const Option& o, Chance& ch) {
  int p = frame(s, fi).player;
  pop(s, fi);
  card_tokens_set(s.players[p], card_by_id(227), o.a[0]);
  sr::reveal_until(s, p, o.a[0] == kWazaSmall ? is_small : is_large, ch);
}

void okapi_options(State& s, int fi, std::vector<Option>& out) {
  int p = frame(s, fi).player;
  if (card_tokens_get(s.players[p], card_by_id(253)) > 0 && okapi_playable(s, p)) out.push_back(opt(Head::yes));
  out.push_back(opt(Head::no));
}

void okapi_apply(State& s, int fi, const Option& o, Chance&) {
  int p = frame(s, fi).player;
  pop(s, fi);
  if (o.head != Head::yes) return;
  CardIdx okapi = card_by_id(253);
  card_tokens_set(s.players[p], okapi, card_tokens_get(s.players[p], okapi) - 1);
  push(s, FrameKind::paid_sponsor, p);
}

}  // namespace

void register_sponsor_effect_frames() {
  register_choice(FrameKind::waza_kind, waza_options, waza_apply);
  register_choice(FrameKind::okapi_stable, okapi_options, okapi_apply);
}

}  // namespace board_ai::arknova
