// Sponsors action, sponsor buildings, paid sponsors (actions/sponsors.py) and moving animals into
// a new reptile house or aviary (actions/relocate.py).

#include <algorithm>

#include "arknova_internal.h"

namespace board_ai::arknova {

void add_zoo_card(Player& pl, CardIdx c) {
  if (pl.n_zoo_cards >= kMaxZooCards) engine_error("too many zoo cards");
  pl.zoo_cards[pl.n_zoo_cards++] = c;
}

std::vector<CellMask> building_spots(const State& s, int p, CardIdx c, bool* has_building) {
  const BuildingSpec& spec = card(c).building;
  *has_building = spec.present;
  if (!spec.present) return {};
  const Player& pl = s.players[p];
  Shape shape = card_shape(c);
  PlacementQuery q;
  q.kind = BuildingKind::kUnique;
  q.size = spec.size;
  q.build2 = pl.is_upgraded(kBuild);
  q.shape = &shape;
  q.need_rock = spec.rock;
  q.need_water = spec.water;
  q.min_border = spec.min_border;
  q.adjacent = spec.needs_adjacency;
  q.any_terrain = pl.in_zoo(card_by_id(219));
  return pl.zoo.legal_placements(q);
}

bool can_play_sponsor(const State& s, int p, CardIdx c) {
  if (!is_sponsor(c) || unmet_requirements(s.players[p], c) != 0) return false;
  bool has_building = false;
  std::vector<CellMask> spots = building_spots(s, p, c, &has_building);
  return !has_building || !spots.empty();
}

void play_sponsor(State& s, int p, CardIdx c, Chance& ch) {
  Player& pl = s.players[p];
  const CardDef& d = card(c);
  IconCounts before = icons(pl);
  add_zoo_card(pl, c);
  IconCounts added{};
  const auto& ci = card_icons(c);
  for (int k = 0; k < kNumIcons; ++k) added[k] = ci[k];
  icons_played(s, p, added, before, ch);
  if (has_immediate(c)) immediate_effect(s, c, p);
  gain(s, p, EffectKind::kEffReputation, d.reputation);
  gain(s, p, EffectKind::kEffConservation, d.conservation);
  gain(s, p, EffectKind::kEffAppeal, d.appeal);
  if (d.building.present) {
    int fi = push(s, FrameKind::sponsor_building, p);
    frame(s, fi).d[sponsor_building_card] = c;
  }
}

namespace {

int sponsor_budget(const State& s, int p, int strength) { return is_upgraded(s, p, kSponsors) ? strength + 1 : strength; }

}  // namespace

int sponsor_strength_needed(const State& s, int p, CardIdx c) {
  return std::max(1, sr::sponsor_level(s, p, c) - (s.players[p].is_upgraded(kSponsors) ? 1 : 0));
}

namespace {

struct Playable {
  Option key;
  CardIdx card;
};

std::vector<Playable> playable(const State& s, int p, int budget) {
  const Player& pl = s.players[p];
  std::vector<Playable> out;
  auto consider = [&](Option key, CardIdx c, int extra) {
    if (is_sponsor(c) && sr::sponsor_level(s, p, c) <= budget && pl.money >= extra && can_play_sponsor(s, p, c)) {
      out.push_back({key, c});
    }
  };
  for (CardIdx c : pl.hand_cards()) consider(opt(Head::play, c), c, 0);
  if (is_upgraded(s, p, kSponsors)) {
    for (int i : cards_in_range(s, p)) consider(opt(Head::play_display, i), s.display[i], i + 1);
  }
  return out;
}

bool sponsors_feasible(State&, int, int) { return true; }

void sponsors_start(State& s, int p, int strength) {
  int fi = push(s, FrameKind::act_sponsors, p);
  Frame& f = frame(s, fi);
  f.d[sponsors_strength] = static_cast<std::int16_t>(strength);
  f.d[sponsors_budget] = static_cast<std::int16_t>(sponsor_budget(s, p, strength));
  f.d[sponsors_played] = 0;
}

void sponsors_options(State& s, int fi, std::vector<Option>& out) {
  const Frame& f = frame(s, fi);
  if (f.d[sponsors_played] == 0) out.push_back(opt(Head::break_));
  for (const Playable& pl : playable(s, f.player, f.d[sponsors_budget])) out.push_back(pl.key);
  if (f.d[sponsors_played]) out.push_back(opt(Head::done));
}

void sponsors_apply(State& s, int fi, const Option& o, Chance& ch) {
  Frame& f = frame(s, fi);
  int p = f.player;
  Player& pl = s.players[p];
  if (o.head == Head::break_ || o.head == Head::done) {
    int strength = f.d[sponsors_strength];
    pop(s, fi);
    if (o.head == Head::break_) {
      bool upgraded = is_upgraded(s, p, kSponsors);
      gain(s, p, EffectKind::kEffMoney, strength * (upgraded ? 2 : 1));
      advance_break(s, p, strength);
    }
    return;
  }
  CardIdx c;
  if (o.head == Head::play) {
    c = o.a[0];
    remove_from_hand(s, p, c);
  } else {
    pl.money -= o.a[0] + 1;
    c = take_from_display(s, o.a[0]);
  }
  f.d[sponsors_budget] -= sr::sponsor_level(s, p, c);
  ++f.d[sponsors_played];
  if (!is_upgraded(s, p, kSponsors)) pop(s, fi);
  play_sponsor(s, p, c, ch);
}

void building_options(State& s, int fi, std::vector<Option>& out) {
  const Frame& f = frame(s, fi);
  bool has_building = false;
  std::vector<CellMask> spots = building_spots(s, f.player, f.d[sponsor_building_card], &has_building);
  for (CellMask c : spots) {
    Option o = opt(Head::place);
    o.cells = c;
    out.push_back(o);
  }
  if (spots.empty()) out.push_back(opt_skip());
}

void building_apply(State& s, int fi, const Option& o, Chance& ch) {
  const Frame f = frame(s, fi);
  pop(s, fi);
  if (o.head != Head::place) return;
  CardIdx c = f.d[sponsor_building_card];
  place_building(s, f.player, BuildingKind::kUnique, card(c).building.size, o.cells, ch, c);
}

void paid_options(State& s, int fi, std::vector<Option>& out) {
  int p = frame(s, fi).player;
  const Player& pl = s.players[p];
  for (CardIdx c : pl.hand_cards()) {
    if (can_play_sponsor(s, p, c) && pl.money >= sr::sponsor_level(s, p, c)) out.push_back(opt(Head::play, c));
  }
  out.push_back(opt_skip());
}

void paid_apply(State& s, int fi, const Option& o, Chance& ch) {
  int p = frame(s, fi).player;
  pop(s, fi);
  if (o.head != Head::play) return;
  s.players[p].money -= sr::sponsor_level(s, p, o.a[0]);
  remove_from_hand(s, p, o.a[0]);
  play_sponsor(s, p, o.a[0], ch);
}

// ---- relocate --------------------------------------------------------------------------

struct Movable {
  CardIdx card;
  int spaces;
};

std::vector<Movable> movable(const State& s, int p, int building) {
  const Player& pl = s.players[p];
  const Building& b = pl.zoo.buildings[building];
  std::vector<Movable> found;
  for (int h = 0; h < pl.n_housing; ++h) {
    const HousingEntry& e = pl.housing[h];
    if (pl.zoo.buildings[e.building].kind != BuildingKind::kEnclosure) continue;
    const CardDef& d = card(e.card);
    for (int i = 0; i < d.n_special; ++i) {
      if (special_building(d.special[i].kind) == b.kind && b.size - b.used >= d.special[i].size && terrain_ok(pl, b, d)) {
        found.push_back({e.card, d.special[i].size});
      }
    }
  }
  return found;
}

void move_options(State& s, int fi, std::vector<Option>& out) {
  const Frame& f = frame(s, fi);
  for (const Movable& m : movable(s, f.player, f.d[move_animals_building])) out.push_back(opt(Head::move, m.card));
  out.push_back(opt(Head::done));
}

void move_apply(State& s, int fi, const Option& o, Chance&) {
  if (o.head == Head::done) {
    pop(s, fi);
    return;
  }
  const Frame& f = frame(s, fi);
  int p = f.player;
  int building = f.d[move_animals_building];
  Player& pl = s.players[p];
  int spaces = -1;
  for (const Movable& m : movable(s, p, building)) {
    if (m.card == o.a[0]) spaces = m.spaces;
  }
  if (spaces < 0) engine_error("move_animals: animal cannot move");
  HousingEntry* h = find_housing(pl, o.a[0]);
  if (g_upstream_rules) {
    pl.zoo.buildings[h->building].occupied = false;
  } else {
    vacate_standard_enclosure(pl, static_cast<int>(h - pl.housing));
  }
  pl.zoo.buildings[building].used += spaces;
  h->building = static_cast<std::int8_t>(building);
  h->used = static_cast<std::int8_t>(spaces);
  if (movable(s, p, building).empty()) pop(s, fi);
}

}  // namespace

void after_special_built(State& s, int p, int building) {
  BuildingKind k = s.players[p].zoo.buildings[building].kind;
  if ((k == BuildingKind::kReptileHouse || k == BuildingKind::kAviary) && !movable(s, p, building).empty()) {
    int fi = push(s, FrameKind::move_animals, p);
    frame(s, fi).d[move_animals_building] = static_cast<std::int16_t>(building);
  }
}

void register_sponsors_action() {
  register_action(kSponsors, sponsors_feasible, sponsors_start);
  register_choice(FrameKind::act_sponsors, sponsors_options, sponsors_apply);
  register_choice(FrameKind::sponsor_building, building_options, building_apply);
  register_choice(FrameKind::paid_sponsor, paid_options, paid_apply);
}

void register_relocate_frames() { register_choice(FrameKind::move_animals, move_options, move_apply); }

}  // namespace board_ai::arknova
