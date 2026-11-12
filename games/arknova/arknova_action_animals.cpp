// Animals action: play animal cards from hand (or from the display on side II) into enclosures
// (ark-nova-ai arknova/actions/animals.py).

#include <algorithm>

#include "arknova_internal.h"

namespace board_ai::arknova {

namespace {

int animal_count(const State& s, int p, int strength) {
  return kAnimalsTable[is_upgraded(s, p, kAnimals)][std::min(strength, 5) - 1];
}

}  // namespace

int animals_strength_needed(const Player& pl) {
  const auto& row = kAnimalsTable[pl.is_upgraded(kAnimals)];
  int strength = 1;
  while (strength < 5 && row[strength - 1] <= 0) ++strength;
  return strength;
}

bool terrain_ok(const Player& pl, const Building& b, const CardDef& c) {
  return sr::ignores_terrain(pl) || (pl.zoo.adjacent_terrain(b.cells, Terrain::kRock) >= c.enc_rock &&
                                     pl.zoo.adjacent_terrain(b.cells, Terrain::kWater) >= c.enc_water);
}

BuildingKind special_building(SpecialKind k) {
  switch (k) {
    case SpecialKind::kReptileHouse: return BuildingKind::kReptileHouse;
    case SpecialKind::kAviary: return BuildingKind::kAviary;
    case SpecialKind::kPettingZoo: return BuildingKind::kPettingZoo;
    default: engine_error("bad special enclosure");
  }
}

int animal_cost(const Player& pl, CardIdx c) {
  const CardDef& d = card(c);
  int discount = 0;
  for (int k = 0; k < kNumContinents; ++k) {
    if (pl.has_partner_zoo(k)) discount += kPartnerZooDiscount * d.continent_icons[k];
  }
  return std::max(0, d.cost - discount - sr::animal_cost_reduction(pl, c));
}

std::vector<Housing> housings(const Player& pl, CardIdx c) {
  const CardDef& d = card(c);
  std::vector<Housing> found;
  const Zoo& zoo = pl.zoo;
  if (d.enc_standard >= 0) {
    for (int b = 0; b < zoo.n_buildings; ++b) {
      const Building& bb = zoo.buildings[b];
      if (bb.kind == BuildingKind::kEnclosure && !bb.occupied && bb.capacity() >= d.enc_standard && terrain_ok(pl, bb, d)) {
        found.push_back({b, 0});
      }
    }
    int flock = -1;
    for (int i = 0; i < d.n_abilities; ++i) {
      if (d.abilities[i].keyword == Keyword::kFLOCK_ANIMAL) {
        flock = d.abilities[i].value;
        break;
      }
    }
    if (flock >= 0) {
      for (int b = 0; b < zoo.n_buildings; ++b) {
        const Building& bb = zoo.buildings[b];
        if (bb.kind != BuildingKind::kEnclosure || !bb.occupied || bb.capacity() < flock || !terrain_ok(pl, bb, d)) continue;
        bool herbivore = false;
        for (int h = 0; h < pl.n_housing; ++h) {
          if (pl.housing[h].building == b && card(pl.housing[h].card).tags[kIcon_herbivore]) herbivore = true;
        }
        if (herbivore) found.push_back({b, 0});
      }
    }
  }
  for (int i = 0; i < d.n_special; ++i) {
    BuildingKind kind = special_building(d.special[i].kind);
    for (int b = 0; b < zoo.n_buildings; ++b) {
      const Building& bb = zoo.buildings[b];
      if (bb.kind == kind && bb.size - bb.used >= d.special[i].size && terrain_ok(pl, bb, d)) {
        found.push_back({b, d.special[i].size});
      }
    }
  }
  return found;
}

HousingEntry* find_housing(Player& pl, CardIdx c) {
  for (int i = 0; i < pl.n_housing; ++i) {
    if (pl.housing[i].card == c) return &pl.housing[i];
  }
  return nullptr;
}

void remove_zoo_card(Player& pl, CardIdx c) {
  auto* end = pl.zoo_cards + pl.n_zoo_cards;
  auto* it = std::find(pl.zoo_cards, end, c);
  if (it == end) engine_error("card not in the zoo");
  std::copy(it + 1, end, it);
  --pl.n_zoo_cards;
}

void vacate_standard_enclosure(Player& pl, int h) {
  const int home = pl.housing[h].building;
  const CardDef& d = card(pl.housing[h].card);
  int fits = -1, sized = -1;
  for (int b = 0; b < pl.zoo.n_buildings; ++b) {
    const Building& bb = pl.zoo.buildings[b];
    if (bb.kind != BuildingKind::kEnclosure || !bb.occupied || bb.capacity() < d.enc_standard) continue;
    if (sized < 0 || bb.capacity() < pl.zoo.buildings[sized].capacity()) sized = b;
    if (terrain_ok(pl, bb, d) && (fits < 0 || bb.capacity() < pl.zoo.buildings[fits].capacity())) fits = b;
  }
  const int flipped = g_upstream_rules ? home : fits >= 0 ? fits : sized >= 0 ? sized : home;
  pl.housing[h].building = static_cast<std::int8_t>(flipped);
  for (int i = 0; i < pl.n_housing; ++i) {
    if (i != h && pl.housing[i].building == flipped) pl.housing[i].building = static_cast<std::int8_t>(home);
  }
  for (int b : {home, flipped}) {
    bool occupied = false;
    for (int i = 0; i < pl.n_housing; ++i) occupied |= i != h && pl.housing[i].building == b;
    pl.zoo.buildings[b].occupied = occupied;
  }
}

void release_animal(State& s, int p, CardIdx c) {
  Player& pl = s.players[p];
  remove_zoo_card(pl, c);
  HousingEntry* h = find_housing(pl, c);
  if (!h) engine_error("release_animal: animal has no housing");
  int idx = static_cast<int>(h - pl.housing);
  Building& b = pl.zoo.buildings[h->building];
  if (b.kind == BuildingKind::kEnclosure) {
    vacate_standard_enclosure(pl, idx);
  } else {
    b.used -= h->used;
  }
  for (int i = idx + 1; i < pl.n_housing; ++i) pl.housing[i - 1] = pl.housing[i];
  --pl.n_housing;
  pl.appeal = static_cast<std::int16_t>(std::max(0, pl.appeal - card(c).appeal));
  s.discard.push_back(c);
}

namespace {

struct Play {
  Option key;
  CardIdx card;
  int building;
  int spaces;
  int cost;
};

std::vector<Play> plays(const State& s, int p, bool small_only = false) {
  const Player& pl = s.players[p];
  struct Source {
    Head head;
    int ref;
    CardIdx card;
    int extra;
  };
  std::vector<Source> sources;
  for (CardIdx c : pl.hand_cards()) sources.push_back({Head::play, c, c, 0});
  if (is_upgraded(s, p, kAnimals)) {
    for (int i : cards_in_range(s, p)) sources.push_back({Head::play_display, i, s.display[i], i + 1});
  }
  std::vector<Play> out;
  for (const Source& src : sources) {
    CardIdx c = src.card;
    if (!is_animal(c) || !sr::requirements_ok(s, p, c) || !sr::animal_allowed(pl, c) || (small_only && !is_small(c))) continue;
    int cost = animal_cost(pl, c) + src.extra;
    if (cost > pl.money) continue;
    for (const Housing& h : housings(pl, c)) {
      out.push_back({opt(src.head, src.ref, h.building), c, h.building, h.spaces, cost});
    }
  }
  return out;
}

// Side II at strength 5+ gains its reputation before playing, so judge the display range after it.
bool animals_feasible(State& s, int p, int strength) {
  if (animal_count(s, p, strength) <= 0) return false;
  Player& pl = s.players[p];
  const auto reputation = pl.reputation;
  if (!g_upstream_rules && is_upgraded(s, p, kAnimals) && strength >= kAnimalsIIReputationStrength &&
      pl.reputation < kReputationMax &&
      (pl.reputation < kReputationCapWithoutCardsII || pl.is_upgraded(kCards))) {
    ++pl.reputation;
  }
  const bool any = !plays(s, p).empty();
  pl.reputation = reputation;
  return any;
}

void animals_start(State& s, int p, int strength) {
  int fi = push(s, FrameKind::act_animals, p);
  Frame& f = frame(s, fi);
  f.d[fd::animals_remaining] = static_cast<std::int16_t>(animal_count(s, p, strength));
  f.d[fd::animals_played] = 0;
  f.d[fd::animals_phase] = fd::kAnimalsPlay;
  f.d[fd::animals_all_small] = 1;
  if (is_upgraded(s, p, kAnimals) && strength >= kAnimalsIIReputationStrength) gain(s, p, EffectKind::kEffReputation, 1);
}

std::vector<int> small_in_display(const State& s, int p) {
  std::vector<int> out;
  for (int i : cards_in_range(s, p)) {
    if (is_animal(s.display[i]) && is_small(s.display[i])) out.push_back(i);
  }
  return out;
}

void advance(State& s, int fi) {
  Frame& f = frame(s, fi);
  int p = f.player;
  if (f.d[fd::animals_phase] == fd::kAnimalsPlay) {
    if (!(f.d[fd::animals_played] && f.d[fd::animals_all_small] && sr::has_small_program(s.players[p]))) {
      pop(s, fi);
      return;
    }
    f.d[fd::animals_phase] = fd::kAnimalsBonus;
    if (!plays(s, p, true).empty()) return;
  }
  if (f.d[fd::animals_phase] == fd::kAnimalsBonus) {
    f.d[fd::animals_phase] = fd::kAnimalsTake;
    if (!small_in_display(s, p).empty()) return;
  }
  pop(s, fi);
}

void animals_options(State& s, int fi, std::vector<Option>& out) {
  const Frame& f = frame(s, fi);
  int p = f.player;
  if (f.d[fd::animals_phase] == fd::kAnimalsTake) {
    std::vector<int> small = small_in_display(s, p);
    for (int i : small) out.push_back(opt(Head::take_small, i));
    if (small.empty()) out.push_back(opt(Head::done));
    return;
  }
  std::vector<Play> ps = plays(s, p, f.d[fd::animals_phase] == fd::kAnimalsBonus);
  for (const Play& pl : ps) out.push_back(pl.key);
  if (f.d[fd::animals_played] || ps.empty() || f.d[fd::animals_phase] == fd::kAnimalsBonus) out.push_back(opt(Head::done));
}

void animals_apply(State& s, int fi, const Option& o, Chance& ch) {
  Frame& f = frame(s, fi);
  int p = f.player;
  Player& pl = s.players[p];
  if (o.head == Head::done) {
    if (f.d[fd::animals_phase] == fd::kAnimalsTake) {
      pop(s, fi);
    } else {
      advance(s, fi);
    }
    return;
  }
  if (o.head == Head::take_small) {
    pop(s, fi);
    to_hand(s, p, take_from_display(s, o.a[0]), true);
    return;
  }
  std::vector<Play> ps = plays(s, p, f.d[fd::animals_phase] == fd::kAnimalsBonus);
  auto match = std::find_if(ps.begin(), ps.end(), [&](const Play& m) { return m.key == o; });
  if (match == ps.end()) engine_error("act_animals: play not found");
  Play m = *match;
  pl.money -= m.cost;
  if (o.head == Head::play) {
    remove_from_hand(s, p, m.card);
  } else {
    take_from_display(s, o.a[0]);
  }
  Building& b = pl.zoo.buildings[m.building];
  const bool filled = b.kind == BuildingKind::kEnclosure && !b.occupied;
  if (b.kind == BuildingKind::kEnclosure) {
    b.occupied = true;
  } else {
    b.used += m.spaces;
  }
  if (pl.n_housing >= kMaxHousing) engine_error("too many housed animals");
  pl.housing[pl.n_housing++] = {m.card, static_cast<std::int8_t>(m.building), static_cast<std::int8_t>(m.spaces)};
  IconCounts before = icons(pl);
  if (pl.n_zoo_cards >= kMaxZooCards) engine_error("too many zoo cards");
  pl.zoo_cards[pl.n_zoo_cards++] = m.card;
  --f.d[fd::animals_remaining];
  ++f.d[fd::animals_played];
  bool in_play_phase = f.d[fd::animals_phase] == fd::kAnimalsPlay;
  f.d[fd::animals_all_small] = f.d[fd::animals_all_small] && is_small(m.card);
  if (!in_play_phase || f.d[fd::animals_remaining] <= 0 || (g_upstream_rules && plays(s, p).empty())) advance(s, fi);
  const CardDef& c = card(m.card);
  gain(s, p, EffectKind::kEffAppeal, c.appeal);
  gain(s, p, EffectKind::kEffConservation, c.conservation);
  gain(s, p, EffectKind::kEffReputation, c.reputation);
  sr::on_animal_played(s, p, m.card);
  if (filled) map_enclosure_filled(s, p, m.building);
  resolve_abilities(s, p, m.card, ch);
  IconCounts added{};
  const auto& ci = card_icons(m.card);
  for (int k = 0; k < kNumIcons; ++k) added[k] = ci[k];
  icons_played(s, p, added, before, ch);
}

}  // namespace

void register_animals_action() {
  register_action(kAnimals, animals_feasible, animals_start);
  register_choice(FrameKind::act_animals, animals_options, animals_apply);
}

}  // namespace board_ai::arknova
