// Cards and Build actions (ark-nova-ai arknova/actions/cards_action.py, actions/build.py).

#include <algorithm>

#include "arknova_internal.h"

namespace board_ai::arknova {
namespace {

// ---- cards: advance the break 2, then draw (deck, or display in range on side II) or snap ----

bool cards_feasible(State&, int, int) { return true; }

void cards_start(State& s, int p, int strength) {
  bool upgraded = is_upgraded(s, p, kCards);
  const CardsRow& row = kCardsTable[upgraded][std::min(strength, 5) - 1];
  int fi = push(s, FrameKind::act_cards, p);
  Frame& f = frame(s, fi);
  f.d[fd::cards_draw] = row.draw;
  f.d[fd::cards_discard] = row.discard;
  f.d[fd::cards_may_snap] = row.may_snap;
  f.d[fd::cards_upgraded] = upgraded;
  f.d[fd::cards_drawn] = 0;
  advance_break(s, p, 2);
}

void cards_options(State& s, int fi, std::vector<Option>& out) {
  const Frame& f = frame(s, fi);
  std::size_t n0 = out.size();
  if (!s.deck.empty() || !s.discard.empty()) out.push_back(opt(Head::deck));
  if (f.d[fd::cards_upgraded]) {
    for (int i : cards_in_range(s, f.player)) out.push_back(opt(Head::display, i));
  }
  if (f.d[fd::cards_may_snap] && f.d[fd::cards_drawn] == 0) {
    for (int i = 0; i < kDisplaySize; ++i) {
      if (s.display[i] != kNoCard) out.push_back(opt(Head::snap, i));
    }
  }
  if (out.size() == n0) out.push_back(opt_skip());
}

void cards_apply(State& s, int fi, const Option& o, Chance& ch) {
  Frame& f = frame(s, fi);
  int p = f.player;
  if (o.head == Head::snap || o.head == Head::skip) {
    pop(s, fi);
    if (o.head == Head::snap) to_hand(s, p, take_from_display(s, o.a[0]), true);
    return;
  }
  if (o.head == Head::deck && !f.d[fd::cards_upgraded]) {
    for (int i = 0; i < f.d[fd::cards_draw]; ++i) {
      CardIdx c = draw_from_deck(s, ch);
      if (c != kNoCard) to_hand(s, p, c, false);
    }
    f.d[fd::cards_drawn] = f.d[fd::cards_draw];
  } else {
    if (o.head == Head::deck) {
      to_hand(s, p, draw_from_deck(s, ch), false);
    } else {
      to_hand(s, p, take_from_display(s, o.a[0]), true);
    }
    ++f.d[fd::cards_drawn];
  }
  if (f.d[fd::cards_drawn] >= f.d[fd::cards_draw]) {
    int discard = f.d[fd::cards_discard];
    pop(s, fi);
    if (discard && s.players[p].hand_size) {
      int d = push(s, FrameKind::discard_hand, p);
      frame(s, d).d[fd::discard_count] = static_cast<std::int16_t>(discard);
    }
  }
}

// ---- build: side I one building (size <= strength); side II several different ones ------------

constexpr KindSize kSideI[] = {
    {BuildingKind::kKiosk, 1},     {BuildingKind::kPavilion, 1},  {BuildingKind::kEnclosure, 1},
    {BuildingKind::kEnclosure, 2}, {BuildingKind::kEnclosure, 3}, {BuildingKind::kEnclosure, 4},
    {BuildingKind::kEnclosure, 5}, {BuildingKind::kPettingZoo, 3},
};
constexpr KindSize kSideIIExtra[] = {{BuildingKind::kReptileHouse, 5}, {BuildingKind::kAviary, 5}};

bool built_contains(const Frame* f, BuildingKind kind, int size) {
  if (!f) return false;
  for (int i = 0; i < f->d[fd::build_built_n]; ++i) {
    if (f->d[fd::build_built + 2 * i] == static_cast<int>(kind) && f->d[fd::build_built + 2 * i + 1] == size) return true;
  }
  return false;
}

// Calls fn(kind, size, placements) for every building that can be built right now; stops early
// when fn returns false.
template <class Fn>
void build_candidates(const State& s, int p, int budget, const Frame* built, Fn&& fn) {
  const Player& pl = s.players[p];
  bool upgraded = is_upgraded(s, p, kBuild);
  auto visit = [&](const KindSize& ks) {
    if (ks.size > budget || built_contains(built, ks.kind, ks.size)) return true;
    if (kBuildCostPerSpace * ks.size > pl.money) return true;
    PlacementQuery q;
    q.kind = ks.kind;
    q.size = ks.size;
    q.build2 = upgraded;
    q.any_terrain = sr::ignores_terrain(pl);
    std::vector<CellMask> placements = pl.zoo.legal_placements(q);
    if (placements.empty()) return true;
    return fn(ks.kind, ks.size, placements);
  };
  for (const KindSize& ks : kSideI) {
    if (!visit(ks)) return;
  }
  if (upgraded) {
    for (const KindSize& ks : kSideIIExtra) {
      if (!visit(ks)) return;
    }
  }
}

bool build_feasible(State& s, int p, int strength) {
  bool any = false;
  build_candidates(s, p, strength, nullptr, [&](BuildingKind, int, const std::vector<CellMask>&) {
    any = true;
    return false;
  });
  return any;
}

void build_start(State& s, int p, int strength) {
  int fi = push(s, FrameKind::act_build, p);
  Frame& f = frame(s, fi);
  f.d[fd::build_budget] = static_cast<std::int16_t>(strength);
  f.d[fd::build_upgraded] = is_upgraded(s, p, kBuild);
  f.d[fd::build_normal_done] = 0;
  f.d[fd::build_engineer_used] = 0;
  f.d[fd::build_built_n] = 0;
}

bool normal_open(const State& s, int p, const Frame& f) {
  if (f.d[fd::build_normal_done] || f.d[fd::build_budget] <= 0) return false;
  bool any = false;
  build_candidates(s, p, f.d[fd::build_budget], &f, [&](BuildingKind, int, const std::vector<CellMask>&) {
    any = true;
    return false;
  });
  return any;
}

template <class Fn>
void engineer_candidates(const State& s, int p, const Frame& f, Fn&& fn) {
  const Player& pl = s.players[p];
  if (f.d[fd::build_engineer_used] || !sr::has_engineer(pl)) return;
  for (int i = 0; i < f.d[fd::build_built_n]; ++i) {
    auto kind = static_cast<BuildingKind>(f.d[fd::build_built + 2 * i]);
    int size = f.d[fd::build_built + 2 * i + 1];
    if (!sr::engineer_kind(kind) || kBuildCostPerSpace * size > pl.money) continue;
    PlacementQuery q;
    q.kind = kind;
    q.size = size;
    q.build2 = f.d[fd::build_upgraded];
    q.any_terrain = sr::ignores_terrain(pl);
    std::vector<CellMask> placements = pl.zoo.legal_placements(q);
    if (!placements.empty() && !fn(kind, size, placements)) return;
  }
}

void build_options(State& s, int fi, std::vector<Option>& out) {
  const Frame& f = frame(s, fi);
  int p = f.player;
  std::size_t n0 = out.size();
  auto add = [&](Head h) {
    return [&out, h](BuildingKind kind, int size, const std::vector<CellMask>& placements) {
      for (CellMask c : placements) {
        Option o = opt(h, static_cast<int>(kind), size);
        o.cells = c;
        out.push_back(o);
      }
      return true;
    };
  };
  if (!f.d[fd::build_normal_done]) build_candidates(s, p, f.d[fd::build_budget], &f, add(Head::build));
  engineer_candidates(s, p, f, add(Head::engineer));
  if (f.d[fd::build_built_n]) {
    out.push_back(opt(Head::done));
  } else if (out.size() == n0) {
    out.push_back(opt_skip());
  }
}

void build_apply(State& s, int fi, const Option& o, Chance&) {
  if (o.head != Head::build && o.head != Head::engineer) {
    pop(s, fi);
    return;
  }
  Frame& f = frame(s, fi);
  int p = f.player;
  std::uint16_t uid = f.uid;
  auto kind = static_cast<BuildingKind>(o.a[0]);
  int size = o.a[1];
  s.players[p].money -= kBuildCostPerSpace * size;
  if (o.head == Head::engineer) {
    f.d[fd::build_engineer_used] = 1;
  } else {
    f.d[fd::build_budget] -= size;
    int n = f.d[fd::build_built_n];
    if (n >= fd::kMaxBuilt) engine_error("act_build: too many buildings");
    f.d[fd::build_built + 2 * n] = static_cast<std::int16_t>(kind);
    f.d[fd::build_built + 2 * n + 1] = static_cast<std::int16_t>(size);
    f.d[fd::build_built_n] = static_cast<std::int16_t>(n + 1);
    if (!f.d[fd::build_upgraded] || f.d[fd::build_budget] <= 0) f.d[fd::build_normal_done] = 1;
  }
  bool engineer_possible = sr::has_engineer(s.players[p]) && !f.d[fd::build_engineer_used];
  bool finished = f.d[fd::build_normal_done] && !engineer_possible;
  if (finished) pop(s, fi);
  place_building(s, p, kind, size, o.cells);
  if (finished) return;
  int cur = find_frame(s, uid);
  if (cur < 0) engine_error("act_build: frame vanished");
  bool engineer_any = false;
  engineer_candidates(s, p, frame(s, cur), [&](BuildingKind, int, const std::vector<CellMask>&) {
    engineer_any = true;
    return false;
  });
  if (!normal_open(s, p, frame(s, cur)) && !engineer_any) pop(s, cur);
}

}  // namespace

std::vector<KindSize> placeable_buildings(const State& s, int p) {
  const Player& pl = s.players[p];
  const bool upgraded = pl.is_upgraded(kBuild);
  std::vector<KindSize> out;
  auto visit = [&](const KindSize& ks) {
    PlacementQuery q;
    q.kind = ks.kind;
    q.size = ks.size;
    q.build2 = upgraded;
    q.any_terrain = sr::ignores_terrain(pl);
    if (!pl.zoo.legal_placements(q).empty()) out.push_back(ks);
  };
  for (const KindSize& ks : kSideI) visit(ks);
  if (upgraded) {
    for (const KindSize& ks : kSideIIExtra) visit(ks);
  }
  return out;
}

void register_cards_action() {
  register_action(kCards, cards_feasible, cards_start);
  register_choice(FrameKind::act_cards, cards_options, cards_apply);
}

void register_build_action() {
  register_action(kBuild, build_feasible, build_start);
  register_choice(FrameKind::act_build, build_options, build_apply);
}

}  // namespace board_ai::arknova
