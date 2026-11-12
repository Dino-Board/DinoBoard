// Special rules of zoo maps 1-8 (ark-nova-ai arknova/map_rules.py). Map 0, 3 and A have none
// beyond their layout. Each rule checks the zoo's map, so other maps are unaffected.

#include <algorithm>

#include "arknova_internal.h"

namespace board_ai::arknova {
namespace {

bool rule_is(const Player& pl, MapRule r) { return pl.zoo.def().rule == r; }

// Cells next to some cell of `cells` (map_rules._adjacent: a cell of `cells` itself counts when it
// is next to another one).
CellMask touching(CellMask cells) {
  CellMask out = 0;
  for (CellMask m = cells; m; m &= m - 1) {
    for (int n : map_cell(__builtin_ctzll(m)).nbr) {
      if (n >= 0) out |= CellMask{1} << n;
    }
  }
  return out;
}

bool any_rule_cell_covered(const Player& pl) { return (pl.zoo.covered() & pl.zoo.def().rule_cells) != 0; }
bool all_rule_cells_covered(const Player& pl) {
  return (pl.zoo.def().rule_cells & ~pl.zoo.covered()) == 0;
}

// The hand cards an option key names (the card fields of option_to_json).
void option_cards(const State& s, const Option& o, std::vector<CardIdx>& out) {
  switch (o.head) {
    case Head::keep: case Head::discard_final: case Head::discard: case Head::pick: case Head::sell:
    case Head::tuck: case Head::discard_hand: case Head::move:
      for (int i = 0; i < o.n; ++i) out.push_back(o.a[i]);
      break;
    case Head::take:
      if (s.top().kind != FrameKind::ab_snap) out.push_back(o.a[0]);
      break;
    case Head::play: out.push_back(o.a[0]); break;
    case Head::support: case Head::release: case Head::play_project: case Head::play_release:
    case Head::play_project_display: case Head::play_release_display:
      if (o.head != Head::play_project_display && o.head != Head::play_release_display) out.push_back(o.a[0]);
      if (o.a[3] == kExtraRelease) out.push_back(o.a[4]);
      break;
    default: break;
  }
}

std::vector<Option> base_options(State& s) {
  std::vector<Option> out;
  registry().options[static_cast<int>(s.top().kind)](s, s.stack_n - 1, out);
  return out;
}

// map_rules._harbor_leaves_options: discarding c must not leave the pending decision with nothing
// to choose. Only cards that the pending options name are checked, by computing the options
// without the card.
bool harbor_leaves_options(State& s, const std::vector<Option>& options, CardIdx c) {
  if (s.top().kind == FrameKind::discard_hand) return true;
  bool named = false;
  std::vector<CardIdx> cards;
  for (const Option& o : options) {
    cards.clear();
    option_cards(s, o, cards);
    if (std::find(cards.begin(), cards.end(), c) != cards.end()) {
      named = true;
      break;
    }
  }
  if (!named) return true;
  Player& pl = s.players[s.top().player];
  pl.hand[c] = 0;
  --pl.hand_size;
  bool left = !base_options(s).empty();
  pl.hand[c] = 1;
  ++pl.hand_size;
  return left;
}

}  // namespace

int map_condition_allowance(const Player& pl) {
  return rule_is(pl, MapRule::kResearchInstitute) && any_rule_cell_covered(pl) ? 1 : 0;
}

int map_sponsor_level_delta(const Player& pl) {
  return rule_is(pl, MapRule::kHollywood) && all_rule_cells_covered(pl) ? -1 : 0;
}

void map_after_build(State& s, int p, int building, Chance& ch) {
  Player& pl = s.players[p];
  Building& b = pl.zoo.buildings[building];
  if (rule_is(pl, MapRule::kHollywood)) {
    for (int i = popcount(b.cells & pl.zoo.def().rule_cells); i > 0; --i) sr::reveal_until(s, p, is_sponsor, ch);
  }
  if (rule_is(pl, MapRule::kOutdoorArea) && b.kind == BuildingKind::kEnclosure &&
      (touching(b.cells) & pl.zoo.def().rule_cells)) {
    pl.zoo.buildings[building].extra_size = kOutdoorExtraSize;
  }
}

void map_enclosure_filled(State& s, int p, int building) {
  const Player& pl = s.players[p];
  if (rule_is(pl, MapRule::kObservationTower) && (touching(pl.zoo.buildings[building].cells) & pl.zoo.def().rule_cells)) {
    gain(s, p, EffectKind::kEffAppeal, kObservationAppeal);
  }
}

int map_break_money(const Player& pl) {
  const MapDef& md = pl.zoo.def();
  if (md.rule == MapRule::kParkRestaurant) {
    int n = 0;
    for (CellMask m = md.rule_cells; m; m &= m - 1) {
      for (int c : map_cell(__builtin_ctzll(m)).nbr) n += c >= 0 && pl.zoo.cell_owner[c] != kNoBuilding;
    }
    return n;
  }
  if (md.rule == MapRule::kIceCream && all_rule_cells_covered(pl)) return pl.zoo.count(BuildingKind::kKiosk);
  return 0;
}

void map_break_income(State& s, int p) { gain(s, p, EffectKind::kEffMoney, map_break_money(s.players[p])); }

bool harbor_active(const Player& pl) { return rule_is(pl, MapRule::kHarbor) && any_rule_cell_covered(pl); }

void harbor_options(State& s, std::vector<Option>& out) {
  const int p = s.top().player;
  if (!harbor_active(s.players[p]) || p != s.current || s.harbor_used) return;
  for (int i = 0; i < s.stack_n; ++i) {
    FrameKind k = s.stack[i].kind;
    if (k == FrameKind::break_ || k == FrameKind::setup_keep || k == FrameKind::final_scoring) return;
  }
  std::vector<Option> base = out;
  for (CardIdx c : s.players[p].hand_cards()) {
    if (harbor_leaves_options(s, base, c)) out.push_back(opt(Head::harbor, c));
  }
}

void harbor_apply(State& s, const Option& o) {
  const int p = s.top().player;
  Player& pl = s.players[p];
  remove_from_hand(s, p, o.a[0]);
  s.discard.push_back(o.a[0]);
  s.harbor_used = true;
  gain(s, p, EffectKind::kEffMoney, kHarborMoney);
  // A hand discard that this discard has satisfied (hand empty, or down to the limit at a break)
  // ends here: otherwise it would offer nothing or force discards below the limit.
  const Frame& top = s.top();
  if (top.kind == FrameKind::discard_hand && top.player == p) {
    const int limit = top.d[fd::discard_limit];
    if (pl.hand_size == 0 || (limit != kNull && pl.hand_size <= limit)) pop(s, s.stack_n - 1);
  }
}

}  // namespace board_ai::arknova
