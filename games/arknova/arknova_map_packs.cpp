// Zoo Map Packs 1 and 2 (maps 9-14) and the Tournament 1 map: their effects (Marine Worlds glossary:
// Mark, Adapt, Cut Down, Extra Shift, Animal Magnet, Shark Attack) and map rules (map pack rulebooks).

#include <algorithm>

#include "arknova_internal.h"

namespace board_ai::arknova {
namespace {

bool rule_is(const Player& pl, MapRule r) { return pl.zoo.def().rule == r; }

constexpr int kLastMarkerConservation = 1;
constexpr int kStoredMoney = 2;   // Caves: break income per stored card
constexpr int kCutDownMoney = 2;  // per space of the removed enclosure
constexpr int kHandBoost = 1;
constexpr int kAIConcealOrder[] = {1, 2, 4, 5};  // Artificial Intelligence: slots concealed in turn
constexpr int kAIMaxStrength = 6;

void push_free_kiosk_or_pavilion(State& s, int p) {
  int fi = push(s, FrameKind::free_build, p);
  Frame& f = frame(s, fi);
  f.d[fd::free_build_n] = 2;
  f.d[fd::free_build_kinds] = static_cast<std::int16_t>(BuildingKind::kKiosk);
  f.d[fd::free_build_kinds + 1] = 1;
  f.d[fd::free_build_kinds + 2] = static_cast<std::int16_t>(BuildingKind::kPavilion);
  f.d[fd::free_build_kinds + 3] = 1;
}

void push_drawn(State& s, int p, EffectKind kind, int amount) {
  int fi = push(s, FrameKind::drawn_effect, p);
  frame(s, fi).d[fd::drawn_kind] = static_cast<std::int16_t>(kind);
  frame(s, fi).d[fd::drawn_amount] = static_cast<std::int16_t>(amount);
}

// ---- Mark ------------------------------------------------------------------------------

void mark_options(State& s, int fi, std::vector<Option>& out) {
  (void)fi;
  for (int i = 0; i < kDisplaySize; ++i) {
    if (s.display[i] != kNoCard && is_animal(s.display[i]) && s.display_mark[i] < 0) out.push_back(opt(Head::mark, i));
  }
  if (out.empty()) out.push_back(opt_skip());
}

void mark_apply(State& s, int fi, const Option& o, Chance&) {
  const int p = frame(s, fi).player;
  pop(s, fi);
  if (o.head == Head::mark) s.display_mark[o.a[0]] = static_cast<std::int8_t>(p);
}

// ---- Shark Attack: discard animals in reputation range for half their appeal ------------

std::vector<int> shark_targets(const State& s, int p) {
  std::vector<int> out;
  for (int i : cards_in_range(s, p)) {
    if (is_animal(s.display[i])) out.push_back(i);
  }
  return out;
}

void shark_options(State& s, int fi, std::vector<Option>& out) {
  for (int i : shark_targets(s, frame(s, fi).player)) out.push_back(opt(Head::shark, i));
  out.push_back(opt(Head::done));
}

void shark_apply(State& s, int fi, const Option& o, Chance&) {
  Frame& f = frame(s, fi);
  const int p = f.player;
  if (o.head == Head::shark) {
    f.d[fd::shark_appeal] = static_cast<std::int16_t>(f.d[fd::shark_appeal] + card(s.display[o.a[0]]).appeal);
    discard_from_display(s, o.a[0]);  // the display is refilled when the turn ends
    --f.d[fd::left];
    if (f.d[fd::left] > 0 && !shark_targets(s, p).empty()) return;
  }
  const int appeal = f.d[fd::shark_appeal] / 2;
  pop(s, fi);
  gain(s, p, EffectKind::kEffAppeal, appeal);
}

// ---- Cut Down: an empty standard enclosure goes back to the supply ----------------------

void cut_down_options(State& s, int fi, std::vector<Option>& out) {
  const Zoo& zoo = s.players[frame(s, fi).player].zoo;
  for (int b = 0; b < zoo.n_buildings; ++b) {
    if (zoo.buildings[b].kind == BuildingKind::kEnclosure && !zoo.buildings[b].occupied) out.push_back(opt(Head::cut_down, b));
  }
  out.push_back(opt_skip());
}

void cut_down_apply(State& s, int fi, const Option& o, Chance& ch) {
  const int p = frame(s, fi).player;
  pop(s, fi);
  if (o.head != Head::cut_down) return;
  Player& pl = s.players[p];
  const int b = o.a[0];
  const int size = pl.zoo.buildings[b].size;
  pl.zoo.remove(b);
  for (int i = 0; i < pl.n_housing; ++i) {
    if (pl.housing[i].building == b) engine_error("cut down: the enclosure houses an animal");
    if (pl.housing[i].building > b) --pl.housing[i].building;
  }
  gain(s, p, EffectKind::kEffMoney, kCutDownMoney * size);
  map_quarters_changed(s, p, ch);
}

// ---- Extra Shift: a worker comes back from the Association board -------------------------

void extra_shift_options(State& s, int fi, std::vector<Option>& out) {
  const Player& pl = s.players[frame(s, fi).player];
  for (int t = 0; t < kNumTasks; ++t) {
    if (pl.workers_out[t] > 0) out.push_back(opt(Head::shift, t));
  }
  out.push_back(opt_skip());
}

void extra_shift_apply(State& s, int fi, const Option& o, Chance&) {
  const int p = frame(s, fi).player;
  pop(s, fi);
  if (o.head == Head::shift) --s.players[p].workers_out[o.a[0]];
}

// ---- Caves: store a hand card face down ----------------------------------------------------

void store_options(State& s, int fi, std::vector<Option>& out) {
  for (CardIdx c : s.players[frame(s, fi).player].hand_cards()) out.push_back(opt(Head::store, c));
  out.push_back(opt_skip());
}

void store_apply(State& s, int fi, const Option& o, Chance&) {
  const int p = frame(s, fi).player;
  pop(s, fi);
  if (o.head != Head::store) return;
  Player& pl = s.players[p];
  if (pl.n_stored >= kMapPackSlots) engine_error("store: no free storage slot");
  remove_from_hand(s, p, o.a[0]);
  pl.stored[pl.n_stored++] = o.a[0];
}

// ---- Geographical Zoo: continent markers ------------------------------------------------------

void geo_bonus(State& s, int p, int bonus) {
  switch (bonus) {
    case kGeoReputation: gain(s, p, EffectKind::kEffReputation, 1); break;
    case kGeoMoney: gain(s, p, EffectKind::kEffMoney, 4); break;
    case kGeoClever: gain(s, p, EffectKind::kEffClever, 1); break;
    case kGeoAppeal: gain(s, p, EffectKind::kEffAppeal, 2); break;
    case kGeoKioskPavilion: push_free_kiosk_or_pavilion(s, p); break;
    default: engine_error("continent marker: bad bonus");
  }
}

int marker_choices(const State& s, const Frame& f) {
  const int markers = s.players[f.player].continent_markers;
  return f.d[fd::marker_any] == 1 ? markers : markers & f.d[fd::marker_areas];
}

void marker_options(State& s, int fi, std::vector<Option>& out) {
  const int choices = marker_choices(s, frame(s, fi));
  for (int c = 0; c < kNumContinents; ++c) {
    if (!(choices >> c & 1)) continue;
    for (int b = 0; b < kNumGeoBonuses; ++b) out.push_back(opt(Head::continent, c, b));
  }
  out.push_back(opt_skip());
}

void marker_apply(State& s, int fi, const Option& o, Chance&) {
  Frame& f = frame(s, fi);
  const int p = f.player;
  Player& pl = s.players[p];
  if (o.head != Head::continent) {
    pop(s, fi);
    return;
  }
  pl.continent_markers = static_cast<std::uint8_t>(pl.continent_markers & ~(1u << o.a[0]));
  f.d[fd::marker_areas] = static_cast<std::int16_t>(f.d[fd::marker_areas] & ~(1 << o.a[0]));
  // The map token removes one marker; a housed animal with two continents may reach two areas.
  if (f.d[fd::marker_any] == 1 || marker_choices(s, f) == 0) pop(s, fi);
  geo_bonus(s, p, o.a[1]);
  if (pl.continent_markers == 0) gain(s, p, EffectKind::kEffConservation, kLastMarkerConservation);
}

void push_markers(State& s, int p, bool any, int areas) {
  int fi = push(s, FrameKind::continent_marker, p);
  frame(s, fi).d[fd::marker_any] = any ? 1 : 0;
  frame(s, fi).d[fd::marker_areas] = static_cast<std::int16_t>(areas);
}

// ---- Lagoon: a person sponsor for every new worker ---------------------------------------------

// Reveals cards until a person sponsor; it goes to p's hand, the others under the deck in the
// order they were revealed. The discard pile is reshuffled into the deck once when it runs out.
void find_person(State& s, int p, Chance* ch) {
  std::vector<CardIdx> revealed;
  bool reshuffled = false;
  CardIdx found = kNoCard;
  while (found == kNoCard) {
    if (s.deck.empty()) {
      if (reshuffled || !ch || s.discard.empty()) break;
      reshuffled = true;
      s.deck.swap(s.discard);
      s.discard.clear();
      ch->shuffle(s.deck);
      continue;
    }
    CardIdx c = s.deck.back();
    s.deck.pop_back();
    if (card(c).type == CardType::kSponsor && card(c).human) {
      found = c;
    } else {
      revealed.push_back(c);
    }
  }
  // The deck's bottom is its front: the first card revealed ends up above the later ones.
  s.deck.insert(s.deck.begin(), revealed.rbegin(), revealed.rend());
  if (found != kNoCard) to_hand(s, p, found, true);
}

void person_options(State& s, int fi, std::vector<Option>& out) {
  const int p = frame(s, fi).player;
  for (CardIdx c : s.players[p].hand_cards()) {
    if (card(c).type == CardType::kSponsor && card(c).human && can_play_sponsor(s, p, c)) out.push_back(opt(Head::play, c));
  }
  out.push_back(opt_skip());
}

void person_apply(State& s, int fi, const Option& o, Chance& ch) {
  const int p = frame(s, fi).player;
  pop(s, fi);
  if (o.head != Head::play) return;
  remove_from_hand(s, p, o.a[0]);
  play_sponsor(s, p, o.a[0], ch);
}

// ---- effects that draw: Wave, finding a person sponsor, Scavenging ----------------------------

void drawn_effect_auto(State& s, int fi, Chance& ch) {
  const Frame f = frame(s, fi);
  pop(s, fi);
  switch (static_cast<EffectKind>(f.d[fd::drawn_kind])) {
    case EffectKind::kEffWave:  // discard the card in folder 1 and replenish
      if (s.display[0] != kNoCard) discard_from_display(s, 0);
      refill_display(s, ch);
      break;
    case EffectKind::kEffFindPerson: find_person(s, f.player, &ch); break;
    case EffectKind::kEffScavenging: resolve_keyword(s, f.player, Keyword::kSCAVENGING, f.d[fd::drawn_amount], ch); break;
    default: engine_error("drawn_effect: bad effect");
  }
}

// ---- Rescue Station ------------------------------------------------------------------------

void rescue(State& s, int p, CardIdx c, Chance& ch) {
  Player& pl = s.players[p];
  if (pl.n_rescued >= kMapPackSlots) engine_error("rescue: no free slot under the map");
  const IconCounts before = icons(pl);
  pl.rescued[pl.n_rescued++] = c;
  add_zoo_card(pl, c);
  IconCounts added = icons(pl);
  for (int k = 0; k < kNumIcons; ++k) added[k] = static_cast<std::int16_t>(added[k] - before[k]);
  icons_played(s, p, added, before, ch);
}

}  // namespace

bool is_rescued(const Player& pl, CardIdx c) {
  return std::find(pl.rescued, pl.rescued + pl.n_rescued, c) != pl.rescued + pl.n_rescued;
}

// Rescue Station: a discarded Animal card (not a Petting Zoo Animal) may go under the map.
bool rescuable(const State& s, int p, CardIdx c) {
  const Player& pl = s.players[p];
  return pl.n_rescued < kMapPackSlots && is_animal(c) && card(c).tags[kIcon_pet] == 0;
}

void rescue_card(State& s, int p, CardIdx c, Chance& ch) { rescue(s, p, c, ch); }

void map_pack_gain(State& s, int p, EffectKind kind, int amount) {
  Player& pl = s.players[p];
  switch (kind) {
    case EffectKind::kEffRescue: {  // Digging 1 whose discarded animal may go under the map
      int fi = push(s, FrameKind::ab_dig, p);
      frame(s, fi).d[fd::left] = static_cast<std::int16_t>(amount);
      frame(s, fi).d[fd::dig_rescue] = 1;
      break;
    }
    case EffectKind::kEffStore:
      if (pl.n_stored < kMapPackSlots && pl.hand_size > 0) push(s, FrameKind::ab_store, p);
      break;
    case EffectKind::kEffExtraShift:
      if (pl.active_workers() < pl.workers) push(s, FrameKind::ab_extra_shift, p);
      break;
    case EffectKind::kEffStrength:
      pl.concealed = static_cast<std::int8_t>(std::min<int>(std::size(kAIConcealOrder), pl.concealed + amount));
      break;
    case EffectKind::kEffMark:
      for (int i = 0; i < amount; ++i) {
        if (s.turn_depth > 0) {  // "after finishing this action"
          Frame f;
          f.kind = FrameKind::ab_mark;
          f.player = static_cast<std::int8_t>(p);
          after_finishing(s, f);
        } else {
          push(s, FrameKind::ab_mark, p);
        }
      }
      break;
    case EffectKind::kEffWave:
      for (int i = 0; i < amount; ++i) push_drawn(s, p, kind, 1);
      break;
    case EffectKind::kEffSharkAttack:
      if (!shark_targets(s, p).empty()) {
        int fi = push(s, FrameKind::ab_shark, p);
        frame(s, fi).d[fd::left] = static_cast<std::int16_t>(amount);
        frame(s, fi).d[fd::shark_appeal] = 0;
      }
      break;
    case EffectKind::kEffAdapt: {  // draw X Final Scoring cards, then discard X of them
      int drawn = 0;
      while (drawn < amount && !s.final_deck.empty() && pl.n_final < kMaxFinalCards) {
        pl.final_cards[pl.n_final++] = s.final_deck.back();
        s.final_deck.pop_back();
        ++drawn;
      }
      for (int i = 0; i < drawn; ++i) push(s, FrameKind::discard_final, p);
      break;
    }
    case EffectKind::kEffCutDown:
      for (int b = 0; b < pl.zoo.n_buildings; ++b) {
        if (pl.zoo.buildings[b].kind == BuildingKind::kEnclosure && !pl.zoo.buildings[b].occupied) {
          push(s, FrameKind::ab_cut_down, p);
          break;
        }
      }
      break;
    case EffectKind::kEffAnimalMagnet:  // every animal in the display, whatever the reputation
      for (int i = 0; i < kDisplaySize; ++i) {
        if (s.display[i] != kNoCard && is_animal(s.display[i])) to_hand(s, p, take_from_display(s, i), true);
      }
      break;
    case EffectKind::kEffContinentMarker:
      if (pl.continent_markers) push_markers(s, p, true, 0);
      break;
    case EffectKind::kEffFindPerson:
      for (int i = 0; i < amount; ++i) push_drawn(s, p, kind, 1);
      break;
    case EffectKind::kEffPersonSponsor: push(s, FrameKind::person_sponsor, p); break;
    case EffectKind::kEffScavenging: push_drawn(s, p, kind, amount); break;
    default: engine_error("map_pack_gain: not a map pack effect");
  }
}

void map_animal_housed(State& s, int p, CardIdx c, int building) {
  Player& pl = s.players[p];
  if (!rule_is(pl, MapRule::kGeographical) || !pl.continent_markers) return;
  const MapDef& md = pl.zoo.def();
  const CellMask cells = pl.zoo.buildings[building].cells;
  int areas = 0;
  for (int k = 0; k < kNumContinents && k < md.n_areas; ++k) {
    if (card(c).tags[kContinentIcon0 + k] && (md.areas[k] & cells) && (pl.continent_markers >> k & 1)) areas |= 1 << k;
  }
  if (areas) push_markers(s, p, false, areas);
}

void map_quarters_changed(State& s, int p, Chance& ch) {
  Player& pl = s.players[p];
  if (!rule_is(pl, MapRule::kDrawingBoard)) return;
  const MapDef& md = pl.zoo.def();
  const CellMask covered = pl.zoo.covered();
  std::uint8_t now = 0;
  for (int a = 0; a < md.n_areas; ++a) {
    if ((md.areas[a] & ~covered) == 0) now = static_cast<std::uint8_t>(now | 1u << a);
  }
  const std::uint8_t fresh = static_cast<std::uint8_t>(now & ~pl.quarters);
  pl.quarters = now;
  for (int a = 0; a < kDrawingBoardAreas; ++a) {
    if (!(fresh >> a & 1)) continue;
    switch (a) {
      case 0: gain(s, p, EffectKind::kEffMoney, 3); break;
      case 1: gain(s, p, EffectKind::kEffAppeal, 2); break;
      case 2: gain(s, p, EffectKind::kEffReputation, 1); break;
      default: hunter(s, p, 4, ch); break;
    }
  }
}

void map_pack_break_income(State& s, int p, Chance& ch) {
  Player& pl = s.players[p];
  if (rule_is(pl, MapRule::kCaves)) gain(s, p, EffectKind::kEffMoney, kStoredMoney * pl.n_stored);
  if (rule_is(pl, MapRule::kDrawingBoard)) {
    const std::uint8_t covered = pl.quarters;
    if (covered & 1) gain(s, p, EffectKind::kEffMoney, 3);
    if (covered & 2) gain(s, p, EffectKind::kEffAppeal, 2);
    if (covered & 4) gain(s, p, EffectKind::kEffReputation, 1);
    if (covered & 8) hunter(s, p, 4, ch);
  }
}

void lagoon_setup(State& s, int p) {
  if (rule_is(s.players[p], MapRule::kLagoon)) find_person(s, p, nullptr);
}

int hand_boost_strength(const Player& pl) { return pl.hand_boost == kBoostPending ? kHandBoost : 0; }

void spend_hand_boost(Player& pl) {
  if (pl.hand_boost == kBoostPending) pl.hand_boost = kBoostUsed;
}

bool hand_boost_open(const State& s, int p) {
  const Player& pl = s.players[p];
  return rule_is(pl, MapRule::kTournament) && !pl.tokens[0] && pl.hand_boost == kBoostFree;
}

void map_pack_turn_options(State& s, std::vector<Option>& out) {
  const Frame& top = s.top();
  const int p = top.player;
  if (p != s.current) return;
  const Player& pl = s.players[p];
  if ((top.kind == FrameKind::turn || top.kind == FrameKind::extra_action) && hand_boost_open(s, p)) {
    for (CardIdx c : pl.hand_cards()) out.push_back(opt(Head::hand_boost, c));
  }
  if (top.kind == FrameKind::turn && rule_is(pl, MapRule::kCaves)) {
    for (int i = 0; i < pl.n_stored; ++i) out.push_back(opt(Head::unstore, i));
  }
}

bool is_map_pack_turn_option(const Option& o) { return o.head == Head::hand_boost || o.head == Head::unstore; }

void map_pack_turn_apply(State& s, const Option& o) {
  const int p = s.top().player;
  Player& pl = s.players[p];
  if (o.head == Head::hand_boost) {
    remove_from_hand(s, p, o.a[0]);
    s.discard.push_back(o.a[0]);
    pl.hand_boost = kBoostPending;
    return;
  }
  const int i = o.a[0];
  const CardIdx c = pl.stored[i];
  std::copy(pl.stored + i + 1, pl.stored + pl.n_stored, pl.stored + i);
  pl.stored[--pl.n_stored] = kNoCard;
  to_hand(s, p, c, false);
}

int ai_strength(const Player& pl, int slot) {
  // Concealed slots show no number: an action card there has the next visible number to its right.
  int strength = slot;
  auto concealed = [&](int n) {
    for (int i = 0; i < pl.concealed; ++i) {
      if (kAIConcealOrder[i] == n) return true;
    }
    return false;
  };
  while (strength < kAIMaxStrength && concealed(strength)) ++strength;
  return strength;
}

void register_map_pack_frames() {
  register_choice(FrameKind::ab_mark, mark_options, mark_apply);
  register_choice(FrameKind::ab_shark, shark_options, shark_apply);
  register_choice(FrameKind::ab_cut_down, cut_down_options, cut_down_apply);
  register_choice(FrameKind::ab_extra_shift, extra_shift_options, extra_shift_apply);
  register_choice(FrameKind::ab_store, store_options, store_apply);
  register_choice(FrameKind::continent_marker, marker_options, marker_apply);
  register_choice(FrameKind::person_sponsor, person_options, person_apply);
  register_auto(FrameKind::drawn_effect, drawn_effect_auto);
}

}  // namespace board_ai::arknova
