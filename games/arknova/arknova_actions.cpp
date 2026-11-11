#include "arknova_actions.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "arknova_internal.h"

namespace board_ai::arknova {

static_assert(atom::kPublicSlots == kFrameData - fd::cards_list, "a public list fills a frame from fd::cards_list");

namespace {

[[noreturn]] void action_error(const std::string& msg) { throw std::logic_error("arknova actions: " + msg); }

int lowest_cell(CellMask cells) {
  if (!cells) action_error("empty cell set");
  return __builtin_ctzll(cells);
}

bool contains(const AtomSequence& seq, ActionId a) { return std::find(seq.begin(), seq.end(), a) != seq.end(); }

class Mapper {
 public:
  Mapper(const ArknovaState& st, const State& s) : st_(st), s_(s), f_(s.top()), p_(f_.player), pl_(s.players[p_]) {}

  // Appends the option's atom sequences. One per option, except a setup keep (see Decision):
  // there only the sequences that start with the atoms chosen so far are listed.
  void sequences(const Option& o, std::vector<AtomSequence>& out) const {
    if (o.head == Head::keep && f_.kind == FrameKind::setup_keep) {
      setup_sequences(o, out);
    } else {
      out.push_back(single(o));
    }
  }

 private:
  AtomSequence single(const Option& o) const {
    switch (o.head) {
      case Head::skip: return {atom::kSkip};
      case Head::done: return {atom::kDone};
      case Head::no: return {atom::kNo};
      case Head::yes: return {atom::kYes};
      case Head::money:
        // A conservation space gives 5 money; the victim of Pilfering pays it.
        return {f_.kind == FrameKind::ab_pilfer ? atom::kGiveMoney : atom::kMoney};
      case Head::deck: return {atom::kDeck};
      case Head::donate: return {atom::kDonate};
      case Head::reputation: return {atom::kReputation};
      case Head::repeat: return {atom::kRepeat};
      case Head::worker: return {atom::kWorker};
      case Head::replenish: return {atom::kReplenish};
      case Head::card: return {atom::kGiveCard};
      case Head::break_: return {atom::kBreak};
      case Head::act: return {atom::kAct + o.a[0] * atom::kMaxX + o.a[1]};
      case Head::xtoken: return {atom::kXtoken + o.a[0]};
      case Head::upgrade: return {atom::kUpgrade + o.a[0]};
      case Head::clever: return {atom::kClever + o.a[0]};
      case Head::multiplier: return {atom::kMultiplier + o.a[0]};
      case Head::boost: return {atom::kBoost + (o.a[0] == 1 ? 0 : 1)};
      case Head::partner_zoo: return {atom::kPartnerZoo + o.a[0]};
      case Head::university: return {atom::kUniversity + o.a[0]};
      case Head::tile: return {atom::kTile + o.a[0]};
      case Head::kind: return {atom::kWaza + (o.a[0] == kWazaSmall ? 0 : 1)};
      case Head::target: return {atom::kTarget + (o.a[0] - p_ + kPlayers) % kPlayers};
      case Head::keep: return {private_slot(o.a[0])};
      case Head::pick: return {private_slot(o.a[0])};
      case Head::take:
        if (f_.kind == FrameKind::ab_snap) return {display(atom::kDisplayTake, o.a[0])};
        return {public_slot(o.a[0])};
      case Head::discard_final: return {final_slot(o.a[0])};
      case Head::move: return {zoo_card(atom::kZoo, o.a[0])};
      case Head::sell: case Head::tuck: case Head::discard: case Head::discard_hand:
        return {hand_card(atom::kHandDiscard, o.a[0])};
      case Head::display: case Head::take_small: return {display(atom::kDisplayTake, o.a[0])};
      case Head::discard_display: return {display(atom::kDisplayDiscard, o.a[0])};
      case Head::snap: return {display(atom::kDisplaySnap, o.a[0])};
      case Head::play:
        if (o.n >= 2) return {hand_card(atom::kHand, o.a[0]), building(o.a[1])};
        return {hand_card(atom::kHand, o.a[0])};
      case Head::play_display:
        if (o.n >= 2) return {display(atom::kDisplayTake, o.a[0]), building(o.a[1])};
        return {display(atom::kDisplayTake, o.a[0])};
      case Head::support: case Head::release: return project_atoms(project_slot(o.a[0]), o);
      case Head::play_project: case Head::play_release: return project_atoms(hand_card(atom::kHand, o.a[0]), o);
      case Head::play_project_display: case Head::play_release_display:
        return project_atoms(display(atom::kDisplayTake, o.a[0]), o);
      case Head::build: case Head::engineer: case Head::place: return placement(o);
      default: action_error(std::string("unsupported option head ") + head_name(o.head));
    }
  }

  // The 4 discards are a set. With the atoms chosen so far as the head, the option is listed once
  // per discard that can come next, so every remaining discard is a legal next atom.
  void setup_sequences(const Option& o, std::vector<AtomSequence>& out) const {
    AtomSequence discards;
    for (CardIdx c : pl_.hand_cards()) {
      if (std::find(o.a, o.a + o.n, c) == o.a + o.n) discards.push_back(hand_card(atom::kHandDiscard, c));
    }
    const AtomSequence& chosen = st_.prefix;
    for (ActionId a : chosen) {
      if (!contains(discards, a)) return;  // this option keeps a card that was already discarded
    }
    AtomSequence rest;
    for (ActionId a : discards) {
      if (!contains(chosen, a)) rest.push_back(a);
    }
    if (chosen.size() + rest.size() != discards.size()) action_error("a setup discard was chosen twice");
    if (rest.empty()) {
      out.push_back(chosen);
      return;
    }
    for (std::size_t next = 0; next < rest.size(); ++next) {
      AtomSequence seq = chosen;
      seq.push_back(rest[next]);
      for (std::size_t i = 0; i < rest.size(); ++i) {
        if (i != next) seq.push_back(rest[i]);
      }
      out.push_back(std::move(seq));
    }
  }

  ActionId hand_card(int base, int c) const {
    if (c < 0 || c >= num_cards() || !pl_.hand[c]) action_error("card not in the hand");
    return base + c;
  }

  ActionId display(int base, int index) const {
    if (index < 0 || index >= kDisplaySize || s_.display[index] == kNoCard) action_error("empty display folder");
    return base + index;
  }

  ActionId private_slot(int c) const {
    int n_off, list_off;
    private_list_offsets(f_.kind, &n_off, &list_off);
    if (n_off < 0) action_error("private card choice outside a private frame");
    for (int j = 0; j < f_.d[n_off]; ++j) {
      if (f_.d[list_off + j] == c) return atom::kPrivate + j;
    }
    action_error("card not in the private list");
  }

  ActionId public_slot(int c) const {
    int j = -1;
    if (f_.kind == FrameKind::ab_hunter) {
      for (int k = 0; k < f_.d[fd::cards_n]; ++k) {
        if (f_.d[fd::cards_list + k] == c) j = k;
      }
    } else if (f_.kind == FrameKind::ab_project) {
      auto it = std::find(s_.base_project_deck.begin(), s_.base_project_deck.end(), c);
      if (it != s_.base_project_deck.end()) j = static_cast<int>(it - s_.base_project_deck.begin());
    } else {
      action_error("public list choice outside ab_hunter / ab_project");
    }
    if (j < 0) action_error("card not in the public list");
    // A Hunter reveal cannot be longer (it is a frame list); the base project pile has at most
    // the base project cards not in play.
    if (j >= atom::kPublicSlots) action_error("public list longer than its slots");
    return atom::kPublic + j;
  }

  ActionId final_slot(int c) const {
    for (int i = 0; i < pl_.n_final; ++i) {
      if (pl_.final_cards[i] == c) return atom::kFinal + i;
    }
    action_error("card not among the final cards");
  }

  ActionId zoo_card(int base, int c) const {
    for (int i = 0; i < pl_.n_zoo_cards; ++i) {
      if (pl_.zoo_cards[i] == c) return base + c;
    }
    action_error("card not in the zoo");
  }

  ActionId project_slot(int c) const {
    for (int i = 0; i < s_.n_base_projects; ++i) {
      if (s_.base_projects[i] == c) return atom::kProject + i;
    }
    for (int i = 0; i < s_.n_projects; ++i) {
      if (s_.projects[i] == c) return atom::kProject + kBaseProjects + i;
    }
    action_error("project not in play");
  }

  ActionId building(int b) const {
    if (b < 0 || b >= pl_.zoo.n_buildings) action_error("bad building id");
    return atom::kBuilding + b;
  }

  AtomSequence project_atoms(ActionId first, const Option& o) const {
    ActionId extra = o.a[3] == kExtraNone ? atom::kNoExtra : zoo_card(atom::kZooGive, o.a[4]);
    return {first, atom::kPosition + o.a[1] - 1, atom::kMapToken + o.a[2], extra};
  }

  AtomSequence placement(const Option& o) const {
    BuildingKind kind;
    int size;
    Shape shape;
    if (o.n >= 2) {
      kind = static_cast<BuildingKind>(o.a[0]);
      size = o.a[1];
      shape = standard_shape(kind, size);
    } else {
      if (f_.kind != FrameKind::sponsor_building) action_error("placement without a building kind");
      CardIdx c = f_.d[sponsor_building_card];
      kind = BuildingKind::kUnique;
      size = card(c).building.size;
      shape = card_shape(c);
    }
    int verb = o.head == Head::build ? 0 : o.head == Head::engineer ? 1 : 2;
    int orient = placement_orientation(shape, o.cells);
    return {atom::kBuildingType + verb * atom::kBuildingTypes + building_type(kind, size),
            atom::cell(lowest_cell(o.cells), orient)};
  }

  const ArknovaState& st_;
  const State& s_;
  const Frame& f_;
  int p_;
  const Player& pl_;
};

}  // namespace

int building_type(BuildingKind kind, int size) {
  switch (kind) {
    case BuildingKind::kEnclosure:
      if (size < 1 || size > 5) action_error("enclosure size out of range");
      return size - 1;
    case BuildingKind::kKiosk: return 5;
    case BuildingKind::kPavilion: return 6;
    case BuildingKind::kPettingZoo: return 7;
    case BuildingKind::kReptileHouse: return 8;
    case BuildingKind::kAviary: return 9;
    case BuildingKind::kUnique: return 10;
    default: action_error("unknown building kind");
  }
}

Decision current_decision(const ArknovaState& st) {
  State& s = st.synced_core();
  if (s.over || !s.stack_n) action_error("no pending decision");
  std::vector<Option> options;
  for (const Option& o : legal_options(s)) {
    if (std::find(options.begin(), options.end(), o) == options.end()) options.push_back(o);
  }
  if (options.empty()) action_error("decision without options");
  Mapper map(st, s);
  Decision d;
  d.options.reserve(options.size());
  d.atoms.reserve(options.size());
  for (const Option& o : options) {
    const std::size_t before = d.atoms.size();
    map.sequences(o, d.atoms);
    d.options.insert(d.options.end(), d.atoms.size() - before, o);
  }
  check_factored_options(d.atoms);
  return d;
}

bool prefix_consistent(const ArknovaState& st) {
  if (st.prefix.empty()) return true;
  State& s = st.synced_core();
  Mapper map(st, s);
  std::vector<AtomSequence> seqs;
  for (const Option& o : legal_options(s)) {
    seqs.clear();
    map.sequences(o, seqs);
    for (const AtomSequence& seq : seqs) {
      if (atom_prefix_matches(seq, st.prefix)) return true;
    }
  }
  return false;
}

void canonicalize_prefix(ArknovaState& st) {
  const State& s = st.core;  // frame kinds are public, so the core need not be synced
  if (!s.over && s.stack_n && s.top().kind == FrameKind::setup_keep) std::sort(st.prefix.begin(), st.prefix.end());
}

std::string atom_name(ActionId a) {
  static const char* kSimpleNames[] = {"skip", "done", "no", "yes", "money", "deck", "donate",
                                       "reputation", "repeat", "worker", "replenish", "give_card", "break",
                                       "no_extra", "hidden_hand_card", "give_money"};
  static const char* kTypeNames[] = {"enclosure1", "enclosure2", "enclosure3", "enclosure4", "enclosure5", "kiosk",
                                     "pavilion", "petting_zoo", "reptile_house", "aviary", "unique"};
  static const char* kVerbNames[] = {"build", "engineer", "place"};
  auto in = [a](int lo, int n) { return a >= lo && a < lo + n; };
  auto num = [](int v) { return std::to_string(v); };
  using namespace atom;
  if (in(kSimple, kNumSimple)) return kSimpleNames[a];
  if (in(kAct, kNumActions * kMaxX)) {
    int i = a - kAct;
    return std::string("act:") + kActionNames[i / kMaxX] + "+" + num(i % kMaxX);
  }
  if (in(kXtoken, kNumActions)) return std::string("xtoken:") + kActionNames[a - kXtoken];
  if (in(kUpgrade, kNumActions)) return std::string("upgrade:") + kActionNames[a - kUpgrade];
  if (in(kClever, kNumActions)) return std::string("clever:") + kActionNames[a - kClever];
  if (in(kMultiplier, kNumActions)) return std::string("multiplier:") + kActionNames[a - kMultiplier];
  if (in(kBoost, 2)) return a == kBoost ? "boost:1" : "boost:5";
  if (in(kPartnerZoo, kNumContinents)) return std::string("partner_zoo:") + kContinentNames[a - kPartnerZoo];
  if (in(kUniversity, kNumUniversities)) return std::string("university:") + kUniversityNames[a - kUniversity];
  if (in(kTile, kNumBonusTiles)) return std::string("tile:") + kBonusTileNames[a - kTile];
  if (in(kWaza, 2)) return a == kWaza ? "kind:small" : "kind:large";
  if (in(kTarget, kTargets)) return a == kTarget ? "target:self" : "target:opponent";
  if (in(kBuildingType, kBuildVerbs * kBuildingTypes)) {
    int i = a - kBuildingType;
    return std::string(kVerbNames[i / kBuildingTypes]) + ":" + kTypeNames[i % kBuildingTypes];
  }
  if (in(kPosition, 3)) return "position:" + num(a - kPosition + 1);
  if (in(kMapToken, kNumMapTokens)) return "map_token:" + num(a - kMapToken);
  if (in(kHand, kCardSlots)) return "hand:" + num(a - kHand);
  if (in(kHandDiscard, kCardSlots)) return "hand_discard:" + num(a - kHandDiscard);
  if (in(kDisplayTake, kDisplaySize)) return "display:" + num(a - kDisplayTake);
  if (in(kDisplayDiscard, kDisplaySize)) return "display_discard:" + num(a - kDisplayDiscard);
  if (in(kDisplaySnap, kDisplaySize)) return "snap:" + num(a - kDisplaySnap);
  if (in(kProject, kProjectSlots)) return "project:" + num(a - kProject);
  if (in(kZoo, kCardSlots)) return "zoo:" + num(a - kZoo);
  if (in(kZooGive, kCardSlots)) return "zoo_give:" + num(a - kZooGive);
  if (in(kPublic, kPublicSlots)) return "list:" + num(a - kPublic);
  if (in(kPrivate, kPrivateCards)) return "private:" + num(a - kPrivate);
  if (in(kFinal, kMaxFinalCards)) return "final_discard:" + num(a - kFinal);
  if (in(kBuilding, kMaxBuildings)) return "enclosure:" + num(a - kBuilding);
  if (in(kCell, kMapCells * kCellHeads)) {
    int i = a - kCell;
    return "cell:" + num(i / kCellHeads) + "/o" + num(i % kCellHeads);
  }
  action_error("atom out of range: " + num(a));
}

}  // namespace board_ai::arknova
