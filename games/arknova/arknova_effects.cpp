// Shared game effects (ark-nova-ai arknova/effects.py): gains, track bonuses, the display,
// placing buildings, and the small choice frames those push.

#include <algorithm>

#include "arknova_internal.h"

namespace board_ai::arknova {

// ---- options -------------------------------------------------------------------------

Option opt(Head h) {
  Option o;
  o.head = h;
  return o;
}
Option opt(Head h, int a0) {
  Option o = opt(h);
  o.n = 1;
  o.a[0] = static_cast<std::int16_t>(a0);
  return o;
}
Option opt(Head h, int a0, int a1) {
  Option o = opt(h, a0);
  o.n = 2;
  o.a[1] = static_cast<std::int16_t>(a1);
  return o;
}
Option opt(Head h, int a0, int a1, int a2) {
  Option o = opt(h, a0, a1);
  o.n = 3;
  o.a[2] = static_cast<std::int16_t>(a2);
  return o;
}
Option opt_skip() { return opt(Head::skip); }

bool Option::operator==(const Option& o) const {
  if (head != o.head || n != o.n || cells != o.cells) return false;
  for (int i = 0; i < n; ++i) {
    if (a[i] != o.a[i]) return false;
  }
  return true;
}

// ---- deck and display ------------------------------------------------------------------

CardIdx draw_from_deck(State& s, Chance& ch) {
  if (s.deck.empty()) {
    s.deck.swap(s.discard);
    s.discard.clear();
    ch.shuffle(s.deck);
  }
  if (s.deck.empty()) return kNoCard;
  CardIdx c = s.deck.back();
  s.deck.pop_back();
  return c;
}

void to_hand(State& s, int p, CardIdx c, bool is_public) {
  Player& pl = s.players[p];
  if (c == kNoCard) engine_error("to_hand: no card");
  if (pl.hand[c]) engine_error("to_hand: card already in hand");
  pl.hand[c] = 1;
  pl.hand_known[c] = is_public ? 1 : 0;
  ++pl.hand_size;
}

void remove_from_hand(State& s, int p, CardIdx c) {
  Player& pl = s.players[p];
  if (c < 0 || !pl.hand[c]) engine_error("remove_from_hand: card not in hand");
  pl.hand[c] = 0;
  pl.hand_known[c] = 0;
  pl.maybe_tucked[c] = 0;
  --pl.hand_size;
}

std::vector<int> cards_in_range(const State& s, int p) {
  int folders = reputation_folders(s.players[p].reputation);
  std::vector<int> out;
  for (int i = 0; i < folders; ++i) {
    if (s.display[i] != kNoCard) out.push_back(i);
  }
  return out;
}

CardIdx take_from_display(State& s, int index) {
  CardIdx c = s.display[index];
  if (c == kNoCard) engine_error("take_from_display: empty folder");
  s.display[index] = kNoCard;
  return c;
}

void refill_display(State& s, Chance& ch) {
  CardIdx kept[kDisplaySize];
  int n = 0;
  for (CardIdx c : s.display) {
    if (c != kNoCard) kept[n++] = c;
  }
  while (n < kDisplaySize) {
    CardIdx c = draw_from_deck(s, ch);
    if (c == kNoCard) break;
    kept[n++] = c;
  }
  for (int i = 0; i < kDisplaySize; ++i) s.display[i] = i < n ? kept[i] : kNoCard;
}

void advance_break(State& s, int p, int steps) {
  if (s.break_by >= 0 || steps <= 0) return;
  s.break_pos = static_cast<std::int8_t>(std::min(s.break_pos + steps, kBreakTrack));
  if (s.break_pos >= kBreakTrack) {
    s.break_by = static_cast<std::int8_t>(p);
    gain(s, p, EffectKind::kEffXtoken, 1);
  }
}

bool is_upgraded(const State& s, int p, int action) {
  if (s.borrowed_action >= 0 && s.borrowed_action == action) return s.borrowed_upgraded;
  return s.players[p].is_upgraded(action);
}

// ---- action card tokens ------------------------------------------------------------

bool affected(const State& s, int q) {
  const Player& pl = s.players[q];
  return pl.appeal >= kInteractiveMinAppeal && !pl.in_zoo(card_by_id(225));
}

bool has_token(const State& s, int p, int action, TokenKind t) {
  const Player& pl = s.players[p];
  for (int i = 0; i < pl.n_action_tokens[action]; ++i) {
    if (pl.action_tokens[action][i] == t) return true;
  }
  return false;
}

bool add_action_token(State& s, int p, int action, TokenKind t) {
  Player& pl = s.players[p];
  if (t != TokenKind::kMultiplier && has_token(s, p, action, t)) return false;
  if (pl.n_action_tokens[action] >= kMaxActionTokens) engine_error("too many action tokens");
  pl.action_tokens[action][pl.n_action_tokens[action]++] = t;
  return true;
}

void remove_action_token(State& s, int p, int action, TokenKind t) {
  Player& pl = s.players[p];
  int n = pl.n_action_tokens[action];
  for (int i = 0; i < n; ++i) {
    if (pl.action_tokens[action][i] == t) {
      for (int j = i + 1; j < n; ++j) pl.action_tokens[action][j - 1] = pl.action_tokens[action][j];
      --pl.n_action_tokens[action];
      return;
    }
  }
}

bool has_venom(const State& s, int p) {
  for (int a = 0; a < kNumActions; ++a) {
    if (has_token(s, p, a, TokenKind::kVenom)) return true;
  }
  return false;
}

// ---- icons ------------------------------------------------------------------------

IconCounts icons(const Player& pl) {
  IconCounts counts{};
  for (int i = 0; i < pl.n_zoo_cards; ++i) {
    const auto& ci = card_icons(pl.zoo_cards[i]);
    for (int k = 0; k < kNumIcons; ++k) counts[k] += ci[k];
  }
  for (int i = 0; i < pl.n_partner_zoos; ++i) ++counts[kContinentIcon0 + pl.partner_zoos[i]];
  for (int i = 0; i < pl.n_universities; ++i) {
    if (pl.universities[i] == kUniScience2) counts[kIcon_science] += 2;
    if (pl.universities[i] == kUniScienceRep) counts[kIcon_science] += 1;
  }
  return counts;
}

void icons_played(State& s, int p, const IconCounts& added, const IconCounts& before, Chance& ch) {
  bool any = false;
  std::vector<int> new_icons;
  for (int k = 0; k < kNumIcons; ++k) {
    if (!added[k]) continue;
    any = true;
    if (!before[k]) new_icons.push_back(k);
  }
  if (!any) return;
  for (int step = 0; step < kPlayers; ++step) {
    int owner = (p + step) % kPlayers;
    // The trigger may add cards to the zoo; iterate over a copy like the Python list(...).
    const Player& o = s.players[owner];
    std::vector<CardIdx> zoo(o.zoo_cards, o.zoo_cards + o.n_zoo_cards);
    for (CardIdx c : zoo) {
      if (has_icon_trigger(c)) icon_trigger(s, c, owner, p, added, new_icons, ch);
    }
  }
}

void score(State& s, int p, EffectKind kind, int amount) {
  Player& pl = s.players[p];
  if (kind == EffectKind::kEffConservation) {
    pl.conservation += amount;
  } else if (kind == EffectKind::kEffAppeal) {
    pl.appeal += amount;
  } else {
    engine_error("score: bad kind");
  }
}

// ---- gains ------------------------------------------------------------------------

namespace {

void push_free_build(State& s, int p, std::initializer_list<std::pair<BuildingKind, int>> kinds) {
  int fi = push(s, FrameKind::free_build, p);
  Frame& f = frame(s, fi);
  f.d[fd::free_build_n] = static_cast<std::int16_t>(kinds.size());
  int i = fd::free_build_kinds;
  for (auto [k, size] : kinds) {
    f.d[i++] = static_cast<std::int16_t>(k);
    f.d[i++] = static_cast<std::int16_t>(size);
  }
}

void gain_conservation(State& s, int p, int amount) {
  Player& pl = s.players[p];
  int old = pl.conservation;
  pl.conservation += amount;
  int now = pl.conservation;
  auto crossed = [&](int space) { return old < space && space <= now; };
  if (crossed(kConservationFinalCardDiscard) && !s.final_card_discarded) push_final_card_discards(s);
  for (int t = 1; t >= 0; --t) {
    if (crossed(kConservationTileSpaces[t])) {
      int fi = push(s, FrameKind::conservation_tile, p);
      frame(s, fi).d[fd::tile_space] = static_cast<std::int16_t>(kConservationTileSpaces[t]);
    }
  }
  if (crossed(kConservationUpgradeOrWorker)) push(s, FrameKind::upgrade_or_worker, p);
}

void gain_reputation(State& s, int p, int amount) {
  Player& pl = s.players[p];
  for (int i = 0; i < amount; ++i) {
    if (pl.reputation >= kReputationMax) {
      pl.appeal += 1;
      continue;
    }
    if (pl.reputation >= kReputationCapWithoutCardsII && !pl.is_upgraded(kCards)) break;
    ++pl.reputation;
    bool present = false;
    Effect bonus = reputation_bonus(pl.reputation, &present);
    if (present) gain(s, p, bonus.kind, 1);
  }
}

void apply_effect(State& s, int p, const Effect& e) {
  if (e.amount) gain(s, p, e.kind, e.amount);
}

}  // namespace

void gain(State& s, int p, EffectKind kind, int amount) {
  Player& pl = s.players[p];
  if (amount <= 0) return;
  switch (kind) {
    case EffectKind::kEffMoney: pl.money += amount; break;
    case EffectKind::kEffAppeal: pl.appeal += amount; break;
    case EffectKind::kEffXtoken: pl.xtokens = static_cast<std::int8_t>(std::min(kMaxXtokens, pl.xtokens + amount)); break;
    case EffectKind::kEffConservation: gain_conservation(s, p, amount); break;
    case EffectKind::kEffReputation: gain_reputation(s, p, amount); break;
    case EffectKind::kEffWorker:
      for (int i = 0; i < amount; ++i) {
        if (pl.workers < kMaxWorkers) {
          ++pl.workers;
          if (pl.workers == kMaxWorkers) apply_effect(s, p, kMap0LastWorker);
        }
      }
      break;
    case EffectKind::kEffUpgrade:
      for (int i = 0; i < amount; ++i) push(s, FrameKind::upgrade, p);
      break;
    case EffectKind::kEffCard:
      for (int i = 0; i < amount; ++i) push(s, FrameKind::take_card, p);
      break;
    case EffectKind::kEffClever:
      for (int i = 0; i < amount; ++i) push(s, FrameKind::clever, p);
      break;
    case EffectKind::kEffEnclosure: push_free_build(s, p, {{BuildingKind::kEnclosure, amount}}); break;
    case EffectKind::kEffKiosk:
      for (int i = 0; i < amount; ++i) push_free_build(s, p, {{BuildingKind::kKiosk, 1}});
      break;
    case EffectKind::kEffMultiplier: push(s, FrameKind::place_multiplier, p); break;
    case EffectKind::kEffPartnerZoo: push(s, FrameKind::gain_partner_zoo, p); break;
    case EffectKind::kEffUniversity: push(s, FrameKind::gain_university, p); break;
    case EffectKind::kEffSponsorForMoney: push(s, FrameKind::paid_sponsor, p); break;
  }
}

void push_final_card_discards(State& s) {
  s.final_card_discarded = true;
  for (int q = kPlayers - 1; q >= 0; --q) {
    if (s.players[q].n_final > 1) push(s, FrameKind::discard_final, q);
  }
}

void take_map_token(State& s, int p, int index) {
  Player& pl = s.players[p];
  if (!pl.tokens[index]) engine_error("take_map_token: token already taken");
  pl.tokens[index] = false;
  apply_effect(s, p, kMap0Tokens[index].effect);
}

void token_income(State& s, int p) {
  for (int i = 0; i < kNumMapTokens; ++i) {
    if (kMap0Tokens[i].repeats && !s.players[p].tokens[i]) apply_effect(s, p, kMap0Tokens[i].effect);
  }
}

bool can_take_partner_zoo(const State& s, int p, int continent) {
  const Player& pl = s.players[p];
  int slot = pl.n_partner_zoos;
  return (s.partner_zoos_available >> continent & 1) && !pl.has_partner_zoo(continent) && slot < kMaxPartnerZoos &&
         (slot < 2 || pl.is_upgraded(kAssociation));
}

void take_partner_zoo(State& s, int p, int continent, Chance& ch) {
  Player& pl = s.players[p];
  int slot = pl.n_partner_zoos;
  IconCounts before = icons(pl);
  pl.partner_zoos[pl.n_partner_zoos++] = static_cast<std::int8_t>(continent);
  s.partner_zoos_available &= static_cast<std::uint8_t>(~(1u << continent));
  IconCounts added{};
  added[kContinentIcon0 + continent] = 1;
  icons_played(s, p, added, before, ch);
  apply_effect(s, p, kMap0PartnerZooSlots[slot]);
}

bool can_take_university(const State& s, int p, int university) {
  const Player& pl = s.players[p];
  return (s.universities_available >> university & 1) && !pl.has_university(static_cast<University>(university)) &&
         pl.n_universities < kNumUniversities;
}

void take_university(State& s, int p, int university, Chance& ch) {
  Player& pl = s.players[p];
  int slot = pl.n_universities;
  IconCounts before = icons(pl);
  pl.universities[pl.n_universities++] = static_cast<std::int8_t>(university);
  s.universities_available &= static_cast<std::uint8_t>(~(1u << university));
  IconCounts added{};
  if (university == kUniScience2) added[kIcon_science] = 2;
  if (university == kUniScienceRep) added[kIcon_science] = 1;
  icons_played(s, p, added, before, ch);
  apply_effect(s, p, kMap0UniversitySlots[slot]);
  gain(s, p, EffectKind::kEffReputation, university == kUniHand ? 1 : university == kUniScienceRep ? 2 : 0);
}

// ---- buildings --------------------------------------------------------------------

int place_building(State& s, int p, BuildingKind kind, int size, CellMask cells, CardIdx name) {
  Player& pl = s.players[p];
  std::vector<PlacementBonus> bonuses;
  int b = pl.zoo.place(kind, size, cells, name, &bonuses);
  after_build_hooks(s, p, b);
  if (kind == BuildingKind::kPavilion) gain(s, p, EffectKind::kEffAppeal, 1);
  if (!pl.full_cover_scored && pl.zoo.is_fully_covered()) {
    pl.full_cover_scored = true;
    gain(s, p, EffectKind::kEffAppeal, kFullCoverAppeal);
  }
  for (auto it = bonuses.rbegin(); it != bonuses.rend(); ++it) apply_effect(s, p, placement_bonus_effect(*it));
  return b;
}

// ---- choice frames ------------------------------------------------------------------

namespace {

void upgrade_options(State& s, int fi, std::vector<Option>& out) {
  const Player& pl = s.players[frame(s, fi).player];
  std::size_t n0 = out.size();
  for (int a = 0; a < kNumActions; ++a) {
    if (!pl.is_upgraded(a)) out.push_back(opt(Head::upgrade, a));
  }
  if (out.size() == n0) out.push_back(opt_skip());
}

void upgrade_apply(State& s, int fi, const Option& o, Chance&) {
  int p = frame(s, fi).player;
  pop(s, fi);
  if (o.head == Head::upgrade) s.players[p].upgraded |= static_cast<std::uint8_t>(1u << o.a[0]);
}

void upgrade_or_worker_options(State& s, int fi, std::vector<Option>& out) {
  const Player& pl = s.players[frame(s, fi).player];
  std::vector<Option> opts;
  upgrade_options(s, fi, opts);
  if (pl.workers < kMaxWorkers) {
    for (const Option& o : opts) {
      if (o.head != Head::skip) out.push_back(o);
    }
    out.push_back(opt(Head::worker));
  } else {
    out.insert(out.end(), opts.begin(), opts.end());
  }
}

void upgrade_or_worker_apply(State& s, int fi, const Option& o, Chance& ch) {
  if (o.head == Head::worker) {
    int p = frame(s, fi).player;
    pop(s, fi);
    gain(s, p, EffectKind::kEffWorker, 1);
  } else {
    upgrade_apply(s, fi, o, ch);
  }
}

int tile_space_index(int space) { return space == kConservationTileSpaces[0] ? 0 : 1; }

void tile_options(State& s, int fi, std::vector<Option>& out) {
  int t = tile_space_index(frame(s, fi).d[fd::tile_space]);
  out.push_back(opt(Head::money));
  for (int i = 0; i < s.n_bonus_tiles[t]; ++i) out.push_back(opt(Head::tile, s.bonus_tiles[t][i]));
}

void tile_apply(State& s, int fi, const Option& o, Chance&) {
  int p = frame(s, fi).player;
  int t = tile_space_index(frame(s, fi).d[fd::tile_space]);
  pop(s, fi);
  if (o.head == Head::money) {
    gain(s, p, EffectKind::kEffMoney, kConservationTileMoney);
    return;
  }
  int n = s.n_bonus_tiles[t];
  int i = 0;
  while (i < n && s.bonus_tiles[t][i] != o.a[0]) ++i;
  if (i == n) engine_error("conservation_tile: tile not available");
  for (int j = i + 1; j < n; ++j) s.bonus_tiles[t][j - 1] = s.bonus_tiles[t][j];
  --s.n_bonus_tiles[t];
  apply_effect(s, p, bonus_tile_effect(static_cast<BonusTile>(o.a[0])));
}

void take_card_options(State& s, int fi, std::vector<Option>& out) {
  std::size_t n0 = out.size();
  if (!s.deck.empty() || !s.discard.empty()) out.push_back(opt(Head::deck));
  for (int i : cards_in_range(s, frame(s, fi).player)) out.push_back(opt(Head::display, i));
  if (out.size() == n0) out.push_back(opt_skip());
}

void take_card_apply(State& s, int fi, const Option& o, Chance& ch) {
  int p = frame(s, fi).player;
  pop(s, fi);
  if (o.head == Head::deck) {
    to_hand(s, p, draw_from_deck(s, ch), false);
  } else if (o.head == Head::display) {
    to_hand(s, p, take_from_display(s, o.a[0]), true);
    if (s.turn_depth == 0) refill_display(s, ch);
  }
}

void clever_options(State& s, int fi, std::vector<Option>& out) {
  const Player& pl = s.players[frame(s, fi).player];
  for (int i = 1; i < kNumActions; ++i) out.push_back(opt(Head::clever, pl.slots[i]));
  out.push_back(opt_skip());
}

void clever_apply(State& s, int fi, const Option& o, Chance&) {
  int p = frame(s, fi).player;
  pop(s, fi);
  if (o.head != Head::clever) return;
  Player& pl = s.players[p];
  auto* it = std::find(pl.slots, pl.slots + kNumActions, o.a[0]);
  std::rotate(pl.slots, it, it + 1);
}

void free_build_options(State& s, int fi, std::vector<Option>& out) {
  const Frame& f = frame(s, fi);
  const Player& pl = s.players[f.player];
  for (int k = 0; k < f.d[fd::free_build_n]; ++k) {
    PlacementQuery q;
    q.kind = static_cast<BuildingKind>(f.d[fd::free_build_kinds + 2 * k]);
    q.size = f.d[fd::free_build_kinds + 2 * k + 1];
    q.build2 = pl.is_upgraded(kBuild);
    q.any_terrain = pl.in_zoo(card_by_id(219));
    for (CellMask c : pl.zoo.legal_placements(q)) {
      Option o = opt(Head::place, static_cast<int>(q.kind), q.size);
      o.cells = c;
      out.push_back(o);
    }
  }
  out.push_back(opt_skip());
}

void free_build_apply(State& s, int fi, const Option& o, Chance&) {
  int p = frame(s, fi).player;
  pop(s, fi);
  if (o.head == Head::place) place_building(s, p, static_cast<BuildingKind>(o.a[0]), o.a[1], o.cells);
}

void multiplier_options(State& s, int, std::vector<Option>& out) {
  for (int a = 0; a < kNumActions; ++a) out.push_back(opt(Head::multiplier, a));
}

void multiplier_apply(State& s, int fi, const Option& o, Chance&) {
  int p = frame(s, fi).player;
  pop(s, fi);
  add_action_token(s, p, o.a[0], TokenKind::kMultiplier);
}

void partner_zoo_options(State& s, int fi, std::vector<Option>& out) {
  std::size_t n0 = out.size();
  for (int c = 0; c < kNumContinents; ++c) {
    if (can_take_partner_zoo(s, frame(s, fi).player, c)) out.push_back(opt(Head::partner_zoo, c));
  }
  if (out.size() == n0) out.push_back(opt_skip());
}

void partner_zoo_apply(State& s, int fi, const Option& o, Chance& ch) {
  int p = frame(s, fi).player;
  pop(s, fi);
  if (o.head == Head::partner_zoo) take_partner_zoo(s, p, o.a[0], ch);
}

void university_options(State& s, int fi, std::vector<Option>& out) {
  std::size_t n0 = out.size();
  for (int u = 0; u < kNumUniversities; ++u) {
    if (can_take_university(s, frame(s, fi).player, u)) out.push_back(opt(Head::university, u));
  }
  if (out.size() == n0) out.push_back(opt_skip());
}

void university_apply(State& s, int fi, const Option& o, Chance& ch) {
  int p = frame(s, fi).player;
  pop(s, fi);
  if (o.head == Head::university) take_university(s, p, o.a[0], ch);
}

void discard_final_options(State& s, int fi, std::vector<Option>& out) {
  const Player& pl = s.players[frame(s, fi).player];
  for (int i = 0; i < pl.n_final; ++i) out.push_back(opt(Head::discard_final, pl.final_cards[i]));
}

void discard_final_apply(State& s, int fi, const Option& o, Chance&) {
  Player& pl = s.players[frame(s, fi).player];
  pop(s, fi);
  auto* end = pl.final_cards + pl.n_final;
  auto* it = std::find(pl.final_cards, end, o.a[0]);
  if (it == end) engine_error("discard_final: card not held");
  std::copy(it + 1, end, it);
  --pl.n_final;
  s.final_deck.insert(s.final_deck.begin(), static_cast<CardIdx>(o.a[0]));
}

void discard_hand_options(State& s, int fi, std::vector<Option>& out) {
  for (CardIdx c : s.players[frame(s, fi).player].hand_cards()) out.push_back(opt(Head::discard, c));
}

void discard_hand_apply(State& s, int fi, const Option& o, Chance&) {
  Frame& f = frame(s, fi);
  Player& pl = s.players[f.player];
  remove_from_hand(s, f.player, o.a[0]);
  s.discard.push_back(o.a[0]);
  f.d[fd::discard_count] = static_cast<std::int16_t>((f.d[fd::discard_count] == kNull ? 1 : f.d[fd::discard_count]) - 1);
  int limit = f.d[fd::discard_limit];
  if ((limit != kNull && pl.hand_size <= limit) || (limit == kNull && f.d[fd::discard_count] <= 0) || pl.hand_size == 0) {
    pop(s, fi);
  }
}

}  // namespace

void register_effect_frames() {
  register_choice(FrameKind::upgrade, upgrade_options, upgrade_apply);
  register_choice(FrameKind::upgrade_or_worker, upgrade_or_worker_options, upgrade_or_worker_apply);
  register_choice(FrameKind::conservation_tile, tile_options, tile_apply);
  register_choice(FrameKind::take_card, take_card_options, take_card_apply);
  register_choice(FrameKind::clever, clever_options, clever_apply);
  register_choice(FrameKind::free_build, free_build_options, free_build_apply);
  register_choice(FrameKind::place_multiplier, multiplier_options, multiplier_apply);
  register_choice(FrameKind::gain_partner_zoo, partner_zoo_options, partner_zoo_apply);
  register_choice(FrameKind::gain_university, university_options, university_apply);
  register_choice(FrameKind::discard_final, discard_final_options, discard_final_apply);
  register_choice(FrameKind::discard_hand, discard_hand_options, discard_hand_apply);
}

}  // namespace board_ai::arknova
