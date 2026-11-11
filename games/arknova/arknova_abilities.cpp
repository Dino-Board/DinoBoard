// Animal abilities (ark-nova-ai arknova/abilities.py, abilities_basic.py, abilities_action.py).
// Each keyword resolves when its animal is played, last printed ability pushed first.

#include <algorithm>

#include "arknova_internal.h"

namespace board_ai::arknova {
namespace {

constexpr int kSunBathingPrice = 4;
constexpr int kPouchAppeal = 2;

std::vector<CardIdx> frame_list(const Frame& f, int n_at, int first) {
  return std::vector<CardIdx>(f.d + first, f.d + first + f.d[n_at]);
}

void set_frame_list(Frame& f, int n_at, int first, const std::vector<CardIdx>& cards) {
  if (first + static_cast<int>(cards.size()) > kFrameData) engine_error("frame card list too long");
  for (int i = first; i < kFrameData; ++i) f.d[i] = kNull;
  f.d[n_at] = static_cast<std::int16_t>(cards.size());
  std::copy(cards.begin(), cards.end(), f.d + first);
}

void push_left(State& s, FrameKind k, int p, int left) {
  int fi = push(s, k, p);
  frame(s, fi).d[fd::left] = static_cast<std::int16_t>(left);
}

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

void draw_into_hand(State& s, int p, int n, Chance& ch) {
  for (int i = 0; i < n; ++i) {
    CardIdx c = draw_from_deck(s, ch);
    if (c != kNoCard) to_hand(s, p, c, false);
  }
}

int icons_everywhere(const State& s, int icon) {
  int n = 0;
  for (const Player& pl : s.players) n += icons(pl)[icon];
  return n;
}

void discard_all(State& s, const std::vector<CardIdx>& cards) {
  s.discard.insert(s.discard.end(), cards.begin(), cards.end());
}

// ---- pick N from a drawn set (Perception, Scavenging) ------------------------------------

void pick_options(State& s, int fi, std::vector<Option>& out) {
  for (CardIdx c : frame_list(frame(s, fi), fd::pick_n, fd::pick_cards)) out.push_back(opt(Head::pick, c));
}

void pick_apply(State& s, int fi, const Option& o, Chance&) {
  Frame& f = frame(s, fi);
  std::vector<CardIdx> cards = frame_list(f, fd::pick_n, fd::pick_cards);
  cards.erase(std::find(cards.begin(), cards.end(), o.a[0]));
  set_frame_list(f, fd::pick_n, fd::pick_cards, cards);
  to_hand(s, f.player, o.a[0], false);
  --f.d[fd::pick_count];
  if (f.d[fd::pick_count] <= 0 || cards.empty()) {
    discard_all(s, cards);
    pop(s, fi);
  }
}

// ---- Hunter ------------------------------------------------------------------------

void hunter_options(State& s, int fi, std::vector<Option>& out) {
  for (CardIdx c : frame_list(frame(s, fi), fd::cards_n, fd::cards_list)) {
    if (is_animal(c)) out.push_back(opt(Head::take, c));
  }
  out.push_back(opt_skip());
}

void hunter_apply(State& s, int fi, const Option& o, Chance&) {
  const Frame f = frame(s, fi);
  pop(s, fi);
  std::vector<CardIdx> revealed = frame_list(f, fd::cards_n, fd::cards_list);
  if (o.head == Head::take) {
    revealed.erase(std::find(revealed.begin(), revealed.end(), o.a[0]));
    to_hand(s, f.player, o.a[0], true);
  }
  discard_all(s, revealed);
}

// ---- Snapping ------------------------------------------------------------------------

void snap_options(State& s, int fi, std::vector<Option>& out) {
  for (int i = 0; i < kDisplaySize; ++i) {
    if (s.display[i] != kNoCard) out.push_back(opt(Head::take, i));
  }
  if (frame(s, fi).d[fd::snap_can_replenish]) out.push_back(opt(Head::replenish));
  out.push_back(opt_skip());
}

void snap_apply(State& s, int fi, const Option& o, Chance& ch) {
  Frame& f = frame(s, fi);
  if (o.head == Head::skip) {
    pop(s, fi);
  } else if (o.head == Head::replenish) {
    f.d[fd::snap_can_replenish] = 0;
    refill_display(s, ch);
  } else {
    to_hand(s, f.player, take_from_display(s, o.a[0]), true);
    --f.d[fd::left];
    f.d[fd::snap_can_replenish] = f.d[fd::snap_replenish_between] && f.d[fd::left] > 0;
    f.d[fd::snap_replenish_between] = 0;
    if (f.d[fd::left] <= 0) pop(s, fi);
  }
}

// ---- Sun Bathing, Digging, Pouch ---------------------------------------------------------

void hand_options(State& s, int fi, std::vector<Option>& out, Head verb) {
  for (CardIdx c : s.players[frame(s, fi).player].hand_cards()) out.push_back(opt(verb, c));
  out.push_back(opt(Head::done));
}

void sun_options(State& s, int fi, std::vector<Option>& out) { hand_options(s, fi, out, Head::sell); }

void sun_apply(State& s, int fi, const Option& o, Chance&) {
  Frame& f = frame(s, fi);
  int p = f.player;
  Player& pl = s.players[p];
  --f.d[fd::left];
  if (o.head == Head::done || f.d[fd::left] <= 0 || (o.head == Head::sell && pl.hand_size <= 1)) pop(s, fi);
  if (o.head == Head::sell) {
    remove_from_hand(s, p, o.a[0]);
    s.discard.push_back(o.a[0]);
    gain(s, p, EffectKind::kEffMoney, kSunBathingPrice);
  }
}

void dig_options(State& s, int fi, std::vector<Option>& out) {
  for (int i = 0; i < kDisplaySize; ++i) {
    if (s.display[i] != kNoCard) out.push_back(opt(Head::discard_display, i));
  }
  for (CardIdx c : s.players[frame(s, fi).player].hand_cards()) out.push_back(opt(Head::discard_hand, c));
  out.push_back(opt(Head::done));
}

void dig_apply(State& s, int fi, const Option& o, Chance& ch) {
  Frame& f = frame(s, fi);
  int p = f.player;
  --f.d[fd::left];
  if (o.head == Head::done || f.d[fd::left] <= 0) pop(s, fi);
  if (o.head == Head::discard_display) {
    CardIdx gone = take_from_display(s, o.a[0]);
    refill_display(s, ch);
    s.discard.push_back(gone);
  } else if (o.head == Head::discard_hand) {
    CardIdx drawn = draw_from_deck(s, ch);
    remove_from_hand(s, p, o.a[0]);
    s.discard.push_back(o.a[0]);
    if (drawn != kNoCard) to_hand(s, p, drawn, false);
  }
}

void pouch_options(State& s, int fi, std::vector<Option>& out) { hand_options(s, fi, out, Head::tuck); }

void pouch_apply(State& s, int fi, const Option& o, Chance&) {
  Frame& f = frame(s, fi);
  int p = f.player;
  Player& pl = s.players[p];
  --f.d[fd::left];
  if (o.head == Head::done || f.d[fd::left] <= 0 || pl.hand_size <= 1) pop(s, fi);
  if (o.head == Head::tuck) {
    const bool seen = pl.hand_known[o.a[0]] || pl.maybe_tucked[o.a[0]];
    remove_from_hand(s, p, o.a[0]);
    if (pl.n_tucked >= kMaxTucked) engine_error("too many tucked cards");
    pl.tucked[pl.n_tucked++] = o.a[0];
    // Face down: the opponent cannot tell which of the cards it saw in the hand went.
    for (int c = 0; c < kCardSlots; ++c) {
      if (pl.hand[c] && pl.hand_known[c]) pl.maybe_tucked[c] = 1;
    }
    if (seen) pl.maybe_tucked[o.a[0]] = 1;
    gain(s, p, EffectKind::kEffAppeal, kPouchAppeal);
  }
}

// ---- Resistance, Assertion, Dominance ---------------------------------------------------

void resist_options(State& s, int fi, std::vector<Option>& out) {
  for (CardIdx c : frame_list(frame(s, fi), fd::cards_n, fd::cards_list)) out.push_back(opt(Head::keep, c));
}

void add_final_card(Player& pl, CardIdx c) {
  if (pl.n_final >= kMaxFinalCards) engine_error("too many final scoring cards");
  pl.final_cards[pl.n_final++] = c;
}

void resist_apply(State& s, int fi, const Option& o, Chance&) {
  const Frame f = frame(s, fi);
  pop(s, fi);
  add_final_card(s.players[f.player], o.a[0]);
  for (CardIdx c : frame_list(f, fd::cards_n, fd::cards_list)) {
    if (c != o.a[0]) s.final_deck.insert(s.final_deck.begin(), c);
  }
}

void project_options(State& s, int fi, std::vector<Option>& out) {
  int only = frame(s, fi).d[fd::project_only];
  for (CardIdx c : s.base_project_deck) {
    if (only == kNull || c == only) out.push_back(opt(Head::take, c));
  }
  out.push_back(opt_skip());
}

void project_apply(State& s, int fi, const Option& o, Chance&) {
  int p = frame(s, fi).player;
  pop(s, fi);
  if (o.head != Head::take) return;
  auto it = std::find(s.base_project_deck.begin(), s.base_project_deck.end(), o.a[0]);
  if (it == s.base_project_deck.end()) engine_error("ab_project: card not in the base project deck");
  s.base_project_deck.erase(it);
  to_hand(s, p, o.a[0], true);
}

// ---- Boost, Pilfering, Hypnosis -------------------------------------------------------------

void boost_options(State&, int, std::vector<Option>& out) {
  out.push_back(opt(Head::boost, 1));
  out.push_back(opt(Head::boost, 5));
  out.push_back(opt_skip());
}

void boost_apply(State& s, int fi, const Option& o, Chance&) {
  const Frame f = frame(s, fi);
  pop(s, fi);
  if (o.head != Head::boost) return;
  std::int8_t* slots = s.players[f.player].slots;
  int action = f.d[fd::boost_action];
  std::int8_t* it = std::find(slots, slots + kNumActions, action);
  std::copy(it + 1, slots + kNumActions, it);
  int pos = o.a[0] - 1;
  std::copy_backward(slots + pos, slots + kNumActions - 1, slots + kNumActions);
  slots[pos] = static_cast<std::int8_t>(action);
}

std::vector<int> targets(const State& s, int p, bool conservation, int minimum) {
  auto value = [&](int q) -> int { return conservation ? s.players[q].conservation : s.players[q].appeal; };
  int top = value(0);
  for (int q = 1; q < kPlayers; ++q) top = std::max(top, value(q));
  std::vector<int> out;
  if (top < minimum) return out;
  for (int q = 0; q < kPlayers; ++q) {
    if (value(q) == top && q != p && affected(s, q)) out.push_back(q);
  }
  return out;
}

void push_pilfer(State& s, int victim, int thief) {
  int fi = push(s, FrameKind::ab_pilfer, victim);
  frame(s, fi).d[fd::pilfer_thief] = static_cast<std::int16_t>(thief);
}

void push_candidates(State& s, FrameKind k, int p, const std::vector<int>& cands) {
  int fi = push(s, k, p);
  Frame& f = frame(s, fi);
  f.d[fd::candidates_n] = static_cast<std::int16_t>(cands.size());
  for (std::size_t i = 0; i < cands.size(); ++i) f.d[fd::candidates + i] = static_cast<std::int16_t>(cands[i]);
}

void pilfer(State& s, int p, bool conservation, int minimum) {
  std::vector<int> t = targets(s, p, conservation, minimum);
  if (t.size() == 1) {
    push_pilfer(s, t[0], p);
  } else if (!t.empty()) {
    push_candidates(s, FrameKind::ab_pilfer_pick, p, t);
  }
}

void candidate_options(State& s, int fi, std::vector<Option>& out) {
  const Frame& f = frame(s, fi);
  for (int i = 0; i < f.d[fd::candidates_n]; ++i) out.push_back(opt(Head::target, f.d[fd::candidates + i]));
}

void pilfer_pick_apply(State& s, int fi, const Option& o, Chance&) {
  int p = frame(s, fi).player;
  pop(s, fi);
  push_pilfer(s, o.a[0], p);
}

void pilfer_options(State& s, int fi, std::vector<Option>& out) {
  if (s.players[frame(s, fi).player].hand_size) out.push_back(opt(Head::card));
  out.push_back(opt(Head::money));
}

void pilfer_apply(State& s, int fi, const Option& o, Chance& ch) {
  const Frame f = frame(s, fi);
  pop(s, fi);
  int victim = f.player, thief = f.d[fd::pilfer_thief];
  if (o.head == Head::card) {
    CardIdx c = ch.pick(s.players[victim].hand_cards());
    remove_from_hand(s, victim, c);
    to_hand(s, thief, c, true);
  } else {
    int paid = std::min<int>(5, s.players[victim].money);
    s.players[victim].money -= paid;
    s.players[thief].money += paid;
  }
}

void offer_hypnosis(State& s, int p, int target, int reach) {
  unsigned mask = 0;
  int n = reach > 0 ? reach : 1;
  for (int i = 0; i < n && i < kNumActions; ++i) mask |= 1u << s.players[target].slots[i];
  offer_extra_action(s, p, mask, target);
}

void hypnosis_pick_apply(State& s, int fi, const Option& o, Chance&) {
  const Frame f = frame(s, fi);
  pop(s, fi);
  offer_hypnosis(s, f.player, o.a[0], f.d[fd::hypnosis_reach]);
}

// ---- keywords -------------------------------------------------------------------------

int action_of(Keyword k) {
  switch (k) {
    case Keyword::kBOOST_ANIMAL: return kAnimals;
    case Keyword::kBOOST_ASSOCIATION: case Keyword::kMULTIPLIER_ASSOCIATION: case Keyword::kACTION_ASSOCIATION:
      return kAssociation;
    case Keyword::kBOOST_BUILDING: case Keyword::kMULTIPLIER_BUILDING: case Keyword::kACTION_BUILDING: return kBuild;
    case Keyword::kBOOST_CARD: case Keyword::kMULTIPLIER_CARD: case Keyword::kACTION_CARDS: return kCards;
    case Keyword::kBOOST_SPONSORS: case Keyword::kMULTIPLIER_SPONSORS: case Keyword::kACTION_SPONSORS: return kSponsors;
    default: engine_error("keyword has no action");
  }
}

void resolve_keyword(State& s, int p, Keyword k, int value, Chance& ch) {
  Player& me = s.players[p];
  switch (k) {
    case Keyword::kFLOCK_ANIMAL: return;
    case Keyword::kSPRINT: draw_into_hand(s, p, value, ch); return;
    case Keyword::kJUMPING:
      advance_break(s, p, value);
      gain(s, p, EffectKind::kEffMoney, value);
      return;
    case Keyword::kINVENTIVE: gain(s, p, EffectKind::kEffXtoken, 1); return;
    case Keyword::kINVENTIVE_BEAR: gain(s, p, EffectKind::kEffXtoken, std::min(3, icons_everywhere(s, kIcon_bear))); return;
    case Keyword::kINVENTIVE_PRIMARY: {
      int n = icons(me)[kIcon_primate];
      gain(s, p, EffectKind::kEffXtoken, n >= 5 ? 3 : n >= 3 ? 2 : n >= 1 ? 1 : 0);
      return;
    }
    case Keyword::kPACK: gain(s, p, EffectKind::kEffAppeal, icons(me)[kIcon_predator]); return;
    case Keyword::kPETTING_ZOO_ANIMAL: gain(s, p, EffectKind::kEffAppeal, 3 * icons(me)[kIcon_pet]); return;
    case Keyword::kICONIC_ANIMAL:
      gain(s, p, EffectKind::kEffAppeal, std::min(8, icons_everywhere(s, kContinentIcon0 + value)));
      return;
    case Keyword::kFULL_THROATED: gain(s, p, EffectKind::kEffWorker, 1); return;
    case Keyword::kPERCEPTION_4: {
      std::vector<CardIdx> drawn;
      for (int i = 0; i < 4; ++i) drawn.push_back(draw_from_deck(s, ch));
      offer_pick(s, p, drawn, 2);
      return;
    }
    case Keyword::kSCAVENGING: {
      ch.shuffle(s.discard);
      std::vector<CardIdx> taken;
      int n = std::min<int>(value, static_cast<int>(s.discard.size()));
      for (int i = 0; i < n; ++i) {
        taken.push_back(s.discard.back());
        s.discard.pop_back();
      }
      offer_pick(s, p, taken, 1);
      return;
    }
    case Keyword::kHUNTER: hunter(s, p, value, ch); return;
    case Keyword::kSNAPPING_1: case Keyword::kSNAPPING_2: {
      int n = k == Keyword::kSNAPPING_1 ? 1 : 2;
      int fi = push(s, FrameKind::ab_snap, p);
      Frame& f = frame(s, fi);
      f.d[fd::left] = static_cast<std::int16_t>(n);
      f.d[fd::snap_can_replenish] = 0;
      f.d[fd::snap_replenish_between] = n > 1;
      return;
    }
    case Keyword::kSPONSOR_MAGNET:
      for (int i = 0; i < kDisplaySize; ++i) {
        if (s.display[i] != kNoCard && is_sponsor(s.display[i])) to_hand(s, p, take_from_display(s, i), true);
      }
      return;
    case Keyword::kSUN_BATHING:
      if (me.hand_size && value > 0) push_left(s, FrameKind::ab_sun, p, value);
      return;
    case Keyword::kDIGGING:
      if (value > 0) push_left(s, FrameKind::ab_dig, p, value);
      return;
    case Keyword::kPOUCH:
      if (me.hand_size && value > 0) push_left(s, FrameKind::ab_pouch, p, value);
      return;
    case Keyword::kRESISTANCE: {
      std::vector<CardIdx> drawn;
      int n = std::min<int>(2, static_cast<int>(s.final_deck.size()));
      for (int i = 0; i < n; ++i) {
        drawn.push_back(s.final_deck.back());
        s.final_deck.pop_back();
      }
      if (drawn.size() == 1) {
        add_final_card(me, drawn[0]);
      } else if (!drawn.empty()) {
        int fi = push(s, FrameKind::ab_resist, p);
        set_frame_list(frame(s, fi), fd::cards_n, fd::cards_list, drawn);
      }
      return;
    }
    case Keyword::kASSERTION:
      if (!s.base_project_deck.empty()) push(s, FrameKind::ab_project, p);
      return;
    case Keyword::kDOMINANCE: {
      CardIdx primates = card_by_id(108);
      if (std::find(s.base_project_deck.begin(), s.base_project_deck.end(), primates) != s.base_project_deck.end()) {
        int fi = push(s, FrameKind::ab_project, p);
        frame(s, fi).d[fd::project_only] = primates;
      }
      return;
    }
    case Keyword::kCLEVER: {
      Frame f;
      f.kind = FrameKind::clever;
      f.player = static_cast<std::int8_t>(p);
      after_finishing(s, f);
      return;
    }
    case Keyword::kBOOST_ANIMAL: case Keyword::kBOOST_ASSOCIATION: case Keyword::kBOOST_BUILDING:
    case Keyword::kBOOST_CARD: case Keyword::kBOOST_SPONSORS: {
      Frame f;
      f.kind = FrameKind::ab_boost;
      f.player = static_cast<std::int8_t>(p);
      f.d[fd::boost_action] = static_cast<std::int16_t>(action_of(k));
      after_finishing(s, f);
      return;
    }
    case Keyword::kMULTIPLIER_ASSOCIATION: case Keyword::kMULTIPLIER_BUILDING: case Keyword::kMULTIPLIER_CARD:
    case Keyword::kMULTIPLIER_SPONSORS:
      add_action_token(s, p, action_of(k), TokenKind::kMultiplier);
      return;
    case Keyword::kACTION_ASSOCIATION: case Keyword::kACTION_BUILDING: case Keyword::kACTION_CARDS:
    case Keyword::kACTION_SPONSORS:
      offer_extra_action(s, p, 1u << action_of(k));
      return;
    case Keyword::kDETERMINATION: offer_extra_action(s, p, (1u << kNumActions) - 1, -1, false); return;
    case Keyword::kPOSTURING:
      for (int i = 0; i < (value > 0 ? value : 1); ++i) {
        push_free_build(s, p, {{BuildingKind::kKiosk, 1}, {BuildingKind::kPavilion, 1}});
      }
      return;
    case Keyword::kPEACOCKING: push_free_build(s, p, {{BuildingKind::kAviary, 5}}); return;
    case Keyword::kVENOM:
      for (int q = 0; q < kPlayers; ++q) {
        const Player& victim = s.players[q];
        if (q == p || victim.appeal <= me.appeal || !affected(s, q)) continue;
        int left = value > 0 ? value : 1;
        std::int8_t slots[kNumActions];
        std::copy(victim.slots, victim.slots + kNumActions, slots);
        for (int action : slots) {
          if (left <= 0) break;
          if (add_action_token(s, q, action, TokenKind::kVenom)) --left;
        }
      }
      return;
    case Keyword::kCONSTRICTION:
      for (int q = 0; q < kPlayers; ++q) {
        const Player& victim = s.players[q];
        if (q == p || !affected(s, q)) continue;
        int left = (victim.appeal > me.appeal) + (victim.conservation > me.conservation);
        std::int8_t slots[kNumActions];
        std::copy(victim.slots, victim.slots + kNumActions, slots);
        for (int i = kNumActions - 1; i >= 0; --i) {
          if (left <= 0) break;
          if (add_action_token(s, q, slots[i], TokenKind::kConstriction)) --left;
        }
      }
      return;
    case Keyword::kPILFERING_1: pilfer(s, p, false, 0); return;
    case Keyword::kPILFERING_2:
      pilfer(s, p, true, 1);
      pilfer(s, p, false, 0);
      return;
    case Keyword::kHYPNOSIS: {
      std::vector<int> t = targets(s, p, false, 0);
      if (t.size() == 1) {
        offer_hypnosis(s, p, t[0], value);
      } else if (!t.empty()) {
        push_candidates(s, FrameKind::ab_hypnosis_pick, p, t);
        frame(s, s.stack_n - 1).d[fd::hypnosis_reach] = static_cast<std::int16_t>(value);
      }
      return;
    }
    case Keyword::kNone: break;
  }
  engine_error(std::string("unknown ability ") + keyword_name(k));
}

}  // namespace

void offer_pick(State& s, int p, const std::vector<CardIdx>& drawn, int keep) {
  std::vector<CardIdx> revealed;
  for (CardIdx c : drawn) {
    if (c != kNoCard) revealed.push_back(c);
  }
  if (static_cast<int>(revealed.size()) <= keep) {
    for (CardIdx c : revealed) to_hand(s, p, c, false);
    return;
  }
  int fi = push(s, FrameKind::ab_pick, p);
  set_frame_list(frame(s, fi), fd::pick_n, fd::pick_cards, revealed);
  frame(s, fi).d[fd::pick_count] = static_cast<std::int16_t>(keep);
}

void hunter(State& s, int p, int value, Chance& ch) {
  std::vector<CardIdx> revealed;
  for (int i = 0; i < value; ++i) {
    CardIdx c = draw_from_deck(s, ch);
    if (c != kNoCard) revealed.push_back(c);
  }
  bool any_animal = std::any_of(revealed.begin(), revealed.end(), is_animal);
  if (any_animal) {
    int fi = push(s, FrameKind::ab_hunter, p);
    set_frame_list(frame(s, fi), fd::cards_n, fd::cards_list, revealed);
  } else {
    discard_all(s, revealed);
  }
}

void resolve_abilities(State& s, int p, CardIdx c, Chance& ch) {
  const CardDef& d = card(c);
  for (int i = d.n_abilities - 1; i >= 0; --i) resolve_keyword(s, p, d.abilities[i].keyword, d.abilities[i].value, ch);
}

void register_ability_frames() {
  register_choice(FrameKind::ab_pick, pick_options, pick_apply);
  register_choice(FrameKind::ab_hunter, hunter_options, hunter_apply);
  register_choice(FrameKind::ab_snap, snap_options, snap_apply);
  register_choice(FrameKind::ab_sun, sun_options, sun_apply);
  register_choice(FrameKind::ab_dig, dig_options, dig_apply);
  register_choice(FrameKind::ab_pouch, pouch_options, pouch_apply);
  register_choice(FrameKind::ab_resist, resist_options, resist_apply);
  register_choice(FrameKind::ab_project, project_options, project_apply);
  register_choice(FrameKind::ab_boost, boost_options, boost_apply);
  register_choice(FrameKind::ab_pilfer_pick, candidate_options, pilfer_pick_apply);
  register_choice(FrameKind::ab_pilfer, pilfer_options, pilfer_apply);
  register_choice(FrameKind::ab_hypnosis_pick, candidate_options, hypnosis_pick_apply);
}

}  // namespace board_ai::arknova
