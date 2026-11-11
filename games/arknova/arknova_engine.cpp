// Game setup, the turn loop, extra actions, breaks and final scoring (ark-nova-ai arknova/engine.py).

#include <algorithm>
#include <stdexcept>

#include "arknova_internal.h"

namespace board_ai::arknova {

void engine_error(const std::string& msg) { throw std::runtime_error("arknova engine: " + msg); }

bool g_upstream_rules = false;

// ---- frame kinds and option heads ----------------------------------------------------

namespace {
constexpr const char* kFrameKindNames[] = {
#define ARKNOVA_NAME(k) #k,
    ARKNOVA_FRAME_KINDS(ARKNOVA_NAME)
#undef ARKNOVA_NAME
};
constexpr const char* kHeadNames[] = {
#define ARKNOVA_NAME(k) #k,
    ARKNOVA_OPTION_HEADS(ARKNOVA_NAME)
#undef ARKNOVA_NAME
};

Registry& registry_storage() {
  static Registry r;
  return r;
}
}  // namespace

const char* frame_kind_name(FrameKind k) {
  if (k == FrameKind::break_) return "break";
  return kFrameKindNames[static_cast<int>(k)];
}

bool is_auto(FrameKind k) { return k >= FrameKind::break_; }

const char* head_name(Head h) {
  if (h == Head::break_) return "break";
  if (h == Head::return_) return "return";
  return kHeadNames[static_cast<int>(h)];
}

// ---- registry ---------------------------------------------------------------------

Registry& registry() {
  static const bool registered = [] {
    if (num_cards() > kCardSlots) engine_error("kCardSlots is smaller than the card count");
    // The longest Hunter reveal must fit a frame's card list: a printed Hunter X, or one card per
    // predator icon in a zoo (sponsor 252), which no zoo can exceed the game's total of.
    int longest_reveal = 0;
    for (int c = 0; c < num_cards(); ++c) {
      const CardDef& d = card(static_cast<CardIdx>(c));
      longest_reveal += card_icons(static_cast<CardIdx>(c))[kIcon_predator];
      for (int i = 0; i < d.n_abilities; ++i) {
        if (d.abilities[i].keyword == Keyword::kHUNTER && d.abilities[i].value > kFrameData - fd::cards_list) {
          engine_error("kFrameData is too small for a printed Hunter reveal");
        }
      }
    }
    if (longest_reveal > kFrameData - fd::cards_list) engine_error("kFrameData is too small for the longest Hunter reveal");
    register_engine_frames();
    register_effect_frames();
    register_cards_action();
    register_build_action();
    register_animals_action();
    register_association_action();
    register_sponsors_action();
    register_relocate_frames();
    register_ability_frames();
    register_sponsor_effect_frames();
    Registry& r = registry_storage();
    for (int k = 0; k < static_cast<int>(FrameKind::kNumFrameKinds); ++k) {
      FrameKind kind = static_cast<FrameKind>(k);
      bool ok = is_auto(kind) ? (r.autos[k] && !r.options[k]) : (r.options[k] && r.apply[k] && !r.autos[k]);
      if (!ok) engine_error(std::string("frame kind registered wrongly: ") + frame_kind_name(kind));
    }
    for (int a = 0; a < kNumActions; ++a) {
      if (!r.feasible[a] || !r.start[a]) engine_error(std::string("action not registered: ") + kActionNames[a]);
    }
    return true;
  }();
  (void)registered;
  return registry_storage();
}

void register_choice(FrameKind k, OptionsFn options, ApplyFn apply) {
  registry_storage().options[static_cast<int>(k)] = options;
  registry_storage().apply[static_cast<int>(k)] = apply;
}

void register_auto(FrameKind k, AutoFn fn) { registry_storage().autos[static_cast<int>(k)] = fn; }

void register_action(ActionCard a, FeasibleFn feasible, StartFn start) {
  registry_storage().feasible[a] = feasible;
  registry_storage().start[a] = start;
}

// ---- frame stack ------------------------------------------------------------------

int push(State& s, FrameKind kind, int player) {
  if (s.stack_n >= kMaxStack) engine_error("frame stack overflow");
  Frame& f = s.stack[s.stack_n];
  f = Frame{};
  f.kind = kind;
  f.player = static_cast<std::int8_t>(player);
  f.uid = s.next_uid++;
  if (s.next_uid == 0) s.next_uid = 1;
  return s.stack_n++;
}

void pop(State& s, int fi) {
  if (fi < 0 || fi >= s.stack_n) engine_error("pop: frame is not on the stack");
  for (int i = fi + 1; i < s.stack_n; ++i) s.stack[i - 1] = s.stack[i];
  --s.stack_n;
}

int find_frame(const State& s, std::uint16_t uid) {
  for (int i = s.stack_n - 1; i >= 0; --i) {
    if (s.stack[i].uid == uid) return i;
  }
  return -1;
}

// ---- players and state ------------------------------------------------------------

int Player::strength(int action) const {
  for (int i = 0; i < kNumActions; ++i) {
    if (slots[i] == action) return i + 1;
  }
  engine_error("strength: action not in slots");
}

int Player::active_workers() const {
  int out = 0;
  for (int t = 0; t < kNumTasks; ++t) out += workers_out[t];
  return workers - out;
}

int Player::hand_limit() const { return has_university(kUniHand) ? kHandLimitUniversity : kHandLimit; }

int Player::projects_supported() const {
  int n = 0;
  for (bool t : tokens) n += !t;
  return n;
}

bool Player::has_university(University u) const {
  return std::find(universities, universities + n_universities, u) != universities + n_universities;
}

bool Player::has_partner_zoo(int continent) const {
  return std::find(partner_zoos, partner_zoos + n_partner_zoos, continent) != partner_zoos + n_partner_zoos;
}

bool Player::in_zoo(CardIdx c) const {
  return std::find(zoo_cards, zoo_cards + n_zoo_cards, c) != zoo_cards + n_zoo_cards;
}

std::vector<CardIdx> Player::hand_cards() const {
  std::vector<CardIdx> out;
  out.reserve(hand_size);
  for (int c = 0; c < kCardSlots; ++c) {
    if (hand[c]) out.push_back(static_cast<CardIdx>(c));
  }
  return out;
}

TurnCtx& State::turn() {
  if (turn_depth == 0) engine_error("no turn in progress");
  return turns[turn_depth - 1];
}

const TurnCtx& State::turn() const {
  if (turn_depth == 0) engine_error("no turn in progress");
  return turns[turn_depth - 1];
}

ProjectTokens* State::tokens_of(CardIdx project) {
  for (int i = 0; i < n_project_tokens; ++i) {
    if (project_tokens[i].project == project) return &project_tokens[i];
  }
  return nullptr;
}

const ProjectTokens* State::tokens_of(CardIdx project) const {
  return const_cast<State*>(this)->tokens_of(project);
}

// ---- setup --------------------------------------------------------------------------

namespace {

constexpr int kSetupDraw = 8;
constexpr int kSetupKeep = 4;

template <class T>
void rng_shuffle(std::vector<T>& v, std::mt19937_64& rng) {
  for (int i = static_cast<int>(v.size()) - 1; i > 0; --i) {
    int j = static_cast<int>(std::uniform_int_distribution<int>(0, i)(rng));
    std::swap(v[i], v[j]);
  }
}

}  // namespace

void new_game(State& s, std::mt19937_64& rng) {
  registry();
  s = State{};
  for (int i = 0; i < kPlayers; ++i) {
    std::vector<int> others = {kBuild, kCards, kAssociation, kSponsors};
    rng_shuffle(others, rng);
    Player& pl = s.players[i];
    pl.slots[0] = kAnimals;
    for (int k = 0; k < 4; ++k) pl.slots[k + 1] = static_cast<std::int8_t>(others[k]);
    pl.appeal = static_cast<std::int16_t>(i);
  }
  std::vector<CardIdx> endgame, base;
  for (int c = 0; c < num_cards(); ++c) {
    const CardDef& d = card(static_cast<CardIdx>(c));
    if (d.type == CardType::kEndgame) {
      endgame.push_back(static_cast<CardIdx>(c));
    } else if (d.type == CardType::kProject && d.proj_category == ProjCategory::kProjBase) {
      base.push_back(static_cast<CardIdx>(c));
    } else {
      s.deck.push_back(static_cast<CardIdx>(c));
    }
  }
  rng_shuffle(s.deck, rng);
  for (int i = 0; i < kDisplaySize; ++i) {
    s.display[i] = s.deck.back();
    s.deck.pop_back();
  }
  s.final_deck = endgame;
  rng_shuffle(s.final_deck, rng);
  rng_shuffle(base, rng);
  s.n_base_projects = kBaseProjects;
  for (int i = 0; i < kBaseProjects; ++i) {
    s.base_projects[i] = base[i];
    ProjectTokens& t = s.project_tokens[s.n_project_tokens++];
    t = ProjectTokens{};
    t.project = base[i];
    t.slot[i] = kBlocked;  // 2 players: base project i has its slot i+1 blocked
  }
  s.base_project_deck.assign(base.begin() + kBaseProjects, base.end());
  s.partner_zoos_available = (1u << kNumContinents) - 1;
  s.universities_available = (1u << kNumUniversities) - 1;
  std::vector<int> tiles;
  for (int t = 0; t < kNumBonusTiles; ++t) tiles.push_back(t);
  rng_shuffle(tiles, rng);
  for (int sp = 0; sp < 2; ++sp) {
    s.n_bonus_tiles[sp] = 2;
    s.bonus_tiles[sp][0] = static_cast<std::int8_t>(tiles[2 * sp]);
    s.bonus_tiles[sp][1] = static_cast<std::int8_t>(tiles[2 * sp + 1]);
  }
  push(s, FrameKind::turn, 0);
  for (int p = kPlayers - 1; p >= 0; --p) {
    Player& pl = s.players[p];
    for (int i = 0; i < 2; ++i) {
      pl.final_cards[pl.n_final++] = s.final_deck.back();
      s.final_deck.pop_back();
    }
    for (int i = 0; i < kSetupDraw; ++i) {
      to_hand(s, p, s.deck.back(), false);
      s.deck.pop_back();
    }
    push(s, FrameKind::setup_keep, p);
  }
}

namespace {

void setup_options(State& s, int fi, std::vector<Option>& out) {
  std::vector<CardIdx> hand = s.players[frame(s, fi).player].hand_cards();
  int n = static_cast<int>(hand.size());
  if (n < kSetupKeep) return;
  int idx[kSetupKeep] = {0, 1, 2, 3};
  while (true) {
    Option o = opt(Head::keep);
    o.n = kSetupKeep;
    for (int k = 0; k < kSetupKeep; ++k) o.a[k] = hand[idx[k]];
    out.push_back(o);
    int k = kSetupKeep - 1;
    while (k >= 0 && idx[k] == n - kSetupKeep + k) --k;
    if (k < 0) break;
    ++idx[k];
    for (int j = k + 1; j < kSetupKeep; ++j) idx[j] = idx[j - 1] + 1;
  }
}

void setup_apply(State& s, int fi, const Option& o, Chance&) {
  int p = frame(s, fi).player;
  pop(s, fi);
  for (CardIdx c : s.players[p].hand_cards()) {
    if (std::find(o.a, o.a + kSetupKeep, c) != o.a + kSetupKeep) continue;
    remove_from_hand(s, p, c);
    s.discard.push_back(c);
  }
}

// ---- turn ---------------------------------------------------------------------------

int card_strength(const State& s, int owner, int action) {
  int penalty = has_token(s, owner, action, TokenKind::kConstriction) ? kConstrictionPenalty : 0;
  return s.players[owner].strength(action) - penalty;
}

void turn_options_impl(State& s, int fi, std::vector<Option>& out, bool charge_venom) {
  int p = frame(s, fi).player;
  Player& pl = s.players[p];
  bool poisoned = has_venom(s, p);
  const Registry& r = registry();
  std::size_t n0 = out.size();
  std::int8_t slots[kNumActions];
  std::copy(pl.slots, pl.slots + kNumActions, slots);
  for (int action : slots) {
    int cost = poisoned && !has_token(s, p, action, TokenKind::kVenom) ? kVenomCost : 0;
    if (charge_venom && pl.money < cost) continue;
    int base = card_strength(s, p, action);
    int paid = std::min<int>(pl.money, cost);
    pl.money -= paid;
    for (int x = 0; x <= pl.xtokens; ++x) {
      if (base + x >= 1 && r.feasible[action](s, p, base + x)) out.push_back(opt(Head::act, action, x));
    }
    pl.money += paid;
    if (pl.xtokens < kMaxXtokens) out.push_back(opt(Head::xtoken, action));
  }
  if (out.size() == n0 && charge_venom) turn_options_impl(s, fi, out, false);
}

void turn_options(State& s, int fi, std::vector<Option>& out) { turn_options_impl(s, fi, out, true); }

int use_card(State& s, int owner, int action) {
  int strength = card_strength(s, owner, action);
  remove_action_token(s, owner, action, TokenKind::kVenom);
  remove_action_token(s, owner, action, TokenKind::kConstriction);
  return strength;
}

void move_to_slot_one(Player& pl, int action) {
  auto* it = std::find(pl.slots, pl.slots + kNumActions, action);
  if (it == pl.slots + kNumActions) engine_error("action not in slots");
  std::rotate(pl.slots, it, it + 1);
}

void turn_apply(State& s, int fi, const Option& o, Chance&) {
  int p = frame(s, fi).player;
  pop(s, fi);
  Player& pl = s.players[p];
  int action = o.a[0];
  bool xtoken = o.head == Head::xtoken;
  pl.last_action = xtoken ? kLastActionXtoken : static_cast<std::int8_t>(action);
  if (has_venom(s, p) && !has_token(s, p, action, TokenKind::kVenom)) pl.money -= std::min<int>(pl.money, kVenomCost);
  int strength = use_card(s, p, action);
  s.turn_depth = 1;
  s.turns[0] = TurnCtx{};
  s.turns[0].action = static_cast<std::int8_t>(action);
  s.turns[0].xtoken = xtoken;
  int te = push(s, FrameKind::turn_end, p);
  frame(s, te).d[fd::phase] = 0;
  if (xtoken) {
    gain(s, p, EffectKind::kEffXtoken, 1);
    return;
  }
  strength += o.a[1];
  pl.xtokens -= o.a[1];
  s.turns[0].strength = static_cast<std::int8_t>(strength);
  s.turns[0].has_strength = true;
  registry().start[action](s, p, strength);
}

void repeat_options(State& s, int fi, std::vector<Option>& out) {
  int p = frame(s, fi).player;
  const TurnCtx& t = s.turn();
  bool can = t.xtoken ? s.players[p].xtokens < kMaxXtokens : registry().feasible[t.action](s, p, t.strength);
  if (can) out.push_back(opt(Head::repeat));
  out.push_back(opt_skip());
}

void repeat_apply(State& s, int fi, const Option& o, Chance&) {
  int p = frame(s, fi).player;
  pop(s, fi);
  if (o.head != Head::repeat) return;
  const TurnCtx& t = s.turn();
  if (t.xtoken) {
    gain(s, p, EffectKind::kEffXtoken, 1);
  } else {
    registry().start[t.action](s, p, t.strength);
  }
}

void push_after_frames(State& s) {
  TurnCtx t = s.turn();
  for (int i = t.n_after - 1; i >= 0; --i) {
    int fi = push(s, t.after[i].kind, t.after[i].player);
    std::copy(std::begin(t.after[i].d), std::end(t.after[i].d), frame(s, fi).d);
  }
}

void check_end(State& s, bool during_break) {
  if (s.end_by >= 0) return;
  for (int q = 0; q < kPlayers; ++q) {
    if (!during_break && q != s.current) continue;
    if (s.players[q].victory_points() >= 0) {
      s.end_by = static_cast<std::int8_t>(q);
      s.final_turns = static_cast<std::int8_t>(during_break ? kPlayers : kPlayers - 1);
      return;
    }
  }
}

void turn_end(State& s, int fi, Chance& ch) {
  int p = frame(s, fi).player;
  Player& pl = s.players[p];
  if (frame(s, fi).d[fd::phase] == 0) {
    int action = s.turn().action;
    if (has_token(s, p, action, TokenKind::kMultiplier)) {
      remove_action_token(s, p, action, TokenKind::kMultiplier);
      push(s, FrameKind::multiplier_repeat, p);
      return;
    }
    frame(s, fi).d[fd::phase] = 1;
    move_to_slot_one(pl, action);
    push_after_frames(s);
    return;
  }
  pop(s, fi);
  refill_display(s, ch);
  s.turn_depth = 0;
  check_end(s, false);
  push(s, FrameKind::next_turn, p);
  if (s.break_by >= 0) {
    int b = push(s, FrameKind::break_, s.break_by);
    frame(s, b).d[fd::phase] = 0;
  }
}

// ---- extra actions -----------------------------------------------------------------

void extra_options(State& s, int fi, std::vector<Option>& out) {
  const Frame f = frame(s, fi);
  int p = f.player, owner = f.d[fd::extra_owner];
  const Registry& r = registry();
  std::size_t n0 = out.size();
  for (int action = 0; action < kNumActions; ++action) {
    if (!(f.d[fd::extra_actions] >> action & 1)) continue;
    int base = card_strength(s, owner, action);
    if (owner != p) {
      s.borrowed_action = static_cast<std::int8_t>(action);
      s.borrowed_upgraded = s.players[owner].is_upgraded(action);
    } else {
      s.borrowed_action = -1;
    }
    for (int x = 0; x <= s.players[p].xtokens; ++x) {
      if (base + x >= 1 && r.feasible[action](s, p, base + x)) out.push_back(opt(Head::act, action, x));
    }
    s.borrowed_action = -1;
    s.borrowed_upgraded = false;
  }
  if (f.d[fd::extra_optional] || out.size() == n0) out.push_back(opt_skip());
}

void extra_apply(State& s, int fi, const Option& o, Chance&) {
  const Frame f = frame(s, fi);
  pop(s, fi);
  if (o.head != Head::act) return;
  int p = f.player, owner = f.d[fd::extra_owner];
  int action = o.a[0], x = o.a[1];
  Player& pl = s.players[p];
  pl.last_action = static_cast<std::int8_t>(action);
  if (owner == p && has_venom(s, p) && !has_token(s, p, action, TokenKind::kVenom)) {
    pl.money -= std::min<int>(pl.money, kVenomCost);
  }
  int strength = use_card(s, owner, action) + x;
  pl.xtokens -= x;
  if (s.turn_depth >= kMaxTurnDepth) engine_error("turn context overflow");
  TurnCtx& t = s.turns[s.turn_depth++];
  t = TurnCtx{};
  t.action = static_cast<std::int8_t>(action);
  t.strength = static_cast<std::int8_t>(strength);
  t.has_strength = true;
  if (owner != p) {
    s.borrowed_action = static_cast<std::int8_t>(action);
    s.borrowed_upgraded = s.players[owner].is_upgraded(action);
  }
  int ee = push(s, FrameKind::extra_end, p);
  frame(s, ee).d[fd::phase] = 0;
  frame(s, ee).d[fd::extra_end_owner] = static_cast<std::int16_t>(owner);
  registry().start[action](s, p, strength);
}

void extra_end(State& s, int fi, Chance&) {
  Frame& f = frame(s, fi);
  if (f.d[fd::phase] == 0) {
    f.d[fd::phase] = 1;
    move_to_slot_one(s.players[f.d[fd::extra_end_owner]], s.turn().action);
    s.borrowed_action = -1;
    s.borrowed_upgraded = false;
    push_after_frames(s);
    return;
  }
  pop(s, fi);
  --s.turn_depth;
}

void next_turn(State& s, int fi, Chance&) {
  pop(s, fi);
  if (s.final_turns >= 0) {
    if (s.final_turns == 0) {
      int f = push(s, FrameKind::final_scoring, s.current);
      frame(s, f).d[fd::phase] = 0;
      return;
    }
    --s.final_turns;
  }
  s.current = static_cast<std::int8_t>((s.current + 1) % kPlayers);
  push(s, FrameKind::turn, s.current);
}

// ---- break --------------------------------------------------------------------------

void break_auto(State& s, int fi, Chance& ch) {
  Frame& f = frame(s, fi);
  int phase = f.d[fd::phase];
  if (phase == 0) {
    f.d[fd::phase] = 1;
    for (int q = kPlayers - 1; q >= 0; --q) {
      const Player& pl = s.players[q];
      if (pl.hand_size > pl.hand_limit()) {
        int d = push(s, FrameKind::discard_hand, q);
        frame(s, d).d[fd::discard_limit] = static_cast<std::int16_t>(pl.hand_limit());
      }
    }
  } else if (phase == 1) {
    f.d[fd::phase] = 2;
    for (Player& pl : s.players) {
      std::fill(std::begin(pl.workers_out), std::end(pl.workers_out), 0);
      std::fill(std::begin(pl.n_action_tokens), std::end(pl.n_action_tokens), 0);
    }
    s.partner_zoos_available = (1u << kNumContinents) - 1;
    s.universities_available = (1u << kNumUniversities) - 1;
    for (int index = 0; index < 2; ++index) {
      if (s.display[index] != kNoCard) {
        s.discard.push_back(s.display[index]);
        s.display[index] = kNoCard;
      }
    }
    refill_display(s, ch);
    f.d[fd::break_order_n] = kPlayers;
    for (int i = 0; i < kPlayers; ++i) f.d[fd::break_order + i] = static_cast<std::int16_t>((f.player + i) % kPlayers);
  } else if (phase == 2) {
    int n = f.d[fd::break_order_n];
    if (n == 0) {
      f.d[fd::phase] = 3;
      return;
    }
    int p = f.d[fd::break_order];
    for (int i = 1; i < n; ++i) f.d[fd::break_order + i - 1] = f.d[fd::break_order + i];
    f.d[fd::break_order + n - 1] = kNull;
    f.d[fd::break_order_n] = static_cast<std::int16_t>(n - 1);
    Player& pl = s.players[p];
    gain(s, p, EffectKind::kEffMoney, appeal_income(pl.appeal) + pl.zoo.kiosk_income());
    token_income(s, p);
    std::vector<CardIdx> zoo(pl.zoo_cards, pl.zoo_cards + pl.n_zoo_cards);
    for (auto it = zoo.rbegin(); it != zoo.rend(); ++it) {
      if (has_income(*it)) income_effect(s, *it, p);
    }
  } else {
    pop(s, fi);
    s.break_pos = 0;
    s.break_by = -1;
    check_end(s, true);
  }
}

// ---- end of game --------------------------------------------------------------------

void final_scoring(State& s, int fi, Chance&) {
  if (frame(s, fi).d[fd::phase] == 0) {
    frame(s, fi).d[fd::phase] = 1;
    if (!s.final_card_discarded) push_final_card_discards(s);
    return;
  }
  pop(s, fi);
  for (int p = 0; p < kPlayers; ++p) {
    const Player& pl = s.players[p];
    std::vector<CardIdx> cards(pl.zoo_cards, pl.zoo_cards + pl.n_zoo_cards);
    cards.insert(cards.end(), pl.final_cards, pl.final_cards + pl.n_final);
    for (CardIdx c : cards) {
      if (has_endgame(c)) endgame_effect(s, c, p);
    }
  }
  for (int p = 0; p < kPlayers; ++p) s.scores[p] = static_cast<std::int16_t>(s.players[p].victory_points());
  s.stack_n = 0;
  s.over = true;
}

}  // namespace

void offer_extra_action(State& s, int p, unsigned action_mask, int owner, bool optional) {
  Frame f;
  f.kind = FrameKind::extra_action;
  f.player = static_cast<std::int8_t>(p);
  f.d[fd::extra_actions] = static_cast<std::int16_t>(action_mask);
  f.d[fd::extra_optional] = optional ? 1 : 0;
  f.d[fd::extra_owner] = static_cast<std::int16_t>(owner < 0 ? p : owner);
  after_finishing(s, f);
}

void after_finishing(State& s, const Frame& f) {
  TurnCtx& t = s.turn();
  if (t.n_after >= kMaxAfter) engine_error("too many deferred frames");
  t.after[t.n_after++] = f;
}

int effective_strength(const State& s, int owner, int action) { return card_strength(s, owner, action); }

std::vector<int> winners(const State& s) {
  int best_score = -100000, best_projects = -1;
  for (int p = 0; p < kPlayers; ++p) {
    int sc = s.scores[p], pr = s.players[p].projects_supported();
    if (sc > best_score || (sc == best_score && pr > best_projects)) {
      best_score = sc;
      best_projects = pr;
    }
  }
  std::vector<int> out;
  for (int p = 0; p < kPlayers; ++p) {
    if (s.scores[p] == best_score && s.players[p].projects_supported() == best_projects) out.push_back(p);
  }
  return out;
}

// ---- step -----------------------------------------------------------------------------

std::vector<Option> legal_options(State& s) {
  if (s.over) return {};
  const Registry& r = registry();
  if (s.stack_n == 0) engine_error("empty stack");
  std::vector<Option> out;
  int fi = s.stack_n - 1;
  OptionsFn fn = r.options[static_cast<int>(s.stack[fi].kind)];
  if (!fn) engine_error(std::string("not a choice frame: ") + frame_kind_name(s.stack[fi].kind));
  fn(s, fi, out);
  return out;
}

namespace {

void settle(State& s, Chance& ch) {
  const Registry& r = registry();
  while (!s.over) {
    if (s.stack_n == 0) engine_error("empty stack while the game is running");
    int fi = s.stack_n - 1;
    int k = static_cast<int>(s.stack[fi].kind);
    if (r.autos[k]) {
      r.autos[k](s, fi, ch);
      continue;
    }
    if (legal_options(s).empty()) engine_error(std::string("choice frame has no options: ") + frame_kind_name(s.stack[fi].kind));
    return;
  }
}

}  // namespace

void step(State& s, const Option& option, Chance& ch) {
  if (s.over) engine_error("the game is over");
  std::vector<Option> legal = legal_options(s);
  if (std::find(legal.begin(), legal.end(), option) == legal.end()) {
    engine_error("illegal option " + option_to_json(s, option) + " for " + frame_kind_name(s.top().kind));
  }
  int fi = s.stack_n - 1;
  registry().apply[static_cast<int>(s.stack[fi].kind)](s, fi, option, ch);
  settle(s, ch);
}

void register_engine_frames() {
  register_choice(FrameKind::setup_keep, setup_options, setup_apply);
  register_choice(FrameKind::turn, turn_options, turn_apply);
  register_choice(FrameKind::multiplier_repeat, repeat_options, repeat_apply);
  register_choice(FrameKind::extra_action, extra_options, extra_apply);
  register_auto(FrameKind::extra_end, extra_end);
  register_auto(FrameKind::turn_end, turn_end);
  register_auto(FrameKind::next_turn, next_turn);
  register_auto(FrameKind::break_, break_auto);
  register_auto(FrameKind::final_scoring, final_scoring);
}

}  // namespace board_ai::arknova
