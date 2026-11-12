#include "arknova_state.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "../../engine/core/snapshot_io.h"
#include "../../engine/core/viz_runtime.h"
#include "arknova_internal.h"
#include "arknova_json.h"
#include "arknova_rules.h"

namespace board_ai::arknova {

namespace {

constexpr int kPackVersion = 0xA45;

void fill_slots(std::int16_t* first, int n, std::int16_t v) { std::fill(first, first + n, v); }

bool valid_card(int v) { return v >= 0 && v < num_cards(); }

class Reader {
 public:
  explicit Reader(const std::vector<int>& v) : v_(v) {}
  int next() {
    if (i_ >= v_.size()) throw std::runtime_error("arknova: packed public state is truncated");
    return v_[i_++];
  }
  template <class T>
  T as() {
    return static_cast<T>(next());
  }
  int count(int max, const char* what) {
    int n = next();
    if (n < 0 || n > max) throw std::runtime_error(std::string("arknova: packed public state: bad count for ") + what);
    return n;
  }
  void finish() const {
    if (i_ != v_.size()) throw std::runtime_error("arknova: packed public state has trailing data");
  }

 private:
  const std::vector<int>& v_;
  std::size_t i_ = 0;
};

void pack_frame(std::vector<int>& out, const Frame& f, bool scrub_private) {
  out.push_back(static_cast<int>(f.kind));
  out.push_back(f.player);
  int n_off = -1, list_off = -1;
  if (scrub_private) private_list_offsets(f.kind, &n_off, &list_off);
  for (int k = 0; k < kFrameData; ++k) {
    bool hidden = list_off >= 0 && k >= list_off && k < list_off + kPrivateCards && f.d[k] != kNull;
    out.push_back(hidden ? kNull : f.d[k]);
  }
}

Frame unpack_frame(Reader& r) {
  Frame f;
  int kind = r.next();
  if (kind < 0 || kind >= static_cast<int>(FrameKind::kNumFrameKinds)) throw std::runtime_error("arknova: bad frame kind");
  f.kind = static_cast<FrameKind>(kind);
  f.player = r.as<std::int8_t>();
  for (int k = 0; k < kFrameData; ++k) f.d[k] = r.as<std::int16_t>();
  return f;
}

}  // namespace

// ---- private frames --------------------------------------------------------------------

bool is_private_frame(FrameKind k) { return k == FrameKind::ab_pick || k == FrameKind::ab_resist; }

void private_list_offsets(FrameKind k, int* n_off, int* list_off) {
  *n_off = *list_off = -1;
  if (k == FrameKind::ab_pick) {
    *n_off = fd::pick_n;
    *list_off = fd::pick_cards;
  } else if (k == FrameKind::ab_resist) {
    *n_off = fd::cards_n;
    *list_off = fd::cards_list;
  }
}

std::vector<int> private_frames_of(const State& s, int p) {
  std::vector<int> out;
  for (int i = 0; i < s.stack_n; ++i) {
    if (is_private_frame(s.stack[i].kind) && s.stack[i].player == p) out.push_back(i);
  }
  if (static_cast<int>(out.size()) > kPrivateFrames) throw std::runtime_error("arknova: too many private frames");
  return out;
}

// ---- schema --------------------------------------------------------------------------------

const viz::VisibilitySchema& ArknovaState::schema() {
  static const viz::VisibilitySchema s = [] {
    viz::VisibilitySchema schema;
    schema.n_players = kPlayers;
    viz::declare_field(schema, "public", viz::all_public({}, kPlayers));
    viz::declare_field(schema, "hand", viz::owner_only_first_axis({kPlayers, kCardSlots}, kPlayers));
    viz::declare_field(schema, "final_cards", viz::owner_only_first_axis({kPlayers, kMaxFinalCards}, kPlayers));
    viz::declare_field(schema, "tucked", viz::owner_only_first_axis({kPlayers, kMaxTucked}, kPlayers));
    viz::declare_field(schema, "private_cards",
                       viz::owner_only_first_axis({kPlayers, kPrivateFrames, kPrivateCards}, kPlayers));
    viz::declare_field(schema, "final_known", viz::owner_only_first_axis({kPlayers, kFinalKnown}, kPlayers));
    return schema;
  }();
  return s;
}

ArknovaState::ArknovaState() {
  fill_slots(&hand[0][0], kPlayers * kCardSlots, 0);
  fill_slots(&final_cards[0][0], kPlayers * kMaxFinalCards, kEmptySlot);
  fill_slots(&tucked[0][0], kPlayers * kMaxTucked, kEmptySlot);
  fill_slots(&private_cards[0][0][0], kPlayers * kPrivateFrames * kPrivateCards, kEmptySlot);
  fill_slots(&final_known[0][0], kPlayers * kFinalKnown, kEmptySlot);
}

void ArknovaState::reset_with_seed(std::uint64_t seed) {
  IGameState::reset_step_count_base();
  *this = ArknovaState();
  std::mt19937_64 rng(seed);
  // Both zoos use one map, drawn uniformly per game.
  const int map = static_cast<int>(rng() % static_cast<std::uint64_t>(num_maps()));
  new_game(core, rng, map);
  viz::init_viz(*this, schema());
  absorb_step(*this);
  advance_forced(*this, rng);
}

void ArknovaState::reset_from_json(const std::string& json) {
  IGameState::reset_step_count_base();
  *this = ArknovaState();
  state_from_json(core, json);
  viz::init_viz(*this, schema());
  absorb_step(*this);
}

int ArknovaState::current_player() const { return core.stack_n ? core.top().player : core.current; }

int ArknovaState::winner() const {
  std::vector<int> w = winners();
  return w.size() == 1 ? w[0] : -1;
}

std::vector<int> ArknovaState::winners() const {
  if (!core.over) return {};
  return arknova::winners(core);
}

std::vector<int> ArknovaState::losers() const {
  if (!core.over) return {};
  std::vector<int> w = winners();
  std::vector<int> out;
  for (int p = 0; p < kPlayers; ++p) {
    if (std::find(w.begin(), w.end(), p) == w.end()) out.push_back(p);
  }
  return out;
}

bool ArknovaState::is_turn_start() const {
  return !core.over && core.stack_n && core.top().kind == FrameKind::turn && prefix.empty();
}

// ---- core <-> hidden fields -----------------------------------------------------------------

State& ArknovaState::synced_core() const {
  if (!core_stale_) return core;
  for (int p = 0; p < kPlayers; ++p) {
    Player& pl = core.players[p];
    pl.hand.fill(0);
    for (int c = 0; c < num_cards(); ++c) pl.hand[c] = hand[p][c] == 1;
    for (int i = 0; i < pl.n_final; ++i) pl.final_cards[i] = valid_card(final_cards[p][i]) ? final_cards[p][i] : kNoCard;
    for (int i = 0; i < pl.n_tucked; ++i) pl.tucked[i] = valid_card(tucked[p][i]) ? tucked[p][i] : kNoCard;
    std::vector<int> frames = private_frames_of(core, p);
    for (std::size_t k = 0; k < frames.size(); ++k) {
      Frame& f = core.stack[frames[k]];
      int n_off, list_off;
      private_list_offsets(f.kind, &n_off, &list_off);
      for (int j = 0; j < f.d[n_off]; ++j) {
        f.d[list_off + j] = valid_card(private_cards[p][k][j]) ? private_cards[p][k][j] : kNull;
      }
    }
  }
  core_stale_ = false;
  return core;
}

void ArknovaState::absorb_core() {
  const State& s = core;
  for (int p = 0; p < kPlayers; ++p) {
    const Player& pl = s.players[p];
    // Endgame cards p returned to the final deck: keep those still there, add new returns (from
    // p's final cards or p's private Resistance draw before this step).
    std::vector<CardIdx> known;
    auto in_final_deck = [&](int c) { return std::find(s.final_deck.begin(), s.final_deck.end(), c) != s.final_deck.end(); };
    for (int i = 0; i < kFinalKnown; ++i) {
      if (valid_card(final_known[p][i]) && in_final_deck(final_known[p][i])) known.push_back(final_known[p][i]);
    }
    auto note_return = [&](int c) {
      if (valid_card(c) && in_final_deck(c) && std::find(known.begin(), known.end(), c) == known.end()) known.push_back(c);
    };
    for (int i = 0; i < kMaxFinalCards; ++i) note_return(final_cards[p][i]);
    for (int k = 0; k < kPrivateFrames; ++k) {
      for (int j = 0; j < kPrivateCards; ++j) {
        int c = private_cards[p][k][j];
        if (valid_card(c) && card(static_cast<CardIdx>(c)).type == CardType::kEndgame) note_return(c);
      }
    }
    if (static_cast<int>(known.size()) > kFinalKnown) throw std::runtime_error("arknova: too many known final deck cards");
    fill_slots(final_known[p], kFinalKnown, kEmptySlot);
    std::copy(known.begin(), known.end(), final_known[p]);

    for (int c = 0; c < kCardSlots; ++c) hand[p][c] = pl.hand[c];

    fill_slots(final_cards[p], kMaxFinalCards, kEmptySlot);
    std::copy(pl.final_cards, pl.final_cards + pl.n_final, final_cards[p]);
    fill_slots(tucked[p], kMaxTucked, kEmptySlot);
    std::copy(pl.tucked, pl.tucked + pl.n_tucked, tucked[p]);
    fill_slots(&private_cards[p][0][0], kPrivateFrames * kPrivateCards, kEmptySlot);
    std::vector<int> frames = private_frames_of(s, p);
    for (std::size_t k = 0; k < frames.size(); ++k) {
      const Frame& f = s.stack[frames[k]];
      int n_off, list_off;
      private_list_offsets(f.kind, &n_off, &list_off);
      if (f.d[n_off] > kPrivateCards) throw std::runtime_error("arknova: too many private frame cards");
      for (int j = 0; j < f.d[n_off]; ++j) private_cards[p][k][j] = f.d[list_off + j];
    }
  }
  core_stale_ = false;
}

// ---- packed public state ----------------------------------------------------------------------

void ArknovaState::pack_public(std::vector<int>& out) const {
  const State& s = core;
  out.clear();
  out.push_back(kPackVersion);
  out.insert(out.end(), {s.map, s.harbor_used});
  for (int p = 0; p < kPlayers; ++p) {
    const Player& pl = s.players[p];
    out.insert(out.end(), {pl.appeal, pl.money, pl.conservation, pl.reputation, pl.xtokens, pl.upgraded, pl.workers});
    out.insert(out.end(), pl.slots, pl.slots + kNumActions);
    out.insert(out.end(), pl.workers_out, pl.workers_out + kNumTasks);
    out.push_back(pl.n_partner_zoos);
    out.insert(out.end(), pl.partner_zoos, pl.partner_zoos + pl.n_partner_zoos);
    out.push_back(pl.n_universities);
    out.insert(out.end(), pl.universities, pl.universities + pl.n_universities);
    for (bool t : pl.tokens) out.push_back(t);
    out.push_back(pl.hand_size);
    std::size_t at = out.size();
    out.push_back(0);
    for (int c = 0; c < num_cards(); ++c) {
      if (pl.shown_in_hand(static_cast<CardIdx>(c))) out.push_back(c);
    }
    out[at] = static_cast<int>(out.size() - at - 1);
    at = out.size();
    out.push_back(0);
    for (int c = 0; c < num_cards(); ++c) {
      if (pl.maybe_tucked[c]) out.push_back(c);
    }
    out[at] = static_cast<int>(out.size() - at - 1);
    out.push_back(pl.n_zoo_cards);
    out.insert(out.end(), pl.zoo_cards, pl.zoo_cards + pl.n_zoo_cards);
    out.push_back(pl.n_final);
    out.push_back(pl.zoo.n_buildings);
    for (int b = 0; b < pl.zoo.n_buildings; ++b) {
      const Building& bb = pl.zoo.buildings[b];
      out.insert(out.end(), {static_cast<int>(bb.kind), bb.size, static_cast<int>(bb.cells & 0xFFFFFFFFu),
                             static_cast<int>(bb.cells >> 32), bb.occupied, bb.used, bb.name, bb.extra_size});
    }
    out.push_back(pl.n_housing);
    for (int i = 0; i < pl.n_housing; ++i) out.insert(out.end(), {pl.housing[i].card, pl.housing[i].building, pl.housing[i].used});
    for (int a = 0; a < kNumActions; ++a) {
      out.push_back(pl.n_action_tokens[a]);
      for (int i = 0; i < pl.n_action_tokens[a]; ++i) out.push_back(static_cast<int>(pl.action_tokens[a][i]));
    }
    out.push_back(pl.n_card_tokens);
    for (int i = 0; i < pl.n_card_tokens; ++i) out.insert(out.end(), {pl.card_tokens[i].card, pl.card_tokens[i].n});
    out.insert(out.end(), {pl.n_tucked, pl.full_cover_scored, pl.last_action});
  }
  out.push_back(s.current);
  out.push_back(static_cast<int>(s.deck.size()));
  out.push_back(static_cast<int>(s.discard.size()));
  out.insert(out.end(), s.discard.begin(), s.discard.end());
  out.insert(out.end(), s.display, s.display + kDisplaySize);
  out.push_back(static_cast<int>(s.final_deck.size()));
  out.insert(out.end(), {s.break_pos, s.break_by, s.n_base_projects});
  out.insert(out.end(), s.base_projects, s.base_projects + s.n_base_projects);
  out.push_back(static_cast<int>(s.base_project_deck.size()));
  out.insert(out.end(), s.base_project_deck.begin(), s.base_project_deck.end());
  out.insert(out.end(), {s.borrowed_action, s.borrowed_upgraded, s.n_projects});
  out.insert(out.end(), s.projects, s.projects + s.n_projects);
  out.push_back(s.n_project_tokens);
  for (int i = 0; i < s.n_project_tokens; ++i) {
    const ProjectTokens& t = s.project_tokens[i];
    out.insert(out.end(), {t.project, t.slot[0], t.slot[1], t.slot[2]});
  }
  out.insert(out.end(), {s.donations, s.partner_zoos_available, s.universities_available});
  for (int t = 0; t < 2; ++t) {
    out.push_back(s.n_bonus_tiles[t]);
    out.insert(out.end(), s.bonus_tiles[t], s.bonus_tiles[t] + s.n_bonus_tiles[t]);
  }
  out.insert(out.end(), {s.final_card_discarded, s.end_by, s.final_turns, s.over, s.scores[0], s.scores[1]});
  out.push_back(s.turn_depth);
  for (int d = 0; d < s.turn_depth; ++d) {
    const TurnCtx& t = s.turns[d];
    out.insert(out.end(), {t.action, t.xtoken, t.strength, t.has_strength, t.n_after});
    for (int i = 0; i < t.n_after; ++i) {
      if (is_private_frame(t.after[i].kind)) throw std::logic_error("arknova: a private frame was deferred");
      pack_frame(out, t.after[i], false);
    }
  }
  out.push_back(s.stack_n);
  for (int i = 0; i < s.stack_n; ++i) pack_frame(out, s.stack[i], true);
  out.push_back(static_cast<int>(prefix.size()));
  out.insert(out.end(), prefix.begin(), prefix.end());
}

void ArknovaState::unpack_public(const std::vector<int>& in) {
  Reader r(in);
  if (r.next() != kPackVersion) throw std::runtime_error("arknova: packed public state has the wrong version");
  core = State{};
  core.map = r.as<std::int8_t>();
  if (core.map < 0 || core.map >= num_maps()) throw std::runtime_error("arknova: packed public state has a bad map");
  core.harbor_used = r.next() != 0;
  for (int p = 0; p < kPlayers; ++p) {
    Player& pl = core.players[p];
    pl.zoo.map = core.map;
    pl.appeal = r.as<std::int16_t>();
    pl.money = r.as<std::int16_t>();
    pl.conservation = r.as<std::int16_t>();
    pl.reputation = r.as<std::int8_t>();
    pl.xtokens = r.as<std::int8_t>();
    pl.upgraded = r.as<std::uint8_t>();
    pl.workers = r.as<std::int8_t>();
    for (auto& v : pl.slots) v = r.as<std::int8_t>();
    for (auto& v : pl.workers_out) v = r.as<std::int8_t>();
    pl.n_partner_zoos = static_cast<std::int8_t>(r.count(kMaxPartnerZoos, "partner zoos"));
    for (int i = 0; i < pl.n_partner_zoos; ++i) pl.partner_zoos[i] = r.as<std::int8_t>();
    pl.n_universities = static_cast<std::int8_t>(r.count(kNumUniversities, "universities"));
    for (int i = 0; i < pl.n_universities; ++i) pl.universities[i] = r.as<std::int8_t>();
    for (bool& t : pl.tokens) t = r.next() != 0;
    pl.hand_size = r.as<std::int16_t>();
    int n_known = r.count(kCardSlots, "known hand cards");
    for (int i = 0; i < n_known; ++i) pl.hand_known.at(static_cast<std::size_t>(r.next())) = 1;
    int n_maybe = r.count(kCardSlots, "maybe tucked cards");
    for (int i = 0; i < n_maybe; ++i) pl.maybe_tucked.at(static_cast<std::size_t>(r.next())) = 1;
    pl.n_zoo_cards = static_cast<std::int8_t>(r.count(kMaxZooCards, "zoo cards"));
    for (int i = 0; i < pl.n_zoo_cards; ++i) pl.zoo_cards[i] = r.as<CardIdx>();
    pl.n_final = static_cast<std::int8_t>(r.count(kMaxFinalCards, "final cards"));
    std::fill(pl.final_cards, pl.final_cards + kMaxFinalCards, kNoCard);
    int n_buildings = r.count(kMaxBuildings, "buildings");
    for (int b = 0; b < n_buildings; ++b) {
      auto kind = static_cast<BuildingKind>(r.next());
      int size = r.next();
      CellMask cells = static_cast<CellMask>(static_cast<std::uint32_t>(r.next()));
      cells |= static_cast<CellMask>(static_cast<std::uint32_t>(r.next())) << 32;
      int occupied = r.next(), used = r.next();
      auto name = r.as<CardIdx>();
      int extra = r.next();
      int id = pl.zoo.place(kind, size, cells, name, nullptr);
      pl.zoo.buildings[id].occupied = occupied != 0;
      pl.zoo.buildings[id].used = static_cast<std::int8_t>(used);
      pl.zoo.buildings[id].extra_size = static_cast<std::int8_t>(extra);
    }
    pl.n_housing = static_cast<std::int8_t>(r.count(kMaxHousing, "housing"));
    for (int i = 0; i < pl.n_housing; ++i) {
      pl.housing[i].card = r.as<CardIdx>();
      pl.housing[i].building = r.as<std::int8_t>();
      pl.housing[i].used = r.as<std::int8_t>();
    }
    for (int a = 0; a < kNumActions; ++a) {
      pl.n_action_tokens[a] = static_cast<std::int8_t>(r.count(kMaxActionTokens, "action tokens"));
      for (int i = 0; i < pl.n_action_tokens[a]; ++i) pl.action_tokens[a][i] = r.as<TokenKind>();
    }
    pl.n_card_tokens = static_cast<std::int8_t>(r.count(kMaxCardTokens, "card tokens"));
    for (int i = 0; i < pl.n_card_tokens; ++i) {
      pl.card_tokens[i].card = r.as<CardIdx>();
      pl.card_tokens[i].n = r.as<std::int8_t>();
    }
    pl.n_tucked = static_cast<std::int8_t>(r.count(kMaxTucked, "tucked cards"));
    std::fill(pl.tucked, pl.tucked + kMaxTucked, kNoCard);
    pl.full_cover_scored = r.next() != 0;
    pl.last_action = r.as<std::int8_t>();
  }
  State& s = core;
  s.current = r.as<std::int8_t>();
  s.deck.assign(static_cast<std::size_t>(r.count(kCardSlots, "deck")), kNoCard);
  s.discard.resize(static_cast<std::size_t>(r.count(kCardSlots, "discard")));
  for (CardIdx& c : s.discard) c = r.as<CardIdx>();
  for (CardIdx& c : s.display) c = r.as<CardIdx>();
  s.final_deck.assign(static_cast<std::size_t>(r.count(kCardSlots, "final deck")), kNoCard);
  s.break_pos = r.as<std::int8_t>();
  s.break_by = r.as<std::int8_t>();
  s.n_base_projects = static_cast<std::int8_t>(r.count(kBaseProjects, "base projects"));
  for (int i = 0; i < s.n_base_projects; ++i) s.base_projects[i] = r.as<CardIdx>();
  s.base_project_deck.resize(static_cast<std::size_t>(r.count(kCardSlots, "base project deck")));
  for (CardIdx& c : s.base_project_deck) c = r.as<CardIdx>();
  s.borrowed_action = r.as<std::int8_t>();
  s.borrowed_upgraded = r.next() != 0;
  s.n_projects = static_cast<std::int8_t>(r.count(kMaxProjectsInPlay + 2, "projects"));
  for (int i = 0; i < s.n_projects; ++i) s.projects[i] = r.as<CardIdx>();
  s.n_project_tokens = static_cast<std::int8_t>(r.count(kMaxProjectsInPlay + 2, "project tokens"));
  for (int i = 0; i < s.n_project_tokens; ++i) {
    s.project_tokens[i].project = r.as<CardIdx>();
    for (auto& v : s.project_tokens[i].slot) v = r.as<std::int8_t>();
  }
  s.donations = r.as<std::int8_t>();
  s.partner_zoos_available = r.as<std::uint8_t>();
  s.universities_available = r.as<std::uint8_t>();
  for (int t = 0; t < 2; ++t) {
    s.n_bonus_tiles[t] = static_cast<std::int8_t>(r.count(2, "bonus tiles"));
    for (int i = 0; i < s.n_bonus_tiles[t]; ++i) s.bonus_tiles[t][i] = r.as<std::int8_t>();
  }
  s.final_card_discarded = r.next() != 0;
  s.end_by = r.as<std::int8_t>();
  s.final_turns = r.as<std::int8_t>();
  s.over = r.next() != 0;
  s.scores[0] = r.as<std::int16_t>();
  s.scores[1] = r.as<std::int16_t>();
  s.turn_depth = static_cast<std::int8_t>(r.count(kMaxTurnDepth, "turn depth"));
  for (int d = 0; d < s.turn_depth; ++d) {
    TurnCtx& t = s.turns[d];
    t.action = r.as<std::int8_t>();
    t.xtoken = r.next() != 0;
    t.strength = r.as<std::int8_t>();
    t.has_strength = r.next() != 0;
    t.n_after = static_cast<std::int8_t>(r.count(kMaxAfter, "deferred frames"));
    for (int i = 0; i < t.n_after; ++i) t.after[i] = unpack_frame(r);
  }
  s.stack_n = static_cast<std::int8_t>(r.count(kMaxStack, "stack"));
  for (int i = 0; i < s.stack_n; ++i) {
    s.stack[i] = unpack_frame(r);
    s.stack[i].uid = static_cast<std::uint16_t>(i + 1);
  }
  s.next_uid = static_cast<std::uint16_t>(s.stack_n + 1);
  int n_prefix = r.count(kMaxPrefix, "prefix");
  prefix.resize(static_cast<std::size_t>(n_prefix));
  for (ActionId& a : prefix) a = r.next();
  r.finish();
  core_stale_ = true;
}

// ---- hashing, snapshot I/O, masking -----------------------------------------------------------

namespace {

std::vector<int>& pack_buffer() {
  thread_local std::vector<int> buf;
  return buf;
}

std::int16_t& hidden_slot(ArknovaState& s, viz::FieldId fid, const std::vector<int>& idx) {
  switch (fid) {
    case kFieldHand: return s.hand[idx.at(0)][idx.at(1)];
    case kFieldFinalCards: return s.final_cards[idx.at(0)][idx.at(1)];
    case kFieldTucked: return s.tucked[idx.at(0)][idx.at(1)];
    case kFieldPrivateCards: return s.private_cards[idx.at(0)][idx.at(1)][idx.at(2)];
    case kFieldFinalKnown: return s.final_known[idx.at(0)][idx.at(1)];
    default: throw std::invalid_argument("arknova: not a card slot field");
  }
}

int hidden_value(const ArknovaState& s, viz::FieldId fid, const std::vector<int>& idx) {
  return hidden_slot(const_cast<ArknovaState&>(s), fid, idx);
}

int any_to_int(const std::any& v) {
  if (v.type() == typeid(int)) return std::any_cast<int>(v);
  if (v.type() == typeid(long)) return static_cast<int>(std::any_cast<long>(v));
  if (v.type() == typeid(long long)) return static_cast<int>(std::any_cast<long long>(v));
  throw std::invalid_argument("arknova: slot value is not an int");
}

}  // namespace

StateHash64 ArknovaState::state_hash() const {
  Hasher h;
  std::vector<int>& buf = pack_buffer();
  pack_public(buf);
  for (int v : buf) h.add(static_cast<std::uint32_t>(v));
  h.combine_bytes(hand, sizeof hand);
  h.combine_bytes(final_cards, sizeof final_cards);
  h.combine_bytes(tucked, sizeof tucked);
  h.combine_bytes(private_cards, sizeof private_cards);
  h.combine_bytes(final_known, sizeof final_known);
  // Draw order never enters the hash; the deck contents follow from every other zone.
  h.add(core.deck.size());
  h.add(core.final_deck.size());
  return h.finalize();
}

void ArknovaState::hash_field_slot(Hasher& h, viz::FieldId fid, const std::vector<int>& idx) const {
  if (fid == kFieldPublic) {
    std::vector<int>& buf = pack_buffer();
    pack_public(buf);
    h.add(buf.size());
    for (int v : buf) h.add(static_cast<std::uint32_t>(v));
    return;
  }
  h.add(hidden_value(*this, fid, idx) + 2);
}

std::any ArknovaState::read_field_slot(viz::FieldId fid, const std::vector<int>& idx) const {
  if (fid == kFieldPublic) {
    std::vector<int> buf;
    pack_public(buf);
    return std::any(std::move(buf));
  }
  return std::any(hidden_value(*this, fid, idx));
}

void ArknovaState::write_field_slot(viz::FieldId fid, const std::vector<int>& idx, const std::any& value) {
  if (fid == kFieldPublic) {
    unpack_public(viz::detail::any_to_int_vec(value, "arknova public"));
    return;
  }
  hidden_slot(*this, fid, idx) = static_cast<std::int16_t>(any_to_int(value));
  core_stale_ = true;
}

void ArknovaState::mask_field_slot(viz::FieldId fid, const std::vector<int>& idx) {
  if (fid == kFieldPublic) return;
  hidden_slot(*this, fid, idx) = kHiddenCard;
}

void ArknovaState::mask_all_hidden_slots(const viz::VisibilitySchema& sch, int perspective) {
  IGameState::mask_all_hidden_slots(sch, perspective);
  // The core keeps truth copies of hidden data; drop what the perspective cannot see so nothing
  // reading the masked core can learn it.
  for (int p = 0; p < kPlayers; ++p) {
    if (p == perspective) continue;
    Player& pl = core.players[p];
    for (int c = 0; c < kCardSlots; ++c) {
      if (!pl.shown_in_hand(static_cast<CardIdx>(c))) pl.hand[c] = pl.hand_known[c] = 0;
    }
    std::fill(pl.final_cards, pl.final_cards + kMaxFinalCards, kNoCard);
    std::fill(pl.tucked, pl.tucked + kMaxTucked, kNoCard);
    for (int fi : private_frames_of(core, p)) {
      Frame& f = core.stack[fi];
      int n_off, list_off;
      private_list_offsets(f.kind, &n_off, &list_off);
      for (int j = 0; j < f.d[n_off]; ++j) f.d[list_off + j] = kNull;
    }
  }
  std::fill(core.deck.begin(), core.deck.end(), kNoCard);
  std::fill(core.final_deck.begin(), core.final_deck.end(), kNoCard);
  core_stale_ = true;
}

}  // namespace board_ai::arknova
