#include "arknova_json.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include "arknova_internal.h"

namespace board_ai::arknova {
namespace {

// ---- a minimal JSON value, parser and writer ---------------------------------------------

struct J {
  enum Type { kNull, kBool, kNum, kStr, kArr, kObj } t = kNull;
  bool b = false;
  long long num = 0;
  std::string s;
  std::vector<J> arr;
  std::vector<std::pair<std::string, J>> obj;

  bool is_null() const { return t == kNull; }
  const J* find(const std::string& key) const {
    for (const auto& kv : obj) {
      if (kv.first == key) return &kv.second;
    }
    return nullptr;
  }
  const J& at(const std::string& key) const {
    const J* v = find(key);
    if (!v) throw std::runtime_error("arknova json: missing key " + key);
    return *v;
  }
  long long i() const {
    if (t != kNum) throw std::runtime_error("arknova json: expected a number");
    return num;
  }
  bool boolean() const {
    if (t != kBool) throw std::runtime_error("arknova json: expected a bool");
    return b;
  }
  const std::string& str() const {
    if (t != kStr) throw std::runtime_error("arknova json: expected a string");
    return s;
  }
  const std::vector<J>& list() const {
    if (t != kArr) throw std::runtime_error("arknova json: expected a list");
    return arr;
  }
};

class Parser {
 public:
  explicit Parser(const std::string& text) : p_(text.c_str()), end_(text.c_str() + text.size()) {}
  J parse() {
    J v = value();
    ws();
    if (p_ != end_) fail("trailing characters");
    return v;
  }

 private:
  const char* p_;
  const char* end_;

  [[noreturn]] void fail(const char* what) { throw std::runtime_error(std::string("arknova json parse: ") + what); }
  void ws() {
    while (p_ < end_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\r' || *p_ == '\t')) ++p_;
  }
  bool lit(const char* w) {
    std::size_t n = std::strlen(w);
    if (static_cast<std::size_t>(end_ - p_) >= n && std::strncmp(p_, w, n) == 0) {
      p_ += n;
      return true;
    }
    return false;
  }
  std::string string() {
    if (*p_ != '"') fail("expected a string");
    ++p_;
    std::string out;
    while (p_ < end_ && *p_ != '"') {
      if (*p_ == '\\') {
        ++p_;
        if (p_ >= end_) fail("bad escape");
        char c = *p_;
        if (c == 'n') out += '\n';
        else if (c == 't') out += '\t';
        else if (c == 'u') {
          if (end_ - p_ < 5) fail("bad unicode escape");
          unsigned code = static_cast<unsigned>(std::stoul(std::string(p_ + 1, p_ + 5), nullptr, 16));
          if (code < 0x80) out += static_cast<char>(code);
          else if (code < 0x800) {
            out += static_cast<char>(0xC0 | (code >> 6));
            out += static_cast<char>(0x80 | (code & 0x3F));
          } else {
            out += static_cast<char>(0xE0 | (code >> 12));
            out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (code & 0x3F));
          }
          p_ += 4;
        } else out += c;
        ++p_;
      } else {
        out += *p_++;
      }
    }
    if (p_ >= end_) fail("unterminated string");
    ++p_;
    return out;
  }
  J value() {
    ws();
    if (p_ >= end_) fail("unexpected end");
    J v;
    char c = *p_;
    if (c == '{') {
      v.t = J::kObj;
      ++p_;
      ws();
      if (*p_ == '}') {
        ++p_;
        return v;
      }
      while (true) {
        ws();
        std::string key = string();
        ws();
        if (*p_ != ':') fail("expected ':'");
        ++p_;
        v.obj.emplace_back(key, value());
        ws();
        if (*p_ == ',') {
          ++p_;
          continue;
        }
        if (*p_ == '}') {
          ++p_;
          return v;
        }
        fail("expected ',' or '}'");
      }
    }
    if (c == '[') {
      v.t = J::kArr;
      ++p_;
      ws();
      if (*p_ == ']') {
        ++p_;
        return v;
      }
      while (true) {
        v.arr.push_back(value());
        ws();
        if (*p_ == ',') {
          ++p_;
          continue;
        }
        if (*p_ == ']') {
          ++p_;
          return v;
        }
        fail("expected ',' or ']'");
      }
    }
    if (c == '"') {
      v.t = J::kStr;
      v.s = string();
      return v;
    }
    if (lit("null")) return v;
    if (lit("true")) {
      v.t = J::kBool;
      v.b = true;
      return v;
    }
    if (lit("false")) {
      v.t = J::kBool;
      return v;
    }
    if (c == '-' || (c >= '0' && c <= '9')) {
      char* stop = nullptr;
      v.t = J::kNum;
      v.num = std::strtoll(p_, &stop, 10);
      if (stop == p_) fail("bad number");
      p_ = stop;
      return v;
    }
    fail("unexpected character");
  }
};

class Writer {
 public:
  std::string out;
  void raw(const std::string& s) { out += s; }
  void str(const std::string& s) {
    out += '"';
    for (char c : s) {
      if (c == '"' || c == '\\') out += '\\';
      out += c;
    }
    out += '"';
  }
  void num(long long v) { out += std::to_string(v); }
  void boolean(bool v) { out += v ? "true" : "false"; }
  void null() { out += "null"; }
  void key(const std::string& k) {
    sep();
    str(k);
    out += ':';
  }
  void begin(char c) {
    sep();
    out += c;
  }
  void end(char c) { out += c; }
  // Called before every value or key: a comma unless this is the first item of its container.
  void sep() {
    if (!out.empty() && out.back() != '[' && out.back() != '{' && out.back() != ':') out += ',';
  }
};

// Value writers: each writes a separator first.
void w_num(Writer& w, long long v) {
  w.sep();
  w.num(v);
}
void w_bool(Writer& w, bool v) {
  w.sep();
  w.boolean(v);
}
void w_str(Writer& w, const std::string& v) {
  w.sep();
  w.str(v);
}
void w_null(Writer& w) {
  w.sep();
  w.null();
}
void w_card(Writer& w, CardIdx c) {
  if (c == kNoCard) {
    w_null(w);
  } else {
    w_str(w, card_id_string(c));
  }
}
template <class It>
void w_cards(Writer& w, It first, It last) {
  w.begin('[');
  for (It it = first; it != last; ++it) w_card(w, *it);
  w.end(']');
}
void w_cells(Writer& w, CellMask cells) {
  w.begin('[');
  for (int c = 0; c < kMapCells; ++c) {
    if (!(cells >> c & 1)) continue;
    w.begin('[');
    w_num(w, map_cell(c).q);
    w_num(w, map_cell(c).r);
    w.end(']');
  }
  w.end(']');
}

CardIdx parse_card(const J& v) {
  if (v.is_null()) return kNoCard;
  return card_by_id(std::stoi(v.str()));
}

int name_index(const std::string& name, const char* const* names, int n, const char* what) {
  for (int i = 0; i < n; ++i) {
    if (name == names[i]) return i;
  }
  throw std::runtime_error(std::string("arknova json: unknown ") + what + " " + name);
}
int action_index(const std::string& name) { return name_index(name, kActionNames, kNumActions, "action"); }
int kind_index(const std::string& name) {
  return name_index(name, kBuildingKindNames, static_cast<int>(BuildingKind::kNumKinds), "building kind");
}

CellMask parse_cells(const J& v) {
  CellMask m = 0;
  for (const J& c : v.list()) {
    int idx = cell_index(static_cast<int>(c.list().at(0).i()), static_cast<int>(c.list().at(1).i()));
    if (idx < 0) throw std::runtime_error("arknova json: cell off the map");
    m |= CellMask{1} << idx;
  }
  return m;
}

// ---- frame data descriptors ---------------------------------------------------------------

struct Field {
  const char* name;
  char type;  // I int, B bool, C card, L card list, P player list, A action, M action list, K kinds list,
              // T task list, H animals phase, O "only" card list
  int off;
  int off2;
};

const std::vector<Field>& frame_fields(FrameKind k) {
  static const std::map<FrameKind, std::vector<Field>> table = {
      {FrameKind::ab_boost, {{"action", 'A', fd::boost_action, 0}}},
      {FrameKind::ab_dig, {{"left", 'I', fd::left, 0}}},
      {FrameKind::ab_sun, {{"left", 'I', fd::left, 0}}},
      {FrameKind::ab_pouch, {{"left", 'I', fd::left, 0}}},
      {FrameKind::ab_hunter, {{"cards", 'L', fd::cards_n, fd::cards_list}}},
      {FrameKind::ab_resist, {{"cards", 'L', fd::cards_n, fd::cards_list}}},
      {FrameKind::ab_hypnosis_pick, {{"candidates", 'P', fd::candidates_n, fd::candidates}, {"reach", 'I', fd::hypnosis_reach, 0}}},
      {FrameKind::ab_pilfer_pick, {{"candidates", 'P', fd::candidates_n, fd::candidates}}},
      {FrameKind::ab_pick, {{"cards", 'L', fd::pick_n, fd::pick_cards}, {"count", 'I', fd::pick_count, 0}}},
      {FrameKind::ab_pilfer, {{"thief", 'I', fd::pilfer_thief, 0}}},
      {FrameKind::ab_project, {{"only", 'O', fd::project_only, 0}}},
      {FrameKind::ab_snap,
       {{"left", 'I', fd::left, 0}, {"can_replenish", 'B', fd::snap_can_replenish, 0},
        {"replenish_between", 'B', fd::snap_replenish_between, 0}}},
      {FrameKind::act_animals,
       {{"remaining", 'I', fd::animals_remaining, 0}, {"played", 'I', fd::animals_played, 0},
        {"phase", 'H', fd::animals_phase, 0}, {"all_small", 'B', fd::animals_all_small, 0}}},
      {FrameKind::act_association,
       {{"strength", 'I', fd::assoc_strength, 0}, {"remaining", 'I', fd::assoc_remaining, 0},
        {"used", 'T', fd::assoc_used, 0}, {"donated", 'B', fd::assoc_donated, 0}, {"side2", 'B', fd::assoc_side2, 0}}},
      {FrameKind::act_build,
       {{"budget", 'I', fd::build_budget, 0}, {"built", 'K', fd::build_built_n, fd::build_built},
        {"upgraded", 'B', fd::build_upgraded, 0}, {"normal_done", 'B', fd::build_normal_done, 0},
        {"engineer_used", 'B', fd::build_engineer_used, 0}}},
      {FrameKind::act_cards,
       {{"draw", 'I', fd::cards_draw, 0}, {"discard", 'I', fd::cards_discard, 0}, {"may_snap", 'B', fd::cards_may_snap, 0},
        {"upgraded", 'B', fd::cards_upgraded, 0}, {"drawn", 'I', fd::cards_drawn, 0}}},
      {FrameKind::act_sponsors,
       {{"strength", 'I', sponsors_strength, 0}, {"budget", 'I', sponsors_budget, 0}, {"played", 'I', sponsors_played, 0}}},
      {FrameKind::break_, {{"phase", 'I', fd::phase, 0}, {"order", 'P', fd::break_order_n, fd::break_order}}},
      {FrameKind::conservation_tile, {{"space", 'I', fd::tile_space, 0}}},
      {FrameKind::discard_hand, {{"count", 'I', fd::discard_count, 0}, {"limit", 'I', fd::discard_limit, 0}}},
      {FrameKind::extra_action,
       {{"actions", 'M', fd::extra_actions, 0}, {"optional", 'B', fd::extra_optional, 0}, {"owner", 'I', fd::extra_owner, 0}}},
      {FrameKind::extra_end, {{"phase", 'I', fd::phase, 0}, {"owner", 'I', fd::extra_end_owner, 0}}},
      {FrameKind::final_scoring, {{"phase", 'I', fd::phase, 0}}},
      {FrameKind::free_build, {{"kinds", 'K', fd::free_build_n, fd::free_build_kinds}}},
      {FrameKind::move_animals, {{"building", 'I', move_animals_building, 0}}},
      {FrameKind::sponsor_building, {{"card", 'C', sponsor_building_card, 0}}},
      {FrameKind::turn_end, {{"phase", 'I', fd::phase, 0}}},
  };
  static const std::vector<Field> none;
  auto it = table.find(k);
  return it == table.end() ? none : it->second;
}

constexpr const char* kAnimalPhases[] = {"play", "bonus", "take"};

void write_frame_data(Writer& w, const Frame& f) {
  w.begin('{');
  for (const Field& fl : frame_fields(f.kind)) {
    std::int16_t v = f.d[fl.off];
    if (v == kNull) continue;
    w.key(fl.name);
    switch (fl.type) {
      case 'I': w_num(w, v); break;
      case 'B': w_bool(w, v != 0); break;
      case 'C': w_card(w, v); break;
      case 'A': w_str(w, kActionNames[v]); break;
      case 'H': w_str(w, kAnimalPhases[v]); break;
      case 'O':
        w.begin('[');
        w_card(w, v);
        w.end(']');
        break;
      case 'L': w_cards(w, f.d + fl.off2, f.d + fl.off2 + v); break;
      case 'P':
        w.begin('[');
        for (int i = 0; i < v; ++i) w_num(w, f.d[fl.off2 + i]);
        w.end(']');
        break;
      case 'M':
        w.begin('[');
        for (int a = 0; a < kNumActions; ++a) {
          if (v >> a & 1) w_str(w, kActionNames[a]);
        }
        w.end(']');
        break;
      case 'T':
        w.begin('[');
        for (int t = 0; t < kNumTasks; ++t) {
          if (v >> t & 1) w_str(w, kTaskNames[t]);
        }
        w.end(']');
        break;
      case 'K':
        w.begin('[');
        for (int i = 0; i < v; ++i) {
          w.begin('[');
          w_str(w, kBuildingKindNames[f.d[fl.off2 + 2 * i]]);
          w_num(w, f.d[fl.off2 + 2 * i + 1]);
          w.end(']');
        }
        w.end(']');
        break;
      default: throw std::logic_error("bad frame field type");
    }
  }
  w.end('}');
}

void write_frame(Writer& w, const Frame& f) {
  w.begin('{');
  w.key("kind");
  w_str(w, frame_kind_name(f.kind));
  w.key("player");
  w_num(w, f.player);
  w.key("data");
  write_frame_data(w, f);
  w.end('}');
}

FrameKind frame_kind_from(const std::string& name) {
  for (int k = 0; k < static_cast<int>(FrameKind::kNumFrameKinds); ++k) {
    if (name == frame_kind_name(static_cast<FrameKind>(k))) return static_cast<FrameKind>(k);
  }
  throw std::runtime_error("arknova json: unknown frame kind " + name);
}

Frame read_frame(const J& v) {
  Frame f;
  f.kind = frame_kind_from(v.at("kind").str());
  f.player = static_cast<std::int8_t>(v.at("player").i());
  const J& data = v.at("data");
  for (const auto& kv : data.obj) {
    const Field* fl = nullptr;
    for (const Field& c : frame_fields(f.kind)) {
      if (kv.first == c.name) fl = &c;
    }
    if (!fl) throw std::runtime_error("arknova json: unknown data key " + kv.first + " for " + v.at("kind").str());
    const J& x = kv.second;
    if (x.is_null()) continue;
    std::int16_t& slot = f.d[fl->off];
    switch (fl->type) {
      case 'I': slot = static_cast<std::int16_t>(x.i()); break;
      case 'B': slot = x.boolean(); break;
      case 'C': slot = parse_card(x); break;
      case 'A': slot = static_cast<std::int16_t>(action_index(x.str())); break;
      case 'H': slot = static_cast<std::int16_t>(name_index(x.str(), kAnimalPhases, 3, "animals phase")); break;
      case 'O':
        if (x.list().size() != 1) throw std::runtime_error("arknova json: 'only' must hold one card");
        slot = parse_card(x.list()[0]);
        break;
      case 'L':
        slot = static_cast<std::int16_t>(x.list().size());
        for (std::size_t i = 0; i < x.list().size(); ++i) f.d[fl->off2 + i] = parse_card(x.list()[i]);
        break;
      case 'P':
        slot = static_cast<std::int16_t>(x.list().size());
        for (std::size_t i = 0; i < x.list().size(); ++i) f.d[fl->off2 + i] = static_cast<std::int16_t>(x.list()[i].i());
        break;
      case 'M':
        slot = 0;
        for (const J& a : x.list()) slot |= static_cast<std::int16_t>(1 << action_index(a.str()));
        break;
      case 'T':
        slot = 0;
        for (const J& t : x.list()) slot |= static_cast<std::int16_t>(1 << name_index(t.str(), kTaskNames, kNumTasks, "task"));
        break;
      case 'K':
        slot = static_cast<std::int16_t>(x.list().size());
        for (std::size_t i = 0; i < x.list().size(); ++i) {
          f.d[fl->off2 + 2 * i] = static_cast<std::int16_t>(kind_index(x.list()[i].list().at(0).str()));
          f.d[fl->off2 + 2 * i + 1] = static_cast<std::int16_t>(x.list()[i].list().at(1).i());
        }
        break;
      default: throw std::logic_error("bad frame field type");
    }
  }
  return f;
}

// ---- turn context ------------------------------------------------------------------------

void write_turn(Writer& w, const State& s, int depth) {
  w.begin('{');
  if (depth > 0) {
    const TurnCtx& t = s.turns[depth - 1];
    w.key("action");
    w_str(w, kActionNames[t.action]);
    w.key("after");
    w.begin('[');
    for (int i = 0; i < t.n_after; ++i) {
      w.begin('[');
      w_str(w, frame_kind_name(t.after[i].kind));
      w_num(w, t.after[i].player);
      write_frame_data(w, t.after[i]);
      w.end(']');
    }
    w.end(']');
    w.key("xtoken");
    w_bool(w, t.xtoken);
    if (t.has_strength) {
      w.key("strength");
      w_num(w, t.strength);
    }
    if (depth > 1) {
      w.key("outer");
      write_turn(w, s, depth - 1);
    }
  }
  w.end('}');
}

void read_turn(State& s, const J& v, int depth_from_top, std::vector<TurnCtx>& out) {
  if (v.obj.empty()) return;
  TurnCtx t;
  t.action = static_cast<std::int8_t>(action_index(v.at("action").str()));
  t.xtoken = v.at("xtoken").boolean();
  if (const J* st = v.find("strength")) {
    t.strength = static_cast<std::int8_t>(st->i());
    t.has_strength = true;
  }
  for (const J& a : v.at("after").list()) {
    J frame_json;
    frame_json.t = J::kObj;
    frame_json.obj.emplace_back("kind", a.list().at(0));
    frame_json.obj.emplace_back("player", a.list().at(1));
    frame_json.obj.emplace_back("data", a.list().at(2));
    if (t.n_after >= kMaxAfter) throw std::runtime_error("arknova json: too many deferred frames");
    t.after[t.n_after++] = read_frame(frame_json);
  }
  if (const J* outer = v.find("outer")) read_turn(s, *outer, depth_from_top + 1, out);
  out.push_back(t);
}

// ---- players ------------------------------------------------------------------------------

void write_player(Writer& w, const Player& pl) {
  w.begin('{');
  w.key("appeal"); w_num(w, pl.appeal);
  w.key("money"); w_num(w, pl.money);
  w.key("conservation"); w_num(w, pl.conservation);
  w.key("reputation"); w_num(w, pl.reputation);
  w.key("xtokens"); w_num(w, pl.xtokens);
  w.key("upgraded");
  w.begin('[');
  for (int a = 0; a < kNumActions; ++a) {
    if (pl.is_upgraded(a)) w_str(w, kActionNames[a]);
  }
  w.end(']');
  w.key("workers"); w_num(w, pl.workers);
  w.key("workers_out");
  w.begin('{');
  for (int t = 0; t < kNumTasks; ++t) {
    if (!pl.workers_out[t]) continue;
    w.key(kTaskNames[t]);
    w_num(w, pl.workers_out[t]);
  }
  w.end('}');
  w.key("partner_zoos");
  w.begin('[');
  for (int i = 0; i < pl.n_partner_zoos; ++i) w_str(w, kContinentNames[pl.partner_zoos[i]]);
  w.end(']');
  w.key("universities");
  w.begin('[');
  for (int i = 0; i < pl.n_universities; ++i) w_str(w, kUniversityNames[pl.universities[i]]);
  w.end(']');
  w.key("tokens");
  w.begin('[');
  for (bool t : pl.tokens) w_bool(w, t);
  w.end(']');
  std::vector<CardIdx> hand = pl.hand_cards(), known;
  for (CardIdx c : hand) {
    if (pl.hand_known[c]) known.push_back(c);
  }
  w.key("hand"); w_cards(w, hand.begin(), hand.end());
  w.key("public_hand"); w_cards(w, known.begin(), known.end());
  w.key("zoo_cards"); w_cards(w, pl.zoo_cards, pl.zoo_cards + pl.n_zoo_cards);
  w.key("final_cards"); w_cards(w, pl.final_cards, pl.final_cards + pl.n_final);
  w.key("housing");
  w.begin('{');
  for (int i = 0; i < pl.n_housing; ++i) {
    w.key(card_id_string(pl.housing[i].card));
    w.begin('[');
    w_num(w, pl.housing[i].building);
    w_num(w, pl.housing[i].used);
    w.end(']');
  }
  w.end('}');
  w.key("action_tokens");
  w.begin('{');
  for (int a = 0; a < kNumActions; ++a) {
    if (!pl.n_action_tokens[a]) continue;
    w.key(kActionNames[a]);
    w.begin('[');
    for (int i = 0; i < pl.n_action_tokens[a]; ++i) w_str(w, kTokenKindNames[static_cast<int>(pl.action_tokens[a][i])]);
    w.end(']');
  }
  w.end('}');
  w.key("card_tokens");
  w.begin('{');
  CardIdx waza = card_by_id(227);
  for (int i = 0; i < pl.n_card_tokens; ++i) {
    const CardTokens& t = pl.card_tokens[i];
    w.key(card_id_string(t.card));
    if (t.card == waza) {
      w_str(w, t.n == kWazaSmall ? "small" : "large");
    } else {
      w_num(w, t.n);
    }
  }
  w.end('}');
  w.key("tucked"); w_cards(w, pl.tucked, pl.tucked + pl.n_tucked);
  w.key("full_cover_scored"); w_bool(w, pl.full_cover_scored);
  w.key("last_action");
  if (pl.last_action < 0) {
    w_null(w);
  } else {
    w_str(w, pl.last_action == kLastActionXtoken ? "xtoken" : kActionNames[pl.last_action]);
  }
  w.key("slots");
  w.begin('[');
  for (int a : pl.slots) w_str(w, kActionNames[a]);
  w.end(']');
  w.key("buildings");
  w.begin('[');
  for (int b = 0; b < pl.zoo.n_buildings; ++b) {
    const Building& bb = pl.zoo.buildings[b];
    w.begin('{');
    w.key("id"); w_num(w, b);
    w.key("kind"); w_str(w, kBuildingKindNames[static_cast<int>(bb.kind)]);
    w.key("size"); w_num(w, bb.size);
    w.key("cells"); w_cells(w, bb.cells);
    w.key("occupied"); w_bool(w, bb.occupied);
    w.key("used"); w_num(w, bb.used);
    w.key("name"); w_card(w, bb.name);
    w.key("extra_size"); w_num(w, bb.extra_size);
    w.end('}');
  }
  w.end(']');
  w.end('}');
}

template <class T, std::size_t N>
std::int8_t fill_list(T (&dst)[N], const std::vector<T>& src, const char* what) {
  if (src.size() > N) throw std::runtime_error(std::string("arknova json: too many ") + what);
  std::copy(src.begin(), src.end(), dst);
  return static_cast<std::int8_t>(src.size());
}

std::vector<CardIdx> parse_cards(const J& v) {
  std::vector<CardIdx> out;
  for (const J& c : v.list()) out.push_back(parse_card(c));
  return out;
}

void read_player(Player& pl, const J& v) {
  pl = Player{};
  pl.appeal = static_cast<std::int16_t>(v.at("appeal").i());
  pl.money = static_cast<std::int16_t>(v.at("money").i());
  pl.conservation = static_cast<std::int16_t>(v.at("conservation").i());
  pl.reputation = static_cast<std::int8_t>(v.at("reputation").i());
  pl.xtokens = static_cast<std::int8_t>(v.at("xtokens").i());
  for (const J& a : v.at("upgraded").list()) pl.upgraded |= static_cast<std::uint8_t>(1u << action_index(a.str()));
  pl.workers = static_cast<std::int8_t>(v.at("workers").i());
  for (const auto& kv : v.at("workers_out").obj) {
    pl.workers_out[name_index(kv.first, kTaskNames, kNumTasks, "task")] = static_cast<std::int8_t>(kv.second.i());
  }
  for (const J& c : v.at("partner_zoos").list()) {
    pl.partner_zoos[pl.n_partner_zoos++] = static_cast<std::int8_t>(name_index(c.str(), kContinentNames, kNumContinents, "continent"));
  }
  for (const J& u : v.at("universities").list()) {
    pl.universities[pl.n_universities++] =
        static_cast<std::int8_t>(name_index(u.str(), kUniversityNames, kNumUniversities, "university"));
  }
  const auto& tokens = v.at("tokens").list();
  if (tokens.size() != kNumMapTokens) throw std::runtime_error("arknova json: map token count");
  for (int i = 0; i < kNumMapTokens; ++i) pl.tokens[i] = tokens[i].boolean();
  for (CardIdx c : parse_cards(v.at("hand"))) {
    pl.hand[c] = 1;
    ++pl.hand_size;
  }
  for (CardIdx c : parse_cards(v.at("public_hand"))) {
    if (pl.hand[c]) pl.hand_known[c] = 1;
  }
  pl.n_zoo_cards = fill_list(pl.zoo_cards, parse_cards(v.at("zoo_cards")), "zoo cards");
  pl.n_final = fill_list(pl.final_cards, parse_cards(v.at("final_cards")), "final cards");
  for (const auto& kv : v.at("housing").obj) {
    if (pl.n_housing >= kMaxHousing) throw std::runtime_error("arknova json: too many housed animals");
    HousingEntry& h = pl.housing[pl.n_housing++];
    h.card = card_by_id(std::stoi(kv.first));
    h.building = static_cast<std::int8_t>(kv.second.list().at(0).i());
    h.used = static_cast<std::int8_t>(kv.second.list().at(1).i());
  }
  for (const auto& kv : v.at("action_tokens").obj) {
    int a = action_index(kv.first);
    for (const J& t : kv.second.list()) {
      if (pl.n_action_tokens[a] >= kMaxActionTokens) throw std::runtime_error("arknova json: too many action tokens");
      pl.action_tokens[a][pl.n_action_tokens[a]++] =
          static_cast<TokenKind>(name_index(t.str(), kTokenKindNames, 3, "action token"));
    }
  }
  for (const auto& kv : v.at("card_tokens").obj) {
    CardIdx c = card_by_id(std::stoi(kv.first));
    int n = kv.second.t == J::kStr ? (kv.second.str() == "small" ? kWazaSmall : kWazaLarge) : static_cast<int>(kv.second.i());
    card_tokens_set(pl, c, n);
  }
  pl.n_tucked = fill_list(pl.tucked, parse_cards(v.at("tucked")), "tucked cards");
  pl.full_cover_scored = v.at("full_cover_scored").boolean();
  const J& last = v.at("last_action");
  pl.last_action = last.is_null() ? -1 : last.str() == "xtoken" ? kLastActionXtoken : static_cast<std::int8_t>(action_index(last.str()));
  const auto& slots = v.at("slots").list();
  if (slots.size() != kNumActions) throw std::runtime_error("arknova json: slot count");
  for (int i = 0; i < kNumActions; ++i) pl.slots[i] = static_cast<std::int8_t>(action_index(slots[i].str()));
  for (const J& b : v.at("buildings").list()) {
    if (b.at("id").i() != pl.zoo.n_buildings) throw std::runtime_error("arknova json: building ids out of order");
    int id = pl.zoo.place(static_cast<BuildingKind>(kind_index(b.at("kind").str())), static_cast<int>(b.at("size").i()),
                          parse_cells(b.at("cells")), parse_card(b.at("name")), nullptr);
    Building& bb = pl.zoo.buildings[id];
    bb.occupied = b.at("occupied").boolean();
    bb.used = static_cast<std::int8_t>(b.at("used").i());
    bb.extra_size = static_cast<std::int8_t>(b.at("extra_size").i());
  }
}

void w_opt_int(Writer& w, int v) {
  if (v < 0) {
    w_null(w);
  } else {
    w_num(w, v);
  }
}

}  // namespace

std::string card_id_string(CardIdx c) {
  char buf[8];
  std::snprintf(buf, sizeof buf, "%03d", card(c).id);
  return buf;
}

std::string state_to_json(const State& s) {
  Writer w;
  w.begin('{');
  w.key("players");
  w.begin('[');
  for (const Player& pl : s.players) write_player(w, pl);
  w.end(']');
  w.key("current"); w_num(w, s.current);
  w.key("deck"); w_cards(w, s.deck.begin(), s.deck.end());
  std::vector<CardIdx> discard = s.discard;
  std::sort(discard.begin(), discard.end());
  w.key("discard"); w_cards(w, discard.begin(), discard.end());
  w.key("display"); w_cards(w, s.display, s.display + kDisplaySize);
  w.key("final_deck"); w_cards(w, s.final_deck.begin(), s.final_deck.end());
  w.key("break_pos"); w_num(w, s.break_pos);
  w.key("break_by"); w_opt_int(w, s.break_by);
  w.key("base_projects"); w_cards(w, s.base_projects, s.base_projects + s.n_base_projects);
  std::vector<CardIdx> bpd = s.base_project_deck;
  std::sort(bpd.begin(), bpd.end());
  w.key("base_project_deck"); w_cards(w, bpd.begin(), bpd.end());
  w.key("borrowed");
  if (s.borrowed_action < 0) {
    w_null(w);
  } else {
    w.begin('[');
    w_str(w, kActionNames[s.borrowed_action]);
    w_bool(w, s.borrowed_upgraded);
    w.end(']');
  }
  w.key("projects"); w_cards(w, s.projects, s.projects + s.n_projects);
  w.key("project_tokens");
  w.begin('{');
  for (int i = 0; i < s.n_project_tokens; ++i) {
    const ProjectTokens& t = s.project_tokens[i];
    w.key(card_id_string(t.project));
    w.begin('{');
    for (int k = 0; k < 3; ++k) {
      if (t.slot[k] < 0) continue;
      w.key(std::to_string(k + 1));
      if (t.slot[k] == kBlocked) {
        w_str(w, "blocked");
      } else {
        w_num(w, t.slot[k]);
      }
    }
    w.end('}');
  }
  w.end('}');
  w.key("donations"); w_num(w, s.donations);
  w.key("partner_zoos_available");
  w.begin('[');
  for (int c = 0; c < kNumContinents; ++c) {
    if (s.partner_zoos_available >> c & 1) w_str(w, kContinentNames[c]);
  }
  w.end(']');
  w.key("universities_available");
  w.begin('[');
  for (int u = 0; u < kNumUniversities; ++u) {
    if (s.universities_available >> u & 1) w_str(w, kUniversityNames[u]);
  }
  w.end(']');
  w.key("bonus_tiles");
  w.begin('{');
  for (int t = 0; t < 2; ++t) {
    w.key(std::to_string(kConservationTileSpaces[t]));
    w.begin('[');
    for (int i = 0; i < s.n_bonus_tiles[t]; ++i) w_str(w, kBonusTileNames[s.bonus_tiles[t][i]]);
    w.end(']');
  }
  w.end('}');
  w.key("final_card_discarded"); w_bool(w, s.final_card_discarded);
  w.key("end_by"); w_opt_int(w, s.end_by);
  w.key("final_turns"); w_opt_int(w, s.final_turns);
  w.key("over"); w_bool(w, s.over);
  w.key("scores");
  if (!s.over) {
    w_null(w);
  } else {
    w.begin('[');
    for (int p = 0; p < kPlayers; ++p) w_num(w, s.scores[p]);
    w.end(']');
  }
  w.key("turn");
  write_turn(w, s, s.turn_depth);
  w.key("stack");
  w.begin('[');
  for (int i = 0; i < s.stack_n; ++i) write_frame(w, s.stack[i]);
  w.end(']');
  w.end('}');
  return w.out;
}

void state_from_json(State& s, const std::string& json) {
  registry();
  J v = Parser(json).parse();
  s = State{};
  const auto& players = v.at("players").list();
  if (players.size() != kPlayers) throw std::runtime_error("arknova json: player count");
  for (int p = 0; p < kPlayers; ++p) read_player(s.players[p], players[p]);
  s.current = static_cast<std::int8_t>(v.at("current").i());
  s.deck = parse_cards(v.at("deck"));
  s.discard = parse_cards(v.at("discard"));
  const auto& display = v.at("display").list();
  if (display.size() != kDisplaySize) throw std::runtime_error("arknova json: display size");
  for (int i = 0; i < kDisplaySize; ++i) s.display[i] = parse_card(display[i]);
  s.final_deck = parse_cards(v.at("final_deck"));
  s.break_pos = static_cast<std::int8_t>(v.at("break_pos").i());
  s.break_by = v.at("break_by").is_null() ? -1 : static_cast<std::int8_t>(v.at("break_by").i());
  s.n_base_projects = fill_list(s.base_projects, parse_cards(v.at("base_projects")), "base projects");
  s.base_project_deck = parse_cards(v.at("base_project_deck"));
  const J& borrowed = v.at("borrowed");
  if (!borrowed.is_null()) {
    s.borrowed_action = static_cast<std::int8_t>(action_index(borrowed.list().at(0).str()));
    s.borrowed_upgraded = borrowed.list().at(1).boolean();
  }
  s.n_projects = fill_list(s.projects, parse_cards(v.at("projects")), "projects");
  for (const auto& kv : v.at("project_tokens").obj) {
    if (s.n_project_tokens >= kMaxProjectsInPlay + 2) throw std::runtime_error("arknova json: too many projects");
    ProjectTokens& t = s.project_tokens[s.n_project_tokens++];
    t.project = card_by_id(std::stoi(kv.first));
    for (const auto& slot : kv.second.obj) {
      int pos = std::stoi(slot.first);
      t.slot[pos - 1] = slot.second.t == J::kStr ? kBlocked : static_cast<std::int8_t>(slot.second.i());
    }
  }
  s.donations = static_cast<std::int8_t>(v.at("donations").i());
  for (const J& c : v.at("partner_zoos_available").list()) {
    s.partner_zoos_available |= static_cast<std::uint8_t>(1u << name_index(c.str(), kContinentNames, kNumContinents, "continent"));
  }
  for (const J& u : v.at("universities_available").list()) {
    s.universities_available |=
        static_cast<std::uint8_t>(1u << name_index(u.str(), kUniversityNames, kNumUniversities, "university"));
  }
  for (int t = 0; t < 2; ++t) {
    const J* tiles = v.at("bonus_tiles").find(std::to_string(kConservationTileSpaces[t]));
    if (!tiles) continue;
    for (const J& name : tiles->list()) {
      s.bonus_tiles[t][s.n_bonus_tiles[t]++] =
          static_cast<std::int8_t>(name_index(name.str(), kBonusTileNames, kNumBonusTiles, "bonus tile"));
    }
  }
  s.final_card_discarded = v.at("final_card_discarded").boolean();
  s.end_by = v.at("end_by").is_null() ? -1 : static_cast<std::int8_t>(v.at("end_by").i());
  s.final_turns = v.at("final_turns").is_null() ? -1 : static_cast<std::int8_t>(v.at("final_turns").i());
  s.over = v.at("over").boolean();
  if (!v.at("scores").is_null()) {
    for (int p = 0; p < kPlayers; ++p) s.scores[p] = static_cast<std::int16_t>(v.at("scores").list().at(p).i());
  }
  std::vector<TurnCtx> turns;
  read_turn(s, v.at("turn"), 0, turns);
  if (turns.size() > kMaxTurnDepth) throw std::runtime_error("arknova json: turn nesting too deep");
  s.turn_depth = static_cast<std::int8_t>(turns.size());
  for (std::size_t i = 0; i < turns.size(); ++i) s.turns[i] = turns[i];
  for (const J& f : v.at("stack").list()) {
    int fi = push(s, FrameKind::turn, 0);
    std::uint16_t uid = s.stack[fi].uid;
    s.stack[fi] = read_frame(f);
    s.stack[fi].uid = uid;
  }
}

// ---- option keys ---------------------------------------------------------------------------

std::string option_to_json(const State& s, const Option& o) {
  Writer w;
  w.begin('[');
  w_str(w, head_name(o.head));
  FrameKind top = s.stack_n ? s.top().kind : FrameKind::turn;
  auto card_args = [&](int n) {
    for (int i = 0; i < n; ++i) w_card(w, o.a[i]);
  };
  switch (o.head) {
    case Head::act:
      w_str(w, kActionNames[o.a[0]]);
      w_num(w, o.a[1]);
      break;
    case Head::xtoken: case Head::upgrade: case Head::clever: case Head::multiplier:
      w_str(w, kActionNames[o.a[0]]);
      break;
    case Head::keep: case Head::discard_final: case Head::discard: case Head::pick: case Head::sell: case Head::tuck:
    case Head::discard_hand: case Head::move:
      card_args(o.n);
      break;
    case Head::tile: w_str(w, kBonusTileNames[o.a[0]]); break;
    case Head::display: case Head::snap: case Head::take_small: case Head::discard_display: case Head::target:
    case Head::boost:
      w_num(w, o.a[0]);
      break;
    case Head::take:
      if (top == FrameKind::ab_snap) {
        w_num(w, o.a[0]);
      } else {
        w_card(w, o.a[0]);
      }
      break;
    case Head::partner_zoo: w_str(w, kContinentNames[o.a[0]]); break;
    case Head::university: w_str(w, kUniversityNames[o.a[0]]); break;
    case Head::kind: w_str(w, o.a[0] == kWazaSmall ? "small" : "large"); break;
    case Head::place: case Head::build: case Head::engineer:
      if (o.n >= 2) {
        w_str(w, kBuildingKindNames[o.a[0]]);
        w_num(w, o.a[1]);
      }
      w_cells(w, o.cells);
      break;
    case Head::play:
      w_card(w, o.a[0]);
      if (o.n >= 2) w_num(w, o.a[1]);
      break;
    case Head::play_display:
      w_num(w, o.a[0]);
      if (o.n >= 2) w_num(w, o.a[1]);
      break;
    case Head::support: case Head::release: case Head::play_project: case Head::play_release:
    case Head::play_project_display: case Head::play_release_display:
      if (o.head == Head::play_project_display || o.head == Head::play_release_display) {
        w_num(w, o.a[0]);
      } else {
        w_card(w, o.a[0]);
      }
      w_num(w, o.a[1]);
      w_num(w, o.a[2]);
      if (o.a[3] == kExtraRelease) w_card(w, o.a[4]);
      if (o.a[3] == kExtraToken) w_str(w, "token:" + card_id_string(o.a[4]));
      break;
    default:
      break;
  }
  w.end(']');
  return w.out;
}

std::string option_label(const State& s, const Option& o) { return option_to_json(s, o); }

}  // namespace board_ai::arknova
