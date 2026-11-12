#pragma once

// Ark Nova rules core: a faithful C++ port of ark-nova-ai's Python engine (2 players, both zoos on
// the same map: Map 0, 1-8 or A).
//
// The game is a stack of decision frames. The top frame is either a choice (its player picks one
// Option) or an auto frame (resolved by the engine). `step` applies an option and then resolves
// auto frames until the next choice, exactly like engine.step / engine._settle in Python.
// Option keys mirror the Python tuples so that the two engines can be compared move by move
// (option_to_json gives the Python key).
//
// All state is fixed-size except the hidden decks (deck, final_deck, base_project_deck), which are
// ordered vectors popped from the back like the Python lists.

#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "arknova_cards.h"
#include "arknova_map.h"

namespace board_ai::arknova {

constexpr std::int16_t kNull = -32768;  // "None" in frame data and option fields
constexpr int kCardSlots = 240;  // >= num_cards(); checked at startup
constexpr int kMaxZooCards = 64;
constexpr int kMaxFinalCards = 6;
constexpr int kMaxHousing = 48;
constexpr int kMaxActionTokens = 4;
constexpr int kMaxCardTokens = 12;
constexpr int kMaxTucked = 24;
constexpr int kMaxStack = 48;
// A frame's card list (Hunter reveal) starts at d[1]: room for every predator icon of the game
// (the Spotted Hyena Compound reveals one card per predator icon in the zoo); checked at startup.
constexpr int kFrameData = 32;
constexpr int kMaxTurnDepth = 4;
constexpr int kMaxAfter = 8;
constexpr int kMaxProjectsInPlay = kBaseProjects + kProjectsAboveBoard;

enum class TokenKind : std::uint8_t { kVenom, kConstriction, kMultiplier };
constexpr const char* kTokenKindNames[] = {"venom", "constriction", "multiplier"};
constexpr std::int8_t kLastActionXtoken = kNumActions;  // Player::last_action value for "xtoken"

// ---- frames -------------------------------------------------------------------

#define ARKNOVA_FRAME_KINDS(X)                                                                     \
  X(ab_boost) X(ab_dig) X(ab_hunter) X(ab_hypnosis_pick) X(ab_pick) X(ab_pilfer) X(ab_pilfer_pick)  \
  X(ab_pouch) X(ab_project) X(ab_resist) X(ab_snap) X(ab_sun) X(act_animals) X(act_association)    \
  X(act_build) X(act_cards) X(act_sponsors) X(clever) X(conservation_tile) X(discard_final)         \
  X(discard_hand) X(extra_action) X(free_build) X(gain_partner_zoo) X(gain_university)             \
  X(move_animals) X(multiplier_repeat) X(okapi_stable) X(paid_sponsor) X(place_multiplier)          \
  X(setup_keep) X(sponsor_building) X(take_card) X(turn) X(upgrade) X(upgrade_or_worker)            \
  X(waza_kind)                                                                                      \
  /* auto frames */                                                                                 \
  X(break_) X(extra_end) X(final_scoring) X(next_turn) X(turn_end)

enum class FrameKind : std::uint8_t {
#define ARKNOVA_ENUM(k) k,
  ARKNOVA_FRAME_KINDS(ARKNOVA_ENUM)
#undef ARKNOVA_ENUM
  kNumFrameKinds
};
const char* frame_kind_name(FrameKind k);  // the Python kind ("break" for break_)
bool is_auto(FrameKind k);

struct Frame {
  FrameKind kind = FrameKind::turn;
  std::int8_t player = 0;
  std::uint16_t uid = 0;  // identity while on the stack (Python compares frames with `is`)
  std::int16_t d[kFrameData];
  Frame() { std::fill(std::begin(d), std::end(d), kNull); }
};

// ---- options ------------------------------------------------------------------

// The first element of a Python option key.
#define ARKNOVA_OPTION_HEADS(X)                                                                    \
  X(act) X(xtoken) X(keep) X(repeat) X(skip) X(upgrade) X(worker) X(money) X(tile) X(deck)         \
  X(display) X(clever) X(place) X(multiplier) X(partner_zoo) X(university) X(discard_final)        \
  X(discard) X(done) X(snap) X(build) X(engineer) X(play) X(play_display) X(take_small)            \
  X(reputation) X(donate) X(support) X(play_project) X(play_project_display) X(release)            \
  X(play_release) X(play_release_display) X(break_) X(pick) X(take) X(target) X(move) X(boost)     \
  X(yes) X(no) X(replenish) X(kind) X(card) X(discard_display) X(discard_hand) X(tuck) X(sponsor)  \
  X(project) X(stable) X(return_) X(sell) X(harbor)

enum class Head : std::uint8_t {
#define ARKNOVA_ENUM(k) k,
  ARKNOVA_OPTION_HEADS(ARKNOVA_ENUM)
#undef ARKNOVA_ENUM
  kNumHeads
};
const char* head_name(Head h);

// Option fields are interpreted per head (see option_to_json). Cell sets go in `cells`.
struct Option {
  Head head = Head::skip;
  std::int8_t n = 0;
  std::int16_t a[8] = {kNull, kNull, kNull, kNull, kNull, kNull, kNull, kNull};
  CellMask cells = 0;
  bool operator==(const Option& o) const;
};

// ---- players and state --------------------------------------------------------

struct HousingEntry {
  CardIdx card = kNoCard;
  std::int8_t building = kNoBuilding;
  std::int8_t used = 0;
};
struct CardTokens {
  CardIdx card = kNoCard;
  std::int8_t n = 0;
};

struct Player {
  std::int16_t appeal = 0;
  std::int16_t money = kStartMoney;
  std::int16_t conservation = 0;
  std::int8_t reputation = 1;
  std::int8_t xtokens = 0;
  std::uint8_t upgraded = 0;  // bit per ActionCard
  std::int8_t workers = 1;
  std::int8_t slots[kNumActions] = {kAnimals, kBuild, kCards, kAssociation, kSponsors};  // slots[0] is slot 1
  std::int8_t workers_out[kNumTasks] = {};  // workers placed per task since the last break
  std::int8_t n_partner_zoos = 0;
  std::int8_t partner_zoos[kMaxPartnerZoos] = {};
  std::int8_t n_universities = 0;
  std::int8_t universities[kNumUniversities] = {};
  bool tokens[kNumMapTokens] = {true, true, true, true, true, true, true};  // still on the map's left edge
  std::array<std::uint8_t, kCardSlots> hand{};        // membership by CardIdx
  std::array<std::uint8_t, kCardSlots> hand_known{};  // the opponent saw the card enter the hand (public_hand)
  // Seen in the hand when a card was tucked face down, so now in the hand or tucked: the opponent
  // no longer knows which. Information only (rules never read it, the JSON does not carry it).
  std::array<std::uint8_t, kCardSlots> maybe_tucked{};
  bool shown_in_hand(CardIdx c) const { return hand[c] && hand_known[c] && !maybe_tucked[c]; }
  std::int16_t hand_size = 0;
  std::int8_t n_zoo_cards = 0;
  CardIdx zoo_cards[kMaxZooCards] = {};  // played animals and sponsors, in play order
  std::int8_t n_final = 0;
  CardIdx final_cards[kMaxFinalCards] = {};
  Zoo zoo;
  std::int8_t n_housing = 0;
  HousingEntry housing[kMaxHousing];
  std::int8_t n_action_tokens[kNumActions] = {};
  TokenKind action_tokens[kNumActions][kMaxActionTokens] = {};
  std::int8_t n_card_tokens = 0;
  CardTokens card_tokens[kMaxCardTokens];
  std::int8_t n_tucked = 0;
  CardIdx tucked[kMaxTucked] = {};
  bool full_cover_scored = false;
  std::int8_t last_action = -1;  // ActionCard, kLastActionXtoken, or -1

  int strength(int action) const;
  bool is_upgraded(int action) const { return upgraded >> action & 1; }
  int active_workers() const;
  int hand_limit() const;
  int projects_supported() const;
  int victory_points() const { return arknova::victory_points(appeal, conservation); }
  bool has_university(University u) const;
  bool has_partner_zoo(int continent) const;
  bool in_zoo(CardIdx c) const;
  std::vector<CardIdx> hand_cards() const;  // ascending CardIdx
};
struct ProjectTokens {
  CardIdx project = kNoCard;
  std::int8_t slot[3] = {-1, -1, -1};  // per slot position 1..3: -1 free, player index, or kBlocked
};
constexpr std::int8_t kBlocked = 9;

// A main action in progress (state.turn in Python); nested for extra actions (`outer`).
struct TurnCtx {
  std::int8_t action = -1;
  bool xtoken = false;
  std::int8_t strength = 0;
  bool has_strength = false;  // unset for an X-token turn
  std::int8_t n_after = 0;
  Frame after[kMaxAfter];
};

struct State {
  std::int8_t map = 0;  // map_def index; every zoo is on this map (Zoo::map)
  Player players[kPlayers];
  std::int8_t current = 0;
  bool harbor_used = false;  // turn_flags["harbor"]: the Map 4 harbour was used this turn
  std::vector<CardIdx> deck;  // hidden; drawn from the back
  std::vector<CardIdx> discard;
  CardIdx display[kDisplaySize] = {kNoCard, kNoCard, kNoCard, kNoCard, kNoCard, kNoCard};  // kNoCard = taken
  std::vector<CardIdx> final_deck;  // hidden; drawn from the back
  std::int8_t break_pos = 0;
  std::int8_t break_by = -1;
  std::int8_t n_base_projects = 0;
  CardIdx base_projects[kBaseProjects] = {};
  std::vector<CardIdx> base_project_deck;  // unused base projects
  std::int8_t borrowed_action = -1;  // set while acting with another player's card (Hypnosis)
  bool borrowed_upgraded = false;
  std::int8_t n_projects = 0;
  CardIdx projects[kMaxProjectsInPlay + 2] = {};  // above the board, index 0 is leftmost
  std::int8_t n_project_tokens = 0;
  ProjectTokens project_tokens[kMaxProjectsInPlay + 2];
  std::int8_t donations = 0;
  std::uint8_t partner_zoos_available = 0;  // bit per Continent
  std::uint8_t universities_available = 0;  // bit per University
  std::int8_t n_bonus_tiles[2] = {};        // per conservation tile space (5, 8)
  std::int8_t bonus_tiles[2][2] = {};
  bool final_card_discarded = false;
  std::int8_t end_by = -1;
  std::int8_t final_turns = -1;
  bool over = false;
  std::int16_t scores[kPlayers] = {};
  std::int8_t turn_depth = 0;  // 0 = no turn in progress (state.turn == {})
  TurnCtx turns[kMaxTurnDepth];
  std::int8_t stack_n = 0;
  std::uint16_t next_uid = 1;
  Frame stack[kMaxStack];

  TurnCtx& turn();
  const TurnCtx& turn() const;
  const Frame& top() const { return stack[stack_n - 1]; }
  int to_move() const { return top().player; }
  ProjectTokens* tokens_of(CardIdx project);
  const ProjectTokens* tokens_of(CardIdx project) const;
};

// ---- randomness -------------------------------------------------------------------

// The engine's only random events. Normal play samples them; the differential test replays the
// Python engine's outcomes.
class Chance {
 public:
  virtual ~Chance() = default;
  virtual void shuffle(std::vector<CardIdx>& cards) = 0;
  // One of `candidates` (non-empty, ascending), chosen uniformly.
  virtual CardIdx pick(const std::vector<CardIdx>& candidates) = 0;
};

// ---- engine API ---------------------------------------------------------------

[[noreturn]] void engine_error(const std::string& msg);

// Off by default. The differential test turns it on to reproduce ark-nova-ai where that engine
// departs from the rulebook: it decides whether a second animal can be played before the first
// one's effects resolve (p. 12 example), judges Animals II feasibility before the strength-5
// reputation gain (p. 13), and frees the leaving animal's own enclosure instead of the smallest
// occupied standard enclosure meeting its needs (pp. 12, 15).
extern bool g_upstream_rules;

// Python new_game on map `map` (map_def index) with this engine's own shuffles (the differential
// test imports Python's setup).
void new_game(State& s, std::mt19937_64& rng, int map = 0);
std::vector<Option> legal_options(State& s);
void step(State& s, const Option& option, Chance& chance);
std::vector<int> winners(const State& s);
std::string option_to_json(const State& s, const Option& o);
std::string option_label(const State& s, const Option& o);

}  // namespace board_ai::arknova
