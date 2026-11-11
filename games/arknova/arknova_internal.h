#pragma once

// Shared machinery of the rules core (not part of the public API in arknova_core.h): the frame
// and action registries, frame data layouts, and the effect helpers of effects.py.

#include <array>
#include <vector>

#include "arknova_core.h"

namespace board_ai::arknova {

using IconCounts = std::array<std::int16_t, kNumIcons>;

// ---- registries (state.CHOICES / AUTOS / ACTIONS) ---------------------------------------

using OptionsFn = void (*)(State& s, int fi, std::vector<Option>& out);
using ApplyFn = void (*)(State& s, int fi, const Option& o, Chance& ch);
using AutoFn = void (*)(State& s, int fi, Chance& ch);
using FeasibleFn = bool (*)(State& s, int p, int strength);
using StartFn = void (*)(State& s, int p, int strength);

struct Registry {
  OptionsFn options[static_cast<int>(FrameKind::kNumFrameKinds)] = {};
  ApplyFn apply[static_cast<int>(FrameKind::kNumFrameKinds)] = {};
  AutoFn autos[static_cast<int>(FrameKind::kNumFrameKinds)] = {};
  FeasibleFn feasible[kNumActions] = {};
  StartFn start[kNumActions] = {};
};
Registry& registry();
void register_choice(FrameKind k, OptionsFn options, ApplyFn apply);
void register_auto(FrameKind k, AutoFn fn);
void register_action(ActionCard a, FeasibleFn feasible, StartFn start);

// Each module registers its frames once; called by registry().
void register_engine_frames();
void register_effect_frames();
void register_cards_action();
void register_build_action();
void register_animals_action();
void register_association_action();
void register_sponsors_action();
void register_relocate_frames();
void register_ability_frames();
void register_sponsor_effect_frames();

// ---- frame stack -----------------------------------------------------------------------

int push(State& s, FrameKind kind, int player);  // returns the frame index
void pop(State& s, int fi);                       // removes frame fi (need not be the top)
inline Frame& frame(State& s, int fi) { return s.stack[fi]; }
int find_frame(const State& s, std::uint16_t uid);  // stack index, or -1 when no longer on the stack

// Frame data layouts (Frame::d offsets) by kind. Lists store their length first.
namespace fd {
constexpr int phase = 0;  // turn_end, extra_end, final_scoring, break
constexpr int extra_end_owner = 1;
constexpr int break_order_n = 1, break_order = 2;
constexpr int extra_actions = 0, extra_optional = 1, extra_owner = 2;  // actions: bit mask
constexpr int tile_space = 0;
constexpr int free_build_n = 0, free_build_kinds = 1;  // pairs (BuildingKind, size)
constexpr int discard_count = 0, discard_limit = 1;
constexpr int cards_draw = 0, cards_discard = 1, cards_may_snap = 2, cards_upgraded = 3, cards_drawn = 4;
constexpr int build_budget = 0, build_upgraded = 1, build_normal_done = 2, build_engineer_used = 3,
              build_built_n = 4, build_built = 5;  // built: pairs (BuildingKind, size)
constexpr int kMaxBuilt = 10;
constexpr int animals_remaining = 0, animals_played = 1, animals_phase = 2, animals_all_small = 3;
constexpr int kAnimalsPlay = 0, kAnimalsBonus = 1, kAnimalsTake = 2;  // animals_phase values
constexpr int assoc_strength = 0, assoc_remaining = 1, assoc_used = 2 /* Task bit mask */, assoc_donated = 3,
              assoc_side2 = 4;
}  // namespace fd

// Project support options (association.py): a[0] project card or display index, a[1] slot
// position, a[2] map token index, a[3] extra kind, a[4] extra card.
constexpr int kExtraNone = 0, kExtraRelease = 1, kExtraToken = 2;
inline Icon project_icon(ProjIcon pi) { return static_cast<Icon>(static_cast<int>(pi) - 1); }
inline bool project_icon_is_plain(ProjIcon pi) {
  return pi >= ProjIcon::kPIcon_bird && pi <= ProjIcon::kPIcon_rock;
}

// ---- animals (actions/animals.py) ----------------------------------------------------

struct Housing {
  int building;
  int spaces;  // 0 for a standard enclosure
};
std::vector<Housing> housings(const Player& pl, CardIdx c);
int animal_cost(const Player& pl, CardIdx c);
void release_animal(State& s, int p, CardIdx c);
HousingEntry* find_housing(Player& pl, CardIdx c);
// The animal of housing entry `h` (in a standard enclosure) leaves it. Per the rules the enclosure
// flipped back to empty is the smallest occupied standard enclosure meeting all of the animal's
// requirements (else its size requirement); the animals recorded there move to the leaver's home.
// The caller then removes or re-targets entry `h`.
void vacate_standard_enclosure(Player& pl, int h);
// abilities.resolve: the abilities of an animal that was just played, in printed order.
void resolve_abilities(State& s, int p, CardIdx c, Chance& ch);
// Removes a card from the zoo_cards list (keeping the order of the others).
void remove_zoo_card(Player& pl, CardIdx c);
void add_zoo_card(Player& pl, CardIdx c);
bool terrain_ok(const Player& pl, const Building& b, const CardDef& c);
BuildingKind special_building(SpecialKind k);

// ---- abilities (abilities*.py) -----------------------------------------------------------

namespace fd {
constexpr int left = 0;  // ab_sun, ab_dig, ab_pouch, ab_snap
constexpr int snap_can_replenish = 1, snap_replenish_between = 2;
constexpr int pick_count = 0, pick_n = 1, pick_cards = 2;  // ab_pick
constexpr int cards_n = 0, cards_list = 1;                   // ab_hunter, ab_resist
constexpr int project_only = 0;                              // ab_project: kNull or a card
constexpr int boost_action = 0;
constexpr int pilfer_thief = 0;
constexpr int candidates_n = 0, candidates = 1, hypnosis_reach = 4;  // ab_pilfer_pick, ab_hypnosis_pick
}  // namespace fd

// abilities_basic._offer_pick: keep `keep` of the drawn cards (kNoCard entries are dropped).
void offer_pick(State& s, int p, const std::vector<CardIdx>& drawn, int keep);
// abilities_basic.hunter: reveal `value` cards; an animal among them may be taken.
void hunter(State& s, int p, int value, Chance& ch);

// ---- sponsors (actions/sponsors.py) ----------------------------------------------------

constexpr int sponsors_strength = 0, sponsors_budget = 1, sponsors_played = 2;
constexpr int sponsor_building_card = 0;
constexpr int move_animals_building = 0;
// Legal positions for the card's unique building; `has_building` is false for cards without one.
std::vector<CellMask> building_spots(const State& s, int p, CardIdx c, bool* has_building);
bool can_play_sponsor(const State& s, int p, CardIdx c);
// Puts a sponsor into play (the caller has paid and removed it from where it came from).
void play_sponsor(State& s, int p, CardIdx c, Chance& ch);

// ---- options -------------------------------------------------------------------------

Option opt(Head h);
Option opt(Head h, int a0);
Option opt(Head h, int a0, int a1);
Option opt(Head h, int a0, int a1, int a2);
Option opt_skip();

// ---- deck and display ------------------------------------------------------------------

CardIdx draw_from_deck(State& s, Chance& ch);  // kNoCard when deck and discard are empty
void to_hand(State& s, int p, CardIdx c, bool is_public);
void remove_from_hand(State& s, int p, CardIdx c);
std::vector<int> cards_in_range(const State& s, int p);
CardIdx take_from_display(State& s, int index);
void refill_display(State& s, Chance& ch);
void advance_break(State& s, int p, int steps);
bool is_upgraded(const State& s, int p, int action);

// ---- action card tokens ------------------------------------------------------------

constexpr int kInteractiveMinAppeal = 5;
constexpr int kVenomCost = 2;
constexpr int kConstrictionPenalty = 2;
bool affected(const State& s, int q);
bool has_token(const State& s, int p, int action, TokenKind t);
bool add_action_token(State& s, int p, int action, TokenKind t);
void remove_action_token(State& s, int p, int action, TokenKind t);
bool has_venom(const State& s, int p);

// ---- icons and gains -------------------------------------------------------------------

IconCounts icons(const Player& pl);
// Announce icons added to player p's zoo; `before` is that zoo's icon count beforehand.
void icons_played(State& s, int p, const IconCounts& added, const IconCounts& before, Chance& ch);
void score(State& s, int p, EffectKind kind, int amount);
void gain(State& s, int p, EffectKind kind, int amount = 1);
inline void gain(State& s, int p, const Effect& e) { gain(s, p, e.kind, e.amount); }
void push_final_card_discards(State& s);
void take_map_token(State& s, int p, int index);
void token_income(State& s, int p);
bool can_take_partner_zoo(const State& s, int p, int continent);
void take_partner_zoo(State& s, int p, int continent, Chance& ch);
bool can_take_university(const State& s, int p, int university);
void take_university(State& s, int p, int university, Chance& ch);
int place_building(State& s, int p, BuildingKind kind, int size, CellMask cells, CardIdx name = kNoCard);

// ---- engine (engine.py) ----------------------------------------------------------------

void offer_extra_action(State& s, int p, unsigned action_mask, int owner = -1, bool optional = true);
void after_finishing(State& s, const Frame& f);
// Strength of owner's action card as it would be used now: its slot, less a Constriction token.
int effective_strength(const State& s, int owner, int action);

// ---- what an action card needs, for the encoder -----------------------------------------

// The lowest strength (X-tokens included) at which player p's own action card does it: Animals
// plays an animal, Sponsors plays sponsor c, Association supports a project.
int animals_strength_needed(const Player& pl);
int sponsor_strength_needed(const State& s, int p, CardIdx c);
int project_strength_needed(const Player& pl);

struct KindSize {
  BuildingKind kind;
  int size;
};
// The buildings player p's own Build card offers that have a legal spot on its map now (money
// and the action's strength aside).
std::vector<KindSize> placeable_buildings(const State& s, int p);

// ---- association facts for the encoder (actions/association.py) -------------------------

// How many of a project's icon (or animals / continents / sizes it counts) the zoo has.
int project_icon_count(const Player& pl, ProjIcon icon);
// Money the next donation costs.
int next_donation_cost(const State& s);
// Per slot position 1..3: whether player p could support the project there right now (slot free,
// not yet supported by p, condition met, a map token left). The strength and the workers of an
// Association action are not part of it. Works for projects in play, in a hand or on the display.
std::array<bool, 3> supportable_levels(const State& s, int p, CardIdx project);

// ---- card tokens (Player::card_tokens; a dict with setdefault semantics in Python) ------

CardTokens* find_card_tokens(Player& pl, CardIdx c);
const CardTokens* find_card_tokens(const Player& pl, CardIdx c);
int card_tokens_get(const Player& pl, CardIdx c, int fallback = 0);
int card_tokens_setdefault(Player& pl, CardIdx c, int value);
void card_tokens_set(Player& pl, CardIdx c, int value);
// Card 227 (Waza Special Assignment) stores the chosen kind in its card_tokens entry.
constexpr int kWazaSmall = 1, kWazaLarge = 2;

// ---- sponsor rules (sponsor_rules.py) and card conditions (cards.unmet_requirements) ------

int unmet_requirements(const Player& pl, CardIdx c);
namespace sr {
bool has(const Player& pl, int printed_id);
int task_value(const Player& pl, int task);
std::vector<CardIdx> breeding_cards(const Player& pl);
void spend_breeding_token(Player& pl, CardIdx c);
bool multi_support_release(const Player& pl);
void release_bonus(State& s, int p);
std::vector<CardIdx> releasable(const Player& pl, CardIdx project, int slot_index);
int waza_kind(const Player& pl);  // 0, kWazaSmall or kWazaLarge
bool animal_allowed(const Player& pl, CardIdx c);
int animal_cost_reduction(const Player& pl, CardIdx c);
bool requirements_ok(const Player& pl, CardIdx c);
int sponsor_level(const State& s, int p, CardIdx c);
void on_animal_played(State& s, int p, CardIdx c);
bool has_small_program(const Player& pl);
bool ignores_terrain(const Player& pl);
bool has_engineer(const Player& pl);
inline bool engineer_kind(BuildingKind k) {
  return k == BuildingKind::kKiosk || k == BuildingKind::kPavilion || k == BuildingKind::kEnclosure;
}
// Reveal deck cards until test(card) holds; that card goes to p's hand (public), the others to
// the discard pile. The discard pile is reshuffled at most once. Returns the card or kNoCard.
CardIdx reveal_until(State& s, int p, bool (*test)(CardIdx), Chance& ch);
}  // namespace sr

// ---- card effect hooks (sponsor_effects.py / final_scoring.py) -------------------------

// ICON_TRIGGERS: card `trigger` in owner's zoo reacts to icons played into p's zoo.
void icon_trigger(State& s, CardIdx trigger, int owner, int p, const IconCounts& added,
                  const std::vector<int>& new_icons, Chance& ch);
bool has_icon_trigger(CardIdx c);
// What icon_trigger reacts to: the icon (-1 for Explorer, which counts icons new to the zoo),
// whether plays into any zoo count, and the gain per icon when it is a plain gain (amount 0 when
// the effect is something else: a free building, a card, an ability).
struct IconTriggerSpec {
  int icon = -1;
  bool any_zoo = false;
  EffectKind kind = EffectKind::kEffMoney;
  int amount = 0;
};
IconTriggerSpec icon_trigger_spec(CardIdx c);
bool has_immediate(CardIdx c);
void immediate_effect(State& s, CardIdx c, int p);
bool has_income(CardIdx c);
void income_effect(State& s, CardIdx c, int p);
bool has_endgame(CardIdx c);
void endgame_effect(State& s, CardIdx c, int p);
void after_build_hooks(State& s, int p, int building);  // AFTER_BUILD_HOOKS, applied in reverse
void after_special_built(State& s, int p, int building);  // relocate.py hook

}  // namespace board_ai::arknova
