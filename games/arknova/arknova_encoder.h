#pragma once

// Variable-length token encoder for the set-style (transformer) network; see TokenLayout
// (variable_tokens, token_positions). Every collection is encoded at its actual size.
//
// "My" = the perspective, who is the player to move whenever the policy is queried. Tokens:
//   global        the table: break track, decks, donations, end of game, association board, the
//                 frames waiting below the pending decision, the composition of the discard pile
//                 and of the regular cards I have not seen (deck and the opponent's hidden cards,
//                 from the belief tracker's known_cards), and which final scoring cards are unseen
//   decision      the pending decision: its frame kind, the typed data of that frame, the turn it
//                 belongs to and the atoms already chosen (prefix)
//   player  x2    tracks, resources, workers, icons, zoo summary (me, then the opponent)
//   action  x10   one per action card of each player: slot, strength as it would be used now,
//                 side, Venom / Constriction / Multiplier tokens on that card
//   card          one per animal / sponsor / final scoring card in my hand, in the opponent's hand
//                 (seen cards with their entity, one entity-0 "unknown" token per unseen card),
//                 among the cards the opponent may have tucked face down, on the display, in either
//                 zoo, in the top frame's public list, in my pending private list and among my
//                 final cards. Carries the card's printed facts and what they mean now for its
//                 holder (the opponent for its hand, maybe-tucked and zoo cards, else me); a
//                 sponsor also what its effects would give (from the rules' effect code).
//   project       the same for conservation project cards, plus the projects in play: who holds
//                 which level, and which levels each player could support now
//   cell    x58   the map both zoos are on, with its axial hex position (relative position bias in
//                 the network): what the map prints there (terrain, placement bonus as an effect,
//                 what the map's special rule makes of the cell), my buildings, the opponent's
//   enclosure     one per standard or special enclosure of each zoo (the buildings that house
//                 animals); mine sit at their lowest cell's position
//   reward  x7    the map's left-edge tokens: what each gives, whether it pays again every break,
//                 who has taken it
// Nothing names the map: everything that differs between maps is in these features, so that
// what is learnt on one map carries over to the others.
// The full list of features is the ARKNOVA_*_FEATURES tables below; layout_json() exports it and
// docs/games/arknova_api.md documents it (tests/arknova checks the two against each other).
//
// Pointer atoms (arknova_actions.h) bind by anchor to the matching "my" token. Head 0 takes,
// plays or picks the object, head 1 gives it up (discard, sell, tuck, release), head 2 snaps a
// display card. Map cells use heads 0..5 for the orientation of the building placed with that
// cell as its lowest; my enclosure tokens use head 0 (house the animal there). Global atoms use
// the global head. Atoms already chosen in the pending decision show as flags on their tokens and
// as one-hots in the decision token.
//
// The encoder reads hand contents only from the masked adapter fields (ArknovaState::hand ...),
// never from the core's copies, and everything else it reads from the core is public.

#include <string>
#include <vector>

#include "../../engine/core/feature_encoder.h"
#include "arknova_actions.h"

namespace board_ai::arknova {

// Feature tables: X(name, width). Offsets follow from the order; a feature occupies
// [name, name + width). All features of a token type start at ArknovaTokenEncoder::kHeader.

// The composition of a pool of regular cards (animals, sponsors, projects); the same block for
// the discard pile and for the unseen cards. Each column is the pool's count (or sum) over the
// count (or sum) among all regular cards of the game.
#define ARKNOVA_POOL_FEATURES(X, pool)                                                             \
  X(pool##_type, 3) X(pool##_animal_size, 5) X(pool##_icons, 15) X(pool##_sponsor_level, 4)        \
  X(pool##_project_category, 4) X(pool##_appeal, 1) X(pool##_conservation, 1)

#define ARKNOVA_GLOBAL_FEATURES(X)                                                                 \
  X(break_pos, 1) X(break_by, 2) X(deck, 1) X(discard, 1) X(final_deck, 1) X(donations, 1)         \
  X(donation_cost, 1) X(end_by, 2) X(final_turns, 1) X(final_card_discarded, 1) X(my_turn, 1)      \
  X(partner_zoo_available, 5) X(university_available, 3) X(bonus_tiles, 18) X(base_pile, 1)        \
  X(stack_depth, 1) X(pending, 42) X(pending_opponent, 1) X(after_clever, 1) X(after_boost, 1)     \
  X(after_extra_action, 1) ARKNOVA_POOL_FEATURES(X, discard) ARKNOVA_POOL_FEATURES(X, unseen)      \
  X(unseen_final, 11) X(trade_used, 1)

// What a map reward gives (a placement bonus, a left-edge token, a side-panel slot): per
// EffectKind its amount, money / 10, an enclosure's size / 5, anything else / 3.
#define ARKNOVA_EFFECT_WIDTH 16

#define ARKNOVA_DECISION_FEATURES(X)                                                               \
  X(frame, 42) X(to_move, 1) X(prefix_len, 1) X(turn_action, 5) X(turn_strength, 1)                \
  X(turn_xtoken, 1) X(turn_depth, 1) X(turn_borrowed, 1) X(budget, 1) X(remaining, 1)              \
  X(done_count, 1) X(side2, 1) X(discard_after, 1) X(may_snap, 1) X(normal_done, 1)                \
  X(engineer_used, 1) X(animals_phase, 3) X(all_small, 1) X(tasks_used, 4) X(donated, 1)           \
  X(optional, 1) X(extra_actions, 5) X(extra_from_opponent, 1) X(boost_action, 5)                  \
  X(tile_space, 2) X(can_replenish, 1) X(replenish_between, 1) X(only_one, 1) X(built, 11)         \
  X(offered, 11) X(list_len, 1) X(chosen_verb, 3) X(chosen_type, 11) X(chosen_position, 3)         \
  X(chosen_map_token, 7) X(chosen_harbor, 1)

#define ARKNOVA_PLAYER_FEATURES(X)                                                                 \
  X(me, 1) X(opponent, 1) X(to_move, 1) X(appeal, 1) X(money, 1) X(conservation, 1)                \
  X(reputation, 1) X(xtokens, 1) X(workers, 1) X(active_workers, 1) X(workers_out, 4)              \
  X(hand_size, 1) X(hand_limit, 1) X(zoo_cards, 1) X(tucked, 1) X(final_cards, 1)                  \
  X(partner_zoos, 5) X(universities, 3) X(map_tokens, 7) X(icons, 15) X(victory_points, 1)         \
  X(poisoned, 1) X(covered_cells, 1) X(buildings, 7) X(kiosk_income, 1) X(appeal_income, 1)       \
  X(display_range, 1) X(placeable, 11) X(next_partner_zoo, ARKNOVA_EFFECT_WIDTH)                    \
  X(next_university, ARKNOVA_EFFECT_WIDTH) X(last_worker, ARKNOVA_EFFECT_WIDTH) X(map_income, 1)    \
  X(can_trade, 1) X(ignore_conditions, 1) X(sponsor_discount, 1) X(group_left, 1)

#define ARKNOVA_ACTION_CARD_FEATURES(X)                                                            \
  X(mine, 1) X(opponent, 1) X(action, 5) X(slot, 5) X(strength, 1) X(effective_strength, 1)        \
  X(upgraded, 1) X(venom, 1) X(constriction, 1) X(multiplier, 1) X(venom_cost, 1) X(in_use, 1)

// Where a card is; shared by card and project tokens (same offsets). Only a card token can be
// unknown, in a zoo, a final card or a sponsor placing its building; only a project can be in play.
#define ARKNOVA_LOCATION_FEATURES(X)                                                               \
  X(my_hand, 1) X(opp_hand, 1) X(display, 1) X(public_list, 1) X(private_list, 1)                  \
  X(opp_maybe_tucked, 1) X(chosen, 1) X(chosen_discard, 1) X(display_pos, 1) X(in_range, 1)

#define ARKNOVA_CARD_FEATURES(X)                                                                   \
  ARKNOVA_LOCATION_FEATURES(X)                                                                     \
  X(unknown, 1) X(my_zoo, 1) X(opp_zoo, 1) X(my_final, 1) X(pending, 1)                            \
  X(type, 3) X(cost, 1) X(money_now, 1) X(affordable, 1) X(size, 1) X(appeal, 1)                   \
  X(conservation, 1) X(reputation, 1) X(level, 1) X(rock, 1) X(water, 1) X(no_standard, 1)         \
  X(special, 6) X(tags, 15) X(req, 20) X(unmet, 1) X(req_ok, 1) X(has_place, 1) X(ability, 44)     \
  X(effect, 5) X(building, 6) X(scoring, 8) X(yield, 2) X(tokens, 1) X(immediate, 4)               \
  X(income, 3) X(trigger_icon, 15) X(trigger_any, 1) X(trigger_gain, 3) X(strength_short, 1)

#define ARKNOVA_PROJECT_FEATURES(X)                                                                \
  ARKNOVA_LOCATION_FEATURES(X)                                                                     \
  X(in_play, 1) X(category, 4) X(icon, 19) X(level_required, 3) X(level_size_min, 3) X(level_size_max, 3)        \
  X(level_conservation, 3) X(level_reputation, 3) X(play_bonus, 4) X(occupied_free, 3)             \
  X(occupied_me, 3) X(occupied_opp, 3) X(occupied_blocked, 3) X(can_me, 3) X(can_opp, 3)           \
  X(count_me, 1) X(count_opp, 1) X(affordable, 1) X(strength_short, 1)

// A cell's map rule features say what covering it does under the map's special rule, in terms of
// the effect rather than of the map: the rule cells themselves are landmarks.
#define ARKNOVA_CELL_FEATURES(X)                                                                   \
  X(terrain, 3) X(build2_only, 1) X(border, 1) X(bonus, ARKNOVA_EFFECT_WIDTH) X(covered, 1)        \
  X(adjacent, 1) X(buildable, 1) X(kind, 7) X(anchor, 1) X(anchor_dq, 1) X(anchor_dr, 1)           \
  X(rock_neighbors, 1) X(water_neighbors, 1) X(kiosk_income, 1) X(kiosks_adjacent, 1)              \
  X(landmark, 1) X(fill_appeal, 1) X(extra_capacity, 1) X(break_money, 1) X(unlocks_trade, 1)      \
  X(unlocks_condition, 1) X(cover_draw_sponsor, 1) X(group, 1) X(group_kiosk_income, 1)            \
  X(group_sponsor_discount, 1) X(opp_kind, 7) X(opp_adjacent, 1)

#define ARKNOVA_ENCLOSURE_FEATURES(X)                                                              \
  X(mine, 1) X(opponent, 1) X(kind, 7) X(size, 1) X(capacity, 1) X(occupied, 1) X(used, 1)         \
  X(free, 1) X(rock, 1) X(water, 1) X(kiosks_adjacent, 1) X(animals, 1) X(target, 1)

// One per left-edge token of the map (taken when supporting a project; map_token:<index> atoms).
#define ARKNOVA_REWARD_FEATURES(X)                                                                 \
  X(index, 7) X(effect, ARKNOVA_EFFECT_WIDTH) X(repeats, 1) X(taken_me, 1) X(taken_opp, 1)         \
  X(chosen, 1)

// Offsets (relative to the first feature) as enumerators: gf::break_pos, df::frame, ...
#define ARKNOVA_FEATURE_ENUM(name, width) name, name##_last = name + (width)-1,
namespace gf { enum : int { ARKNOVA_GLOBAL_FEATURES(ARKNOVA_FEATURE_ENUM) kCount }; }
namespace df { enum : int { ARKNOVA_DECISION_FEATURES(ARKNOVA_FEATURE_ENUM) kCount }; }
namespace pf { enum : int { ARKNOVA_PLAYER_FEATURES(ARKNOVA_FEATURE_ENUM) kCount }; }
namespace af { enum : int { ARKNOVA_ACTION_CARD_FEATURES(ARKNOVA_FEATURE_ENUM) kCount }; }
namespace cf { enum : int { ARKNOVA_CARD_FEATURES(ARKNOVA_FEATURE_ENUM) kCount }; }
namespace jf { enum : int { ARKNOVA_PROJECT_FEATURES(ARKNOVA_FEATURE_ENUM) kCount }; }
namespace xf { enum : int { ARKNOVA_CELL_FEATURES(ARKNOVA_FEATURE_ENUM) kCount }; }
namespace ef { enum : int { ARKNOVA_ENCLOSURE_FEATURES(ARKNOVA_FEATURE_ENUM) kCount }; }
namespace rf { enum : int { ARKNOVA_REWARD_FEATURES(ARKNOVA_FEATURE_ENUM) kCount }; }
#undef ARKNOVA_FEATURE_ENUM

class ArknovaTokenEncoder final : public IFeatureEncoder {
 public:
  enum TokenType {
    kGlobalToken, kDecisionToken, kPlayerToken, kActionCardToken, kCardToken, kProjectToken, kCellToken,
    kEnclosureToken, kRewardToken, kTokenTypeCount
  };

  // Token header: valid, entity id, anchor id, token type, has position, q, r.
  static constexpr int kHeader = 7;
  static constexpr int kFeatures = cf::kCount;  // the widest token type
  static constexpr int kTokenDim = kHeader + kFeatures;

  // Entity ids: 0 = none.
  static constexpr int kEntityCard = 1;                           // + CardIdx
  static constexpr int kEntityCell = kEntityCard + kCardSlots;    // + map cell
  static constexpr int kEntityAction = kEntityCell + kMapCells;   // + ActionCard
  static constexpr int kNumEntities = kEntityAction + kNumActions;

  // Pointer heads.
  static constexpr int kHeadTake = 0;     // play / take / pick; an enclosure: house the animal there
  static constexpr int kHeadDiscard = 1;  // discard / sell / tuck / release
  static constexpr int kHeadSnap = 2;     // display card: snap it
  static constexpr int kNumHeads = atom::kCellHeads;  // a cell: head = orientation of the placement

  // Anchor ids are 1 + index; the bases below are indices.
  static constexpr int kAnchorHand = 0;
  static constexpr int kAnchorZoo = kAnchorHand + kCardSlots;
  static constexpr int kAnchorDisplay = kAnchorZoo + kCardSlots;
  static constexpr int kAnchorProject = kAnchorDisplay + kDisplaySize;
  static constexpr int kAnchorPublic = kAnchorProject + atom::kProjectSlots;
  static constexpr int kAnchorPrivate = kAnchorPublic + atom::kPublicSlots;
  static constexpr int kAnchorFinal = kAnchorPrivate + kPrivateCards;
  static constexpr int kAnchorBuilding = kAnchorFinal + kMaxFinalCards;
  static constexpr int kAnchorCell = kAnchorBuilding + kMaxBuildings;
  static constexpr int kNumAnchors = kAnchorCell + kMapCells;

  ArknovaTokenEncoder();

  int action_space() const override { return atom::kNumAtoms; }
  int feature_dim() const override { return 0; }
  const TokenLayout* token_layout() const override { return &layout_; }

  void encode_features(const IGameState& masked_state, int perspective_player, const IBeliefTracker* tracker,
                       std::vector<float>* out) const override;

  // The layout as JSON: token header, per token type its features (name, offset, width), entity
  // and anchor ranges. For tests and documentation.
  static std::string layout_json();

 private:
  TokenLayout layout_;
};

}  // namespace board_ai::arknova
