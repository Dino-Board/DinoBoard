#pragma once

// Factored action space of the Ark Nova adapter.
//
// Every core Option becomes a short sequence of atoms (engine/core/factored_action.h). Global
// atoms name a choice directly (an action card and X-token count, a bonus tile, a building type);
// pointer atoms name an object the encoder exposes as an anchor token: a card in the mover's
// hand (by card id), a display position, a project in play, a card in the mover's zoo (by card
// id), an entry of the public card list on the top frame (Hunter reveal, base project deck), an
// entry of the mover's private card list (Perception / Resistance draw), one of the mover's final
// cards, one of the mover's enclosures (standard or special), or a cell of the mover's map.
//
// One atom has one meaning. In particular an atom never stands for gaining (playing, taking,
// keeping) something in one decision and for giving it up (discarding, selling, tucking, paying)
// in another: cards have separate "take" and "discard" atoms, and money has kMoney (take 5 from
// a conservation space) and kGiveMoney (pay the Pilfering thief).
//
// Option -> atoms:
//   setup keep                   the 4 discarded hand cards as hand_discard atoms, in any order
//   play / play_display animal   [hand | display, enclosure]
//   support / release / play_*   [project | hand | display, position, map token,
//                                 no_extra | zoo_give (released animal or breeding card)]
//   build / engineer / place     [building type, cell(lowest covered cell, orientation)]
//   pick / resist keep           private slot      take (Hunter, base project)  public slot
//   discard_final                final slot        move                         zoo card
//   play (sponsor)               hand              display / take_small / take (Snapping)  display
//   sell / tuck / discard / discard_hand           hand_discard
//   discard_display              display_discard   snap                         display_snap
// Each remaining head is one global atom. kHiddenHandCard is never legal: it is what the
// opponent observes for a Pouch tuck (ArknovaRules::public_action).

#include <string>
#include <vector>

#include "../../engine/core/factored_action.h"
#include "arknova_state.h"

namespace board_ai::arknova {

namespace atom {

enum Simple : int {
  kSkip, kDone, kNo, kYes, kMoney, kDeck, kDonate, kReputation, kRepeat, kWorker, kReplenish, kGiveCard,
  kBreak, kNoExtra, kHiddenHandCard, kGiveMoney, kNumSimple
};

constexpr int kMaxX = kMaxXtokens + 1;
constexpr int kBuildingTypes = 11;  // enclosure 1..5, kiosk, pavilion, petting zoo, reptile house, aviary, unique
constexpr int kBuildVerbs = 3;      // build, engineer, place
constexpr int kTargets = kPlayers;  // relative to the mover
constexpr int kProjectSlots = kBaseProjects + kMaxProjectsInPlay + 2;
// A frame's card list starts at d[1] (fd::cards_list), so no public list is longer than this.
constexpr int kPublicSlots = kFrameData - 1;
constexpr int kCellHeads = kMaxOrientations;  // the placement's orientation

constexpr int kSimple = 0;
constexpr int kAct = kSimple + kNumSimple;
constexpr int kXtoken = kAct + kNumActions * kMaxX;
constexpr int kUpgrade = kXtoken + kNumActions;
constexpr int kClever = kUpgrade + kNumActions;
constexpr int kMultiplier = kClever + kNumActions;
constexpr int kBoost = kMultiplier + kNumActions;  // boost to slot 1, slot 5
constexpr int kPartnerZoo = kBoost + 2;
constexpr int kUniversity = kPartnerZoo + kNumContinents;
constexpr int kTile = kUniversity + kNumUniversities;
constexpr int kWaza = kTile + kNumBonusTiles;  // small, large
constexpr int kTarget = kWaza + 2;
constexpr int kBuildingType = kTarget + kTargets;
constexpr int kPosition = kBuildingType + kBuildVerbs * kBuildingTypes;
constexpr int kMapToken = kPosition + 3;
constexpr int kNumGlobal = kMapToken + kNumMapTokens;
// Pointer atoms. Cards have a "take" atom (play it, take it, keep it) and a "discard" atom (give
// it up); the two are different pointer heads of the same token.
constexpr int kHand = kNumGlobal;                        // play a hand card
constexpr int kHandDiscard = kHand + kCardSlots;         // discard / sell / tuck a hand card
constexpr int kDisplayTake = kHandDiscard + kCardSlots;  // take or play a display card
constexpr int kDisplayDiscard = kDisplayTake + kDisplaySize;  // discard a display card (Digging)
constexpr int kDisplaySnap = kDisplayDiscard + kDisplaySize;  // Cards action: snap
constexpr int kProject = kDisplaySnap + kDisplaySize;
constexpr int kZoo = kProject + kProjectSlots;  // move an animal into a new special enclosure
constexpr int kZooGive = kZoo + kCardSlots;     // release an animal / spend a breeding token
constexpr int kPublic = kZooGive + kCardSlots;
constexpr int kPrivate = kPublic + kPublicSlots;
constexpr int kFinal = kPrivate + kPrivateCards;  // discard a final scoring card
constexpr int kBuilding = kFinal + kMaxFinalCards;  // house the animal in this enclosure (building id)
constexpr int kCell = kBuilding + kMaxBuildings;
constexpr int kNumAtoms = kCell + kMapCells * kCellHeads;

constexpr int cell(int c, int orientation) { return kCell + c * kCellHeads + orientation; }

}  // namespace atom

// Index of a building in the building type atoms (0 .. atom::kBuildingTypes - 1).
int building_type(BuildingKind kind, int size);

// The pending decision of the player to move, as (option, atom sequence) pairs. An option has
// one sequence, except a setup keep: its 4 discards may be named in any order, so it is listed
// once per possible next discard (the atoms already chosen, the next one, then the rest).
struct Decision {
  std::vector<Option> options;
  std::vector<AtomSequence> atoms;
};
Decision current_decision(const ArknovaState& st);

// Whether some option of the pending decision starts with the prefix. A world sampled for a
// player other than the mover must satisfy this: the mover's chosen atoms may point at its
// hidden cards.
bool prefix_consistent(const ArknovaState& st);

// The atoms chosen for a setup keep are a set: keeps them sorted, so that the same discards
// chosen in a different order are the same state.
void canonicalize_prefix(ArknovaState& st);

std::string atom_name(ActionId a);

}  // namespace board_ai::arknova
