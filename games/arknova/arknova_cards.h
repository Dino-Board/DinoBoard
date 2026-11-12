#pragma once

// Ark Nova card data and the fixed numbers of the base game (2 players, map 0).
// The numbers mirror ark-nova-ai's arknova/tables.py; card facts come from the generated
// arknova_cards_data.inc (games/arknova/tools/gen_cards.py).

#include <array>
#include <cstdint>

namespace board_ai::arknova {

constexpr int kPlayers = 2;

// ---- icons, continents, actions ---------------------------------------------

enum Icon : std::uint8_t {
  kIcon_bird, kIcon_herbivore, kIcon_predator, kIcon_primate, kIcon_reptile, kIcon_bear, kIcon_pet,
  kIcon_africa, kIcon_americas, kIcon_asia, kIcon_australia, kIcon_europe,
  kIcon_science, kIcon_water, kIcon_rock,
  kNumIcons
};
constexpr int kNumAnimalCategories = 7;  // kIcon_bird .. kIcon_pet
constexpr int kContinentIcon0 = kIcon_africa;

enum Continent : std::uint8_t { kCont_africa, kCont_americas, kCont_asia, kCont_australia, kCont_europe, kNumContinents };
constexpr const char* kContinentNames[] = {"africa", "americas", "asia", "australia", "europe"};
constexpr const char* kIconNames[] = {"bird", "herbivore", "predator", "primate", "reptile", "bear", "pet",
                                      "africa", "americas", "asia", "australia", "europe",
                                      "science", "water", "rock"};

enum ActionCard : std::uint8_t { kAnimals, kBuild, kCards, kAssociation, kSponsors, kNumActions };
constexpr const char* kActionNames[] = {"animals", "build", "cards", "association", "sponsors"};

enum University : std::uint8_t { kUniHand, kUniScience2, kUniScienceRep, kNumUniversities };
constexpr const char* kUniversityNames[] = {"hand", "science2", "science_rep"};

// Requirement slots beyond the icons (CardDef::req is indexed by Icon, then these).
enum ReqExtra : std::uint8_t {
  kReqPartnerZoo = kNumIcons, kReqAnimalsII, kReqSponsorsII, kReqAppeal, kReqReputation, kNumReqs
};

// ---- card definitions -------------------------------------------------------

enum class CardType : std::uint8_t { kEndgame, kProject, kSponsor, kAnimal };
enum class SpecialKind : std::uint8_t { kNone, kReptileHouse, kAviary, kPettingZoo };
enum class ProjCategory : std::uint8_t { kNone, kProjBase, kProjRelease, kProjBreed, kProjNormal };
enum class ProjIcon : std::uint8_t {
  kNone,
  kPIcon_bird, kPIcon_herbivore, kPIcon_predator, kPIcon_primate, kPIcon_reptile, kPIcon_bear, kPIcon_pet,
  kPIcon_africa, kPIcon_americas, kPIcon_asia, kPIcon_australia, kPIcon_europe,
  kPIcon_science, kPIcon_water, kPIcon_rock,
  kPIconAllAnimals, kPIconAllContinents, kPIconAnimalSize2, kPIconAnimalSize4,
};
enum class Reward : std::uint8_t { kRewardConservation, kRewardReputation };
enum class RewardReq : std::uint8_t {
  kRReqNone, kRReqRelease, kRReqAllAnimals, kRReqWater, kRReqRock, kRReqScience, kRReqAnimalSize2,
  kRReq_bird, kRReq_herbivore, kRReq_predator, kRReq_primate, kRReq_reptile, kRReq_bear, kRReq_pet,
  kRReq_africa, kRReq_americas, kRReq_asia, kRReq_australia, kRReq_europe,
};

enum class Keyword : std::uint8_t {
  kNone,
  kACTION_ASSOCIATION, kACTION_BUILDING, kACTION_CARDS, kACTION_SPONSORS, kASSERTION,
  kBOOST_ANIMAL, kBOOST_ASSOCIATION, kBOOST_BUILDING, kBOOST_CARD, kBOOST_SPONSORS,
  kCLEVER, kCONSTRICTION, kDETERMINATION, kDIGGING, kDOMINANCE, kFLOCK_ANIMAL,
  kFULL_THROATED, kHUNTER, kHYPNOSIS, kICONIC_ANIMAL, kINVENTIVE, kINVENTIVE_BEAR,
  kINVENTIVE_PRIMARY, kJUMPING, kMULTIPLIER_ASSOCIATION, kMULTIPLIER_BUILDING,
  kMULTIPLIER_CARD, kMULTIPLIER_SPONSORS, kPACK, kPEACOCKING, kPERCEPTION_4,
  kPETTING_ZOO_ANIMAL, kPILFERING_1, kPILFERING_2, kPOSTURING, kPOUCH, kRESISTANCE,
  kSCAVENGING, kSNAPPING_1, kSNAPPING_2, kSPONSOR_MAGNET, kSPRINT, kSUN_BATHING, kVENOM,
};

// Effect kinds that `gain` understands (effects.py gain); also used for bonus tiles,
// placement bonuses and map side-panel slots.
enum class EffectKind : std::uint8_t {
  kEffMoney, kEffAppeal, kEffXtoken, kEffConservation, kEffReputation, kEffWorker, kEffUpgrade,
  kEffCard, kEffClever, kEffEnclosure, kEffKiosk, kEffMultiplier, kEffPartnerZoo, kEffUniversity,
  kEffSponsorForMoney, kEffSpecialEnclosure, kNumEffectKinds
};
constexpr const char* kEffectKindNames[] = {
    "money", "appeal", "xtoken", "conservation", "reputation", "worker", "upgrade", "card", "clever", "enclosure",
    "kiosk", "multiplier", "partner_zoo", "university", "sponsor_for_money", "special_enclosure"};
struct Effect {
  EffectKind kind = EffectKind::kEffMoney;
  std::int8_t amount = 0;
};

constexpr int kMaxShape = 6;

struct SpecialSlot {
  SpecialKind kind = SpecialKind::kNone;
  std::int8_t size = 0;
};
struct Ability {
  Keyword keyword = Keyword::kNone;
  std::int8_t value = -1;  // -1 = none; ICONIC_ANIMAL stores a Continent
};
struct BuildingSpec {
  bool present = false;
  std::int8_t n = 0;
  std::int8_t shape[kMaxShape][2] = {};
  std::int8_t size = 0, rock = 0, water = 0, min_border = 0;
  bool needs_adjacency = false;
};
struct ProjectReward {
  Reward reward = Reward::kRewardConservation;
  std::int8_t amount = 0;
  RewardReq requirement = RewardReq::kRReqNone;
};
struct ProjectSlot {
  std::int8_t required = -1;  // -1 = none
  std::int8_t size_min = -1, size_max = -1;
  std::int8_t n_rewards = 0;
  ProjectReward rewards[2];
};
struct ScoringStep {
  std::int8_t threshold = 0, conservation = 0;
};

struct CardDef {
  std::int16_t id = 0;  // printed number: 1..11 endgame, 101.. project, 201.. sponsor, 401.. animal
  CardType type = CardType::kAnimal;
  const char* name = "";
  std::int8_t cost = 0, size = 0, appeal = 0, conservation = 0, reputation = 0, level = 0;
  bool human = false;
  std::int8_t enc_standard = -1;  // -1 = no standard enclosure
  std::int8_t enc_rock = 0, enc_water = 0;
  std::int8_t n_special = 0;
  SpecialSlot special[2];
  std::array<std::int8_t, kNumIcons> tags{};
  std::uint8_t continents = 0;  // bit per Continent
  std::array<std::int8_t, kNumContinents> continent_icons{};  // printed continent icons, with repeats
  std::array<std::int8_t, kNumReqs> req{};
  std::int8_t min_reputation = -1, max_appeal = -1;
  std::int8_t n_abilities = 0;
  Ability abilities[2];
  bool lasting_effect = false;  // a sponsor with income / passive / endgame effects
  BuildingSpec building;
  ProjCategory proj_category = ProjCategory::kNone;
  ProjIcon proj_icon = ProjIcon::kNone;
  std::int8_t n_slots = 0;
  ProjectSlot slots[3];
  std::int8_t n_play_bonus = 0;
  Effect play_bonus[2];
  std::int8_t n_scoring = 0;
  ScoringStep scoring[4];
};

// Cards are addressed by CardIdx: the position in printed-id order (0 .. kNumCards-1).
using CardIdx = std::int16_t;
constexpr CardIdx kNoCard = -1;

int num_cards();
const CardDef& card(CardIdx c);
// CardIdx of a printed id; throws on an unknown id.
CardIdx card_by_id(int printed_id);
inline bool is_animal(CardIdx c) { return card(c).type == CardType::kAnimal; }
inline bool is_sponsor(CardIdx c) { return card(c).type == CardType::kSponsor; }
inline bool is_project(CardIdx c) { return card(c).type == CardType::kProject; }
inline bool is_small(CardIdx c) { return is_animal(c) && card(c).size <= 2; }
inline bool is_large(CardIdx c) { return is_animal(c) && card(c).size >= 4; }
bool has_keyword(CardIdx c, Keyword k);
// Icons one card adds to a zoo: tags plus rock/water requirement icons (cards.icons_of).
const std::array<std::int8_t, kNumIcons>& card_icons(CardIdx c);
const char* keyword_name(Keyword k);

// ---- fixed numbers (tables.py, 2 players) -----------------------------------

constexpr int kStartMoney = 25;
constexpr int kMaxXtokens = 5;
constexpr int kHandLimit = 3;
constexpr int kHandLimitUniversity = 5;
constexpr int kBuildCostPerSpace = 2;
constexpr int kPartnerZooDiscount = 3;
constexpr int kFullCoverAppeal = 7;
constexpr int kDisplaySize = 6;
constexpr int kMaxWorkers = 4;
constexpr int kMaxPartnerZoos = 4;
constexpr int kBreakTrack = 9;
constexpr int kProjectsAboveBoard = 2;
constexpr int kBaseProjects = 3;
constexpr int kAnimalsIIReputationStrength = 5;

struct CardsRow {
  std::int8_t draw, discard;
  bool may_snap;
};
constexpr CardsRow kCardsTable[2][5] = {
    {{1, 1, false}, {1, 0, false}, {2, 1, false}, {2, 0, false}, {3, 1, true}},
    {{1, 0, false}, {2, 1, false}, {2, 0, true}, {3, 1, true}, {4, 1, true}},
};
constexpr std::int8_t kAnimalsTable[2][5] = {{0, 1, 1, 1, 2}, {1, 1, 2, 2, 2}};

enum Task : std::uint8_t { kTaskReputation, kTaskPartnerZoo, kTaskUniversity, kTaskProject, kNumTasks };
constexpr const char* kTaskNames[] = {"reputation", "partner_zoo", "university", "project"};
constexpr std::int8_t kTaskValue[kNumTasks] = {2, 3, 4, 5};
constexpr int kTaskReputationGain = 2;
constexpr std::int8_t kDonationCosts[] = {2, 5, 7, 10};
constexpr int kNumDonationCosts = 4;
constexpr int kDonationFinalCost = 12;

int appeal_income(int appeal);
int target_appeal(int conservation);
inline int victory_points(int appeal, int conservation) { return appeal - target_appeal(conservation); }

constexpr int kReputationMax = 15;
constexpr int kReputationCapWithoutCardsII = 9;
// Reputation track bonus at each space (kEffMoney with amount 0 = none).
Effect reputation_bonus(int space, bool* present);
int reputation_folders(int reputation);
int folder_reputation(int folder);

constexpr int kConservationUpgradeOrWorker = 2;
constexpr int kConservationTileSpaces[] = {5, 8};
constexpr int kConservationTileMoney = 5;
constexpr int kConservationFinalCardDiscard = 10;

enum BonusTile : std::uint8_t {
  kTileMoney10, kTileXtoken3, kTileReputation2, kTileUniversity, kTilePartnerZoo, kTileEnclosure3,
  kTileCards3, kTileMultiplier, kTileSponsor, kNumBonusTiles
};
constexpr const char* kBonusTileNames[] = {"money10", "xtoken3", "reputation2", "university", "partner_zoo",
                                           "enclosure3", "cards3", "multiplier", "sponsor"};
Effect bonus_tile_effect(BonusTile t);

// Placement bonuses of all maps (tables.PLACEMENT_BONUS, then maps.EXTRA_PLACEMENT_BONUS).
enum PlacementBonus : std::uint8_t {
  kPBNone, kPBMoney5, kPBMoney10, kPBXtoken, kPBRep2, kPBCard, kPBClever, kPBMoney2, kPBRep1, kPBKiosk,
  kPBWorker1, kPBPartnerZoo, kPBUniversity, kPBMultiplier, kPBSponsorForMoney, kNumPlacementBonuses
};
constexpr const char* kPlacementBonusNames[] = {"", "money5", "money10", "xtoken", "rep2", "card", "clever",
                                                "money2", "rep1", "kiosk", "worker1", "partner_zoo",
                                                "university", "multiplier", "sponsor_for_money"};
Effect placement_bonus_effect(PlacementBonus b);

constexpr int kNumMapTokens = 7;

}  // namespace board_ai::arknova
