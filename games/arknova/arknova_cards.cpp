#include "arknova_cards.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace board_ai::arknova {
namespace {

#include "arknova_cards_data.inc"

struct CardTable {
  std::vector<CardDef> cards;
  std::vector<std::array<std::int8_t, kNumIcons>> icons;
  std::vector<CardIdx> by_id;  // printed id -> CardIdx

  CardTable() : cards(kNumCards) {
    fill_cards(cards.data());
    icons.resize(kNumCards);
    for (int i = 0; i < kNumCards; ++i) {
      const CardDef& c = cards[i];
      icons[i] = c.tags;
      if (c.enc_rock && !c.tags[kIcon_rock]) icons[i][kIcon_rock] += c.enc_rock;
      if (c.enc_water && !c.tags[kIcon_water]) icons[i][kIcon_water] += c.enc_water;
      if (c.id >= static_cast<int>(by_id.size())) by_id.resize(c.id + 1, kNoCard);
      by_id[c.id] = static_cast<CardIdx>(i);
    }
  }
};

const CardTable& table() {
  static const CardTable t;
  return t;
}

}  // namespace

int num_cards() { return kNumCards; }

const CardDef& card(CardIdx c) {
  if (c < 0 || c >= kNumCards) throw std::out_of_range("arknova: bad card index " + std::to_string(c));
  return table().cards[c];
}

CardIdx card_by_id(int printed_id) {
  const auto& by_id = table().by_id;
  if (printed_id < 0 || printed_id >= static_cast<int>(by_id.size()) || by_id[printed_id] == kNoCard) {
    throw std::out_of_range("arknova: unknown card id " + std::to_string(printed_id));
  }
  return by_id[printed_id];
}

bool has_keyword(CardIdx c, Keyword k) {
  const CardDef& d = card(c);
  for (int i = 0; i < d.n_abilities; ++i) {
    if (d.abilities[i].keyword == k) return true;
  }
  return false;
}

const std::array<std::int8_t, kNumIcons>& card_icons(CardIdx c) {
  card(c);
  return table().icons[c];
}

const char* keyword_name(Keyword k) {
  if (k == Keyword::kNone) return "";
  return kKeywordNames[static_cast<int>(k) - 1];
}

int appeal_income(int appeal) {
  static constexpr int kThresholds[][2] = {
      {0, 5},    {1, 6},    {2, 7},    {3, 8},    {4, 9},    {5, 10},   {7, 11},   {9, 12},   {11, 13},
      {13, 14},  {15, 15},  {17, 16},  {20, 17},  {23, 18},  {26, 19},  {29, 20},  {32, 21},  {36, 22},
      {40, 23},  {44, 24},  {48, 25},  {52, 26},  {56, 27},  {61, 28},  {66, 29},  {71, 30},  {76, 31},
      {81, 32},  {86, 33},  {91, 34},  {96, 35},  {102, 36}, {108, 37},
  };
  int income = kThresholds[0][1];
  for (const auto& t : kThresholds) {
    if (appeal >= t[0]) income = t[1];
  }
  return income;
}

int target_appeal(int conservation) {
  int c = conservation < 0 ? 0 : conservation;
  int label = c <= 10 ? 2 * c - 14 : 6 + 3 * (c - 10);
  return 100 - label;
}

Effect reputation_bonus(int space, bool* present) {
  *present = true;
  switch (space) {
    case 5: return {EffectKind::kEffUpgrade, 1};
    case 8: return {EffectKind::kEffWorker, 1};
    case 10: case 13: return {EffectKind::kEffCard, 1};
    case 11: case 14: return {EffectKind::kEffConservation, 1};
    case 12: case 15: return {EffectKind::kEffXtoken, 1};
    default: *present = false; return {};
  }
}

int reputation_folders(int reputation) {
  static constexpr int kTops[] = {1, 3, 6, 9, 12, 15};
  for (int i = 0; i < 6; ++i) {
    if (reputation <= kTops[i]) return i + 1;
  }
  return 6;
}

int folder_reputation(int folder) {
  static constexpr int kTops[] = {1, 3, 6, 9, 12};
  return folder == 1 ? 0 : kTops[folder - 2] + 1;
}

Effect bonus_tile_effect(BonusTile t) {
  switch (t) {
    case kTileMoney10: return {EffectKind::kEffMoney, 10};
    case kTileXtoken3: return {EffectKind::kEffXtoken, 3};
    case kTileReputation2: return {EffectKind::kEffReputation, 2};
    case kTileUniversity: return {EffectKind::kEffUniversity, 1};
    case kTilePartnerZoo: return {EffectKind::kEffPartnerZoo, 1};
    case kTileEnclosure3: return {EffectKind::kEffEnclosure, 3};
    case kTileCards3: return {EffectKind::kEffCard, 3};
    case kTileMultiplier: return {EffectKind::kEffMultiplier, 1};
    case kTileSponsor: return {EffectKind::kEffSponsorForMoney, 1};
    default: throw std::invalid_argument("arknova: bad bonus tile");
  }
}

Effect placement_bonus_effect(PlacementBonus b) {
  switch (b) {
    case kPBMoney5: return {EffectKind::kEffMoney, 5};
    case kPBMoney10: return {EffectKind::kEffMoney, 10};
    case kPBXtoken: return {EffectKind::kEffXtoken, 1};
    case kPBRep2: return {EffectKind::kEffReputation, 2};
    case kPBCard: return {EffectKind::kEffCard, 1};
    case kPBClever: return {EffectKind::kEffClever, 1};
    case kPBMoney2: return {EffectKind::kEffMoney, 2};
    case kPBRep1: return {EffectKind::kEffReputation, 1};
    case kPBKiosk: return {EffectKind::kEffKiosk, 1};
    case kPBWorker1: return {EffectKind::kEffWorker, 1};
    case kPBPartnerZoo: return {EffectKind::kEffPartnerZoo, 1};
    case kPBUniversity: return {EffectKind::kEffUniversity, 1};
    case kPBMultiplier: return {EffectKind::kEffMultiplier, 1};
    case kPBSponsorForMoney: return {EffectKind::kEffSponsorForMoney, 1};
    default: throw std::invalid_argument("arknova: bad placement bonus");
  }
}

}  // namespace board_ai::arknova
