#include "arknova_encoder.h"

#include <algorithm>
#include <array>
#include <optional>
#include <stdexcept>
#include <string>

#include "arknova_belief.h"
#include "arknova_internal.h"

namespace board_ai::arknova {

namespace {

using Enc = ArknovaTokenEncoder;

constexpr int kNumFrameKinds = static_cast<int>(FrameKind::kNumFrameKinds);
constexpr int kNumKeywords = static_cast<int>(Keyword::kVENOM);  // Keyword 1 .. kVENOM
constexpr int kNumProjIcons = static_cast<int>(ProjIcon::kPIconAnimalSize4);
constexpr int kNumProjCategories = static_cast<int>(ProjCategory::kProjNormal);
constexpr int kNumBuildingKinds = static_cast<int>(BuildingKind::kNumKinds);

// The feature tables' widths against the enums they hold.
static_assert(gf::pending_last - gf::pending + 1 == kNumFrameKinds);
static_assert(gf::partner_zoo_available_last - gf::partner_zoo_available + 1 == kNumContinents);
static_assert(gf::university_available_last - gf::university_available + 1 == kNumUniversities);
static_assert(gf::bonus_tiles_last - gf::bonus_tiles + 1 == 2 * kNumBonusTiles);
static_assert(df::frame_last - df::frame + 1 == kNumFrameKinds);
static_assert(df::turn_action_last - df::turn_action + 1 == kNumActions);
static_assert(df::tasks_used_last - df::tasks_used + 1 == kNumTasks);
static_assert(df::built_last - df::built + 1 == atom::kBuildingTypes);
static_assert(df::offered_last - df::offered + 1 == atom::kBuildingTypes);
static_assert(df::chosen_verb_last - df::chosen_verb + 1 == atom::kBuildVerbs);
static_assert(df::chosen_type_last - df::chosen_type + 1 == atom::kBuildingTypes);
static_assert(df::chosen_map_token_last - df::chosen_map_token + 1 == kNumMapTokens);
static_assert(pf::workers_out_last - pf::workers_out + 1 == kNumTasks);
static_assert(pf::partner_zoos_last - pf::partner_zoos + 1 == kNumContinents);
static_assert(pf::universities_last - pf::universities + 1 == kNumUniversities);
static_assert(pf::map_tokens_last - pf::map_tokens + 1 == kNumMapTokens);
static_assert(pf::icons_last - pf::icons + 1 == kNumIcons);
static_assert(af::action_last - af::action + 1 == kNumActions);
static_assert(af::slot_last - af::slot + 1 == kNumActions);
static_assert(cf::tags_last - cf::tags + 1 == kNumIcons);
static_assert(cf::req_last - cf::req + 1 == kNumReqs);
static_assert(cf::ability_last - cf::ability + 1 == kNumKeywords);
static_assert(jf::icon_last - jf::icon + 1 == kNumProjIcons);
static_assert(jf::category_last - jf::category + 1 == kNumProjCategories);
static_assert(xf::bonus_last - xf::bonus + 1 == kNumPlacementBonuses);
static_assert(pf::buildings_last - pf::buildings + 1 == kNumBuildingKinds);
static_assert(xf::kind_last - xf::kind + 1 == kNumBuildingKinds);
static_assert(ef::kind_last - ef::kind + 1 == kNumBuildingKinds);
static_assert(cf::trigger_icon_last - cf::trigger_icon + 1 == kNumIcons);
// Card and project tokens share the location block.
static_assert(static_cast<int>(jf::my_hand) == static_cast<int>(cf::my_hand) &&
              static_cast<int>(jf::in_range) == static_cast<int>(cf::in_range) &&
              static_cast<int>(jf::in_play) == static_cast<int>(cf::in_range) + 1 &&
              static_cast<int>(cf::unknown) == static_cast<int>(cf::in_range) + 1);
static_assert(gf::kCount <= Enc::kFeatures && df::kCount <= Enc::kFeatures && pf::kCount <= Enc::kFeatures &&
              af::kCount <= Enc::kFeatures && jf::kCount <= Enc::kFeatures && xf::kCount <= Enc::kFeatures &&
              ef::kCount <= Enc::kFeatures);
// The discard and unseen pools share one block layout.
constexpr int kPoolWidth = gf::discard_conservation_last - gf::discard_type + 1;
static_assert(gf::unseen_type == gf::discard_type + kPoolWidth &&
              gf::unseen_conservation_last == gf::unseen_type + kPoolWidth - 1);
static_assert(gf::discard_icons_last - gf::discard_icons + 1 == kNumIcons);
static_assert(gf::discard_project_category_last - gf::discard_project_category + 1 == kNumProjCategories);
constexpr int kAnimalSizes = gf::discard_animal_size_last - gf::discard_animal_size + 1;
constexpr int kLowestSponsorLevel = 3;  // sponsor level columns: 3 or less, 4, 5, 6 or more
constexpr int kSponsorLevels = gf::discard_sponsor_level_last - gf::discard_sponsor_level + 1;
constexpr int kFinalCardTypes = gf::unseen_final_last - gf::unseen_final + 1;

bool valid_card(int v) { return v >= 0 && v < num_cards(); }

using PoolRow = std::array<float, kPoolWidth>;

// Per CardIdx, what the card adds to each pool column: its count (or value) in the column's bucket
// over that bucket's count (or sum) among all regular cards; zero for final scoring cards.
const std::vector<PoolRow>& pool_shares() {
  static const std::vector<PoolRow> shares = [] {
    auto at = [](int feature) { return feature - gf::discard_type; };
    std::vector<PoolRow> rows(static_cast<std::size_t>(num_cards()), PoolRow{});
    PoolRow total{};
    for (int i = 0; i < num_cards(); ++i) {
      const CardIdx c = static_cast<CardIdx>(i);
      const CardDef& d = card(c);
      PoolRow& r = rows[static_cast<std::size_t>(i)];
      switch (d.type) {
        case CardType::kAnimal:
          r[at(gf::discard_type)] = 1.0f;
          r[at(gf::discard_animal_size) + d.size - 1] = 1.0f;
          r[at(gf::discard_appeal)] = d.appeal;
          r[at(gf::discard_conservation)] = d.conservation;
          break;
        case CardType::kSponsor:
          r[at(gf::discard_type) + 1] = 1.0f;
          r[at(gf::discard_sponsor_level) +
            std::clamp(d.level - kLowestSponsorLevel, 0, kSponsorLevels - 1)] = 1.0f;
          break;
        case CardType::kProject:
          r[at(gf::discard_type) + 2] = 1.0f;
          r[at(gf::discard_project_category) + static_cast<int>(d.proj_category) - 1] = 1.0f;
          break;
        case CardType::kEndgame:
          continue;
      }
      for (int k = 0; k < kNumIcons; ++k) r[at(gf::discard_icons) + k] = card_icons(c)[k];
      for (int k = 0; k < kPoolWidth; ++k) total[k] += r[k];
    }
    for (PoolRow& r : rows) {
      for (int k = 0; k < kPoolWidth; ++k) r[k] = total[k] > 0.0f ? r[k] / total[k] : 0.0f;
    }
    return rows;
  }();
  return shares;
}

// Per CardIdx, the unseen_final column of a final scoring card (they come in CardIdx order), or -1.
const std::vector<int>& final_columns() {
  static const std::vector<int> columns = [] {
    std::vector<int> out(static_cast<std::size_t>(num_cards()), -1);
    int n = 0;
    for (int i = 0; i < num_cards(); ++i) {
      if (card(static_cast<CardIdx>(i)).type == CardType::kEndgame) out[static_cast<std::size_t>(i)] = n++;
    }
    return out;
  }();
  return columns;
}

int lowest_cell(CellMask cells) { return __builtin_ctzll(cells); }

// The X of "keyword X", or -1. Iconic Animal stores a continent there, which is one of the card's
// own continent icons (check_card_table), so it has no number of its own.
int ability_number(const Ability& ab) { return ab.keyword == Keyword::kICONIC_ANIMAL ? -1 : ab.value; }

// A card token's place: location feature, anchor index and pointer atoms (or -1), display index.
struct Where {
  int loc;
  int anchor = -1;
  ActionId take = -1;
  ActionId discard = -1;
  int display = -1;
};

class Writer {
 public:
  Writer(const ArknovaState& st, int me, std::vector<float>* out)
      : st_(st), s_(st.core), me_(me), opp_(1 - me), out_(out) {
    if (s_.stack_n) top_ = &s_.top();
    for (int i : cards_in_range(s_, me_)) in_range_[i] = true;
  }

  // Appends a token and returns its features; valid until the next token is appended. `anchor` is
  // an anchor index (ArknovaTokenEncoder::kAnchor*) or -1, `pos_cell` the map cell it sits at or -1.
  float* add(int type, int entity, int anchor, int pos_cell = -1) {
    const std::size_t at = out_->size();
    out_->resize(at + Enc::kTokenDim, 0.0f);
    float* t = out_->data() + at;
    t[0] = 1.0f;
    t[1] = static_cast<float>(entity);
    t[2] = static_cast<float>(anchor + 1);
    t[3] = static_cast<float>(type);
    if (pos_cell >= 0) {
      t[4] = 1.0f;
      t[5] = static_cast<float>(map_cell(pos_cell).q);
      t[6] = static_cast<float>(map_cell(pos_cell).r);
    }
    return t + Enc::kHeader;
  }

  bool chosen(ActionId a) const { return std::find(st_.prefix.begin(), st_.prefix.end(), a) != st_.prefix.end(); }

  void global_token() {
    float* f = add(Enc::kGlobalToken, 0, -1);
    f[gf::break_pos] = s_.break_pos / static_cast<float>(kBreakTrack);
    if (s_.break_by == me_) f[gf::break_by] = 1.0f;
    if (s_.break_by == opp_) f[gf::break_by + 1] = 1.0f;
    f[gf::deck] = static_cast<float>(s_.deck.size()) / 200.0f;
    f[gf::discard] = static_cast<float>(s_.discard.size()) / 200.0f;
    f[gf::final_deck] = static_cast<float>(s_.final_deck.size()) / 12.0f;
    f[gf::donations] = s_.donations / 5.0f;
    f[gf::donation_cost] = next_donation_cost(s_) / 12.0f;
    if (s_.end_by == me_) f[gf::end_by] = 1.0f;
    if (s_.end_by == opp_) f[gf::end_by + 1] = 1.0f;
    f[gf::final_turns] = s_.final_turns >= 0 ? s_.final_turns / 2.0f : 0.0f;
    f[gf::final_card_discarded] = s_.final_card_discarded ? 1.0f : 0.0f;
    f[gf::my_turn] = s_.current == me_ ? 1.0f : 0.0f;
    for (int c = 0; c < kNumContinents; ++c) f[gf::partner_zoo_available + c] = s_.partner_zoos_available >> c & 1;
    for (int u = 0; u < kNumUniversities; ++u) f[gf::university_available + u] = s_.universities_available >> u & 1;
    for (int sp = 0; sp < 2; ++sp) {
      for (int i = 0; i < s_.n_bonus_tiles[sp]; ++i) f[gf::bonus_tiles + sp * kNumBonusTiles + s_.bonus_tiles[sp][i]] = 1.0f;
    }
    f[gf::base_pile] = static_cast<float>(s_.base_project_deck.size()) / 10.0f;
    // The frames below the pending decision: what still has to be resolved after it.
    f[gf::stack_depth] = std::min<int>(s_.stack_n, 10) / 10.0f;
    int below[kNumFrameKinds] = {};
    for (int i = 0; i + 1 < s_.stack_n; ++i) {
      const Frame& fr = s_.stack[i];
      ++below[static_cast<int>(fr.kind)];
      if (!is_auto(fr.kind) && fr.player == opp_) f[gf::pending_opponent] = 1.0f;
    }
    for (int k = 0; k < kNumFrameKinds; ++k) f[gf::pending + k] = std::min(below[k], 4) / 4.0f;
    // "After finishing" effects queued by the turns in progress.
    int after[3] = {};
    for (int d = 0; d < s_.turn_depth; ++d) {
      const TurnCtx& t = s_.turns[d];
      for (int i = 0; i < t.n_after; ++i) {
        switch (t.after[i].kind) {
          case FrameKind::clever: ++after[0]; break;
          case FrameKind::ab_boost: ++after[1]; break;
          case FrameKind::extra_action: ++after[2]; break;
          default: throw std::logic_error("arknova encoder: unexpected deferred frame");
        }
      }
    }
    f[gf::after_clever] = std::min(after[0], 4) / 4.0f;
    f[gf::after_boost] = std::min(after[1], 4) / 4.0f;
    f[gf::after_extra_action] = std::min(after[2], 4) / 4.0f;
    // The discard pile (public), and the cards whose place I do not know: the deck and the
    // opponent's hidden hand, tucked cards and private draws (the belief tracker's pools), less
    // the cards it may have tucked, which have tokens of their own.
    const std::vector<PoolRow>& share = pool_shares();
    auto add_pool = [&](int base, int c) {
      const PoolRow& r = share[static_cast<std::size_t>(c)];
      for (int k = 0; k < kPoolWidth; ++k) f[base + k] += r[k];
    };
    for (CardIdx c : s_.discard) add_pool(gf::discard_type, c);
    const std::vector<std::uint8_t> known = known_cards(st_, me_, false);
    const Player& opp = s_.players[opp_];
    for (int c = 0; c < num_cards(); ++c) {
      if (known[static_cast<std::size_t>(c)] || opp.maybe_tucked[c]) continue;
      const int final_column = final_columns()[static_cast<std::size_t>(c)];
      if (final_column >= 0) {
        f[gf::unseen_final + final_column] = 1.0f;
      } else {
        add_pool(gf::unseen_type, c);
      }
    }
  }

  void decision_token() {
    float* f = add(Enc::kDecisionToken, 0, -1);
    f[df::to_move] = st_.current_player() == me_ ? 1.0f : 0.0f;
    f[df::prefix_len] = static_cast<float>(st_.prefix.size()) / kMaxPrefix;
    if (s_.turn_depth > 0) {
      const TurnCtx& tc = s_.turn();
      if (tc.action >= 0) f[df::turn_action + tc.action] = 1.0f;
      if (tc.has_strength) f[df::turn_strength] = tc.strength / 10.0f;
      f[df::turn_xtoken] = tc.xtoken ? 1.0f : 0.0f;
    }
    f[df::turn_depth] = s_.turn_depth / static_cast<float>(kMaxTurnDepth);
    f[df::turn_borrowed] = s_.borrowed_action >= 0 ? 1.0f : 0.0f;
    for (ActionId a : st_.prefix) {
      if (a >= atom::kBuildingType && a < atom::kBuildingType + atom::kBuildVerbs * atom::kBuildingTypes) {
        int i = a - atom::kBuildingType;
        f[df::chosen_verb + i / atom::kBuildingTypes] = 1.0f;
        f[df::chosen_type + i % atom::kBuildingTypes] = 1.0f;
      } else if (a >= atom::kPosition && a < atom::kPosition + 3) {
        f[df::chosen_position + (a - atom::kPosition)] = 1.0f;
      } else if (a >= atom::kMapToken && a < atom::kMapToken + kNumMapTokens) {
        f[df::chosen_map_token + (a - atom::kMapToken)] = 1.0f;
      }
    }
    if (!top_) return;
    const Frame& t = *top_;
    const int p = t.player;
    f[df::frame + static_cast<int>(t.kind)] = 1.0f;
    // Frame data by kind (the layouts of arknova_internal.h); kNull reads as 0.
    auto d = [&t](int off) { return t.d[off] == kNull ? 0 : static_cast<int>(t.d[off]); };
    auto type_list = [&](int at, int n_off, int first) {
      for (int i = 0; i < d(n_off); ++i) {
        f[at + building_type(static_cast<BuildingKind>(t.d[first + 2 * i]), t.d[first + 2 * i + 1])] = 1.0f;
      }
    };
    switch (t.kind) {
      case FrameKind::act_cards:
        f[df::remaining] = (d(fd::cards_draw) - d(fd::cards_drawn)) / 5.0f;
        f[df::done_count] = d(fd::cards_drawn) / 5.0f;
        f[df::side2] = d(fd::cards_upgraded) ? 1.0f : 0.0f;
        f[df::discard_after] = static_cast<float>(d(fd::cards_discard));
        f[df::may_snap] = d(fd::cards_may_snap) && d(fd::cards_drawn) == 0 ? 1.0f : 0.0f;
        break;
      case FrameKind::act_build:
        f[df::budget] = d(fd::build_budget) / 10.0f;
        f[df::done_count] = d(fd::build_built_n) / 5.0f;
        f[df::side2] = d(fd::build_upgraded) ? 1.0f : 0.0f;
        f[df::normal_done] = d(fd::build_normal_done) ? 1.0f : 0.0f;
        f[df::engineer_used] = d(fd::build_engineer_used) ? 1.0f : 0.0f;
        type_list(df::built, fd::build_built_n, fd::build_built);
        break;
      case FrameKind::act_animals:
        f[df::remaining] = d(fd::animals_remaining) / 5.0f;
        f[df::done_count] = d(fd::animals_played) / 5.0f;
        f[df::side2] = is_upgraded(s_, p, kAnimals) ? 1.0f : 0.0f;
        f[df::animals_phase + d(fd::animals_phase)] = 1.0f;
        f[df::all_small] = d(fd::animals_all_small) ? 1.0f : 0.0f;
        break;
      case FrameKind::act_association:
        f[df::budget] = d(fd::assoc_remaining) / 10.0f;
        f[df::side2] = d(fd::assoc_side2) ? 1.0f : 0.0f;
        for (int task = 0; task < kNumTasks; ++task) f[df::tasks_used + task] = d(fd::assoc_used) >> task & 1;
        f[df::donated] = d(fd::assoc_donated) ? 1.0f : 0.0f;
        break;
      case FrameKind::act_sponsors:
        f[df::budget] = d(sponsors_budget) / 10.0f;
        f[df::done_count] = d(sponsors_played) / 5.0f;
        f[df::side2] = is_upgraded(s_, p, kSponsors) ? 1.0f : 0.0f;
        break;
      case FrameKind::ab_sun: case FrameKind::ab_dig: case FrameKind::ab_pouch:
        f[df::remaining] = d(fd::left) / 5.0f;
        break;
      case FrameKind::ab_snap:
        f[df::remaining] = d(fd::left) / 5.0f;
        f[df::can_replenish] = d(fd::snap_can_replenish) ? 1.0f : 0.0f;
        f[df::replenish_between] = d(fd::snap_replenish_between) ? 1.0f : 0.0f;
        break;
      case FrameKind::ab_pick:
        f[df::remaining] = d(fd::pick_count) / 5.0f;
        f[df::list_len] = d(fd::pick_n) / 10.0f;
        break;
      case FrameKind::ab_hunter: case FrameKind::ab_resist:
        f[df::list_len] = d(fd::cards_n) / 10.0f;
        break;
      case FrameKind::ab_project:
        f[df::list_len] = static_cast<float>(s_.base_project_deck.size()) / 10.0f;
        f[df::only_one] = t.d[fd::project_only] != kNull ? 1.0f : 0.0f;
        break;
      case FrameKind::discard_hand: {
        // At a break the frame holds the hand limit, after a Cards action the number to discard.
        const int hand = s_.players[p].hand_size;
        const int left = t.d[fd::discard_limit] != kNull ? std::max(0, hand - d(fd::discard_limit))
                         : t.d[fd::discard_count] == kNull ? 1 : d(fd::discard_count);
        f[df::remaining] = std::min(left, hand) / 5.0f;
        break;
      }
      case FrameKind::extra_action:
        f[df::optional] = d(fd::extra_optional) ? 1.0f : 0.0f;
        for (int a = 0; a < kNumActions; ++a) f[df::extra_actions + a] = d(fd::extra_actions) >> a & 1;
        f[df::extra_from_opponent] = d(fd::extra_owner) != p ? 1.0f : 0.0f;
        break;
      case FrameKind::ab_boost:
        f[df::boost_action + d(fd::boost_action)] = 1.0f;
        break;
      case FrameKind::conservation_tile:
        f[df::tile_space + (d(fd::tile_space) == kConservationTileSpaces[0] ? 0 : 1)] = 1.0f;
        break;
      case FrameKind::free_build:
        type_list(df::offered, fd::free_build_n, fd::free_build_kinds);
        break;
      default:
        break;  // no data, or data shown elsewhere (sponsor_building: card flag, move_animals: building flag)
    }
  }

  void player_token(int p) {
    const Player& pl = s_.players[p];
    float* f = add(Enc::kPlayerToken, 0, -1);
    f[p == me_ ? pf::me : pf::opponent] = 1.0f;
    f[pf::to_move] = st_.current_player() == p ? 1.0f : 0.0f;
    f[pf::appeal] = pl.appeal / 60.0f;
    f[pf::money] = pl.money / 50.0f;
    f[pf::conservation] = pl.conservation / 30.0f;
    f[pf::reputation] = pl.reputation / static_cast<float>(kReputationMax);
    f[pf::xtokens] = pl.xtokens / static_cast<float>(kMaxXtokens);
    f[pf::workers] = pl.workers / static_cast<float>(kMaxWorkers);
    f[pf::active_workers] = pl.active_workers() / static_cast<float>(kMaxWorkers);
    for (int i = 0; i < kNumTasks; ++i) f[pf::workers_out + i] = pl.workers_out[i] / static_cast<float>(kMaxWorkers);
    f[pf::hand_size] = pl.hand_size / 20.0f;
    f[pf::hand_limit] = pl.hand_limit() / 5.0f;
    f[pf::zoo_cards] = pl.n_zoo_cards / 30.0f;
    f[pf::tucked] = pl.n_tucked / 10.0f;
    f[pf::final_cards] = pl.n_final / static_cast<float>(kMaxFinalCards);
    for (int i = 0; i < pl.n_partner_zoos; ++i) f[pf::partner_zoos + pl.partner_zoos[i]] = 1.0f;
    for (int i = 0; i < pl.n_universities; ++i) f[pf::universities + pl.universities[i]] = 1.0f;
    for (int i = 0; i < kNumMapTokens; ++i) f[pf::map_tokens + i] = pl.tokens[i] ? 1.0f : 0.0f;
    IconCounts ic = icons(pl);
    for (int i = 0; i < kNumIcons; ++i) f[pf::icons + i] = ic[i] / 10.0f;
    f[pf::victory_points] = pl.victory_points() / 30.0f;
    f[pf::poisoned] = has_venom(s_, p) ? 1.0f : 0.0f;
    f[pf::covered_cells] = popcount(pl.zoo.covered()) / 50.0f;
    for (int b = 0; b < pl.zoo.n_buildings; ++b) f[pf::buildings + static_cast<int>(pl.zoo.buildings[b].kind)] += 0.1f;
    f[pf::kiosk_income] = pl.zoo.kiosk_income() / 10.0f;
    f[pf::appeal_income] = appeal_income(pl.appeal) / 40.0f;
    f[pf::display_range] = reputation_folders(pl.reputation) / static_cast<float>(kDisplaySize);
    for (const KindSize& ks : placeable_buildings(s_, p)) f[pf::placeable + building_type(ks.kind, ks.size)] = 1.0f;
  }

  void action_card_tokens(int p) {
    const Player& pl = s_.players[p];
    const bool poisoned = has_venom(s_, p);
    // The card of the action in progress: the turn player's, or the other player's when borrowed.
    int used_action = -1, used_owner = -1;
    if (s_.turn_depth > 0) {
      used_action = s_.turn().action;
      used_owner = s_.borrowed_action == used_action ? 1 - s_.current : s_.current;
    }
    for (int a = 0; a < kNumActions; ++a) {
      float* f = add(Enc::kActionCardToken, Enc::kEntityAction + a, -1);
      f[p == me_ ? af::mine : af::opponent] = 1.0f;
      f[af::action + a] = 1.0f;
      const int strength = pl.strength(a);
      f[af::slot + strength - 1] = 1.0f;
      f[af::strength] = strength / 5.0f;
      f[af::effective_strength] = effective_strength(s_, p, a) / 5.0f;
      f[af::upgraded] = pl.is_upgraded(a) ? 1.0f : 0.0f;
      const bool venom = has_token(s_, p, a, TokenKind::kVenom);
      f[af::venom] = venom ? 1.0f : 0.0f;
      f[af::constriction] = has_token(s_, p, a, TokenKind::kConstriction) ? 1.0f : 0.0f;
      int multipliers = 0;
      for (int i = 0; i < pl.n_action_tokens[a]; ++i) multipliers += pl.action_tokens[a][i] == TokenKind::kMultiplier;
      f[af::multiplier] = multipliers / static_cast<float>(kMaxActionTokens);
      f[af::venom_cost] = poisoned && !venom ? 1.0f : 0.0f;
      f[af::in_use] = p == used_owner && a == used_action ? 1.0f : 0.0f;
    }
  }

  // One token for a card at a place: a project token for a conservation project, else a card token.
  float* card_token(int c, const Where& w) {
    const bool known = valid_card(c);
    const bool project = known && card(static_cast<CardIdx>(c)).type == CardType::kProject;
    // Zoos hold animals and sponsors, final cards come from the final scoring deck, the projects in
    // play are projects; a token type without the location's feature would set another feature.
    const bool card_place = w.loc == cf::my_zoo || w.loc == cf::opp_zoo || w.loc == cf::my_final;
    if (project ? card_place : w.loc == jf::in_play) {
      throw std::logic_error("arknova encoder: a card at a place its token type has no feature for");
    }
    float* f = add(project ? Enc::kProjectToken : Enc::kCardToken, known ? Enc::kEntityCard + c : 0, w.anchor);
    f[w.loc] = 1.0f;
    if (w.take >= 0 && chosen(w.take)) f[cf::chosen] = 1.0f;
    if (w.discard >= 0 && chosen(w.discard)) f[cf::chosen_discard] = 1.0f;
    if (w.display >= 0) {
      f[cf::display_pos] = (w.display + 1) / static_cast<float>(kDisplaySize);
      f[cf::in_range] = in_range_[w.display] ? 1.0f : 0.0f;
    }
    if (!known) {
      f[cf::unknown] = 1.0f;
      return f;
    }
    const bool in_zoo = w.loc == cf::my_zoo || w.loc == cf::opp_zoo;
    // The sponsor whose unique building is being placed.
    if (in_zoo && top_ && top_->kind == FrameKind::sponsor_building && top_->d[sponsor_building_card] == c) {
      f[cf::pending] = 1.0f;
    }
    if (project) {
      project_features(f, static_cast<CardIdx>(c), w);
    } else {
      card_features(f, static_cast<CardIdx>(c), w);
    }
    return f;
  }

  void cell_tokens() {
    const Player& pl = s_.players[me_];
    const Zoo& zoo = pl.zoo;
    const CellMask covered = zoo.covered();
    const CellMask adjacent = neighbor_mask(covered);
    const bool build2 = pl.is_upgraded(kBuild);
    const bool any_terrain = sr::ignores_terrain(pl);
    for (int c = 0; c < kMapCells; ++c) {
      const MapCell& mc = map_cell(c);
      float* f = add(Enc::kCellToken, Enc::kEntityCell + c, Enc::kAnchorCell + c, c);
      const CellMask bit = CellMask{1} << c;
      f[xf::terrain + static_cast<int>(mc.terrain)] = 1.0f;
      f[xf::build2_only] = mc.build2_only ? 1.0f : 0.0f;
      f[xf::border] = mc.border ? 1.0f : 0.0f;
      f[xf::bonus + mc.bonus] = 1.0f;
      f[xf::covered] = covered & bit ? 1.0f : 0.0f;
      f[xf::adjacent] = adjacent & bit ? 1.0f : 0.0f;
      const bool free_terrain = mc.terrain == Terrain::kPlain || any_terrain;
      f[xf::buildable] = !(covered & bit) && free_terrain && (!mc.build2_only || build2) ? 1.0f : 0.0f;
      const int b = zoo.cell_owner[c];
      if (b >= 0) {
        // The cells of one building share its kind and point at its lowest cell (where an
        // enclosure's token sits).
        const Building& bb = zoo.buildings[b];
        const MapCell& low = map_cell(lowest_cell(bb.cells));
        f[xf::kind + static_cast<int>(bb.kind)] = 1.0f;
        f[xf::anchor] = lowest_cell(bb.cells) == c ? 1.0f : 0.0f;
        f[xf::anchor_dq] = (low.q - mc.q) / 4.0f;
        f[xf::anchor_dr] = (low.r - mc.r) / 4.0f;
        if (bb.kind == BuildingKind::kKiosk) f[xf::kiosk_income] = zoo.kiosk_income_of(b) / 6.0f;
      }
      int rock = 0, water = 0, kiosks = 0;
      for (int nb : mc.nbr) {
        if (nb < 0) continue;
        rock += map_cell(nb).terrain == Terrain::kRock;
        water += map_cell(nb).terrain == Terrain::kWater;
        kiosks += zoo.cell_owner[nb] >= 0 && zoo.buildings[zoo.cell_owner[nb]].kind == BuildingKind::kKiosk;
      }
      f[xf::rock_neighbors] = rock / 6.0f;
      f[xf::water_neighbors] = water / 6.0f;
      f[xf::kiosks_adjacent] = kiosks / 3.0f;
    }
  }

  // The buildings that house animals. A kiosk, pavilion or unique building has no token: on my
  // map its cells show it, and each player token counts the buildings by kind.
  void enclosure_tokens(int p) {
    const Player& pl = s_.players[p];
    const Zoo& zoo = pl.zoo;
    const bool mine = p == me_;
    int target = -1;  // the new special enclosure animals may move into
    if (top_ && top_->kind == FrameKind::move_animals && top_->player == p) target = top_->d[move_animals_building];
    for (int b = 0; b < zoo.n_buildings; ++b) {
      const Building& bb = zoo.buildings[b];
      if (bb.kind != BuildingKind::kEnclosure && !is_special(bb.kind)) continue;
      float* f = add(Enc::kEnclosureToken, 0, mine ? Enc::kAnchorBuilding + b : -1, mine ? lowest_cell(bb.cells) : -1);
      f[mine ? ef::mine : ef::opponent] = 1.0f;
      f[ef::kind + static_cast<int>(bb.kind)] = 1.0f;
      f[ef::size] = bb.size / 5.0f;
      if (bb.kind == BuildingKind::kEnclosure) {
        f[ef::capacity] = bb.capacity() / 5.0f;
        f[ef::occupied] = bb.occupied ? 1.0f : 0.0f;
        f[ef::free] = bb.occupied ? 0.0f : bb.capacity() / 5.0f;
      } else {
        f[ef::capacity] = bb.size / 5.0f;
        f[ef::used] = bb.used / 5.0f;
        f[ef::free] = (bb.size - bb.used) / 5.0f;
      }
      f[ef::rock] = zoo.adjacent_terrain(bb.cells, Terrain::kRock) / 6.0f;
      f[ef::water] = zoo.adjacent_terrain(bb.cells, Terrain::kWater) / 6.0f;
      int kiosks = 0;
      for (int a : zoo.adjacent_buildings(b)) kiosks += zoo.buildings[a].kind == BuildingKind::kKiosk;
      f[ef::kiosks_adjacent] = kiosks / 3.0f;
      int animals = 0;
      for (int i = 0; i < pl.n_housing; ++i) animals += pl.housing[i].building == b;
      f[ef::animals] = animals / 4.0f;
      f[ef::target] = b == target ? 1.0f : 0.0f;
    }
  }

  const ArknovaState& st_;
  const State& s_;
  int me_, opp_;
  const Frame* top_ = nullptr;

 private:
  // Whose card it is for the "what would it do now" features: the opponent's for its hand,
  // maybe-tucked and zoo cards, mine otherwise.
  int holder_of(const Where& w) const {
    return w.loc == cf::opp_hand || w.loc == cf::opp_maybe_tucked || w.loc == cf::opp_zoo ? opp_ : me_;
  }

  // Strength points the holder's action card lacks for it now, before X-tokens, / 5.
  float strength_short(int holder, int action, int needed) const {
    return std::clamp(needed - effective_strength(s_, holder, action), 0, 5) / 5.0f;
  }

  // Printed facts of an animal, sponsor or final scoring card, and what they mean now for its holder.
  void card_features(float* f, CardIdx c, const Where& w) {
    const CardDef& d = card(c);
    const int holder = holder_of(w);
    const Player& owner = s_.players[holder];
    const bool animal = d.type == CardType::kAnimal, sponsor = d.type == CardType::kSponsor;
    f[cf::type + (animal ? 0 : sponsor ? 1 : 2)] = 1.0f;
    f[cf::cost] = d.cost / 36.0f;
    f[cf::size] = d.size / 5.0f;
    f[cf::appeal] = d.appeal / 10.0f;
    f[cf::conservation] = d.conservation / 5.0f;
    f[cf::reputation] = d.reputation / 5.0f;
    f[cf::level] = d.level / 6.0f;
    f[cf::rock] = d.enc_rock / 2.0f;
    f[cf::water] = d.enc_water / 2.0f;
    f[cf::no_standard] = animal && d.enc_standard < 0 ? 1.0f : 0.0f;
    for (int k = 0; k < d.n_special; ++k) {
      const int kind = static_cast<int>(d.special[k].kind) - 1;  // reptile house, aviary, petting zoo
      f[cf::special + 2 * kind] = 1.0f;
      f[cf::special + 2 * kind + 1] = d.special[k].size / 5.0f;
    }
    for (int i = 0; i < kNumIcons; ++i) f[cf::tags + i] = d.tags[i] / 2.0f;
    for (int r = 0; r < kNumReqs; ++r) {
      if (!d.req[r]) continue;
      f[cf::req + r] = r == kReqReputation ? d.min_reputation / static_cast<float>(kReputationMax)
                       : r == kReqAppeal ? d.max_appeal / 50.0f
                                         : d.req[r] / 2.0f;
    }
    for (int i = 0; i < d.n_abilities; ++i) {
      // One feature per keyword: 1 for a keyword without a number, (X + 1) / 6 for "keyword X".
      const Ability& ab = d.abilities[i];
      f[cf::ability + static_cast<int>(ab.keyword) - 1] = ability_number(ab) >= 0 ? (ab.value + 1) / 6.0f : 1.0f;
    }
    f[cf::effect + 0] = has_immediate(c) ? 1.0f : 0.0f;
    f[cf::effect + 1] = has_icon_trigger(c) ? 1.0f : 0.0f;
    f[cf::effect + 2] = has_income(c) ? 1.0f : 0.0f;
    f[cf::effect + 3] = has_endgame(c) ? 1.0f : 0.0f;
    f[cf::effect + 4] = d.lasting_effect ? 1.0f : 0.0f;
    if (d.building.present) {
      f[cf::building + 0] = 1.0f;
      f[cf::building + 1] = d.building.size / 5.0f;
      f[cf::building + 2] = d.building.rock / 2.0f;
      f[cf::building + 3] = d.building.water / 2.0f;
      f[cf::building + 4] = d.building.min_border / 2.0f;
      f[cf::building + 5] = d.building.needs_adjacency ? 1.0f : 0.0f;
    }
    for (int i = 0; i < d.n_scoring; ++i) {
      f[cf::scoring + 2 * i] = d.scoring[i].threshold / 24.0f;
      f[cf::scoring + 2 * i + 1] = d.scoring[i].conservation / 4.0f;
    }
    for (int p = 0; p < kPlayers; ++p) {
      const int n = card_tokens_get(s_.players[p], c);
      if (n > 0) f[cf::tokens] = n / 5.0f;
    }
    if (animal || sponsor) f[cf::unmet] = unmet_requirements(owner, c) / 3.0f;

    const bool held = w.loc == cf::my_zoo || w.loc == cf::opp_zoo || w.loc == cf::my_final;
    if (!held && (animal || sponsor)) {
      // What playing it now would take: money (with the holder's discounts and the display
      // folder), its conditions, a place for it in the holder's zoo, and action card strength.
      const int money = (w.display >= 0 ? w.display + 1 : 0) + (animal ? animal_cost(owner, c) : 0);
      f[cf::money_now] = money / 36.0f;
      f[cf::affordable] = owner.money >= money ? 1.0f : 0.0f;
      bool req_ok, has_place;
      if (animal) {
        req_ok = sr::requirements_ok(owner, c) && sr::animal_allowed(owner, c);
        has_place = !housings(owner, c).empty();
        f[cf::strength_short] = strength_short(holder, kAnimals, animals_strength_needed(owner));
      } else {
        req_ok = unmet_requirements(owner, c) == 0;
        bool has_building = false;
        const bool spot = !building_spots(s_, holder, c, &has_building).empty();
        has_place = !has_building || spot;
        f[cf::strength_short] = strength_short(holder, kSponsors, sponsor_strength_needed(s_, holder, c));
      }
      f[cf::req_ok] = req_ok ? 1.0f : 0.0f;
      f[cf::has_place] = has_place ? 1.0f : 0.0f;
    }
    // What the card's end-of-game effect would score right now for the player holding it (me for
    // my final cards and the ones I am offered).
    if (has_endgame(c) && (held || w.loc == cf::private_list)) {
      State& x = scratch();
      Player& pl = x.players[holder];
      const auto appeal = pl.appeal, conservation = pl.conservation;
      endgame_effect(x, c, static_cast<int>(&pl - x.players));
      f[cf::yield + 0] = (pl.conservation - conservation) / 4.0f;
      f[cf::yield + 1] = (pl.appeal - appeal) / 10.0f;
      pl.appeal = appeal;
      pl.conservation = conservation;
    }
    if (sponsor) sponsor_features(f, c, w, held);
  }

  // A sponsor's effects, measured by running the rules' effect code on a copy of the state: what
  // its immediate effect would give its holder if played now, what its break income pays the
  // holder (as if in its zoo when not yet there), and what each icon that triggers it brings.
  void sponsor_features(float* f, CardIdx c, const Where& w, bool held) {
    auto part = [](int v, float top) { return std::min(static_cast<float>(std::max(v, 0)), top) / top; };
    const bool income = has_income(c), immediate = !held && has_immediate(c);
    if (income || immediate) {
      const int p = holder_of(w);
      State x = s_;
      Player& pl = x.players[p];
      if (!held) add_zoo_card(pl, c);  // the effects run with the card in the zoo
      struct Mark { int money, appeal, conservation, xtokens, workers, frames; };
      auto mark = [&] { return Mark{pl.money, pl.appeal, pl.conservation, pl.xtokens, pl.workers, x.stack_n}; };
      // Cards to take and free buildings come as frames pushed on the stack.
      auto pushed = [&](const Mark& m, FrameKind kind) {
        int n = 0;
        for (int i = m.frames; i < x.stack_n; ++i) n += x.stack[i].kind == kind && x.stack[i].player == p;
        return n;
      };
      if (income) {
        const Mark m = mark();
        income_effect(x, c, p);
        f[cf::income + 0] = part(pl.money - m.money, 10.0f);
        f[cf::income + 1] = part(pl.conservation - m.conservation, 2.0f);
        f[cf::income + 2] = part(pl.xtokens - m.xtokens + pushed(m, FrameKind::take_card), 2.0f);
      }
      if (immediate) {
        const Mark m = mark();
        immediate_effect(x, c, p);
        f[cf::immediate + 0] = part(pl.money - m.money, 20.0f);
        f[cf::immediate + 1] = part(pl.appeal - m.appeal, 10.0f);
        f[cf::immediate + 2] = part(pl.conservation - m.conservation, 5.0f);
        f[cf::immediate + 3] = part(pushed(m, FrameKind::take_card) + pl.workers - m.workers +
                                        pushed(m, FrameKind::free_build), 2.0f);
      }
    }
    if (has_icon_trigger(c)) {
      const IconTriggerSpec t = icon_trigger_spec(c);
      if (t.icon >= 0) f[cf::trigger_icon + t.icon] = 1.0f;
      f[cf::trigger_any] = t.any_zoo ? 1.0f : 0.0f;
      if (t.amount > 0 && t.kind == EffectKind::kEffMoney) {
        f[cf::trigger_gain + 0] = t.amount / 3.0f;
      } else if (t.amount > 0 && t.kind == EffectKind::kEffAppeal) {
        f[cf::trigger_gain + 1] = t.amount / 2.0f;
      } else {
        f[cf::trigger_gain + 2] = 1.0f;  // a track step, an X-token, a free building, a card, an ability
      }
    }
  }

  void project_features(float* f, CardIdx c, const Where& w) {
    const CardDef& d = card(c);
    const Player& mine = s_.players[me_];
    f[jf::category + static_cast<int>(d.proj_category) - 1] = 1.0f;
    f[jf::icon + static_cast<int>(d.proj_icon) - 1] = 1.0f;
    for (int k = 0; k < d.n_slots; ++k) {
      const ProjectSlot& slot = d.slots[k];
      if (slot.required >= 0) f[jf::level_required + k] = slot.required / 8.0f;
      if (slot.size_min >= 0) f[jf::level_size_min + k] = slot.size_min / 5.0f;
      if (slot.size_max >= 0) f[jf::level_size_max + k] = slot.size_max / 5.0f;
      for (int i = 0; i < slot.n_rewards; ++i) {
        const int at = slot.rewards[i].reward == Reward::kRewardConservation ? jf::level_conservation : jf::level_reputation;
        f[at + k] += slot.rewards[i].amount / 5.0f;
      }
    }
    for (int i = 0; i < d.n_play_bonus; ++i) {
      const Effect& e = d.play_bonus[i];
      const int at = e.kind == EffectKind::kEffMoney ? 0 : e.kind == EffectKind::kEffAppeal ? 1
                     : e.kind == EffectKind::kEffConservation ? 2 : 3;  // reputation; checked at startup
      f[jf::play_bonus + at] += e.amount / (at < 2 ? 10.0f : 5.0f);
    }
    if (const ProjectTokens* pt = s_.tokens_of(c)) {
      for (int k = 0; k < d.n_slots; ++k) {
        const int at = pt->slot[k] == -1 ? jf::occupied_free : pt->slot[k] == me_ ? jf::occupied_me
                       : pt->slot[k] == opp_ ? jf::occupied_opp : jf::occupied_blocked;
        f[at + k] = 1.0f;
      }
    }
    const std::array<bool, 3> can_me = supportable_levels(s_, me_, c), can_opp = supportable_levels(s_, opp_, c);
    for (int k = 0; k < 3; ++k) {
      f[jf::can_me + k] = can_me[k] ? 1.0f : 0.0f;
      f[jf::can_opp + k] = can_opp[k] ? 1.0f : 0.0f;
    }
    f[jf::count_me] = project_icon_count(mine, d.proj_icon) / 8.0f;
    f[jf::count_opp] = project_icon_count(s_.players[opp_], d.proj_icon) / 8.0f;
    f[jf::affordable] = w.display < 0 || mine.money >= w.display + 1 ? 1.0f : 0.0f;
    const int holder = holder_of(w);
    f[jf::strength_short] = strength_short(holder, kAssociation, project_strength_needed(s_.players[holder]));
  }

  // A private copy of the core for end-of-game effects, which write the score they award.
  State& scratch() {
    if (!scratch_) scratch_.emplace(s_);
    return *scratch_;
  }

  std::vector<float>* out_;
  bool in_range_[kDisplaySize] = {};
  std::optional<State> scratch_;
};

// The encoding relies on these properties of the card table; a card that breaks one would be
// encoded ambiguously, so refuse to start.
void check_card_table() {
  int valued[kNumKeywords + 1] = {};  // 0 unseen, 1 always with a number, 2 never
  int final_cards = 0;
  for (int i = 0; i < num_cards(); ++i) {
    const CardDef& d = card(static_cast<CardIdx>(i));
    const std::string name = d.name;
    final_cards += d.type == CardType::kEndgame;
    if (d.type == CardType::kAnimal && (d.size < 1 || d.size > kAnimalSizes)) {
      throw std::logic_error("arknova encoder: animal size outside the pool columns: " + name);
    }
    for (int k = 0; k < d.n_abilities; ++k) {
      const int kw = static_cast<int>(d.abilities[k].keyword);
      if (kw < 1 || kw > kNumKeywords) throw std::logic_error("arknova encoder: unknown ability keyword on " + name);
      if (d.abilities[k].keyword == Keyword::kICONIC_ANIMAL) {
        const int continent = d.abilities[k].value;
        if (continent < 0 || continent >= kNumContinents || !d.tags[kContinentIcon0 + continent]) {
          throw std::logic_error("arknova encoder: an Iconic Animal without its continent icon: " + name);
        }
      }
      const int v = ability_number(d.abilities[k]);
      if (v > 5) throw std::logic_error("arknova encoder: ability number above 5 on " + name);
      const int kind = v >= 0 ? 1 : 2;
      if (valued[kw] && valued[kw] != kind) {
        throw std::logic_error("arknova encoder: a keyword appears with and without a number: " + name);
      }
      valued[kw] = kind;
    }
    for (int k = 0; k < d.n_special; ++k) {
      if (d.special[k].kind == SpecialKind::kNone) throw std::logic_error("arknova encoder: bad special enclosure on " + name);
    }
    if (d.type == CardType::kSponsor && has_icon_trigger(static_cast<CardIdx>(i))) {
      const IconTriggerSpec t = icon_trigger_spec(static_cast<CardIdx>(i));
      if (t.icon >= kNumIcons || (t.kind == EffectKind::kEffMoney && t.amount > 3) ||
          (t.kind == EffectKind::kEffAppeal && t.amount > 2)) {
        throw std::logic_error("arknova encoder: icon trigger outside its features: " + name);
      }
    }
    if (d.type != CardType::kProject) continue;
    if (d.proj_category == ProjCategory::kNone || d.proj_icon == ProjIcon::kNone || d.n_slots != 3) {
      throw std::logic_error("arknova encoder: project without category, icon or 3 levels: " + name);
    }
    for (int k = 0; k < d.n_play_bonus; ++k) {
      const EffectKind e = d.play_bonus[k].kind;
      if (e != EffectKind::kEffMoney && e != EffectKind::kEffAppeal && e != EffectKind::kEffConservation &&
          e != EffectKind::kEffReputation) {
        throw std::logic_error("arknova encoder: play bonus kind without a feature: " + name);
      }
    }
  }
  if (final_cards != kFinalCardTypes) throw std::logic_error("arknova encoder: unseen_final needs one column per final card");
}

struct FeatureDesc {
  const char* name;
  int offset;
  int width;
};
#define ARKNOVA_DESC_TABLE(table, ns, FEATURES)                \
  const FeatureDesc table[] = {FEATURES(ARKNOVA_DESC_##ns)};
#define ARKNOVA_DESC_gf(name, width) {#name, gf::name, width},
#define ARKNOVA_DESC_df(name, width) {#name, df::name, width},
#define ARKNOVA_DESC_pf(name, width) {#name, pf::name, width},
#define ARKNOVA_DESC_af(name, width) {#name, af::name, width},
#define ARKNOVA_DESC_cf(name, width) {#name, cf::name, width},
#define ARKNOVA_DESC_jf(name, width) {#name, jf::name, width},
#define ARKNOVA_DESC_xf(name, width) {#name, xf::name, width},
#define ARKNOVA_DESC_ef(name, width) {#name, ef::name, width},
ARKNOVA_DESC_TABLE(kGlobalDesc, gf, ARKNOVA_GLOBAL_FEATURES)
ARKNOVA_DESC_TABLE(kDecisionDesc, df, ARKNOVA_DECISION_FEATURES)
ARKNOVA_DESC_TABLE(kPlayerDesc, pf, ARKNOVA_PLAYER_FEATURES)
ARKNOVA_DESC_TABLE(kActionCardDesc, af, ARKNOVA_ACTION_CARD_FEATURES)
ARKNOVA_DESC_TABLE(kCardDesc, cf, ARKNOVA_CARD_FEATURES)
ARKNOVA_DESC_TABLE(kProjectDesc, jf, ARKNOVA_PROJECT_FEATURES)
ARKNOVA_DESC_TABLE(kCellDesc, xf, ARKNOVA_CELL_FEATURES)
ARKNOVA_DESC_TABLE(kEnclosureDesc, ef, ARKNOVA_ENCLOSURE_FEATURES)
#undef ARKNOVA_DESC_TABLE

template <std::size_t N>
void append_type(std::string& out, const char* name, const FeatureDesc (&table)[N]) {
  out += std::string("{\"name\":\"") + name + "\",\"features\":[";
  for (std::size_t i = 0; i < N; ++i) {
    if (i) out += ',';
    out += std::string("[\"") + table[i].name + "\"," + std::to_string(table[i].offset) + "," +
           std::to_string(table[i].width) + "]";
  }
  out += "]}";
}

void append_range(std::string& out, const char* name, int base, int count, bool last = false) {
  out += std::string("[\"") + name + "\"," + std::to_string(base) + "," + std::to_string(count) + "]";
  if (!last) out += ',';
}

}  // namespace

ArknovaTokenEncoder::ArknovaTokenEncoder() {
  check_card_table();
  layout_.variable_tokens = true;
  layout_.token_positions = true;
  layout_.token_dim = kTokenDim;
  layout_.num_token_types = kTokenTypeCount;
  layout_.num_entity_ids = kNumEntities;
  layout_.num_pointer_heads = kNumHeads;
  layout_.action_token.assign(atom::kNumAtoms, -1);
  layout_.action_head.assign(atom::kNumAtoms, -1);
  layout_.num_anchor_ids = kNumAnchors;
  layout_.anchor_action.assign(static_cast<std::size_t>(kNumAnchors * kNumHeads), -1);
  auto bind = [this](int anchor, int head, int a) {
    layout_.anchor_action[static_cast<std::size_t>(anchor * kNumHeads + head)] = a;
  };
  for (int c = 0; c < kCardSlots; ++c) {
    bind(kAnchorHand + c, kHeadTake, atom::kHand + c);
    bind(kAnchorHand + c, kHeadDiscard, atom::kHandDiscard + c);
    bind(kAnchorZoo + c, kHeadTake, atom::kZoo + c);
    bind(kAnchorZoo + c, kHeadDiscard, atom::kZooGive + c);
  }
  for (int i = 0; i < kDisplaySize; ++i) {
    bind(kAnchorDisplay + i, kHeadTake, atom::kDisplayTake + i);
    bind(kAnchorDisplay + i, kHeadDiscard, atom::kDisplayDiscard + i);
    bind(kAnchorDisplay + i, kHeadSnap, atom::kDisplaySnap + i);
  }
  for (int i = 0; i < atom::kProjectSlots; ++i) bind(kAnchorProject + i, kHeadTake, atom::kProject + i);
  for (int i = 0; i < atom::kPublicSlots; ++i) bind(kAnchorPublic + i, kHeadTake, atom::kPublic + i);
  for (int i = 0; i < kPrivateCards; ++i) bind(kAnchorPrivate + i, kHeadTake, atom::kPrivate + i);
  for (int i = 0; i < kMaxFinalCards; ++i) bind(kAnchorFinal + i, kHeadDiscard, atom::kFinal + i);
  for (int b = 0; b < kMaxBuildings; ++b) bind(kAnchorBuilding + b, kHeadTake, atom::kBuilding + b);
  for (int c = 0; c < kMapCells; ++c) {
    for (int h = 0; h < atom::kCellHeads; ++h) bind(kAnchorCell + c, h, atom::cell(c, h));
  }
}

std::string ArknovaTokenEncoder::layout_json() {
  std::string out = "{\"token_dim\":" + std::to_string(kTokenDim) + ",\"header\":" + std::to_string(kHeader) +
                    ",\"header_fields\":[\"valid\",\"entity\",\"anchor\",\"type\",\"has_position\",\"q\",\"r\"]," +
                    "\"num_heads\":" + std::to_string(kNumHeads) + ",\"types\":[";
  append_type(out, "global", kGlobalDesc);
  out += ',';
  append_type(out, "decision", kDecisionDesc);
  out += ',';
  append_type(out, "player", kPlayerDesc);
  out += ',';
  append_type(out, "action_card", kActionCardDesc);
  out += ',';
  append_type(out, "card", kCardDesc);
  out += ',';
  append_type(out, "project", kProjectDesc);
  out += ',';
  append_type(out, "cell", kCellDesc);
  out += ',';
  append_type(out, "enclosure", kEnclosureDesc);
  out += "],\"entities\":[";
  append_range(out, "card", kEntityCard, kCardSlots);
  append_range(out, "cell", kEntityCell, kMapCells);
  append_range(out, "action_card", kEntityAction, kNumActions, true);
  out += "],\"anchors\":[";  // anchor ids, i.e. 1 + index
  append_range(out, "hand", 1 + kAnchorHand, kCardSlots);
  append_range(out, "zoo", 1 + kAnchorZoo, kCardSlots);
  append_range(out, "display", 1 + kAnchorDisplay, kDisplaySize);
  append_range(out, "project", 1 + kAnchorProject, atom::kProjectSlots);
  append_range(out, "public_list", 1 + kAnchorPublic, atom::kPublicSlots);
  append_range(out, "private_list", 1 + kAnchorPrivate, kPrivateCards);
  append_range(out, "final", 1 + kAnchorFinal, kMaxFinalCards);
  append_range(out, "enclosure", 1 + kAnchorBuilding, kMaxBuildings);
  append_range(out, "cell", 1 + kAnchorCell, kMapCells, true);
  out += "]}";
  return out;
}

void ArknovaTokenEncoder::encode_features(const IGameState& masked_state, int perspective_player,
                                          const IBeliefTracker*, std::vector<float>* out) const {
  const auto& st = checked_cast<ArknovaState>(masked_state);
  if (perspective_player < 0 || perspective_player >= kPlayers) {
    throw std::invalid_argument("arknova encoder: perspective out of range");
  }
  const State& s = st.core;
  const int me = perspective_player, opp = 1 - me;
  Writer w(st, me, out);

  w.global_token();
  w.decision_token();
  w.player_token(me);
  w.player_token(opp);
  w.action_card_tokens(me);
  w.action_card_tokens(opp);

  for (int c = 0; c < num_cards(); ++c) {
    if (st.hand[me][c] == 1) w.card_token(c, {cf::my_hand, kAnchorHand + c, atom::kHand + c, atom::kHandDiscard + c});
  }
  int unseen = s.players[opp].hand_size;
  for (int c = 0; c < num_cards(); ++c) {
    if (st.hand[opp][c] != 1) continue;
    w.card_token(c, {cf::opp_hand});
    --unseen;
  }
  if (unseen < 0) throw std::logic_error("arknova encoder: more seen opponent hand cards than its hand size");
  for (int i = 0; i < unseen; ++i) w.card_token(kHiddenCard, {cf::opp_hand});
  for (int c = 0; c < num_cards(); ++c) {
    if (s.players[opp].maybe_tucked[c]) w.card_token(c, {cf::opp_maybe_tucked});
  }
  for (int i = 0; i < kDisplaySize; ++i) {
    if (s.display[i] == kNoCard) continue;
    w.card_token(s.display[i], {cf::display, kAnchorDisplay + i, atom::kDisplayTake + i, -1, i});
  }
  for (int p : {me, opp}) {
    const Player& pl = s.players[p];
    for (int i = 0; i < pl.n_zoo_cards; ++i) {
      const int c = pl.zoo_cards[i];
      if (p == me) {
        w.card_token(c, {cf::my_zoo, kAnchorZoo + c, atom::kZoo + c});
      } else {
        w.card_token(c, {cf::opp_zoo});
      }
    }
  }
  if (w.top_) {
    const Frame& f = *w.top_;
    std::vector<int> list;
    if (f.kind == FrameKind::ab_hunter) {
      for (int j = 0; j < f.d[fd::cards_n]; ++j) list.push_back(f.d[fd::cards_list + j]);
    } else if (f.kind == FrameKind::ab_project) {
      list.assign(s.base_project_deck.begin(), s.base_project_deck.end());
    }
    // A Hunter reveal is a frame list, so it has at most kPublicSlots cards; the base project
    // pile only shrinks from the base project cards left after setup.
    if (static_cast<int>(list.size()) > atom::kPublicSlots) {
      throw std::logic_error("arknova encoder: public list longer than its slots");
    }
    for (int j = 0; j < static_cast<int>(list.size()); ++j) {
      w.card_token(list[j], {cf::public_list, kAnchorPublic + j, atom::kPublic + j});
    }
    if (is_private_frame(f.kind) && f.player == me) {
      std::vector<int> frames = private_frames_of(s, me);
      int k = static_cast<int>(std::find(frames.begin(), frames.end(), s.stack_n - 1) - frames.begin());
      int n_off, list_off;
      private_list_offsets(f.kind, &n_off, &list_off);
      for (int j = 0; j < f.d[n_off]; ++j) {
        w.card_token(st.private_cards[me][k][j], {cf::private_list, kAnchorPrivate + j, atom::kPrivate + j});
      }
    }
  }
  for (int i = 0; i < s.players[me].n_final; ++i) {
    w.card_token(st.final_cards[me][i], {cf::my_final, kAnchorFinal + i});
  }
  for (int i = 0; i < s.n_base_projects; ++i) {
    w.card_token(s.base_projects[i], {jf::in_play, kAnchorProject + i, atom::kProject + i});
  }
  for (int i = 0; i < s.n_projects; ++i) {
    const int slot = kBaseProjects + i;
    w.card_token(s.projects[i], {jf::in_play, kAnchorProject + slot, atom::kProject + slot});
  }
  w.cell_tokens();
  w.enclosure_tokens(me);
  w.enclosure_tokens(opp);
}

}  // namespace board_ai::arknova
