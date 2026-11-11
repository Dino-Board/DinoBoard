#include "splendor_net_adapter.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "../../engine/core/masked_state.h"
#include "../../engine/core/viz_runtime.h"

namespace board_ai::splendor {

namespace {

void encode_card(const SplendorCard* card, std::vector<float>* out) {
  if (!card) {
    out->insert(out->end(), 13, 0.0f);
    return;
  }
  out->push_back(1.0f);
  out->push_back(static_cast<float>(card->tier) / 3.0f);
  out->push_back(static_cast<float>(card->points) / 5.0f);
  for (int c = 0; c < kColorCount; ++c) {
    out->push_back(card->bonus == c ? 1.0f : 0.0f);
  }
  for (int c = 0; c < kColorCount; ++c) {
    out->push_back(static_cast<float>(card->cost[static_cast<size_t>(c)]) / 7.0f);
  }
}

void encode_hidden_reserved_placeholder(std::vector<float>* out) {
  out->push_back(1.0f);
  out->insert(out->end(), 12, 0.0f);
}

}  // namespace

// Layout reminder (kFeatureDim = 6 + 15*P + kNobleCount*6 + 12*13 + 39*P + 7):
//   Public (kFeatureDim - 39):
//     - bank (6)
//     - per-player stats in perspective order (15 * P)
//     - nobles (kNobleCount * 6)
//     - tableau (12 * 13)
//     - opp reserved slots (39 * (P-1)): visible → card, hidden → placeholder
//     - metadata (7)
//   Private (39):
//     - perspective's own 3 reserved slots × 13 dims (real cards — visible
//       AND hidden both carry the real card ID since perspective knows them)
template <int NPlayers>
void SplendorFeatureEncoder<NPlayers>::encode_features(
    const IGameState& state,
    int perspective_player,
    const IBeliefTracker* /*tracker*/,
    std::vector<float>* out) const {
  const auto* s = dynamic_cast<const SplendorState<NPlayers>*>(&state);
  if (!s || !out || perspective_player < 0 || perspective_player >= Cfg::kPlayers) return;
  const SplendorData<NPlayers>& d = s->persistent.data();
  const auto& cards = splendor_card_pool();
  const auto& nobles = splendor_nobles();

  for (int i = 0; i < kTokenTypes; ++i) {
    const float denom = (i == 5) ? 5.0f : static_cast<float>(Cfg::kGemCount);
    out->push_back(static_cast<float>(d.bank[static_cast<size_t>(i)]) / denom);
  }

  for (int pi = 0; pi < Cfg::kPlayers; ++pi) {
    const int pid = (perspective_player + pi) % Cfg::kPlayers;
    for (int i = 0; i < kTokenTypes; ++i) {
      out->push_back(static_cast<float>(d.player_gems[pid][static_cast<size_t>(i)]) / 10.0f);
    }
    for (int i = 0; i < kColorCount; ++i) {
      out->push_back(static_cast<float>(d.player_bonuses[pid][static_cast<size_t>(i)]) / 7.0f);
    }
    out->push_back(static_cast<float>(d.player_points[pid]) / 20.0f);
    out->push_back(static_cast<float>(d.reserved_size[pid]) / 3.0f);
    out->push_back(static_cast<float>(d.player_cards_count[pid]) / 20.0f);
    out->push_back(static_cast<float>(d.player_nobles_count[pid]) / 3.0f);
  }

  for (int i = 0; i < Cfg::kNobleCount; ++i) {
    if (i < d.nobles_size) {
      const int nid = d.nobles[static_cast<size_t>(i)];
      if (nid >= 0 && nid < static_cast<int>(nobles.size())) {
        for (int c = 0; c < kColorCount; ++c) {
          out->push_back(static_cast<float>(nobles[static_cast<size_t>(nid)][static_cast<size_t>(c)]) / 4.0f);
        }
        out->push_back(1.0f);
      } else {
        out->insert(out->end(), 6, 0.0f);
      }
    } else {
      out->insert(out->end(), 6, 0.0f);
    }
  }

  for (int tier = 0; tier < 3; ++tier) {
    for (int slot = 0; slot < 4; ++slot) {
      const bool exists = slot < d.tableau_size[static_cast<size_t>(tier)];
      if (!exists) {
        out->insert(out->end(), 13, 0.0f);
        continue;
      }
      const int cid = d.tableau[static_cast<size_t>(tier)][static_cast<size_t>(slot)];
      if (cid < 0 || cid >= static_cast<int>(cards.size())) {
        out->insert(out->end(), 13, 0.0f);
      } else {
        encode_card(&cards[static_cast<size_t>(cid)], out);
      }
    }
  }

  // Non-perspective players' reserved slots. The encoder reads the
  // masked clone's `reserved` field directly: face-down opp slots arrive
  // as kHiddenCardId (written by mask_field_slot / mask_all_hidden_slots).
  // Visible (face-up) opp slots carry the real cid. There is no viz
  // query here — the placeholder IS the visibility signal.
  for (int pi = 1; pi < Cfg::kPlayers; ++pi) {
    const int pid = (perspective_player + pi) % Cfg::kPlayers;
    for (int slot = 0; slot < 3; ++slot) {
      const bool exists = slot < d.reserved_size[pid];
      if (!exists) {
        out->insert(out->end(), 13, 0.0f);
        continue;
      }
      const int cid = d.reserved[pid][static_cast<size_t>(slot)];
      if (cid == kHiddenCardId) {
        encode_hidden_reserved_placeholder(out);
        continue;
      }
      if (cid < 0 || cid >= static_cast<int>(cards.size())) {
        out->insert(out->end(), 13, 0.0f);
      } else {
        encode_card(&cards[static_cast<size_t>(cid)], out);
      }
    }
  }

  out->push_back(static_cast<float>(std::min(d.plies, kMaxPlies)) / static_cast<float>(kMaxPlies));
  out->push_back(d.current_player == perspective_player ? 1.0f : 0.0f);
  out->push_back(static_cast<float>(std::max(0, d.pending_returns)) / 5.0f);
  const SplendorTurnStage stage = static_cast<SplendorTurnStage>(d.stage);
  out->push_back(stage == SplendorTurnStage::kNormal ? 1.0f : 0.0f);
  out->push_back(stage == SplendorTurnStage::kReturnTokens ? 1.0f : 0.0f);
  out->push_back(stage == SplendorTurnStage::kChooseNoble ? 1.0f : 0.0f);
  out->push_back(d.first_player == perspective_player ? 1.0f : 0.0f);

  // Perspective's own 3 reserved slots, full card detail regardless of
  // public visibility — perspective is the owner so they always know.
  for (int slot = 0; slot < 3; ++slot) {
    const bool exists = slot < d.reserved_size[perspective_player];
    if (!exists) {
      out->insert(out->end(), 13, 0.0f);
      continue;
    }
    const int cid =
        d.reserved[perspective_player][static_cast<size_t>(slot)];
    if (cid < 0 || cid >= static_cast<int>(cards.size())) {
      out->insert(out->end(), 13, 0.0f);
    } else {
      encode_card(&cards[static_cast<size_t>(cid)], out);
    }
  }
}

template <int NPlayers>
SplendorTokenEncoder<NPlayers>::SplendorTokenEncoder() {
  static_assert(Cfg::kBuyFaceupCount == 12 && Cfg::kReserveFaceupCount == 12);
  static_assert(Cfg::kReserveDeckCount == 3 && Cfg::kBuyReservedCount == 3);
  static_assert(Cfg::kPlayers <= kSeatSlots);
  static_assert(kGlobalFeatures <= kCardFeatures && kPlayerFeatures <= kCardFeatures &&
                kNobleFeatures <= kCardFeatures && kDeckFeatures <= kCardFeatures);
  layout_.num_tokens = kNumTokens;
  layout_.token_dim = kTokenDim;
  layout_.num_token_types = kTokenTypeCount;
  layout_.num_entity_ids = 1 + static_cast<int>(splendor_card_pool().size()) +
                           static_cast<int>(splendor_nobles().size());
  layout_.num_pointer_heads = kPointerHeadCount;

  layout_.token_types.assign(kNumTokens, kCardToken);
  layout_.token_types[0] = kGlobalToken;
  for (int p = 0; p < Cfg::kPlayers; ++p) layout_.token_types[kPlayerBase + p] = kPlayerToken;
  for (int i = 0; i < Cfg::kNobleCount; ++i) layout_.token_types[kNobleBase + i] = kNobleToken;
  for (int t = 0; t < 3; ++t) layout_.token_types[kDeckBase + t] = kDeckToken;

  layout_.action_token.assign(Cfg::kActionSpace, -1);
  layout_.action_head.assign(Cfg::kActionSpace, -1);
  auto bind = [this](int action, int token, int head) {
    layout_.action_token[static_cast<size_t>(action)] = token;
    layout_.action_head[static_cast<size_t>(action)] = head;
  };
  for (int i = 0; i < 12; ++i) {
    bind(Cfg::kBuyFaceupOffset + i, kTableauBase + i, kBuyHead);
    bind(Cfg::kReserveFaceupOffset + i, kTableauBase + i, kReserveFaceupHead);
  }
  for (int t = 0; t < 3; ++t) bind(Cfg::kReserveDeckOffset + t, kDeckBase + t, kReserveDeckHead);
  // Policy is only queried for the player to move, whose reserves are the
  // perspective-relative seat-0 block.
  for (int s = 0; s < 3; ++s) bind(Cfg::kBuyReservedOffset + s, kReservedBase + s, kBuyHead);
  for (int i = 0; i < Cfg::kChooseNobleCount; ++i) {
    bind(Cfg::kChooseNobleOffset + i, kNobleBase + i, kChooseNobleHead);
  }
}

template <int NPlayers>
void SplendorTokenEncoder<NPlayers>::encode_features(
    const IGameState& state,
    int perspective_player,
    const IBeliefTracker* /*tracker*/,
    std::vector<float>* out) const {
  const auto* s = dynamic_cast<const SplendorState<NPlayers>*>(&state);
  if (!s) throw std::runtime_error("SplendorTokenEncoder: state is not a SplendorState");
  if (!out) throw std::runtime_error("SplendorTokenEncoder: out is null");
  if (perspective_player < 0 || perspective_player >= Cfg::kPlayers) {
    throw std::runtime_error("SplendorTokenEncoder: perspective_player out of range");
  }
  const SplendorData<NPlayers>& d = s->persistent.data();
  const auto& cards = splendor_card_pool();
  const auto& nobles = splendor_nobles();
  const int num_cards = static_cast<int>(cards.size());
  const int me = perspective_player;
  auto rel = [me](int pid) { return (pid - me + Cfg::kPlayers) % Cfg::kPlayers; };

  const std::size_t base = out->size();
  out->resize(base + static_cast<std::size_t>(kFeatureDim), 0.0f);
  auto token = [&](int i) { return out->data() + base + static_cast<std::size_t>(i) * kTokenDim; };

  {
    float* t = token(0);
    t[0] = 1.0f;
    float* f = t + 2;
    for (int i = 0; i < kTokenTypes; ++i) {
      const float denom = (i == 5) ? 5.0f : static_cast<float>(Cfg::kGemCount);
      f[i] = static_cast<float>(d.bank[static_cast<size_t>(i)]) / denom;
    }
    f[6] = static_cast<float>(std::min(d.plies, kMaxPlies)) / static_cast<float>(kMaxPlies);
    f[7] = static_cast<float>(std::max(0, d.pending_returns)) / 5.0f;
    const auto stage = static_cast<SplendorTurnStage>(d.stage);
    f[8] = stage == SplendorTurnStage::kNormal ? 1.0f : 0.0f;
    f[9] = stage == SplendorTurnStage::kReturnTokens ? 1.0f : 0.0f;
    f[10] = stage == SplendorTurnStage::kChooseNoble ? 1.0f : 0.0f;
    f[11] = d.final_round_remaining >= 0 ? 1.0f : 0.0f;
    f[12] = d.current_player == me ? 1.0f : 0.0f;
  }

  for (int pid = 0; pid < Cfg::kPlayers; ++pid) {
    float* t = token(kPlayerBase + rel(pid));
    t[0] = 1.0f;
    float* f = t + 2;
    f[rel(pid)] = 1.0f;
    f[4] = d.current_player == pid ? 1.0f : 0.0f;
    f[5] = d.first_player == pid ? 1.0f : 0.0f;
    for (int i = 0; i < kTokenTypes; ++i) {
      f[6 + i] = static_cast<float>(d.player_gems[pid][static_cast<size_t>(i)]) / 10.0f;
    }
    for (int c = 0; c < kColorCount; ++c) {
      f[12 + c] = static_cast<float>(d.player_bonuses[pid][static_cast<size_t>(c)]) / 7.0f;
    }
    f[17] = static_cast<float>(d.player_points[pid]) / 20.0f;
    f[18] = static_cast<float>(d.reserved_size[pid]) / 3.0f;
    f[19] = static_cast<float>(d.player_cards_count[pid]) / 20.0f;
    f[20] = static_cast<float>(d.player_nobles_count[pid]) / 3.0f;
  }

  // Card token features: [0] in tableau, [1..4] owner seat (reserves only),
  // [5] face down, [6] identity unknown to perspective, [7..9] tier,
  // [10] points, [11..15] bonus color, [16..20] cost, [21..25] per-color
  // shortfall for `buyer` after bonuses and gems, [26] shortfall left after
  // gold, [27] affordable by `buyer`.
  auto write_card = [&](float* t, int cid, int owner_rel, bool face_down, int buyer) {
    if (cid < 0 || cid >= num_cards) {
      throw std::runtime_error("SplendorTokenEncoder: card id " + std::to_string(cid) + " out of range");
    }
    const SplendorCard& card = cards[static_cast<size_t>(cid)];
    t[0] = 1.0f;
    t[1] = static_cast<float>(1 + cid);
    float* f = t + 2;
    f[0] = owner_rel < 0 ? 1.0f : 0.0f;
    if (owner_rel >= 0) f[1 + owner_rel] = 1.0f;
    f[5] = face_down ? 1.0f : 0.0f;
    f[7 + card.tier - 1] = 1.0f;
    f[10] = static_cast<float>(card.points) / 5.0f;
    f[11 + card.bonus] = 1.0f;
    int shortfall = 0;
    for (int c = 0; c < kColorCount; ++c) {
      const int cost = card.cost[static_cast<size_t>(c)];
      f[16 + c] = static_cast<float>(cost) / 7.0f;
      const int need = std::max(0, cost - static_cast<int>(d.player_bonuses[buyer][static_cast<size_t>(c)]));
      const int miss = std::max(0, need - static_cast<int>(d.player_gems[buyer][static_cast<size_t>(c)]));
      f[21 + c] = static_cast<float>(miss) / 7.0f;
      shortfall += miss;
    }
    const int after_gold = std::max(0, shortfall - static_cast<int>(d.player_gems[buyer][5]));
    f[26] = static_cast<float>(after_gold) / 10.0f;
    f[27] = after_gold == 0 ? 1.0f : 0.0f;
  };

  for (int tier = 0; tier < 3; ++tier) {
    for (int slot = 0; slot < d.tableau_size[static_cast<size_t>(tier)]; ++slot) {
      write_card(token(kTableauBase + tier * 4 + slot),
                 d.tableau[static_cast<size_t>(tier)][static_cast<size_t>(slot)],
                 /*owner_rel=*/-1, /*face_down=*/false, /*buyer=*/me);
    }
  }

  for (int pid = 0; pid < Cfg::kPlayers; ++pid) {
    const int r = rel(pid);
    for (int slot = 0; slot < d.reserved_size[pid]; ++slot) {
      float* t = token(kReservedBase + r * 3 + slot);
      const int cid = d.reserved[pid][static_cast<size_t>(slot)];
      const bool face_down = d.reserved_visible[pid][static_cast<size_t>(slot)] == 0;
      if (cid != kHiddenCardId) {
        write_card(t, cid, r, face_down, /*buyer=*/pid);
        continue;
      }
      const int tier_idx = d.reserved_tier[pid][static_cast<size_t>(slot)];
      if (tier_idx < 0 || tier_idx >= 3) {
        throw std::runtime_error("SplendorTokenEncoder: hidden reserve without a tier");
      }
      t[0] = 1.0f;
      float* f = t + 2;
      f[1 + r] = 1.0f;
      f[5] = 1.0f;
      f[6] = 1.0f;
      f[7 + tier_idx] = 1.0f;
    }
  }

  for (int i = 0; i < d.nobles_size; ++i) {
    const int nid = d.nobles[static_cast<size_t>(i)];
    if (nid < 0 || nid >= static_cast<int>(nobles.size())) {
      throw std::runtime_error("SplendorTokenEncoder: noble id " + std::to_string(nid) + " out of range");
    }
    float* t = token(kNobleBase + i);
    t[0] = 1.0f;
    t[1] = static_cast<float>(1 + num_cards + nid);
    float* f = t + 2;
    for (int c = 0; c < kColorCount; ++c) {
      const int req = nobles[static_cast<size_t>(nid)][static_cast<size_t>(c)];
      f[c] = static_cast<float>(req) / 4.0f;
      f[5 + c] = static_cast<float>(std::max(0, req - static_cast<int>(d.player_bonuses[me][static_cast<size_t>(c)]))) / 4.0f;
    }
  }

  for (int tier = 0; tier < 3; ++tier) {
    float* t = token(kDeckBase + tier);
    t[0] = 1.0f;
    float* f = t + 2;
    f[tier] = 1.0f;
    f[3] = static_cast<float>(d.deck_sizes[static_cast<size_t>(tier)]) / 40.0f;
  }
}

template <int NPlayers>
void SplendorBeliefTracker<NPlayers>::init(
    IGameState& state, int /*perspective*/, const AnyMap& payload) {
  // Perspective-agnostic: tracker holds only public card-multiset
  // aggregates. Per-perspective private knowledge (own reserved card
  // ids) is read from state.viz=1 slots in randomize_unseen, not here.
  //
  // Mid-game bootstrap: `pack_init_payload` ships the complete seen-card
  // set, so an `initial_observation` taken at any ply restores the same
  // memory a session would have built by streaming every action from the
  // start. Without it, re-deriving from the current tableau would miss
  // cards that already left the hidden pool (bought / already flipped),
  // and `randomize_unseen` would sample decks larger than `deck_sizes`.
  if (initialized_) return;
  seen_cards_.clear();
  initialized_ = true;

  auto seen_it = payload.find("seen_cards");
  if (seen_it != payload.end()) {
    for (int cid :
         any_int_list(seen_it->second, "SplendorBeliefTracker::init seen_cards")) {
      if (cid >= 0) seen_cards_.insert(cid);
    }
    return;
  }

  // Fallback (opening payload / selfplay): seed from the visible tableau.
  // Tableau is schema field "tableau" with shape {3, 4} and base
  // viz=all_public — every slot is visible to every perspective. Read
  // each slot via read_field_slot. Empty slots return -1 (per
  // write_field_slot: tableau slots beyond tableau_size carry -1).
  const viz::FieldId fid_tableau_size =
      viz::schema_lookup(state.schema_ref(), "tableau_size");
  const viz::FieldId fid_tableau =
      viz::schema_lookup(state.schema_ref(), "tableau");
  for (int t = 0; t < 3; ++t) {
    const int size = std::any_cast<int>(
        state.read_field_slot(fid_tableau_size, {t}));
    for (int slot = 0; slot < size; ++slot) {
      const int cid = std::any_cast<int>(
          state.read_field_slot(fid_tableau, {t, slot}));
      if (cid >= 0) seen_cards_.insert(cid);
    }
  }
  // Nobles are public but aren't "cards" for the deck-pool belief.
  // Reserved slots at game start are all empty — nothing to add.
}

template <int NPlayers>
AnyMap SplendorBeliefTracker<NPlayers>::pack_init_payload(
    const IGameState& gt_state, int perspective) const {
  // Complete seen-card restore. `seen_cards_` accumulates every card that
  // has publicly left the hidden pool: tableau arrivals (deck_flip), visible
  // reserved cards, and purchased cards (including blind reserves whose
  // identity is revealed when bought). Rather than replay that history, the
  // GT state already partitions the pool: a card is still in the hidden pool
  // iff it sits in a deck or in a face-down reserve slot. Everything else is
  // public. Face-down reserves are excluded for every seat — the streaming
  // tracker never adds them either; the owner's own blind reserve is carried
  // by state viz and consumed by randomize_unseen's observer_known set.
  const auto* s = dynamic_cast<const SplendorState<NPlayers>*>(&gt_state);
  if (!s || perspective < 0 || perspective >= NPlayers) return {};
  const auto& d = s->persistent.data();
  const auto& cards = splendor_card_pool();
  const int total = static_cast<int>(cards.size());

  std::vector<char> unseen(static_cast<size_t>(total), 0);
  for (int t = 0; t < 3; ++t) {
    for (int cid : d.decks[static_cast<size_t>(t)]) {
      if (cid >= 0 && cid < total) unseen[static_cast<size_t>(cid)] = 1;
    }
  }

  // Face-down reserves are not "seen" in the tracker's sense — the
  // streaming tracker never adds them (a blind reserve's identity is the
  // perspective's own private knowledge, carried by state viz and consumed
  // by randomize_unseen's observer_known set, not by seen_cards_). Face-up
  // reserves came off a public tableau slot, so they are already seen.
  for (int p = 0; p < NPlayers; ++p) {
    for (int slot = 0; slot < d.reserved_size[static_cast<size_t>(p)]; ++slot) {
      if (d.reserved_visible[static_cast<size_t>(p)][static_cast<size_t>(slot)] != 0) {
        continue;
      }
      const int cid = d.reserved[static_cast<size_t>(p)][static_cast<size_t>(slot)];
      if (cid >= 0 && cid < total) unseen[static_cast<size_t>(cid)] = 1;
    }
  }

  std::vector<int> seen;
  seen.reserve(static_cast<size_t>(total));
  for (int cid = 0; cid < total; ++cid) {
    if (!unseen[static_cast<size_t>(cid)]) seen.push_back(cid);
  }
  AnyMap out;
  out["seen_cards"] = std::any(std::move(seen));
  return out;
}

template <int NPlayers>
void SplendorBeliefTracker<NPlayers>::observe_public_event(
    int /*actor*/,
    ActionId /*action*/,
    const std::vector<PublicEvent>& events) {
  // deck_flip events carry tableau card IDs revealed by drawing
  // from the deck to replace a bought/reserved card. reserved_card_bought
  // carries the id of a reserved card leaving the reserve row face-up.
  // Both are public to every observer, so accumulate unconditionally.
  //
  // No `self_reserve_deck` branch: the actor's new reserve cid is
  // already on state.reserved with viz=1 for the owner — owners read
  // it through state, not through the tracker. Other observers learn
  // nothing from a blind reserve, which is correct.
  for (const auto& ev : events) {
    if (ev.first == "deck_flip" || ev.first == "reserved_card_bought") {
      const int cid = std::any_cast<int>(ev.second.at("card_id"));
      if (cid >= 0) seen_cards_.insert(cid);
    }
  }
}

// randomize_unseen produces a world whose public fields are byte-equal
// across any two trackers with the same observation history, regardless
// of the input state's hidden contents. Called per-sim at MCTS root for
// determinization, AND at the end of each apply_observation to re-sample
// session state_'s hidden fields into a fresh tracker-consistent world.
//
// Canonical unseen-pool formula:
//   unseen_pool = full_pool − seen_cards − observer's viz=1 reserved
//
// Algorithm:
//   1. unseen_by_tier[t] = pool[t] − seen_cards − observer-known reserved
//      (the observer's own reserved card ids, found via
//      state.viz_["reserved"][p, slot, observer] == 1).
//   2. For every reserved slot the observer cannot see (viz=0): consume
//      one card from the matching tier of unseen_by_tier. The tier comes
//      from the public schema field `reserved_tier[p, slot]` (set by
//      rules at reserve-from-deck / reserve-from-tableau time), NOT from
//      the slot's current cid — across sims the cid is whatever a prior
//      sim's randomize_unseen left there, so reverse-deriving the tier
//      from `cards[cid].tier` produced inconsistent deck-size reductions
//      between sims sharing the same DAG node, which let
//      `decks[t].empty()` disagree at the same hash and split the
//      legal-action set (BUG: action-24 mismatch).
//   3. Remaining unseen_by_tier[t] → data.decks[t]. Size is exactly
//      |unseen_by_tier[t]| − (observer-hidden reserves of tier t),
//      which equals truth's deck size by construction.
template <int NPlayers>
void SplendorBeliefTracker<NPlayers>::randomize_unseen(IGameState& state, int observer, std::mt19937_64& rng) const {
  auto* s = dynamic_cast<SplendorState<NPlayers>*>(&state);
  if (!s) return;

  SplendorData<NPlayers> data = s->persistent.data();
  const auto& cards = splendor_card_pool();
  const int total_cards = static_cast<int>(cards.size());

  // Observer-known reserved cids: state.viz_["reserved"][p, slot, observer]==1.
  // These are already-fixed in the produced world and must be excluded
  // from the unseen pool so we don't double-deal them.
  const auto& reserved_viz = viz::viz_get(state, "reserved");
  std::unordered_set<int> observer_known;
  observer_known.reserve(static_cast<size_t>(Cfg::kPlayers * 3));
  auto is_visible_to_observer = [&](int p, int slot) -> bool {
    const std::vector<int> idx_slot{p, slot};
    const std::size_t base = viz::flat_offset_data_only(reserved_viz.shape, idx_slot);
    return reserved_viz.data[base + static_cast<std::size_t>(observer)] != 0;
  };
  for (int p = 0; p < Cfg::kPlayers; ++p) {
    for (int slot = 0; slot < data.reserved_size[p]; ++slot) {
      if (is_visible_to_observer(p, slot)) {
        const int cid = data.reserved[p][static_cast<size_t>(slot)];
        if (cid >= 0 && cid < total_cards) observer_known.insert(cid);
      }
    }
  }

  std::array<std::vector<int>, 3> unseen_by_tier{};
  for (int cid = 0; cid < total_cards; ++cid) {
    if (seen_cards_.count(cid) != 0) continue;
    if (observer_known.count(cid) != 0) continue;
    const int tier_idx = cards[static_cast<size_t>(cid)].tier - 1;
    if (tier_idx >= 0 && tier_idx < 3) {
      unseen_by_tier[static_cast<size_t>(tier_idx)].push_back(cid);
    }
  }

  for (int t = 0; t < 3; ++t) {
    std::shuffle(unseen_by_tier[static_cast<size_t>(t)].begin(),
                 unseen_by_tier[static_cast<size_t>(t)].end(), rng);
  }

  std::array<size_t, 3> idx{0, 0, 0};
  for (int p = 0; p < Cfg::kPlayers; ++p) {
    for (int slot = 0; slot < data.reserved_size[p]; ++slot) {
      if (is_visible_to_observer(p, slot)) continue;  // observer-known, keep
      // Tier comes from the public reserved_tier slot, not from the
      // (potentially stale across sims) cid in `reserved`.
      const int tier_idx = data.reserved_tier[p][static_cast<size_t>(slot)];
      if (tier_idx < 0 || tier_idx >= 3) continue;
      auto& pool = unseen_by_tier[static_cast<size_t>(tier_idx)];
      auto& i = idx[static_cast<size_t>(tier_idx)];
      if (i >= pool.size()) {
        throw std::runtime_error(
            "SplendorBeliefTracker::randomize_unseen: unseen tier-" +
            std::to_string(tier_idx + 1) +
            " pool exhausted while dealing hidden reserves "
            "(tracker accounting disagrees with state)");
      }
      data.reserved[p][static_cast<size_t>(slot)] =
          static_cast<std::int16_t>(pool[i++]);
    }
  }

  // Deck content = remaining unseen pool per tier. Size is determined by
  // the (pool - seen - opp_hidden) formula, not by preservation of the
  // input state's deck size.
  for (int t = 0; t < 3; ++t) {
    auto& pool = unseen_by_tier[static_cast<size_t>(t)];
    auto& i = idx[static_cast<size_t>(t)];
    auto& deck = data.decks[static_cast<size_t>(t)];
    deck.clear();
    deck.reserve(pool.size() - i);
    for (; i < pool.size(); ++i) {
      deck.push_back(static_cast<std::int16_t>(pool[i]));
    }
    if (static_cast<int>(deck.size()) != data.deck_sizes[static_cast<size_t>(t)]) {
      throw std::runtime_error(
          "SplendorBeliefTracker::randomize_unseen: sampled tier-" +
          std::to_string(t + 1) + " deck has " + std::to_string(deck.size()) +
          " cards but public deck_sizes says " +
          std::to_string(data.deck_sizes[static_cast<size_t>(t)]));
    }
  }

  // Caller-owned rng now drives all subsequent draws via the
  // do_action_fast(rng) path; nothing on state to reseed.
  (void)rng;

  auto node = std::make_shared<SplendorPersistentNode<NPlayers>>();
  node->action_from_parent = -1;
  node->materialized = std::make_shared<const SplendorData<NPlayers>>(std::move(data));
  s->persistent = SplendorPersistentState<NPlayers>(node);
  s->undo_stack.clear();
}

template <int NPlayers>
AnyMap SplendorBeliefTracker<NPlayers>::serialize() const {
  AnyMap out;
  // Canonical form: sorted vector of seen card IDs (unordered_set iteration
  // order varies). Two trackers with the same seen set produce equal output.
  // Perspective-agnostic: trackers fed the same observation stream from
  // different seats produce equal output (no perspective_player field).
  std::vector<int> seen(seen_cards_.begin(), seen_cards_.end());
  std::sort(seen.begin(), seen.end());
  out["seen_cards"] = seen;
  out["initialized"] = initialized_;
  return out;
}

template class SplendorFeatureEncoder<2>;
template class SplendorFeatureEncoder<3>;
template class SplendorFeatureEncoder<4>;
template class SplendorTokenEncoder<2>;
template class SplendorTokenEncoder<3>;
template class SplendorTokenEncoder<4>;
template class SplendorBeliefTracker<2>;
template class SplendorBeliefTracker<3>;
template class SplendorBeliefTracker<4>;

}  // namespace board_ai::splendor
