#include "coup_net_adapter.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "../../engine/core/viz_runtime.h"

namespace board_ai::coup {

namespace {

// Maps a claim-bearing action to the character being claimed, or -1 if the
// action itself is not a claim.
int claim_role_for_action(ActionId action) {
  if (action == kTaxAction) return kDuke;
  if (action >= kAssassinateOffset && action < kAssassinateOffset + kAssassinateCount) return kAssassin;
  if (action >= kStealOffset && action < kStealOffset + kStealCount) return kCaptain;
  if (action == kExchangeAction) return kAmbassador;
  if (action == kBlockDukeAction) return kDuke;
  if (action == kBlockContessaAction) return kContessa;
  if (action == kBlockCaptainAction) return kCaptain;
  if (action == kBlockAmbassadorAction) return kAmbassador;
  return -1;
}

template <int NPlayers>
int influence_count(const CoupData<NPlayers>& d, int p) {
  int n = 0;
  if (!d.revealed[p][0]) ++n;
  if (!d.revealed[p][1]) ++n;
  return n;
}

int action_type_index(ActionId action) {
  if (action == kIncomeAction) return 0;
  if (action == kForeignAidAction) return 1;
  if (action >= kCoupOffset && action < kCoupOffset + kCoupCount) return 2;
  if (action == kTaxAction) return 3;
  if (action >= kAssassinateOffset && action < kAssassinateOffset + kAssassinateCount) return 4;
  if (action >= kStealOffset && action < kStealOffset + kStealCount) return 5;
  if (action == kExchangeAction) return 6;
  return -1;
}

// Two-card multiset table (Plan §5.1). 15 entries; m=0..14 corresponds
// to (R_a, R_b) with R_a ≤ R_b in role-id order:
//   m=0  (D,D)         m=5  (As,As)        m=10 (Cap,Amb)
//   m=1  (D,As)        m=6  (As,Cap)       m=11 (Cap,Con)
//   m=2  (D,Cap)       m=7  (As,Amb)       m=12 (Amb,Amb)
//   m=3  (D,Amb)       m=8  (As,Con)       m=13 (Amb,Con)
//   m=4  (D,Con)       m=9  (Cap,Cap)      m=14 (Con,Con)
struct TwoCardTable {
  std::array<std::array<int, kCharacterCount>, 15> need;
  std::array<std::array<int, kCharacterCount>, kCharacterCount> id;
  TwoCardTable() {
    for (auto& row : need) row.fill(0);
    int m = 0;
    for (int a = 0; a < kCharacterCount; ++a) {
      for (int b = a; b < kCharacterCount; ++b) {
        ++need[m][a];
        ++need[m][b];
        id[a][b] = m;
        id[b][a] = m;
        ++m;
      }
    }
  }
};
const TwoCardTable& two_card_table() {
  static const TwoCardTable t{};
  return t;
}

}  // namespace

template <int NPlayers>
const std::array<std::array<int, kCharacterCount>, 15>&
CoupBeliefTracker<NPlayers>::need_2card() {
  return two_card_table().need;
}

template <int NPlayers>
int CoupBeliefTracker<NPlayers>::multiset_id_2card(int a, int b) {
  return two_card_table().id[a][b];
}

// Masked-state read pattern (BUG-018 / encoder contract): the encoder
// reads slots through the perspective-masked clone. viz=1 slots carry
// truth, viz=0 slots carry placeholder sentinels (kPlaceholderInt8 =
// INT8_MIN for CharId/std::int8_t fields). The placeholder never
// equals any legitimate character id (0..4), so a `== c` test on a
// placeholder is always false — char-OH naturally encodes as all-zero
// for hidden slots without any explicit placeholder branch.

// Belief-net plan §16. Encoder rewrite — one consolidated per-player
// loop (no separate "private" pass), single global block, all
// hand-craft signals removed in favor of belief-net tracker raw counts.
// The masked clone handles privacy: viz=1 → truth, viz=0 → placeholder
// (kPlaceholderInt8 = INT8_MIN); `card == c` on placeholder is always
// false → char-OH zeros out. No `is_self` gate.
template <int NPlayers>
void CoupFeatureEncoder<NPlayers>::encode_features(
    const IGameState& masked_state,
    int perspective_player,
    const IBeliefTracker* tracker,
    std::vector<float>* out) const {
  const auto* s = dynamic_cast<const CoupState<NPlayers>*>(&masked_state);
  if (!s || !out || perspective_player < 0 ||
      perspective_player >= NPlayers) {
    return;
  }
  const auto& d = s->data;
  const auto* bt =
      dynamic_cast<const CoupBeliefTracker<NPlayers>*>(tracker);

  auto clamp_norm = [](int x) -> float {
    if (x < 0) return 0.0f;
    if (x > 4) return 1.0f;
    return static_cast<float>(x) * 0.25f;
  };

  // ---- Per-player block (46 each), rotated so perspective is index 0 ----
  for (int pi = 0; pi < NPlayers; ++pi) {
    const int pid = (perspective_player + pi) % NPlayers;

    // alive(1), coins/12(1)
    out->push_back(d.alive[pid] ? 1.0f : 0.0f);
    out->push_back(static_cast<float>(d.coins[pid]) / 12.0f);

    // 2 x { revealed?(1) + char-OH(5) } = 12.
    // The masked clone makes this work uniformly: viz=1 (own unrevealed
    // + any-perspective revealed) carries truth; viz=0 (other
    // perspective unrevealed) carries placeholder → char-OH zeros.
    // `revealed?` flag is always public.
    for (int sl = 0; sl < 2; ++sl) {
      const bool rev = d.revealed[pid][sl];
      out->push_back(rev ? 1.0f : 0.0f);
      const std::int8_t card = d.influence[pid][sl];
      for (int c = 0; c < kCharacterCount; ++c) {
        out->push_back(card == c ? 1.0f : 0.0f);
      }
    }

    // role flags (4): active / target / blocker / challenger.
    out->push_back(d.active_player == pid ? 1.0f : 0.0f);
    out->push_back(d.action_target == pid ? 1.0f : 0.0f);
    out->push_back(d.blocker == pid ? 1.0f : 0.0f);
    out->push_back(d.challenger == pid ? 1.0f : 0.0f);

    // 4 x belief-net raw counts (5 each = 20).
    for (int c = 0; c < kCharacterCount; ++c) {
      out->push_back(bt ? clamp_norm(bt->pre_claim_count(pid, c)) : 0.0f);
    }
    for (int c = 0; c < kCharacterCount; ++c) {
      out->push_back(bt ? clamp_norm(bt->post_claim_count(pid, c)) : 0.0f);
    }
    for (int c = 0; c < kCharacterCount; ++c) {
      out->push_back(
          bt ? clamp_norm(bt->pre_challenge_initiated(pid, c)) : 0.0f);
    }
    for (int c = 0; c < kCharacterCount; ++c) {
      out->push_back(
          bt ? clamp_norm(bt->post_challenge_initiated(pid, c)) : 0.0f);
    }

    // last_reshuffle_kind OH (3): {None, Exchange, RevealTruthful}.
    int kind = 0;
    if (bt) kind = static_cast<int>(bt->last_reshuffle_kind(pid));
    for (int j = 0; j < 3; ++j) {
      out->push_back(kind == j ? 1.0f : 0.0f);
    }
    // last_revealed_role OH (5). -1 unless kind == RevealTruthful.
    int rev_role = bt ? bt->last_revealed_role(pid) : -1;
    for (int c = 0; c < kCharacterCount; ++c) {
      out->push_back(rev_role == c ? 1.0f : 0.0f);
    }

    // exchange_complete_count / 4 clamped to [0, 1] (1). Counts the number
    // of times this player has successfully completed an Exchange
    // (Ambassador action, including the case where the claim was
    // challenged but proven truthful) — i.e. exchange_complete events.
    out->push_back(bt ? clamp_norm(bt->exchange_complete_count(pid)) : 0.0f);
  }

  // ---- Global block (47 + N) ----
  // stage OH (11)
  const int stage_idx = static_cast<int>(d.stage);
  for (int i = 0; i < kStageCount; ++i) {
    out->push_back(i == stage_idx ? 1.0f : 0.0f);
  }

  // declared_action_type OH (7).
  const int action_type = action_type_index(d.declared_action);
  constexpr int kActionTypeCount = 7;
  for (int i = 0; i < kActionTypeCount; ++i) {
    out->push_back(i == action_type ? 1.0f : 0.0f);
  }

  // claimed_character OH (5).
  for (int c = 0; c < kCharacterCount; ++c) {
    out->push_back(d.claimed_character == c ? 1.0f : 0.0f);
  }

  // block_character OH (5).
  for (int c = 0; c < kCharacterCount; ++c) {
    out->push_back(d.block_character == c ? 1.0f : 0.0f);
  }

  // pending_claimer relative-to-perspective OH (N). Tracker is
  // authoritative; `randomize_unseen` does not use this field but the
  // encoder does (see plan §16 table).
  for (int pi = 0; pi < NPlayers; ++pi) {
    const int pid = (perspective_player + pi) % NPlayers;
    const bool match = bt && bt->pending_claimer() == pid;
    out->push_back(match ? 1.0f : 0.0f);
  }

  // pending_challenged (1).
  out->push_back((bt && bt->pending_challenged()) ? 1.0f : 0.0f);

  // ply / 200 (1).
  constexpr float kMaxPlies = 200.0f;
  out->push_back(static_cast<float>(d.ply) / kMaxPlies);

  // remaining[R] (5): observer-derived unseen pool. Total 3 per role
  // minus publicly revealed minus observer's own unrevealed. Other
  // players' unrevealed slots are viz=0 → placeholder, contribute
  // nothing — exactly what the belief net is estimating.
  std::array<int, kCharacterCount> revealed_count{};
  for (int p = 0; p < NPlayers; ++p) {
    for (int sl = 0; sl < kInfluencePerPlayer; ++sl) {
      if (d.revealed[p][sl]) {
        const int role = static_cast<int>(d.influence[p][sl]);
        if (role >= 0 && role < kCharacterCount) {
          ++revealed_count[role];
        }
      }
    }
  }
  std::array<int, kCharacterCount> own_unrevealed{};
  for (int sl = 0; sl < kInfluencePerPlayer; ++sl) {
    if (!d.revealed[perspective_player][sl]) {
      const int role = static_cast<int>(d.influence[perspective_player][sl]);
      if (role >= 0 && role < kCharacterCount) {
        ++own_unrevealed[role];
      }
    }
  }
  // remaining[R] normalized by kCardsPerCharacter → [0, 1].
  constexpr float kInvCardsPerChar =
      1.0f / static_cast<float>(kCardsPerCharacter);
  for (int r = 0; r < kCharacterCount; ++r) {
    const int rem =
        kCardsPerCharacter - revealed_count[r] - own_unrevealed[r];
    out->push_back(static_cast<float>(std::max(0, rem)) * kInvCardsPerChar);
  }

  // 2 x exchange_drawn { occupied?(1) + char-OH(5) } = 12. The masked
  // clone's viz controlled by `reveal_slot_to(active)` — only during the two
  // ExchangeReturn stages on the active player's perspective does this
  // carry truth; otherwise placeholder → both `occupied?` and OH zero.
  // `occupied = (xd != placeholder && xd >= 0)`. The first half of the
  // condition catches viz=0 (placeholder = INT8_MIN, < 0); the second
  // catches the `-1` in-bounds sentinel rules use after Return1.
  for (int i = 0; i < kExchangeDrawSlots; ++i) {
    const std::int8_t xd = d.exchange_drawn[i];
    const bool occupied = xd >= 0;
    out->push_back(occupied ? 1.0f : 0.0f);
    for (int c = 0; c < kCharacterCount; ++c) {
      out->push_back((occupied && xd == c) ? 1.0f : 0.0f);
    }
  }
}

template <int NPlayers>
void CoupBeliefTracker<NPlayers>::init(
    IGameState& /*state*/, int perspective,
    const AnyMap& /*payload*/) {
  // Walker-driven init: `viz::apply_public_snapshot` already wrote the
  // public projection (every viz=1 slot's value, plus the perspective's
  // full viz slice) into `state` before this call — including the
  // perspective's owner_only influence slots, which ride directly on the
  // wire's per-slot value pairs. The tracker has no perspective-private
  // bootstrap to seed; just remember perspective for `randomize_unseen`
  // and reset signal memory.
  perspective_player_ = perspective;
  for (auto& row : pre_claim_counts_) row.fill(0);
  for (auto& row : post_claim_counts_) row.fill(0);
  for (auto& row : pre_challenge_initiated_) row.fill(0);
  for (auto& row : post_challenge_initiated_) row.fill(0);
  last_reshuffle_kind_.fill(ReshuffleKind::kNone);
  last_revealed_role_.fill(-1);
  exchange_complete_counts_.fill(0);
  pending_claimer_ = -1;
  pending_claim_role_ = -1;
  pending_challenged_ = false;
}

template <int NPlayers>
void CoupBeliefTracker<NPlayers>::observe_public_event(
    int actor,
    ActionId action,
    const std::vector<PublicEvent>& events) {
  // Tracker fed by two inputs:
  //   1. Action-level: claim openings (post_claim_counts_) and challenge
  //      declarations (post_challenge_initiated_).
  //   2. Event-level: stage-boundary events emitted by
  //      `coup_events::extract_events_only`. Reshuffle events
  //      (claim_resolved_truthful / block_resolved_truthful /
  //      exchange_complete) promote post→pre so the belief feature
  //      extractor can distinguish "evidence carried over a reshuffle"
  //      from "evidence accrued since".

  // ----- Action-level (claim opening + challenge declaration) -----
  const int claimed = claim_role_for_action(action);
  if (claimed >= 0) {
    pending_claimer_ = actor;
    pending_claim_role_ = claimed;
    pending_challenged_ = false;
    if (actor >= 0 && actor < NPlayers) {
      post_claim_counts_[actor][claimed] += 1;
    }
  } else if (action == kChallengeAction) {
    if (pending_claim_role_ >= 0 && actor >= 0 && actor < NPlayers) {
      post_challenge_initiated_[actor][pending_claim_role_] += 1;
    }
    pending_challenged_ = true;
  }

  // ----- Event-level -----
  for (const auto& evt : events) {
    const auto& payload = evt.second;
    if (evt.first == "claim_resolved_truthful") {
      auto it_c = payload.find("claimer");
      auto it_r = payload.find("role");
      if (it_c == payload.end() || it_r == payload.end()) continue;
      int c = std::any_cast<int>(it_c->second);
      int r = std::any_cast<int>(it_r->second);
      if (c < 0 || c >= NPlayers) continue;
      if (r < 0 || r >= kCharacterCount) continue;
      promote_post_to_pre(c);
      last_reshuffle_kind_[c] = ReshuffleKind::kRevealTruthful;
      last_revealed_role_[c] = static_cast<std::int8_t>(r);
      if (pending_claimer_ == c && pending_claim_role_ == r) {
        pending_claimer_ = -1;
        pending_claim_role_ = -1;
        pending_challenged_ = false;
      }
    } else if (evt.first == "block_resolved_truthful") {
      auto it_b = payload.find("blocker");
      auto it_r = payload.find("role");
      if (it_b == payload.end() || it_r == payload.end()) continue;
      int b = std::any_cast<int>(it_b->second);
      int r = std::any_cast<int>(it_r->second);
      if (b < 0 || b >= NPlayers) continue;
      if (r < 0 || r >= kCharacterCount) continue;
      promote_post_to_pre(b);
      last_reshuffle_kind_[b] = ReshuffleKind::kRevealTruthful;
      last_revealed_role_[b] = static_cast<std::int8_t>(r);
      if (pending_claimer_ == b && pending_claim_role_ == r) {
        pending_claimer_ = -1;
        pending_claim_role_ = -1;
        pending_challenged_ = false;
      }
    } else if (evt.first == "claim_unchallenged") {
      auto it_c = payload.find("claimer");
      auto it_r = payload.find("role");
      if (it_c == payload.end() || it_r == payload.end()) continue;
      int c = std::any_cast<int>(it_c->second);
      int r = std::any_cast<int>(it_r->second);
      if (c < 0 || c >= NPlayers) continue;
      if (r < 0 || r >= kCharacterCount) continue;
      if (pending_claimer_ == c && pending_claim_role_ == r) {
        pending_claimer_ = -1;
        pending_claim_role_ = -1;
        pending_challenged_ = false;
      }
    } else if (evt.first == "block_unchallenged") {
      auto it_b = payload.find("blocker");
      auto it_r = payload.find("role");
      if (it_b == payload.end() || it_r == payload.end()) continue;
      int b = std::any_cast<int>(it_b->second);
      int r = std::any_cast<int>(it_r->second);
      if (b < 0 || b >= NPlayers) continue;
      if (r < 0 || r >= kCharacterCount) continue;
      if (pending_claimer_ == b && pending_claim_role_ == r) {
        pending_claimer_ = -1;
        pending_claim_role_ = -1;
        pending_challenged_ = false;
      }
    } else if (evt.first == "exchange_complete") {
      auto it_p = payload.find("player");
      if (it_p == payload.end()) continue;
      int p = std::any_cast<int>(it_p->second);
      if (p < 0 || p >= NPlayers) continue;
      promote_post_to_pre(p);
      last_reshuffle_kind_[p] = ReshuffleKind::kExchange;
      last_revealed_role_[p] = -1;
      exchange_complete_counts_[p] += 1;
    }
  }
}

// randomize_unseen: viz-driven determinization. Reads which slots the
// observer can legitimately see from `state.viz_` (rules are sole viz
// writer), consumes those slots from the pool of remaining role counts,
// and weighted-samples the rest into viz=0 slots.
//
// Slot inventory:
//   - influence[p][s] (owner_only_first_axis): observer sees iff
//       viz["influence"][p, s, observer] == 1. Base owner_only_first_axis
//       gives viz[p, *, p] = 1 for the owner. Rules call reveal_slot on
//       lose-influence and on successful challenge → viz[p, s, *] = 1.
//   - exchange_drawn[i] (all_hidden base): observer sees iff
//       viz["exchange_drawn"][i, observer] == 1. Rules call
//       reveal_slot_to(active_player) during exchange.
//   - deck_count[c] (all_hidden): never visible. The pool's residual
//       distribution is written here; deck_size scalar is public and
//       preserved.
template <int NPlayers>
void CoupBeliefTracker<NPlayers>::randomize_unseen(
    IGameState& state, int observer, std::mt19937_64& rng) const {
  auto* s = dynamic_cast<CoupState<NPlayers>*>(&state);
  if (!s) return;
  auto& d = s->data;
  if (observer < 0 || observer >= NPlayers) return;

  const auto& influence_viz = viz::viz_get(state, "influence");
  const auto& exchange_viz = viz::viz_get(state, "exchange_drawn");
  const int n_viewers = influence_viz.viewer_count();
  if (observer >= n_viewers) return;

  auto influence_visible = [&](int p, int sl) -> bool {
    const std::size_t base = viz::flat_offset_data_only(
        influence_viz.shape, std::vector<int>{p, sl});
    return influence_viz.data[base + static_cast<std::size_t>(observer)] != 0;
  };
  auto exchange_visible = [&](int i) -> bool {
    const std::size_t base = viz::flat_offset_data_only(
        exchange_viz.shape, std::vector<int>{i});
    return exchange_viz.data[base + static_cast<std::size_t>(observer)] != 0;
  };

  // ------------------------------------------------------------------
  // Step 1: compute `remaining[role]` — copies of each role still
  // unaccounted for from the observer's perspective. Visible influence
  // and visible exchange_drawn slots consume from the pool; everything
  // else (including the entire deck and any opp's hidden slots) must
  // come out of the residual.
  // ------------------------------------------------------------------
  std::array<int, kCharacterCount> remaining{};
  for (int c = 0; c < kCharacterCount; ++c) {
    remaining[static_cast<size_t>(c)] = kCardsPerCharacter;
  }
  auto consume = [&](CharId card) {
    if (card >= 0 && card < kCharacterCount &&
        remaining[static_cast<size_t>(card)] > 0) {
      remaining[static_cast<size_t>(card)]--;
    }
  };

  for (int p = 0; p < NPlayers; ++p) {
    for (int sl = 0; sl < 2; ++sl) {
      if (influence_visible(p, sl)) consume(d.influence[p][sl]);
    }
  }
  // Exchange in-flight count comes from the public fields `stage` +
  // `exchange_held_count` rather than from `d.exchange_drawn[i]` itself.
  // Per DEC-003, viz=0 slots hold semantically-undefined bytes (stale
  // values left over from prior ply's apply_public_snapshot). Reading
  // `d.exchange_drawn[i] >= 0` on a viz=0 slot to decide if it's
  // "occupied" violates the contract: e.g. after kExchangeReturn1
  // sets the truth slot to -1 and resets viz, the session's stored
  // value (e.g. 0 = Duke from the prior ply) makes the slot look
  // occupied to the next sim's randomize_unseen — pool inconsistent.
  const bool exchange_active =
      (d.stage == CoupStage::kExchangeReturn1 ||
       d.stage == CoupStage::kExchangeReturn2);
  int drawn_in_flight = 0;
  if (exchange_active && d.active_player >= 0 && d.active_player < NPlayers) {
    drawn_in_flight =
        static_cast<int>(d.exchange_held_count) -
        influence_count(d, d.active_player);
    if (drawn_in_flight < 0) drawn_in_flight = 0;
    if (drawn_in_flight > kExchangeDrawSlots) {
      drawn_in_flight = kExchangeDrawSlots;
    }
  }
  int visible_drawn_count = 0;
  for (int i = 0; i < kExchangeDrawSlots; ++i) {
    if (exchange_active && exchange_visible(i) && d.exchange_drawn[i] >= 0) {
      consume(d.exchange_drawn[i]);
      ++visible_drawn_count;
    }
  }

  // ------------------------------------------------------------------
  // Step 2: enumerate hidden slots, grouped by owner.
  //   - For each opponent: the (≤ 2) viz=0 influence slots they hold +
  //     any in-flight exchange-drawn slot owned by an active opponent.
  //   - For the active player: their own in-flight exchange-drawn
  //     slots when they're the observer's opponent (covered above) or
  //     simply ignored when they're the observer themselves (observer-
  //     owned exchange slots are viz=1 to observer, won't be hidden).
  //   - Deck residual (deck_size copies, content unordered).
  // ------------------------------------------------------------------
  constexpr int kXBase = 100;
  struct OwnerSlots {
    int owner;
    std::vector<int> slot_idx;  // 0/1 for influence, kXBase+i for exchange
  };
  std::array<OwnerSlots, NPlayers> per_owner{};
  for (int p = 0; p < NPlayers; ++p) per_owner[p].owner = p;
  for (int p = 0; p < NPlayers; ++p) {
    for (int sl = 0; sl < 2; ++sl) {
      if (!influence_visible(p, sl)) {
        per_owner[p].slot_idx.push_back(sl);
      }
    }
  }
  // A drawn slot the observer cannot see holds an undefined value on a
  // session state (e.g. the observer's own slot emptied by Return1 still
  // carries the returned character). Reset it to the rules' "empty"
  // marker; slots that do hold a hidden in-flight card are refilled below.
  for (int i = 0; i < kExchangeDrawSlots; ++i) {
    if (!exchange_visible(i)) d.exchange_drawn[i] = -1;
  }
  if (exchange_active && drawn_in_flight > visible_drawn_count) {
    int hidden_to_assign = drawn_in_flight - visible_drawn_count;
    const int active = d.active_player;
    if (active >= 0 && active < NPlayers) {
      // Rules treat a slot as occupied iff the exchanging player sees it
      // (exchange_slot_live), so the hidden cards must go to exactly those
      // slots. That column of `exchange_drawn` is not synced to other
      // seats (VizSync::kReceiverOnly), which is fine: another seat's
      // exchange is never in flight at a search root — only on a truth
      // clone, where the column is exact.
      for (int i = 0; i < kExchangeDrawSlots && hidden_to_assign > 0; ++i) {
        const std::size_t base = viz::flat_offset_data_only(
            exchange_viz.shape, std::vector<int>{i});
        const bool live_for_active =
            exchange_viz.data[base + static_cast<std::size_t>(active)] != 0;
        if (!exchange_visible(i) && live_for_active) {
          per_owner[active].slot_idx.push_back(kXBase + i);
          --hidden_to_assign;
        }
      }
    }
    if (hidden_to_assign != 0) {
      throw std::logic_error(
          "CoupBeliefTracker::randomize_unseen: another seat's exchange is "
          "in flight but its drawn slots are not marked visible to it "
          "(observer=" + std::to_string(observer) + ")");
    }
  }

  // Total hidden slots (excluding deck) — used for the pool sanity check.
  int hidden_owner_slots = 0;
  for (int p = 0; p < NPlayers; ++p) {
    hidden_owner_slots += static_cast<int>(per_owner[p].slot_idx.size());
  }
  const int deck_size = static_cast<int>(d.deck_size);
  const int total_hidden_slots = hidden_owner_slots + deck_size;

  const int total_remaining =
      std::accumulate(remaining.begin(), remaining.end(), 0);
  if (total_remaining != total_hidden_slots) {
    throw std::logic_error(
        "CoupBeliefTracker::randomize_unseen: pool inconsistency "
        "(remaining=" + std::to_string(total_remaining) +
        ", hidden_slots=" + std::to_string(total_hidden_slots) +
        ", observer=" + std::to_string(observer) + ")");
  }

  // Observer's own hidden slots cannot exist — observer's influence is
  // viz=1 to themselves by construction.
  if (!per_owner[observer].slot_idx.empty()) {
    throw std::logic_error(
        "CoupBeliefTracker::randomize_unseen: hidden slot owned by "
        "observer (viz mis-tagged) — observer=" +
        std::to_string(observer));
  }

  // ------------------------------------------------------------------
  // Step 3: fill viz=0 slots from `remaining`.
  //
  // - Belief-net path (pi_valid_ && pi_observer_==observer): per-opp
  //   multiset categorical, branch selected by alive count
  //   (alive=2 → 15-way two-card, alive=1 → 5-way single), with
  //   feasibility mask `need[m][R] ≤ remaining[R]`. Belief is per-opp,
  //   not per-slot, so opps are processed independently.
  //
  // - Fallback (no belief net): just shuffle `remaining` over every
  //   hidden slot uniformly without replacement. No per-opp aggregation,
  //   no priors, no signals_ — every viz=0 slot draws from the same
  //   residual pool with weight = remaining[R].
  // ------------------------------------------------------------------
  const bool use_belief_net = pi_valid_ && pi_observer_ == observer;
  auto opp_index = [observer](int p) {
    return (p - observer - 1 + NPlayers) % NPlayers;
  };

  auto write_owner_slots = [&](int owner,
                               const std::array<int, kCharacterCount>& fills) {
    int next_role = 0;
    int remaining_copies = fills[next_role];
    for (int slot_idx : per_owner[owner].slot_idx) {
      while (remaining_copies == 0 && next_role + 1 < kCharacterCount) {
        ++next_role;
        remaining_copies = fills[next_role];
      }
      const CharId picked = static_cast<CharId>(next_role);
      --remaining_copies;
      if (slot_idx >= kXBase) {
        d.exchange_drawn[slot_idx - kXBase] = picked;
      } else {
        d.influence[owner][slot_idx] = picked;
      }
    }
  };

  // ------------------------------------------------------------------
  // Observer's certain knowledge of the court deck (CoupData::deck_known):
  // `known[c]` copies of c lie in the deck or in one of the `tainted`
  // slots. Every produced world must honor it — otherwise an opponent
  // gets dealt a card the observer itself put back into the deck. The
  // other seats' knowledge rows are hidden from the observer: clear them.
  // ------------------------------------------------------------------
  std::array<int, kCharacterCount> known{};
  int known_total = 0;
  for (int c = 0; c < kCharacterCount; ++c) {
    known[c] = std::max(0, static_cast<int>(d.deck_known[observer][c]));
    known_total += known[c];
  }
  std::vector<std::pair<int, int>> tainted;
  for (int q = 0; q < NPlayers; ++q) {
    if (q == observer) continue;
    for (int sl = 0; sl < kInfluencePerPlayer; ++sl) {
      if (d.deck_known_taint[observer][q][sl] > 0) tainted.emplace_back(q, sl);
    }
    d.deck_known[q].fill(0);
    for (auto& row : d.deck_known_taint[q]) row.fill(0);
  }

  // One complete draw of all hidden owner slots plus the deck residual
  // from the same pool. With `pin_known_in_deck` the known cards are
  // placed in the deck first (exact when no slot is tainted: they are
  // specific cards that never left the deck). (Body keeps the pre-existing
  // indentation of steps 3 and 4.)
  const std::array<int, kCharacterCount> pool = remaining;
  std::array<int, kCharacterCount> deck_fill{};
  auto sample_world = [&](bool pin_known_in_deck) {
  remaining = pool;
  deck_fill.fill(0);
  int deck_remaining = deck_size;
  if (pin_known_in_deck) {
    for (int c = 0; c < kCharacterCount; ++c) {
      if (known[c] > remaining[c]) {
        throw std::logic_error(
            "CoupBeliefTracker::randomize_unseen: deck_known exceeds the "
            "unseen pool (observer=" + std::to_string(observer) + ")");
      }
      remaining[c] -= known[c];
      deck_fill[c] = known[c];
    }
    if (known_total > deck_size) {
      throw std::logic_error(
          "CoupBeliefTracker::randomize_unseen: deck_known exceeds deck_size");
    }
    deck_remaining -= known_total;
  }

  if (use_belief_net) {
    std::vector<int> opp_order;
    opp_order.reserve(NPlayers - 1);
    for (int p = 0; p < NPlayers; ++p) {
      if (p == observer) continue;
      if (per_owner[p].slot_idx.empty()) continue;
      opp_order.push_back(p);
    }
    std::shuffle(opp_order.begin(), opp_order.end(), rng);

    const auto& need_2 = two_card_table().need;

    auto sample_branch = [&](const std::vector<double>& branch_logits,
                             const std::vector<bool>& feasible) -> int {
      double m = -std::numeric_limits<double>::infinity();
      for (size_t h = 0; h < branch_logits.size(); ++h) {
        if (feasible[h] && branch_logits[h] > m) m = branch_logits[h];
      }
      if (!std::isfinite(m)) return -1;
      std::vector<double> w(branch_logits.size(), 0.0);
      double total = 0.0;
      for (size_t h = 0; h < branch_logits.size(); ++h) {
        if (!feasible[h]) continue;
        w[h] = std::exp(branch_logits[h] - m);
        total += w[h];
      }
      if (!(total > 0.0)) return -1;
      std::uniform_real_distribution<double> dist(0.0, total);
      const double u = dist(rng);
      double acc = 0.0;
      for (size_t h = 0; h < branch_logits.size(); ++h) {
        acc += w[h];
        if (u < acc) return static_cast<int>(h);
      }
      for (int h = static_cast<int>(branch_logits.size()) - 1; h >= 0; --h) {
        if (w[h] > 0.0) return h;
      }
      return -1;
    };

    for (int owner : opp_order) {
      const int n_unknown = static_cast<int>(per_owner[owner].slot_idx.size());
      std::array<int, kCharacterCount> fills{};

      if (n_unknown == 2) {
        std::vector<double> logits(15, 0.0);
        std::vector<bool> feasible(15, true);
        const int oi = opp_index(owner);
        for (int h = 0; h < 15; ++h) {
          const float p = pi_hand_[oi][CoupBeliefTracker<NPlayers>::
                                          kBeliefBranch2 + h];
          logits[h] = std::log(static_cast<double>(p) + 1e-30);
        }
        for (int h = 0; h < 15; ++h) {
          for (int r = 0; r < kCharacterCount; ++r) {
            if (need_2[h][r] > remaining[r]) { feasible[h] = false; break; }
          }
        }
        const int picked = sample_branch(logits, feasible);
        if (picked < 0) {
          throw std::logic_error(
              "CoupBeliefTracker::randomize_unseen: no feasible 2-card "
              "multiset for owner=" + std::to_string(owner));
        }
        for (int r = 0; r < kCharacterCount; ++r) fills[r] = need_2[picked][r];
      } else if (n_unknown == 1) {
        std::vector<double> logits(kCharacterCount, 0.0);
        std::vector<bool> feasible(kCharacterCount, true);
        const int oi = opp_index(owner);
        for (int r = 0; r < kCharacterCount; ++r) {
          const float p = pi_hand_[oi][CoupBeliefTracker<NPlayers>::
                                          kBeliefBranch1 + r];
          logits[r] = std::log(static_cast<double>(p) + 1e-30);
        }
        for (int r = 0; r < kCharacterCount; ++r) {
          if (remaining[r] <= 0) feasible[r] = false;
        }
        const int picked = sample_branch(logits, feasible);
        if (picked < 0) {
          throw std::logic_error(
              "CoupBeliefTracker::randomize_unseen: no feasible single role "
              "for owner=" + std::to_string(owner));
        }
        fills[picked] = 1;
      } else {
        // n_unknown ∈ {3, 4}: only mid-Exchange. Belief head is
        // alive-conditioned at {1, 2}, so this branch falls through to
        // pool-uniform per-slot.
        for (int s = 0; s < n_unknown; ++s) {
          int total = 0;
          for (int r = 0; r < kCharacterCount; ++r) {
            total += remaining[r] - fills[r];
          }
          if (total <= 0) {
            throw std::logic_error(
                "CoupBeliefTracker::randomize_unseen: pool exhausted "
                "during n_unknown>2 fallback");
          }
          std::uniform_int_distribution<int> dist(0, total - 1);
          int u = dist(rng);
          int picked = 0;
          for (int r = 0; r < kCharacterCount; ++r) {
            const int avail = remaining[r] - fills[r];
            if (u < avail) { picked = r; break; }
            u -= avail;
          }
          fills[picked] += 1;
        }
      }

      for (int r = 0; r < kCharacterCount; ++r) remaining[r] -= fills[r];
      write_owner_slots(owner, fills);
    }
  } else {
    // No belief net → shuffle remaining over every hidden owner slot
    // uniformly without replacement. Per-opp aggregation is irrelevant
    // here; we just need per-slot draws.
    for (int p = 0; p < NPlayers; ++p) {
      if (p == observer) continue;
      std::array<int, kCharacterCount> fills{};
      for (int s = 0, n = static_cast<int>(per_owner[p].slot_idx.size());
           s < n; ++s) {
        int total = 0;
        for (int r = 0; r < kCharacterCount; ++r) total += remaining[r] - fills[r];
        if (total <= 0) {
          throw std::logic_error(
              "CoupBeliefTracker::randomize_unseen: pool exhausted "
              "during fallback shuffle (owner=" + std::to_string(p) + ")");
        }
        std::uniform_int_distribution<int> dist(0, total - 1);
        int u = dist(rng);
        int picked = 0;
        for (int r = 0; r < kCharacterCount; ++r) {
          const int avail = remaining[r] - fills[r];
          if (u < avail) { picked = r; break; }
          u -= avail;
        }
        fills[picked] += 1;
      }
      for (int r = 0; r < kCharacterCount; ++r) remaining[r] -= fills[r];
      write_owner_slots(p, fills);
    }
  }

  // ------------------------------------------------------------------
  // Step 4: deck residual — uniform-without-replacement from `remaining`.
  // We don't need per-position identity; deck is a multiset slot. Just
  // dump the residual into deck_count.
  // ------------------------------------------------------------------
  while (deck_remaining > 0) {
    int total = 0;
    for (int r = 0; r < kCharacterCount; ++r) total += remaining[r];
    if (total <= 0) {
      throw std::logic_error(
          "CoupBeliefTracker::randomize_unseen: deck residual underflow");
    }
    std::uniform_int_distribution<int> dist(0, total - 1);
    int u = dist(rng);
    int picked = 0;
    for (int r = 0; r < kCharacterCount; ++r) {
      if (u < remaining[r]) { picked = r; break; }
      u -= remaining[r];
    }
    --remaining[picked];
    ++deck_fill[picked];
    --deck_remaining;
  }
  };  // sample_world

  auto honors_known = [&]() {
    std::array<int, kCharacterCount> have = deck_fill;
    for (const auto& [q, sl] : tainted) {
      const int c = d.influence[q][sl];
      if (c >= 0 && c < kCharacterCount) ++have[c];
    }
    // Cards another seat has drawn for an exchange still in flight came
    // straight from the deck (never a search root; truth clones only).
    if (exchange_active && d.active_player != observer) {
      for (int slot_idx : per_owner[d.active_player].slot_idx) {
        if (slot_idx < kXBase) continue;
        const int c = d.exchange_drawn[slot_idx - kXBase];
        if (c >= 0 && c < kCharacterCount) ++have[c];
      }
    }
    for (int c = 0; c < kCharacterCount; ++c) {
      if (known[c] > have[c]) return false;
    }
    return true;
  };

  if (known_total == 0) {
    sample_world(false);
  } else if (tainted.empty()) {
    sample_world(true);
  } else {
    // A known card may have moved into a tainted slot: sample freely and
    // keep the first world that honors the claim. If none turns up, pin
    // the known cards in the deck — always a consistent world.
    constexpr int kMaxRejections = 64;
    bool ok = false;
    for (int attempt = 0; attempt < kMaxRejections && !ok; ++attempt) {
      sample_world(false);
      ok = honors_known();
    }
    if (!ok) sample_world(true);
  }

  for (int c = 0; c < kCharacterCount; ++c) {
    d.deck_count[static_cast<size_t>(c)] =
        static_cast<std::int8_t>(deck_fill[static_cast<size_t>(c)]);
  }
}

template <int NPlayers>
void CoupBeliefTracker<NPlayers>::prepare_for_root(
    const IGameState& root_state,
    int root_player,
    const IBeliefFeatureExtractor* extractor,
    const IBeliefEvaluator* evaluator) {
  // Plan §2.1: extract per-perspective features, run the belief net,
  // softmax(/T) per opp row, cache pi_ in this session-shared tracker.
  // sim_tracker (clone) inherits pi_ by value — each sim reads it
  // during `randomize_unseen` without re-evaluating the network.
  pi_valid_ = false;
  pi_observer_ = -1;
  for (auto& row : pi_hand_) row.fill(0.0f);

  if (!extractor || !evaluator) return;
  if (root_player < 0 || root_player >= NPlayers) return;
  if (NPlayers <= 1) return;

  std::vector<float> features;
  if (!extractor->extract_from_state(root_state, root_player, this,
                                     &features)) {
    return;
  }
  std::vector<float> logits;
  if (!evaluator->evaluate(features, &logits)) return;
  constexpr int kHandDim = kBeliefHandDim;
  const int expected = (NPlayers - 1) * kHandDim;
  if (static_cast<int>(logits.size()) != expected) return;

  // Per-opp two independent branch softmaxes (multiset Plan §5):
  // [0..14] is the two-card multiset head; [15..19] is the single-card
  // head. Each branch is normalized along its own axis at fixed temperature 1.
  // randomize_unseen reads only the branch matching that opp's current
  // alive count, so the two normalizers don't need to be commensurate.
  const float inv_T = 1.0f / board_ai::kBeliefTemperature;
  auto softmax_branch = [&](int opp, int base, int len) -> bool {
    float m = -std::numeric_limits<float>::infinity();
    for (int k = 0; k < len; ++k) {
      const float z = logits[opp * kHandDim + base + k] * inv_T;
      if (z > m) m = z;
    }
    if (!std::isfinite(m)) return false;
    float sum = 0.0f;
    std::array<float, std::max(kBeliefBranch1Size, kBeliefBranch2Size)> ez{};
    for (int k = 0; k < len; ++k) {
      const float z = logits[opp * kHandDim + base + k] * inv_T;
      ez[k] = std::exp(z - m);
      sum += ez[k];
    }
    if (sum <= 0.0f || !std::isfinite(sum)) return false;
    for (int k = 0; k < len; ++k) {
      pi_hand_[opp][base + k] = ez[k] / sum;
    }
    return true;
  };
  for (int i = 0; i < NPlayers - 1; ++i) {
    if (!softmax_branch(i, kBeliefBranch2, kBeliefBranch2Size)) return;
    if (!softmax_branch(i, kBeliefBranch1, kBeliefBranch1Size)) return;
  }
  pi_valid_ = true;
  pi_observer_ = root_player;
}

template <int NPlayers>
AnyMap CoupBeliefTracker<NPlayers>::serialize() const {
  // Canonical, perspective-agnostic dump used by sim_tracker /
  // perspective-invariance tests. Tracker content is a public-event
  // derivation, so two trackers fed the same observation stream from
  // different seats must produce equal output.
  AnyMap out;
  auto flatten = [&](const auto& mat) {
    std::vector<int> flat;
    flat.reserve(static_cast<size_t>(NPlayers * kCharacterCount));
    for (int p = 0; p < NPlayers; ++p) {
      for (int c = 0; c < kCharacterCount; ++c) {
        flat.push_back(mat[p][c]);
      }
    }
    return flat;
  };
  out["pre_claim_counts"] = flatten(pre_claim_counts_);
  out["post_claim_counts"] = flatten(post_claim_counts_);
  out["pre_challenge_initiated"] = flatten(pre_challenge_initiated_);
  out["post_challenge_initiated"] = flatten(post_challenge_initiated_);
  std::vector<int> kinds;
  kinds.reserve(NPlayers);
  for (int p = 0; p < NPlayers; ++p) {
    kinds.push_back(static_cast<int>(last_reshuffle_kind_[p]));
  }
  out["last_reshuffle_kind"] = kinds;
  std::vector<int> roles;
  roles.reserve(NPlayers);
  for (int p = 0; p < NPlayers; ++p) {
    roles.push_back(static_cast<int>(last_revealed_role_[p]));
  }
  out["last_revealed_role"] = roles;
  std::vector<int> exch_counts;
  exch_counts.reserve(NPlayers);
  for (int p = 0; p < NPlayers; ++p) {
    exch_counts.push_back(exchange_complete_counts_[p]);
  }
  out["exchange_complete_counts"] = exch_counts;
  out["pending_claimer"] = pending_claimer_;
  out["pending_claim_role"] = pending_claim_role_;
  out["pending_challenged"] = pending_challenged_;
  return out;
}

// CoupBeliefFeatureExtractor — Plan §1.3.
//
// Reads the perspective-masked state + CoupBeliefTracker and emits a
// flat feature vector in the layout documented in coup_net_adapter.h.
// All inputs are public derivations; viz=0 slots arrive as
// kPlaceholderInt8 (never compared).
template <int NPlayers>
void CoupBeliefFeatureExtractor<NPlayers>::extract(
    const IGameState& masked_state,
    int perspective_player,
    const IBeliefTracker* tracker,
    std::vector<float>* out) const {
  const auto* s = dynamic_cast<const CoupState<NPlayers>*>(&masked_state);
  if (!s || !out || perspective_player < 0 ||
      perspective_player >= NPlayers) {
    return;
  }
  const auto& d = s->data;
  const auto* bt =
      dynamic_cast<const CoupBeliefTracker<NPlayers>*>(tracker);
  out->clear();
  out->reserve(static_cast<size_t>(kFeatureDim));

  // --- Global block (6) ---
  // remaining[R]: total(3) − publicly revealed R − observer's own
  //              unrevealed R slots. All other slots are viz=0 to
  //              observer and contribute nothing (would-be R cards in
  //              opps' hands are exactly what the belief net is
  //              estimating).
  std::array<int, kCharacterCount> revealed_count{};
  for (int p = 0; p < NPlayers; ++p) {
    for (int sl = 0; sl < kInfluencePerPlayer; ++sl) {
      if (d.revealed[p][sl]) {
        const int role = static_cast<int>(d.influence[p][sl]);
        if (role >= 0 && role < kCharacterCount) {
          ++revealed_count[role];
        }
      }
    }
  }
  std::array<int, kCharacterCount> own_unrevealed{};
  for (int sl = 0; sl < kInfluencePerPlayer; ++sl) {
    if (!d.revealed[perspective_player][sl]) {
      const int role = static_cast<int>(d.influence[perspective_player][sl]);
      if (role >= 0 && role < kCharacterCount) {
        ++own_unrevealed[role];
      }
    }
  }
  for (int r = 0; r < kCharacterCount; ++r) {
    const int remaining =
        kCardsPerCharacter - revealed_count[r] - own_unrevealed[r];
    out->push_back(static_cast<float>(std::max(0, remaining)));
  }
  out->push_back(static_cast<float>(d.ply) / 200.0f);

  // --- Per-opp block (28 × (N-1)) ---
  // Walk i ∈ [0, N-2]; opp = (perspective + 1 + i) mod N. Order matches
  // the belief net's output rows (Plan §1.2.1 opp_to_player).
  auto clamp_norm = [](int x) -> float {
    if (x < 0) return 0.0f;
    if (x > 4) return 1.0f;
    return static_cast<float>(x) * 0.25f;  // /4
  };
  for (int i = 0; i < NPlayers - 1; ++i) {
    const int opp = (perspective_player + 1 + i) % NPlayers;

    // pre_claim_counts (5)
    for (int r = 0; r < kCharacterCount; ++r) {
      out->push_back(bt ? clamp_norm(bt->pre_claim_count(opp, r)) : 0.0f);
    }
    // post_claim_counts (5)
    for (int r = 0; r < kCharacterCount; ++r) {
      out->push_back(bt ? clamp_norm(bt->post_claim_count(opp, r)) : 0.0f);
    }
    // pre_challenge_initiated (5)
    for (int r = 0; r < kCharacterCount; ++r) {
      out->push_back(
          bt ? clamp_norm(bt->pre_challenge_initiated(opp, r)) : 0.0f);
    }
    // post_challenge_initiated (5)
    for (int r = 0; r < kCharacterCount; ++r) {
      out->push_back(
          bt ? clamp_norm(bt->post_challenge_initiated(opp, r)) : 0.0f);
    }
    // last_reshuffle_kind one-hot (3): {None, Exchange, RevealTruthful}
    int kind = 0;
    if (bt) {
      const auto k = bt->last_reshuffle_kind(opp);
      kind = static_cast<int>(k);
    }
    for (int j = 0; j < 3; ++j) {
      out->push_back(kind == j ? 1.0f : 0.0f);
    }
    // last_revealed_role one-hot (5). Only valid when kind ==
    // RevealTruthful; otherwise tracker stores -1 → all zero.
    int rev_role = bt ? bt->last_revealed_role(opp) : -1;
    for (int r = 0; r < kCharacterCount; ++r) {
      out->push_back(rev_role == r ? 1.0f : 0.0f);
    }
  }
}

// CoupBeliefLabelExtractor — Plan §2.2.
//
// Reads truth CoupState and emits the per-observer label tuple. Runs
// only on the GT runner (selfplay's emit path); never reachable from
// the AI session.
template <int NPlayers>
void CoupBeliefLabelExtractor<NPlayers>::extract(
    const IGameState& truth_state,
    int observer,
    std::vector<std::vector<int>>* out_hand_counts,
    std::vector<int>* out_remaining,
    std::vector<int>* out_alive) const {
  const auto& s = board_ai::checked_cast<CoupState<NPlayers>>(truth_state);
  const auto& d = s.data;

  out_hand_counts->assign(NPlayers - 1,
                          std::vector<int>(kCharacterCount, 0));
  out_remaining->assign(kCharacterCount, 0);
  out_alive->assign(NPlayers - 1, 0);

  // Public reveals across every seat.
  std::array<int, kCharacterCount> revealed_count{};
  for (int p = 0; p < NPlayers; ++p) {
    for (int sl = 0; sl < kInfluencePerPlayer; ++sl) {
      if (d.revealed[p][sl]) {
        const int role = static_cast<int>(d.influence[p][sl]);
        if (role >= 0 && role < kCharacterCount) {
          ++revealed_count[role];
        }
      }
    }
  }
  // Observer's own unrevealed influence.
  std::array<int, kCharacterCount> own_unrevealed{};
  for (int sl = 0; sl < kInfluencePerPlayer; ++sl) {
    if (!d.revealed[observer][sl]) {
      const int role = static_cast<int>(d.influence[observer][sl]);
      if (role >= 0 && role < kCharacterCount) {
        ++own_unrevealed[role];
      }
    }
  }
  for (int r = 0; r < kCharacterCount; ++r) {
    const int rem =
        kCardsPerCharacter - revealed_count[r] - own_unrevealed[r];
    (*out_remaining)[r] = std::max(0, rem);
  }

  // Per-opp hand_counts + alive flag. opp i = (observer + 1 + i) % N.
  for (int i = 0; i < NPlayers - 1; ++i) {
    const int opp = (observer + 1 + i) % NPlayers;
    int alive_slots = 0;
    for (int sl = 0; sl < kInfluencePerPlayer; ++sl) {
      if (!d.revealed[opp][sl]) {
        ++alive_slots;
        const int role = static_cast<int>(d.influence[opp][sl]);
        if (role >= 0 && role < kCharacterCount) {
          ++(*out_hand_counts)[i][role];
        }
      }
    }
    (*out_alive)[i] = alive_slots;  // 0/1/2 — multiset-aware (Plan §7).
  }
}

template class CoupFeatureEncoder<2>;
template class CoupFeatureEncoder<3>;
template class CoupFeatureEncoder<4>;
template class CoupBeliefTracker<2>;
template class CoupBeliefTracker<3>;
template class CoupBeliefTracker<4>;
template class CoupBeliefFeatureExtractor<2>;
template class CoupBeliefFeatureExtractor<3>;
template class CoupBeliefFeatureExtractor<4>;
template class CoupBeliefLabelExtractor<2>;
template class CoupBeliefLabelExtractor<3>;
template class CoupBeliefLabelExtractor<4>;

}  // namespace board_ai::coup
