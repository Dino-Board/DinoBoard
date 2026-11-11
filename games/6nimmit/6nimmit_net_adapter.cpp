#include "6nimmit_net_adapter.h"

#include <algorithm>
#include <any>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "../../engine/core/game_interfaces.h"
#include "../../engine/core/viz_runtime.h"
#include "../../engine/search/net_mcts.h"

namespace board_ai::sixnimmt {

template <int NPlayers>
void SixNimmtFeatureEncoder<NPlayers>::encode_features(
    const IGameState& state,
    int perspective_player,
    const IBeliefTracker* /*tracker*/,
    std::vector<float>* out) const {
  const auto* s = dynamic_cast<const SixNimmtState<NPlayers>*>(&state);
  if (!s || !out || perspective_player < 0 ||
      perspective_player >= NPlayers) {
    return;
  }
  const auto& d = s->data;
  out->reserve(static_cast<size_t>(Cfg::kFeatureDim));

  // ---- own_hand (§1.5): 104 ----
  // perspective sees own hand viz=1 → truth values 0/1; opp slots are
  // not read (we only emit perspective_player's row).
  for (int c = 0; c < kNumCards; ++c) {
    const std::int8_t v =
        d.hand[static_cast<size_t>(perspective_player)][static_cast<size_t>(c)];
    out->push_back(v == 1 ? 1.0f : 0.0f);
  }

  // ---- table dense (§1.5): 4 row-tail card_ids + 4 row lengths +
  //      4 row heads totals = 12. Normalize cards by 104, lengths by 5,
  //      heads by 30 (rough upper bound; ALGO §5 normalize-loosely-then-
  //      let-the-net-rescale).
  for (int r = 0; r < kNumRows; ++r) {
    int len = 0;
    int tail_card = 0;
    int heads = 0;
    for (int s_idx = 0; s_idx < kRowCapacity; ++s_idx) {
      int c = d.table[static_cast<size_t>(r)][static_cast<size_t>(s_idx)];
      if (c >= 1 && c <= kNumCards) {
        tail_card = c;
        heads += head_value(c);
        ++len;
      } else {
        break;
      }
    }
    out->push_back(static_cast<float>(tail_card) /
                   static_cast<float>(kNumCards));
    out->push_back(static_cast<float>(len) /
                   static_cast<float>(kRowCapacity));
    out->push_back(static_cast<float>(heads) / 30.0f);
  }

  // ---- bullhead_count[N] (§1.5): per-player accumulated heads,
  //      normalized by 30. Saves the net the head_value lookup.
  for (int pi = 0; pi < NPlayers; ++pi) {
    const int pid = (perspective_player + pi) % NPlayers;
    int heads = 0;
    for (int c = 1; c <= kNumCards; ++c) {
      if (d.bullhead_pile[static_cast<size_t>(pid)][static_cast<size_t>(c - 1)]) {
        heads += head_value(c);
      }
    }
    out->push_back(static_cast<float>(heads) / 30.0f);
  }

  // ---- meta: current_round one-hot (10) + small_phase_active (1)
  //           + small_player one-hot rotated own-first (N) ----
  for (int r = 0; r < kNumRounds; ++r) {
    out->push_back(d.current_round == r ? 1.0f : 0.0f);
  }
  out->push_back(d.small_phase_active ? 1.0f : 0.0f);
  for (int pi = 0; pi < NPlayers; ++pi) {
    const int pid = (perspective_player + pi) % NPlayers;
    out->push_back(d.small_player == pid ? 1.0f : 0.0f);
  }

  // ---- pending_commits per seat (rotated own-first): 1 hidden flag +
  //      104 one-hot per seat. Reads the masked clone — viz=0 slots are
  //      kPlaceholderInt8, viz=1 slots carry the revealed card_id (or 0
  //      if the slot was already consumed by the apply loop / not yet
  //      committed). The encoder treats placeholder as "hidden" (flag=1
  //      with all one-hot zero) and revealed-with-zero as "revealed but
  //      empty" (flag=0, one-hot zero).
  for (int pi = 0; pi < NPlayers; ++pi) {
    const int pid = (perspective_player + pi) % NPlayers;
    const std::int8_t pc =
        d.pending_commits[static_cast<size_t>(pid)];
    const bool hidden = (pc == kPlaceholderInt8);
    out->push_back(hidden ? 1.0f : 0.0f);
    for (int c = 1; c <= kNumCards; ++c) {
      const bool match = !hidden && (pc == static_cast<std::int8_t>(c));
      out->push_back(match ? 1.0f : 0.0f);
    }
  }
}

template <int NPlayers>
void SixNimmtBeliefTracker<NPlayers>::init(
    IGameState& state, int perspective, const AnyMap& payload) {
  if (perspective < 0 || perspective >= NPlayers) return;
  auto* s = dynamic_cast<SixNimmtState<NPlayers>*>(&state);
  if (!s) return;

  // payload "own_hand" is a vector<int> of card numbers (1..104) — used
  // only by the API / web path where session seed != truth seed. The
  // selfplay / arena path reseeds the session from truth's seed and
  // sends an empty payload (own hand already present).
  auto it_hand = payload.find("own_hand");
  if (it_hand != payload.end()) {
    if (it_hand->second.type() == typeid(std::vector<int>)) {
      const auto& cards = std::any_cast<const std::vector<int>&>(it_hand->second);
      // Wipe perspective's hand row first (independently-seeded session
      // will have a different bitmap), then write the cards.
      auto& row = s->data.hand[static_cast<size_t>(perspective)];
      row.fill(0);
      for (int c : cards) {
        if (c >= 1 && c <= kNumCards) {
          row[static_cast<size_t>(c - 1)] = 1;
        }
      }
    }
  }
  // viz for hand[perspective] is already viz=1 to perspective via base
  // owner_only_first_axis (init_viz seeded it). No reveal needed.
}

template <int NPlayers>
AnyMap SixNimmtBeliefTracker<NPlayers>::pack_init_payload(
    const IGameState& gt_state, int perspective) const {
  if (perspective < 0 || perspective >= NPlayers) return {};
  const auto* s = dynamic_cast<const SixNimmtState<NPlayers>*>(&gt_state);
  if (!s) return {};
  std::vector<int> cards;
  cards.reserve(kHandSize);
  for (int c = 1; c <= kNumCards; ++c) {
    if (s->data.hand[static_cast<size_t>(perspective)]
                     [static_cast<size_t>(c - 1)]) {
      cards.push_back(c);
    }
  }
  AnyMap out;
  out["own_hand"] = std::any(cards);
  return out;
}

template <int NPlayers>
void SixNimmtBeliefTracker<NPlayers>::observe_public_event(
    int /*actor*/, ActionId /*action*/,
    const std::vector<PublicEvent>& events) {
  // Public play history (`played_by_`) is the only memory the tracker
  // carries — every other public derivation lives on state. Consume
  // `card_played` events emitted by the events extractor.
  for (const auto& ev : events) {
    if (ev.first != "card_played") continue;
    const auto& payload = ev.second;
    auto it_p = payload.find("player");
    auto it_c = payload.find("card");
    if (it_p == payload.end() || it_c == payload.end()) continue;
    int p = std::any_cast<int>(it_p->second);
    int c = std::any_cast<int>(it_c->second);
    if (p < 0 || p >= NPlayers) continue;
    if (c < 1 || c > kNumCards) continue;
    played_by_[static_cast<size_t>(p)]
              [static_cast<size_t>(c - 1)] = 1;
  }
}

template <int NPlayers>
AnyMap SixNimmtBeliefTracker<NPlayers>::serialize() const {
  // Canonical form: per-player ordered list of played card ids. Used by
  // tracker-equivalence tests across selfplay / API paths.
  AnyMap out;
  std::vector<int> flat;
  flat.reserve(static_cast<size_t>(NPlayers * kNumCards));
  for (int p = 0; p < NPlayers; ++p) {
    for (int c = 1; c <= kNumCards; ++c) {
      if (played_by_[static_cast<size_t>(p)]
                    [static_cast<size_t>(c - 1)]) {
        flat.push_back(p * kNumCards + c);
      }
    }
  }
  out["played_flat"] = std::any(flat);
  return out;
}

template <int NPlayers>
void SixNimmtBeliefTracker<NPlayers>::randomize_unseen(
    IGameState& state, int observer, std::mt19937_64& rng) const {
  auto* s = dynamic_cast<SixNimmtState<NPlayers>*>(&state);
  if (!s) return;
  if (observer < 0 || observer >= NPlayers) return;
  auto& d = s->data;

  // unseen_pool = {1..104} \ table \ bullhead_pile[*] \ hand[observer]
  //               \ pending_commits[observer]
  //               \ ⋃_p played_by_[p]   (cards already played publicly)
  std::array<bool, kNumCards + 1> seen{};  // seen[card] for 1..104
  for (int r = 0; r < kNumRows; ++r) {
    for (int s_idx = 0; s_idx < kRowCapacity; ++s_idx) {
      int c = d.table[static_cast<size_t>(r)][static_cast<size_t>(s_idx)];
      if (c >= 1 && c <= kNumCards) seen[static_cast<size_t>(c)] = true;
    }
  }
  for (int p = 0; p < NPlayers; ++p) {
    for (int c = 1; c <= kNumCards; ++c) {
      if (d.bullhead_pile[static_cast<size_t>(p)][static_cast<size_t>(c - 1)]) {
        seen[static_cast<size_t>(c)] = true;
      }
    }
  }
  for (int c = 1; c <= kNumCards; ++c) {
    if (d.hand[static_cast<size_t>(observer)][static_cast<size_t>(c - 1)]) {
      seen[static_cast<size_t>(c)] = true;
    }
  }
  // Any pending slot viz=1 to observer holds truth — exclude from pool so
  // it isn't double-sampled into another opp's hand. Covers (a) observer's
  // own pre-seal pending (always viz=1 to self) and (b) opp pendings
  // post-seal in the SMALL / apply window where rules have flipped them
  // to all-public via reveal_slot.
  const auto& pc_viz = viz::viz_get(state, "pending_commits");
  auto pending_visible_to_observer = [&](int owner_p) -> bool {
    const std::size_t base = viz::flat_offset_data_only(
        pc_viz.shape, std::vector<int>{owner_p});
    return pc_viz.data[base + static_cast<std::size_t>(observer)] != 0;
  };
  for (int p = 0; p < NPlayers; ++p) {
    if (!pending_visible_to_observer(p)) continue;
    const int pc = d.pending_commits[static_cast<size_t>(p)];
    if (pc >= 1 && pc <= kNumCards) seen[static_cast<size_t>(pc)] = true;
  }
  // Tracker public history: any card the events stream already attributed
  // to some player p is fully accounted for by table / bullhead in the
  // current state, but if a future code path drops one of those before
  // the apply loop completes, this guard keeps the pool honest.
  for (int p = 0; p < NPlayers; ++p) {
    for (int c = 1; c <= kNumCards; ++c) {
      if (played_by_[static_cast<size_t>(p)]
                    [static_cast<size_t>(c - 1)]) {
        seen[static_cast<size_t>(c)] = true;
      }
    }
  }

  std::vector<int> unseen_pool;
  unseen_pool.reserve(kNumCards);
  for (int c = 1; c <= kNumCards; ++c) {
    if (!seen[static_cast<size_t>(c)]) unseen_pool.push_back(c);
  }

  // §1.6: each opp's "hidden total to observer" = NUM_ROUNDS - current_round
  // (hand cards still to play, plus possibly 1 already in pending). One
  // sequential PL slice per opp; remaining pool (本局未发的牌) is discarded.
  const int n = kNumRounds - d.current_round;

  // Fail-loud invariant: unseen_pool must have enough cards to fill all
  // opps' hidden slots. Off-by-one in schema/state would surface here.
  const int needed = n * (NPlayers - 1);
  if (static_cast<int>(unseen_pool.size()) < needed) {
    throw std::logic_error(
        "6nimmit: unseen_pool too small in randomize_unseen — "
        "schema or state bookkeeping bug");
  }

  // PL sampler: with belief active, sample n cards out of `pool` weighted
  // by `weights[c]`. Sequential without replacement: each draw is a
  // weighted multinomial over the still-remaining cards. With belief
  // inactive (pi_valid_ false or different observer), fall back to a
  // uniform sample = a swap-with-back-then-pop n times — equivalent to
  // shuffling and taking the prefix, but linear in n.
  const bool use_belief = pi_valid_ && pi_observer_ == observer;

  for (int opp = 0; opp < NPlayers; ++opp) {
    if (opp == observer) continue;

    // See .claude/skills/game-rules/simultaneous.md ("What the framework
    // already gives you for free"): tracker no longer touches opp's
    // pending_commits — sim's `current_player()` scan + opponent_selection
    // (Smooth-UCT prior) will naturally drive opp's commit decision as a
    // sim-side ISMCTS opp node, and DAG hash sentinel auto-merges the
    // info set across different opp commits.
    //
    // Two cases for "is opp's pending already public":
    //  (a) viz=1 to observer (post-seal SMALL / apply window): truth
    //      lives in pending_commits[opp] and must be excluded from the
    //      unseen pool (already done above) but we still leave the slot
    //      alone here.
    //  (b) viz=0: session's slot reads stale 0 (placeholder) — the
    //      simultaneous-action design depends on this for "opp not
    //      committed yet" semantics. Rules will sim-side write the
    //      pending value via do_action_fast → opp commits in sim.
    const bool opp_pending_visible = pending_visible_to_observer(opp);
    // Normalize opp's viz=0 pending slot back to the business sentinel 0
    // ("not committed yet"). The session's masked clone writes
    // kPlaceholderInt8 here, which would otherwise fool rules' scan
    // (`pending_commits == 0` check) into treating opp as committed.
    if (!opp_pending_visible) {
      d.pending_commits[static_cast<size_t>(opp)] = 0;
    }
    d.hand[static_cast<size_t>(opp)].fill(0);

    // Hidden hand draws to make: total opp possesses (n) minus the one
    // already pinned in a viz=1 pending slot, when applicable.
    const int hand_draws = n - (opp_pending_visible ? 1 : 0);
    if (hand_draws <= 0) continue;

    // belief_probs_ is keyed by `i = (opp - observer - 1 + N) % N` —
    // matches the label extractor's opp-rotation.
    const int opp_idx = (opp - observer - 1 + NPlayers) % NPlayers;

    std::vector<int> sampled;
    sampled.reserve(static_cast<size_t>(hand_draws));

    for (int k = 0; k < hand_draws; ++k) {
      const int m = static_cast<int>(unseen_pool.size());
      if (m <= 0) {
        throw std::logic_error(
            "6nimmit: unseen_pool exhausted mid-sample — "
            "should be caught by `needed` precheck");
      }
      int picked = -1;
      if (use_belief) {
        // PL weight per remaining card = belief_probs_[opp_idx][c-1].
        // If the marginal sum is non-positive (numerically degenerate
        // posterior over the residual pool — e.g. tracker just saw all
        // high-mass cards drop), fall back to uniform for this draw.
        double total = 0.0;
        for (int j = 0; j < m; ++j) {
          const int c = unseen_pool[static_cast<size_t>(j)];
          total += static_cast<double>(
              belief_probs_[static_cast<size_t>(opp_idx)]
                           [static_cast<size_t>(c - 1)]);
        }
        if (total > 0.0 && std::isfinite(total)) {
          std::uniform_real_distribution<double> uni(0.0, total);
          double r = uni(rng);
          double acc = 0.0;
          for (int j = 0; j < m; ++j) {
            const int c = unseen_pool[static_cast<size_t>(j)];
            acc += static_cast<double>(
                belief_probs_[static_cast<size_t>(opp_idx)]
                             [static_cast<size_t>(c - 1)]);
            if (r <= acc) {
              picked = j;
              break;
            }
          }
          if (picked < 0) picked = m - 1;
        }
      }
      if (picked < 0) {
        std::uniform_int_distribution<int> uni(0, m - 1);
        picked = uni(rng);
      }
      const int card = unseen_pool[static_cast<size_t>(picked)];
      sampled.push_back(card);
      // Swap-pop the consumed entry — O(1).
      unseen_pool[static_cast<size_t>(picked)] =
          unseen_pool[static_cast<size_t>(m - 1)];
      unseen_pool.pop_back();
    }

    // See .claude/skills/game-rules/simultaneous.md ("What the framework
    // already gives you for free" + "The sentinel convention"): write
    // sampled cards into opp's hand, do NOT touch pending_commits[opp]. If
    // pending is viz=1 (post-seal
    // window) it stays as the truth value already in the slot; if
    // viz=0 the session-side stale 0 carries the "opp not yet
    // committed" semantics that lets sim's current_player() scan find
    // opp as a placeholder and route through opponent_selection.
    for (int c : sampled) {
      d.hand[static_cast<size_t>(opp)]
            [static_cast<size_t>(c - 1)] = 1;
    }
  }
  // Remaining unseen_pool (本局未发) is discarded — no slot stores it.
}

template <int NPlayers>
void SixNimmtBeliefTracker<NPlayers>::prepare_for_root(
    const IGameState& root_state, int root_player,
    const IBeliefFeatureExtractor* extractor,
    const IBeliefEvaluator* evaluator) {
  // Reset cache.
  pi_valid_ = false;
  pi_observer_ = -1;
  for (auto& row : belief_probs_) row.fill(0.0f);

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

  const int expected = (NPlayers - 1) * kNumCards;
  if (static_cast<int>(logits.size()) != expected) return;

  // Per-opp output is 104 sigmoid logits. P(opp holds card c) =
  // sigmoid(logit) at fixed temperature 1.
  const float inv_T = 1.0f / board_ai::kBeliefTemperature;
  for (int i = 0; i < NPlayers - 1; ++i) {
    const int base = i * kNumCards;
    for (int c = 0; c < kNumCards; ++c) {
      const double z =
          static_cast<double>(logits[static_cast<size_t>(base + c)]) *
          static_cast<double>(inv_T);
      const double p = 1.0 / (1.0 + std::exp(-z));
      belief_probs_[static_cast<size_t>(i)][static_cast<size_t>(c)] =
          static_cast<float>(p);
    }
  }
  pi_valid_ = true;
  pi_observer_ = root_player;
}

// ============================================================
// SixNimmtBeliefFeatureExtractor — public belief-net input.
// ============================================================

template <int NPlayers>
void SixNimmtBeliefFeatureExtractor<NPlayers>::extract(
    const IGameState& masked_state,
    int perspective_player,
    const IBeliefTracker* tracker,
    std::vector<float>* out) const {
  const auto* s =
      dynamic_cast<const SixNimmtState<NPlayers>*>(&masked_state);
  if (!s || !out || perspective_player < 0 ||
      perspective_player >= NPlayers) {
    if (out) {
      out->assign(static_cast<size_t>(kFeatureDim), 0.0f);
    }
    return;
  }
  const auto& d = s->data;
  const auto* bt =
      dynamic_cast<const SixNimmtBeliefTracker<NPlayers>*>(tracker);

  out->clear();
  out->reserve(static_cast<size_t>(kFeatureDim));

  // [B] own_played (104) + [C] opp_played_union (104)
  std::array<float, kNumCards> own_played{};
  std::array<float, kNumCards> opp_played_union{};
  if (bt) {
    for (int c = 1; c <= kNumCards; ++c) {
      if (bt->played_by(perspective_player, c)) {
        own_played[static_cast<size_t>(c - 1)] = 1.0f;
      }
      for (int p = 0; p < NPlayers; ++p) {
        if (p == perspective_player) continue;
        if (bt->played_by(p, c)) {
          opp_played_union[static_cast<size_t>(c - 1)] = 1.0f;
          break;
        }
      }
    }
  }

  // [A] unseen_pool = 1..104 \ own_hand \ ⋃_p played_by_ \ table \
  //     bullhead piles — the same pool randomize_unseen deals from.
  // Table and piles are subtracted directly (all_public): the four
  // starting row heads were never "played" by anyone, so played_by_
  // alone would leave them — cards lying face up — marked unseen.
  // own_hand from masked state (perspective sees own hand viz=1).
  std::array<bool, kNumCards + 1> seen{};
  for (int r = 0; r < kNumRows; ++r) {
    for (int s_idx = 0; s_idx < kRowCapacity; ++s_idx) {
      const int c = d.table[static_cast<size_t>(r)][static_cast<size_t>(s_idx)];
      if (c >= 1 && c <= kNumCards) seen[static_cast<size_t>(c)] = true;
    }
  }
  for (int p = 0; p < NPlayers; ++p) {
    for (int c = 1; c <= kNumCards; ++c) {
      if (d.bullhead_pile[static_cast<size_t>(p)][static_cast<size_t>(c - 1)]) {
        seen[static_cast<size_t>(c)] = true;
      }
    }
  }
  for (int c = 1; c <= kNumCards; ++c) {
    if (d.hand[static_cast<size_t>(perspective_player)]
              [static_cast<size_t>(c - 1)] == 1) {
      seen[static_cast<size_t>(c)] = true;
    }
  }
  if (bt) {
    for (int p = 0; p < NPlayers; ++p) {
      for (int c = 1; c <= kNumCards; ++c) {
        if (bt->played_by(p, c)) seen[static_cast<size_t>(c)] = true;
      }
    }
  }
  // pending_commits visible to observer: own pending always viz=1; opp
  // pending viz=1 only after seal (incl. SMALL window). masked clone
  // contract: viz=0 reads kPlaceholderInt8 (out of 1..104 → skipped),
  // viz=1 reads the true card_id → excluded from unseen_pool. Same
  // single-loop trick as the PV encoder; no SMALL-phase branching.
  for (int p = 0; p < NPlayers; ++p) {
    const std::int8_t pc =
        d.pending_commits[static_cast<size_t>(p)];
    if (pc >= 1 && pc <= kNumCards) {
      seen[static_cast<size_t>(pc)] = true;
    }
  }
  // [A] write
  for (int c = 1; c <= kNumCards; ++c) {
    out->push_back(seen[static_cast<size_t>(c)] ? 0.0f : 1.0f);
  }
  // [B] write
  for (int c = 0; c < kNumCards; ++c) {
    out->push_back(own_played[static_cast<size_t>(c)]);
  }
  // [C] write
  for (int c = 0; c < kNumCards; ++c) {
    out->push_back(opp_played_union[static_cast<size_t>(c)]);
  }

  // [D] table dense (12) — same shape as PV encoder's table block.
  for (int r = 0; r < kNumRows; ++r) {
    int len = 0;
    int tail_card = 0;
    int heads = 0;
    for (int s_idx = 0; s_idx < kRowCapacity; ++s_idx) {
      int c = d.table[static_cast<size_t>(r)][static_cast<size_t>(s_idx)];
      if (c >= 1 && c <= kNumCards) {
        tail_card = c;
        heads += head_value(c);
        ++len;
      } else {
        break;
      }
    }
    out->push_back(static_cast<float>(tail_card) /
                   static_cast<float>(kNumCards));
    out->push_back(static_cast<float>(len) /
                   static_cast<float>(kRowCapacity));
    out->push_back(static_cast<float>(heads) / 30.0f);
  }

  // [E] bullhead counts per seat, rotated own-first, normalized by 30.
  for (int pi = 0; pi < NPlayers; ++pi) {
    const int pid = (perspective_player + pi) % NPlayers;
    int heads = 0;
    for (int c = 1; c <= kNumCards; ++c) {
      if (d.bullhead_pile[static_cast<size_t>(pid)]
                          [static_cast<size_t>(c - 1)]) {
        heads += head_value(c);
      }
    }
    out->push_back(static_cast<float>(heads) / 30.0f);
  }

  // [F] meta: small_phase_active + small_player one-hot rotated own-first.
  out->push_back(d.small_phase_active ? 1.0f : 0.0f);
  for (int pi = 0; pi < NPlayers; ++pi) {
    const int pid = (perspective_player + pi) % NPlayers;
    out->push_back(d.small_player == pid ? 1.0f : 0.0f);
  }
}

// ============================================================
// SixNimmtBeliefLabelExtractor — GT-side opp-hand multi-hot.
// ============================================================

template <int NPlayers>
void SixNimmtBeliefLabelExtractor<NPlayers>::extract(
    const IGameState& truth_state,
    int observer,
    std::vector<std::vector<int>>* out_hand_counts,
    std::vector<int>* out_remaining,
    std::vector<int>* out_alive) const {
  const auto& s = checked_cast<SixNimmtState<NPlayers>>(truth_state);
  const auto& d = s.data;

  out_hand_counts->assign(static_cast<size_t>(NPlayers - 1),
                          std::vector<int>(kClassCount, 0));
  const auto& pc_viz = viz::viz_get(truth_state, "pending_commits");
  auto pending_visible_to_observer = [&](int owner_p) -> bool {
    const std::size_t base = viz::flat_offset_data_only(
        pc_viz.shape, std::vector<int>{owner_p});
    return pc_viz.data[base + static_cast<std::size_t>(observer)] != 0;
  };
  for (int i = 0; i < NPlayers - 1; ++i) {
    const int opp = (observer + 1 + i) % NPlayers;
    auto& row = (*out_hand_counts)[static_cast<size_t>(i)];
    for (int c = 1; c <= kNumCards; ++c) {
      if (d.hand[static_cast<size_t>(opp)]
                [static_cast<size_t>(c - 1)] == 1) {
        row[static_cast<size_t>(c - 1)] = 1;
      }
    }
    // Sealed pending stays in opp's possession only while still hidden
    // to observer. Once viz=1 (post-seal SMALL / apply window) the input
    // extractor excludes that card from `unseen_pool`, so the label must
    // exclude it too — otherwise the BCE target asks the head to predict
    // a card the input has already disclosed as "not held".
    int pc = d.pending_commits[static_cast<size_t>(opp)];
    if (pc >= 1 && pc <= kNumCards && !pending_visible_to_observer(opp)) {
      row[static_cast<size_t>(pc - 1)] = 1;
    }
  }
  // remaining / alive_per_opp unused here — framework requires the shapes.
  out_remaining->assign(static_cast<size_t>(kClassCount), 0);
  out_alive->assign(static_cast<size_t>(NPlayers - 1), 1);
}

template class SixNimmtFeatureEncoder<2>;
template class SixNimmtFeatureEncoder<3>;
template class SixNimmtFeatureEncoder<4>;
template class SixNimmtBeliefTracker<2>;
template class SixNimmtBeliefTracker<3>;
template class SixNimmtBeliefTracker<4>;
template class SixNimmtBeliefFeatureExtractor<2>;
template class SixNimmtBeliefFeatureExtractor<3>;
template class SixNimmtBeliefFeatureExtractor<4>;
template class SixNimmtBeliefLabelExtractor<2>;
template class SixNimmtBeliefLabelExtractor<3>;
template class SixNimmtBeliefLabelExtractor<4>;

}  // namespace board_ai::sixnimmt
