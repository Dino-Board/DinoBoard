#include "maskmen_net_adapter.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "../../engine/core/masked_state.h"
#include "../../engine/core/visibility_schema.h"

namespace board_ai::maskmen {

namespace {

// Push the public block: shared between the policy/value encoder and
// the belief feature extractor.
//   own hand counts (normalized by kStartingHand)        : 6
//   last_play_type one-hot                                : 6
//   last_play=none bit                                    : 1
//   last_play_size / kMaxPlaySize                         : 1
//   is_first_play_of_round                                : 1
//   strength_above closure flat                           : 36
// Total: 51
template <int NPlayers>
void push_public_block(const MaskmenData<NPlayers>& d,
                       int perspective,
                       std::vector<float>* out) {
  // own hand normalized
  for (int t = 0; t < kTypeCount; ++t) {
    const int v = d.hand_count[static_cast<size_t>(perspective)]
                              [static_cast<size_t>(t)];
    out->push_back(v < 0 ? 0.0f
                         : static_cast<float>(v) /
                               static_cast<float>(kStartingHand));
  }
  // last_play_type one-hot + none bit
  const int lpt = d.last_play_type;
  for (int t = 0; t < kTypeCount; ++t) {
    out->push_back(lpt == t ? 1.0f : 0.0f);
  }
  out->push_back(lpt < 0 ? 1.0f : 0.0f);
  // last_play_size normalized
  out->push_back(static_cast<float>(d.last_play_size) /
                 static_cast<float>(kMaxPlaySize));
  // is_first_play_of_round
  out->push_back(static_cast<float>(d.is_first_play_of_round));
  // strength_above flat
  for (int a = 0; a < kTypeCount; ++a) {
    for (int b = 0; b < kTypeCount; ++b) {
      out->push_back(static_cast<float>(
          d.strength_above[static_cast<size_t>(a)][static_cast<size_t>(b)]));
    }
  }
}

// Encode a single play into the output vector.
// Per-play: type one-hot (6) + size/3.0 (1) + actor one-hot seat-rotated (N) + is_first_in_round (1)
template <int NPlayers>
void push_play(const MaskmenPlay& play, int perspective, bool is_first,
               std::vector<float>* out) {
  for (int t = 0; t < kTypeCount; ++t) {
    out->push_back(play.type == t ? 1.0f : 0.0f);
  }
  out->push_back(static_cast<float>(play.size) / 3.0f);
  const int rotated = (play.actor - perspective + NPlayers) % NPlayers;
  for (int k = 0; k < NPlayers; ++k) {
    out->push_back(rotated == k ? 1.0f : 0.0f);
  }
  out->push_back(is_first ? 1.0f : 0.0f);
}

template <int NPlayers>
void push_chain_padding(int slots, std::vector<float>* out) {
  constexpr int kWidth = 6 + 1 + NPlayers + 1;
  for (int i = 0; i < slots * kWidth; ++i) {
    out->push_back(0.0f);
  }
}

// Push chain blocks shared by PV encoder and belief feature extractor.
// Encoding order (reverse — most recent first):
//   1. Current round chain: up to 6 plays, reversed
//   2. History chains: up to 8 rounds × 6 plays each, both layers reversed
//   3. current_round_idx: completed rounds / 15.0
template <int NPlayers>
void push_chain_blocks(const MaskmenBeliefTracker<NPlayers>* mtr,
                       int perspective, std::vector<float>* out) {
  constexpr int kMaxLen = MaskmenFeatureLayout<NPlayers>::kMaxChainLen;
  constexpr int kMaxRounds = MaskmenFeatureLayout<NPlayers>::kMaxHistoryRounds;

  if (!mtr) {
    constexpr int kDim = MaskmenFeatureLayout<NPlayers>::kCurrentChainDim +
                         MaskmenFeatureLayout<NPlayers>::kHistoryChainDim + 1;
    for (int i = 0; i < kDim; ++i) out->push_back(0.0f);
    return;
  }

  // Current round chain (reversed)
  const auto& cur = mtr->current_round_plays();
  const int cur_len = std::min(static_cast<int>(cur.size()), kMaxLen);
  for (int i = cur_len - 1; i >= 0; --i) {
    push_play<NPlayers>(cur[static_cast<size_t>(i)], perspective, i == 0, out);
  }
  push_chain_padding<NPlayers>(kMaxLen - cur_len, out);

  // History chains (reversed per round, rounds already most-recent-first)
  const auto& hist = mtr->history_plays();
  const int num_rounds = std::min(static_cast<int>(hist.size()), kMaxRounds);
  for (int r = 0; r < num_rounds; ++r) {
    const auto& round = hist[static_cast<size_t>(r)];
    const int rlen = std::min(static_cast<int>(round.size()), kMaxLen);
    for (int i = rlen - 1; i >= 0; --i) {
      push_play<NPlayers>(round[static_cast<size_t>(i)], perspective, i == 0,
                          out);
    }
    push_chain_padding<NPlayers>(kMaxLen - rlen, out);
  }
  push_chain_padding<NPlayers>(
      (kMaxRounds - num_rounds) * kMaxLen, out);

  // current_round_idx
  out->push_back(static_cast<float>(hist.size()) / 15.0f);
}

}  // namespace

template <int NPlayers>
void MaskmenFeatureEncoder<NPlayers>::encode_features(
    const IGameState& state,
    int perspective_player,
    const IBeliefTracker* tracker,
    std::vector<float>* out) const {
  const auto& s = checked_cast<MaskmenState<NPlayers>>(state);
  const auto& d = s.data;
  if (!out) return;
  if (perspective_player < 0 || perspective_player >= NPlayers) {
    throw std::logic_error(
        "MaskmenFeatureEncoder::encode_features: invalid perspective");
  }
  out->reserve(static_cast<size_t>(Layout::kFeatureDim));

  push_public_block<NPlayers>(d, perspective_player, out);

  const auto* mtr =
      dynamic_cast<const MaskmenBeliefTracker<NPlayers>*>(tracker);

  for (int rot = 0; rot < NPlayers; ++rot) {
    const int seat = (perspective_player + rot) % NPlayers;
    // hand_total / kStartingHand. For self use direct count; for opps
    // prefer the tracker's public memory if available; else fall back to
    // summing the visible hand row (works for self only — for opps the
    // row is placeholder, but the pre-tracker bootstrap path also passes
    // null tracker, in which case the framework hasn't started belief
    // yet and we just emit 1.0).
    int hand_total = 0;
    if (seat == perspective_player) {
      for (int t = 0; t < kTypeCount; ++t) {
        const int v = d.hand_count[static_cast<size_t>(seat)]
                                  [static_cast<size_t>(t)];
        if (v >= 0) hand_total += v;
      }
    } else if (mtr) {
      hand_total = mtr->hand_total_for(seat);
    } else {
      hand_total = kStartingHand;
    }
    out->push_back(static_cast<float>(hand_total) /
                   static_cast<float>(kStartingHand));
    // passed_this_round
    out->push_back(static_cast<float>(
        d.passed_this_round[static_cast<size_t>(seat)]));
    // is_current_player
    out->push_back(seat == d.current_player_ ? 1.0f : 0.0f);
    // belief expected per-type counts
    if (seat == perspective_player) {
      for (int t = 0; t < kTypeCount; ++t) {
        const int v = d.hand_count[static_cast<size_t>(seat)]
                                  [static_cast<size_t>(t)];
        out->push_back(v < 0 ? 0.0f
                             : static_cast<float>(v) /
                                   static_cast<float>(kCardsPerType));
      }
    } else if (mtr && mtr->pi_valid() &&
               mtr->pi_observer() == perspective_player) {
      const int opp_idx =
          (seat - perspective_player - 1 + NPlayers) % NPlayers;
      for (int t = 0; t < kTypeCount; ++t) {
        out->push_back(mtr->pi_expected(opp_idx, t) /
                       static_cast<float>(kCardsPerType));
      }
    } else {
      // Uniform pool prior for that opp.
      for (int t = 0; t < kTypeCount; ++t) {
        out->push_back(0.0f);
      }
    }
  }

  push_chain_blocks<NPlayers>(mtr, perspective_player, out);
}

// ---------------------------------------------------------------------------
// MaskmenBeliefTracker
// ---------------------------------------------------------------------------

template <int NPlayers>
void MaskmenBeliefTracker<NPlayers>::init(IGameState& state, int perspective,
                                          const AnyMap& /*payload*/) {
  perspective_ = perspective;
  played_per_type_.fill(0);
  for (int p = 0; p < NPlayers; ++p) {
    hand_total_[static_cast<size_t>(p)] = static_cast<std::int8_t>(kStartingHand);
    passed_this_round_[static_cast<size_t>(p)] = 0;
  }
  pi_valid_ = false;
  pi_observer_ = -1;
  for (int i = 0; i < NPlayers - 1; ++i) {
    pi_expected_[static_cast<size_t>(i)].fill(0.0f);
  }
  current_round_plays_.clear();
  history_plays_.clear();
  // No private bootstrap to write — Maskmen masks each opp's hand row
  // but has no payload to apply; observer's own hand is already viz=1.
  (void)state;
}

template <int NPlayers>
void MaskmenBeliefTracker<NPlayers>::observe_public_event(
    int /*actor*/,
    ActionId /*action*/,
    const std::vector<PublicEvent>& events) {
  for (const auto& ev : events) {
    if (ev.first == "played") {
      const auto& m = ev.second;
      auto it_p = m.find("player");
      auto it_t = m.find("type");
      auto it_s = m.find("size");
      if (it_p == m.end() || it_t == m.end() || it_s == m.end()) {
        throw std::logic_error(
            "MaskmenBeliefTracker: 'played' event missing fields");
      }
      const int p = std::any_cast<int>(it_p->second);
      const int t = std::any_cast<int>(it_t->second);
      const int sz = std::any_cast<int>(it_s->second);
      played_per_type_[static_cast<size_t>(t)] =
          static_cast<std::int8_t>(played_per_type_[static_cast<size_t>(t)] + sz);
      hand_total_[static_cast<size_t>(p)] =
          static_cast<std::int8_t>(hand_total_[static_cast<size_t>(p)] - sz);
      current_round_plays_.push_back(MaskmenPlay{p, t, sz});
    } else if (ev.first == "round_end") {
      passed_this_round_.fill(0);
      history_plays_.insert(history_plays_.begin(),
                            std::move(current_round_plays_));
      current_round_plays_.clear();
      if (static_cast<int>(history_plays_.size()) >
          MaskmenFeatureLayout<NPlayers>::kMaxHistoryRounds) {
        history_plays_.pop_back();
      }
    }
  }
}

template <int NPlayers>
void MaskmenBeliefTracker<NPlayers>::randomize_unseen(
    IGameState& state, int observer, std::mt19937_64& rng) const {
  auto& s = checked_cast<MaskmenState<NPlayers>>(state);
  auto& d = s.data;

  // Compute the unseen pool per type.
  // pool[t] is an UPPER BOUND on opponents' total of type t — for N<4 the
  // dealer silently discards (60 - 15N) cards whose composition is hidden,
  // so cap_sum can exceed total_unseen. IPF column targets are
  //   target[t] = cap[t] * (total_unseen / cap_sum)
  // (proportional fair-share assumption: discards drawn uniformly from
  // unseen). Urn caps draws-of-type-t at cap[t].
  std::array<int, kTypeCount> cap{};
  std::array<int, kTypeCount> own{};
  for (int t = 0; t < kTypeCount; ++t) {
    const int v = d.hand_count[static_cast<size_t>(observer)]
                              [static_cast<size_t>(t)];
    own[static_cast<size_t>(t)] = (v < 0 ? 0 : v);
    cap[static_cast<size_t>(t)] =
        kCardsPerType - own[static_cast<size_t>(t)] -
        played_per_type_[static_cast<size_t>(t)];
    if (cap[static_cast<size_t>(t)] < 0) {
      throw std::logic_error(
          "MaskmenBeliefTracker::randomize_unseen: negative cap");
    }
  }

  std::array<int, NPlayers> opp_remaining{};
  int total_unseen = 0;
  for (int p = 0; p < NPlayers; ++p) {
    if (p == observer) {
      opp_remaining[static_cast<size_t>(p)] = 0;
    } else {
      opp_remaining[static_cast<size_t>(p)] =
          hand_total_[static_cast<size_t>(p)];
      total_unseen += hand_total_[static_cast<size_t>(p)];
    }
  }

  int cap_sum = 0;
  for (int t = 0; t < kTypeCount; ++t) cap_sum += cap[static_cast<size_t>(t)];
  if (cap_sum < total_unseen) {
    throw std::logic_error(
        "MaskmenBeliefTracker::randomize_unseen: cap_sum < total_unseen");
  }
  // Column targets for IPF: scale caps so they sum to total_unseen.
  std::array<double, kTypeCount> col_target{};
  for (int t = 0; t < kTypeCount; ++t) {
    col_target[static_cast<size_t>(t)] =
        cap_sum > 0
            ? static_cast<double>(cap[static_cast<size_t>(t)]) *
                  static_cast<double>(total_unseen) /
                  static_cast<double>(cap_sum)
            : 0.0;
  }

  // Build (NPlayers-1) × kTypeCount expectation matrix M.
  // Row order: opp_idx i = 0..N-2, mapping seat = (observer+1+i) % N.
  std::array<std::array<double, kTypeCount>, NPlayers - 1> M{};
  const bool use_pi = pi_valid_ && pi_observer_ == observer;
  for (int i = 0; i < NPlayers - 1; ++i) {
    const int seat = (observer + 1 + i) % NPlayers;
    for (int t = 0; t < kTypeCount; ++t) {
      double v;
      if (use_pi) {
        v = static_cast<double>(pi_expected_[static_cast<size_t>(i)]
                                            [static_cast<size_t>(t)]);
      } else {
        // Uniform fallback: opp_total * col_target_t / total_unseen
        const double rem = static_cast<double>(
            opp_remaining[static_cast<size_t>(seat)]);
        v = total_unseen > 0
                ? rem * col_target[static_cast<size_t>(t)] /
                      static_cast<double>(total_unseen)
                : 0.0;
      }
      if (v < 1e-12) v = 1e-12;
      M[static_cast<size_t>(i)][static_cast<size_t>(t)] = v;
    }
  }

  // IPF: alternately rescale rows to match opp_remaining and columns
  // to match pool. 30 iters, tol 1e-5.
  for (int iter = 0; iter < 30; ++iter) {
    double max_delta = 0.0;
    // Row scale → row sum = opp_remaining[seat]
    for (int i = 0; i < NPlayers - 1; ++i) {
      const int seat = (observer + 1 + i) % NPlayers;
      const double target =
          static_cast<double>(opp_remaining[static_cast<size_t>(seat)]);
      double rs = 0.0;
      for (int t = 0; t < kTypeCount; ++t) {
        rs += M[static_cast<size_t>(i)][static_cast<size_t>(t)];
      }
      if (rs <= 0.0) continue;
      const double scale = target / rs;
      max_delta = std::max(max_delta, std::abs(scale - 1.0));
      for (int t = 0; t < kTypeCount; ++t) {
        M[static_cast<size_t>(i)][static_cast<size_t>(t)] *= scale;
      }
    }
    // Column scale → column sum = col_target[t]
    for (int t = 0; t < kTypeCount; ++t) {
      const double target = col_target[static_cast<size_t>(t)];
      double cs = 0.0;
      for (int i = 0; i < NPlayers - 1; ++i) {
        cs += M[static_cast<size_t>(i)][static_cast<size_t>(t)];
      }
      if (cs <= 0.0) continue;
      const double scale = target / cs;
      max_delta = std::max(max_delta, std::abs(scale - 1.0));
      for (int i = 0; i < NPlayers - 1; ++i) {
        M[static_cast<size_t>(i)][static_cast<size_t>(t)] *= scale;
      }
    }
    if (max_delta < 1e-5) break;
  }

  // Floor to keep support.
  for (int i = 0; i < NPlayers - 1; ++i) {
    for (int t = 0; t < kTypeCount; ++t) {
      if (M[static_cast<size_t>(i)][static_cast<size_t>(t)] < 1e-12) {
        M[static_cast<size_t>(i)][static_cast<size_t>(t)] = 1e-12;
      }
    }
  }

  // Weighted urn: pick (i,t) with weight M[i][t] * cap_left[t], decrement.
  // cap_left[t] is the per-type ceiling on how many cards of that type
  // can still be assigned to opps (kCardsPerType - own[t] - played[t]
  // minus what we've already drawn). Discards consume from this same
  // budget but invisibly — the urn just stops drawing of a type once
  // its cap hits zero.
  std::array<int, kTypeCount> cap_left = cap;
  std::array<int, NPlayers> opp_left = opp_remaining;
  std::array<std::array<int, kTypeCount>, NPlayers> draw{};
  for (int p = 0; p < NPlayers; ++p) {
    draw[static_cast<size_t>(p)].fill(0);
  }

  while (true) {
    // Build weighted choices over (i, t) where opp_left[seat]>0 and cap_left[t]>0
    double total_w = 0.0;
    std::array<std::array<double, kTypeCount>, NPlayers - 1> W{};
    for (int i = 0; i < NPlayers - 1; ++i) {
      const int seat = (observer + 1 + i) % NPlayers;
      if (opp_left[static_cast<size_t>(seat)] <= 0) {
        W[static_cast<size_t>(i)].fill(0.0);
        continue;
      }
      for (int t = 0; t < kTypeCount; ++t) {
        if (cap_left[static_cast<size_t>(t)] <= 0) {
          W[static_cast<size_t>(i)][static_cast<size_t>(t)] = 0.0;
          continue;
        }
        const double w =
            M[static_cast<size_t>(i)][static_cast<size_t>(t)] *
            static_cast<double>(cap_left[static_cast<size_t>(t)]);
        W[static_cast<size_t>(i)][static_cast<size_t>(t)] = w;
        total_w += w;
      }
    }
    if (total_w <= 0.0) break;

    std::uniform_real_distribution<double> u(0.0, total_w);
    double r = u(rng);
    int pick_i = -1, pick_t = -1;
    for (int i = 0; i < NPlayers - 1 && pick_i < 0; ++i) {
      for (int t = 0; t < kTypeCount && pick_i < 0; ++t) {
        const double w = W[static_cast<size_t>(i)][static_cast<size_t>(t)];
        if (w <= 0.0) continue;
        r -= w;
        if (r <= 0.0) {
          pick_i = i;
          pick_t = t;
        }
      }
    }
    if (pick_i < 0) {
      // Float underflow safety: pick the last viable cell.
      for (int i = NPlayers - 2; i >= 0 && pick_i < 0; --i) {
        for (int t = kTypeCount - 1; t >= 0 && pick_i < 0; --t) {
          if (W[static_cast<size_t>(i)][static_cast<size_t>(t)] > 0.0) {
            pick_i = i;
            pick_t = t;
          }
        }
      }
    }
    if (pick_i < 0) break;
    const int seat = (observer + 1 + pick_i) % NPlayers;
    draw[static_cast<size_t>(seat)][static_cast<size_t>(pick_t)] += 1;
    cap_left[static_cast<size_t>(pick_t)] -= 1;
    opp_left[static_cast<size_t>(seat)] -= 1;
  }

  // Validate every opp got fully assigned.
  for (int p = 0; p < NPlayers; ++p) {
    if (p == observer) continue;
    if (opp_left[static_cast<size_t>(p)] != 0) {
      throw std::logic_error(
          "MaskmenBeliefTracker::randomize_unseen: opp_left nonzero after urn");
    }
  }

  // Write the sampled hands into state for opps (observer's row is viz=1).
  for (int p = 0; p < NPlayers; ++p) {
    if (p == observer) continue;
    for (int t = 0; t < kTypeCount; ++t) {
      d.hand_count[static_cast<size_t>(p)][static_cast<size_t>(t)] =
          static_cast<std::int8_t>(draw[static_cast<size_t>(p)][static_cast<size_t>(t)]);
    }
  }
}

template <int NPlayers>
void MaskmenBeliefTracker<NPlayers>::prepare_for_root(
    const IGameState& root_state,
    int root_player,
    const IBeliefFeatureExtractor* extractor,
    const IBeliefEvaluator* evaluator) {
  pi_valid_ = false;
  pi_observer_ = -1;
  if (!extractor || !evaluator) return;
  std::vector<float> features;
  if (!extractor->extract_from_state(root_state, root_player, this,
                                     &features)) {
    throw std::logic_error(
        "MaskmenBeliefTracker::prepare_for_root: extractor produced wrong size");
  }
  std::vector<float> logits;
  if (!evaluator->evaluate(features, &logits)) {
    throw std::logic_error(
        "MaskmenBeliefTracker::prepare_for_root: evaluator returned false");
  }
  const int expected = (NPlayers - 1) * kTypeCount;
  if (static_cast<int>(logits.size()) != expected) {
    throw std::logic_error(
        "MaskmenBeliefTracker::prepare_for_root: logit count mismatch");
  }
  for (int i = 0; i < NPlayers - 1; ++i) {
    for (int t = 0; t < kTypeCount; ++t) {
      float v = logits[static_cast<size_t>(i * kTypeCount + t)];
      if (!std::isfinite(v) || v < 1e-6f) v = 1e-6f;
      if (v > static_cast<float>(kCardsPerType)) v = static_cast<float>(kCardsPerType);
      pi_expected_[static_cast<size_t>(i)][static_cast<size_t>(t)] = v;
    }
  }
  pi_valid_ = true;
  pi_observer_ = root_player;
}

template <int NPlayers>
AnyMap MaskmenBeliefTracker<NPlayers>::serialize() const {
  AnyMap m;
  std::vector<int> played(kTypeCount, 0);
  for (int t = 0; t < kTypeCount; ++t) {
    played[static_cast<size_t>(t)] =
        played_per_type_[static_cast<size_t>(t)];
  }
  m["played_per_type"] = std::any(played);
  std::vector<int> ht(NPlayers, 0);
  std::vector<int> pp(NPlayers, 0);
  for (int p = 0; p < NPlayers; ++p) {
    ht[static_cast<size_t>(p)] = hand_total_[static_cast<size_t>(p)];
    pp[static_cast<size_t>(p)] = passed_this_round_[static_cast<size_t>(p)];
  }
  m["hand_total"] = std::any(ht);
  m["passed_this_round"] = std::any(pp);

  // Serialize chain data: flatten plays as [actor, type, size, ...]
  std::vector<int> cur_chain;
  for (const auto& play : current_round_plays_) {
    cur_chain.push_back(play.actor);
    cur_chain.push_back(play.type);
    cur_chain.push_back(play.size);
  }
  m["current_round_plays"] = std::any(cur_chain);

  std::vector<std::vector<int>> hist;
  for (const auto& round : history_plays_) {
    std::vector<int> r;
    for (const auto& play : round) {
      r.push_back(play.actor);
      r.push_back(play.type);
      r.push_back(play.size);
    }
    hist.push_back(std::move(r));
  }
  m["history_plays"] = std::any(hist);

  return m;
}

// ---------------------------------------------------------------------------
// MaskmenBeliefFeatureExtractor
// ---------------------------------------------------------------------------

template <int NPlayers>
void MaskmenBeliefFeatureExtractor<NPlayers>::extract(
    const IGameState& state,
    int perspective_player,
    const IBeliefTracker* tracker,
    std::vector<float>* out) const {
  const auto& s = checked_cast<MaskmenState<NPlayers>>(state);
  const auto& d = s.data;
  if (!out) return;
  out->reserve(static_cast<size_t>(kFeatureDim));

  push_public_block<NPlayers>(d, perspective_player, out);

  const auto* mtr =
      dynamic_cast<const MaskmenBeliefTracker<NPlayers>*>(tracker);

  // played_per_type / kCardsPerType
  for (int t = 0; t < kTypeCount; ++t) {
    int v = 0;
    if (mtr) {
      v = mtr->played_per_type(t);
    }
    out->push_back(static_cast<float>(v) / static_cast<float>(kCardsPerType));
  }

  // Per-opp (observer-rotated): hand_total/start, passed, is_current
  for (int i = 0; i < NPlayers - 1; ++i) {
    const int seat = (perspective_player + 1 + i) % NPlayers;
    const int ht = mtr ? mtr->hand_total_for(seat) : kStartingHand;
    out->push_back(static_cast<float>(ht) / static_cast<float>(kStartingHand));
    out->push_back(static_cast<float>(
        d.passed_this_round[static_cast<size_t>(seat)]));
    out->push_back(seat == d.current_player_ ? 1.0f : 0.0f);
  }

  push_chain_blocks<NPlayers>(mtr, perspective_player, out);
}

// ---------------------------------------------------------------------------
// MaskmenBeliefLabelExtractor
// ---------------------------------------------------------------------------

template <int NPlayers>
void MaskmenBeliefLabelExtractor<NPlayers>::extract(
    const IGameState& truth_state,
    int observer,
    std::vector<std::vector<int>>* out_hand_counts,
    std::vector<int>* out_remaining,
    std::vector<int>* out_alive) const {
  const auto& s = checked_cast<MaskmenState<NPlayers>>(truth_state);
  const auto& d = s.data;

  out_hand_counts->assign(NPlayers - 1, std::vector<int>(kTypeCount, 0));
  out_remaining->assign(kTypeCount, 0);
  out_alive->assign(NPlayers - 1, 0);

  // Compute publicly played per type from truth: kCardsPerType * 1 minus
  // (sum of all hands per type). hand_count is truth here.
  std::array<int, kTypeCount> in_hands{};
  for (int p = 0; p < NPlayers; ++p) {
    for (int t = 0; t < kTypeCount; ++t) {
      in_hands[static_cast<size_t>(t)] +=
          d.hand_count[static_cast<size_t>(p)][static_cast<size_t>(t)];
    }
  }
  std::array<int, kTypeCount> own{};
  for (int t = 0; t < kTypeCount; ++t) {
    own[static_cast<size_t>(t)] =
        d.hand_count[static_cast<size_t>(observer)][static_cast<size_t>(t)];
  }
  // remaining[t] = kCardsPerType - publicly_played - own
  // publicly_played = kCardsPerType - in_hands[t]
  for (int t = 0; t < kTypeCount; ++t) {
    (*out_remaining)[static_cast<size_t>(t)] =
        in_hands[static_cast<size_t>(t)] - own[static_cast<size_t>(t)];
  }

  for (int i = 0; i < NPlayers - 1; ++i) {
    const int seat = (observer + 1 + i) % NPlayers;
    int sum = 0;
    for (int t = 0; t < kTypeCount; ++t) {
      const int v =
          d.hand_count[static_cast<size_t>(seat)][static_cast<size_t>(t)];
      (*out_hand_counts)[static_cast<size_t>(i)][static_cast<size_t>(t)] = v;
      sum += v;
    }
    (*out_alive)[static_cast<size_t>(i)] = sum > 0 ? 1 : 0;
  }
}

template class MaskmenFeatureEncoder<2>;
template class MaskmenFeatureEncoder<3>;
template class MaskmenFeatureEncoder<4>;
template class MaskmenBeliefTracker<2>;
template class MaskmenBeliefTracker<3>;
template class MaskmenBeliefTracker<4>;
template class MaskmenBeliefFeatureExtractor<2>;
template class MaskmenBeliefFeatureExtractor<3>;
template class MaskmenBeliefFeatureExtractor<4>;
template class MaskmenBeliefLabelExtractor<2>;
template class MaskmenBeliefLabelExtractor<3>;
template class MaskmenBeliefLabelExtractor<4>;

}  // namespace board_ai::maskmen
