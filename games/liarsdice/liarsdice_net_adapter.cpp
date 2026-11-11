#include "liarsdice_net_adapter.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../engine/core/masked_state.h"
#include "../../engine/core/viz_runtime.h"
#include "liarsdice_rules.h"
#include "outcome_index.h"

namespace board_ai::liarsdice {

namespace {

// Append a single 6-dim bid descriptor (X dense + Y one-hot 5 over face
// 2..6). pad=true emits the all-zero placeholder ("no bid"). x is
// normalized by kTotalDice so it lands in [0, 1].
template <int NPlayers>
inline void append_bid_features(std::vector<float>* out, int x, int y,
                                bool pad) {
  if (pad) {
    for (int i = 0; i < 6; ++i) out->push_back(0.0f);
    return;
  }
  constexpr float kInvTotalDice =
      1.0f / static_cast<float>(LiarsdiceConfig<NPlayers>::kTotalDice);
  out->push_back(static_cast<float>(x) * kInvTotalDice);
  for (int f = 2; f <= 6; ++f) {
    out->push_back(f == y ? 1.0f : 0.0f);
  }
}

// Append one seat's last-K=5 bid block in reverse order (slot 0 = most
// recent). PLAN §1.6.
template <int NPlayers>
inline void append_seat_bid_block(
    std::vector<float>* out,
    const LiarsdiceBeliefTracker<NPlayers>* tracker, int seat) {
  constexpr int K = LiarsdiceConfig<NPlayers>::kBidsPerSeat;
  const auto& bids = tracker->bids_for(seat);
  const int n = static_cast<int>(bids.size());
  for (int k = 0; k < K; ++k) {
    if (k < n) {
      const auto& bid = bids[static_cast<size_t>(n - 1 - k)];
      append_bid_features<NPlayers>(out, bid.first, bid.second, /*pad=*/false);
    } else {
      append_bid_features<NPlayers>(out, 0, 0, /*pad=*/true);
    }
  }
}

}  // namespace

// ============================================================
// LiarsdiceFeatureEncoder
// ============================================================

template <int NPlayers>
void LiarsdiceFeatureEncoder<NPlayers>::encode_features(
    const IGameState& masked_state,
    int perspective_player,
    const IBeliefTracker* tracker,
    std::vector<float>* out) const {
  const auto& s = checked_cast<LiarsdiceState<NPlayers>>(masked_state);
  const auto& d = s.data;
  out->clear();
  out->reserve(static_cast<size_t>(Cfg::kFeatureDim));

  // own dice_count[perspective] (viz=1 owner-only). Hidden bytes are
  // kPlaceholderInt8 — for the perspective's OWN row they are always
  // visible, so we never read placeholder here. Defensive: clamp <0 → 0.
  // Normalized by kDicePerPlayer so each face count lands in [0, 1].
  constexpr float kInvDicePerPlayer = 1.0f / static_cast<float>(kDicePerPlayer);
  for (int f = 0; f < kFaceCount; ++f) {
    int v = d.dice_count[static_cast<size_t>(perspective_player)]
                        [static_cast<size_t>(f)];
    if (v == kPlaceholderInt8 || v < 0) v = 0;
    out->push_back(static_cast<float>(v) * kInvDicePerPlayer);
  }

  // current_bid_count scalar (public). Normalized by kTotalDice → [0, 1].
  constexpr float kInvTotalDice = 1.0f / static_cast<float>(Cfg::kTotalDice);
  out->push_back(static_cast<float>(d.current_bid_count) * kInvTotalDice);

  // current_bid_face one-hot[6]: idx 0 = no-bid (face=0); idx 1..5 = face 2..6.
  for (int idx = 0; idx < 6; ++idx) {
    int target = (idx == 0) ? 0 : (idx + 1);  // 0, 2, 3, 4, 5, 6
    out->push_back(d.current_bid_face == target ? 1.0f : 0.0f);
  }

  // per-seat last-5 bid block, rotated so seat k=0 is the perspective.
  const auto* lt =
      dynamic_cast<const LiarsdiceBeliefTracker<NPlayers>*>(tracker);
  if (lt == nullptr) {
    // No tracker installed → all-zero bid history block.
    for (int i = 0; i < Cfg::kBidHistoryFeatures; ++i) {
      out->push_back(0.0f);
    }
  } else {
    for (int k = 0; k < NPlayers; ++k) {
      const int seat = (perspective_player + k) % NPlayers;
      append_seat_bid_block<NPlayers>(out, lt, seat);
    }
  }
}

// ============================================================
// LiarsdiceBeliefTracker
// ============================================================

template <int NPlayers>
void LiarsdiceBeliefTracker<NPlayers>::init(
    IGameState& /*state*/, int perspective, const AnyMap& /*payload*/) {
  perspective_ = perspective;
  for (auto& v : bids_by_player_) v.clear();
  pi_valid_ = false;
  pi_observer_ = -1;
  for (auto& row : pi_outcome_) row.fill(0.0f);
}

template <int NPlayers>
void LiarsdiceBeliefTracker<NPlayers>::observe_public_event(
    int actor, ActionId action,
    const std::vector<PublicEvent>& /*events*/) {
  using Cfg = LiarsdiceConfig<NPlayers>;
  if (actor < 0 || actor >= NPlayers) return;
  if (is_challenge(action, Cfg::kTotalDice)) return;  // single-round terminal
  const int x = bid_count_from_action(action);
  const int y = bid_face_from_action(action);
  bids_by_player_[static_cast<size_t>(actor)].emplace_back(x, y);
}

template <int NPlayers>
void LiarsdiceBeliefTracker<NPlayers>::randomize_unseen(
    IGameState& state, int observer, std::mt19937_64& rng) const {
  auto& s = checked_cast<LiarsdiceState<NPlayers>>(state);
  auto& d = s.data;

  // For each opp != observer, sample one of the 252 enumerated Σ=5
  // outcomes from the cached posterior (uniform fallback if no belief
  // net is installed). Write back as dice_count[opp][f] = n_f.
  const bool use_belief = pi_valid_ && pi_observer_ == observer;
  for (int opp = 0; opp < NPlayers; ++opp) {
    if (opp == observer) continue;
    // PLAN §1.5.4: opp_idx = (opp - observer - 1 + N) % N — clockwise
    // index from the observer.
    const int opp_idx = (opp - observer - 1 + NPlayers) % NPlayers;

    int outcome_idx = -1;
    if (use_belief) {
      const auto& pi = pi_outcome_[static_cast<size_t>(opp_idx)];
      std::uniform_real_distribution<float> uni(0.0f, 1.0f);
      const float r = uni(rng);
      float acc = 0.0f;
      for (int i = 0; i < kOutcomeCount; ++i) {
        acc += pi[static_cast<size_t>(i)];
        if (r <= acc) {
          outcome_idx = i;
          break;
        }
      }
      if (outcome_idx < 0) outcome_idx = kOutcomeCount - 1;
    } else {
      std::uniform_int_distribution<int> uni(0, kOutcomeCount - 1);
      outcome_idx = uni(rng);
    }

    const Outcome& out = index_to_outcome(outcome_idx);
    for (int f = 0; f < kFaceCount; ++f) {
      d.dice_count[static_cast<size_t>(opp)][static_cast<size_t>(f)] =
          out[static_cast<size_t>(f)];
    }
  }
}

template <int NPlayers>
void LiarsdiceBeliefTracker<NPlayers>::prepare_for_root(
    const IGameState& root_state, int root_player,
    const IBeliefFeatureExtractor* extractor,
    const IBeliefEvaluator* evaluator) {
  pi_valid_ = false;
  pi_observer_ = -1;
  for (auto& row : pi_outcome_) row.fill(0.0f);

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
  constexpr int kCountBins = kDicePerPlayer + 1;       // 6
  constexpr int kPerOppLogits = kFaceCount * kCountBins;  // 36
  const int expected = (NPlayers - 1) * kPerOppLogits;
  if (static_cast<int>(logits.size()) != expected) return;

  // Per-opp output is 36 = 6 faces × 6 count bins. For each face do an
  // independent 6-way softmax at fixed temperature 1 to target
  // per-face marginal q_f[c] = P(n_f = c). The 252-joint over Σ=5
  // multisets is initialized by score(o) = ∏_f q_f[n_f] (renormalized),
  // then refined by IPF so that every per-face marginal of the joint
  // matches q_f exactly (up to tol). Without IPF the Σ=5 constraint
  // reweights ∏ q_f and the resulting joint's marginals drift away from
  // the network's outputs.
  const float inv_T = 1.0f / board_ai::kBeliefTemperature;
  constexpr int kIpfMaxIter = 10;
  constexpr double kIpfTol = 1e-4;
  for (int i = 0; i < NPlayers - 1; ++i) {
    const int base = i * kPerOppLogits;

    // 6 × 6 marginal table: p_fc[f][c] = P(n_f = c) — IPF target.
    std::array<std::array<double, kCountBins>, kFaceCount> p_fc{};
    for (int f = 0; f < kFaceCount; ++f) {
      const int fbase = base + f * kCountBins;
      double m = -std::numeric_limits<double>::infinity();
      for (int c = 0; c < kCountBins; ++c) {
        const double z =
            static_cast<double>(logits[static_cast<size_t>(fbase + c)]) *
            inv_T;
        if (z > m) m = z;
      }
      if (!std::isfinite(m)) return;
      double sum = 0.0;
      std::array<double, kCountBins> ez{};
      for (int c = 0; c < kCountBins; ++c) {
        const double z =
            static_cast<double>(logits[static_cast<size_t>(fbase + c)]) *
            inv_T;
        ez[static_cast<size_t>(c)] = std::exp(z - m);
        sum += ez[static_cast<size_t>(c)];
      }
      if (sum <= 0.0 || !std::isfinite(sum)) return;
      for (int c = 0; c < kCountBins; ++c) {
        p_fc[static_cast<size_t>(f)][static_cast<size_t>(c)] =
            ez[static_cast<size_t>(c)] / sum;
      }
    }

    // IPF starting point: π[k] = ∏_f q_f[n_f^(k)], then renormalize on the
    // 252-simplex.
    std::array<double, kOutcomeCount> pi{};
    double total = 0.0;
    for (int k = 0; k < kOutcomeCount; ++k) {
      const Outcome& o = index_to_outcome(k);
      double score = 1.0;
      for (int f = 0; f < kFaceCount; ++f) {
        const int n_f = o[static_cast<size_t>(f)];
        score *= p_fc[static_cast<size_t>(f)][static_cast<size_t>(n_f)];
      }
      pi[static_cast<size_t>(k)] = score;
      total += score;
    }
    if (!(total > 0.0) || !std::isfinite(total)) return;
    const double inv_total = 1.0 / total;
    for (int k = 0; k < kOutcomeCount; ++k) pi[static_cast<size_t>(k)] *= inv_total;

    // IPF: for each face f, rescale π so its marginal in dimension f
    // equals q_f. Iterate until max marginal error < tol or kIpfMaxIter.
    // Bins with q_f[c] > 0 but current marginal == 0 are unreachable
    // under the support of π and are left alone (standard IPF behavior).
    for (int it = 0; it < kIpfMaxIter; ++it) {
      double max_err = 0.0;
      for (int f = 0; f < kFaceCount; ++f) {
        std::array<double, kCountBins> m_f{};
        for (int k = 0; k < kOutcomeCount; ++k) {
          const int n_f = index_to_outcome(k)[static_cast<size_t>(f)];
          m_f[static_cast<size_t>(n_f)] += pi[static_cast<size_t>(k)];
        }
        std::array<double, kCountBins> scale{};
        for (int c = 0; c < kCountBins; ++c) {
          const double q = p_fc[static_cast<size_t>(f)][static_cast<size_t>(c)];
          const double m = m_f[static_cast<size_t>(c)];
          scale[static_cast<size_t>(c)] = (m > 0.0) ? (q / m) : 1.0;
          const double err = std::fabs(q - m);
          if (err > max_err) max_err = err;
        }
        for (int k = 0; k < kOutcomeCount; ++k) {
          const int n_f = index_to_outcome(k)[static_cast<size_t>(f)];
          pi[static_cast<size_t>(k)] *= scale[static_cast<size_t>(n_f)];
        }
      }
      if (max_err < kIpfTol) break;
    }

    // Final renormalize (IPF preserves Σπ=1 in exact arithmetic; floating-
    // point drift makes the safety divide cheap and worthwhile).
    double final_total = 0.0;
    for (int k = 0; k < kOutcomeCount; ++k) final_total += pi[static_cast<size_t>(k)];
    if (!(final_total > 0.0) || !std::isfinite(final_total)) return;
    const double inv_final = 1.0 / final_total;
    auto& row = pi_outcome_[static_cast<size_t>(i)];
    for (int k = 0; k < kOutcomeCount; ++k) {
      row[static_cast<size_t>(k)] =
          static_cast<float>(pi[static_cast<size_t>(k)] * inv_final);
    }
  }
  pi_valid_ = true;
  pi_observer_ = root_player;
}

template <int NPlayers>
AnyMap LiarsdiceBeliefTracker<NPlayers>::serialize() const {
  // Canonical perspective-agnostic dump: flatten bids_by_player_ into a
  // (seat, bid-index, X|Y) tensor. Two trackers fed the same public
  // action stream produce equal output.
  AnyMap out;
  std::vector<int> bid_lengths;
  bid_lengths.reserve(NPlayers);
  std::vector<int> flat_bids;
  for (int p = 0; p < NPlayers; ++p) {
    const auto& v = bids_by_player_[static_cast<size_t>(p)];
    bid_lengths.push_back(static_cast<int>(v.size()));
    for (const auto& bid : v) {
      flat_bids.push_back(bid.first);
      flat_bids.push_back(bid.second);
    }
  }
  out["bid_lengths"] = bid_lengths;
  out["flat_bids"] = flat_bids;
  return out;
}

// ============================================================
// LiarsdiceBeliefFeatureExtractor — per-seat last-5 bid slice only.
// ============================================================

template <int NPlayers>
void LiarsdiceBeliefFeatureExtractor<NPlayers>::extract(
    const IGameState& /*masked_state*/, int perspective_player,
    const IBeliefTracker* tracker, std::vector<float>* out) const {
  out->clear();
  out->reserve(static_cast<size_t>(kFeatureDim));

  const auto* lt =
      dynamic_cast<const LiarsdiceBeliefTracker<NPlayers>*>(tracker);
  if (lt == nullptr) {
    for (int i = 0; i < kFeatureDim; ++i) out->push_back(0.0f);
    return;
  }
  for (int k = 0; k < NPlayers; ++k) {
    const int seat = (perspective_player + k) % NPlayers;
    append_seat_bid_block<NPlayers>(out, lt, seat);
  }
}

// ============================================================
// LiarsdiceBeliefLabelExtractor — GT-side opp dice → per-face one-hot.
// ============================================================

template <int NPlayers>
void LiarsdiceBeliefLabelExtractor<NPlayers>::extract(
    const IGameState& truth_state, int observer,
    std::vector<std::vector<int>>* out_hand_counts,
    std::vector<int>* out_remaining,
    std::vector<int>* out_alive) const {
  const auto& s = checked_cast<LiarsdiceState<NPlayers>>(truth_state);
  const auto& d = s.data;

  // hand_counts is repurposed as a flat per-face one-hot row per opp:
  // length kClassCount = 6 faces * 6 count bins. For each face f set
  // slot (f * kCountBins + n_f) = 1.
  constexpr int kCountBins = kDicePerPlayer + 1;
  out_hand_counts->assign(static_cast<size_t>(NPlayers - 1),
                          std::vector<int>(kClassCount, 0));
  for (int i = 0; i < NPlayers - 1; ++i) {
    const int opp = (observer + 1 + i) % NPlayers;  // PLAN §1.5.4
    int total = 0;
    for (int f = 0; f < kFaceCount; ++f) {
      const int n_f =
          d.dice_count[static_cast<size_t>(opp)][static_cast<size_t>(f)];
      if (n_f < 0 || n_f > kDicePerPlayer) {
        throw std::logic_error(
            "LiarsdiceBeliefLabelExtractor: opp dice_count out of range");
      }
      total += n_f;
      (*out_hand_counts)[static_cast<size_t>(i)]
                       [static_cast<size_t>(f * kCountBins + n_f)] = 1;
    }
    if (total != kDicePerPlayer) {
      throw std::logic_error(
          "LiarsdiceBeliefLabelExtractor: opp dice_count not Σ=5");
    }
  }
  // remaining / alive_per_opp unused in liarsdice's belief-target wiring,
  // but the framework requires their shapes. Emit zero / one fillers.
  out_remaining->assign(static_cast<size_t>(kClassCount), 0);
  out_alive->assign(static_cast<size_t>(NPlayers - 1), 1);
}

// ============================================================
// Explicit template instantiations
// ============================================================

template class LiarsdiceFeatureEncoder<2>;
template class LiarsdiceFeatureEncoder<3>;
template class LiarsdiceFeatureEncoder<4>;
template class LiarsdiceBeliefTracker<2>;
template class LiarsdiceBeliefTracker<3>;
template class LiarsdiceBeliefTracker<4>;
template class LiarsdiceBeliefFeatureExtractor<2>;
template class LiarsdiceBeliefFeatureExtractor<3>;
template class LiarsdiceBeliefFeatureExtractor<4>;
template class LiarsdiceBeliefLabelExtractor<2>;
template class LiarsdiceBeliefLabelExtractor<3>;
template class LiarsdiceBeliefLabelExtractor<4>;

}  // namespace board_ai::liarsdice
