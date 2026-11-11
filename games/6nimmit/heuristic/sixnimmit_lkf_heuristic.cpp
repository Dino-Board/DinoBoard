#include "sixnimmit_lkf_heuristic.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "../6nimmit_state.h"
#include "../6nimmit_rules.h"
#include "sixnimmit_rule_heuristic.h"

namespace board_ai::sixnimmt {

namespace {

// ---------------------------------------------------------------------
// Algorithm constants from games/6nimmit/参考/lkf_zhc_cyb2.cpp.
// ---------------------------------------------------------------------
constexpr int kRolloutT = 5;
constexpr double kInf = 1e20;
constexpr double kDiscount = 0.9;

struct LkfStack {
  int t = 0;   // top card (0 = empty)
  int n = 0;   // length
  int h = 0;  // sum of head values

  bool full() const { return n == kRowCapacity; }
  void push(int x) {
    t = x;
    h += head_value(x);
    ++n;
  }
  int refresh(int x) {
    int res = -h;
    n = 1;
    t = x;
    h = head_value(x);
    return res;
  }
  int value() const { return -h; }
};

// `ecb_local`: cards already publicly visible (table + bullhead piles +
// already-revealed apply queue cards). Members of `c` (own hand) start
// out NOT in ecb_local; the dfs flips ecb_local[ec]=true while exploring
// a chance branch, mirroring the reference's global ecb[].
struct LkfStatus {
  std::array<LkfStack, kNumRows> s{};
  // cards: own remaining hand (size n)
  std::array<int, kHandSize> c{};
  int n = 0;

  void remove(int mc) {
    int q = 0;
    for (int i = 0; i < n; ++i) {
      if (c[static_cast<size_t>(i)] != mc) {
        c[static_cast<size_t>(q++)] = c[static_cast<size_t>(i)];
      }
    }
    n = q;
  }

  // Insert single card x; return score delta for the player playing x
  // (negative when x eats a row or refreshes onto an empty slot).
  int add(int x) {
    int r = -1;
    for (int i = 0; i < kNumRows; ++i) {
      if (s[static_cast<size_t>(i)].t < x &&
          (r == -1 || s[static_cast<size_t>(i)].t >
                          s[static_cast<size_t>(r)].t)) {
        r = i;
      }
    }
    if (r == -1) {
      int maxv = -9999;
      for (int i = 0; i < kNumRows; ++i) {
        int v = s[static_cast<size_t>(i)].value();
        if (v > maxv) {
          maxv = v;
          r = i;
        }
      }
      return s[static_cast<size_t>(r)].refresh(x);
    } else if (s[static_cast<size_t>(r)].full()) {
      return s[static_cast<size_t>(r)].refresh(x);
    } else {
      s[static_cast<size_t>(r)].push(x);
      return 0;
    }
  }

  // SMALL phase: x must refresh some row, opp's y is then auto-placed.
  // Returns the row index that minimizes (cost(x) + opp_gain(y)).
  int best_small_stack(int x, int y) const {
    int maxv = -10000;
    int maxi = 0;
    for (int i = 0; i < kNumRows; ++i) {
      LkfStatus ns = *this;
      int v = 0;
      v += ns.s[static_cast<size_t>(i)].refresh(x);
      v -= ns.add(y);
      if (v > maxv) {
        maxv = v;
        maxi = i;
      }
    }
    return maxi;
  }

  // Two cards simultaneously; returns net score for the x player (the
  // player of the smaller card).
  int add(int x, int y) {
    if (x > y) return -add(y, x);
    int r = -1;
    for (int i = 0; i < kNumRows; ++i) {
      if (s[static_cast<size_t>(i)].t < x &&
          (r == -1 || s[static_cast<size_t>(i)].t >
                          s[static_cast<size_t>(r)].t)) {
        r = i;
      }
    }
    if (r == -1) {
      r = best_small_stack(x, y);
      return s[static_cast<size_t>(r)].refresh(x) - add(y);
    } else if (s[static_cast<size_t>(r)].full()) {
      return s[static_cast<size_t>(r)].refresh(x) - add(y);
    } else {
      s[static_cast<size_t>(r)].push(x);
      return -add(y);
    }
  }

  // Monte Carlo rollout: shuffle our remaining hand, pair each with a
  // uniform-random opponent card from the full 1..104 range (matches the
  // reference's `rand() % 104 + 1`), play out the sequence.
  double estimate(std::mt19937& rng) const {
    double res = 0.0;
    std::uniform_int_distribution<int> opp_dist(1, kNumCards);
    for (int trial = 0; trial < kRolloutT; ++trial) {
      LkfStatus ns = *this;
      std::array<int, kHandSize> cm{};
      for (int i = 0; i < n; ++i) cm[static_cast<size_t>(i)] = c[static_cast<size_t>(i)];
      std::shuffle(cm.begin(), cm.begin() + n, rng);
      for (int i = 0; i < n; ++i) {
        int d = opp_dist(rng);
        res += ns.add(cm[static_cast<size_t>(i)], d);
      }
    }
    res /= kRolloutT;
    return res;
  }
};

// log(C(n, m)) via precomputed log-factorials.
struct CombTables {
  std::array<double, kNumCards + 1> log_fact{};
  // cmaxprob[nc][k][i] = P(max-rank == i in a k-subset of nc items).
  // nc up to kNumCards, k up to kHandSize.
  std::array<std::array<std::array<double, kNumCards>, kHandSize + 1>,
             kNumCards + 1>
      cmaxprob{};

  CombTables() {
    log_fact[0] = 0.0;
    for (int i = 1; i <= kNumCards; ++i) {
      log_fact[static_cast<size_t>(i)] =
          log_fact[static_cast<size_t>(i - 1)] + std::log(static_cast<double>(i));
    }
    auto C_log = [&](int n, int m) {
      return log_fact[static_cast<size_t>(n)] -
             log_fact[static_cast<size_t>(m)] -
             log_fact[static_cast<size_t>(n - m)];
    };
    for (int nc = 0; nc <= kNumCards; ++nc) {
      int kmax = std::min(kHandSize, nc);
      for (int k = 1; k <= kmax; ++k) {
        for (int i = 0; i < nc; ++i) {
          if (nc - i >= k) {
            cmaxprob[static_cast<size_t>(nc)][static_cast<size_t>(k)]
                    [static_cast<size_t>(i)] =
                std::exp(C_log(nc - i - 1, k - 1) - C_log(nc, k));
          }
        }
      }
    }
  }
};

const CombTables& comb_tables() {
  static const CombTables t;
  return t;
}

struct CybDist {
  int n = 0;
  std::array<std::pair<double, int>, kNumCards> dist{};
};

// 2-stage CYB distribution over likely opponent-card values, given the
// current LkfStatus + ecb (cards publicly seen).
//
// `ecb_local` is the same in/out array used by dfs (sized kNumCards+1,
// 1-indexed, true = card already publicly observed and thus excluded
// from opp's possible holdings).
void cyb_get_prob_dist(const LkfStatus& s,
                       const std::array<bool, kNumCards + 1>& ecb_local,
                       CybDist* out) {
  const auto& tab = comb_tables();
  std::array<LkfStack, kNumRows> si = s.s;
  std::sort(si.begin(), si.end(),
            [](const LkfStack& a, const LkfStack& b) { return a.t < b.t; });

  std::array<int, kNumRows + 1> bp{};
  for (int r = 0; r < kNumRows; ++r) bp[static_cast<size_t>(r)] = si[static_cast<size_t>(r)].t;
  bp[kNumRows] = kNumCards + 1;  // sentinel = 105

  double minh = kInf;
  for (int r = 0; r < kNumRows; ++r) {
    minh = std::min(minh, static_cast<double>(si[static_cast<size_t>(r)].h));
  }

  std::array<bool, kNumCards + 1> cb{};
  std::array<double, kNumCards + 1> prob{};
  for (int i = 1; i <= kNumCards; ++i) cb[static_cast<size_t>(i)] = ecb_local[static_cast<size_t>(i)];
  for (int i = 0; i < s.n; ++i) cb[static_cast<size_t>(s.c[static_cast<size_t>(i)])] = false;

  int ncb = 0;
  for (int i = 1; i <= kNumCards; ++i) if (!cb[static_cast<size_t>(i)]) ++ncb;
  if (ncb == 0) {
    out->n = 0;
    return;
  }

  for (int i = 1; i <= kNumCards; ++i) {
    prob[static_cast<size_t>(i)] = cb[static_cast<size_t>(i)] ? 0.0 : (1.0 / ncb);
  }

  constexpr int kStages = 2;
  for (int stage = 1; stage <= kStages; ++stage) {
    if (stage == kStages) {
      for (int i = 0; i < s.n; ++i) cb[static_cast<size_t>(s.c[static_cast<size_t>(i)])] = true;
    }
    for (int i = 1; i <= kNumCards; ++i) prob[static_cast<size_t>(i)] += prob[static_cast<size_t>(i - 1)];

    int dn = 0;
    for (int i = 1; i < bp[0]; ++i) {
      if (!cb[static_cast<size_t>(i)]) {
        out->dist[static_cast<size_t>(dn++)] = {-minh, i};
      }
    }
    for (int r = 0; r < kNumRows; ++r) {
      for (int i = bp[static_cast<size_t>(r)] + 1; i < bp[static_cast<size_t>(r + 1)]; ++i) {
        if (!cb[static_cast<size_t>(i)]) {
          if (si[static_cast<size_t>(r)].n == 5) {
            double p = 1.0 - (prob[static_cast<size_t>(i)] -
                              prob[static_cast<size_t>(bp[static_cast<size_t>(r)] - 1)]);
            out->dist[static_cast<size_t>(dn++)] = {-si[static_cast<size_t>(r)].h * p, i};
          } else if (si[static_cast<size_t>(r)].n == 4) {
            double p = prob[static_cast<size_t>(i)] -
                       prob[static_cast<size_t>(bp[static_cast<size_t>(r)] - 1)];
            double q = prob[static_cast<size_t>(bp[static_cast<size_t>(r + 1)])] -
                       prob[static_cast<size_t>(i)];
            out->dist[static_cast<size_t>(dn++)] = {si[static_cast<size_t>(r)].h * (q - p), i};
          } else {
            out->dist[static_cast<size_t>(dn++)] = {0.0, i};
          }
        }
      }
    }

    if (stage == kStages) {
      // Random shuffle + keep first half (matches reference's
      // random_shuffle + dist_n = ceil(dist_n * 0.5)).
      // The lkf reference relies on rand() global; we use a fixed local
      // engine seeded by the dist contents for reproducibility.
      std::seed_seq seed{dn, static_cast<int>(std::lround(minh)), bp[0], bp[1]};
      std::mt19937 sh_rng(seed);
      std::shuffle(out->dist.begin(), out->dist.begin() + dn, sh_rng);
      dn = static_cast<int>(std::ceil(dn * 0.5));
    }

    std::sort(out->dist.begin(), out->dist.begin() + dn,
              [](const std::pair<double, int>& a,
                 const std::pair<double, int>& b) { return a.first > b.first; });

    for (int i = 0, j = 0; i < dn; i = j) {
      while (j < dn && std::abs(out->dist[static_cast<size_t>(i)].first -
                                out->dist[static_cast<size_t>(j)].first) < 1e-3) {
        ++j;
      }
      double p_sum = 0.0;
      int hand_size_for_lookup = std::min(s.n, kHandSize);
      for (int k = i; k < j; ++k) {
        p_sum += tab.cmaxprob[static_cast<size_t>(dn)]
                              [static_cast<size_t>(hand_size_for_lookup)]
                              [static_cast<size_t>(k)];
      }
      double mean = (j > i) ? (p_sum / (j - i)) : 0.0;
      for (int k = i; k < j; ++k) out->dist[static_cast<size_t>(k)].first = mean;
    }

    if (stage == kStages) {
      while (dn > 0 && out->dist[static_cast<size_t>(dn - 1)].first < 0.001) --dn;
    } else {
      for (int i = 1; i <= kNumCards; ++i) prob[static_cast<size_t>(i)] = 0.0;
      double tp = 0.0;
      for (int i = 0; i < dn; ++i) {
        prob[static_cast<size_t>(out->dist[static_cast<size_t>(i)].second)] =
            out->dist[static_cast<size_t>(i)].first;
        tp += out->dist[static_cast<size_t>(i)].first;
      }
      if (tp > 0) {
        for (int i = 1; i <= kNumCards; ++i) prob[static_cast<size_t>(i)] /= tp;
      }
    }
    out->n = dn;
  }
}

// Minimax with chance node — returns (value, best_card_at_root).
double dfs(const LkfStatus& s, int depth, int depth_limit,
           std::array<bool, kNumCards + 1>& ecb_local,
           std::mt19937& rollout_rng, int* best_card) {
  if (s.n == 0) return 0.0;
  if (depth == depth_limit) return s.estimate(rollout_rng);

  CybDist cyb;
  cyb_get_prob_dist(s, ecb_local, &cyb);

  double res = -kInf;
  for (int i = 0; i < s.n; ++i) {
    int mc = s.c[static_cast<size_t>(i)];
    double cur = 0.0;
    double tp = 0.0;
    for (int j = 0; j < cyb.n; ++j) {
      double p = cyb.dist[static_cast<size_t>(j)].first;
      int ec = cyb.dist[static_cast<size_t>(j)].second;
      LkfStatus ns = s;
      ns.remove(mc);
      int t = ns.add(mc, ec);
      tp += p;
      ecb_local[static_cast<size_t>(ec)] = true;
      cur += p * (t + kDiscount * dfs(ns, depth + 1, depth_limit,
                                      ecb_local, rollout_rng, nullptr));
      ecb_local[static_cast<size_t>(ec)] = false;
    }
    if (tp > 0) cur /= tp;
    if (cur > res) {
      res = cur;
      if (depth == 0 && best_card) *best_card = mc;
    }
  }
  return res;
}

template <int NPlayers>
void collect_lkf_status(const SixNimmtState<NPlayers>& s,
                        LkfStatus* out) {
  const auto& d = s.data;
  for (int r = 0; r < kNumRows; ++r) {
    LkfStack& st = out->s[static_cast<size_t>(r)];
    st = LkfStack{};
    for (int sidx = 0; sidx < kRowCapacity; ++sidx) {
      int c = d.table[static_cast<size_t>(r)][static_cast<size_t>(sidx)];
      if (c >= 1 && c <= kNumCards) {
        st.t = c;
        st.h += head_value(c);
        ++st.n;
      } else {
        break;
      }
    }
  }
  int hand_n = 0;
  const int me = d.current_player;
  for (int c = 1; c <= kNumCards; ++c) {
    if (d.hand[static_cast<size_t>(me)][static_cast<size_t>(c - 1)] == 1) {
      out->c[static_cast<size_t>(hand_n++)] = c;
    }
  }
  out->n = hand_n;
}

template <int NPlayers>
void build_ecb(const SixNimmtState<NPlayers>& s,
               std::array<bool, kNumCards + 1>* ecb_local) {
  const auto& d = s.data;
  ecb_local->fill(false);
  for (int r = 0; r < kNumRows; ++r) {
    for (int sidx = 0; sidx < kRowCapacity; ++sidx) {
      int c = d.table[static_cast<size_t>(r)][static_cast<size_t>(sidx)];
      if (c >= 1 && c <= kNumCards) (*ecb_local)[static_cast<size_t>(c)] = true;
    }
  }
  for (int p = 0; p < NPlayers; ++p) {
    for (int c = 1; c <= kNumCards; ++c) {
      if (d.bullhead_pile[static_cast<size_t>(p)]
                          [static_cast<size_t>(c - 1)] == 1) {
        (*ecb_local)[static_cast<size_t>(c)] = true;
      }
    }
  }
  // After seal, every committed seat's pending_commits is viz=public
  // and holds its revealed card; an unsealed (mid-commit) round still
  // has owner-only viz so heuristics from any single perspective only
  // see their own pending. Either way a non-zero pending_commits[p]
  // for p != observer can only happen after seal — safe to fold.
  for (int p = 0; p < NPlayers; ++p) {
    int c = d.pending_commits[static_cast<size_t>(p)];
    if (c >= 1 && c <= kNumCards) (*ecb_local)[static_cast<size_t>(c)] = true;
  }
}

template <int NPlayers>
HeuristicResult pick_2p(IGameState& state, const IGameRules& rules,
                        std::uint64_t rng_seed) {
  const auto& s = checked_cast<SixNimmtState<NPlayers>>(state);
  const auto& d = s.data;
  auto legal = rules.legal_actions(state);

  HeuristicResult result;
  result.actions = legal;
  result.scores.assign(legal.size(), 0.0);
  if (legal.empty()) return result;

  LkfStatus status;
  collect_lkf_status<NPlayers>(s, &status);
  std::array<bool, kNumCards + 1> ecb_local{};
  build_ecb<NPlayers>(s, &ecb_local);

  // SMALL phase: my card is pending_commits[small_player]; opp's card
  // is the only other non-zero pending_commits entry (viz=public after
  // seal, so the masked-state heuristic sees them all).
  if (d.small_phase_active) {
    int small_p = d.small_player;
    int my_card = (small_p >= 0 && small_p < NPlayers)
                      ? d.pending_commits[static_cast<size_t>(small_p)]
                      : 0;
    int opp_card = 0;
    for (int p = 0; p < NPlayers; ++p) {
      if (p == small_p) continue;
      int c = d.pending_commits[static_cast<size_t>(p)];
      if (c >= 1 && c <= kNumCards) { opp_card = c; break; }
    }
    if (opp_card == 0) opp_card = my_card;  // degenerate: pick any
    int chosen_row = status.best_small_stack(my_card, opp_card);
    ActionId target = kSmallChooseRowOffset + chosen_row;
    bool legal_ok = false;
    for (ActionId a : legal) if (a == target) { legal_ok = true; break; }
    if (!legal_ok) target = legal.front();
    for (size_t i = 0; i < legal.size(); ++i) {
      result.scores[i] = (legal[i] == target) ? 1.0 : 0.0;
    }
    return result;
  }

  // COMMIT phase: depth=2 (hand>4) or depth=3 (hand<=4).
  int depth_limit = (status.n > 4) ? 2 : 3;
  std::mt19937 rollout_rng(rng_seed != 0 ? rng_seed : 0xC0FFEEULL);
  int best_card = 0;
  dfs(status, 0, depth_limit, ecb_local, rollout_rng, &best_card);
  if (best_card < 1 || best_card > kNumCards) {
    // Defensive fallback to first legal commit.
    for (ActionId a : legal) {
      if (a >= kCommitOffset && a < kCommitOffset + kNumCards) {
        best_card = a - kCommitOffset + 1;
        break;
      }
    }
  }

  ActionId target = kCommitOffset + (best_card - 1);
  bool legal_ok = false;
  for (ActionId a : legal) if (a == target) { legal_ok = true; break; }
  if (!legal_ok) target = legal.front();
  for (size_t i = 0; i < legal.size(); ++i) {
    result.scores[i] = (legal[i] == target) ? 1.0 : 0.0;
  }
  return result;
}

}  // namespace

template <int NPlayers>
HeuristicResult SixNimmtLkfHeuristic<NPlayers>::pick(
    IGameState& masked_state,
    const IBeliefTracker* tracker,
    const IGameRules& rules,
    std::uint64_t rng_seed) const {
  if constexpr (NPlayers == 2) {
    return pick_2p<NPlayers>(masked_state, rules, rng_seed);
  } else {
    // 3p/4p: lkf's CYB distribution is built for 1v1; multi-opp chance
    // nodes need separate work. Defer to rule_v1.
    SixNimmtRuleHeuristic<NPlayers> fallback("lkf_v1_fallback_rule_v1");
    return fallback.pick(masked_state, tracker, rules, rng_seed);
  }
}

template class SixNimmtLkfHeuristic<2>;
template class SixNimmtLkfHeuristic<3>;
template class SixNimmtLkfHeuristic<4>;

}  // namespace board_ai::sixnimmt
