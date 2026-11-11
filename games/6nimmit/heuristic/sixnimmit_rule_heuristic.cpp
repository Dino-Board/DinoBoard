#include "sixnimmit_rule_heuristic.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include "../6nimmit_state.h"
#include "../6nimmit_rules.h"

namespace board_ai::sixnimmt {

namespace {

struct StackInfo {
  int row;
  int top;
  int length;
  int heads;
};

struct IntervalInfo {
  int num;
  std::vector<int> cards;
  int stack_idx = -1;
  bool bad = false;
  bool danger = false;
  bool dead = false;
};

template <int NPlayers>
void collect_stacks(const SixNimmtState<NPlayers>& s,
                    std::array<StackInfo, kNumRows>* stacks) {
  const auto& d = s.data;
  for (int r = 0; r < kNumRows; ++r) {
    int len = 0;
    int tail = 0;
    int heads_sum = 0;
    for (int sidx = 0; sidx < kRowCapacity; ++sidx) {
      int c = d.table[static_cast<size_t>(r)][static_cast<size_t>(sidx)];
      if (c >= 1 && c <= kNumCards) {
        tail = c;
        heads_sum += head_value(c);
        ++len;
      } else {
        break;
      }
    }
    (*stacks)[static_cast<size_t>(r)] = StackInfo{r, tail, len, heads_sum};
  }
}

std::array<StackInfo, kNumRows> sorted_stacks(
    const std::array<StackInfo, kNumRows>& stacks) {
  std::array<StackInfo, kNumRows> out = stacks;
  std::sort(out.begin(), out.end(),
            [](const StackInfo& a, const StackInfo& b) {
              return a.top < b.top;
            });
  return out;
}

template <int NPlayers>
std::array<IntervalInfo, 5> build_intervals(
    const SixNimmtState<NPlayers>& s, int perspective,
    const std::array<StackInfo, kNumRows>& selfstacks) {
  const auto& d = s.data;
  std::array<IntervalInfo, 5> intervals{};
  for (int i = 0; i < 5; ++i) intervals[static_cast<size_t>(i)].num = i;

  std::vector<int> hand;
  hand.reserve(kNumCards);
  for (int c = 1; c <= kNumCards; ++c) {
    if (d.hand[static_cast<size_t>(perspective)]
              [static_cast<size_t>(c - 1)] == 1) {
      hand.push_back(c);
    }
  }

  for (int card : hand) {
    int t = 0;
    for (int i = 0; i < kNumRows; ++i) {
      if (selfstacks[static_cast<size_t>(i)].top < card) t = i + 1;
    }
    intervals[static_cast<size_t>(t)].cards.push_back(card);
  }

  for (int i = 1; i < 5; ++i) {
    auto& iv = intervals[static_cast<size_t>(i)];
    iv.stack_idx = i - 1;
    const auto& st = selfstacks[static_cast<size_t>(i - 1)];
    if (static_cast<int>(iv.cards.size()) >= 2 && st.length <= 2) iv.bad = true;
    if (st.length == kRowCapacity) iv.dead = true;
  }
  for (int i = 2; i < 5; ++i) {
    auto& iv = intervals[static_cast<size_t>(i)];
    const auto& prev_st = selfstacks[static_cast<size_t>(i - 2)];
    if (prev_st.length == kRowCapacity) {
      iv.danger = true;
      iv.bad = false;
    }
  }
  return intervals;
}

int cost_for_card(const std::array<StackInfo, kNumRows>& stacks, int card) {
  int arg = 0;
  int max_top = 0;
  for (int i = 0; i < kNumRows; ++i) {
    int top = stacks[static_cast<size_t>(i)].top;
    if (top < card && top > max_top) {
      max_top = top;
      arg = i;
    }
  }
  if (max_top == 0) return 0;
  if (stacks[static_cast<size_t>(arg)].length == kRowCapacity) {
    return stacks[static_cast<size_t>(arg)].heads;
  }
  return 0;
}

template <int NPlayers>
int findsmall_index(const std::array<StackInfo, kNumRows>& stacks,
                    int my_card, int opp_card_estimate) {
  if constexpr (NPlayers == 2) {
    int best_idx = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (int r = 0; r < kNumRows; ++r) {
      auto sim = stacks;
      sim[static_cast<size_t>(r)].top = my_card;
      sim[static_cast<size_t>(r)].length = 1;
      sim[static_cast<size_t>(r)].heads = head_value(my_card);
      double score = static_cast<double>(cost_for_card(sim, opp_card_estimate)) -
                     static_cast<double>(stacks[static_cast<size_t>(r)].heads);
      if (score > best_score) {
        best_score = score;
        best_idx = r;
      }
    }
    return best_idx;
  } else {
    int min_heads = std::numeric_limits<int>::max();
    int best_idx = 0;
    int best_top = -1;
    for (int r = 0; r < kNumRows; ++r) {
      const auto& st = stacks[static_cast<size_t>(r)];
      if (st.heads < min_heads ||
          (st.heads == min_heads && st.top > best_top)) {
        min_heads = st.heads;
        best_top = st.top;
        best_idx = r;
      }
    }
    return best_idx;
  }
}

template <int NPlayers>
int commit_choice(
    const std::array<StackInfo, kNumRows>& selfstacks,
    const std::array<IntervalInfo, 5>& intervals) {
  auto IM = [&](int i) -> const IntervalInfo& {
    return intervals[static_cast<size_t>(4 - i)];
  };

  std::vector<int> all_cards;
  for (int i = 0; i < 5; ++i) {
    for (int c : intervals[static_cast<size_t>(i)].cards) {
      all_cards.push_back(c);
    }
  }
  std::sort(all_cards.begin(), all_cards.end());

  const auto& smalls = intervals[0].cards;
  std::vector<int> deadcards;
  for (int i = 1; i < 5; ++i) {
    if (intervals[static_cast<size_t>(i)].dead) {
      for (int c : intervals[static_cast<size_t>(i)].cards) {
        deadcards.push_back(c);
      }
    }
  }
  std::sort(deadcards.begin(), deadcards.end());

  if (static_cast<int>(deadcards.size() + smalls.size()) ==
      static_cast<int>(all_cards.size())) {
    if (smalls.empty()) {
      return deadcards.front();
    }
    int min_heads = std::numeric_limits<int>::max();
    for (const auto& st : selfstacks) {
      if (st.heads < min_heads) min_heads = st.heads;
    }
    if (min_heads <= 3) {
      return all_cards.front();
    }
    if (smalls.size() >= 2 &&
        smalls[smalls.size() - 2] >= smalls.back() - 3) {
      return smalls[smalls.size() - 2];
    }
    return smalls.back();
  }

  for (int i = 0; i < 4; ++i) {
    const auto& iv = IM(i);
    if (iv.stack_idx < 0) continue;
    const auto& st = selfstacks[static_cast<size_t>(iv.stack_idx)];
    if (st.length == 4 && !iv.cards.empty() &&
        iv.cards.front() <= st.top + 3) {
      return iv.cards.front();
    }
  }

  for (int i = 0; i < 4; ++i) {
    const auto& iv = IM(i);
    if (iv.stack_idx < 0) continue;
    const auto& st = selfstacks[static_cast<size_t>(iv.stack_idx)];
    if (st.length == 3 && !iv.cards.empty()) {
      return iv.cards.back();
    }
  }

  if (!intervals[1].cards.empty()) {
    const auto& st = selfstacks[static_cast<size_t>(intervals[1].stack_idx)];
    if (st.length == 1) {
      return intervals[1].cards.front();
    }
  }

  int skip = -1;
  bool found = false;
  for (int i = 0; i < 4; ++i) {
    const auto& iv = IM(i);
    if (!iv.bad) continue;
    if (iv.stack_idx < 0) continue;
    const auto& st = selfstacks[static_cast<size_t>(iv.stack_idx)];
    int ilen = static_cast<int>(iv.cards.size());

    if (!found && ilen == 2 && st.length == 1) {
      if (i != 3 && static_cast<int>(IM(i + 1).cards.size()) > 0) {
        return iv.cards.front();
      } else {
        skip = i;
        continue;
      }
    } else if (i == 2 && st.length == 1 && IM(i + 1).stack_idx >= 0 &&
               selfstacks[static_cast<size_t>(IM(i + 1).stack_idx)].length == 1) {
      return iv.cards.front();
    } else if (i == 3 && st.length == 1) {
      return iv.cards.front();
    } else if (st.length == 1) {
      return iv.cards.front();
    } else if (i == 0 || i == 1) {
      return iv.cards.back();
    } else if (i == 2) {
      if (ilen >= 2 && iv.cards.back() <= iv.cards[ilen - 2] + 4 &&
          selfstacks[3].length == 1) {
        return iv.cards[ilen - 2];
      }
      return iv.cards.back();
    } else {
      if (ilen >= 2 && iv.cards.back() <= iv.cards[ilen - 2] + 4) {
        return iv.cards[ilen - 2];
      }
    }
    found = true;
  }
  if (skip != -1) {
    return IM(skip).cards.front();
  }
  if (IM(3).bad && !IM(3).cards.empty()) {
    return IM(3).cards.back();
  }

  static constexpr int T[4] = {3, 2, 1, 0};
  for (int idx : T) {
    const auto& iv = IM(idx);
    if (iv.stack_idx < 0) continue;
    const auto& st = selfstacks[static_cast<size_t>(iv.stack_idx)];
    if (static_cast<int>(iv.cards.size()) == 1 && st.length == 2) {
      return iv.cards.front();
    }
  }
  for (int idx : T) {
    const auto& iv = IM(idx);
    if (iv.stack_idx < 0) continue;
    const auto& st = selfstacks[static_cast<size_t>(iv.stack_idx)];
    if (static_cast<int>(iv.cards.size()) == 1 && st.length == 1) {
      return iv.cards.front();
    }
  }
  if (!intervals[0].cards.empty()) {
    int min_heads = std::numeric_limits<int>::max();
    for (const auto& st : selfstacks) {
      if (st.heads < min_heads) min_heads = st.heads;
    }
    if (min_heads <= 3) {
      return all_cards.front();
    }
  }
  for (int i = 0; i < 4; ++i) {
    const auto& iv = IM(i);
    if (iv.stack_idx < 0) continue;
    const auto& st = selfstacks[static_cast<size_t>(iv.stack_idx)];
    if (st.length == 4 && !iv.cards.empty()) {
      return iv.cards.front();
    }
  }
  for (int i = 0; i < 4; ++i) {
    const auto& iv = IM(i);
    if (iv.danger && !iv.cards.empty()) {
      return iv.cards.back();
    }
  }
  return all_cards.front();
}

template <int NPlayers>
HeuristicResult pick_impl(IGameState& state, const IGameRules& rules) {
  const auto& s = checked_cast<SixNimmtState<NPlayers>>(state);
  const auto& d = s.data;
  auto legal = rules.legal_actions(state);

  HeuristicResult result;
  result.actions = legal;
  result.scores.assign(legal.size(), 0.0);
  if (legal.empty()) return result;

  std::array<StackInfo, kNumRows> stacks{};
  collect_stacks<NPlayers>(s, &stacks);
  auto selfstacks = sorted_stacks(stacks);

  if (d.small_phase_active) {
    int my_card = (d.small_player >= 0 && d.small_player < NPlayers)
                      ? d.pending_commits[static_cast<size_t>(d.small_player)]
                      : 0;
    int opp_estimate = 0;
    {
      std::array<bool, kNumCards + 1> seen{};
      for (int r = 0; r < kNumRows; ++r) {
        for (int sidx = 0; sidx < kRowCapacity; ++sidx) {
          int c = d.table[static_cast<size_t>(r)][static_cast<size_t>(sidx)];
          if (c >= 1 && c <= kNumCards) seen[static_cast<size_t>(c)] = true;
        }
      }
      for (int p = 0; p < NPlayers; ++p) {
        for (int c = 1; c <= kNumCards; ++c) {
          if (d.bullhead_pile[static_cast<size_t>(p)]
                              [static_cast<size_t>(c - 1)] == 1) {
            seen[static_cast<size_t>(c)] = true;
          }
        }
      }
      for (int c = 1; c <= kNumCards; ++c) {
        if (d.hand[static_cast<size_t>(d.current_player)]
                  [static_cast<size_t>(c - 1)] == 1) {
          seen[static_cast<size_t>(c)] = true;
        }
      }
      for (int c = kNumCards; c >= 1; --c) {
        if (!seen[static_cast<size_t>(c)]) { opp_estimate = c; break; }
      }
      if (opp_estimate == 0) opp_estimate = my_card;
    }

    int chosen_row = findsmall_index<NPlayers>(stacks, my_card, opp_estimate);
    for (size_t i = 0; i < legal.size(); ++i) {
      ActionId a = legal[i];
      if (a == kSmallChooseRowOffset + chosen_row) {
        result.scores[i] = 1.0;
      } else {
        result.scores[i] = 0.0;
      }
    }
    return result;
  }

  int perspective = d.current_player;
  auto intervals = build_intervals<NPlayers>(s, perspective, selfstacks);
  int chosen_card = commit_choice<NPlayers>(selfstacks, intervals);
  ActionId target = kCommitOffset + (chosen_card - 1);
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
HeuristicResult SixNimmtRuleHeuristic<NPlayers>::pick(
    IGameState& masked_state,
    const IBeliefTracker* /*tracker*/,
    const IGameRules& rules,
    std::uint64_t /*rng_seed*/) const {
  return pick_impl<NPlayers>(masked_state, rules);
}

template class SixNimmtRuleHeuristic<2>;
template class SixNimmtRuleHeuristic<3>;
template class SixNimmtRuleHeuristic<4>;

}  // namespace board_ai::sixnimmt
