#include "loveletter_rule_heuristic.h"

#include <algorithm>
#include <array>
#include <random>

#include "../loveletter_state.h"

namespace board_ai::loveletter {

namespace {

constexpr int kGuard = 1, kPriest = 2, kBaron = 3, kHandmaid = 4;
constexpr int kPrince = 5, kKing = 6, kCountess = 7, kPrincess = 8;

template <int NPlayers>
double score_action(const LoveLetterState<NPlayers>& s, ActionId a,
                    std::mt19937_64& /*rng*/) {
  const auto& d = s.data;
  const int actor = d.current_player;
  const int my_hand = d.hand[actor];
  const int my_drawn = d.drawn_card;

  // Action ids carry the target's absolute seat (decode_action in
  // loveletter_rules.cpp), not an offset from the actor. Reading them as
  // an offset made every seat but the last score "Prince on myself" as an
  // attack on an opponent.
  auto target_from_offset = [&](int target_seat) { return target_seat; };

  if (a >= kGuardOffset && a < kGuardOffset + kGuardCount) {
    const int rel = a - kGuardOffset;
    const int target_off = rel / 7;
    const int guess = 2 + (rel % 7);
    const int target = target_from_offset(target_off);
    if (target == actor || !d.alive[target]) return -100.0;
    if (d.protected_flags[target]) return -5.0;
    int seen = 0;
    std::array<int, 9> played{};
    for (int p = 0; p < NPlayers; ++p) {
      for (int c = 1; c <= kCardTypes; ++c) {
        int n = d.discard_count[p][c];
        played[c] += n;
        seen += n;
      }
    }
    played[my_hand]++;
    if (my_drawn > 0) played[my_drawn]++;
    for (int c = 1; c <= kCardTypes; ++c) {
      played[c] += d.face_up_count[c];
    }
    const std::array<int, 9> total = {0, 5, 2, 2, 2, 2, 1, 1, 1};
    double prob_guess = std::max(0.0, static_cast<double>(total[guess] - played[guess]));
    return 0.6 + prob_guess * 0.25;
  }

  if (a >= kPriestOffset && a < kPriestOffset + kPriestCount) {
    const int target = target_from_offset(a - kPriestOffset);
    if (target == actor || !d.alive[target]) return -100.0;
    if (d.protected_flags[target]) return -5.0;
    return 2.5;
  }

  if (a >= kBaronOffset && a < kBaronOffset + kBaronCount) {
    const int target = target_from_offset(a - kBaronOffset);
    if (target == actor || !d.alive[target]) return -100.0;
    if (d.protected_flags[target]) return -5.0;
    int kept = (my_hand == kBaron) ? my_drawn : my_hand;
    return static_cast<double>(kept) * 0.5 - 1.5;
  }

  if (a == kHandmaidAction) {
    return 2.2;
  }

  if (a >= kPrinceOffset && a < kPrinceOffset + kPrinceCount) {
    const int target = target_from_offset(a - kPrinceOffset);
    if (!d.alive[target]) return -100.0;
    if (target != actor && d.protected_flags[target]) return -5.0;
    if (target == actor) {
      int kept = (my_hand == kPrince) ? my_drawn : my_hand;
      if (kept == kCountess) return 0.3;
      return -1.0;
    }
    return 1.8;
  }

  if (a >= kKingOffset && a < kKingOffset + kKingCount) {
    const int target = target_from_offset(a - kKingOffset);
    if (target == actor || !d.alive[target]) return -100.0;
    if (d.protected_flags[target]) return -5.0;
    int my_other = (my_hand == kKing) ? my_drawn : my_hand;
    if (my_other <= 3) return 2.0;
    return -0.5;
  }

  if (a == kCountessAction) {
    if (my_hand == kKing || my_drawn == kKing ||
        my_hand == kPrince || my_drawn == kPrince) {
      return 0.5;
    }
    return -0.8;
  }

  if (a == kPrincessAction) {
    return -50.0;
  }

  return 0.0;
}

}  // namespace

template <int NPlayers>
HeuristicResult LoveLetterRuleHeuristic<NPlayers>::pick(
    IGameState& masked_state,
    const IBeliefTracker* /*tracker*/,
    const IGameRules& rules,
    std::uint64_t rng_seed) const {
  auto& s = checked_cast<LoveLetterState<NPlayers>>(masked_state);
  auto legal = rules.legal_actions(masked_state);
  std::mt19937_64 rng(rng_seed);

  HeuristicResult result;
  result.actions = legal;
  result.scores.reserve(legal.size());
  for (ActionId a : legal) {
    result.scores.push_back(score_action<NPlayers>(s, a, rng));
  }
  return result;
}

template class LoveLetterRuleHeuristic<2>;
template class LoveLetterRuleHeuristic<3>;
template class LoveLetterRuleHeuristic<4>;

}  // namespace board_ai::loveletter
