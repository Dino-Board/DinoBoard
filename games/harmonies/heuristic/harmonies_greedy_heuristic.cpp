#include "harmonies_greedy_heuristic.h"

#include <algorithm>
#include <array>
#include <queue>

#include "../../../engine/core/game_interfaces.h"
#include "../harmonies_card_data.h"
#include "../harmonies_rules.h"
#include "../harmonies_state.h"

namespace board_ai::harmonies {

namespace {

inline std::pair<int, int> apply_xform(int t, int dq, int dr) {
  const int a = kTransforms[t][0];
  const int b = kTransforms[t][1];
  const int c = kTransforms[t][2];
  const int d = kTransforms[t][3];
  return {a * dq + b * dr, c * dq + d * dr};
}

// Full color stack (bottom-up) for each tower_type, terminated by 0.
static constexpr int kTowerColors[][3] = {
  {0,0,0},  // 0 empty
  {1,0,0},  // 1 Blue
  {2,0,0},  // 2 Gray
  {3,0,0},  // 3 Brown
  {4,0,0},  // 4 Green
  {5,0,0},  // 5 Yellow
  {6,0,0},  // 6 Red
  {6,6,0},  // 7 house2 (Red+Red)
  {3,3,0},  // 8 wood2  (Brown+Brown)
  {3,4,0},  // 9 tree2  (Brown+Green)
  {2,2,0},  // 10 mountain2 (Gray+Gray)
  {3,3,4},  // 11 tree3  (Brown+Brown+Green)
  {2,2,2},  // 12 mountain3 (Gray+Gray+Gray)
};

// For a given card + player board, find the best (anchor, orientation) that
// maximizes the fraction of pattern pieces already satisfied.
// Returns: {best_fraction, best_anchor, best_orientation, unsatisfied_colors}
// unsatisfied_colors: for the best placement, which colors are still needed
// (as a count array indexed 0..5 for colors 1..6).
template <int NPlayers, int Side>
struct CardFitResult {
  double fraction;
  int anchor;
  int orient;
  int missing_pieces;
  std::array<int, kColors> needed_colors;
};

template <int NPlayers, int Side>
CardFitResult<NPlayers, Side> best_card_fit(
    const HarmoniesState<NPlayers, Side>& s, int player, int card_id) {
  const auto& card = kCards[card_id - 1];
  CardFitResult<NPlayers, Side> best{};
  best.fraction = -1.0;
  best.anchor = -1;
  best.orient = -1;
  best.missing_pieces = card.pattern_size;
  best.needed_colors.fill(0);

  using Geo = Board<Side>;
  for (int anchor = 0; anchor < Geo::kHexes; ++anchor) {
    if (s.has_animal[player][anchor]) continue;
    const int ax_q = Geo::kAxialQ[anchor];
    const int ax_r = Geo::kAxialR[anchor];
    for (int t = 0; t < 6; ++t) {
      int sat = 0;
      bool blocked = false;
      std::array<int, kColors> need{};
      for (int i = 0; i < card.pattern_size; ++i) {
        const auto& piece = card.pieces[i];
        const auto rel = apply_xform(t, piece.dq, piece.dr);
        const int tq = ax_q + rel.first;
        const int tr = ax_r + rel.second;
        const int th = Geo::axial_to_hex(tq, tr);
        if (th < 0) { blocked = true; break; }
        const int tt = static_cast<int>(s.tower_type[player][th]);
        const std::uint16_t bit = static_cast<std::uint16_t>(1u << tt);
        if (piece.tower_mask & bit) {
          ++sat;
        } else if (tt != 0) {
          blocked = true; break;
        } else {
          // Empty hex — figure out ALL colors needed to build the target tower.
          for (int try_tt = 1; try_tt < kTowerTypes; ++try_tt) {
            if (piece.tower_mask & (1u << try_tt)) {
              for (int lv = 0; lv < 3; ++lv) {
                int col = kTowerColors[try_tt][lv];
                if (col == 0) break;
                ++need[col - 1];
              }
              break;
            }
          }
        }
      }
      if (blocked) continue;
      const double frac = static_cast<double>(sat) / card.pattern_size;
      if (frac > best.fraction) {
        best.fraction = frac;
        best.anchor = anchor;
        best.orient = t;
        best.missing_pieces = card.pattern_size - sat;
        best.needed_colors = need;
      }
    }
  }
  if (best.fraction < 0.0) best.fraction = 0.0;
  return best;
}

// Evaluate how well a set of selected tokens matches the needs of held cards.
template <int NPlayers, int Side>
double slot_card_synergy(const HarmoniesState<NPlayers, Side>& s, int player,
                         const std::array<int, kColors>& slot_colors) {
  double synergy = 0.0;
  for (int i = 0; i < kHandSize; ++i) {
    const int cid = s.held_cards[player][i];
    if (cid <= 0) continue;
    const auto fit = best_card_fit(s, player, cid);
    if (fit.anchor < 0) continue;
    // Count how many tokens in this slot match what the card needs.
    int matched = 0;
    int total_needed = 0;
    for (int c = 0; c < kColors; ++c) {
      total_needed += fit.needed_colors[c];
      matched += std::min(slot_colors[c], fit.needed_colors[c]);
    }
    if (total_needed <= 0) continue;
    const auto& card = kCards[cid - 1];
    double card_value = static_cast<double>(card.scores[card.num_cubes - 1]);
    synergy += (static_cast<double>(matched) / total_needed) * card_value * 0.8;
  }
  return synergy;
}

template <int NPlayers, int Side>
double evaluate(const HarmoniesState<NPlayers, Side>& s, int player) {
  double v = static_cast<double>(s.animal_scores[player])
           + static_cast<double>(s.landscape_scores[player]);

  const auto& tt = s.tower_type[player];

  // ======== CARD FIT POTENTIAL (guide token placement toward completing cards) ========
  for (int i = 0; i < kHandSize; ++i) {
    const int cid = s.held_cards[player][i];
    if (cid <= 0) continue;
    const auto& card = kCards[cid - 1];
    const int cubes_init = std::max<int>(1, s.cube_initial[player][i]);
    const int cubes_left = s.cube_remaining[player][i];
    const int cubes_placed = cubes_init - cubes_left;

    // Next cube's incremental value.
    const int next_idx = cubes_placed;
    double next_score;
    if (next_idx < card.num_cubes) {
      next_score = static_cast<double>(card.scores[next_idx]);
      if (next_idx > 0) next_score -= card.scores[next_idx - 1];
    } else {
      next_score = 0.0;
    }

    const auto fit = best_card_fit(s, player, cid);

    // Reward based on how close the board is to enabling the next cube placement.
    // Scale by the value of the next cube to prioritize high-value cubes.
    if (fit.fraction >= 1.0) {
      // Pattern fully satisfied — next cube can be placed right now.
      v += next_score * 3.0;
    } else if (fit.missing_pieces == 1) {
      v += next_score * 1.5;
    } else if (fit.missing_pieces == 2) {
      v += next_score * 0.6;
    } else {
      v += fit.fraction * next_score * 0.3;
    }
  }

  // ======== HAND MANAGEMENT ========
  int hand_count = 0;
  for (int i = 0; i < kHandSize; ++i) {
    if (s.held_cards[player][i] > 0) ++hand_count;
  }
  v += hand_count * 2.0;

  // ======== LANDSCAPE SHAPING (secondary signal) ========
  int occupied = 0;
  for (int h = 0; h < Board<Side>::kHexes; ++h) {
    if (tt[h] != 0) ++occupied;
  }
  v += occupied * 0.3;

  return v;
}

template <int NPlayers, int Side>
double score_action(HarmoniesState<NPlayers, Side>& s, const IGameRules& rules,
                    ActionId a, std::mt19937_64& rng) {
  using Geo = Board<Side>;
  const int player = s.current_player_;

  // ---- SELECT_SLOT: evaluate slot-card synergy before cloning ----
  if (a >= kActSelectSlotBase && a < kActSelectSlotBase + kSlots) {
    const int slot = static_cast<int>(a - kActSelectSlotBase);
    std::array<int, kColors> slot_colors{};
    for (int c = 0; c < kColors; ++c) {
      slot_colors[c] = s.central_board_counts[slot][c];
    }
    double synergy = slot_card_synergy(s, player, slot_colors);

    const IGameState& base = s;
    auto clone = base.clone_state();
    rules.do_action_fast(*clone, a, rng);
    auto& cs = checked_cast<HarmoniesState<NPlayers, Side>>(*clone);
    HarmoniesRules<NPlayers, Side>::recompute_landscape_scores(cs);
    return evaluate(cs, player) + synergy;
  }

  // ---- TAKE_CARD: bonus for card fit with current board ----
  if (a >= kActTakeCardBase && a < kActPlaceTokenBase) {
    const IGameState& base = s;
    auto clone = base.clone_state();
    rules.do_action_fast(*clone, a, rng);
    auto& cs = checked_cast<HarmoniesState<NPlayers, Side>>(*clone);
    HarmoniesRules<NPlayers, Side>::recompute_landscape_scores(cs);
    double v = evaluate(cs, player);
    // Extra bonus: how well does the taken card synergize with existing cards?
    const int cid = static_cast<int>(a - kActTakeCardBase) + 1;
    const auto fit = best_card_fit(cs, player, cid);
    const auto& card = kCards[cid - 1];
    double card_score = static_cast<double>(card.scores[card.num_cubes - 1]);
    // High fit fraction = card pattern already partially on board = great pick.
    v += fit.fraction * card_score * 1.0;
    // Prefer cards with fewer missing pieces (easier to complete).
    v -= fit.missing_pieces * 1.0;
    return v;
  }

  // ---- PLACE_CUBE: always valuable — actual score from partial scoring ----
  if (a >= Geo::kActPlaceCubeBase && a < Geo::kActEndTurn) {
    const IGameState& base = s;
    auto clone = base.clone_state();
    rules.do_action_fast(*clone, a, rng);
    auto& cs = checked_cast<HarmoniesState<NPlayers, Side>>(*clone);
    HarmoniesRules<NPlayers, Side>::recompute_landscape_scores(cs);
    return evaluate(cs, player) + 8.0;
  }

  // ---- PLACE_TOKEN: bonus if it advances a card pattern ----
  if (a >= kActPlaceTokenBase && a < Geo::kActPlaceCubeBase) {
    const int hex = static_cast<int>(a - kActPlaceTokenBase) / kColors;
    const IGameState& base = s;
    auto clone = base.clone_state();
    rules.do_action_fast(*clone, a, rng);
    auto& cs = checked_cast<HarmoniesState<NPlayers, Side>>(*clone);
    HarmoniesRules<NPlayers, Side>::recompute_landscape_scores(cs);
    double v = evaluate(cs, player);

    // Check if this placement improved any card's best fit.
    for (int i = 0; i < kHandSize; ++i) {
      const int cid = cs.held_cards[player][i];
      if (cid <= 0) continue;
      const auto fit_after = best_card_fit(cs, player, cid);
      const auto fit_before = best_card_fit(s, player, cid);
      double delta = fit_after.fraction - fit_before.fraction;
      if (delta > 0.0) {
        const auto& card = kCards[cid - 1];
        double card_score = static_cast<double>(card.scores[card.num_cubes - 1]);
        v += delta * card_score * 1.5;
      }
    }

    // Penalty for placing on a hex far from existing tiles (discourage scatter).
    bool has_neighbor = false;
    for (int d = 0; d < 6; ++d) {
      const int nb = Geo::neighbor(hex, d);
      if (nb >= 0 && s.tower_type[player][nb] != 0) {
        has_neighbor = true; break;
      }
    }
    if (!has_neighbor && s.space_remaining[player] < Geo::kHexes - 2) {
      v -= 1.0;
    }

    return v;
  }

  // ---- END_TURN / other ----
  {
    const IGameState& base = s;
    auto clone = base.clone_state();
    rules.do_action_fast(*clone, a, rng);
    auto& cs = checked_cast<HarmoniesState<NPlayers, Side>>(*clone);
    HarmoniesRules<NPlayers, Side>::recompute_landscape_scores(cs);
    return evaluate(cs, player);
  }
}

}  // namespace

template <int NPlayers, int Side>
HeuristicResult HarmoniesGreedyHeuristic<NPlayers, Side>::pick(
    IGameState& masked_state,
    const IBeliefTracker* /*tracker*/,
    const IGameRules& rules,
    std::uint64_t rng_seed) const {
  auto& s = checked_cast<HarmoniesState<NPlayers, Side>>(masked_state);
  auto legal = rules.legal_actions(masked_state);
  HeuristicResult result;
  result.actions = legal;
  result.scores.reserve(legal.size());
  std::mt19937_64 rng(rng_seed ? rng_seed : 1ULL);
  for (ActionId a : legal) {
    result.scores.push_back(score_action<NPlayers, Side>(s, rules, a, rng));
  }
  return result;
}

#define HARMONIES_INST_HarmoniesGreedyHeuristic(N, S) template class HarmoniesGreedyHeuristic<N, S>;
HARMONIES_FOR_EACH_VARIANT(HARMONIES_INST_HarmoniesGreedyHeuristic)
#undef HARMONIES_INST_HarmoniesGreedyHeuristic

}  // namespace board_ai::harmonies
