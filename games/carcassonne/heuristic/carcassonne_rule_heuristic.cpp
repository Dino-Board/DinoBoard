#include "carcassonne_rule_heuristic.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include "../carcassonne_rules.h"
#include "../carcassonne_state.h"

namespace board_ai::carcassonne {

namespace {

// ============================================================
// Phased weights
//
// Every weight in RuleWeights is an `std::array<T, 3>` indexed by game
// phase: [0] early, [1] mid, [2] late. The phase boundaries
// `t_early` / `t_mid` are also profile fields, so a profile can choose
// e.g. {0.30, 0.65} or {0.40, 0.75} to shift when "midgame" starts.
//
// Phase selection is by `progress = (kDeckSlots - deck_count) / kDeckSlots`:
//   progress < t_early  → use [0] early
//   progress < t_mid    → use [1] mid
//   else                → use [2] late
//
// A weight that should NOT vary with phase fills all three slots with
// the same value (helper `K(x)` below). A weight that linearly ramps
// uses three distinct values; the score functions step at the phase
// boundaries (no smooth interpolation — phases are discrete).
// ============================================================

template <typename T>
using Phased = std::array<T, 3>;

template <typename T>
constexpr Phased<T> K(T v) {
  return {v, v, v};
}

template <typename T>
constexpr Phased<T> P(T e, T m, T l) {
  return {e, m, l};
}

// Pick the phase slot for a given progress, using a profile's
// (t_early, t_mid) thresholds.
template <typename T>
T pick_phased(const Phased<T>& w, double progress,
              double t_early, double t_mid) {
  if (progress < t_early) return w[0];
  if (progress < t_mid) return w[1];
  return w[2];
}

// ============================================================
// RuleWeights
//
// Three coexisting evaluation paths for cities/roads (controlled by
// profile-set weights):
//   (1) Legacy ownership: empty/owned/my/opp_*. Old asymmetric branch
//       family. A 1-1 city evaluates to 0 even though rules pay both
//       owners full points; the marginal path corrects this.
//   (2) Open-edge structural: city/road completion_bonus/open_penalty
//       and the *_meeple_open_penalty pair. Captures "completability"
//       (open_edges = remaining edges to close).
//   (3) Marginal: marginal_*_complete/_open. When > 0, score_place_action
//       uses marginal_feature_delta (before/after award diff respecting
//       award_to_majority). The legacy ownership weights for that path
//       must be 0 to avoid double-counting.
// ============================================================
struct RuleWeights {
  // Phase boundaries
  double t_early = 0.33;
  double t_mid = 0.67;

  // ---- PV evaluator settings ----
  // The rule heuristic returns raw scores; the PV evaluator wraps them
  // in softmax(score * prior_temperature) to produce MCTS priors. Larger
  // raw scores (sharper profiles) want lower temperatures so the softmax
  // distinguishes them less aggressively. value_margin_scale is the tanh
  // divisor for the V head: V = tanh((final_me - final_opp) / scale).
  double prior_temperature = 0.55;
  double value_margin_scale = 28.0;

  // ---- Meeple supply pressure (multiplier applied at every meeple-
  //      placement target). Bucketed by remaining: 0-1, 2, 3, many. ----
  Phased<double> supply_low = K(2.3);    // 0-1 remaining
  Phased<double> supply_two = K(1.25);   // 2 remaining
  Phased<double> supply_three = K(0.55); // 3 remaining
  Phased<double> supply_many = K(0.18);  // 4+ remaining

  // ---- Tile placement: structural board features ----
  Phased<double> adjacent = K(0.95);      // each occupied neighbor
  Phased<double> center_dist = K(0.025);  // Manhattan distance penalty

  // ---- Tile placement: cities, legacy ownership path (1) ----
  Phased<double> empty_city_complete = K(0.45);
  Phased<double> owned_city_complete = K(1.35);
  Phased<double> open_city_base = K(0.18);
  Phased<double> my_city_base = K(0.45);
  Phased<double> my_city_points = K(0.12);
  Phased<double> opp_city_base = K(0.35);
  Phased<double> opp_city_points = K(0.10);

  // ---- Tile placement: roads, legacy ownership path (1) ----
  Phased<double> empty_road_complete = K(0.25);
  Phased<double> owned_road_complete = K(1.05);
  Phased<double> open_road_base = K(0.07);
  Phased<double> my_road_base = K(0.20);
  Phased<double> my_road_points = K(0.06);
  Phased<double> opp_road_base = K(0.25);
  Phased<double> opp_road_points = K(0.07);

  // ---- Tile placement: cloisters / chapel ----
  Phased<double> chapel_place_base = K(0.8);
  Phased<double> chapel_place_occupancy = K(0.35);
  // Adjacent-cloister tile bonus (this placed tile raises an existing
  // cloister's occupancy). Off in production sharp.
  Phased<double> tile_cloister_complete = K(0.0);
  Phased<double> tile_cloister_near = K(0.0);

  // ---- Tile placement: farms (place-time delta) ----
  // farm_tile_X[phase] = the per-completed-city contribution multiplier
  // when the farm has X (empty/my-leading/opp-leading/tied) ownership.
  Phased<double> farm_tile_empty = K(0.05);
  Phased<double> farm_tile_my = K(0.18);
  Phased<double> farm_tile_opp = K(0.20);
  Phased<double> farm_tile_tied = K(0.04);

  // ---- Open-edge structural path (2) ----
  Phased<double> city_completion_bonus = K(0.0);
  Phased<double> city_open_penalty = K(0.0);
  Phased<double> opp_city_open_bonus = K(0.0);
  Phased<double> road_completion_bonus = K(0.0);
  Phased<double> road_open_penalty = K(0.0);
  Phased<double> opp_road_open_bonus = K(0.0);
  Phased<double> city_meeple_open_penalty = K(0.0);
  Phased<double> road_meeple_open_penalty = K(0.0);

  // ---- Marginal path (3) ----
  Phased<double> marginal_city_complete = K(0.0);
  Phased<double> marginal_city_open = K(0.0);
  Phased<double> marginal_road_complete = K(0.0);
  Phased<double> marginal_road_open = K(0.0);

  // ---- Pass meeple ----
  Phased<double> pass_base = K(0.18);
  Phased<double> pass_supply = K(0.35);

  // ---- Meeple placement: cloister center ----
  Phased<double> cloister_base = K(0.85);
  Phased<double> cloister_occupancy = K(0.45);
  Phased<double> cloister_complete_bonus = K(6.0);
  Phased<double> cloister_supply = K(1.0);

  // ---- Meeple placement: cities ----
  Phased<double> city_meeple_complete_base = K(1.8);
  Phased<double> city_meeple_complete_points = K(1.35);
  Phased<double> city_meeple_base = K(0.75);
  Phased<double> city_meeple_points = K(0.30);
  Phased<double> city_meeple_supply = K(1.0);
  Phased<double> city_small_bonus = K(0.45);
  Phased<double> city_large_penalty = K(0.35);

  // ---- Meeple placement: roads ----
  Phased<double> road_meeple_complete_base = K(0.9);
  Phased<double> road_meeple_complete_points = K(1.05);
  Phased<double> road_meeple_base = K(0.35);
  Phased<double> road_meeple_points = K(0.13);
  Phased<double> road_meeple_supply = K(1.15);

  // ---- Meeple placement: farms ----
  Phased<double> farm_base = K(0.20);
  Phased<double> farm_points = K(0.45);   // multiplied by completed_cities*3
  Phased<double> farm_flat = K(0.75);     // flat per-phase bonus
  Phased<double> farm_supply = K(1.85);
  // farm_empty_early_penalty fires only in EARLY phase (progress < t_early)
  // when 0 completed cities. The old `farm_empty_until` threshold has been
  // collapsed into the phase boundary t_early.
  Phased<double> farm_empty_early_penalty = K(1.25);
  // farm_crowded_penalty fires in EARLY+MID phases when too many of our
  // meeples are already on the board. Old `farm_crowded_until` collapsed
  // into the phase boundary t_mid (only fires when progress < t_mid).
  Phased<double> farm_crowded_penalty = K(0.70);
  Phased<int> farm_crowded_meeples = K(5);
  // Farm potential: incomplete-city contribution. Off in production sharp.
  Phased<double> farm_potential = K(0.0);
};

// ============================================================
// Profile definitions
//
// Each profile_*_weights() returns a complete RuleWeights. Profiles
// can override t_early / t_mid to shift phase boundaries.
// ============================================================

RuleWeights baseline_weights() {
  // Baseline: phase-flat weights. Used as the regression-control profile
  // and as the starting template for sharper profiles.
  return RuleWeights{};
}

RuleWeights sharp_weights() {
  // 2026-06-22 reshape: phased weights re-designed around the strategy
  // guides ("opening = city/cloister, endgame = farm warfare") rather
  // than mechanically sampling the old continuous curves.
  //
  //   EARLY (progress < 0.33):
  //     - city / cloister scoring stays strong (fast points + meeple cycling)
  //     - farms heavily discouraged (penalty + low value, opp leads we
  //       walk away rather than contesting)
  //     - meeple supply pressure tight (we want reserves for surprise small
  //       cities + chapels that show up later)
  //   MID (0.33 - 0.67):
  //     - secure ONE farm if it touches a real completed city
  //     - city/cloister still primary
  //     - supply pressure relaxes a bit
  //   LATE (progress > 0.67):
  //     - farm ranks rise sharply (final 3 pts/city is what wins or loses)
  //     - completing in-flight city/road > new meeple (V head already
  //       projects them, so heuristic rewards finishing not starting)
  //     - supply low_*: relax — late-game meeple has fewer turns to lock
  //
  // PV: sharper rule scores want a lower softmax temperature; tanh scale
  // tightens to 24 because the score range shrinks once final-scoring
  // projection is the only V signal (no projection_weight to mix).
  RuleWeights w = baseline_weights();
  w.prior_temperature = 0.78;
  w.value_margin_scale = 24.0;

  // ---- Supply pressure: highest in early, relaxed by late ----
  // Old code: subtraction-style supply_low_base - supply_low_progress *
  // progress. Phase 0 should be largest. Reduced compared to the old
  // 2.95-1.91 lerp — V head now does its own work on meeple value, so
  // we don't need to scream as loudly here.
  w.supply_low = P(2.60, 2.10, 1.65);
  w.supply_two = P(1.55, 1.30, 1.00);
  w.supply_three = P(0.70, 0.55, 0.40);
  w.supply_many = K(0.18);

  // ---- Structural placement ----
  w.adjacent = K(1.08);
  w.center_dist = K(0.035);

  // ---- Cities (legacy ownership path) ----
  // Slightly stronger early, slightly relaxed late (late: V head already
  // captures completion margin, so the hand-coded city term over-counts).
  w.empty_city_complete = P(0.45, 0.40, 0.32);
  w.owned_city_complete = P(1.80, 1.55, 1.30);
  w.open_city_base = P(0.18, 0.14, 0.10);
  w.my_city_base = P(0.65, 0.55, 0.45);
  w.my_city_points = P(0.18, 0.15, 0.12);
  w.opp_city_base = P(0.60, 0.52, 0.42);
  w.opp_city_points = P(0.16, 0.13, 0.10);

  // ---- Roads (legacy) ----
  // Roads are quick points + meeple cycling — keep stable across phases.
  w.empty_road_complete = K(0.18);
  w.owned_road_complete = K(1.18);
  w.open_road_base = K(0.045);
  w.my_road_base = K(0.25);
  w.my_road_points = K(0.07);
  w.opp_road_base = K(0.34);
  w.opp_road_points = K(0.09);

  // ---- Chapel placement ----
  w.chapel_place_base = K(0.65);
  w.chapel_place_occupancy = K(0.42);
  w.pass_base = K(0.24);
  w.pass_supply = K(0.42);

  // ---- Cloister meeple ----
  // Stronger early — small cloisters are a key way to cycle meeples.
  w.cloister_base = P(0.85, 0.75, 0.65);
  w.cloister_occupancy = K(0.50);
  w.cloister_complete_bonus = K(6.4);
  w.cloister_supply = K(1.15);

  // ---- City meeple ----
  // Push small-city / completing-city placements harder in early/mid,
  // ease late (V head already counts in-flight value).
  w.city_meeple_complete_base = P(2.20, 2.00, 1.70);
  w.city_meeple_complete_points = K(1.55);
  w.city_meeple_base = P(0.85, 0.75, 0.65);
  w.city_meeple_points = K(0.36);
  w.city_meeple_supply = K(1.08);
  w.city_small_bonus = P(0.80, 0.65, 0.50);
  w.city_large_penalty = P(0.70, 0.55, 0.40);

  // ---- Road meeple ----
  w.road_meeple_complete_base = K(0.95);
  w.road_meeple_complete_points = K(1.12);
  w.road_meeple_base = K(0.28);
  w.road_meeple_points = K(0.10);
  w.road_meeple_supply = K(1.42);

  // ---- Farm meeple — strategy-driven phase shape ----
  // STRATEGY_NOTES says: avoid farms early; secure ONE in mid; embrace
  // late. Translated: farm_points jumps from low (early) to high (late).
  w.farm_base = P(-0.20, -0.05, 0.10);    // negative early, positive late
  w.farm_points = P(0.20, 0.65, 1.30);    // 6.5x late vs early
  w.farm_flat = P(-0.10, 0.10, 0.50);     // discourage early commit
  // Supply pressure on farms eases late (we want to commit them then).
  w.farm_supply = P(2.80, 2.00, 1.20);
  // Empty-farm penalty: still strong in early; gone in mid+late.
  // (in_early_phase() already gates this to phase[0] only at runtime —
  // setting all three the same value is fine; the runtime check decides.)
  w.farm_empty_early_penalty = K(2.00);
  // Crowded-farm penalty: in_mid_or_earlier_phase() gates to phase 0/1.
  w.farm_crowded_penalty = K(1.05);
  w.farm_crowded_meeples = K(4);

  // ---- Farm tile-placement (per-completed-city contribution) ----
  // Late phase emphasizes the differential most — endgame farm fights.
  w.farm_tile_empty = P(0.05, 0.15, 0.30);
  w.farm_tile_my = P(0.20, 0.50, 0.95);
  w.farm_tile_opp = P(0.25, 0.65, 1.10);
  w.farm_tile_tied = P(0.05, 0.18, 0.40);
  return w;
}

RuleWeights sharp_cloister_weights() {
  RuleWeights w = sharp_weights();
  w.tile_cloister_complete = K(1.20);
  // Old: progress < 0.55 ? 0.08 : 0.16. Phases t_early=0.33, t_mid=0.67
  // both fall above 0.55 only at late, so early/mid get the lower value.
  w.tile_cloister_near = P(0.08, 0.08, 0.16);
  return w;
}

RuleWeights sharp_reserve_weights() {
  // "Reserve" = stronger meeple-supply pressure, especially early.
  // Bumps on top of sharp_cloister_weights, scaled to new sharp values.
  RuleWeights w = sharp_cloister_weights();
  // Stronger supply pressure (+0.25 early, +0.15 mid, +0.05 late).
  w.supply_low = P(2.85, 2.25, 1.70);
  w.supply_two = P(1.75, 1.40, 1.05);
  w.supply_three = P(0.85, 0.65, 0.45);
  w.pass_supply = K(0.48);
  w.city_meeple_supply = P(1.18, 1.08, 1.08);
  w.road_meeple_supply = P(1.55, 1.50, 1.42);
  // Even less farm in early/mid (reserve = save meeples for hard tactics).
  w.farm_supply = P(3.00, 2.20, 1.30);
  w.farm_base = P(-0.30, -0.10, 0.05);
  w.farm_empty_early_penalty = K(2.20);
  return w;
}

RuleWeights sharp_cloister_reserve_soft_weights() {
  // Lighter reserve pressure than sharp_reserve, with cloister bonus.
  RuleWeights w = sharp_cloister_weights();
  w.tile_cloister_complete = K(1.30);
  // Old: progress < 0.55 ? 0.10 : 0.18 — preserved.
  w.tile_cloister_near = P(0.10, 0.10, 0.18);
  // Mild supply bumps (+0.15 / +0.10 / +0.02).
  w.supply_low = P(2.75, 2.20, 1.67);
  w.supply_two = P(1.65, 1.35, 1.02);
  w.pass_supply = K(0.45);
  w.road_meeple_supply = P(1.48, 1.46, 1.42);
  // Slightly lighter farm push than sharp_reserve.
  w.farm_supply = P(2.95, 2.10, 1.25);
  w.farm_base = P(-0.25, -0.08, 0.07);
  w.farm_empty_early_penalty = K(2.10);
  return w;
}

RuleWeights sharp_sharefix_weights() {
  // Switch city/road ownership to the marginal-delta path that respects
  // award_to_majority semantics. See STRATEGY_NOTES.md "Finding 1".
  RuleWeights w = sharp_weights();
  // Disable legacy ownership branches so we don't double-count.
  w.empty_city_complete = K(0.0);
  w.owned_city_complete = K(0.0);
  w.open_city_base = K(0.0);
  w.my_city_base = K(0.0);
  w.my_city_points = K(0.0);
  w.opp_city_base = K(0.0);
  w.opp_city_points = K(0.0);
  w.empty_road_complete = K(0.0);
  w.owned_road_complete = K(0.0);
  w.open_road_base = K(0.0);
  w.my_road_base = K(0.0);
  w.my_road_points = K(0.0);
  w.opp_road_base = K(0.0);
  w.opp_road_points = K(0.0);
  // Enable marginal scales.
  w.marginal_city_complete = K(1.65);
  w.marginal_city_open = K(0.55);
  w.marginal_road_complete = K(1.18);
  w.marginal_road_open = K(0.30);
  return w;
}

RuleWeights sharp_open_edge_weights() {
  RuleWeights w = sharp_weights();
  w.city_completion_bonus = K(0.6);
  w.city_open_penalty = K(0.10);
  w.opp_city_open_bonus = K(0.20);
  w.road_completion_bonus = K(0.30);
  w.road_open_penalty = K(0.25);
  w.opp_road_open_bonus = K(0.15);
  w.city_meeple_open_penalty = K(0.40);
  w.road_meeple_open_penalty = K(0.35);
  return w;
}

RuleWeights sharp_cloister_complete_weights() {
  RuleWeights w = sharp_weights();
  w.tile_cloister_complete = K(0.9);
  w.tile_cloister_near = K(0.12);
  return w;
}

RuleWeights sharp_farm_potential_weights() {
  RuleWeights w = sharp_weights();
  // Old: farm_potential_base 0.10 + farm_potential_progress 0.30 *
  // progress. Sampled at 0.165/0.5/0.835.
  w.farm_potential = P(0.15, 0.25, 0.35);
  return w;
}

RuleWeights weights_for_profile(CarcassonneHeuristicProfile profile) {
  switch (profile) {
    case CarcassonneHeuristicProfile::kBaseline: return baseline_weights();
    case CarcassonneHeuristicProfile::kSharpPvSoft: return sharp_weights();
    case CarcassonneHeuristicProfile::kSharpCloisterSoftPv:
      return sharp_cloister_weights();
    case CarcassonneHeuristicProfile::kSharpReserveSoftPv:
      return sharp_reserve_weights();
    case CarcassonneHeuristicProfile::kSharpCloisterReserveSoftPv:
      return sharp_cloister_reserve_soft_weights();
    case CarcassonneHeuristicProfile::kSharpSharefix:
      return sharp_sharefix_weights();
    case CarcassonneHeuristicProfile::kSharpOpenEdge:
      return sharp_open_edge_weights();
    case CarcassonneHeuristicProfile::kSharpCloisterComplete:
      return sharp_cloister_complete_weights();
    case CarcassonneHeuristicProfile::kSharpFarmPotential:
      return sharp_farm_potential_weights();
  }
  throw std::runtime_error("carcassonne heuristic unknown profile");
}

// ============================================================
// Helpers
// ============================================================

int opponent_of(int player) { return 1 - player; }

int meeples_on_board(const CarcassonneState& s, int player) {
  int count = 0;
  for (int cell = 0; cell < kBoardCells; ++cell) {
    if (s.board_meeple_owner[static_cast<size_t>(cell)] == player) ++count;
  }
  return count;
}

double game_progress(const CarcassonneState& s) {
  const double drawn = static_cast<double>(kDeckSlots - s.deck_count);
  return std::max(0.0, std::min(1.0, drawn / static_cast<double>(kDeckSlots)));
}

// Pick a phased weight at this state's progress, using the profile's
// own t_early/t_mid thresholds.
template <typename T>
T at_phase(const Phased<T>& w, double progress, const RuleWeights& rw) {
  return pick_phased(w, progress, rw.t_early, rw.t_mid);
}

// True iff progress is in the early phase (progress < t_early).
bool in_early_phase(double progress, const RuleWeights& w) {
  return progress < w.t_early;
}

// True iff progress is in the mid or earlier phase (progress < t_mid).
bool in_mid_or_earlier_phase(double progress, const RuleWeights& w) {
  return progress < w.t_mid;
}

// Mirrors carcassonne_rules::award_to_majority — every tied-best owner
// receives the full points. Tied 1-1 means BOTH players score `points`.
double owner_award(const std::array<int, kPlayers>& owners, int player,
                   double points) {
  int best = 0;
  for (int count : owners) best = std::max(best, count);
  if (best <= 0 || owners[static_cast<size_t>(player)] != best) return 0.0;
  return points;
}

double ownership_delta(const std::array<int, kPlayers>& owners, int player,
                       double points) {
  const int opp = opponent_of(player);
  return owner_award(owners, player, points) -
         owner_award(owners, opp, points);
}

// Marginal feature evaluation — same semantics as before, computes
//   (me_after - opp_after) - (me_before - opp_before)
// using award_to_majority semantics on both ends.
struct BeforeRegion {
  int tiles = 0;
  int shields = 0;
  std::array<int, kPlayers> owners{0, 0};
};

double marginal_feature_delta(
    const FeatureScoreInfo& after, Terrain terrain, int player,
    const std::vector<BeforeRegion>& before_regions) {
  const int opp = opponent_of(player);
  const double points_after = (terrain == kCity)
      ? (after.complete ? 2.0 * static_cast<double>(after.tiles + after.shields)
                        : static_cast<double>(after.tiles + after.shields))
      : static_cast<double>(after.tiles);
  const double me_after = owner_award(after.owners, player, points_after);
  const double opp_after = owner_award(after.owners, opp, points_after);

  double me_before = 0.0;
  double opp_before = 0.0;
  for (const auto& r : before_regions) {
    const double points_before = (terrain == kCity)
        ? static_cast<double>(r.tiles + r.shields)
        : static_cast<double>(r.tiles);
    me_before += owner_award(r.owners, player, points_before);
    opp_before += owner_award(r.owners, opp, points_before);
  }

  return (me_after - opp_after) - (me_before - opp_before);
}

double farm_tile_weight_for_ownership(const std::array<int, kPlayers>& owners, int player,
                            const RuleWeights& w, double progress) {
  const int opp = opponent_of(player);
  const int mine = owners[static_cast<size_t>(player)];
  const int theirs = owners[static_cast<size_t>(opp)];
  if (mine > theirs) {
    return at_phase(w.farm_tile_my, progress, w);
  }
  if (theirs > mine) {
    return -at_phase(w.farm_tile_opp, progress, w);
  }
  if (mine > 0) {
    return at_phase(w.farm_tile_tied, progress, w);
  }
  return at_phase(w.farm_tile_empty, progress, w);
}

double meeple_supply_penalty(const CarcassonneState& s, int player,
                             const RuleWeights& w) {
  const int remaining = s.meeples_remaining[static_cast<size_t>(player)];
  const double progress = game_progress(s);
  if (remaining <= 1) return at_phase(w.supply_low, progress, w);
  if (remaining == 2) return at_phase(w.supply_two, progress, w);
  if (remaining == 3) return at_phase(w.supply_three, progress, w);
  return at_phase(w.supply_many, progress, w);
}

int group_for_action_side(int tile_type, int rotation, Terrain terrain,
                          int side) {
  const auto& def = tile_def(tile_type);
  const auto& groups = terrain == kCity ? def.city_groups : def.road_groups;
  const int count = terrain == kCity ? def.city_group_count
                                     : def.road_group_count;
  for (int gi = 0; gi < count; ++gi) {
    const auto sides =
        rotated_group_sides(groups[static_cast<size_t>(gi)], rotation);
    for (int s : sides) {
      if (s == side) return gi;
    }
  }
  throw std::runtime_error("carcassonne heuristic missing feature group");
}

// ============================================================
// score_place_action — split into per-feature helpers
// ============================================================

double score_place_structural(const CarcassonneState& s, int row, int col,
                              double progress, const RuleWeights& w) {
  constexpr std::array<int, kSides> dr{{-1, 0, 1, 0}};
  constexpr std::array<int, kSides> dc{{0, 1, 0, -1}};
  int adjacent = 0;
  for (int side = 0; side < kSides; ++side) {
    const int nr = row + dr[static_cast<size_t>(side)];
    const int nc = col + dc[static_cast<size_t>(side)];
    if (!s.in_bounds(nr, nc)) continue;
    const int ncell = s.index(nr, nc);
    if (s.board_tile[static_cast<size_t>(ncell)] != kNoTile) adjacent += 1;
  }
  const double center_dist =
      std::abs(static_cast<double>(row) - (kBoardRows - 1) / 2.0) +
      std::abs(static_cast<double>(col) - (kBoardCols - 1) / 2.0);
  return adjacent * at_phase(w.adjacent, progress, w) -
         center_dist * at_phase(w.center_dist, progress, w);
}

double score_place_neighboring_cloisters(const CarcassonneState& tmp, int row,
                                          int col, int player, double progress,
                                          const RuleWeights& w) {
  const double tcc = at_phase(w.tile_cloister_complete, progress, w);
  const double tcn = at_phase(w.tile_cloister_near, progress, w);
  if (tcc == 0.0 && tcn == 0.0) return 0.0;
  double score = 0.0;
  for (int dr = -1; dr <= 1; ++dr) {
    for (int dc = -1; dc <= 1; ++dc) {
      const int cr = row + dr;
      const int cc = col + dc;
      if (!tmp.in_bounds(cr, cc)) continue;
      const int ccell = tmp.index(cr, cc);
      const int chapel_tile = tmp.board_tile[static_cast<size_t>(ccell)];
      if (chapel_tile == kNoTile || !tile_def(chapel_tile).chapel) continue;
      const int owner = tmp.board_meeple_owner[static_cast<size_t>(ccell)];
      if (owner < 0) continue;
      if (tmp.board_meeple_pos[static_cast<size_t>(ccell)] !=
          kMeepleCenterPos) {
        continue;
      }
      const int occupancy = CarcassonneRules::cloister_occupancy(tmp, ccell);
      const double sign = owner == player ? 1.0 : -1.0;
      if (occupancy == 9) {
        score += sign * tcc * 9.0;
      } else if (occupancy >= 6) {
        score += sign * tcn * static_cast<double>(occupancy);
      }
    }
  }
  return score;
}

double score_place_city_road_marginal(
    const CarcassonneState& s, const CarcassonneState& tmp, int row, int col,
    int tile, int rot, int group, Terrain terrain,
    const FeatureScoreInfo& info, int player, double progress,
    const RuleWeights& w) {
  const int opp = opponent_of(player);
  std::vector<BeforeRegion> before_regions;
  std::set<std::pair<int, int>> seen_before_cells;
  const auto sides_in_group_after =
      terrain == kCity
          ? rotated_group_sides(
                tile_def(tile).city_groups[static_cast<size_t>(group)], rot)
          : rotated_group_sides(
                tile_def(tile).road_groups[static_cast<size_t>(group)], rot);
  for (int sd : sides_in_group_after) {
    constexpr std::array<int, kSides> dr2{{-1, 0, 1, 0}};
    constexpr std::array<int, kSides> dc2{{0, 1, 0, -1}};
    const int nr = row + dr2[static_cast<size_t>(sd)];
    const int nc = col + dc2[static_cast<size_t>(sd)];
    if (!s.in_bounds(nr, nc)) continue;
    const int ncell = s.index(nr, nc);
    if (s.board_tile[static_cast<size_t>(ncell)] == kNoTile) continue;
    const int nrot = s.board_rotation[static_cast<size_t>(ncell)];
    const int ntile = s.board_tile[static_cast<size_t>(ncell)];
    const int back = (sd + 2) % kSides;
    if (edge_terrain_at(ntile, nrot, back) != terrain) continue;
    const int ngroup = CarcassonneRules::feature_group_for_side(
        ntile, nrot, terrain, back);
    if (!seen_before_cells.insert({ncell, ngroup}).second) continue;
    const FeatureScoreInfo before_info = (terrain == kCity)
        ? CarcassonneRules::collect_city(s, ncell, back)
        : CarcassonneRules::collect_road(s, ncell, back);
    BeforeRegion r;
    r.tiles = before_info.tiles;
    r.shields = before_info.shields;
    r.owners = before_info.owners;
    before_regions.push_back(r);
  }
  const double marg_w = info.complete
      ? at_phase(terrain == kCity ? w.marginal_city_complete
                                  : w.marginal_road_complete,
                 progress, w)
      : at_phase(terrain == kCity ? w.marginal_city_open
                                  : w.marginal_road_open,
                 progress, w);
  double score = marg_w * marginal_feature_delta(info, terrain, player,
                                                 before_regions);
  if (!info.complete) {
    const int my_owner_count = info.owners[static_cast<size_t>(player)];
    const int opp_owner_count = info.owners[static_cast<size_t>(opp)];
    const Phased<double>& cb = terrain == kCity ? w.city_completion_bonus
                                                : w.road_completion_bonus;
    const Phased<double>& op = terrain == kCity ? w.city_open_penalty
                                                : w.road_open_penalty;
    const Phased<double>& oob = terrain == kCity ? w.opp_city_open_bonus
                                                 : w.opp_road_open_bonus;
    if (my_owner_count > opp_owner_count) {
      score += at_phase(cb, progress, w) /
               static_cast<double>(std::max(1, info.open_edges));
      score -= at_phase(op, progress, w) *
               static_cast<double>(std::max(0, info.open_edges - 1));
    } else if (opp_owner_count > my_owner_count) {
      score += at_phase(oob, progress, w) *
               static_cast<double>(std::max(0, info.open_edges - 1));
    }
  }
  return score;
}

double score_place_city_road_legacy(const FeatureScoreInfo& info,
                                    Terrain terrain, int player,
                                    double progress, const RuleWeights& w) {
  const int opp = opponent_of(player);
  const int my_owner_count = info.owners[static_cast<size_t>(player)];
  const int opp_owner_count = info.owners[static_cast<size_t>(opp)];
  double score = 0.0;
  if (terrain == kCity) {
    const int points = info.complete
        ? 2 * (info.tiles + info.shields)
        : info.tiles + info.shields;
    if (info.complete) {
      const double delta = ownership_delta(info.owners, player,
                                           static_cast<double>(points));
      if (my_owner_count == 0 && opp_owner_count == 0) {
        score += at_phase(w.empty_city_complete, progress, w) *
                 static_cast<double>(points);
      } else {
        score += at_phase(w.owned_city_complete, progress, w) * delta;
      }
    } else {
      score += at_phase(w.open_city_base, progress, w) *
               static_cast<double>(points);
      if (my_owner_count > opp_owner_count) {
        score += at_phase(w.my_city_base, progress, w) +
                 at_phase(w.my_city_points, progress, w) *
                 static_cast<double>(points);
        score += at_phase(w.city_completion_bonus, progress, w) /
                 static_cast<double>(std::max(1, info.open_edges));
        score -= at_phase(w.city_open_penalty, progress, w) *
                 static_cast<double>(std::max(0, info.open_edges - 1));
      } else if (opp_owner_count > my_owner_count) {
        score -= at_phase(w.opp_city_base, progress, w) +
                 at_phase(w.opp_city_points, progress, w) *
                 static_cast<double>(points);
        score += at_phase(w.opp_city_open_bonus, progress, w) *
                 static_cast<double>(std::max(0, info.open_edges - 1));
      }
    }
  } else {  // kRoad
    if (info.complete) {
      const double delta = ownership_delta(info.owners, player,
                                           static_cast<double>(info.tiles));
      if (my_owner_count == 0 && opp_owner_count == 0) {
        score += at_phase(w.empty_road_complete, progress, w) *
                 static_cast<double>(info.tiles);
      } else {
        score += at_phase(w.owned_road_complete, progress, w) * delta;
      }
    } else {
      score += at_phase(w.open_road_base, progress, w) *
               static_cast<double>(info.tiles);
      if (my_owner_count > opp_owner_count) {
        score += at_phase(w.my_road_base, progress, w) +
                 at_phase(w.my_road_points, progress, w) *
                 static_cast<double>(info.tiles);
        score += at_phase(w.road_completion_bonus, progress, w) /
                 static_cast<double>(std::max(1, info.open_edges));
        score -= at_phase(w.road_open_penalty, progress, w) *
                 static_cast<double>(std::max(0, info.open_edges - 1));
      } else if (opp_owner_count > my_owner_count) {
        score -= at_phase(w.opp_road_base, progress, w) +
                 at_phase(w.opp_road_points, progress, w) *
                 static_cast<double>(info.tiles);
        score += at_phase(w.opp_road_open_bonus, progress, w) *
                 static_cast<double>(std::max(0, info.open_edges - 1));
      }
    }
  }
  return score;
}

double score_place_city_road_regions(const CarcassonneState& s,
                                     const CarcassonneState& tmp, int row,
                                     int col, int tile, int rot, int player,
                                     double progress, const RuleWeights& w) {
  double score = 0.0;
  std::array<std::uint8_t, kSides * 2> processed{};
  const int cell = tmp.index(row, col);
  for (int side = 0; side < kSides; ++side) {
    const Terrain terrain = edge_terrain_at(tile, rot, side);
    if (terrain != kCity && terrain != kRoad) continue;
    const int group = group_for_action_side(tile, rot, terrain, side);
    const int key = (terrain == kRoad ? kSides : 0) + group;
    if (processed[static_cast<size_t>(key)] != 0) continue;
    processed[static_cast<size_t>(key)] = 1;

    const auto info = (terrain == kCity)
        ? CarcassonneRules::collect_city(tmp, cell, side)
        : CarcassonneRules::collect_road(tmp, cell, side);

    const double marg_w = info.complete
        ? at_phase(terrain == kCity ? w.marginal_city_complete
                                    : w.marginal_road_complete,
                   progress, w)
        : at_phase(terrain == kCity ? w.marginal_city_open
                                    : w.marginal_road_open,
                   progress, w);
    if (marg_w > 0.0) {
      score += score_place_city_road_marginal(
          s, tmp, row, col, tile, rot, group, terrain, info, player,
          progress, w);
    } else {
      score += score_place_city_road_legacy(info, terrain, player, progress, w);
    }
  }
  return score;
}

double score_place_chapel_self(const CarcassonneState& tmp, int cell, int tile,
                               double progress, const RuleWeights& w) {
  if (!tile_def(tile).chapel) return 0.0;
  return at_phase(w.chapel_place_base, progress, w) +
         at_phase(w.chapel_place_occupancy, progress, w) *
             CarcassonneRules::cloister_occupancy(tmp, cell);
}

double score_place_farms(const CarcassonneState& tmp, int row, int col,
                         int tile, int rot, int player, double progress,
                         const RuleWeights& w) {
  const int cell = tmp.index(row, col);
  const auto& def = tile_def(tile);
  double score = 0.0;
  for (int gi = 0; gi < def.farm_group_count; ++gi) {
    const auto connections = rotated_farm_connections(
        def.farm_groups[static_cast<size_t>(gi)], rot);
    if (connections.empty()) continue;
    const auto farm = CarcassonneRules::collect_farm(tmp, cell, connections[0]);
    score += static_cast<double>(farm.completed_cities) *
             farm_tile_weight_for_ownership(farm.owners, player, w, progress);
  }
  return score;
}

double score_place_action(const CarcassonneState& s, ActionId action,
                          const RuleWeights& w) {
  const int row = CarcassonneRules::decode_row(action);
  const int col = CarcassonneRules::decode_col(action);
  const int rot = CarcassonneRules::decode_rotation(action);
  const int player = s.current_player_;

  CarcassonneState tmp = s;
  const int cell = tmp.index(row, col);
  tmp.board_tile[static_cast<size_t>(cell)] = tmp.current_tile;
  tmp.board_rotation[static_cast<size_t>(cell)] = static_cast<std::int8_t>(rot);
  const int tile = tmp.current_tile;
  const double progress = game_progress(s);

  double score = 0.0;
  score += score_place_structural(s, row, col, progress, w);
  score += score_place_neighboring_cloisters(tmp, row, col, player, progress, w);
  score += score_place_city_road_regions(s, tmp, row, col, tile, rot, player,
                                          progress, w);
  score += score_place_chapel_self(tmp, cell, tile, progress, w);
  score += score_place_farms(tmp, row, col, tile, rot, player, progress, w);
  return score;
}

// ============================================================
// score_meeple_action — split into per-target helpers
// ============================================================

double score_meeple_pass(double supply_penalty, double progress,
                         const RuleWeights& w) {
  return at_phase(w.pass_base, progress, w) +
         at_phase(w.pass_supply, progress, w) * supply_penalty;
}

double score_meeple_cloister(const CarcassonneState& s, int cell,
                             double supply_penalty, double progress,
                             const RuleWeights& w) {
  const int occupancy = CarcassonneRules::cloister_occupancy(s, cell);
  double score = at_phase(w.cloister_base, progress, w) +
                 at_phase(w.cloister_occupancy, progress, w) *
                 static_cast<double>(occupancy);
  if (occupancy == 9) {
    score += at_phase(w.cloister_complete_bonus, progress, w);
  } else {
    score -= at_phase(w.cloister_supply, progress, w) * supply_penalty;
  }
  return score;
}

double score_meeple_city(const FeatureScoreInfo& info, double supply_penalty,
                         double progress, const RuleWeights& w) {
  const int points = info.complete
      ? 2 * (info.tiles + info.shields)
      : info.tiles + info.shields;
  if (info.complete) {
    return at_phase(w.city_meeple_complete_base, progress, w) +
           at_phase(w.city_meeple_complete_points, progress, w) *
           static_cast<double>(points);
  }
  double score = at_phase(w.city_meeple_base, progress, w) +
                 at_phase(w.city_meeple_points, progress, w) *
                 static_cast<double>(points) -
                 at_phase(w.city_meeple_supply, progress, w) * supply_penalty;
  score += at_phase(w.city_completion_bonus, progress, w) /
           static_cast<double>(std::max(1, info.open_edges));
  score -= at_phase(w.city_meeple_open_penalty, progress, w) *
           static_cast<double>(std::max(0, info.open_edges - 1));
  if (info.tiles <= 2) score += at_phase(w.city_small_bonus, progress, w);
  if (info.tiles >= 5) score -= at_phase(w.city_large_penalty, progress, w);
  return score;
}

double score_meeple_road(const FeatureScoreInfo& info, double supply_penalty,
                         double progress, const RuleWeights& w) {
  if (info.complete) {
    return at_phase(w.road_meeple_complete_base, progress, w) +
           at_phase(w.road_meeple_complete_points, progress, w) *
           static_cast<double>(info.tiles);
  }
  return at_phase(w.road_meeple_base, progress, w) +
         at_phase(w.road_meeple_points, progress, w) *
         static_cast<double>(info.tiles) -
         at_phase(w.road_meeple_supply, progress, w) * supply_penalty +
         at_phase(w.road_completion_bonus, progress, w) /
         static_cast<double>(std::max(1, info.open_edges)) -
         at_phase(w.road_meeple_open_penalty, progress, w) *
         static_cast<double>(std::max(0, info.open_edges - 1));
}

double score_meeple_farm(const CarcassonneState& s, const FarmScoreInfo& farm,
                         int player, double progress, double supply_penalty,
                         const RuleWeights& w) {
  const double farm_points =
      static_cast<double>(farm.completed_cities * 3);
  const double farm_potential =
      static_cast<double>(farm.incomplete_cities) *
      at_phase(w.farm_potential, progress, w);
  double score = at_phase(w.farm_base, progress, w) +
                 farm_points * at_phase(w.farm_points, progress, w) +
                 farm_potential +
                 at_phase(w.farm_flat, progress, w) -
                 at_phase(w.farm_supply, progress, w) * supply_penalty;
  // farm_empty_early_penalty: only fire in the EARLY phase (replaces
  // the old farm_empty_until threshold).
  if (farm.completed_cities == 0 && in_early_phase(progress, w)) {
    score -= at_phase(w.farm_empty_early_penalty, progress, w);
  }
  // farm_crowded_penalty: only fire while still in MID-or-earlier
  // phase (replaces the old farm_crowded_until threshold).
  if (meeples_on_board(s, player) >=
          at_phase(w.farm_crowded_meeples, progress, w) &&
      in_mid_or_earlier_phase(progress, w)) {
    score -= at_phase(w.farm_crowded_penalty, progress, w);
  }
  return score;
}

double score_meeple_action(const CarcassonneState& s, ActionId action,
                           const RuleWeights& w) {
  const int player = s.current_player_;
  const int cell = s.index(s.last_row, s.last_col);
  const int tile = s.board_tile[static_cast<size_t>(cell)];
  const int rot = s.board_rotation[static_cast<size_t>(cell)];
  const double supply_penalty = meeple_supply_penalty(s, player, w);
  const double progress = game_progress(s);

  if (action == kMeeplePassAction) {
    return score_meeple_pass(supply_penalty, progress, w);
  }

  const int pos = CarcassonneRules::decode_meeple_position(action);
  if (pos == kMeepleCenterPos) {
    return score_meeple_cloister(s, cell, supply_penalty, progress, w);
  }
  if (pos >= 0 && pos < kSides) {
    const Terrain terrain = edge_terrain_at(tile, rot, pos);
    if (terrain == kCity) {
      const auto info = CarcassonneRules::collect_city(s, cell, pos);
      return score_meeple_city(info, supply_penalty, progress, w);
    }
    if (terrain == kRoad) {
      const auto info = CarcassonneRules::collect_road(s, cell, pos);
      return score_meeple_road(info, supply_penalty, progress, w);
    }
    throw std::runtime_error("carcassonne heuristic meeple feature terrain");
  }
  if (pos >= kSides && pos < kSides + kFarmSides) {
    const auto farm = CarcassonneRules::collect_farm(s, cell, pos - kSides);
    return score_meeple_farm(s, farm, player, progress, supply_penalty, w);
  }
  throw std::runtime_error("carcassonne heuristic invalid meeple action");
}

}  // namespace

CarcassonneProfilePvSettings carcassonne_profile_pv_settings(
    CarcassonneHeuristicProfile profile) {
  const RuleWeights w = weights_for_profile(profile);
  return {w.prior_temperature, w.value_margin_scale};
}

HeuristicResult CarcassonneRuleHeuristic::pick(
    IGameState& masked_state, const IBeliefTracker* /*tracker*/,
    const IGameRules& rules, std::uint64_t rng_seed) const {
  const auto& s = checked_cast<CarcassonneState>(masked_state);
  const auto legal = rules.legal_actions(masked_state);
  const RuleWeights weights = weights_for_profile(profile_);
  HeuristicResult result;
  result.actions = legal;
  result.scores.reserve(legal.size());

  std::mt19937_64 rng(rng_seed);
  for (ActionId action : legal) {
    double score = 0.0;
    if (s.stage == kStagePlaceTile) {
      score = score_place_action(s, action, weights);
    } else if (s.stage == kStageMeeple) {
      score = score_meeple_action(s, action, weights);
    } else {
      throw std::runtime_error("carcassonne heuristic unknown stage");
    }
    score += static_cast<double>(rng() & 0xFFFF) / 65536.0 * 0.001;
    result.scores.push_back(score);
  }
  return result;
}

}  // namespace board_ai::carcassonne
