#pragma once

#include <string>
#include <utility>

#include "../../../engine/core/heuristic.h"

namespace board_ai::carcassonne {

enum class CarcassonneHeuristicProfile {
  kBaseline,
  kSharpPvSoft,
  kSharpCloisterSoftPv,
  kSharpReserveSoftPv,
  kSharpCloisterReserveSoftPv,
  // A/B research profiles (2026-06-22) — see STRATEGY_NOTES.md
  // "Research Findings". Each isolates one hypothesis on top of
  // sharp_weights so eval_model can attribute strength changes.
  kSharpSharefix,           // marginal city/road eval (tied-majority bug fix)
  kSharpOpenEdge,           // turn on completability/open-edge terms in sharp
  kSharpCloisterComplete,   // turn on tile_cloister_* in sharp (not full reserve)
  kSharpFarmPotential,      // turn on farm_potential_* + place-time incomplete cities
};

// PV evaluator settings for a profile. Exposed so the PV evaluator can
// read prior_temperature / value_margin_scale without owning a parallel
// PvWeights struct or duplicating the profile-to-weights mapping. Both
// fields live inside RuleWeights — this struct is just a slim view on
// the relevant subset.
struct CarcassonneProfilePvSettings {
  double prior_temperature;
  double value_margin_scale;
};

CarcassonneProfilePvSettings carcassonne_profile_pv_settings(
    CarcassonneHeuristicProfile profile);

class CarcassonneRuleHeuristic final : public IHeuristic {
 public:
  explicit CarcassonneRuleHeuristic(
      std::string name,
      CarcassonneHeuristicProfile profile =
          CarcassonneHeuristicProfile::kSharpPvSoft)
      : name_(std::move(name)), profile_(profile) {}

  std::string name() const override { return name_; }
  HeuristicResult pick(IGameState& masked_state, const IBeliefTracker* tracker,
                       const IGameRules& rules,
                       std::uint64_t rng_seed) const override;

 private:
  std::string name_;
  CarcassonneHeuristicProfile profile_;
};

}  // namespace board_ai::carcassonne
