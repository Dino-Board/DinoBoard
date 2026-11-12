#pragma once
#include <string>
#include "../../../engine/core/heuristic.h"

// Forward declarations - defined in jaipur_state.h / jaipur_rules.h
namespace board_ai::jaipur {
struct JaipurData;
struct JaipurTrade;
}

namespace board_ai::jaipur {

// Expected value of one face-down bonus token of `tier` from `observer`'s
// knowledge only (full multiset minus the observer's own draws). Shared by
// the rule heuristic, the heuristic PV evaluator and the token encoder.
double jaipur_bonus_expected(int tier, const JaipurData& d, int observer);

// Rule-based heuristic for Jaipur. Scores the current legal actions from the
// acting player's perspective using public info + the actor's own hand:
//  - sell k of g: sum of the top k value tokens + expected bonus value
//  - take good: prefer expensive goods + type synergy with hand
//  - take camels: conditional on herd size / opponent hand fullness (rulebook
//    advice 1-2)
//  - trade atoms: the best complete trade below the atom (value of the goods
//    taken minus the goods / camels paid)
//
// Bonus token values are estimated by the mean of the residual value pool
// the acting player knows (they are face-down; the pile's real composition
// is hidden state and is masked out of the heuristic's input).
class JaipurRuleHeuristic final : public IHeuristic {
 public:
  explicit JaipurRuleHeuristic(std::string instance_name = "rule")
      : name_(std::move(instance_name)) {}

  std::string name() const override { return name_; }

  HeuristicResult pick(IGameState& masked_state,
                       const IBeliefTracker* tracker,
                       const IGameRules& rules,
                       std::uint64_t rng_seed) const override;

 private:
  std::string name_;
  double score_sell(int good_type, int count, const JaipurData& d) const;
  double score_take(int good_type, const JaipurData& d) const;
  double score_camels(const JaipurData& d) const;
  double score_trade(const JaipurTrade& trade, const JaipurData& d) const;
};

}  // namespace board_ai::jaipur
