#pragma once

// Ports of ark-nova-ai's GreedyBot and LookaheadBot (arknova/bot.py): a valuation of the position
// in appeal-equivalents plus a search over the mover's own turns (opponents passive).
//
// Each search runs on one world sampled by the tracker from the masked state, so it never reads
// the opponent's hand or the deck order. Each next atom is scored by the best option below it.

#include <string>
#include <utility>

#include "../../../engine/core/heuristic.h"
#include "../arknova_core.h"

namespace board_ai::arknova {

struct GreedyWeights {
  double money = 0.28;        // per money, for the first kMoneyWorking
  double spare_money = 0.08;  // per money for the next kMoneySpare; beyond is worth nothing
  double conservation = 2.6;  // per conservation point on a card, when estimating a play
  double income = 0.15;       // per money of break income, per break left
  double hold = 0.35;         // share of a playable card's net gain credited while in hand
  double xtoken = 0.8;
  double upgrade = 5.0;  // per upgraded action card, scaled by time left
  double reputation = 0.4;
  double reputation_high = 0.3;  // per reputation step above 9
  double worker = 1.9;           // first extra worker; later ones decay
  double partner = 2.0;
  double university = 2.0;
  double decay = 0.8;
  double sponsor_level = 0.8;  // lasting effects of a sponsor in play, per level
  double slot = 0.15;          // per slot step of an action card
  double idle_enclosure = 0.45;  // share of its cost kept by an empty enclosure nothing fits
};

// bot.evaluate: value of the position for player p.
double greedy_evaluate(const State& s, int p, const GreedyWeights& w = GreedyWeights());

class ArknovaGreedyHeuristic final : public IHeuristic {
 public:
  explicit ArknovaGreedyHeuristic(std::string name) : name_(std::move(name)) {}

  std::string name() const override { return name_; }
  HeuristicResult pick(IGameState& masked_state, const IBeliefTracker* tracker, const IGameRules& rules,
                       std::uint64_t rng_seed) const override;

 private:
  std::string name_;
  GreedyWeights w_;
};

struct LookaheadConfig {
  int depth = 3;     // own turns searched
  int width = 8;     // lines kept per depth
  int variants = 10; // second steps tried per turn option (best of each kind)
  int follow = 8;    // options compared at each later step of a simulated turn
};

// LookaheadBot with reply=False and open_info=False: a beam over the mover's next `depth` whole
// turns. The reference bot plans at the turn decision and replays that plan for the rest of the
// turn; this heuristic is stateless, so every decision of the mover's own turn re-runs the beam
// from the current position (completing this turn first). Other decisions are GreedyBot's.
class ArknovaLookaheadHeuristic final : public IHeuristic {
 public:
  ArknovaLookaheadHeuristic(std::string name, LookaheadConfig cfg) : name_(std::move(name)), cfg_(cfg) {}

  std::string name() const override { return name_; }
  HeuristicResult pick(IGameState& masked_state, const IBeliefTracker* tracker, const IGameRules& rules,
                       std::uint64_t rng_seed) const override;

 private:
  std::string name_;
  LookaheadConfig cfg_;
  GreedyWeights w_;
};

struct MonteCarloConfig {
  LookaheadConfig beam;       // ranks the options and plays the decisions without rollouts
  int candidates = 8;         // best options by the beam compared by rollouts
  int rollouts = 16;          // sampled worlds, each played once from every candidate
  int replies = 2;            // opponent turns in a rollout; it stops at the mover's next turn after the last
  double prior_weight = 0.5;  // weight of a candidate's beam value, added to its rollout mean
  double seconds = 0.0;       // > 0: no further rollouts once a decision has taken this long (at least one)
};

// The beam refined by Monte Carlo rollouts at every decision of the mover's own turn. The beam
// values the options; the ones it keeps to its last depth (at most `candidates`, best first) are
// compared when there are two or more. Each rollout samples a world from the masked state
// (tracker), plays every candidate in it, then everyone's decisions by greedy until the mover's
// turn after the opponent's `replies`-th (to the end once the game end is triggered), and scores
// greedy_evaluate(mover) - greedy_evaluate(opponent) there, or the final score difference with a
// win bonus. All candidates of a rollout share its world and random stream. A candidate's value
// is its rollout mean plus prior_weight times its beam value. Other players' turns are greedy's.
//
// Measured with seats swapped on each deal: beats beam_3x8 64% over 400 games on deals not used
// to choose the settings (66% over the 200 that were), and the same rollouts at the turn decision
// only 60% over 200. Over 200 games each vs beam_3x8: rollouts at the turn decision only 57-59%
// (16 or 32 rollouts alike), a wider beam alone (3x16, 3x24) 49%, candidates the beam dropped
// (ranked by their value after the own turn) 48-50%, deeper rollouts (3 replies) 53%, beam_2x3
// turns inside rollouts 55%. On one core a turn takes about 1.5 s on average, up to ~9 s late in
// the game when rollouts run to the end.
class ArknovaMonteCarloHeuristic final : public IHeuristic {
 public:
  ArknovaMonteCarloHeuristic(std::string name, MonteCarloConfig cfg) : name_(std::move(name)), cfg_(cfg) {}

  std::string name() const override { return name_; }
  HeuristicResult pick(IGameState& masked_state, const IBeliefTracker* tracker, const IGameRules& rules,
                       std::uint64_t rng_seed) const override;

 private:
  std::string name_;
  MonteCarloConfig cfg_;
  GreedyWeights w_;
};

}  // namespace board_ai::arknova
