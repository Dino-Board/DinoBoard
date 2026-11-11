#pragma once

#include <cstdint>
#include <functional>

#include "../core/game_interfaces.h"

namespace board_ai::search {

// scorer(state, perspective, depth_remaining) → tiebreak bonus.
// Game-side scorer folds "prefer faster wins" (depth_remaining) and
// "bigger margin" (state-derived) into one scalar; tail solver applies a
// single auxiliary_weight on top.
using MarginScorer = std::function<float(const IGameState&, int, int)>;

struct TailSolveConfig {
  int depth_limit = 5;
  std::int64_t node_budget = 10'000'000;
  int time_limit_ms = 0;
  // Tiebreak among proven wins only (terminal value >= 1.0):
  //   v += auxiliary_weight * margin_scorer(state, p, depth_remaining)
  // Loss / draw branches are unaffected. Bonus must keep v >= 1.0 so
  // net_mcts.cpp's |value|>=1 adoption gate still triggers, and must stay
  // small relative to 1.0 so a non-terminal cutoff never reaches 1.0.
  // Recommended scorer shape (Splendor / Azul):
  //   `depth_remaining + 0.1f * normalized_margin`,
  // weight = 0.01.
  float auxiliary_weight = 0.0f;
  MarginScorer margin_scorer;
  // Hidden-info games (MCTS has a root belief tracker): the solver runs on
  // belief-sampled worlds, and a proven win is only adopted if the chosen
  // action is also a proven win in this many independently sampled worlds
  // in total. The session state's hidden slots hold stale bytes, so solving
  // it directly is not a proof of anything.
  int belief_worlds = 8;
};

enum class TailSolveOutcome {
  kUnknown = 0,
  kProvenWin = 1,
  kProvenLoss = 2,
  kProvenDraw = 3,
};

struct TailSolveResult {
  TailSolveOutcome outcome = TailSolveOutcome::kUnknown;
  ActionId best_action = -1;
  float value = 0.0f;
  std::int64_t nodes_searched = 0;
  double elapsed_ms = 0.0;
  bool budget_exceeded = false;
};

class ITailSolver {
 public:
  virtual ~ITailSolver() = default;
  virtual TailSolveResult solve(
      IGameState& state,
      const IGameRules& rules,
      const IStateValueModel& value_model,
      int perspective_player,
      const TailSolveConfig& config) const = 0;
};

class AlphaBetaTailSolver final : public ITailSolver {
 public:
  TailSolveResult solve(
      IGameState& state,
      const IGameRules& rules,
      const IStateValueModel& value_model,
      int perspective_player,
      const TailSolveConfig& config) const override;
};

}  // namespace board_ai::search
