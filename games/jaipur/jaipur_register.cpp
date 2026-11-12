#include "../../engine/core/game_registry.h"
#include "../../heuristic/mcts_pv_heuristic.h"
#include "../../heuristic/heuristic_config.h"
#include "jaipur_state.h"
#include "jaipur_rules.h"
#include "jaipur_net_adapter.h"
#include "jaipur_belief_tracker.h"
#include "heuristic/jaipur_rule_heuristic.h"
#include "heuristic/jaipur_heuristic_pv_evaluator.h"
#include "heuristic/jaipur_smart_heuristic.h"

namespace board_ai::jaipur {

namespace {

// events_only_extractor: public goods-flow events, the same for every
// perspective:
//   - "goods_taken"    {player, good_type, count}: a single take.
//   - "goods_sold"     {player, good_type, count}: a sale of `count` cards.
//   - "trade_resolved" {player, take[6], give[7]}: the give atom that
//     completed a trade; per-type counts taken from the market and given to
//     it (give[6] = camels).
// Trade atoms before the last one emit nothing: the prefix is public state.
// There is no private event: the value of a bonus token the actor drew is
// perspective-private knowledge and rides the snapshot as the owner-only
// state slot `bonus_taken` (not tracker memory).
std::vector<PublicEvent> extract_events_only(
    const IGameState& before, ActionId action,
    const IGameState& after, int /*perspective*/) {
  std::vector<PublicEvent> events;
  const auto& db = static_cast<const JaipurState&>(before).data;
  const auto& da = static_cast<const JaipurState&>(after).data;
  const int actor = db.current_player;
  auto flow = [&](const char* kind, int good, int count) {
    AnyMap p;
    p["player"] = std::any(actor);
    p["good_type"] = std::any(good);
    p["count"] = std::any(count);
    events.emplace_back(kind, std::move(p));
  };

  if (is_take_action(action)) {
    flow("goods_taken", action - kActTakeBase, 1);
  } else if (is_sell_action(action)) {
    flow("goods_sold", sell_type(action), sell_count(action));
  } else if (is_trade_give_action(action) && jaipur_trade_idle(da)) {
    std::vector<int> take(db.trade_take.begin(), db.trade_take.end());
    std::vector<int> give(db.trade_give.begin(), db.trade_give.end());
    give[trade_give_type(action)] += trade_give_count(action);
    AnyMap p;
    p["player"] = std::any(actor);
    p["take"] = std::any(take);
    p["give"] = std::any(give);
    events.emplace_back("trade_resolved", std::move(p));
  }
  return events;
}

GameBundle make_jaipur(std::uint64_t seed) {
  GameBundle b;
  b.game_id = "jaipur";
  auto s = std::make_unique<JaipurState>();
  s->reset_with_seed(seed);
  b.state = std::move(s);
  b.rules = std::make_unique<JaipurRules>();
  b.value_model = std::make_unique<DefaultStateValueModel>();
  b.encoder = std::make_unique<JaipurTokenEncoder>();
  b.belief_tracker = std::make_unique<JaipurBeliefTracker>();
  b.state_serializer = make_jaipur_serializer();
  b.action_descriptor = make_jaipur_descriptor();
  b.heuristics.register_named(
      "rule",
      std::make_unique<JaipurRuleHeuristic>("rule"));
  // Heuristic-driven MCTS: a policy/value evaluator backed by the rule
  // heuristic (P = softmax of rule scores, V = tanh of expected-score gap),
  // wrapped in MctsPVHeuristic. This lets the web run a strong,
  // tracker-sampling MCTS AI WITHOUT any trained ONNX model. Simulations /
  // c_puct come from heuristic.json's "rule_mcts" entry.
  auto pv_eval = std::make_unique<JaipurHeuristicPVEvaluator>("jaipur_rule_pv");
  auto* pv_eval_ptr = pv_eval.get();
  b.heuristic_pv_evaluators.push_back(std::move(pv_eval));
  const auto hc = board_ai::heuristic::load_mcts_pv_heuristic_config(
      "jaipur", "rule_mcts", /*require_model_path=*/false);
  b.heuristics.register_named(
      "rule_mcts",
      board_ai::heuristic::MctsPVHeuristic::make_from_evaluator(
          "rule_mcts", pv_eval_ptr, b.value_model.get(),
          hc.simulations, hc.c_puct, hc.world_cap, hc.leaf_batch_size));
  // Composite: MCTS for the first decision of a turn (sell/take/trade
  // choice), fast rule heuristic for the later atoms of a trade.
  const auto& mcts_base = b.heuristics.get("rule_mcts");
  const auto* mcts_ptr = dynamic_cast<const board_ai::heuristic::MctsPVHeuristic*>(
      &mcts_base);
  b.heuristics.register_named(
      "rule_smart",
      std::make_unique<JaipurSmartHeuristic>("rule_smart", mcts_ptr));
  b.install_event_protocol(
      extract_events_only,
      []() -> const viz::VisibilitySchema& { return JaipurState::schema(); });
  // Stalling penalty: any selfplay episode that reaches max_game_plies
  // without a real terminal (piles_exhausted >= 3 or a failed market refill) is routed
  // through this adjudicator. Returning -1 means "no winner"; the framework
  // then assigns z = -1 to BOTH players (selfplay_runner.cpp:592). This is
  // the anti-stall signal — networks that drive games to truncation see
  // the same negative reward as a loss, eliminating the wait-it-out
  // equilibrium we observed in early training (cap_hits 50/100, plies 150).
  b.adjudicator = [](const IGameState&) -> int { return -1; };
  return b;
}

GameRegistrar reg("jaipur", make_jaipur);

}  // namespace

}  // namespace board_ai::jaipur
