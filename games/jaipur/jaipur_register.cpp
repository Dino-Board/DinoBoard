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

// events_only_extractor: per-perspective events derived from the before/after
// state diff + action. Emits:
//   - "goods_taken"  (public): a player took a goods card of type g from the
//     market (take action). Lets every tracker update that player's net_in.
//   - "goods_sold"   (public): a player sold all of type g. Triggers the
//     strong sell signal (sold=true, net_in=0 for that type/player).
//   - "exchange_resolved" (public): an exchange completed, carrying the goods
//     types taken from the market and the goods types given to it, so trackers
//     update net_in per type (take-give, clamped ≥ 0).
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

  // --- Public goods-flow events (every perspective sees these) ---
  if (action >= kActSellBase && action < kActSellBase + kGoodTypes) {
    AnyMap p; p["player"] = std::any(actor);
    p["good_type"] = std::any(action - kActSellBase);
    events.emplace_back("goods_sold", std::move(p));
  } else if (action >= kActTakeBase && action < kActTakeBase + kGoodTypes) {
    AnyMap p; p["player"] = std::any(actor);
    p["good_type"] = std::any(action - kActTakeBase);
    events.emplace_back("goods_taken", std::move(p));
  } else if (db.exchange_phase != kExchPhaseIdle &&
             da.exchange_phase == kExchPhaseIdle) {
    // Exchange COMMITTED (phase → idle via a final commit). There is no
    // cancel action — once an exchange begins it must complete — so any
    // phase→idle transition is a commit. Derive moved goods from the
    // MARKET count delta (robust to which sub-action triggered the commit):
    // a good whose market count dropped was taken into the actor's hand;
    // a good whose market count rose was given from hand.
    std::vector<int> take_types, give_types;
    for (int g = 0; g < kGoodTypes; ++g) {
      int delta = da.market_count[g] - db.market_count[g];
      for (int i = 0; i < -delta; ++i) take_types.push_back(g);  // left market → taken
      for (int i = 0; i < delta; ++i) give_types.push_back(g);   // entered market → given
    }
    AnyMap p; p["player"] = std::any(actor);
    p["take_types"] = std::any(take_types);
    p["give_types"] = std::any(give_types);
    events.emplace_back("exchange_resolved", std::move(p));
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
  b.encoder = std::make_unique<JaipurFeatureEncoder>();
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
  // Composite: MCTS for idle-phase decisions (sell/take/exchange choice),
  // fast rule heuristic for exchange sub-actions (no 20k MCTS per sub-action).
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
