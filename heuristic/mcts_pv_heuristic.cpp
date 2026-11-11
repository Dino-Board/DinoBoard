#include "mcts_pv_heuristic.h"

#include <stdexcept>
#include <utility>

#include "../engine/core/feature_encoder.h"
#include "../engine/infer/onnx_policy_value_evaluator.h"
#include "../engine/search/net_mcts.h"

namespace board_ai::heuristic {

std::unique_ptr<IHeuristic> MctsPVHeuristic::make_from_evaluator(
    std::string name,
    const search::IPolicyValueEvaluator* evaluator,
    const IStateValueModel* value_model,
    int simulations,
    float c_puct,
    int world_cap,
    int leaf_batch_size) {
  if (!evaluator) {
    throw std::runtime_error(
        "MctsPVHeuristic::make_from_evaluator: evaluator must not be null ('" +
        name + "')");
  }
  if (!value_model) {
    throw std::runtime_error(
        "MctsPVHeuristic::make_from_evaluator: value_model must not be null ('" +
        name + "')");
  }
  if (world_cap < 0) {
    throw std::runtime_error(
        "MctsPVHeuristic::make_from_evaluator: world_cap must be non-negative ('" +
        name + "')");
  }
  if (leaf_batch_size < 1) {
    throw std::runtime_error(
        "MctsPVHeuristic::make_from_evaluator: leaf_batch_size must be >= 1 ('" +
        name + "')");
  }
  return std::unique_ptr<IHeuristic>(new MctsPVHeuristic(
      std::move(name), evaluator, /*owned=*/nullptr, value_model,
      simulations, c_puct, world_cap, leaf_batch_size));
}

std::unique_ptr<IHeuristic> MctsPVHeuristic::make_from_onnx(
    std::string name,
    const std::string& model_path,
    const IFeatureEncoder* encoder,
    const IStateValueModel* value_model,
    int simulations,
    float c_puct,
    int world_cap,
    int leaf_batch_size) {
  if (model_path.empty()) {
    throw std::runtime_error(
        "MctsPVHeuristic::make_from_onnx: model_path must not be empty ('" +
        name + "')");
  }
  if (!encoder) {
    throw std::runtime_error(
        "MctsPVHeuristic::make_from_onnx: encoder must not be null ('" +
        name + "')");
  }
  if (!value_model) {
    throw std::runtime_error(
        "MctsPVHeuristic::make_from_onnx: value_model must not be null ('" +
        name + "')");
  }
  if (world_cap < 0) {
    throw std::runtime_error(
        "MctsPVHeuristic::make_from_onnx: world_cap must be non-negative ('" +
        name + "')");
  }
  if (leaf_batch_size < 1) {
    throw std::runtime_error(
        "MctsPVHeuristic::make_from_onnx: leaf_batch_size must be >= 1 ('" +
        name + "')");
  }
  auto onnx = std::make_unique<infer::OnnxPolicyValueEvaluator>(
      model_path, encoder);
  if (!onnx->is_ready()) {
    throw std::runtime_error(
        "MctsPVHeuristic::make_from_onnx: failed to load model '" +
        model_path + "': " + onnx->last_error());
  }
  const search::IPolicyValueEvaluator* eval_ptr = onnx.get();
  auto h = std::unique_ptr<MctsPVHeuristic>(new MctsPVHeuristic(
      std::move(name), eval_ptr, std::move(onnx), value_model,
      simulations, c_puct, world_cap, leaf_batch_size));
  h->model_path_hint_ = model_path;
  h->encoder_hint_ = encoder;
  return h;
}

MctsPVHeuristic::MctsPVHeuristic(
    std::string name,
    const search::IPolicyValueEvaluator* evaluator,
    std::unique_ptr<infer::OnnxPolicyValueEvaluator> owned,
    const IStateValueModel* value_model,
    int simulations,
    float c_puct,
    int world_cap,
    int leaf_batch_size)
    : name_(std::move(name)),
      evaluator_(evaluator),
      owned_evaluator_(std::move(owned)),
      value_model_(value_model),
      simulations_(simulations),
      c_puct_(c_puct),
      world_cap_(world_cap),
      leaf_batch_size_(leaf_batch_size) {}

MctsPVHeuristic::~MctsPVHeuristic() = default;

void MctsPVHeuristic::replace_evaluator(
    const search::IPolicyValueEvaluator* evaluator) {
  if (!evaluator) {
    throw std::runtime_error(
        "MctsPVHeuristic::replace_evaluator: evaluator must not be null ('" +
        name_ + "')");
  }
  evaluator_ = evaluator;
  // Drop the per-bundle ONNX session — the BatchedEvaluator pool now owns
  // the actual model, and freeing here releases ~ONNX session memory per
  // sampling worker (multiplied by num_workers, this matters at scale).
  owned_evaluator_.reset();
}

search::NetMctsConfig MctsPVHeuristic::search_config() const {
  search::NetMctsConfig cfg{};
  cfg.simulations = simulations_;
  cfg.c_puct = c_puct_;
  cfg.world_cap = world_cap_;
  cfg.leaf_batch_size = leaf_batch_size_;
  // A heuristic is a fixed yardstick (eval benchmark opponent, guidance
  // source, web "heuristic" level): it keeps the UCT2 parent count it has
  // always searched with, whatever the engine default or the train-config
  // switch of the model it is measured against. Otherwise win rates against
  // it would not be comparable across that switch or across engine versions.
  cfg.uct2_arrival_visits = false;
  return cfg;
}

HeuristicResult MctsPVHeuristic::pick(IGameState& masked_state,
                                      const IBeliefTracker* tracker,
                                      const IGameRules& rules,
                                      std::uint64_t rng_seed) const {
  search::NetMctsConfig cfg = search_config();
  cfg.root_belief_tracker = tracker;

  search::NetMcts mcts(cfg);
  search::NetMctsStats stats{};
  mcts.search_root(masked_state, rules, *value_model_, *evaluator_,
                   &stats, rng_seed);

  HeuristicResult out;
  out.simulations = stats.simulations_done;
  out.actions = stats.root_actions;
  out.scores.reserve(stats.root_action_visits.size());
  for (int v : stats.root_action_visits) {
    out.scores.push_back(static_cast<double>(v));
  }
  // Fall-back: if MCTS didn't run any sims (shouldn't happen, but guard
  // against zero-budget configs), return uniform scores so the caller's
  // sample_heuristic_index doesn't divide by zero.
  if (out.actions.empty()) {
    out.actions = rules.legal_actions(masked_state);
    out.scores.assign(out.actions.size(), 1.0);
  }
  return out;
}

}  // namespace board_ai::heuristic
