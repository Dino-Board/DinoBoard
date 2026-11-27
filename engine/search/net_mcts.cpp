#include "net_mcts.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <unordered_map>

#include "../core/rng_salt.h"
#include "../core/schema_hash.h"

namespace board_ai::search {

namespace {

class SplitMix64Engine {
 public:
  using result_type = std::uint64_t;

  explicit SplitMix64Engine(std::uint64_t seed) : state_(seed) {}

  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return UINT64_MAX; }

  result_type operator()() {
    return splitmix64(state_);
  }

 private:
  std::uint64_t state_;
};

struct Edge {
  ActionId action = -1;
  float prior = 0.0f;
  int child = -1;           // index into `nodes` vector; -1 if not yet traversed
  int visit_count = 0;
  float value_sum = 0.0f;
  // In-flight descents through this edge awaiting leaf evaluation. Each
  // counts as one extra visit with value -1 during selection only; never
  // touches visit_count / value_sum, so it reverts exactly.
  int virtual_loss = 0;
};

struct Node {
  int to_play = 0;
  bool expanded = false;
  int visit_count = 0;
  float value_sum = 0.0f;
  std::vector<Edge> edges{};
  int virtual_loss = 0;
  // Network value of the node for `to_play`, set at expansion (Gumbel's
  // mixed-value completion of unvisited edges reads it).
  float raw_value = 0.0f;
};

// One (edge, outcome) pair: the traversals of `edge` of node `parent` that
// ended in node `child`. An edge has several outcomes when the action's
// result differs between simulations (visible chance, or the next player's
// private state in the sampled world), so these counts — not the edge's own —
// say how often a path reached `child` through that edge (UCT2 parent count,
// see NetMctsConfig::uct2_arrival_visits).
struct ArrivalKey {
  int parent = -1;
  int edge = -1;
  int child = -1;
  bool operator==(const ArrivalKey& o) const {
    return parent == o.parent && edge == o.edge && child == o.child;
  }
};

struct ArrivalKeyHash {
  std::size_t operator()(const ArrivalKey& k) const {
    std::uint64_t h = murmur3_fmix64(static_cast<std::uint64_t>(
        static_cast<std::uint32_t>(k.parent)));
    h = murmur3_fmix64(h ^ static_cast<std::uint32_t>(k.edge));
    h = murmur3_fmix64(h ^ static_cast<std::uint32_t>(k.child));
    return static_cast<std::size_t>(h);
  }
};

// Same bookkeeping as Edge::visit_count / Edge::virtual_loss, per outcome.
struct ArrivalCount {
  int visit_count = 0;
  int virtual_loss = 0;
};

static float clip_value(float v, float lim) {
  const float cap = std::max(0.0f, lim);
  return std::max(-cap, std::min(cap, v));
}

// Root priors p -> p^(1/T), renormalised (NetMctsConfig::root_policy_temperature).
// Computed relative to the largest prior so the powers stay in range.
static void apply_root_policy_temperature(Node& root, float temperature) {
  if (temperature == 1.0f || root.edges.size() < 2) return;
  float max_p = 0.0f;
  for (const Edge& e : root.edges) max_p = std::max(max_p, e.prior);
  if (max_p <= 0.0f) return;
  const double inv_t = 1.0 / static_cast<double>(temperature);
  std::vector<double> w(root.edges.size(), 0.0);
  double sum = 0.0;
  for (size_t i = 0; i < root.edges.size(); ++i) {
    const double rel = static_cast<double>(std::max(0.0f, root.edges[i].prior)) /
                       static_cast<double>(max_p);
    w[i] = rel > 0.0 ? std::pow(rel, inv_t) : 0.0;
    sum += w[i];
  }
  for (size_t i = 0; i < root.edges.size(); ++i) {
    root.edges[i].prior = static_cast<float>(w[i] / sum);
  }
}

// `total_alpha` > 0: per-action concentration sums to `total_alpha`
// (NetMctsConfig::root_dirichlet_total_alpha), either uniformly (`total_alpha / n`)
// or KataGo-shaped (`shaped == true`: half uniform, half proportional to excess
// log-prior above mean capped at max(0.01, 1/n)). Otherwise fixed `alpha`.
static void apply_root_dirichlet_noise(Node& root, float alpha, float total_alpha,
                                       bool shaped, float epsilon, std::uint64_t seed) {
  if (root.edges.empty()) return;
  if (shaped) {
    const std::size_t n = root.edges.size();
    const double eps = std::max(0.0, std::min(1.0, static_cast<double>(epsilon)));
    if (total_alpha <= 1e-8f || eps <= 1e-8) return;
    const double uniform_prob = 1.0 / static_cast<double>(n);
    const double cap = std::max(0.01, uniform_prob);
    std::vector<double> alpha_prop(n, 0.0);
    double log_sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double p = std::max(0.0, static_cast<double>(root.edges[i].prior));
      alpha_prop[i] = std::log(std::min(cap, p) + 1e-20);
      log_sum += alpha_prop[i];
    }
    const double log_mean = log_sum / static_cast<double>(n);
    double prop_sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      alpha_prop[i] = std::max(0.0, alpha_prop[i] - log_mean);
      prop_sum += alpha_prop[i];
    }
    if (prop_sum <= 1e-12) {
      for (std::size_t i = 0; i < n; ++i) alpha_prop[i] = uniform_prob;
    } else {
      for (std::size_t i = 0; i < n; ++i) {
        alpha_prop[i] = 0.5 * (alpha_prop[i] / prop_sum + uniform_prob);
      }
    }
    SplitMix64Engine rng(seed);
    std::vector<double> noise(n, 0.0);
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double ai = alpha_prop[i] * static_cast<double>(total_alpha);
      if (ai <= 1e-12) continue;
      std::gamma_distribution<double> gamma(ai, 1.0);
      const double g = std::max(0.0, gamma(rng));
      if (std::isfinite(g)) {
        noise[i] = g;
        sum += g;
      }
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) return;
    for (std::size_t i = 0; i < n; ++i) {
      const double nd = noise[i] / sum;
      const double p = std::max(0.0, static_cast<double>(root.edges[i].prior));
      root.edges[i].prior = static_cast<float>((1.0 - eps) * p + eps * nd);
    }
    return;
  }

  const float a = total_alpha > 0.0f
      ? total_alpha / static_cast<float>(root.edges.size())
      : std::max(0.0f, alpha);
  const float eps = std::max(0.0f, std::min(1.0f, epsilon));
  if (a <= 1e-8f || eps <= 1e-8f) return;
  SplitMix64Engine rng(seed);
  std::gamma_distribution<float> gamma(a, 1.0f);
  std::vector<float> noise(root.edges.size(), 0.0f);
  float sum = 0.0f;
  for (size_t i = 0; i < root.edges.size(); ++i) {
    const float g = std::max(0.0f, gamma(rng));
    noise[i] = g;
    sum += g;
  }
  if (sum <= 1e-12f) return;
  for (size_t i = 0; i < root.edges.size(); ++i) {
    const float n = noise[i] / sum;
    const float p = std::max(0.0f, root.edges[i].prior);
    root.edges[i].prior = (1.0f - eps) * p + eps * n;
  }
}

// Multinomial sample from edge priors. Used by kFrozenPrior on opponent
// nodes — descent picks an edge proportional to the prior set at expand
// time. Priors have already been validated finite/non-negative and
// re-normalized to sum to 1.0 in expand_node, so accumulation is stable.
// Falls back to argmax-of-prior if numeric drift makes the cumulative
// sum non-positive (defensive — shouldn't happen given expand_node's
// rejection of zero-mass priors).
static int sample_index_by_prior(
    const std::vector<Edge>& edges, std::mt19937_64& rng) {
  float total = 0.0f;
  for (const auto& e : edges) total += std::max(0.0f, e.prior);
  if (!std::isfinite(total) || total <= 0.0f) {
    int best = 0;
    float best_p = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < edges.size(); ++i) {
      const float p = edges[i].prior;
      if (p > best_p) { best_p = p; best = static_cast<int>(i); }
    }
    return best;
  }
  std::uniform_real_distribution<float> u01(0.0f, total);
  const float r = u01(rng);
  float acc = 0.0f;
  for (size_t i = 0; i < edges.size(); ++i) {
    acc += std::max(0.0f, edges[i].prior);
    if (r <= acc) return static_cast<int>(i);
  }
  return static_cast<int>(edges.size()) - 1;
}

static void validate_leaf_values(
    const std::vector<float>& values,
    int num_players,
    const char* context) {
  if (static_cast<int>(values.size()) < num_players) {
    throw std::runtime_error(
        std::string("MCTS: ") + context + " returned " +
        std::to_string(values.size()) + " values for " +
        std::to_string(num_players) + " players");
  }
  for (int p = 0; p < num_players; ++p) {
    if (!std::isfinite(values[static_cast<size_t>(p)])) {
      throw std::runtime_error(
          std::string("MCTS: ") + context +
          " returned non-finite value for player " + std::to_string(p));
    }
  }
}

static float safe_log(float p) {
  return std::log(std::max(p, std::numeric_limits<float>::min()));
}

// sigma(completedQ) for every edge of `node` (mctx
// qtransform_completed_by_mix_value): visited edges keep their mean value,
// unvisited ones take the mixed value
//   (raw_value + sum_N * sum_{visited} p*q / sum_{visited} p) / (sum_N + 1),
// the completed values are rescaled to [0, 1] over the node and scaled by
// (c_visit + max N) * c_scale. Q reads backed-up visits only; `count_inflight`
// adds in-flight descents to the max-N scale, matching the counts the caller
// selects with.
static void gumbel_sigma_q(const Node& node, const GumbelConfig& g,
                           double value_clip, bool count_inflight,
                           std::vector<float>* out) {
  const size_t n = node.edges.size();
  out->assign(n, 0.0f);
  double sum_n = 0.0;
  double visited_prior = 0.0;
  double weighted_q = 0.0;
  int max_n = 0;
  for (const Edge& e : node.edges) {
    max_n = std::max(max_n, e.visit_count + (count_inflight ? e.virtual_loss : 0));
    if (e.visit_count > 0) {
      const double q = static_cast<double>(e.value_sum) / e.visit_count;
      sum_n += e.visit_count;
      visited_prior += e.prior;
      weighted_q += static_cast<double>(e.prior) * q;
    }
  }
  const double v_mix = (sum_n > 0.0 && visited_prior > 0.0)
      ? (static_cast<double>(node.raw_value) + sum_n * (weighted_q / visited_prior)) / (sum_n + 1.0)
      : static_cast<double>(node.raw_value);
  double lo = std::numeric_limits<double>::infinity();
  double hi = -std::numeric_limits<double>::infinity();
  std::vector<double> completed(n);
  for (size_t i = 0; i < n; ++i) {
    const Edge& e = node.edges[i];
    completed[i] = e.visit_count > 0
        ? static_cast<double>(e.value_sum) / e.visit_count : v_mix;
    lo = std::min(lo, completed[i]);
    hi = std::max(hi, completed[i]);
  }
  const double scale = (static_cast<double>(g.c_visit) + max_n) * g.c_scale;
  const double span = g.rescale_values
      ? std::max(hi - lo, 1e-8)
      : std::max(2.0 * value_clip, 1e-6);
  for (size_t i = 0; i < n; ++i) {
    (*out)[i] = static_cast<float>(scale * (completed[i] - lo) / span);
  }
}

// Gumbel interior selection: argmax pi'(a) - N(a) / (1 + sum N),
// pi' = softmax(logits + sigma(completedQ)).
static int gumbel_interior_edge(const Node& node, const GumbelConfig& g,
                                double value_clip, std::vector<float>& scratch) {
  gumbel_sigma_q(node, g, value_clip, true, &scratch);
  const size_t n = node.edges.size();
  double max_z = -std::numeric_limits<double>::infinity();
  int total_n = 0;
  for (size_t i = 0; i < n; ++i) {
    const double z = safe_log(node.edges[i].prior) + scratch[i];
    scratch[i] = static_cast<float>(z);
    max_z = std::max(max_z, z);
    total_n += node.edges[i].visit_count + node.edges[i].virtual_loss;
  }
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double w = std::exp(static_cast<double>(scratch[i]) - max_z);
    scratch[i] = static_cast<float>(w);
    sum += w;
  }
  int best = -1;
  double best_score = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < n; ++i) {
    const Edge& e = node.edges[i];
    const double score = static_cast<double>(scratch[i]) / sum -
        static_cast<double>(e.visit_count + e.virtual_loss) / (1.0 + total_n);
    if (score > best_score) {
      best_score = score;
      best = static_cast<int>(i);
    }
  }
  return best;
}

static double softmax_into(const std::vector<double>& z, std::vector<float>* out) {
  double max_z = -std::numeric_limits<double>::infinity();
  for (double v : z) max_z = std::max(max_z, v);
  double sum = 0.0;
  out->assign(z.size(), 0.0f);
  for (size_t i = 0; i < z.size(); ++i) {
    if (!std::isfinite(z[i])) continue;
    const double w = std::exp(z[i] - max_z);
    (*out)[i] = static_cast<float>(w);
    sum += w;
  }
  for (float& v : *out) v = static_cast<float>(v / sum);
  return sum;
}

// kRollout. `root_node` is the expanded root (raw network priors on its
// edges); the caller has already handled the forced move and the tail solver.
static ActionId run_rollout_search(
    const IGameState& root,
    const IGameRules& rules,
    const IStateValueModel& value_model,
    const IPolicyValueEvaluator& evaluator,
    const NetMctsConfig& cfg,
    const Node& root_node,
    std::uint64_t parent_seed,
    std::chrono::steady_clock::time_point t0,
    NetMctsStats* stats) {
  const RolloutConfig& rc = cfg.rollout;
  const int root_player = root.current_player();
  const int np = root.num_players();
  const size_t n_edges = root_node.edges.size();

  std::vector<size_t> candidates(n_edges);
  for (size_t i = 0; i < n_edges; ++i) candidates[i] = i;
  if (rc.max_candidates > 0 && static_cast<size_t>(rc.max_candidates) < n_edges) {
    std::stable_sort(candidates.begin(), candidates.end(), [&](size_t a, size_t b) {
      return root_node.edges[a].prior > root_node.edges[b].prior;
    });
    candidates.resize(static_cast<size_t>(rc.max_candidates));
  }
  const size_t n_cand = candidates.size();

  const bool timed = cfg.time_limit_ms > 0.0;
  const auto deadline = t0 + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double, std::milli>(cfg.time_limit_ms));
  const size_t chunk = static_cast<size_t>(std::max(1, cfg.leaf_batch_size));
  const int worlds_per_wave = std::max(1, static_cast<int>(chunk / n_cand));
  const double inv_temp = 1.0 / static_cast<double>(rc.policy_temperature);

  struct Rollout {
    size_t cand = 0;
    std::unique_ptr<IGameState> state;
    std::unique_ptr<IBeliefTracker> tracker;
    std::mt19937_64 rng;
    int plies_left = 0;
    bool done = false;
    std::vector<float> value;
    std::unique_ptr<IGameState> masked;
    std::vector<ActionId> legal;
    std::vector<float> priors;
  };

  auto step = [&](Rollout& ro, ActionId action) {
    const int actor = ro.state->current_player();
    std::unique_ptr<IGameState> before;
    if (ro.tracker && cfg.events_only_extractor) before = ro.state->clone_state();
    rules.do_action_fast(*ro.state, action, ro.rng);
    if (before) {
      std::vector<PublicEvent> events = cfg.events_only_extractor(
          *before, action, *ro.state, root_player);
      ro.tracker->observe_public_event(
          actor, rules.public_action(*before, action), events);
    }
  };

  std::vector<std::vector<double>> value_sums(n_edges, std::vector<double>(static_cast<size_t>(np), 0.0));
  std::vector<int> counts(n_edges, 0);
  std::int64_t evaluations = 0;
  int rollouts_done = 0;
  int max_plies = 0;

  // Timed: a wave starts only if one more wave as long as the last one still
  // ends before the deadline.
  std::chrono::steady_clock::duration last_wave{};
  for (int r0 = 0; r0 < rc.rollouts_per_action; r0 += worlds_per_wave) {
    const auto wave_t0 = std::chrono::steady_clock::now();
    if (timed && r0 > 0 && wave_t0 + last_wave > deadline) break;
    const int r1 = std::min(rc.rollouts_per_action, r0 + worlds_per_wave);
    std::vector<Rollout> wave;
    wave.reserve(static_cast<size_t>(r1 - r0) * n_cand);
    for (int r = r0; r < r1; ++r) {
      // One sampled world per rollout index, shared by every candidate
      // (common random numbers: candidates differ only by their action).
      std::mt19937_64 world_rng(board_ai::rng::derive_subseed(
          parent_seed, "rollout.world", static_cast<std::uint64_t>(r)));
      std::unique_ptr<IGameState> world = root.clone_state();
      std::unique_ptr<IBeliefTracker> world_tracker;
      if (cfg.root_belief_tracker) {
        world_tracker = cfg.root_belief_tracker->clone();
        world_tracker->randomize_unseen(*world, root_player, world_rng);
      }
      for (size_t c : candidates) {
        Rollout ro;
        ro.cand = c;
        ro.state = world->clone_state();
        if (world_tracker) ro.tracker = world_tracker->clone();
        ro.rng = world_rng;
        ro.plies_left = rc.depth;
        step(ro, root_node.edges[c].action);
        wave.push_back(std::move(ro));
      }
    }

    int plies = 0;
    while (true) {
      std::vector<Rollout*> need_eval;
      for (Rollout& ro : wave) {
        if (ro.done) continue;
        if (ro.state->is_terminal()) {
          ro.value = value_model.terminal_values(*ro.state);
          ro.done = true;
          continue;
        }
        ro.legal = rules.legal_actions(*ro.state);
        if (ro.legal.empty()) {
          throw std::runtime_error(
              "rollout search: legal_actions returned empty for non-terminal state");
        }
        ro.masked = make_masked_state(*ro.state, ro.state->schema_ref(),
                                      ro.state->current_player());
        need_eval.push_back(&ro);
      }
      if (need_eval.empty()) break;
      for (size_t b0 = 0; b0 < need_eval.size(); b0 += chunk) {
        const size_t b1 = std::min(need_eval.size(), b0 + chunk);
        std::vector<IPolicyValueEvaluator::BatchItem> items;
        items.reserve(b1 - b0);
        for (size_t b = b0; b < b1; ++b) {
          Rollout& ro = *need_eval[b];
          IPolicyValueEvaluator::BatchItem item;
          item.masked_state = ro.masked.get();
          item.perspective_player = ro.state->current_player();
          item.tracker = ro.tracker ? ro.tracker.get() : cfg.root_belief_tracker;
          item.legal_actions = &ro.legal;
          item.priors = &ro.priors;
          item.values = &ro.value;
          items.push_back(item);
        }
        evaluator.evaluate_batch(items);
        evaluations += static_cast<std::int64_t>(items.size());
      }
      for (Rollout* rp : need_eval) {
        Rollout& ro = *rp;
        validate_leaf_values(ro.value, np, "rollout evaluation");
        ro.masked.reset();
        if (ro.plies_left <= 0) {
          ro.done = true;
          continue;
        }
        if (ro.priors.size() != ro.legal.size()) {
          throw std::runtime_error(
              "rollout search: evaluator returned " + std::to_string(ro.priors.size()) +
              " priors for " + std::to_string(ro.legal.size()) + " actions");
        }
        std::vector<double> w(ro.priors.size());
        for (size_t i = 0; i < w.size(); ++i) {
          const double p = std::max(0.0, static_cast<double>(ro.priors[i]));
          w[i] = inv_temp == 1.0 ? p : std::pow(p, inv_temp);
        }
        std::discrete_distribution<size_t> pick(w.begin(), w.end());
        const ActionId a = ro.legal[pick(ro.rng)];
        ro.value.clear();
        step(ro, a);
        ro.plies_left -= 1;
      }
      ++plies;
    }
    max_plies = std::max(max_plies, plies);

    for (Rollout& ro : wave) {
      auto& sum = value_sums[ro.cand];
      for (int p = 0; p < np; ++p) {
        sum[static_cast<size_t>(p)] += clip_value(ro.value[static_cast<size_t>(p)], cfg.value_clip);
      }
      counts[ro.cand] += 1;
      ++rollouts_done;
    }
    last_wave = std::chrono::steady_clock::now() - wave_t0;
  }

  // One tabular MMD step over the candidates; non-candidates keep pi' = 0.
  std::vector<double> z(n_edges, -std::numeric_limits<double>::infinity());
  const double denom = 1.0 + static_cast<double>(rc.mmd_alpha) * rc.mmd_eta;
  for (size_t c : candidates) {
    const double q = value_sums[c][static_cast<size_t>(root_player)] / counts[c];
    z[c] = (safe_log(root_node.edges[c].prior) + rc.mmd_eta * q) / denom;
  }
  std::vector<float> target;
  softmax_into(z, &target);

  size_t chosen = candidates.front();
  if (rc.sample_action) {
    std::mt19937_64 pick_rng(board_ai::rng::derive_subseed(parent_seed, "rollout.pick"));
    std::discrete_distribution<size_t> pick(target.begin(), target.end());
    chosen = pick(pick_rng);
  } else {
    for (size_t c : candidates) {
      if (target[c] > target[chosen]) chosen = c;
    }
  }

  if (stats) {
    const double sec = std::max(1e-9, std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count());
    stats->simulations_done = rollouts_done;
    stats->expanded_nodes = 1;
    stats->nodes_per_sec = static_cast<double>(evaluations) / sec;
    stats->max_depth_reached = max_plies;
    stats->network_evaluations = evaluations + 1;
    stats->root_actions.clear();
    stats->root_action_visits.clear();
    stats->root_edge_values.assign(n_edges, std::vector<double>(static_cast<size_t>(np), 0.0));
    for (size_t i = 0; i < n_edges; ++i) {
      stats->root_actions.push_back(root_node.edges[i].action);
      stats->root_action_visits.push_back(counts[i]);
      if (counts[i] > 0) {
        for (int p = 0; p < np; ++p) {
          stats->root_edge_values[i][static_cast<size_t>(p)] =
              value_sums[i][static_cast<size_t>(p)] / counts[i];
        }
      }
    }
    stats->root_values = stats->root_edge_values[chosen];
    stats->best_action_value = stats->root_values[static_cast<size_t>(root_player)];
    stats->root_policy_target = std::move(target);
    stats->selected_action = root_node.edges[chosen].action;
  }
  return root_node.edges[chosen].action;
}

}  // namespace

std::vector<int> gumbel_considered_visit_sequence(int num_considered,
                                                  int num_simulations) {
  std::vector<int> seq;
  if (num_simulations <= 0) return seq;
  seq.reserve(static_cast<size_t>(num_simulations));
  if (num_considered <= 1) {
    for (int i = 0; i < num_simulations; ++i) seq.push_back(i);
    return seq;
  }
  const int log2max = static_cast<int>(std::ceil(std::log2(static_cast<double>(num_considered))));
  std::vector<int> visits(static_cast<size_t>(num_considered), 0);
  int considered = num_considered;
  while (static_cast<int>(seq.size()) < num_simulations) {
    const int extra = std::max(1, num_simulations / (log2max * considered));
    for (int k = 0; k < extra; ++k) {
      seq.insert(seq.end(), visits.begin(), visits.begin() + considered);
      for (int i = 0; i < considered; ++i) visits[static_cast<size_t>(i)] += 1;
    }
    considered = std::max(2, considered / 2);
  }
  seq.resize(static_cast<size_t>(num_simulations));
  return seq;
}

ActionId choose_root_action(
    const NetMctsStats& stats,
    double temperature,
    std::uint64_t rng_seed,
    int top_k,
    ActionId fallback_action) {
  if (stats.selected_action >= 0) return stats.selected_action;
  return select_action_from_visits(stats.root_actions, stats.root_action_visits,
                                   temperature, rng_seed, top_k, fallback_action);
}

NetMcts::NetMcts(NetMctsConfig cfg) : cfg_(cfg) {
  if (cfg_.fpu_reduction &&
      !(std::isfinite(*cfg_.fpu_reduction) && *cfg_.fpu_reduction >= 0.0f)) {
    throw std::invalid_argument(
        "NetMctsConfig.fpu_reduction must be a finite value >= 0, got " +
        std::to_string(*cfg_.fpu_reduction));
  }
  if (!(std::isfinite(cfg_.root_policy_temperature) &&
        cfg_.root_policy_temperature > 0.0f)) {
    throw std::invalid_argument(
        "NetMctsConfig.root_policy_temperature must be a finite value > 0, got " +
        std::to_string(cfg_.root_policy_temperature));
  }
  if (!(std::isfinite(cfg_.root_dirichlet_total_alpha) &&
        cfg_.root_dirichlet_total_alpha >= 0.0f)) {
    throw std::invalid_argument(
        "NetMctsConfig.root_dirichlet_total_alpha must be a finite value >= 0, got " +
        std::to_string(cfg_.root_dirichlet_total_alpha));
  }
  if (cfg_.root_dirichlet_total_alpha > 0.0f && cfg_.root_dirichlet_alpha > 0.0f) {
    throw std::invalid_argument(
        "NetMctsConfig: root_dirichlet_alpha and root_dirichlet_total_alpha are "
        "two ways to size the same noise; set only one of them");
  }
  if (cfg_.root_dirichlet_shaped && cfg_.root_dirichlet_total_alpha <= 0.0f) {
    throw std::invalid_argument(
        "NetMctsConfig: root_dirichlet_shaped requires root_dirichlet_total_alpha > 0");
  }
  if (cfg_.algorithm != SearchAlgorithm::kPuct) {
    const char* name = cfg_.algorithm == SearchAlgorithm::kGumbel ? "gumbel" : "rollout";
    if (cfg_.root_dirichlet_alpha > 0.0f || cfg_.root_dirichlet_total_alpha > 0.0f ||
        cfg_.root_dirichlet_shaped || cfg_.root_dirichlet_epsilon > 0.0f ||
        cfg_.root_policy_temperature != 1.0f) {
      throw std::invalid_argument(
          std::string("NetMctsConfig: root Dirichlet noise and root policy temperature are "
                      "PUCT exploration knobs; algorithm=") + name + " does not use them");
    }
    if (cfg_.cover_root_edges) {
      throw std::invalid_argument(
          std::string("NetMctsConfig: cover_root_edges is PUCT-only (algorithm=") + name + ")");
    }
  }
  if (cfg_.algorithm == SearchAlgorithm::kGumbel) {
    const GumbelConfig& g = cfg_.gumbel;
    if (cfg_.time_limit_ms > 0.0) {
      throw std::invalid_argument(
          "NetMctsConfig: algorithm=gumbel needs a fixed simulation budget "
          "(Sequential Halving schedules it up front); time_limit_ms must be 0");
    }
    if (cfg_.opponent_selection != OpponentSelection::kPuct) {
      throw std::invalid_argument(
          "NetMctsConfig: algorithm=gumbel selects interior edges itself; "
          "opponent_selection must be puct");
    }
    if (g.num_considered < 1 || !(std::isfinite(g.c_visit) && g.c_visit >= 0.0f) ||
        !(std::isfinite(g.c_scale) && g.c_scale > 0.0f)) {
      throw std::invalid_argument(
          "NetMctsConfig.gumbel: need num_considered >= 1, c_visit >= 0, c_scale > 0");
    }
  }
  if (cfg_.algorithm == SearchAlgorithm::kRollout) {
    const RolloutConfig& r = cfg_.rollout;
    if (r.depth < 0 || r.rollouts_per_action < 1 || r.max_candidates < 0 ||
        !(std::isfinite(r.mmd_eta) && r.mmd_eta > 0.0f) ||
        !(std::isfinite(r.mmd_alpha) && r.mmd_alpha >= 0.0f) ||
        !(std::isfinite(r.policy_temperature) && r.policy_temperature > 0.0f)) {
      throw std::invalid_argument(
          "NetMctsConfig.rollout: need depth >= 0, rollouts_per_action >= 1, "
          "max_candidates >= 0, mmd_eta > 0, mmd_alpha >= 0, policy_temperature > 0");
    }
  }
}

ActionId select_action_from_visits(
    const std::vector<ActionId>& actions,
    const std::vector<int>& visits,
    double temperature,
    std::uint64_t rng_seed,
    int top_k,
    ActionId fallback_action) {
  (void)fallback_action;
  if (actions.empty()) {
    throw std::invalid_argument("select_action_from_visits: actions is empty");
  }
  if (actions.size() != visits.size()) {
    throw std::invalid_argument(
        "select_action_from_visits: actions/visits size mismatch (" +
        std::to_string(actions.size()) + " vs " +
        std::to_string(visits.size()) + ")");
  }
  if (temperature <= 1e-6) {
    int best_visit = std::numeric_limits<int>::min();
    std::vector<size_t> best_indices;
    best_indices.reserve(actions.size());
    for (size_t i = 0; i < visits.size(); ++i) {
      if (visits[i] < 0) {
        throw std::invalid_argument(
            "select_action_from_visits: negative visit count for action " +
            std::to_string(actions[i]));
      }
      if (visits[i] > best_visit) {
        best_visit = visits[i];
        best_indices.clear();
        best_indices.push_back(i);
      } else if (visits[i] == best_visit) {
        best_indices.push_back(i);
      }
    }
    if (best_indices.empty() || best_visit <= 0) {
      throw std::runtime_error("select_action_from_visits: no visited action to select");
    }
    if (best_indices.size() == 1) return actions[best_indices.front()];
    SplitMix64Engine rng(rng_seed);
    std::uniform_int_distribution<size_t> pick(0, best_indices.size() - 1);
    return actions[best_indices[pick(rng)]];
  }

  // top_k filtering: only the k actions with the most visits participate
  // in the temperature-weighted sample. 0 = no limit, 1 = argmax (handled
  // above by temperature <= 1e-6 path when caller sets temperature ~0).
  std::vector<bool> in_topk(actions.size(), true);
  if (top_k > 0 && top_k < static_cast<int>(actions.size())) {
    // Find the k-th largest visit count.
    std::vector<int> sorted_visits(visits.begin(), visits.end());
    std::nth_element(sorted_visits.begin(),
                     sorted_visits.begin() + (sorted_visits.size() - static_cast<size_t>(top_k)),
                     sorted_visits.end());
    const int threshold = sorted_visits[sorted_visits.size() - static_cast<size_t>(top_k)];
    int count = 0;
    for (size_t i = 0; i < visits.size(); ++i) {
      if (visits[i] > threshold) {
        count++;
      } else if (visits[i] == threshold && count < top_k) {
        count++;
      } else {
        in_topk[i] = false;
      }
    }
  }

  const double inv_t = 1.0 / std::max(1e-6, temperature);
  std::vector<double> weights(actions.size(), 0.0);
  double sum_w = 0.0;
  for (size_t i = 0; i < visits.size(); ++i) {
    if (visits[i] < 0) {
      throw std::invalid_argument(
          "select_action_from_visits: negative visit count for action " +
          std::to_string(actions[i]));
    }
    if (!in_topk[i]) continue;
    const double base = static_cast<double>(visits[i]);
    const double w = (base > 0.0) ? std::pow(base, inv_t) : 0.0;
    if (!std::isfinite(w)) {
      throw std::runtime_error(
          "select_action_from_visits: non-finite sampling weight for action " +
          std::to_string(actions[i]));
    }
    weights[i] = w;
    sum_w += weights[i];
  }
  if (!std::isfinite(sum_w) || sum_w <= 1e-12) {
    throw std::runtime_error("select_action_from_visits: all visit sampling weights are zero");
  }

  SplitMix64Engine rng(rng_seed);
  std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
  return actions[dist(rng)];
}

// ISMCTS search.
//
// Algorithm (single simulation):
//   1. Clone root. If belief_tracker set, call randomize_unseen to sample
//      a belief-consistent hidden world (root determinization). Descent is
//      now deterministic within this sampled world.
//   2. Descend from root, using UCT2 for edge selection:
//        score = Q + c_puct * prior * sqrt(arrivals) / (1 + edge.visits)
//      where `arrivals` counts the traversals of the edge we came through
//      that ended in the current node (see NetMctsConfig::uct2_arrival_visits;
//      the edge's own visit count when the edge has a single outcome).
//      At root it is root_node.visit_count (equals sim count so far).
//   3. At each node, compute the current state's hash using
//      state.state_hash_for_perspective(state.current_player()). Look up
//      in global `node_index` table. If found, reuse (DAG share). If not,
//      create new node.
//   4. On reaching an unexpanded (leaf) node, call evaluator.evaluate(
//      sim_state, current_player, legal_actions) to get priors + values.
//      Store priors on edges (one per legal action).
//   5. Backup leaf_values up the path: node.visit++, edge.visit++ per step.
//
// Invariants:
//   - DAG is acyclic (guaranteed by state.step_count monotonicity in hash).
//   - Nodes at the same (public, acting-player-private, step) are shared,
//     giving info-set statistics aggregation across sampled worlds.
//   - No chance node machinery: physical randomness resolves at root sampling.
ActionId NetMcts::search_root(
    const IGameState& root,
    const IGameRules& rules,
    const IStateValueModel& value_model,
    const IPolicyValueEvaluator& evaluator,
    NetMctsStats* stats,
    std::uint64_t seed) const {
  const auto legal_root = rules.legal_actions(root);
  if (legal_root.empty()) {
    if (stats) *stats = {};
    return -1;
  }

  const int root_player = root.current_player();
  const std::uint64_t parent_seed = (seed != 0)
      ? seed
      : static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());

  // Per-decision belief-net hook. Run once before the tail solver and any
  // sim — both sample worlds through the tracker, which reads the cached
  // pi posterior. const_cast is safe: the tracker pointer is `const` for
  // the read-only sim path (clone()/randomize_unseen) but
  // `prepare_for_root` is the one intended write site, gated on this
  // caller-owned per-decision flow.
  if (cfg_.root_belief_tracker && cfg_.belief_extractor &&
      cfg_.belief_evaluator) {
    auto* mutable_tracker =
        const_cast<IBeliefTracker*>(cfg_.root_belief_tracker);
    mutable_tracker->prepare_for_root(
        root, root_player, cfg_.belief_extractor, cfg_.belief_evaluator);
  }

  if (cfg_.tail_solve_enabled && cfg_.tail_solver) {
    const IBeliefTracker* tracker = cfg_.root_belief_tracker;
    auto sample_world = [&](std::uint64_t k) {
      std::unique_ptr<IGameState> world = root.clone_state();
      if (tracker) {
        std::mt19937_64 world_rng(board_ai::rng::derive_subseed(
            parent_seed, "mcts.tail_solve.world", k));
        tracker->clone()->randomize_unseen(*world, root_player, world_rng);
      }
      return world;
    };

    auto solve_state = sample_world(0);
    TailSolveResult ts = cfg_.tail_solver->solve(
        *solve_state, rules, value_model, root_player, cfg_.tail_solve_config);
    double solve_elapsed_ms = ts.elapsed_ms;
    bool solve_completed = !ts.budget_exceeded;
    bool proven = ts.outcome == TailSolveOutcome::kProvenWin &&
                  ts.value >= 1.0f && ts.best_action >= 0;

    if (proven && tracker) {
      TailSolveConfig verify_cfg = cfg_.tail_solve_config;
      verify_cfg.depth_limit = cfg_.tail_solve_config.depth_limit - 1;
      for (int k = 1; proven && k < cfg_.tail_solve_config.belief_worlds; ++k) {
        auto world = sample_world(static_cast<std::uint64_t>(k));
        if (!rules.validate_action(*world, ts.best_action)) {
          proven = false;
          break;
        }
        rules.do_action_deterministic(*world, ts.best_action);
        if (world->is_terminal()) {
          proven = value_model.terminal_value_for_player(*world, root_player) >= 1.0f;
          continue;
        }
        const TailSolveResult v = cfg_.tail_solver->solve(
            *world, rules, value_model, root_player, verify_cfg);
        solve_elapsed_ms += v.elapsed_ms;
        solve_completed = solve_completed && !v.budget_exceeded;
        proven = v.outcome == TailSolveOutcome::kProvenWin && v.value >= 1.0f;
      }
    }

    if (stats) {
      stats->tail_solve_attempted = true;
      stats->tail_solve_completed = solve_completed;
      stats->tail_solve_elapsed_ms = solve_elapsed_ms;
    }
    if (proven) {
      if (stats) {
        stats->tail_solved = true;
        stats->tail_solve_outcome = ts.outcome;
        stats->tail_solve_value = ts.value;
        stats->simulations_done = 0;
        stats->root_actions = legal_root;
        stats->root_action_visits.assign(legal_root.size(), 0);
        size_t best_idx = 0;
        for (size_t i = 0; i < legal_root.size(); ++i) {
          if (legal_root[i] == ts.best_action) {
            stats->root_action_visits[i] = 1;
            best_idx = i;
            break;
          }
        }
        stats->best_action_value = static_cast<double>(ts.value);
        // Populate root_values / root_edge_values so downstream consumers
        // (analysis path: _human_wr_from_stats / _human_wr_for_action; web
        // pipeline reading root_values[human_player]) get well-formed data.
        // ProvenWin convention: acting player at root scores +1, opponents
        // share -1 evenly so the vector is zero-sum. Without this the
        // analysis worker crashes with IndexError on empty root_values and
        // the expert-mode web pipeline hangs (BUG: Azul end-game freeze).
        const int np = root.num_players();
        const int actor = root.current_player();
        std::vector<double> rv(static_cast<size_t>(np), 0.0);
        if (np >= 2) {
          const double opp = -1.0 / static_cast<double>(np - 1);
          for (int p = 0; p < np; ++p) {
            rv[static_cast<size_t>(p)] = (p == actor) ? 1.0 : opp;
          }
        }
        stats->root_values = rv;
        stats->root_edge_values.assign(legal_root.size(),
                                       std::vector<double>(static_cast<size_t>(np), 0.0));
        if (best_idx < stats->root_edge_values.size()) {
          stats->root_edge_values[best_idx] = rv;
        }
      }
      return ts.best_action;
    }
  }

  const auto t0 = std::chrono::steady_clock::now();
  const bool timed = cfg_.time_limit_ms > 0.0;
  const auto deadline = t0 + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double, std::milli>(timed ? cfg_.time_limit_ms : 0.0));
  // A timed search's `simulations` is only a ceiling and may be huge.
  const size_t reserve_hint = static_cast<size_t>(std::max(512, 2 * (timed
      ? std::min(cfg_.simulations, 1 << 16) : std::min(cfg_.simulations, 1 << 29))));

  std::vector<Node> nodes;
  nodes.reserve(reserve_hint);

  // Global hash → node_index table: the DAG's canonical lookup. Cleared per
  // search_root call (cross-call sharing is not attempted; the step_count in
  // hash would separate states from different sessions anyway).
  std::unordered_map<StateHash64, int> node_index;
  node_index.reserve(reserve_hint);

  // Per-step masked-state clone: overwrite every hidden slot with
  // kPlaceholder for the acting perspective. ALGORITHM_OVERVIEW §5.2
  // contract — sim descent every step masks once, then both the DAG
  // hash lookup and (if the resulting node is unexpanded) the encoder
  // read from the SAME clone.
  //
  // The schema walker visits the FULL slot set: viz=1 slots route to
  // game's `hash_field_slot` (truth value), viz=0 slots emit
  // `kHiddenHashSentinel`. Hashing the live state and hashing the
  // masked clone produce the same digest — viz=0 slots in the clone
  // hold kPlaceholder, but the walker's visible-flag check fires
  // sentinel mode for them either way (BUG-037 postmortem).
  auto materialize_masked = [](const IGameState& s) -> std::unique_ptr<IGameState> {
    return make_masked_state(s, s.schema_ref(), s.current_player());
  };

  auto hash_masked = [](const IGameState& m) -> StateHash64 {
    return m.state_hash_for_perspective(m.current_player());
  };

  auto root_masked = materialize_masked(root);
  const StateHash64 root_hash = hash_masked(*root_masked);
  nodes.push_back(Node{root_player, false, 0, 0.0f, {}});
  node_index[root_hash] = 0;

  // Leaf expansion. Caller passes the live `state` (for legal_actions
  // and terminal_values, which need ground truth), a masked clone
  // already built for the acting player (for the encoder), and the
  // tracker for the same perspective (or null when no tracker is
  // registered for the game).
  auto install_priors = [&](Node& node, const std::vector<ActionId>& legal,
                            std::vector<float>& priors) {
    if (priors.size() != legal.size()) {
      throw std::runtime_error("MCTS: evaluator returned " + std::to_string(priors.size()) +
          " priors but " + std::to_string(legal.size()) + " actions");
    }

    float sum = 0.0f;
    for (size_t i = 0; i < priors.size(); ++i) {
      const float p = priors[i];
      if (!std::isfinite(p)) {
        throw std::runtime_error(
            "MCTS: evaluator returned non-finite prior for action " +
            std::to_string(legal[i]));
      }
      if (p < 0.0f) {
        throw std::runtime_error(
            "MCTS: evaluator returned negative prior for action " +
            std::to_string(legal[i]));
      }
      sum += p;
    }
    if (!std::isfinite(sum) || sum <= 1e-8f) {
      throw std::runtime_error("MCTS: evaluator returned zero prior mass over legal actions");
    }
    for (float& p : priors) {
      p /= sum;
    }

    node.edges.clear();
    node.edges.reserve(legal.size());
    for (size_t i = 0; i < legal.size(); ++i) {
      Edge edge{};
      edge.action = legal[i];
      edge.prior = priors[i];
      node.edges.push_back(std::move(edge));
    }
    node.expanded = true;
  };

  auto expand_node = [&](Node& node, const IGameState& state,
                         const IGameState& masked,
                         const IBeliefTracker* tracker) -> std::vector<float> {
    const auto legal = rules.legal_actions(state);
    if (legal.empty()) {
      if (!state.is_terminal()) {
        throw std::runtime_error(
            "MCTS: legal_actions returned empty for non-terminal state");
      }
      node.expanded = true;
      node.edges.clear();
      return value_model.terminal_values(state);
    }

    std::vector<float> priors;
    std::vector<float> values;
    const bool ok = evaluator.evaluate(masked, node.to_play, tracker, legal, &priors, &values);
    if (!ok) {
      throw std::runtime_error("MCTS: evaluator.evaluate() failed — model not loaded or inference error");
    }
    install_priors(node, legal, priors);
    return values;
  };

  // Root expansion uses the session-shared tracker directly — no sim is
  // active yet, so there's nothing to clone.
  const std::vector<float> root_net_values =
      expand_node(nodes[0], root, *root_masked, cfg_.root_belief_tracker);
  if (!nodes[0].edges.empty()) {
    validate_leaf_values(root_net_values, root.num_players(), "root evaluation");
    nodes[0].raw_value = clip_value(
        root_net_values[static_cast<size_t>(root_player)], cfg_.value_clip);
  }
  std::int64_t network_evaluations = 1;

  // The evaluator's own prior, kept for NetMctsStats::root_priors before the
  // root temperature and noise below reshape the edges.
  std::vector<float> raw_root_priors;
  if (stats) {
    raw_root_priors.reserve(nodes[0].edges.size());
    for (const Edge& e : nodes[0].edges) raw_root_priors.push_back(e.prior);
  }

  // Forced move: with exactly one legal action every simulation would pass
  // through the same root edge, so the visit distribution is one-hot on it
  // whatever the budget. Return it without simulating. The belief hook and
  // the tail solver above ran as usual (their stats are unchanged); the value
  // read-out is the network's evaluation of the root, the only estimate
  // available without search. Analysis mode (cover_root_edges) keeps
  // searching: its purpose is the searched value, not the action.
  if (nodes[0].edges.size() == 1 && !cfg_.cover_root_edges) {
    const int n_players = root.num_players();
    validate_leaf_values(root_net_values, n_players, "root evaluation");
    if (stats) {
      std::vector<double> rv(static_cast<size_t>(n_players), 0.0);
      for (int p = 0; p < n_players; ++p) {
        rv[static_cast<size_t>(p)] = static_cast<double>(clip_value(
            root_net_values[static_cast<size_t>(p)], cfg_.value_clip));
      }
      stats->simulations_done = 0;
      stats->expanded_nodes = 1;
      stats->root_actions = {nodes[0].edges[0].action};
      stats->root_action_visits = {1};
      stats->root_priors = raw_root_priors;
      stats->best_action_value = rv[static_cast<size_t>(root_player)];
      stats->root_values = rv;
      stats->root_edge_values = {rv};
      stats->network_evaluations = 1;
    }
    return nodes[0].edges[0].action;
  }

  if (cfg_.algorithm == SearchAlgorithm::kRollout) {
    if (stats) stats->root_priors = raw_root_priors;
    return run_rollout_search(root, rules, value_model, evaluator, cfg_, nodes[0],
                              parent_seed, t0, stats);
  }

  const std::uint64_t dirichlet_seed = board_ai::rng::derive_subseed(
      parent_seed, "mcts.dirichlet",
      static_cast<std::uint64_t>(root.current_player()));
  apply_root_policy_temperature(nodes[0], cfg_.root_policy_temperature);
  apply_root_dirichlet_noise(
      nodes[0],
      cfg_.root_dirichlet_alpha,
      cfg_.root_dirichlet_total_alpha,
      cfg_.root_dirichlet_shaped,
      cfg_.root_dirichlet_epsilon,
      dirichlet_seed);

  const int np = root.num_players();
  std::vector<std::vector<double>> root_edge_values(
      nodes[0].edges.size(), std::vector<double>(static_cast<size_t>(np), 0.0));

  const int simulations = std::max(1, cfg_.simulations);

  // Gumbel root state: logits of the raw prior, the Gumbel draw per root
  // action, and the Sequential Halving schedule (one entry per simulation).
  const bool gumbel = cfg_.algorithm == SearchAlgorithm::kGumbel;
  std::vector<double> gumbel_logits;
  std::vector<double> gumbel_noise;
  std::vector<int> gumbel_schedule;
  std::vector<float> gumbel_scratch;
  if (gumbel) {
    const size_t n_root = nodes[0].edges.size();
    gumbel_logits.resize(n_root);
    gumbel_noise.assign(n_root, 0.0);
    for (size_t i = 0; i < n_root; ++i) {
      gumbel_logits[i] = safe_log(nodes[0].edges[i].prior);
    }
    if (cfg_.gumbel.noise) {
      SplitMix64Engine grng(board_ai::rng::derive_subseed(parent_seed, "mcts.gumbel"));
      std::uniform_real_distribution<double> u01(0.0, 1.0);
      for (size_t i = 0; i < n_root; ++i) {
        const double u = std::max(u01(grng), 1e-300);
        gumbel_noise[i] = -std::log(-std::log(u));
      }
    }
    const int m = std::min(cfg_.gumbel.num_considered, static_cast<int>(n_root));
    gumbel_schedule = gumbel_considered_visit_sequence(m, simulations);
  }
  // Root action of simulation `sim_index`: among the root edges whose visit
  // count (in-flight included) equals the schedule entry, the best
  // gumbel + logit + sigma(completedQ). The first m simulations visit the
  // Gumbel-Top-m; later entries implicitly halve that set.
  auto gumbel_root_edge = [&](int sim_index) -> int {
    const int target = gumbel_schedule[static_cast<size_t>(sim_index)];
    gumbel_sigma_q(nodes[0], cfg_.gumbel, cfg_.value_clip, true, &gumbel_scratch);
    int best = -1;
    double best_score = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < nodes[0].edges.size(); ++i) {
      const Edge& e = nodes[0].edges[i];
      if (e.visit_count + e.virtual_loss != target) continue;
      const double score = gumbel_noise[i] + gumbel_logits[i] + gumbel_scratch[i];
      if (score > best_score) {
        best_score = score;
        best = static_cast<int>(i);
      }
    }
    if (best < 0) {
      throw std::runtime_error(
          "MCTS gumbel: no root action has the scheduled visit count " +
          std::to_string(target) + " at simulation " + std::to_string(sim_index));
    }
    return best;
  };

  // ISMCTS K × M split (see NetMctsConfig::world_cap):
  //   K = number of distinct sampled worlds (outer loop)
  //   M = sims per world (inner loop, all sharing the same world_seed)
  // world_cap == 0 → legacy K=simulations, M=1 (every sim its own world).
  // world_cap > 0 → K=clamp(world_cap, 1, simulations), M=ceil(sims/K).
  // Total sims still hard-capped at `simulations`; the last world may
  // run fewer than M iterations.
  const int K = (cfg_.world_cap > 0)
      ? std::clamp(cfg_.world_cap, 1, simulations)
      : simulations;
  const int M = (simulations + K - 1) / K;

  std::int64_t dag_reuse_hits = 0;
  std::int64_t depth_out_hits = 0;
  int max_depth_reached_local = 0;

  // Per-(edge, outcome) counts, updated in lockstep with the edge's own
  // visit_count / virtual_loss in backup / add_virtual_loss below. Only
  // maintained when selection reads them.
  std::unordered_map<ArrivalKey, ArrivalCount, ArrivalKeyHash> arrivals;
  if (cfg_.uct2_arrival_visits) {
    arrivals.reserve(reserve_hint);
  }

  // Backup. Standard path-walk; in a DAG each node's visit_count tracks
  // total visits across all parent paths.
  auto backup = [&](const std::vector<int>& path_nodes,
                    const std::vector<int>& path_edges,
                    const std::vector<float>& leaf_values) {
    for (int i = static_cast<int>(path_nodes.size()) - 1; i >= 0; --i) {
      const int node_idx = path_nodes[static_cast<size_t>(i)];
      Node& n = nodes[node_idx];
      const size_t tp = static_cast<size_t>(n.to_play);
      const float v = clip_value(leaf_values[tp], cfg_.value_clip);
      n.visit_count += 1;
      n.value_sum += v;
      if (i > 0) {
        const int parent_idx = path_nodes[static_cast<size_t>(i - 1)];
        const int parent_edge_idx = path_edges[static_cast<size_t>(i - 1)];
        const size_t parent_tp = static_cast<size_t>(nodes[parent_idx].to_play);
        const float pv = clip_value(leaf_values[parent_tp], cfg_.value_clip);
        Edge& parent_edge = nodes[parent_idx].edges[parent_edge_idx];
        parent_edge.visit_count += 1;
        parent_edge.value_sum += pv;
        if (cfg_.uct2_arrival_visits) {
          arrivals[ArrivalKey{parent_idx, parent_edge_idx, node_idx}]
              .visit_count += 1;
        }
        if (parent_idx == 0) {
          auto& rev = root_edge_values[static_cast<size_t>(parent_edge_idx)];
          for (int p = 0; p < np; ++p) {
            const double cv = static_cast<double>(
                clip_value(leaf_values[static_cast<size_t>(p)], cfg_.value_clip));
            rev[static_cast<size_t>(p)] += cv;
          }
        }
      }
    }
  };

  auto add_virtual_loss = [&](const std::vector<int>& path_nodes,
                              const std::vector<int>& path_edges, int delta) {
    for (size_t i = 0; i < path_nodes.size(); ++i) {
      Node& n = nodes[static_cast<size_t>(path_nodes[i])];
      n.virtual_loss += delta;
      if (i < path_edges.size()) {
        n.edges[static_cast<size_t>(path_edges[i])].virtual_loss += delta;
        if (cfg_.uct2_arrival_visits) {
          arrivals[ArrivalKey{path_nodes[i], path_edges[i], path_nodes[i + 1]}]
              .virtual_loss += delta;
        }
      }
    }
  };

  // Leaves waiting for evaluation. Descents that reach a leaf already in
  // flight attach their path to it instead of evaluating it twice.
  struct PendingLeaf {
    int node_idx = -1;
    std::vector<std::pair<std::vector<int>, std::vector<int>>> paths;
    std::unique_ptr<IGameState> masked;
    std::unique_ptr<IBeliefTracker> sim_tracker;
    const IBeliefTracker* encoder_tracker = nullptr;
    std::vector<ActionId> legal;
    std::vector<float> priors;
    std::vector<float> values;
  };
  const int leaf_batch = std::max(1, cfg_.leaf_batch_size);
  std::vector<PendingLeaf> pending;
  std::unordered_map<int, size_t> pending_by_node;
  int inflight_sims = 0;

  auto flush_pending = [&]() {
    if (pending.empty()) return;
    std::vector<IPolicyValueEvaluator::BatchItem> items;
    items.reserve(pending.size());
    for (auto& p : pending) {
      IPolicyValueEvaluator::BatchItem item;
      item.masked_state = p.masked.get();
      item.perspective_player = nodes[static_cast<size_t>(p.node_idx)].to_play;
      item.tracker = p.encoder_tracker;
      item.legal_actions = &p.legal;
      item.priors = &p.priors;
      item.values = &p.values;
      items.push_back(item);
    }
    evaluator.evaluate_batch(items);
    network_evaluations += static_cast<std::int64_t>(pending.size());
    for (auto& p : pending) {
      Node& leaf_node = nodes[static_cast<size_t>(p.node_idx)];
      install_priors(leaf_node, p.legal, p.priors);
      validate_leaf_values(p.values, np, "leaf evaluation");
      leaf_node.raw_value = clip_value(
          p.values[static_cast<size_t>(leaf_node.to_play)], cfg_.value_clip);
      for (const auto& [pn, pe] : p.paths) {
        add_virtual_loss(pn, pe, -1);
        backup(pn, pe, p.values);
      }
    }
    pending.clear();
    pending_by_node.clear();
    inflight_sims = 0;
  };

  // Gumbel: a Sequential Halving phase (constant number of surviving actions,
  // i.e. runs of equal schedule entries of one length) visits every survivor
  // equally whatever the order, so its leaves can be evaluated in batches of
  // up to leaf_batch_size; the halving at its end must see all of them backed up.
  std::vector<char> gumbel_phase_end;
  if (gumbel) {
    const size_t n = gumbel_schedule.size();
    std::vector<int> run_len(n, 0);
    for (size_t i = 0; i < n;) {
      size_t j = i;
      while (j < n && gumbel_schedule[j] == gumbel_schedule[i]) ++j;
      for (size_t k = i; k < j; ++k) run_len[k] = static_cast<int>(j - i);
      i = j;
    }
    gumbel_phase_end.assign(n, 0);
    for (size_t i = 0; i < n; ++i) {
      gumbel_phase_end[i] = i + 1 == n ||
          (gumbel_schedule[i + 1] != gumbel_schedule[i] && run_len[i + 1] != run_len[i]);
    }
  }
  auto gumbel_round_ends = [&](int sim_index) {
    return gumbel && gumbel_phase_end[static_cast<size_t>(sim_index)] != 0;
  };

  int sim = 0;
  auto out_of_time = [&]() {
    return timed && sim > 0 && std::chrono::steady_clock::now() >= deadline;
  };
  // Untimed: K passes of M sims, each pass its own world. Timed: one sim
  // per pass, cycling through the K worlds (kept alive between visits when
  // world_cap bounds them; with world_cap == 0 every sim is its own world
  // anyway, so nothing is kept).
  struct WorldTemplate {
    std::mt19937_64 rng;
    std::unique_ptr<IGameState> state;
    std::unique_ptr<IBeliefTracker> tracker;
  };
  const bool keep_worlds = timed && cfg_.world_cap > 0;
  std::vector<WorldTemplate> kept_worlds;
  if (keep_worlds) kept_worlds.reserve(static_cast<size_t>(K));
  WorldTemplate fresh_world;
  const int passes = timed ? simulations : K;
  const int sims_per_pass = timed ? 1 : M;
  for (int pass = 0; pass < passes && sim < simulations && !out_of_time(); ++pass) {
    const int wk = keep_worlds ? pass % K : pass;
    WorldTemplate* world = nullptr;
    if (keep_worlds && wk < static_cast<int>(kept_worlds.size())) {
      world = &kept_worlds[static_cast<size_t>(wk)];
    } else {
      // Per-world subseed: every sim inside this world starts descent from
      // the same RNG state so randomize_unseen's sampled hidden world is
      // shared byte-equal across the M inner iterations. Divergence between
      // inner sims comes only from the bandit picking different edges as
      // edge visit counts grow.
      const std::uint64_t world_seed = board_ai::rng::derive_subseed(
          parent_seed, "mcts.world", static_cast<std::uint64_t>(wk));

      // Run randomize_unseen ONCE per world (not per inner sim). For
      // belief-aware games this can fire ONNX forward passes (belief-net
      // posterior cached in the tracker, Plackett-Luce sampling, etc.) —
      // amortizing it from K×M to K is the whole point of world_cap.
      // The world_state / world_tracker
      // produced here serve as templates for every inner sim, which clone
      // them fresh before descent. world_rng's state after randomize is
      // captured and copied into each inner's sim_rng so descent's
      // do_action_fast draws stay bit-exact identical to the legacy
      // per-sim re-randomize behavior.
      WorldTemplate made;
      made.rng.seed(world_seed);
      made.state = root.clone_state();
      if (cfg_.root_belief_tracker != nullptr) {
        made.tracker = cfg_.root_belief_tracker->clone();
        made.tracker->randomize_unseen(*made.state, root.current_player(),
                                       made.rng);
      }
      if (keep_worlds) {
        kept_worlds.push_back(std::move(made));
        world = &kept_worlds.back();
      } else {
        fresh_world = std::move(made);
        world = &fresh_world;
      }
    }
    const std::mt19937_64& world_rng = world->rng;
    const std::unique_ptr<IGameState>& world_state = world->state;
    const std::unique_ptr<IBeliefTracker>& world_tracker = world->tracker;

    for (int wm = 0; wm < sims_per_pass && sim < simulations; ++wm, ++sim) {
    std::unique_ptr<IGameState> sim_state = world_state->clone_state();

    // Single per-sim RNG drives descent's do_action_fast draws. Initialized
    // from a copy of `world_rng` post-randomize so descent sees the same
    // RNG state the legacy code did (legacy: re-seed + re-run randomize per
    // sim; new: re-use the post-randomize state captured once per world).
    std::mt19937_64 sim_rng(world_rng);

    // Clone tracker so descent-time observe_public_event (below, after
    // each do_action_fast) doesn't pollute the world-shared template.
    // The clone is a sim-local stack variable; it dies with the sim and
    // the next sim re-clones from the same world template.
    std::unique_ptr<IBeliefTracker> sim_tracker;
    if (world_tracker) {
      sim_tracker = world_tracker->clone();
    }
    auto& sim_step_rng = sim_rng;

    // Path records for backup. For UCT2 we also track which edge we came
    // through INTO each node on the path; the sqrt() in UCB uses that edge's
    // visit_count, not the node's global visit_count (which in a DAG mixes
    // visits from multiple incoming paths).
    std::vector<int> path_nodes;
    std::vector<int> path_edges;     // edge index within path_nodes[i]
    path_nodes.reserve(static_cast<size_t>(cfg_.max_depth + 2));
    path_edges.reserve(static_cast<size_t>(cfg_.max_depth + 2));

    int cur_idx = 0;
    path_nodes.push_back(cur_idx);

    std::vector<float> leaf_values;
    bool leaf_pending = false;
    int depth = 0;

    // At root, there's no incoming edge; use root's node visit_count as the
    // sqrt argument (equals sim count so far, in-flight sims included).
    int incoming_edge_visits = nodes[0].visit_count + nodes[0].virtual_loss;

    // Masked clone of the current sim_state. §5.2: one mask per step,
    // shared by the DAG hash lookup and (if the node turns out
    // unexpanded) the encoder. Materialized lazily because the very
    // first iteration of each sim sits on the root, which was already
    // hashed + expanded above before this loop began.
    std::unique_ptr<IGameState> step_masked;

    while (depth < cfg_.max_depth) {
      if (sim_state->is_terminal()) {
        leaf_values = value_model.terminal_values(*sim_state);
        break;
      }
      if (!nodes[cur_idx].expanded) {
        auto pit = pending_by_node.find(cur_idx);
        if (pit != pending_by_node.end()) {
          // In Gumbel with a large leaf_batch, hitting an already-pending leaf
          // repeatedly wastes simulations and duplicates its single value
          // backup. Flush once the pending batch is big enough to amortize GPU
          // evaluation, then continue descending through the newly expanded node.
          const int min_batch = std::min(leaf_batch, 32);
          const int min_coll_batch = std::min(leaf_batch, 8);
          if (gumbel &&
              (static_cast<int>(pending.size()) >= min_batch ||
               (static_cast<int>(pending.size()) >= min_coll_batch &&
                inflight_sims >= static_cast<int>(pending.size()) * 2))) {
            flush_pending();
            continue;
          }
          pending[pit->second].paths.emplace_back(path_nodes, path_edges);
        } else {
          if (!step_masked) step_masked = materialize_masked(*sim_state);
          // Sim-local tracker (clone of root + descent-time
          // observe_public_event updates) is what the encoder reads, so it
          // sees tracker state matching the sim's actual depth. Falls back
          // to the session-shared tracker if the game registers none.
          PendingLeaf leaf;
          leaf.node_idx = cur_idx;
          leaf.paths.emplace_back(path_nodes, path_edges);
          leaf.legal = rules.legal_actions(*sim_state);
          if (leaf.legal.empty()) {
            throw std::runtime_error(
                "MCTS: legal_actions returned empty for non-terminal state");
          }
          leaf.encoder_tracker = sim_tracker
              ? sim_tracker.get() : cfg_.root_belief_tracker;
          leaf.masked = std::move(step_masked);
          leaf.sim_tracker = std::move(sim_tracker);
          pending_by_node.emplace(cur_idx, pending.size());
          pending.push_back(std::move(leaf));
        }
        add_virtual_loss(path_nodes, path_edges, +1);
        ++inflight_sims;
        leaf_pending = true;
        break;
      }
      if (nodes[cur_idx].edges.empty()) {
        throw std::runtime_error(
            "MCTS: expanded non-terminal node has no legal edges");
      }

      // UCT2 edge selection. sqrt() argument is how often the edge we just
      // came through led here (incoming_edge_visits), NOT the node's global
      // visit_count. Avoids over-exploration bias from DAG's multiple
      // parents (Childs et al. 2008).
      int best_edge = -1;

      // Root-edge coverage (analysis-pipeline only): before PUCT takes over,
      // visit any unvisited root edge once. Same world / same descent / same
      // backup as a normal sim — the only difference is which edge we pick
      // at root. Guarantees action_values is dense over the full legal set.
      if (cfg_.cover_root_edges && cur_idx == 0) {
        for (int ei = 0; ei < static_cast<int>(nodes[cur_idx].edges.size()); ++ei) {
          const Edge& ce = nodes[cur_idx].edges[ei];
          if (ce.visit_count + ce.virtual_loss == 0) {
            best_edge = ei;
            break;
          }
        }
      }

      if (gumbel) {
        best_edge = cur_idx == 0
            ? gumbel_root_edge(sim)
            : gumbel_interior_edge(nodes[cur_idx], cfg_.gumbel, cfg_.value_clip, gumbel_scratch);
      }

      if (best_edge < 0) {
        // Frozen-prior on opponent nodes (Smooth-UCT-style). Root always
        // PUCT regardless. See OpponentSelection comment in net_mcts.h.
        const bool use_frozen_prior =
            cfg_.opponent_selection == OpponentSelection::kFrozenPrior &&
            cur_idx != 0 &&
            nodes[cur_idx].to_play != root_player;
        if (use_frozen_prior) {
          best_edge = sample_index_by_prior(nodes[cur_idx].edges, sim_rng);
        } else {
          const float sqrt_parent = std::sqrt(
              static_cast<float>(std::max(1, incoming_edge_visits)));
          float unvisited_q = 0.0f;
          if (cfg_.fpu_reduction) {
            const Node& node = nodes[cur_idx];
            float visited_prior = 0.0f;
            for (const Edge& e : node.edges) {
              if (e.visit_count + e.virtual_loss > 0) visited_prior += e.prior;
            }
            const float node_q = node.visit_count > 0
                ? node.value_sum / static_cast<float>(node.visit_count) : 0.0f;
            unvisited_q = node_q - *cfg_.fpu_reduction * std::sqrt(visited_prior);
          }
          float best_score = -std::numeric_limits<float>::infinity();
          for (int ei = 0; ei < static_cast<int>(nodes[cur_idx].edges.size()); ++ei) {
            const Edge& e = nodes[cur_idx].edges[ei];
            const int n_eff = e.visit_count + e.virtual_loss;
            float q = unvisited_q;
            if (n_eff > 0) {
              q = (e.value_sum - static_cast<float>(e.virtual_loss)) /
                  static_cast<float>(n_eff);
            }
            const float u = cfg_.c_puct * e.prior * sqrt_parent /
                            (1.0f + static_cast<float>(n_eff));
            const float score = q + u;
            if (score > best_score) {
              best_score = score;
              best_edge = ei;
            }
          }
        }
      }
      if (best_edge < 0) {
        throw std::runtime_error("MCTS: failed to select an edge from expanded node");
      }

      const ActionId chosen_action = nodes[cur_idx].edges[best_edge].action;
      // Hash scope must fully determine the legal action set at an expanded
      // DAG node. If a reused node offers an action illegal in the current
      // sampled world, the game hash/encoder scope is broken and must fail
      // loudly rather than silently truncating the simulation.
      if (cfg_.safe_mode) {
        if (!rules.validate_action(*sim_state, chosen_action)) {
          auto current_legal = rules.legal_actions(*sim_state);
          std::string edges_str;
          for (const auto& e : nodes[cur_idx].edges) {
            if (!edges_str.empty()) edges_str += ",";
            edges_str += std::to_string(e.action);
          }
          std::string legal_str;
          for (auto a : current_legal) {
            if (!legal_str.empty()) legal_str += ",";
            legal_str += std::to_string(a);
          }
          // Dump state_hash to correlate with independent debugging.
          // Error path — ok to mask one more time for the dump.
          const auto h_pub = hash_masked(*materialize_masked(*sim_state));
          const auto h_full = sim_state->state_hash();
          throw std::runtime_error(
              "MCTS: DAG node legal-action mismatch; selected action " +
              std::to_string(chosen_action) + " is not legal in current state. "
              "node_edges=[" + edges_str + "] (size=" +
              std::to_string(nodes[cur_idx].edges.size()) + ") "
              "current_legal=[" + legal_str + "] (size=" +
              std::to_string(current_legal.size()) + ") "
              "current_player=" + std::to_string(sim_state->current_player()) +
              " node_to_play=" + std::to_string(nodes[cur_idx].to_play) +
              " step_count=" + std::to_string(sim_state->step_count()) +
              " depth=" + std::to_string(depth) +
              " hash_pub=" + std::to_string(h_pub) +
              " hash_full=" + std::to_string(h_full));
        }
      }
      const ActionId final_action = nodes[cur_idx].edges[best_edge].action;

      // Snapshot the actor + state-before for sim-local tracker
      // maintenance — the events_only_extractor diff requires both.
      // Skipped when the game registers no extractor (fully-public
      // games) or no tracker — short-circuit before the clone so the
      // perf cost is exactly zero on those paths.
      const int actor_for_event = sim_state->current_player();
      std::unique_ptr<IGameState> state_before;
      if (sim_tracker && cfg_.events_only_extractor) {
        state_before = sim_state->clone_state();
      }

      rules.do_action_fast(*sim_state, final_action, sim_step_rng);

      // Sim-local tracker maintenance: feed the cloned sim_tracker the
      // events from this descent step so the next iteration's encoder
      // (and any subsequent observe_public_event) sees up-to-date
      // public-derived state. Perspective is the root acting player —
      // tracker content is perspective-agnostic by contract
      // (test_tracker_perspective_invariance). This sim_tracker is
      // discarded when the sim ends; cfg_.root_belief_tracker is never
      // written.
      //
      // Uses events_only_extractor (not the full PublicEventExtractor)
      // — the sim path doesn't need the wholesale public_snapshot half
      // of the wire protocol; sim_state.viz is already maintained by
      // do_action_fast and node-player masking happens via
      // make_masked_state directly.
      if (state_before) {
        std::vector<PublicEvent> events = cfg_.events_only_extractor(
            *state_before, final_action, *sim_state, root_player);
        sim_tracker->observe_public_event(
            actor_for_event, rules.public_action(*state_before, final_action), events);
      }

      if (cfg_.on_sim_step) {
        cfg_.on_sim_step(sim, depth, *sim_state, sim_tracker.get());
      }

      // After do_action: hash the live sim_state to look up the DAG.
      // state_hash_for_perspective on the live state is bit-equal to
      // hashing the masked clone (viz=0 slots use kHiddenHashSentinel in
      // both paths). This lets us skip the expensive materialize_masked
      // (clone + mask) on DAG hits — only materialize on cache miss when
      // the encoder actually needs the masked clone for expansion.
      const StateHash64 next_hash =
          sim_state->state_hash_for_perspective(sim_state->current_player());
      int next_idx = -1;
      auto it = node_index.find(next_hash);
      if (it != node_index.end()) {
        next_idx = it->second;
        ++dag_reuse_hits;
        // DAG hit — skip materialize_masked entirely.
        step_masked.reset();
      } else {
        // Cache miss — materialize for the encoder in the next iteration's
        // expand_node call.
        step_masked = materialize_masked(*sim_state);
        nodes.push_back(Node{sim_state->current_player(), false, 0, 0.0f, {}});
        next_idx = static_cast<int>(nodes.size()) - 1;
        node_index[next_hash] = next_idx;
      }

      // Update this edge's `child` field to the resolved target. Since the
      // DAG may have this edge point to different children on different
      // paths (e.g. if the rules-apply leads to different worlds), we always
      // overwrite — but for correctness we expect the same (edge, sim world)
      // to produce the same hash, so this is typically stable.
      nodes[cur_idx].edges[best_edge].child = next_idx;

      path_edges.push_back(best_edge);
      const int parent_idx = cur_idx;
      cur_idx = next_idx;
      path_nodes.push_back(cur_idx);
      // For next iteration's UCT2: the "incoming edge" is the one we just
      // traversed. Count only its traversals that arrived at this node
      // (in-flight ones included, like every other count used by selection);
      // the edge's own count also covers its other outcomes.
      if (cfg_.uct2_arrival_visits) {
        const auto ait =
            arrivals.find(ArrivalKey{parent_idx, best_edge, cur_idx});
        incoming_edge_visits = ait == arrivals.end()
            ? 0 : ait->second.visit_count + ait->second.virtual_loss;
      } else {
        const Edge& traversed = nodes[parent_idx].edges[best_edge];
        incoming_edge_visits = traversed.visit_count + traversed.virtual_loss;
      }
      if (cfg_.on_descent_edge) {
        cfg_.on_descent_edge(sim, depth, parent_idx, best_edge, cur_idx,
                             incoming_edge_visits);
      }
      depth += 1;
    }
    if (depth > max_depth_reached_local) max_depth_reached_local = depth;

    if (leaf_pending) {
      if (inflight_sims >= leaf_batch || gumbel_round_ends(sim)) flush_pending();
      continue;
    }

    // Depth-out fallback: sim ran out of max_depth without hitting a
    // terminal or unexpanded leaf — the descent looped through expanded
    // nodes for the full budget. Treat as a draw-equivalent leaf (zero
    // value to every player) and back up. The state is policy-collapse
    // pathology (e.g. Coup mutual block-deadlock) that the player will
    // need to learn to break, not an evaluator failure — silently
    // throwing here masks the diagnostic. See KNOWN_ISSUES.
    if (leaf_values.empty()) {
      leaf_values.assign(static_cast<size_t>(np), 0.0f);
      ++depth_out_hits;
    }
    validate_leaf_values(leaf_values, np, "leaf evaluation");
    backup(path_nodes, path_edges, leaf_values);
    if (gumbel_round_ends(sim)) flush_pending();
    }  // inner M loop (sims sharing this world)
  }  // outer K loop (worlds)
  flush_pending();

  const Node& root_node = nodes[0];
  // Tie-break random selection when multiple edges share max visit count.
  int best_edge = 0;
  int best_visit = -1;
  std::vector<int> tied_edges;
  for (int ei = 0; ei < static_cast<int>(root_node.edges.size()); ++ei) {
    const int vc = root_node.edges[ei].visit_count;
    if (vc > best_visit) {
      best_visit = vc;
      tied_edges.clear();
      tied_edges.push_back(ei);
    } else if (vc == best_visit) {
      tied_edges.push_back(ei);
    }
  }
  if (tied_edges.size() == 1) {
    best_edge = tied_edges.front();
  } else if (!tied_edges.empty()) {
    SplitMix64Engine tiebreak_rng(
        board_ai::rng::derive_subseed(parent_seed, "mcts.tiebreak"));
    std::uniform_int_distribution<size_t> pick(0, tied_edges.size() - 1);
    best_edge = tied_edges[pick(tiebreak_rng)];
  }

  // Gumbel decision: among the most-visited root actions (the survivors of
  // Sequential Halving) the best gumbel + logit + sigma(completedQ); the
  // training target is softmax(logit + sigma(completedQ)) over every root action.
  std::vector<float> gumbel_target;
  if (gumbel) {
    gumbel_sigma_q(root_node, cfg_.gumbel, cfg_.value_clip, false, &gumbel_scratch);
    double best_score = -std::numeric_limits<double>::infinity();
    std::vector<double> z(root_node.edges.size());
    for (size_t i = 0; i < root_node.edges.size(); ++i) {
      z[i] = gumbel_logits[i] + gumbel_scratch[i];
      if (root_node.edges[i].visit_count != best_visit) continue;
      const double score = gumbel_noise[i] + z[i];
      if (score > best_score) {
        best_score = score;
        best_edge = static_cast<int>(i);
      }
    }
    softmax_into(z, &gumbel_target);
  }

  if (stats) {
    stats->network_evaluations = network_evaluations;
    if (gumbel) {
      stats->selected_action = root_node.edges[best_edge].action;
      stats->root_policy_target = std::move(gumbel_target);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::max(1e-9, std::chrono::duration<double>(t1 - t0).count());
    stats->simulations_done = sim;
    stats->expanded_nodes = static_cast<std::int64_t>(nodes.size());
    stats->nodes_per_sec = static_cast<double>(nodes.size()) / sec;
    stats->dag_reuse_hits = dag_reuse_hits;
    stats->depth_out_hits = depth_out_hits;
    stats->max_depth_reached = max_depth_reached_local;
    stats->root_actions.clear();
    stats->root_action_visits.clear();
    stats->root_actions.reserve(root_node.edges.size());
    stats->root_action_visits.reserve(root_node.edges.size());
    for (const Edge& e : root_node.edges) {
      stats->root_actions.push_back(e.action);
      stats->root_action_visits.push_back(e.visit_count);
    }
    stats->root_priors = std::move(raw_root_priors);
    const Edge& best = root_node.edges[best_edge];
    float q = 0.0f;
    if (best.visit_count > 0) q = best.value_sum / static_cast<float>(best.visit_count);
    stats->best_action_value = static_cast<double>(clip_value(q, cfg_.value_clip));
    stats->root_values.resize(static_cast<size_t>(np));
    if (best.visit_count > 0) {
      const auto& rev = root_edge_values[static_cast<size_t>(best_edge)];
      for (int p = 0; p < np; ++p) {
        stats->root_values[static_cast<size_t>(p)] = rev[static_cast<size_t>(p)] / best.visit_count;
      }
    }
    stats->root_edge_values.resize(root_node.edges.size());
    for (size_t ei = 0; ei < root_node.edges.size(); ++ei) {
      auto& out = stats->root_edge_values[ei];
      out.resize(static_cast<size_t>(np), 0.0);
      const int vc = root_node.edges[ei].visit_count;
      if (vc > 0) {
        const auto& rev = root_edge_values[ei];
        for (int p = 0; p < np; ++p) {
          out[static_cast<size_t>(p)] = rev[static_cast<size_t>(p)] / vc;
        }
      }
    }
  }
  return root_node.edges[best_edge].action;
}

}  // namespace board_ai::search
