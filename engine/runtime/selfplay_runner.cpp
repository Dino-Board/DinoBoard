#include "selfplay_runner.h"

#include <cassert>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

#include "../core/masked_state.h"
#include "../core/rng_salt.h"
#include "../core/snapshot_io.h"

namespace board_ai::runtime {

SelfplayEpisodeResult run_selfplay_episode(
    IGameState& initial_state,
    const IGameRules& rules,
    const IStateValueModel& value_model,
    PolicyEvaluatorFactory evaluator_for_player,
    const SelfplayConfig& config,
    std::uint64_t episode_seed,
    std::vector<IBeliefTracker*> per_perspective_trackers,
    std::vector<IGameState*> per_seat_states,
    PublicStateApplier public_state_applier,
    const IFeatureEncoder* encoder,
    const search::ITailSolver* tail_solver,
    GameAdjudicator adjudicator,
    AuxiliaryScorer auxiliary_scorer,
    const IHeuristic* heuristic,
    TrainingActionFilter training_action_filter,
    TailSolveTrigger tail_solve_trigger,
    EpisodeStatsExtractor episode_stats_extractor,
    int trace_perspective,
    IBeliefTracker* trace_belief_tracker,
    PublicEventExtractor public_event_extractor,
    EventsOnlyExtractor events_only_extractor,
    const IBeliefFeatureExtractor* belief_extractor,
    std::vector<const IBeliefEvaluator*> per_seat_belief_evaluators,
    const IBeliefLabelExtractor* belief_label_extractor) {
  SelfplayEpisodeResult result{};
  if (belief_label_extractor) {
    result.belief_label_class_count =
        belief_label_extractor->label_class_count();
  }

  if (config.tail_solve_enabled && (!tail_solver || !tail_solve_trigger)) {
    throw std::invalid_argument(
        "run_selfplay_episode: tail_solve_enabled=true requires both a "
        "registered ITailSolver and a TailSolveTrigger; one or both are missing.");
  }

  // events_only_extractor non-null requires public_event_extractor non-null:
  // the sim-tracker descent (events_only) is meaningful only if the seat
  // advance path (public_event) is also wired. The other direction is
  // permitted (Azul registers public_event_extractor + public_state_applier
  // directly, no tracker, no events_only).
  if (events_only_extractor && !public_event_extractor) {
    throw std::invalid_argument(
        "run_selfplay_episode: events_only_extractor is registered but "
        "public_event_extractor is null. The hidden-info advance path "
        "needs both; this combination is a binding bug.");
  }

  // Tracing works on any game with a public_event_extractor. Tracker
  // is optional: for games without one (Azul: fully-public counts +
  // snapshot, no tracker registered) the trace still records actions /
  // events / public_snapshot for round-trip / replay tests, only
  // belief_snapshot_after is left empty. Tracker bootstrap and
  // observe_public_event below are guarded individually.
  const bool tracing =
      trace_perspective >= 0 && public_event_extractor;
  if (tracing) {
    result.trace_enabled = true;
    result.trace_perspective = trace_perspective;
  }

  std::unique_ptr<FilteredRulesWrapper> filtered_rules_ptr;
  if (training_action_filter) {
    filtered_rules_ptr = std::make_unique<FilteredRulesWrapper>(rules, training_action_filter);
  }

  auto state = initial_state.clone_state();
  // Snapshot of initial truth for episode_stats_extractor. Some stats
  // (e.g. coup's "did first player actually have Duke when they Tax'd")
  // need to inspect ground-truth hidden fields at game start, which
  // `state` no longer holds after the play loop advances it. Cloning is
  // cheap (one IGameState alloc per episode) compared to the play loop.
  auto initial_truth = initial_state.clone_state();
  int ply = 0;

  // Per-seat session state is mandatory: MCTS / encoder / heuristic /
  // legal-action queries on the AI path read from that seat's session
  // state, never from truth. The session state is advanced via the
  // public-event protocol every ply (mirrors py_engine::advance_ai_view_).
  const int num_players = state->num_players();
  if (static_cast<int>(per_seat_states.size()) != num_players) {
    throw std::runtime_error(
        "run_selfplay_episode: per_seat_states size != num_players");
  }
  for (int p = 0; p < num_players; ++p) {
    if (per_seat_states[p] == nullptr) {
      throw std::runtime_error(
          "run_selfplay_episode: per_seat_states[p] must not be null");
    }
  }
  auto ai_view_for = [&](int seat) -> IGameState& {
    return *per_seat_states[seat];
  };

  // Init each seat's tracker once at episode start. Every public event is
  // fed to every tracker in the main loop below; each tracker's belief
  // accumulates monotonically until game over.
  const bool use_per_perspective = !per_perspective_trackers.empty();
  if (use_per_perspective) {
    const int num_players = state->num_players();
    if (static_cast<int>(per_perspective_trackers.size()) != num_players) {
      throw std::runtime_error(
          "run_selfplay_episode: per_perspective_trackers size != num_players");
    }
    for (int p = 0; p < num_players; ++p) {
      if (per_perspective_trackers[p]) {
        // selfplay path: per-seat session state was reset_with_seed at
        // game start, identical to truth. payload is empty — the
        // tracker's init only seeds its own public memory by reading
        // public slots from state. The session API path constructs a
        // non-empty payload via pack_init_payload to bootstrap
        // perspective-private slots when the session was seeded
        // independently of GT.
        IGameState& seat = *per_seat_states[p];
        per_perspective_trackers[p]->init(seat, p, AnyMap{});
      }
    }
  }

  // No episode-start freshen of per-seat states. Decision-side reads
  // (encoder via masked-state placeholder, hash via kHiddenHashSentinel,
  // MCTS sims via sim_tracker->randomize_unseen at sim entry) are
  // perspective-aware by construction; truth values that linger in
  // per_seat_states' viz=0 slots are structurally unreachable. A future
  // direct-field read of opp-private would be a bug to catch in tests,
  // not silently paper over with a session-side resample.

  // Tracing uses its own separate tracker instance so trace output stays
  // reproducible across refactors of MCTS tracker routing. Initial
  // observation is the snapshot-path's opening twin:
  //   { "public_snapshot": <walker viz::serialize_public_snapshot(state, perspective)>,
  //     "tracker_init":    <tracker.pack_init_payload(state, p)> }
  // Wire shape mirrors per-ply public_snapshot: the public part is
  // produced by exactly the same walker, and `tracker_init` plays the
  // role per-ply `events` plays for `tracker.observe_public_event`.
  if (tracing) {
    AnyMap pub;
    viz::serialize_public_snapshot(
        *state, state->schema_ref(), trace_perspective, pub);
    AnyMap tracker_init_payload;
    if (trace_belief_tracker) {
      tracker_init_payload = trace_belief_tracker->pack_init_payload(
          *state, trace_perspective);
      // Trace's own tracker bootstrap: per_seat_states[trace_perspective]
      // already mirrors truth, so the payload is redundant on selfplay's
      // own state — but feeding the same payload here guarantees the
      // trace tracker's init is exercised on the same path API uses.
      trace_belief_tracker->init(
          *per_seat_states[trace_perspective], trace_perspective,
          tracker_init_payload);
      result.initial_belief_snapshot = trace_belief_tracker->serialize();
    }
    result.initial_observation["public_snapshot"] = std::any(std::move(pub));
    result.initial_observation["tracker_init"] =
        std::any(std::move(tracker_init_payload));
  }

  std::mt19937_64 heuristic_rng(
      board_ai::rng::derive_subseed(episode_seed, "selfplay.heuristic_pick"));
  std::uniform_real_distribution<double> heuristic_dist(0.0, 1.0);
  int heuristic_moves = 0;

  const bool playout_cap = config.fast_simulations > 0;
  if (playout_cap) {
    if (config.fast_simulations >= config.simulations) {
      throw std::invalid_argument(
          "selfplay: fast_simulations (" + std::to_string(config.fast_simulations) +
          ") must be < simulations (" + std::to_string(config.simulations) + ")");
    }
    if (!(config.full_search_prob > 0.0 && config.full_search_prob < 1.0)) {
      throw std::invalid_argument(
          "selfplay: full_search_prob must be in (0, 1) when fast_simulations > 0, got " +
          std::to_string(config.full_search_prob));
    }
  }
  std::mt19937_64 playout_cap_rng(
      board_ai::rng::derive_subseed(episode_seed, "selfplay.playout_cap"));

  // GT step rng for do_action_fast on truth state. Seeded directly from
  // episode_seed (no subseed derivation) so that any other GT runner —
  // notably the API session's `step_rng_` in py_engine.cpp — receiving
  // the same episode seed advances truth through identical draws. The
  // round-trip test `test_public_snapshot_round_trip` depends on this
  // equivalence (it replays selfplay's trace via GameSession.apply_action
  // and asserts public state matches the trace's snapshot).
  std::mt19937_64 step_rng(episode_seed);

  // Per-seat session-state advance: mirrors py_engine::advance_ai_view_.
  // Called after truth has been advanced by `chosen`, with the truth state
  // before/after both available. For each seat:
  //   - hidden-info game (public_event_extractor registered): public_snapshot
  //     overwrites the seat's public fields. Per DEC-003, viz=0 hidden
  //     slots are NOT freshened — they are unread bytes that the framework
  //     structurally hides from the hash, encoder, and sim entry.
  //     do_action_fast is NOT run on the seat — the observer has incomplete
  //     information so replaying the action there would just sample one
  //     world; the public projection arrives whole via the snapshot, and
  //     the eventual decision will be made off whatever world MCTS samples.
  //   - fully-public game (no extractor): the seat just replays do_action_fast
  //     deterministically.
  auto advance_per_seat_states =
      [&](const IGameState& truth_before, const IGameState& truth_after,
          ActionId chosen, int /*actor*/) {
    for (int p = 0; p < num_players; ++p) {
      IGameState& seat = *per_seat_states[p];
      if (!public_event_extractor) {
        // Fully-public game: just replay the action on the seat state.
        const std::uint64_t view_step_seed = board_ai::rng::derive_subseed(
            episode_seed, "selfplay.view_step",
            static_cast<std::uint64_t>(ply) * 17ULL +
                static_cast<std::uint64_t>(p));
        std::mt19937_64 view_step_rng(view_step_seed);
        rules.do_action_fast(seat, chosen, view_step_rng);
        continue;
      }
      PublicEventTrace evt_p = public_event_extractor(
          truth_before, chosen, truth_after, p);
      seat.begin_step_for_session_observe();
      if (public_state_applier && !evt_p.public_snapshot.empty()) {
        public_state_applier(seat, evt_p.public_snapshot, p);
      }
      // tracker.observe_public_event is called in the per_perspective loop
      // immediately below. No hidden-slot freshening — sim entry will
      // sample fresh worlds on a cloned sim_tracker (DEC-003).
    }
  };

  // Belief-net training-sample emit (Plan §2.2). Active when both the
  // feature extractor (AI side) and label extractor (GT side) are
  // present. Independent of belief_evaluator: emit can run with no
  // installed belief.onnx (early training).
  const bool emit_belief_samples =
      belief_extractor != nullptr && belief_label_extractor != nullptr;
  auto emit_belief_for_ply = [&]() {
    if (!emit_belief_samples) return;
    for (int observer = 0; observer < num_players; ++observer) {
      IBeliefTracker* obs_tracker =
          (use_per_perspective &&
           observer < static_cast<int>(per_perspective_trackers.size()))
              ? per_perspective_trackers[observer]
              : nullptr;
      BeliefSample bs{};
      bs.ply = ply;
      bs.observer = observer;
      // Feature side: per-seat session + that seat's tracker. No truth.
      if (!belief_extractor->extract_from_state(
              *per_seat_states[observer], observer, obs_tracker,
              &bs.features)) {
        throw std::runtime_error(
            "run_selfplay_episode: belief feature extractor returned wrong size");
      }
      // Label side: read from truth.
      belief_label_extractor->extract(
          *state, observer, &bs.hand_counts, &bs.remaining, &bs.alive_per_opp);
      result.belief_samples.push_back(std::move(bs));
    }
  };

  while (!state->is_terminal() && ply < config.max_game_plies) {
    const int player = state->current_player();

    // Belief-sample emit at the START of each ply: features describe the
    // observer's pre-action belief state, labels read truth for the same
    // moment. One sample per observer.
    emit_belief_for_ply();

    const bool use_filter = filtered_rules_ptr &&
        config.training_filter_ratio > 0.0 &&
        heuristic_dist(heuristic_rng) < config.training_filter_ratio;
    const IGameRules& effective_rules = use_filter ? *filtered_rules_ptr : rules;

    // AI-path reads come from the acting seat's session state (truth-free).
    IGameState& ai_view = ai_view_for(player);

    const auto legal = effective_rules.legal_actions(ai_view);
    if (legal.empty()) break;
    const std::vector<ActionId> full_legal_storage =
        use_filter ? rules.legal_actions(ai_view) : std::vector<ActionId>{};
    const std::vector<ActionId>& full_legal =
        use_filter ? full_legal_storage : legal;

    const bool use_heuristic = heuristic &&
        config.heuristic_guidance_ratio > 0.0 &&
        heuristic_dist(heuristic_rng) < config.heuristic_guidance_ratio;

    if (use_heuristic) {
      const std::uint64_t hr_seed = board_ai::rng::derive_subseed(
          episode_seed, "selfplay.heuristic", static_cast<std::uint64_t>(ply));
      // Mask the seat's view from its own perspective so the heuristic's
      // viz=0 reads land on placeholder, not truth. Hidden-info paths set
      // ai_view to the seat's session state (already truth-free), but the
      // session may still hold last-ply leftovers in viz=0 slots; masking
      // again before heuristic dispatch is the structural guard.
      auto masked = make_masked_state(ai_view, ai_view.schema_ref(), player);
      const IBeliefTracker* hr_tracker =
          (use_per_perspective && player >= 0 &&
           player < static_cast<int>(per_perspective_trackers.size()))
              ? per_perspective_trackers[player]
              : nullptr;
      auto hr = heuristic->pick(*masked, hr_tracker, effective_rules, hr_seed);

      const size_t n_actions = hr.actions.size();
      std::vector<double> probs(n_actions);
      double max_score = *std::max_element(hr.scores.begin(), hr.scores.end());

      if (config.heuristic_temperature > 1e-6) {
        double inv_temp = 1.0 / config.heuristic_temperature;
        double sum_exp = 0.0;
        for (size_t i = 0; i < n_actions; ++i) {
          probs[i] = std::exp((hr.scores[i] - max_score) * inv_temp);
          sum_exp += probs[i];
        }
        for (auto& p : probs) p /= sum_exp;
      } else {
        int count = 0;
        for (size_t i = 0; i < n_actions; ++i) {
          if (hr.scores[i] >= max_score - 1e-9) ++count;
        }
        for (size_t i = 0; i < n_actions; ++i) {
          probs[i] = (hr.scores[i] >= max_score - 1e-9) ? 1.0 / count : 0.0;
        }
      }

      // Convert to fake visits for policy target
      std::vector<int> fake_visits(n_actions);
      for (size_t i = 0; i < n_actions; ++i) {
        fake_visits[i] = static_cast<int>(probs[i] * 10000.0 + 0.5);
      }

      // Sample action from distribution
      const std::uint64_t h = board_ai::rng::derive_subseed(
          hr_seed, "selfplay.heuristic_sample");
      double r = static_cast<double>(h & 0xFFFFFFFF) / 4294967296.0;
      size_t chosen_idx = n_actions - 1;
      double cumulative = 0.0;
      for (size_t i = 0; i < n_actions; ++i) {
        cumulative += probs[i];
        if (r <= cumulative) { chosen_idx = i; break; }
      }
      ActionId chosen = hr.actions[chosen_idx];

      SelfplaySample sample{};
      sample.ply = ply;
      sample.player = player;
      sample.action_id = chosen;
      sample.policy_action_ids = hr.actions;
      if (encoder) {
        const IBeliefTracker* enc_tracker =
            (use_per_perspective && player >= 0 &&
             player < static_cast<int>(per_perspective_trackers.size()))
                ? per_perspective_trackers[player]
                : nullptr;
        encoder->encode(ai_view, player, enc_tracker, full_legal, &sample.features, &sample.legal_mask);
      }
      sample.policy_action_visits = fake_visits;
      sample.heuristic_simulations = hr.simulations;
      result.samples.push_back(std::move(sample));

      std::unique_ptr<IGameState> state_before = state->clone_state();
      const ActionId observed = rules.public_action(*state_before, chosen);
      effective_rules.do_action_fast(*state, chosen, step_rng);
      if (use_per_perspective) {
        const int num_players = static_cast<int>(per_perspective_trackers.size());
        for (int p = 0; p < num_players; ++p) {
          if (!per_perspective_trackers[p]) continue;
          PublicEventTrace evt_p;
          if (public_event_extractor) {
            evt_p = public_event_extractor(*state_before, chosen, *state, p);
          }
          per_perspective_trackers[p]->observe_public_event(
              player, observed, evt_p.events);
        }
      }
      // Advance per-seat session states via the public-event protocol so
      // the AI path's view of public state stays in sync with truth's
      // public projection while hidden remains a tracker-consistent sample.
      advance_per_seat_states(*state_before, *state, chosen, player);
      if (tracing) {
        SelfplayObservationTrace t{};
        t.ply = ply;
        t.actor = player;
        t.action = observed;
        auto evt = public_event_extractor(
            *state_before, chosen, *state, trace_perspective);
        t.events = evt.events;
        t.public_snapshot = evt.public_snapshot;
        if (trace_belief_tracker) {
          trace_belief_tracker->observe_public_event(
              player, observed, evt.events);
          t.belief_snapshot_after = trace_belief_tracker->serialize();
        }
        result.observation_trace.push_back(std::move(t));
      }
      ply += 1;
      heuristic_moves += 1;
      continue;
    }

    // MCTS root uses the acting player's per-perspective tracker — it has
    // accumulated the full observation history for that seat since game
    // start, giving ISMCTS an accurate belief to sample from.
    IBeliefTracker* mcts_tracker = nullptr;
    if (use_per_perspective && player >= 0 &&
        player < static_cast<int>(per_perspective_trackers.size())) {
      mcts_tracker = per_perspective_trackers[player];
    }

    const bool full_search = !playout_cap ||
        std::uniform_real_distribution<double>(0.0, 1.0)(playout_cap_rng) < config.full_search_prob;
    const auto noise = search::resolve_root_dirichlet_noise(
        config.dirichlet_alpha, config.dirichlet_epsilon,
        config.dirichlet_on_first_n_plies, ply);

    const bool try_tail_solve = config.tail_solve_enabled && tail_solver &&
        tail_solve_trigger && tail_solve_trigger(ai_view, ply);

    search::NetMctsConfig mcts_cfg{};
    mcts_cfg.simulations = full_search ? config.simulations : config.fast_simulations;
    mcts_cfg.world_cap = config.world_cap;
    mcts_cfg.c_puct = config.c_puct;
    mcts_cfg.fpu_reduction = config.fpu_reduction;
    mcts_cfg.uct2_arrival_visits = config.uct2_arrival_visits;
    mcts_cfg.max_depth = config.max_depth;
    mcts_cfg.value_clip = config.value_clip;
    mcts_cfg.root_dirichlet_alpha = full_search ? noise.alpha : 0.0;
    mcts_cfg.root_dirichlet_epsilon = full_search ? noise.epsilon : 0.0;
    mcts_cfg.opponent_selection = config.opponent_selection;
    mcts_cfg.safe_mode = config.safe_mode;
    mcts_cfg.leaf_batch_size = config.leaf_batch_size;
    // ISMCTS: root-sampling hidden info + DAG per-acting-player keying.
    // MCTS uses the per-sim sampled world's rules.legal_actions at each node.
    if (mcts_tracker) {
      mcts_cfg.root_belief_tracker = mcts_tracker;
    }
    // Sim-tracker descent maintenance (SIM_TRACKER_DESCENT_PLAN). Uses
    // the events-only extractor (sim path doesn't need the wholesale
    // public_snapshot the ply-protocol extractor produces — sim_state
    // is itself a GT-equivalent advanced by do_action_fast).
    if (events_only_extractor) {
      mcts_cfg.events_only_extractor = events_only_extractor;
    }
    {
      const IBeliefEvaluator* seat_belief_eval =
          (!per_seat_belief_evaluators.empty() &&
           player >= 0 &&
           player < static_cast<int>(per_seat_belief_evaluators.size()))
              ? per_seat_belief_evaluators[player]
              : nullptr;
      if (belief_extractor && seat_belief_eval) {
        mcts_cfg.belief_extractor = belief_extractor;
        mcts_cfg.belief_evaluator = seat_belief_eval;
      }
    }

    if (try_tail_solve) {
      mcts_cfg.tail_solve_enabled = true;
      mcts_cfg.tail_solve_config = config.tail_solve_config;
      // auxiliary_scorer is the game's score-margin function. Tail solver
      // uses it as a tiebreak among proven wins (terminal value >= 1.0).
      if (auxiliary_scorer && !config.tail_solve_config.margin_scorer) {
        mcts_cfg.tail_solve_config.margin_scorer = auxiliary_scorer;
      }
      mcts_cfg.tail_solver = tail_solver;
    }

    search::NetMcts mcts(mcts_cfg);
    search::NetMctsStats stats{};
    const std::uint64_t mcts_seed = board_ai::rng::derive_subseed(
        episode_seed, "selfplay.mcts", static_cast<std::uint64_t>(ply));
    const search::IPolicyValueEvaluator& seat_evaluator =
        evaluator_for_player(player);
    mcts.search_root(ai_view, effective_rules, value_model, seat_evaluator,
                     &stats, mcts_seed);

    if (stats.tail_solve_attempted) {
      result.tail_solve_attempts += 1;
      result.tail_solve_total_ms += stats.tail_solve_elapsed_ms;
      if (stats.tail_solve_completed) {
        result.tail_solve_completed += 1;
      }
      if (stats.tail_solved) {
        result.tail_solve_successes += 1;
      }
    }

    SelfplaySample sample{};
    sample.ply = ply;
    sample.player = player;
    sample.policy_action_ids = stats.root_actions;
    sample.policy_action_visits = stats.root_action_visits;
    sample.tail_solved = stats.tail_solved;
    sample.full_search = full_search;
    sample.simulations = stats.simulations_done;
    if (encoder && (full_search || config.encode_fast_samples)) {
      const IBeliefTracker* enc_tracker =
          (use_per_perspective && player >= 0 &&
           player < static_cast<int>(per_perspective_trackers.size()))
              ? per_perspective_trackers[player]
              : nullptr;
      encoder->encode(ai_view, player, enc_tracker, full_legal, &sample.features, &sample.legal_mask);
    }

    const double temperature = stats.tail_solved ? 0.0 :
        search::resolve_linear_temperature(config.temperature_schedule, config.temperature, ply);
    const std::uint64_t action_seed = board_ai::rng::derive_subseed(
        episode_seed, "selfplay.action_pick", static_cast<std::uint64_t>(ply));
    const ActionId chosen = search::select_action_from_visits(
        stats.root_actions, stats.root_action_visits, temperature, action_seed,
        config.top_k, legal[0]);

    sample.action_id = chosen;
    result.samples.push_back(std::move(sample));

    std::unique_ptr<IGameState> state_before = state->clone_state();
    const ActionId observed = rules.public_action(*state_before, chosen);
    effective_rules.do_action_fast(*state, chosen, step_rng);
    if (use_per_perspective) {
      // Every seat's tracker sees every action. Each perspective extracts
      // its own perspective-specific events.
      const int num_players = static_cast<int>(per_perspective_trackers.size());
      for (int p = 0; p < num_players; ++p) {
        if (!per_perspective_trackers[p]) continue;
        PublicEventTrace evt_p;
        if (public_event_extractor) {
          evt_p = public_event_extractor(*state_before, chosen, *state, p);
        }
        per_perspective_trackers[p]->observe_public_event(
            player, observed, evt_p.events);
      }
    }
    advance_per_seat_states(*state_before, *state, chosen, player);
    if (tracing) {
      SelfplayObservationTrace t{};
      t.ply = ply;
      t.actor = player;
      t.action = observed;
      auto evt = public_event_extractor(
          *state_before, chosen, *state, trace_perspective);
      t.events = evt.events;
      t.public_snapshot = evt.public_snapshot;
      if (trace_belief_tracker) {
        trace_belief_tracker->observe_public_event(
            player, observed, evt.events);
        t.belief_snapshot_after = trace_belief_tracker->serialize();
      }
      result.observation_trace.push_back(std::move(t));
    }
    ply += 1;
  }

  result.total_plies = ply;
  result.final_state = state->clone_state();

  if (state->is_terminal()) {
    const int num_players = state->num_players();
    float best_val = -2.0f;
    int best_player = -1;
    bool is_draw = true;
    for (int p = 0; p < num_players; ++p) {
      const float v = value_model.terminal_value_for_player(*state, p);
      if (v > best_val) {
        best_val = v;
        best_player = p;
      }
    }
    for (int p = 0; p < num_players; ++p) {
      const float v = value_model.terminal_value_for_player(*state, p);
      if (std::abs(v - best_val) > 1e-6f) {
        is_draw = false;
        break;
      }
    }
    if (is_draw) {
      result.draw = true;
      result.winner = -1;
    } else {
      result.draw = false;
      result.winner = best_player;
    }

    const auto terminal_vals = value_model.terminal_values(*state);
    assert(std::abs(std::accumulate(terminal_vals.begin(), terminal_vals.end(), 0.0f)) < 1e-4f);
    result.terminal_z_values = terminal_vals;
    for (auto& s : result.samples) {
      s.z_values = terminal_vals;
      if (result.draw) {
        s.z = 0.0f;
      } else {
        s.z = (s.player == result.winner) ? 1.0f : -1.0f;
      }
    }
  } else if (adjudicator) {
    const int adj_winner = adjudicator(*state);
    if (adj_winner < 0) {
      result.draw = true;
      result.winner = -1;
    } else {
      result.draw = false;
      result.winner = adj_winner;
    }
    const int num_players = state->num_players();
    // Non-terminal adjudication is a timeout/anti-stalling fallback. Until a
    // game supplies richer custom adjudication values, penalize every seat.
    std::vector<float> adj_vals(num_players, -1.0f);
    result.terminal_z_values = adj_vals;
    for (auto& s : result.samples) {
      s.z_values = adj_vals;
      s.z = -1.0f;
    }
  } else {
    // Game didn't finish and no adjudicator — treat as draw (all zeros).
    const int np = state->num_players();
    std::vector<float> zero_vals(np, 0.0f);
    result.terminal_z_values = zero_vals;
    for (auto& s : result.samples) {
      s.z_values = zero_vals;
      s.z = 0.0f;
    }
    result.draw = true;
    result.winner = -1;
  }

  if (episode_stats_extractor) {
    std::vector<SelfplaySampleView> views;
    views.reserve(result.samples.size());
    for (const auto& s : result.samples) {
      views.push_back({s.ply, s.player, s.action_id, s.features.empty() ? nullptr : &s.features});
    }
    result.custom_stats = episode_stats_extractor(*state, *initial_truth, views);
  }

  return result;
}

}  // namespace board_ai::runtime
