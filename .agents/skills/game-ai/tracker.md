# IBeliefTracker Reference

## Interface

```cpp
class IBeliefTracker {
 public:
  virtual void init(IGameState& state, int perspective,
                    const AnyMap& payload) = 0;

  virtual AnyMap pack_init_payload(
      const IGameState& gt_state, int perspective) const;  // default: empty

  virtual void observe_public_event(
      int actor, ActionId action,
      const std::vector<PublicEvent>& events) = 0;

  virtual void randomize_unseen(IGameState& state, int observer,
                                std::mt19937_64& rng) const = 0;

  virtual std::unique_ptr<IBeliefTracker> clone() const = 0;

  virtual void prepare_for_root(
      const IGameState& root_state, int root_player,
      const IBeliefFeatureExtractor* extractor,
      const IBeliefEvaluator* evaluator);  // default: no-op

  virtual AnyMap serialize() const;  // default: empty
};
```

## Lifecycle

```
Game start:
  1. GT creates truth state with seed and runs any rules-side initial reveals.
  2. GT serializes an opening public_snapshot per perspective.
  3. GT calls pack_init_payload(gt_state, perspective) per perspective.
  4. Each AI session applies the opening snapshot to its independent state.
  5. Session calls tracker.init(session_state, perspective, payload).

Each ply:
  6. GT runs do_action_fast on truth state.
  7. GT extracts public_event_extractor(before, action, after, perspective).
  8. Observer session receives (action_id, events, public_snapshot).
  9. Session bumps step_count via begin_step_for_session_observe().
  10. Session applies public_snapshot; hidden slots are left untouched.
  11. Session calls tracker.observe_public_event(actor, action, events).

MCTS decision:
  12. tracker.prepare_for_root(root_state, root_player, extractor, evaluator).
  13. Split simulations into K worlds x M sims/world using world_cap.
  14. For each world:
      a. world_state = root_state.clone()
      b. world_tracker = session_tracker.clone()
      c. world_tracker.randomize_unseen(world_state, root_player, world_rng)
         exactly once for this world.
  15. For each sim inside that world:
      a. sim_state = world_state.clone()
      b. sim_tracker = world_tracker.clone()
      c. For each sim step:
         - snapshot state_before if events_only_extractor is registered
         - rules.do_action_fast(sim_state, action, sim_rng)
         - sim_tracker.observe_public_event(actor, action, events_only_diff)
      d. sim_tracker is discarded at sim end.
```

Tracker content is perspective-agnostic public-derived memory. Per-perspective
private knowledge (own hand, peeked cards, exchange choices) lives on state with
the proper `viz=1` slice, not in the tracker.

## init

```cpp
void MyTracker::init(IGameState& state, int perspective,
                     const AnyMap& payload) {
  // Write perspective-private bootstrap into this session state, if the
  // opening public snapshot cannot carry it.
  if (payload.count("own_hand")) {
    auto& s = checked_cast<MyGameState>(state);
    s.hand[perspective] = std::any_cast<int8_t>(payload.at("own_hand"));
    // Toggle viz if needed (e.g. owner sees own hand)
  }

  // Seed perspective-agnostic internal memory from public facts on state.
  // e.g. read discard pile, public card counts, etc.
  reset_internal_counts(state);
}
```

- `payload` is game-specific, produced by `pack_init_payload` on the GT side.
- For selfplay/arena: payload is empty (session state already seeded with same seed as truth).
- For API/web: payload carries perspective-private bootstrap (own hand, etc.).
- Do not store perspective-private facts in tracker fields. Store them in state,
  guarded by viz, so hash/mask/encoder all share the same information boundary.

## pack_init_payload

```cpp
AnyMap MyTracker::pack_init_payload(
    const IGameState& gt_state, int perspective) const {
  const auto& s = checked_cast<MyGameState>(gt_state);
  AnyMap payload;
  // Package perspective-private facts that the public broadcast can't reach
  payload["own_hand"] = std::any(s.hand[perspective]);
  return payload;
}
```

## observe_public_event

```cpp
void MyTracker::observe_public_event(
    int actor, ActionId action,
    const std::vector<PublicEvent>& events) {
  // Update internal belief from event payloads ONLY.
  // NO peeking at state or truth.
  for (const auto& [kind, payload] : events) {
    if (kind == "card_revealed") {
      int card = std::any_cast<int>(payload.at("card"));
      int player = std::any_cast<int>(payload.at("player"));
      public_seen_cards_.insert(card);
      // ...
    }
  }
  // Track action history (claims, challenges, etc.)
  action_history_.push_back({actor, action});
}
```

## randomize_unseen

```cpp
void MyTracker::randomize_unseen(
    IGameState& state, int observer,
    std::mt19937_64& rng) const {
  auto& s = checked_cast<MyGameState>(state);

  // 1. Compute unseen pool
  //    unseen = full_pool - tracker.public_seen - state.viz=1[observer]
  const auto& hand_viz = viz::viz_get(state, "hand");
  auto hand_visible = [&](int p) {
    const std::size_t base =
        viz::flat_offset_data_only(hand_viz.shape, std::vector<int>{p});
    return hand_viz.data[base + static_cast<std::size_t>(observer)] != 0;
  };

  std::vector<int> unseen_pool;
  for (int card : all_cards) {
    if (!public_seen_.count(card) && !observer_already_sees_card(s, observer, card)) {
      unseen_pool.push_back(card);
    }
  }

  // 2. Shuffle
  std::shuffle(unseen_pool.begin(), unseen_pool.end(), rng);

  // 3. Fill viz=0 slots
  int pool_idx = 0;
  for (int p = 0; p < num_players; ++p) {
    if (!hand_visible(p)) {
      s.hand[p] = unseen_pool[pool_idx++];
    }
  }
}
```

### Contract
- Fill every slot where `viz[..., observer]=0` with concrete values
- Public slots (viz=1) in the result must depend ONLY on observation history
- Same observation sequence must yield byte-equal public hash across different trackers
- Called at the MCTS world entry on a cloned world tracker. With `world_cap > 0`,
  M inner sims share one sampled world by cloning the already-randomized
  `world_state` / `world_tracker`.
- Never called on the session tracker itself. Session viz=0 slots are unread
  bytes structurally hidden by hash sentinels, encoder placeholders, and
  world-entry resampling.

## When NOT needed

Games that don't need a tracker:
- Fully public + deterministic (TicTacToe, Quoridor): no hidden info at all
- Public physical-randomness (Azul): bag draws are public, no belief to track

Register a tracker only for information-asymmetric games (Love Letter, Splendor, Coup, 6 nimmt!, Liar's Dice).

## What goes on tracker vs. schema

Rule of thumb: **anything do_action_fast / legal_actions reads must be on schema** (BUG-037 — same perspective hash ⇒ same legal actions). Off-schema state is invisible to AI sessions because `public_snapshot` only carries schema fields. The tracker is for **public-derived memory that ground truth doesn't bother storing but encoder/belief net wants** — public action histories, claim/challenge logs, per-round play tallies, claim-resolution counters. The tracker rebuilds itself from the public event stream (`observe_public_event`) and is cloned per perspective, so the same memory exists on every AI session without round-tripping through snapshot.

Examples:
- Liar's Dice `bids_by_player_` — full public bid log, encoder reads last 5 per seat.
- Coup `pre_/post_claim_counts_` — per-role claim tallies split across reshuffles.
- 6 nimmt! per-round played-card history — what each seat dropped this round.

If a value is only ever consumed by the encoder and is fully reconstructable from public events, prefer tracker over schema — it keeps the DAG hash narrow (fewer hot bytes) and avoids snapshot bloat.

## Known issues

- **BUG-017**: tracker content must be perspective-invariant. Two trackers with different perspectives fed the same public event stream must produce equal public-derived aggregates. `test_tracker_perspective_invariance` guards this.

- **BUG-032**: `randomize_unseen` must not read the input state's hidden contents to decide sampling. The unseen pool is derived from `tracker.public_seen` + `state.viz=1[observer]` slots, never from viz=0 slot values.

- **DEC-003**: `randomize_unseen` is called in exactly one place: MCTS world entry on a cloned tracker. The session itself never calls it. Session viz=0 slots are unread bytes -- structurally hidden from hash (kHiddenHashSentinel), encoder (kPlaceholder), and sim (clone + resample).

## events_only_diff for install_event_protocol

Games register their event diff function via `GameBundle::install_event_protocol`:

```cpp
// In register.cpp:
b.install_event_protocol(
    // events_only_diff: extracts events from before/after state diff
    [](const IGameState& before, ActionId action,
       const IGameState& after, int perspective)
        -> std::vector<PublicEvent> {
      // diff before vs after to produce public events
      // Return empty vector if no events for this action
      return events;
    },
    // schema_provider: returns the static schema
    []() -> const viz::VisibilitySchema& {
      return MyGameState::schema();
    });
```

`events_only_diff` is consumed by MCTS sim descent. After each simulated
`do_action_fast`, the search calls it on `(state_before, action, state_after,
root_player)` and feeds the resulting events to the sim-local tracker. If your
encoder reads tracker-derived public aggregates that can change after actions,
this diff must emit enough events to keep the sim tracker synchronized.

For games whose tracker is event-free (all info flows through snapshot/viz):
```cpp
b.install_event_protocol(
    viz::no_events_extractor,
    []() -> const viz::VisibilitySchema& { return MyGameState::schema(); });
```
