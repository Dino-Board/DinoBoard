# IGameRules Reference

## Interface overview

```cpp
class IGameRules {
 public:
  virtual bool validate_action(const IGameState& state, ActionId action) const = 0;
  virtual std::vector<ActionId> legal_actions(const IGameState& state) const = 0;

  // Framework wrappers -- public entry points (NOT virtual):
  void do_action_fast(IGameState& state, ActionId action, std::mt19937_64& rng) const;
  UndoToken do_action_deterministic(IGameState& state, ActionId action) const;
  void undo_action(IGameState& state, const UndoToken& token) const;

 protected:
  // Game-defined impls -- override these:
  virtual void do_action_fast_impl(IGameState& state, ActionId action,
                                   std::mt19937_64& rng) const = 0;
  virtual void undo_action_impl(IGameState& state, const UndoToken& token) const = 0;
  virtual UndoToken do_action_deterministic_impl(IGameState& state, ActionId action) const;
};
```

The framework wrappers handle `step_count_` bookkeeping:
- `do_action_fast`: increments `step_count_` then calls `do_action_fast_impl`
- `do_action_deterministic`: increments `step_count_` then calls `do_action_deterministic_impl`
- `undo_action`: calls `undo_action_impl` then decrements `step_count_`

Game code never sees or touches `step_count_`.

## legal_actions / validate_action

`legal_actions(state)` returns the action ids available to `state.current_player()`. `validate_action(state, action)` should be equivalent to membership in that list and must not mutate the state.

For ISMCTS DAG reuse, legal-action scope must match the perspective hash: if two sampled worlds have the same `state_hash_for_perspective(state.current_player())`, they must produce the same `legal_actions` set for that acting player. Otherwise a reused MCTS edge can become illegal in a different sampled world.

Action ids are a shared contract across rules, encoders, descriptors, tests, API clients, and policy heads. Keep ids stable and inside the configured action space.

## do_action_fast_impl

The hot-path action application. Used by MCTS, selfplay, arena, web/API.

```cpp
void MyGameRules::do_action_fast_impl(
    IGameState& state, ActionId action,
    std::mt19937_64& rng) const {
  auto& s = checked_cast<MyGameState>(state);

  // 1. Apply the action to game fields
  s.board[action] = s.current_player_;

  // 2. Maintain viz for hidden-info games
  //    (fully-public games skip this)
  viz::reveal_slot(state, "drawn_card", {});        // reveal to all
  viz::reveal_slot_to(state, "hand", {target}, actor); // peek
  viz::reset_to_base(state, "hand", schema(), {p}); // re-hide after swap
  viz::swap_slot_owned(state, "hand", p1, p2);      // swap + re-reveal

  // 3. Hidden draws from rng (deck pulls, bag draws)
  if (!s.is_terminal()) {
    int drawn = draw_from_deck(s, rng);
    s.hand[next_player] = drawn;
  }

  // 4. Advance turn / check terminal
  s.current_player_ = next_player(s);
  check_terminal(s);
}
```

### Critical rules

- **Returns void** -- no UndoToken. MCTS/selfplay discard the state after use.
- **NO undo_stack pushes** -- do_action_fast is on the MCTS hot path (millions of calls). Pushing undo records for an undo that never happens is pure waste.
- **rng is caller-owned** -- the state holds no RNG. The runner/session owns one and threads it through.

## Viz maintenance primitives

Rules are the SOLE game-side writer of `state.viz_` (golden standard I1). Available primitives:

| Primitive | Signature | Effect |
|-----------|-----------|--------|
| `reveal_slot` | `(state, "field", {idx...})` | Flip viz to all-1 for all viewers |
| `reveal_slot_to` | `(state, "field", {idx...}, viewer)` | Flip viz ON for one viewer only |
| `reset_to_base` | `(state, "field", schema, {idx...})` | Restore to schema's base_viz |
| `swap_slot` | `(state, "field", {idx_a}, {idx_b})` | Swap viewer-axis rows between two slots |
| `swap_slot_owned` | `(state, "field", owner_a, owner_b)` | Swap + auto-reveal to new owners |

For scalar fields (data rank 0), pass empty `idx` `{}`.

### Viz maintenance examples

```cpp
// Love Letter: Priest peek -- target's hand visible only to actor
viz::reveal_slot_to(state, "hand", {target}, actor);

// Love Letter: King swap -- swap hand contents + viz follows content
viz::swap_slot_owned(state, "hand", actor, target);

// Coup: lose influence -- reveal to all
viz::reveal_slot(state, "influence", {player, slot});

// Coup: replace influence after exchange -- re-hide replaced slot
viz::reset_to_base(state, "influence", schema(), {player, slot});

// Splendor: drawn card revealed to all
viz::reveal_slot(state, "drawn_card", {});
```

## do_action_deterministic_impl + undo (tail solver only)

```cpp
UndoToken MyGameRules::do_action_deterministic_impl(
    IGameState& state, ActionId action) const {
  auto& s = checked_cast<MyGameState>(state);

  // Push undo record BEFORE mutating
  UndoRecord rec{};
  rec.action = action;
  rec.prev_player = s.current_player_;
  rec.prev_winner = s.winner_;
  rec.prev_terminal = s.terminal;
  rec.prev_scores = s.scores;
  rec.prev_viz = s.viz_;  // snapshot viz if rules mutate it
  s.undo_stack.push_back(rec);

  // Apply action (same logic as fast, but NO rng draws)
  // Games with hidden draws MUST suppress them here
  // (e.g. Splendor uses forced_draw_override = -2 sentinel)
  apply_action_logic(s, action);

  return UndoToken{};
}

void MyGameRules::undo_action_impl(
    IGameState& state, const UndoToken& token) const {
  auto& s = checked_cast<MyGameState>(state);
  if (s.undo_stack.empty()) return;
  UndoRecord rec = s.undo_stack.back();
  s.undo_stack.pop_back();

  // Restore everything from the snapshot
  s.board[rec.action] = kEmpty;
  s.current_player_ = rec.prev_player;
  s.winner_ = rec.prev_winner;
  s.terminal = rec.prev_terminal;
  s.scores = rec.prev_scores;
  s.viz_ = rec.prev_viz;

  (void)token;
}
```

### do_action_fast vs do_action_deterministic

| Aspect | `do_action_fast_impl` | `do_action_deterministic_impl` |
|--------|----------------------|-------------------------------|
| Caller | MCTS, selfplay, arena, web | Tail solver only |
| RNG | Caller-owned `std::mt19937_64&` | No RNG -- suppress hidden draws |
| Undo support | **NO** -- never pushes undo_stack | **YES** -- pushes undo_stack |
| Returns | void | UndoToken (paired with undo_action) |
| Hot path | Yes (millions of calls) | No (alpha-beta depth-limited) |

Only games that register a tail solver need a production-quality deterministic/undo path. If a game has physical randomness and registers `tail_solver`, its deterministic path must suppress or model that randomness without sampling hidden truth.

Hidden-information observer sessions do not advance by replaying `do_action_fast_impl` from sampled private bytes. They receive the action plus the public-event protocol: `public_state_applier` overwrites the visible snapshot and the belief tracker consumes events. Fully-public games may replay `do_action_fast` directly because there is no hidden state to leak.

## Common pitfalls

1. **BUG-001**: pushing undo records in `do_action_fast_impl` wastes memory and slows MCTS. Only `do_action_deterministic_impl` pushes.

2. **Never call step_count_ from game code**. The framework wrappers handle it. Game code cannot see it.

3. **validate_action must not mutate state**. It takes `const IGameState&`.

4. **legal_actions must return empty vector (not throw) when terminal**. MCTS checks `is_terminal()` first, but the contract is still "empty = terminal."

5. **Legal-action scope must match the perspective hash**. If two states have the same `state_hash_for_perspective(state.current_player())`, they must expose the same `legal_actions` set for that acting player. ISMCTS reuses DAG nodes by this hash and fails loudly on a reused edge that is illegal in the current sampled world.

6. **Hidden draws in deterministic path**: games with stochastic draws (deck, bag) MUST override `do_action_deterministic_impl` to suppress them. The default forwards to `do_action_fast_impl` with a throwaway RNG, which is WRONG for hidden-info games -- it "peeks" at the deck.

7. **Viz writes belong exclusively in do_action_fast_impl** (rules are the sole game-side viz writer). Never write viz from the encoder, tracker, or any other component.
