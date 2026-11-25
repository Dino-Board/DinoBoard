# External AI Integration (GameSession)

## Overview

An external platform drives a DinoBoard AI through the pybind11 `GameSession` (in
`bindings/py_engine.cpp`) in observation mode. The AI holds no ground truth -- it receives
observations (a public snapshot at the start, then action_id + events + public_snapshot per action)
and returns decisions. The bridges that play on other sites (BGA, hullqin) rebuild a session from
the table's public snapshot for every decision.

## Integration flow

```python
import dinoboard_engine as engine

session = engine.GameSession("splendor", seed=42, model_path="games/splendor/model/splendor_2p.onnx")
# Perspective-specific facts at the start (own hand, ...): the public snapshot plus the tracker's
# bootstrap. A ground-truth session produces it with extract_initial_observation(seat); a bridge
# builds the same two parts from the table it sees.
session.apply_initial_observation(0, obs)

# Feed every action, the opponents' and the AI's own
session.apply_observation(action=5, events=[], public_snapshot=snap)

# Get the AI decision (does not advance the session)
result = session.get_ai_action(simulations=800, temperature=0.0)
action = result["action"]

# Apply to truth externally, then feed the AI's own move back
session.apply_observation(action=action, events=[], public_snapshot=next_snap)

# Inspect belief state (tracker games only)
belief = session.get_belief_snapshot()
```

- `apply_initial_observation` is **required** for snapshot-path games (Azul, Splendor, Love
  Letter, Coup, 6 nimmt!, Liar's Dice). Without it, the session's independently-seeded hidden state
  diverges from truth. Fully-public no-snapshot games (TicTacToe, Quoridor) take action ids only;
  `game_metadata(game_id)["has_public_state_applier"]` tells the two apart.
- `events` are required for tracker games, `[]` otherwise; `public_snapshot` is required for
  snapshot-path games.

## Invariants

1. **No full truth state crosses the boundary.** The session holds an independent IGameState seeded with its own RNG. Public fields arrive through `public_snapshot`, perspective-private bootstrap arrives through the initial observation, and hidden opponent state is sampled only at MCTS sim entry.

2. **The AI cannot cheat.** The tracker never receives truth state through the observation path. `init` consumes a GT-packed `AnyMap` bootstrap on the independent session state, `observe_public_event` takes only the event stream, and `randomize_unseen` writes TO cloned sim state (fills viz=0 slots) instead of reading hidden truth.

3. **Deciding is non-mutating.** `get_ai_action` does not advance the session; it advances when the caller feeds the action back via `apply_observation`. This lets the caller inspect/validate the AI's choice before committing.

4. **Every action is observed.** Both opponent actions AND the AI's own actions. The session learns about its own moves the same way it learns about opponents'.

5. **Seed independence.** Two sessions with different seeds but the same observation stream produce identical `state_hash_for_perspective` at every ply. Enforced by `test_public_hash_excludes_internal_rng`.

## Visibility layers

| Layer | What it sees | Source |
|-------|-------------|--------|
| Ground truth | Everything | Rules engine, full state |
| AI session | Public fields + own private | Snapshots + tracker init payload |
| AI MCTS sim | Public + sampled hidden | `randomize_unseen` on cloned state |
| Network encoder | Masked state (kPlaceholder for hidden) | `make_masked_state` |

## Key methods

| Method | Description |
|--------|-------------|
| `get_ai_action(simulations, temperature, ...)` | Run MCTS and return the chosen action_id (`time_limit_ms` searches for a wall-clock budget) |
| `get_heuristic_action(name, temperature)` | Run a named heuristic and return action_id |
| `configure_tail_solve(enabled, depth_limit, node_budget, ...)` | Set tail solve parameters for subsequent decisions |
| `apply_ai_action(simulations, temperature)` | Convenience method: pick an AI action with MCTS, apply it to the local session, and return the result |
| `get_belief_snapshot()` | Return the tracker's `serialize()` output as a dict |

## Notes

- All heavy C++ work (MCTS, ONNX inference) releases the GIL via
  `py::gil_scoped_release` so the web server's event loop is not blocked.
- The session owns one tracker per perspective, matching the selfplay stack.
- `get_ai_action` is non-mutating; `apply_ai_action` does mutate the `GameSession`.
- `get_belief_snapshot()` returns an empty dict for games without a registered
  tracker, or when the tracker serializes no explicit memory.
