# REST AI API Reference

## Overview

Observation-only REST API for driving an AI player externally. The AI holds no ground truth -- it receives observations (action_id + events + public_snapshot) and returns decisions.

Base path: `/ai/sessions`

## Endpoints

### 1. POST /ai/sessions -- Create session

```json
// Request
{
  "game_id": "splendor",
  "seed": 12345,           // optional, server picks random if omitted
  "my_seat": 0,            // 0-indexed player seat for the AI
  "initial_observation": {  // required for snapshot-path games
    // perspective-specific facts at game start (own hand, etc.)
    // obtained via GameSession.extract_initial_observation(seat)
  }
}

// Response
{
  "session_id": "abc-123",
  "game_id": "splendor",
  "num_players": 2,
  "my_seat": 0,
  "current_player": 0,
  "is_terminal": false
}
```

- `initial_observation` is **required** for snapshot-path games (Azul, Splendor, Love Letter, Coup, 6 nimmt!, Liar's Dice). Without it, the AI session's independently-seeded hidden state diverges from truth.
- **Must be omitted** for fully-public no-snapshot games (TicTacToe, Quoridor). Server rejects mismatches with HTTP 400.

### 2. POST /ai/sessions/{id}/observe -- Record an action

```json
// Request
{
  "action_id": 42,
  "events": [               // required for tracker games, [] otherwise
    {
      "kind": "card_revealed",
      "payload": {"card": 3, "player": 1}
    }
  ],
  "public_snapshot": {       // required for snapshot-path games, {} otherwise
    // truth-side public fields after this action
  }
}

// Response
{
  "actions_observed": 5,
  "current_player": 1,
  "is_terminal": false
}
```

**Important**: `/decide` is non-mutating. It returns the chosen action_id but does NOT advance the session. The caller must:
1. Call `/decide` to get the AI's chosen action
2. Apply the action to ground truth
3. Call `/observe` with the action + events + snapshot (including the AI's own moves)

### 3. POST /ai/sessions/{id}/decide -- Ask AI for next action

```json
// No request body

// Response
{
  "action_id": 7,
  "action_info": {
    "type": "buy",
    "card": 12,
    "card_name": "Ruby Mine"
  },
  "stats": {
    "simulations": 800,
    "root_value": 0.65
  },
  "current_player": 0,
  "is_terminal": false
}
```

- Returns HTTP 409 if it's not the AI's turn or the game is terminal.
- AI strength (simulations, temperature, opponent_selection) is server-controlled via the `web_expert` MCTS profile. Not exposed as wire fields.

### 4. GET /ai/sessions/{id} -- Session status

```json
// Response
{
  "session_id": "abc-123",
  "closed": false,
  "game_id": "splendor",
  "num_players": 2,
  "my_seat": 0,
  "is_terminal": false,
  "current_player": 1,
  "winner": null,
  "actions_observed": 10
}
```

### 5. DELETE /ai/sessions/{id} -- Close session

```json
// Response
{"closed": true}
```

## Integration flow

```
Caller (game server / test harness)         AI API
  |                                           |
  |-- POST /ai/sessions {game_id, seat, obs} ->|  Create
  |<- session_id --------------------------------|
  |                                           |
  |  [Game loop]                              |
  |  GT advances: action -> events + snapshot |
  |-- POST /ai/sessions/{id}/observe --------->|  Feed observation
  |<- actions_observed, current_player --------|
  |                                           |
  |  [AI's turn]                              |
  |-- POST /ai/sessions/{id}/decide ---------->|  Get AI action
  |<- action_id, action_info ------------------|
  |                                           |
  |  Apply action to GT, extract events       |
  |-- POST /ai/sessions/{id}/observe --------->|  Feed AI's own move back
  |<- ... -------------------------------------|
  |                                           |
  |  [Repeat until terminal]                  |
  |                                           |
  |-- DELETE /ai/sessions/{id} --------------->|  Cleanup
  |<- closed ----------------------------------|
```

## Invariants

1. **No full truth state crosses the API boundary.** The session holds an independent IGameState seeded with its own RNG. Public fields arrive through `public_snapshot`, perspective-private bootstrap arrives through `initial_observation`, and hidden opponent state is sampled only at MCTS sim entry.

2. **The AI cannot cheat.** The tracker never receives truth state through the observe path. `init` consumes a GT-packed `AnyMap` bootstrap on the independent session state, `observe_public_event` takes only the event stream, and `randomize_unseen` writes TO cloned sim state (fills viz=0 slots) instead of reading hidden truth.

3. **Decide is non-mutating.** The session state does not advance until the caller feeds the action back via `/observe`. This lets the caller inspect/validate the AI's choice before committing.

4. **Every action goes through observe.** Both opponent actions AND the AI's own actions. The session learns about its own moves the same way it learns about opponents'.

5. **Seed independence.** Two sessions with different seeds but the same observation stream produce identical `state_hash_for_perspective` at every ply. Enforced by `test_public_hash_excludes_internal_rng`.

## Error handling

| HTTP Code | Meaning |
|-----------|---------|
| 400 | Bad request: unknown game_id, illegal observed action, missing initial_observation for snapshot-path game, initial_observation provided for non-snapshot game |
| 404 | Session not found |
| 409 | Conflict: not AI's turn (for decide), game already terminal / closed session |

## Visibility layers

| Layer | What it sees | Source |
|-------|-------------|--------|
| Ground truth | Everything | Rules engine, full state |
| AI session | Public fields + own private | Snapshots + tracker init payload |
| AI MCTS sim | Public + sampled hidden | `randomize_unseen` on cloned state |
| Network encoder | Masked state (kPlaceholder for hidden) | `make_masked_state` |

## Python GameSession Wrapper

The pybind11 `GameSession` class (in `bindings/py_engine.cpp`) wraps the same
AI pipeline for Python-side use (web server, scripts, tests). The REST API is a
thin wrapper around this class (via `platform/ai_service/sessions.py`).

### Key methods

| Method | Description |
|--------|-------------|
| `get_ai_action(simulations, temperature, ...)` | Run MCTS and return the chosen action_id |
| `get_heuristic_action(name, temperature)` | Run a named heuristic and return action_id |
| `configure_tail_solve(enabled, depth_limit, node_budget, ...)` | Set tail solve parameters for subsequent decisions |
| `apply_ai_action(simulations, temperature)` | Convenience method: pick an AI action with MCTS, apply it to the local session, and return the result |
| `get_belief_snapshot()` | Return the tracker's `serialize()` output as a dict |

### Usage pattern

```python
import dinoboard_engine as engine

session = engine.GameSession("splendor", seed=42, model_path="games/splendor/model/splendor_2p.onnx")
session.apply_initial_observation(0, obs)

# Feed opponent action
session.apply_observation(action=5, events=[], public_snapshot=snap)

# Get AI decision
result = session.get_ai_action(simulations=800, temperature=0.0)
action = result["action"]

# Apply to truth externally, then feed the AI's own move back
session.apply_observation(action=action, events=[], public_snapshot=next_snap)

# Inspect belief state (tracker games only)
belief = session.get_belief_snapshot()
```

### Notes

- All heavy C++ work (MCTS, ONNX inference) releases the GIL via
  `py::gil_scoped_release` so the web server's event loop is not blocked.
- The session owns one tracker per perspective, matching the selfplay stack.
- REST `/decide` uses `get_ai_action` and is non-mutating. `apply_ai_action` is
  a local Python convenience method and does mutate the `GameSession`.
- `get_belief_snapshot()` returns an empty dict for games without a registered
  tracker, or when the tracker serializes no explicit memory.
