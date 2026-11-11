# AI Pipeline, Advanced Operations & Replay

## 1. AI Pipeline — How Decision + Analysis Works

Every human-to-play position runs MCTS **once** (precompute). The result serves two purposes:
- Smart hint (if requested)
- Drop-score analysis (after human plays)

### Pipeline state machine

```
[AI move done] → [done] → start precompute (background)
     ↓ human plays
[queued] — submitted to thread pool, waiting for worker
     ↓
[analyzing] — wait for precompute, read action_values[chosen], compute drop
     ↓
[ai_thinking] — AI's MCTS decision
     ↓
[done] — trigger next precompute, loop

Error path: any exception → [error] (with pipe["error"] message)
```

`[analyzing]` is a cache read, NOT a new MCTS. Only when precompute failed (cancelled, timeout) does it run inline MCTS as fallback. Non-expert difficulty skips analyzing + precompute entirely (uses `schedule_pipeline_ai_only` which goes straight from `[queued]` to `[ai_thinking]`).

### Multi-step AI turns: backend walks ONE action, frontend loops

A pipeline worker invocation computes and commits **exactly one** AI action.
For games where one player takes multiple sub-actions in a row (intra-turn
multi-action like Jaipur's exchange phases, or Harmonies' select-slot →
place-tokens → take-card sequence), **do not loop in the worker**. The
frontend `app.js` `pollPipeline` loop already drives this:

```
[ai action arrives] → animate → render → if current_player still ∈ ai_players:
    POST /ai-action again → next worker invocation → next single action → ...
```

Looping inside the worker collapses every sub-action into a single transition
that overwrites `pipe["ai_action"]` / `pipe["ai_action_info"]`. The frontend
only ever sees the LAST sub-action; every intermediate animation is lost. If
a multi-action AI turn appears to "stall" or "skip steps", the bug is almost
always on the **game-rules side** (`do_action_fast` not advancing
`current_player`, an exchange phase machine not transitioning, a
`legal_actions` deadlock) — not in the pipeline. Fix it at the rule level.

Historic regression: Jaipur exchange "卡住" was a `do_action_fast` bug in the
camel-count phase that left `current_player` unchanged forever; "fixing" it
by adding a worker-side loop hid the rules bug AND silently broke
Harmonies' multi-step AI animations, because the frontend's per-step
animation hook only fires once per `/ai-action` round-trip.

### Drop-score calculation

From the precompute MCTS result, each root edge has N-dim Q values:
- `best_wr = (root_values[human_player] + 1) / 2`
- `actual_wr = (action_values[chosen][human_player] + 1) / 2`
- `drop = (best_wr - actual_wr) * 100`

Thresholds: ≥5% = mistake (warn), ≥10% = blunder.

---

## 2. Advanced Operations (Framework-Provided)

All handled by the framework — game code does NOT implement these:

| Operation | Mechanism |
|-----------|-----------|
| **Undo** | Loops `step-back`, skipping AI responses and consecutive same-player moves (via `last_actor`). Cancels pipeline, rebuilds session, triggers new precompute. |
| **Play for opponent** | Steps back to target AI's action start, then human plays in their seat. `config.disableForce: true` hides this for hidden-info games. |
| **Smart hint** | Reuses precompute result. If precompute still running, waits up to 8s. Does not trigger additional MCTS. |
| **Win-rate toggle** | Controls info-panel win-rate pill + mistake markers. Default state from `config.showWinrateDefault` (must be `false` for hidden-info games — values are based on root values which include true hidden state). |

### Interaction flow

**Human plays**: cancel pipeline → apply + render → wait for precompute analysis → show drop → AI thinks → AI plays → render → start next precompute

**Undo**: cancel pipeline → loop step-back (skip consecutive same-actor) → render → start precompute

**Hint**: if precompute ready → show immediately. If still running → show "analyzing..." → auto-update when done.

---

## 3. C++ Components for Web

Two C++ functions registered on GameBundle that the web frontend consumes:

### StateSerializer
`(const IGameState&) -> AnyMap` — serializes game state to JSON for the frontend.

The frontend receives this via `GET /api/games/{session_id}` under
`gameState.state`. Framework fields (`current_player`, `legal_actions`,
`winner`, `human_player`, etc.) stay at the top level; game-specific serializer
fields are not merged into the top-level object.

### ActionDescriptor
`(ActionId) -> AnyMap` — translates an action ID to a human-readable description.

Used by `formatOpponentMove` and `formatSuggestedMove` in the frontend.

---

## 4. Framework-Provided UI

`createApp` injects these automatically — game code does NOT implement them:

| Feature | Description |
|---------|-------------|
| Zoom controls | +/- buttons, persisted to localStorage |
| Sidebar collapse | Toggle button, persisted |
| Info panel | Turn indicator, opponent move pill, win-rate pill, suggestion pill, custom extensions |
| Replay viewer | Auto-shown after game ends (expert difficulty), per-frame drop-score |
| i18n | `t(key)` with auto language detection |

### Allowed imports (game JS should only import these)

| Module | Exports | Use |
|--------|---------|-----|
| `general/app.js` | `createApp(config)` | **Only** entry point |
| `general/i18n.js` | `t`, `tList`, `register`, `getLang`, `setLang` | Translation |
| `general/api.js` | `apiGet`, `apiPost`, `API_BASE` | Only for non-standard API calls (rare) |

**Do NOT import** `sidebar`, `modal`, `info_panel`, `replay`, `animate`, `layout`, `pipeline` — these are framework-internal. Influence info panel content via `config.extensions` or `config.formatOpponentMove`.

---

## 5. Replay Format

Unified JSON format for online games and evaluation scripts:

```json
{
  "game_id": "quoridor",
  "seed": 42,
  "players": { "player_0": {"name": "model", "type": "model"}, ... },
  "result": {"winner": 0, "draw": false, "total_plies": 75},
  "action_history": [42, 130, 5, ...],
  "frames": [...]
}
```

- `frames` optional — online games include full frames with analysis; eval scripts only include `action_history`, frontend builds frames on load via `POST /api/replay/build`
- Expert difficulty only

---

## 6. Internal API Endpoints

Handled by `createApp` internally — game code does not call these directly:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/games` | POST | Create session |
| `/api/games/{id}` | GET | Get state |
| `/api/games/{id}/action` | POST | Submit action |
| `/api/games/{id}/ai-action` | POST | Trigger AI move |
| `/api/games/{id}/ai-hint` | POST | Get hint |
| `/api/games/{id}/step-back` | POST | Undo |
| `/api/games/{id}/pipeline` | GET | Poll AI status |
| `/api/games/{id}/replay` | GET | Get replay frames |
| `/api/replay/build` | POST | Build frames from action_history |
