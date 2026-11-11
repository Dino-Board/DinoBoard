# Game Registration and Build Reference

## File layout

```
games/<game_id>/
  <game_id>_state.h           # State class declaration
  <game_id>_state.cpp         # schema(), reset_with_seed, hash/read/write/mask dispatchers
  <game_id>_rules.h           # Rules class declaration
  <game_id>_rules.cpp         # legal_actions, do_action_fast_impl, undo
  <game_id>_net_adapter.h     # Encoder class declaration
  <game_id>_net_adapter.cpp   # encode_features, action_space, feature_dim
  <game_id>_register.cpp      # GameBundle factory + GameRegistrar static init
  CMakeLists.txt              # CMake build target for this game
  heuristic/                  # Optional: named heuristics
    <game_id>_rule_heuristic.h
    <game_id>_rule_heuristic.cpp
  config/
    game.json                 # Game-level props, per-variant dims
    train.json                # Network shape, training hyperparams, heuristic_guidance
    mcts.json                 # MCTS profiles (selfplay/arena/eval/web_expert/web_casual/analysis)
    web.json                  # Web frontend configuration
    heuristic.json            # Named heuristic pool + selectors
    belief.json               # Optional: belief network config
  web/
    <game_id>.js              # Web frontend (createApp)
  model/                      # ONNX models (gitignored, produced by training)
```

## manifest.json

Single source of truth for which games are live. Located at `games/manifest.json`.

```json
{
  "games": [
    {
      "id": "mygame",
      "enabled": true,
      "framework_whitelist": true,
      "capabilities": ["hidden_info", "snapshot", "tracker"],
      "sources": [
        "mygame_state.cpp",
        "mygame_rules.cpp",
        "mygame_net_adapter.cpp",
        "mygame_register.cpp"
      ]
    }
  ]
}
```

### Field reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | string | required | Game identifier, matches directory name under `games/` |
| `enabled` | bool | true | false skips build entirely; sources stay on disk |
| `framework_whitelist` | bool | false | true = game must satisfy core framework invariants |
| `capabilities` | string[] | [] | Tags consumed by framework test matrix |
| `sources` | string[] | required | Relative to `games/<id>/`, consumed by `setup.py`; keep the game's CMake source list in sync |

### Capability tags

| Tag | Meaning | Triggers |
|-----|---------|----------|
| `hidden_info` | Game has perspective-private slots (viz != all_public) | `test_encoder_respects_hash_scope`, `test_public_hash_excludes_internal_rng` |
| `snapshot` | Registers `public_state_applier` | `test_public_snapshot_round_trip` |
| `tracker` | Registers `belief_tracker` | Tracker-consistency tests |

## Build integration

`setup.py` reads `manifest.json` and compiles each enabled game's `sources` list directly.

The top-level `CMakeLists.txt` also reads `manifest.json`, but only to add each enabled `games/<id>/` subdirectory. Each game still owns a small `games/<id>/CMakeLists.txt` with the same source list, so adding or removing a CMake-built file currently requires updating both manifest `sources` and that game's CMake target.

```cmake
# top-level CMakeLists.txt:
# manifest.json controls which game subdirectories are added.

# games/<id>/CMakeLists.txt:
# add_library(game_<id> STATIC <same source list as manifest>)
```

No manual top-level CMake edit is needed when adding a game, but the new game directory must include its own `CMakeLists.txt`.

## Build verification

```bash
# Full build
python setup.py build_ext --inplace

# Verify registration
python -c "import dinoboard_engine as engine; print(engine.available_games())"
# Should include your game_id

# Run framework tests for your game
pytest tests/framework/ -k "mygame" -v
```

## Config files

### game.json (minimal)

```json
{
  "game_id": "mygame",
  "num_players": 2,
  "variants": {
    "mygame": { "num_players": 2 },
    "mygame_3p": { "num_players": 3 }
  }
}
```

### train.json (minimal)

```json
{
  "network": {
    "type": "mlp",
    "hidden_layers": [256, 256, 256],
    "activation": "relu"
  },
  "training": {
    "learning_rate": 0.001,
    "batch_size": 256
  },
  "mcts": {
    "selfplay": {
      "simulations": 200
    }
  }
}
```

## Invariants cheatsheet

| # | Rule | Enforced by |
|---|------|-------------|
| 1 | Every state field in DAG node identity is a schema slot | `test_public_hash_excludes_internal_rng` |
| 2 | Rules are the sole game-side viz writer | `test_rules_sole_viz_writer` |
| 3 | RNG is not on state, not in hash, not on wire | `test_public_hash_excludes_internal_rng` |
| 4 | Encoder input is always `const IGameState& masked_state` | `test_encoder_respects_hash_scope` |
| 5 | Tracker observe path reads events only; init may write bootstrap payload into session state | Compile-time + API/session tests |
| 6 | `randomize_unseen` called only at MCTS sim entry on clone | Framework code structure |
| 7 | `step_count_` managed by framework, never by game code | `friend class IGameRules` |
| 8 | `state_hash_for_perspective` folds step_count automatically | Framework `schema_hash.h` |
| 9 | Snapshot round-trip preserves public hash | `test_public_snapshot_round_trip` |
| 10 | Same acting-player perspective hash implies same legal action set | MCTS runtime legal-action mismatch check |
| 11 | No off-schema hash escape hatch (`hash_extra_state_fields` removed) | Framework-level removal |
