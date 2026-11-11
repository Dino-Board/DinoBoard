# DinoBoard Onboarding Guide

## What This Is

AlphaZero-style self-play framework. C++ engine (MCTS + game logic) with Python training glue.
Games listed in `games/manifest.json` (tictactoe, quoridor, azul, splendor, loveletter, coup, 6nimmit, liarsdice).
Stack: C++ engine (pybind11) -> Python training pipeline -> ONNX models -> web frontend.

## 5 Hard Rules

1. **No silent degradation** — errors must surface immediately. Never write fallback logic that masks failures: no `catch(...){return false}`, no `dict.get(key, default)` (use `dict["key"]`), no default return values that make the caller think it succeeded. If something is broken, it must crash, not silently produce wrong results.
2. **No Python reimplementation** — all game logic / search is C++. If you're writing Python that "does what the C++ does but slower", stop.
3. **AI cannot read hidden info** — the framework structurally prevents the AI decision path from seeing other players' private data. The interfaces are designed so there is no way to pass truth state to the AI, not even by accident. Details in ALGORITHM_OVERVIEW.md.
4. **GIL release** — any pybind11 binding doing meaningful C++ work (MCTS, ONNX, batch encode) MUST `py::gil_scoped_release`.
5. **Training config-driven** — `game.json` + `mcts.json` + `train.json` + `heuristic.json` + optional `belief.json`. No code changes for training runs. Self-play loss not decreasing is normal (opponent also improves).

## ISMCTS Core Properties (Do Not Break)

1. Root determinization — no chance nodes; each sim samples a complete world via tracker.randomize_unseen.
2. DAG via step_count — nodes share when hash matches; step always increases parent→child, structurally acyclic.
3. UCT2 — UCB numerator uses the edge-specific visit count, not the DAG node's global N.
4. Public hash excludes RNG — session_rng seed does not affect state_hash_for_perspective.

## Task Routing

| Task | Skill |
|------|-------|
| New game state/rules/build | /game-rules |
| Encoder / tracker / registration | /game-ai |
| Training config | /game-config |
| Training issues / hyperparams | /training |
| Evaluate trained models | /eval-model |
| Web frontend | /web-dev |
| Web animations | /web-animation |
| Tests | /game-test |
| Modify MCTS / framework internals | /framework-dev |
| Update docs / skills | /write-docs, /write-skill |

## New Game Workflow

1. Implement state + rules + schema in C++ (`games/<id>/`)
2. Implement encoder + tracker + GameBundle registration
3. Add entry to `games/manifest.json` with capabilities
4. Create config files: `game.json`, `mcts.json`, `train.json`, `web.json`, `heuristic.json`
5. Build + run tests until green (`pytest tests/<id>/`)
6. Build web frontend (`games/<id>/web/`)
7. Write API integration doc (`docs/games/<id>_api.md` — action encoding + events + snapshot fields)
8. Train: `python3 -m training.cli --game <id> --output runs/<dir> --workers N`
9. Copy trained ONNX to `games/<id>/model/`, verify in web UI

## Baseline Check Commands

```bash
# Build
python3 setup.py build_ext --inplace

# Run tests
python3 scripts/run_tests.py -j 8

# Start web server
cd platform && python -m uvicorn app:app --host 0.0.0.0 --port 8000

# Run training — check train.json for selfplay_concurrency first:
#   CPU mode (no selfplay_concurrency or enabled=false):
#     --workers = physical core count minus 2 (reserve for main + OS)
#   Batched mode (selfplay_concurrency.enabled=true):
#     --workers = episodes_per_step (e.g. 100), add --providers coreml,cpu or cuda,cpu
python3 -m training.cli --game <game_id> --output runs/<run_dir> --workers <N>
```

## High-Frequency Failure Modes

1. **Holding GIL or global lock during heavy C++ work** — pybind11 bindings doing MCTS/ONNX MUST `py::gil_scoped_release`. Without it, the web server freezes for seconds per AI move.
2. **dict.get() with default** — hides missing fields. Use `dict["key"]`.
3. **catch(...){return false}** — banned. Re-throw with context.
4. **evaluate() returning false on error** — must throw. A silent `return false` once caused weeks of training on a uniform-random policy without anyone noticing.
5. **Python fallback for C++ logic** — never. Expose via pybind11 instead.
6. **Empty model_path in selfplay/arena/eval** — always an error. No uniform-policy fallback exists. A plain `GameSession(model_path="")` is still allowed for state tests and heuristic-only paths.
7. **Forgetting manifest.json entry** — compiles fine, but game invisible to framework.

## Framework Non-Goals

1. **Exact mixed equilibria** (Texas Hold'em) — AlphaZero trains deterministic policies; cannot converge to exact Nash. Approximate mixing (Coup / Liar's Dice bluffs) is within scope.
2. **Action space explosion** (DouDizhu, Magic: The Gathering) — fixed action_space flat policy head cannot handle; would need hierarchical / autoregressive decoding.
3. **Cooperative games** (Hanabi) — self-play converges to private conventions that only work against the training partner; zero-shot coordination problem.
4. **Single-player games** (solitaire, 2048) — no opponent means no win/loss signal; training cannot bootstrap.
5. **Closed-eye phases** (Werewolf night) — viz is a god-view function; cannot represent "not knowing whether an opponent gained information" (higher-order uncertainty).
