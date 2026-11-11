# DinoBoard — AI Work Instructions

## Before you start

Use the `/onboarding` skill before any non-trivial task.

## Hard rules

1. **No fallback** — fail immediately with throw; never `catch(...){return false;}`, `dict.get(key, default)`, or swallow errors. Use `dict["key"]`, not `.get()`.
2. **No Python rewrite of C++** — selfplay/eval/arena/MCTS/tail solver are all C++. Expose to Python via pybind11.
3. **AI must not read hidden info** — the framework structurally prevents AI decision paths from seeing other players' private data. There is no interface to pass ground-truth state to the AI. See `ALGORITHM_OVERVIEW.md`.
4. **GIL release** — heavy pybind11 work (MCTS / ONNX) must use `py::gil_scoped_release`.
5. **Training is config-driven** — `game.json` + `mcts.json` + `train.json` + `heuristic.json` + optional `belief.json`. Tune parameters without code changes. Self-play loss not decreasing is normal (opponents also get stronger).

## Task routing (use the corresponding skill)

| Task | Skill |
|------|-------|
| Add new game (state/rules/schema) | `/game-rules` |
| Add new game (encoder/tracker/registration/tests) | `/game-ai` |
| Config files | `/game-config` |
| Test acceptance | `/game-test` |
| Web frontend | `/web-dev` |
| Training hyperparameters / diagnose training | `/training` |
| Modify MCTS / framework | Read `ALGORITHM_OVERVIEW.md` first |

## After code changes

```bash
python3 setup.py build_ext --inplace
python3 scripts/run_tests.py -j 8
```
