# DinoBoard

**中文版:** [README_CN.md](README_CN.md)

> **Drop a rulebook. Ship a superhuman AI.**
>
> Tell an LLM "add Azul" in one sentence — it reads the rulebook,
> writes the C++ rules engine, runs the full test suite, trains an
> AI, and generates the web frontend. You just sign off.

An AlphaZero-style framework for **general board games**, turning
"rule engine + AI decision + web play" into an end-to-end loop
that stays trustworthy under LLM-assisted development.

---

## Core mechanisms

Three mechanisms keep the rules engine and AI decision path
trustworthy even under vibe coding:

1. **Per-field visibility tagging** — each game declares a per-slot
   "who can see this" tag. The framework walks this manifest once to
   produce a **masked observation view** shared by snapshot, hash,
   and encoder. Hidden slots structurally read as a placeholder —
   truth can never leak through.

2. **Physical GT/AI separation** — ground truth advances the game;
   per-player AI sessions only see the message stream. The AI has no
   interface to obtain a truth pointer. Structural safety, not
   contractual.

3. **Mandatory web frontend** — training metrics can look fine while
   the AI throws a winning move. A human spots it within 30 seconds
   of play.

See [ALGORITHM_OVERVIEW.md](ALGORITHM_OVERVIEW.md) for the full
algorithm walkthrough.

---

## Games

| Game | Type | 2p | 3p | 4p |
|------|------|----|----|-----|
| TicTacToe | perfect info | 强/✓ | — | — |
| Quoridor | perfect info | 强/✓ | — | — |
| Azul | random | 中等/✓ | 中等/✗ | 中等/✗ |
| Splendor | random + light hidden info | 中等/✓ | 中等/✗ | 中等/✗ |
| Love Letter | random + hidden info | 随机/✓ | 随机/✓ | 随机/✓ |
| Coup | bluff | 随机/✓ | 随机/✗ | 随机/✓ |
| 6 nimmt! | simultaneous + hidden info | 强/✓ | 强/✗ | 强/✗ |
| Liar's Dice | bluff | 中等/✓ | 中等/✓ | 中等/✓ |
| Harmonies | random | 中等/✗ | 中等/✗ | 中等/✗ |
| Jaipur | random + hidden info | 中等/✓ | — | — |
| Carcassonne | random + hidden info | 中等/✗ | — | — |
| Maskmen | hidden info | — | 随机/✓ | 随机/✗ |
| Ark Nova | random + hidden info | 未评估/✗ | — | — |

> Cell format: **heuristic strength / network**. ✓ = trained network available, ✗ = no trained network, — = variant not supported.
>
> Token-transformer networks: Ark Nova by default; Splendor, Carcassonne and Harmonies via `games/<game>/config/train_transformer*.json`.
> 2p web games of these four run the transformer variant (`web_variants` in `game.json`): trained models for Splendor and Carcassonne, randomly initialised ones for Harmonies and Ark Nova.

---

## Quick start

### Requirements

- **C++17 compiler** — Mac: `xcode-select --install`; Linux:
  `apt install build-essential`
- **Python >= 3.9** with `pybind11` and `torch`
- **ONNX Runtime** — Mac: `brew install onnxruntime`; Linux/Windows:
  bundled in the repo or download from
  [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases)

### Build

```bash
python3 -m pip install -e .
python3 -c "import dinoboard_engine; print(dinoboard_engine.available_games())"
```

### Test

```bash
python3 setup.py build_ext --inplace
python3 scripts/run_tests.py -j 8
```

`scripts/run_tests.py` runs pytest files in parallel without requiring
`pytest-xdist`, with isolated per-worker ONNX model caches. Use
`python3 scripts/run_tests.py --serial` when reproducing a file-order issue.

### Train

```bash
GAME=splendor
RUN_DIR=runs/${GAME}_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR"

# CPU (adjust --workers to your core count)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
nohup python3 -m training.cli --game "$GAME" --output "$RUN_DIR" \
    --workers 4 > "$RUN_DIR/train.log" 2>&1 &

# GPU/CoreML: add --providers cuda,cpu or --providers coreml,cpu
# Token transformer on several GPUs: --game splendor_tok \
#   --train-config games/splendor/config/train_transformer_kata_guided.json \
#   --providers cuda,cpu --devices 0,1
```

### Web play

```bash
cd platform && python -m uvicorn app:app --host 0.0.0.0 --port 8000
open http://localhost:8000
```

### Ark Nova on ark-nova-ai's board

DinoBoard has no Ark Nova page of its own. `scripts/arknova_board.py` starts the server of
[ark-nova-ai](https://github.com/collinfeng/ark-nova-ai) (MIT, a separate checkout) with two
opponents on Map 0: `dinoboard`, DinoBoard's token transformer with MCTS
(`games/arknova/model/arknova_2p.onnx`, randomly initialised for now; `--model` picks another), and
ark-nova-ai's `beam_3x8` search.

```bash
python scripts/arknova_board.py --ref ~/ref/ark-nova-ai   # [--model m.onnx] [--profile web_casual]
# the page: cd ~/ref/ark-nova-ai/web && pnpm install && pnpm build, then open http://127.0.0.1:8765
```

### AI API

```
POST /ai/sessions                  → create session
POST /ai/sessions/{id}/observe     → feed public events
POST /ai/sessions/{id}/decide      → get AI action
DELETE /ai/sessions/{id}           → end session
```

Callers only need action IDs and public events — no shared state code
or embedded C++ engine.

---

## Adding a new game

1. Mirror the closest existing game's implementation
2. Copy `tests/<closest>/test_checklist.py`, change `GAME = "..."`
3. Iterate on test failures — `pytest tests/<new_game>/` all-green
   means "game is ready"
4. `python -m training.cli --game <id>` kicks off training
5. Verify in the web UI

---

## Scope boundaries

DinoBoard is built on AlphaZero / ISMCTS. These don't fit:

1. **Precise mixed-strategy equilibria** (Texas Hold'em) — approximate mixing (Coup / Liar's Dice bluff) is in scope
2. **Combinatorial action explosion** (DouDizhu 27k+, MTG deckbuilding)
3. **Cooperative games** (Hanabi) — self-play converges to private conventions
4. **Single-player games** (Solitaire, 2048) — no opponent = no training signal
5. **Closed-eye phases** (Werewolf) — viz is a god-view function

---

## Docs

- [FEATURES_OVERVIEW.md](FEATURES_OVERVIEW.md) — capabilities at a glance
- [ALGORITHM_OVERVIEW.md](ALGORITHM_OVERVIEW.md) — algorithm walkthrough
- [docs/FRAMEWORK_DESIGN_RATIONALE.md](docs/FRAMEWORK_DESIGN_RATIONALE.md) — design motivation
- `docs/games/<game>_api.md` — per-game API integration docs
