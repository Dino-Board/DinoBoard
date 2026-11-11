# Game Configuration Reference

All config lives in `games/<id>/config/`. Six files, each with a distinct scope.

---

## game.json — Game-Level Properties

| Field | Type | Description |
|-------|------|-------------|
| game_id | string | Canonical ID (matches manifest.json) |
| display_name | string | Chinese display name for web UI |
| display_name_en | string | English display name for web UI |
| victory_mode | "winner_takes"\|"loser_pays" | Controls gating threshold variance formula. `winner_takes` (default): 1 winner, N-1 losers. `loser_pays`: 1 loser, N-1 winners. Does NOT affect C++ value model (which reads `winners()`/`losers()` from state). Only consumer: Python training pipeline gating threshold. |
| players.min | int | Minimum player count |
| players.max | int | Maximum player count |
| max_game_plies | int | Maximum plies before forced termination (used by dirichlet_on_first_n_plies sentinel) |
| variants | object | Per-variant dimensions (keyed by variant_id) |
| variants.<v>.feature_dim | int | Input feature dimension for this variant (0 for variable-length token encoders) |
| variants.<v>.action_space | int | Action space size for this variant |
| variants.<v>.num_players | int | Player count for this variant |
| web_variants | object | Optional. Registered variant web games use per player count, e.g. `{"2": "splendor_tok"}`; its model is `games/<base>/model/<variant>_2p.onnx` (platform/model_paths.py). Default `<game_id>_<N>p`, else `<game_id>` |

Variant naming: base `<game_id>` for 2p, `<game_id>_3p`/`<game_id>_4p` for multi.

---

## mcts.json — MCTS Baseline (Flat, No Profile Names)

Shared knobs every profile inherits. All fields are optional (fall back to _BASE_DICT in mcts_profile.py).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| simulations | int | 200 | MCTS simulation count |
| world_cap | int | 0 | 0=every sim its own world; >0 caps K outer determinizations |
| c_puct | float | 1.4 | Exploration constant |
| temperature | float | 0.0 | Action selection temperature |
| temperature_schedule | object | (disabled) | {enabled, initial, final, decay_plies} |
| dirichlet_alpha | float | 0.0 | Root noise alpha (off by default) |
| dirichlet_epsilon | float | 0.0 | Root noise weight |
| dirichlet_on_first_n_plies | int | -1 | -1 = whole game (resolved to max_game_plies) |
| opponent_selection | "puct"\|"prior" | "puct" | Non-root opponent node selection |
| tail_solve_enabled | bool | false | Enable tail solver |
| tail_solve_depth_limit | int | 10 | Tail solver depth |
| tail_solve_node_budget | int | 200000 | Tail solver node cap |
| tail_solve_auxiliary_weight | float | 0.0 | Tail solver aux score weight |
| ai_use_action_filter | bool | false | Enable TrainingActionFilter in AI |
| cover_root_edges | bool | false | Force at least 1 visit on every root child |
| fpu_reduction | float\|null | null | First-play urgency reduction (Leela/KataGo): unvisited children get Q = Q(parent) − r·sqrt(sum of visited priors). null keeps the legacy Q = 0. Splendor uses 0.2 for selfplay and eval |

---

## train.json — Training Configuration

### Top-level blocks:

#### network
| Field | Type | Description |
|-------|------|-------------|
| type | "mlp" \| "token_transformer" | Network type; other values are rejected. |
| hidden_layers | int[] | (mlp) Layer widths, e.g. [512,512,512,512] |
| activation | "relu" | (mlp) Activation function. Only `"relu"` is implemented; other values are rejected. |
| residual | bool | (mlp) Enable residual connections |
| block_size | int | (mlp) Layers per residual block (typically 2) |
| d_model / num_layers / num_heads / ffn_dim | int | (token_transformer) All four required; `d_model % num_heads == 0`. |
| attention | "matmul" \| "fused" | (token_transformer, optional, default "matmul") Exported attention graph only; weights and training are identical. "fused" emits onnxruntime's `com.microsoft::MultiHeadAttention` (long token sets, see `training/performance.md`). |
| relative_position_radius | int ≥ 1 | (token_transformer) Required exactly when the token layout has `token_positions`, rejected otherwise. Attention adds a learned per-head bias per clipped offset (dx, dy) in [-R, R]² between positioned tokens, one table shared by all blocks. |
| anchor_embedding | bool | (token_transformer, optional, default true) false drops the learned per-anchor embedding (needs a layout with anchors; anchors still bind actions), so with relative positions the network sees board positions only as offsets. |

`token_transformer` needs an encoder that exposes a `TokenLayout`
(`engine.game_metadata(id)["token_layout"]`), i.e. a `_tok` variant such as
`splendor_tok`. Such variants share the base game's config directory, so the
transformer recipe lives in a separate file passed with
`--train-config games/<base>/config/train_transformer.json`. That file must not
carry an `mcts` block that differs from the base `train.json` (MCTS profiles
are always resolved from the base file).

#### training
| Field | Type | Description |
|-------|------|-------------|
| steps | int | Total training steps |
| episodes_per_step | int | Self-play episodes per step |
| batch_size | int | Training batch size |
| learning_rate | float | Initial learning rate |
| lr_schedule.type | string | Schedule type: `"cosine"` (recommended), `"step"`, or `"constant"` |
| lr_schedule.lr_min | float | Minimum LR at end of cosine schedule |
| lr_schedule.gamma | float | Decay factor per step (step schedule only) |
| lr_schedule.step_size | int | Steps between decays (step schedule only) |
| grad_clip_norm | float | Gradient clipping norm (default 1.0) |
| weight_decay | float | L2 regularization weight (default 1e-4) |
| train_batches_per_step | int | Gradient updates per step |
| replay_buffer_size | int | Max samples in replay buffer |
| simulations_start | int | Simulations override for initial steps (before model ready) |
| eval_benchmarks | string[] | Benchmark list, e.g. ["heuristic_free"] |
| opponent_pool_enabled | bool | Enable past-model opponent pool |
| opponent_pool_self_ratio | float | Fraction of games vs self (vs pool) |
| gating_accept_win_rate | float | Override auto-computed gating threshold |
| playout_cap | object | KataGo playout cap randomization: `{"full_simulations": int, "fast_simulations": int, "full_search_prob": float}`. Each selfplay MCTS ply is a full search (root noise, training sample) with probability full_search_prob, else a fast search without noise whose sample is dropped. Replaces the profile's simulations; exclusive with simulations_start; requires selfplay_concurrency. Splendor transformer 800/150/0.25: 6.4x games/hour, 1.5x training samples/hour vs 2000 every ply |
| score_head | object | Training-only auxiliary head predicting the final score margin: `{"loss_weight": float, "scale": float}`, both > 0. Target per seat = (own score − best other score) / scale, MSE weighted by loss_weight. Needs the game's `final_score_extractor` (`has_final_score_extractor` in metadata); not exported to ONNX. Can be toggled across resumes |
| opponent_policy_head | object | Training-only auxiliary head predicting the opponent's next move: `{"loss_weight": float}` (> 0; KataGo uses 0.15). Target = normalized root visits of the next ply acted by a different seat (fast PCR searches included); samples with no later opponent ply get weight 0. Not exported to ONNX. Can be toggled across resumes (optimizer groups are matched by name) |
| ema | object | `{"decay": float}`, 0 < decay < 1. Exported ONNX models (selfplay/eval/gating) use an exponential moving average of the weights, updated after every optimizer step; the checkpoint stores it as `ema_state_dict` |
| compact_replay | object | `{"features_dtype": "float16" \| "float32"}`. Replay buffer keeps features as numpy arrays in that dtype and policies sparse (densified per batch); selfplay returns numpy samples. Requires selfplay_concurrency. For large feature vectors (carcassonne_tok: up to ~15k floats, ~30 KB/sample in fp16 vs ~0.45 MB as Python float lists) |
| inference_precision | string | `"fp32"` (default) or `"fp16"`: selfplay/eval/gating run an fp16 copy under `models/fp16/` (fp32 I/O); checkpoints stay fp32. Worth it when GPU-bound (Splendor d128 transformer: +38% sims/s) |
| symmetry_augmentation | bool | Default false. Each sampled training example gets a uniformly random board symmetry of the encoder (`IFeatureEncoder::symmetries`, `engine/core/feature_encoder.h`) applied at batch time to its features, policy target, legal mask and opponent-policy target (`training/symmetry.py`); the replay buffer is unchanged. Rejected when the encoder offers no symmetries (only the Harmonies encoders do: 4 per board side) |

When to use `playout_cap`, `score_head`, `opponent_policy_head`, `ema` (and the MCTS `fpu_reduction`) and what they cost: `training/katago-techniques.md`. Throughput knobs (`inference_precision`, concurrency, providers): `training/performance.md`.

#### training.selfplay_concurrency
| Field | Type | Description |
|-------|------|-------------|
| enabled | bool | Enable batched concurrent inference |
| opponent_pool_max_distinct | int | Max distinct models loaded simultaneously |
| leaf_batch_size | int | Leaves each MCTS search collects (virtual loss) before one batched evaluation; 1 = classic one-leaf search. Required when enabled. |
| evaluator.batch_size | int | Max batch for inference |
| evaluator.batch_timeout_us | int | Microseconds to wait for batch fill |
| evaluator.max_queue_size | int | Max pending requests; must be >= workers * leaf_batch_size |
| evaluator.intra_op_num_threads | int | ONNX intra-op threads |
| evaluator.num_inference_threads | int | Inference threads, each with its own ONNX session. Required when enabled. 2 on a single GPU; keep `workers * leaf_batch_size >= num_inference_threads * batch_size` (see `training/performance.md`). |
| evaluator.devices | list[int] | Optional GPU ids for the cuda/tensorrt providers (default device 0). Thread i uses `devices[i % n]`; `num_inference_threads` must be a multiple of n. CLI `--devices` overrides. |
| evaluator.providers | string[] | ONNX execution providers (empty = CPU) |

#### mcts (profile deltas)
Contains `selfplay` and `eval` profile deltas. Each overrides baseline fields.

Selfplay is the ONLY place dirichlet non-zero values belong.
Eval profile is used for both benchmark eval and latest-vs-best gating.

`uct2_arrival_visits` (bool, default true) may appear only here, in `selfplay`
and/or `eval` (it is rejected in mcts.json and web.json). It sets the parent
count (the sqrt numerator) of a non-root node's exploration term. true: the
traversals of the incoming edge that arrived at this node. false: the
incoming edge's visit count over all its outcomes (the behaviour before this
key existed), which overstates the count at an outcome of probability p by
1/p. Details: `UCT2 parent count` in engine/search/net_mcts.h.
Set it to false in both profiles to resume a run started with the old
behaviour unchanged. It reaches selfplay, eval vs heuristic, gating and
`scripts/eval_model.py`; the web platform always uses true, and the `mcts_pv`
heuristic always uses false so its strength stays a fixed yardstick.

#### heuristic_guidance
| Field | Type | Description |
|-------|------|-------------|
| name | string | Heuristic name from heuristic.json::named |
| hold_steps | int | Steps before guidance starts decaying |
| steps | int | Absolute step at which the ratio reaches 0 (linear from hold_steps; must be > hold_steps; 0 = guidance off) |
| initial_ratio | float | Starting guidance ratio (1.0 = 100% heuristic) |
| temperature | float | Temperature for heuristic action selection |
| until_win_rate | object | Optional win-rate trigger (below); requires `steps: 0` |

Set `initial_ratio: 0.0` and `steps: 0` to disable guidance entirely.

`until_win_rate` = `{benchmark, threshold, decay_steps, max_hold_steps}`: hold
`initial_ratio` until an eval at step ≥ `hold_steps` shows `benchmark` (an
`eval_benchmarks` entry) win rate ≥ `threshold`, or `max_hold_steps` passes;
then decay linearly to 0 over `decay_steps`. The trigger step is saved in
`checkpoint.pt` (`guidance_trigger_step`) so `--resume` keeps the phase.

#### runtime
| Field | Type | Description |
|-------|------|-------------|
| workers | int | Optional (default 4). Concurrent selfplay games; CLI `--workers` overrides |
| eval_every | int | Steps between eval runs |
| eval_games | int | Games per eval run |
| save_every | int | Steps between checkpoint saves |
| seed | int | Optional fixed seed for reproducibility |

---

## web.json — Web/API Profile Deltas

Three required profiles, each a flat object overriding baseline fields:

| Profile | Purpose |
|---------|---------|
| web_expert | Full-strength AI for web play |
| web_casual | Weak AI (low simulations, no tail solve) |
| analysis | Drop-score / smart-hint pipeline. Only `simulations` and `world_cap` are consumed from this profile; the pipeline hard-codes `temperature=0.0`, `cover_root_edges=true`, and unfiltered search so every legal human move has visit data. |

Example:
```json
{
  "web_expert": { "simulations": 5000, "tail_solve_enabled": true },
  "web_casual": { "simulations": 10 },
  "analysis":   { "simulations": 5000, "world_cap": 32 }
}
```

---

## heuristic.json — Named Heuristic Pool + Selectors

### named (heuristic pool)
Each entry has a `type` and type-specific fields:

| Type | Fields | Description |
|------|--------|-------------|
| "rule" | (none) | C++-registered rule heuristic |
| "mcts_pv" | simulations, c_puct, [model_path], [world_cap], [leaf_batch_size] | PV evaluator + MCTS. With model_path: ONNX model as PV. Without model_path: hand-written C++ PV evaluator (e.g. quoridor's rule_mcts_v1). |

`leaf_batch_size` (int >= 1, default 1) is forwarded to the heuristic's own MCTS, same
semantics as `selfplay_concurrency.leaf_batch_size`. With an ONNX `model_path` the
concurrent selfplay/eval entries share one batched evaluator across threads, so
`evaluator.max_queue_size` must be >= threads * this value.

`mcts_pv` fields are real runtime knobs. The C++ game registration reads
`heuristic.json::named.<name>.simulations/c_puct/model_path` when constructing
`MctsPVHeuristic`; do not duplicate these values as hard-coded constants in the
registration file.

### Three independent heuristic consumers

All three pick from the same named pool but are configured independently — changing one does not affect the others.

| Consumer | Configured in | Purpose |
|----------|--------------|---------|
| Selfplay guidance | `train.json::heuristic_guidance.name` | Early training uses heuristic instead of MCTS for a fraction of episodes |
| Eval baseline | `heuristic.json::selectors.eval_vs_heuristic` | Opponent for eval benchmark (model vs heuristic) |
| Web hint | `heuristic.json::selectors.web` | Opponent for web "heuristic" difficulty |

Each selector in heuristic.json: `{ "name": "<named_key>", "temperature": 0.0 }`

---

## belief.json — Belief Network (Optional)

Only for games with belief tracking (coup, 6nimmit, liarsdice).

| Field | Type | Description |
|-------|------|-------------|
| architecture | int[] | Hidden layer widths |
| activation | "relu" | Activation function. Only `"relu"` is implemented; other values are rejected. |
| batch_norm | bool | Enable batch normalization |
| residual | bool | Enable residual connections |
| block_size | int | Layers per residual block |
| training.loss | "ce"\|"bce"\|"mse" | Loss function (cross-entropy, binary cross-entropy, or MSE) |

Belief is enabled by engine capability plus the presence of `belief.json`.
There is no `enabled` field. Belief logit temperature is framework-fixed at
`1.0` and must not be configured.

---

## Profile Resolution Architecture

```
resolve_profile(game_id, profile_name):
  1. Start with _BASE_DICT (in-code defaults, all zeros for dirichlet)
  2. Overlay mcts.json (game's shared baseline)
  3. Overlay profile delta:
     - selfplay/eval -> train.json["mcts"][name]
     - web_expert/web_casual/analysis -> web.json[name]
  4. Resolve dirichlet_on_first_n_plies sentinel (-1 -> max_game_plies)
  5. Validate (opponent_selection in {puct,prior}, tail_solve needs trigger)
  6. Return frozen MctsProfile dataclass
```

Profile names: `selfplay`, `eval`, `web_expert`, `web_casual`, `analysis`

All five profiles share the same mcts.json baseline. Dirichlet defaults to 0/0/0 in baseline -- only selfplay sets non-zero.

---

## Config Principles

- No silent fallbacks for real knobs: MCTS profiles start from explicit
  `_BASE_DICT` defaults plus `mcts.json` and profile deltas, while loaders reject
  unknown fields and missing required top-level config blocks.
- MCTS profile files (`mcts.json`, `train.json::mcts`, `web.json`) reject
  unknown fields via `training.mcts_profile`. Other config files validate only
  the fields their loaders know about; add explicit loader validation when a
  new knob is meant to be real.
- Legacy keys `heuristic_guidance_*` / `heuristic_temperature` at train.json::training level hard-fail with ValueError.
- Heuristic guidance (train.json), eval baseline (heuristic.json::selectors.eval_vs_heuristic), and web hint (heuristic.json::selectors.web) are three independent selectors — changing one does not affect the others.
- `opponent_selection` is set in mcts.json baseline only (inherited by all profiles). Never overridden per-profile, never a wire field.

---

## CLI Flags (training.cli)

| Flag | Description |
|------|-------------|
| `--init-from <path.pt>` | Warmstart from existing checkpoint weights |
| `--resume <output_dir>` | Resume interrupted training from checkpoint.pt in that directory |
| `--providers <comma-sep>` | Override selfplay_concurrency.evaluator.providers (e.g. `"coreml,cpu"` or `"cuda,cpu"`) |
| `--train-config <path>` | Override train.json path |
| `--config-override <path.json>` | Deep-merge a JSON object file into train config |
| `--belief-config <path>` | Override belief.json path |
| `--eval-benchmark <onnx_path ...>` | Additional ONNX models to benchmark against during eval |
| `--safe-mode` | Enable MCTS safe mode: validate DAG edges at every descent step. Slower but catches hash/schema bugs. |

Example:
```bash
python -m training.cli --game coup --output runs/coup_test \
  --init-from output/coup/step_500.pt \
  --providers "coreml,cpu" \
  --config-override overrides/coup_lr.json
```

---

## Example: Minimal Config Set (Love Letter)

```
games/loveletter/config/
  game.json       — game_id, display_name, players, max_game_plies, variants
  mcts.json       — {"c_puct": 1.6, "opponent_selection": "prior"}
  train.json      — network + training + mcts.{selfplay,eval} + heuristic_guidance + runtime
  web.json        — web_expert + web_casual + analysis
  heuristic.json  — named pool + selectors
```
