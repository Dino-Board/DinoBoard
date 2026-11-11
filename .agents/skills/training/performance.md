# Selfplay Throughput Tuning

How to make a run produce more training signal per wall-clock hour. Field
syntax lives in `game-config/reference.md` (`training.selfplay_concurrency`,
`training.inference_precision`, `training.playout_cap`); this file says which
knob to turn and why. Algorithm-level changes that also save compute (playout
cap randomization) are in [katago-techniques.md](katago-techniques.md).

---

## Where the time goes

A step = `episodes_per_step` games, every ply runs an MCTS search, every
simulation needs one network evaluation. SGD is a few seconds; selfplay is
everything else. So the budget is **network evaluations per second**, and the
cost of a step is roughly `episodes × plies × simulations / evals_per_sec`.

Two consequences:

1. Every ONNX `Run()` has a fixed cost (kernel launches: about 1 ms for the
   d128 4-layer transformer, independent of batch size) plus a per-item cost.
   Small batches are all fixed cost; only full batches (~1024) reach the
   GPU's compute ceiling (Splendor transformer fp16: ~2.3 µs/item, about
   400k evaluations/s on one consumer GPU).
2. Past the ceiling, the only lever is **spending fewer simulations per useful
   training sample** — i.e. playout cap randomization.

## Diagnose before tuning

`nvidia-smi` utilization is **not** a saturation signal: it counts time with
any kernel running, so a GPU stuck on launch overhead or serialized sessions
can show 40–60% while being the bottleneck. Measure the evaluator directly
(evaluations/s and average batch size, e.g. by timing `run_batch` in
`batched_policy_value_evaluator.cpp`) and compare against a standalone
single-thread ORT benchmark of the same model at the same batch size.

Rough guide from `nvidia-smi` and `top`:

| Observation | Bottleneck | Turn |
|-------------|-----------|------|
| GPU util / power low, CPU cores idle | Too few leaves in flight | `--workers`, `leaf_batch_size` |
| GPU low, CPU cores saturated | C++ search / encoding | fewer workers per core, cheaper encoder, `world_cap` |
| GPU high | Net is the cost | `inference_precision: "fp16"`, smaller net |
| Neither saturated, sims/s flat as you add workers | Batches too small (fixed `Run()` cost) or GPU at compute ceiling | more leaves in flight so batches fill; then playout cap randomization |
| Memory pressure / OOM-kill | Too many concurrent games or replay too large | fewer workers, smaller `replay_buffer_size`, more WSL memory |

## Knobs, in the order to try them

### 1. Concurrency (`training.selfplay_concurrency`)

All of these are framework-level (`engine/infer`, `engine/search/net_mcts`,
concurrent selfplay binding) and work for every game, including
hidden-information games: leaf batches are flushed only when full or when the
search ends, not per sampled world.

- `runtime.workers` (CLI `--workers` overrides) = concurrent selfplay games,
  capped by `episodes_per_step`. Each game contributes `leaf_batch_size`
  leaves per evaluator round trip, so leaves in flight ≈
  `workers × leaf_batch_size`.
- The rule: **leaves in flight ≥ `num_inference_threads × batch_size`**, so
  every inference thread always takes a full batch. Otherwise each `Run()`
  pays the fixed cost for a fraction of a batch.
- `leaf_batch_size` (virtual loss): 8–16. Larger values make each search less
  sequential (slightly weaker per simulation) but fill batches with fewer
  games and less memory.
- `evaluator.max_queue_size` must be ≥ `workers × leaf_batch_size`
  (validated at startup; also checked against a heuristic MCTS teacher's own
  `leaf_batch_size` during guidance).
- `evaluator.num_inference_threads`: one ORT session per thread. **2 on a
  single GPU.** Sessions on one GPU serialize (under WSL2 strictly), so a
  second thread only hides host-side work (copy-in, result scatter); 3+
  threads split the same leaves into smaller batches and get slower.
- `evaluator.devices` (or `--devices 0,1,2`): multi-GPU. Thread i runs on
  `devices[i % n]`, so use 2 threads per GPU. Check CPU first: Splendor tok on
  3× 4090D + 48 Xeon threads saturates the CPU at ~410 smp/s (GPUs ~25% busy);
  more workers do not help once MCTS is CPU-bound.
- `evaluator.batch_size`: 768–1024 for small nets. Bigger batches (1536+) add
  latency without more throughput once the GPU is compute-bound.
- `evaluator.batch_timeout_us`: short (tens of µs) when leaves in flight
  comfortably exceed `batch_size`; longer only if batches come out tiny.

Measured on the Splendor d128 transformer (fp16, CUDA, WSL2, 800/150 playout
cap, 400 episodes/step), selfplay seconds per step:

| workers × leaf | threads × batch | s/step |
|----------------|-----------------|--------|
| 200 × 8 | 4 × 1024 | 33.6 (batches ~290) |
| 400 × 16 | 4 × 1024 | 28.2 |
| 400 × 16 | 3 × 1024 | 25.9 |
| 400 × 16 | **2 × 1024** | **22.5** |
| 400 × 16 | 2 × 768 | 23.6 |
| 400 × 16 | 2 × 1536 | 24.6 |
| 400 × 8 | 2 × 1024 | 27.8 |
| 400 × 16 | 1 × 1024 | 37.1 |

Things that did not help there: ORT `do_copy_in_default_stream=0`; moving
caller wake-ups to a separate completion thread (the GPU was already the
limit); CUDA graphs (+10% single-threaded, but capture fails with concurrent
sessions).

### 2. Inference precision (`training.inference_precision`)

`"fp16"` exports an fp16 copy next to every ONNX model (`models/fp16/`, fp32
I/O) and uses it for selfplay, eval and gating; checkpoints and SGD stay
fp32. Worth it whenever the GPU is a noticeable share of the cost (Splendor
d128 transformer: about +38% simulations/s). No effect on CPU-only runs.

Long token sets are GPU-bound instead. A fixed-slot Carcassonne encoder (221
tokens, d128 L4, fp16, GPU shared with a Splendor run) did 9.4k sims/s versus
140–215k for splendor_tok. Plain-op attention materialises `[B, heads, S, S]`
scores (fused SDPA halved the PyTorch forward), so set
`network.attention: "fused"`; the model pads the sequence to a multiple of 8,
which onnxruntime's memory-efficient kernel needs. Also use
`training.compact_replay`: that trainer peaked at ~6.5 GB plus ~40 KB per
replay sample.

carcassonne_tok emits only the tokens present (mean ~80, max ~160 versus those
220 slots). PyTorch fp16, batch 1024, GPU shared: about 3x the inference and
training throughput of the fixed slots at 80 tokens, about 1.4x at 160. The
relative position bias is a dense `[B, heads, S, S]` tensor: it must be
contiguous (SDPA is ~10x slower otherwise) and costs +15% at 80 tokens, +35%
at 160. A batch pads to its longest encoding, so mixed-stage batches run at
the late-game length.

### 3. Execution provider (`--providers`)

`cuda,cpu` is the default GPU choice. `tensorrt,cuda,cpu` is supported
(needs the TensorRT libs on `LD_LIBRARY_PATH` via `scripts/gpu_env.sh`);
engines are built per model with a 1..`batch_size` shape profile and cached
in `<model dir>/trt_cache`, so the first step after each export pays a build.
On the Splendor transformer it did **not** beat CUDA fp16 (the run was at the
evaluator-latency ceiling) and doubled GPU power — measure before adopting.

### 4. Fewer simulations per useful sample

Once sims/s is flat, use `training.playout_cap` (see
[katago-techniques.md](katago-techniques.md)). On the Splendor transformer,
800 full / 150 fast / 0.25 gave ~6.7× games/hour and ~1.5–1.9× training
samples/hour versus 2000 simulations on every ply.

### 5. Host resources

- 200–400 concurrent Splendor games plus a 200k-sample replay buffer need well
  over 15 GB. Under WSL2 raise `memory=` in `%UserProfile%\.wslconfig`
  (then `wsl --shutdown`); otherwise the OOM killer ends the run silently.
- Building the extension with `nice -n 19` keeps a running training job from
  being starved.

## Measuring a change

Compare **seconds per step at equal episodes and simulations**, or better,
training samples per hour (the `samples=` growth in `train.log` until the
replay buffer is full). Run each variant for several steps after warm-up:
the first step after an export includes model load (and TensorRT builds).
Measure selfplay and heuristic-guided steps separately: a guided step runs the
teacher's search, not the net under training, and has a different cost.

For a repeatable selfplay benchmark, copy a trained run dir and resume the
copy for one or two steps with `--config-override` carrying the variant
(`--output` must equal `--resume`, pass `--train-config` again,
`--eval-every 0`). Starting from a trained checkpoint skips the guidance
phase and gives realistic game lengths.
Never judge a throughput change by eval win rate — 100 eval games carry about
±10% noise.
