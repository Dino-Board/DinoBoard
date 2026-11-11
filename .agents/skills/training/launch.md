# Launching, Resuming & Stopping Training

Operational runbook for starting a training run, attaching GPU, resuming, and
stopping cleanly. For *what values to set*, see hyperparams.md; for *diagnosing
a sick run*, see diagnostics.md.

---

## The two independent GPU paths

"Use GPU" means two separate things. Know which one you depend on:

| Path | What it accelerates | Dependency | Needs `gpu_env.sh`? |
|------|--------------------|-----------|---------------------|
| PyTorch training | Per-step gradient updates on the PV net | PyTorch's own bundled cuDNN | No — works as long as a CUDA device is visible |
| ONNX self-play inference | Model leaf evaluation in selfplay/eval MCTS | onnxruntime CUDA execution provider | **Yes** — needs the venv's pip `nvidia-cudnn-cu12` on `LD_LIBRARY_PATH` |

`scripts/gpu_env.sh` only fixes the **second** path. It prepends the bundled
ORT GPU libs and the `nvidia-*` pip CUDA/cuDNN `.so` dirs to `LD_LIBRARY_PATH`.

### Why you must source it before `--providers cuda`

The concurrent self-play `BatchedPolicyValueEvaluator` is **created at startup**
(it loads `model_init.onnx`) regardless of heuristic-guidance ratio. With
`--providers cuda` but no `gpu_env.sh`, evaluator creation **hard-crashes** at
step 1:

```
run_concurrent_selfplay_step: evaluator creation failed:
BatchedPolicyValueEvaluator: failed to load model 'model_init.onnx':
... libcudnn_adv.so.9: cannot open shared object file
```

This is NOT a silent CPU fallback — the run dies. Always `source
scripts/gpu_env.sh` in the same shell before launching with `--providers cuda`.

### Heuristic-guidance phase nuance

During heuristic guidance (steps up to `heuristic_guidance.steps`, at ratio
1.0), self-play picks actions via the hand-written C++ PV heuristic, NOT the
ONNX model — so ONNX GPU inference is mostly idle in that window. But PyTorch
training still runs on GPU every step, and after guidance ends the model takes
over self-play and ONNX GPU becomes the critical path. Source `gpu_env.sh` from
the start so you never have to stop/restart to add it mid-run.

---

## Fresh run (from scratch)

Run from the repo root with the training venv's python. `source gpu_env.sh`
first, then launch under `nohup ... &` so closing the terminal doesn't kill it.

```bash
cd /home/chihchi/Projects/DinoBoard && \
source scripts/gpu_env.sh && \
nohup .venv-train-cpu/bin/python -m training.cli \
  --game <game_id> \
  --output runs/<run_name> \
  --workers 100 --providers cuda \
  > runs/<run_name>.nohup.out 2>&1 &
```

- `--output` must be a NEW directory for a from-scratch run (don't reuse a dir
  that already has `checkpoint.pt` unless you intend `--resume`).
- `--workers` = concurrent self-play threads (100 is typical on desktop).
- Drop `--providers cuda` (and the `gpu_env.sh` line) to run pure CPU.

## Resume an interrupted run

Point `--resume` at the existing run dir. It restores from `checkpoint.pt`; if
that's missing it recovers PV weights from the latest `model_step_*.onnx`
(optimizer/LR-schedule state is reset in that fallback).

```bash
cd /home/chihchi/Projects/DinoBoard && \
source scripts/gpu_env.sh && \
nohup .venv-train-cpu/bin/python -m training.cli \
  --game <game_id> \
  --output runs/<run_name> \
  --resume runs/<run_name> \
  --workers 100 --providers cuda \
  > runs/<run_name>.nohup_resume.out 2>&1 &
```

> Config edits (lr, world_cap, guidance, etc.) only fully apply to a
> **fresh** run. Resume keeps the schedule/step counter from the checkpoint —
> if you changed hyperparameters to fix a bad run, start fresh, don't resume.
> Exception: the KataGo-style options (auxiliary heads, EMA, FPU, playout
> cap) are designed to be switched on or off at resume — see
> [katago-techniques.md](katago-techniques.md#resume-behavior).

To run a non-default recipe (e.g. the Splendor transformer), add
`--train-config games/<base>/config/<recipe>.json` to the launch command, and
pass the same file again on every resume.

---

## Monitor

```bash
# Confirm the process is alive
ps aux | grep "training.cli" | grep -v grep

# Follow the metrics log (per-step p_loss/v_loss/lr + eval/gating)
# sleep 3s so nohup has time to create the log file before tail opens it
sleep 3 && tail -f runs/<run_name>/train.log
```

Healthy step-1 signature for a fresh run: `p_loss` near `log(action_space)`
(random policy), `v_loss` ~0.8, `lr` = your configured base, first model
`[PROMOTED]`, `heuristic_ratio:1.00`.

## Verify GPU is actually engaged

`Training device: cuda` in the log only confirms the PyTorch path. To confirm
the ONNX self-play path is on GPU, check that post-guidance self-play steps are
fast (a CPU fallback shows a much higher ETA). `nvidia-smi` should show the
training python holding GPU memory.

## Stop

```bash
# Find the training PID (and any `tail -f` you started)
ps aux | grep -E "training.cli|tail -f.*train.log" | grep -v grep

# Graceful stop
kill <training_pid> <tail_pid>

# Verify it's gone
ps aux | grep "training.cli" | grep -v grep || echo "stopped"
```

SIGTERM or SIGINT makes the pipeline finish the current step, write
`checkpoint.pt` and exit (log line `Stop signal ... stopped after step N`),
so the run is resumable. Use `kill` (SIGTERM) for `nohup ... &` jobs:
background jobs ignore Ctrl+C-style SIGINT. A step can take minutes, so wait
for the process to exit; avoid `kill -9` unless it's hung.

---

## Known Issues

- **`libcudnn_*.so.9: cannot open shared object file`** → forgot to `source
  scripts/gpu_env.sh` before `--providers cuda`. The run crashes at evaluator
  creation, not at the first model use. Source it and relaunch.
- **`--providers cuda` "works" but ETA is huge / GPU idle** → cuDNN missing and
  the CUDA provider silently fell back to CPU for ONNX (PyTorch training may
  still be on GPU, masking the problem). Verify with `gpu_env.sh` + `nvidia-smi`.
- **Changed a hyperparameter but the run ignores it** → you used `--resume`. The
  checkpoint pins the LR schedule and step counter; config changes meant to fix
  a bad run require a fresh `--output` dir.
