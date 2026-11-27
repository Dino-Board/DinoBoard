---
description: "Tune training hyperparameters, prevent policy/value overconfidence, configure KataGo exploration (shaped Dirichlet, exponential temperature, opening/midgame start), selfplay throughput, and diagnose training issues"
---

# Training Tuning & Diagnostics

Read the sub-document matching your current task:

- **Launch / resume / stop a run** (GPU env setup, `gpu_env.sh`, nohup launch, monitoring, clean stop): read [launch.md](launch.md)
- **Hyperparameter selection** (c_puct, network depth/width, lr, KataGo exponential `temperature_schedule`, Dirichlet, opponent pool by game type): read [hyperparams.md](hyperparams.md)
- **Prevent overconfidence** (policy prior collapse vs value outcome memorization, triage table, sample reuse `< 1.0`, layered prevention & rollback checklist): read [overconfidence.md](overconfidence.md)
- **Selfplay exploration / policy lock-in** (`opening_exploration`, KataGo `midgame_start` + tournament fork, exponential `root_policy_temperature`, adaptive/shaped `dirichlet_total_alpha`, `prior_ent` / `policy_kl` logs): read [exploration.md](exploration.md)
- **Diagnose issues** (symptom-to-cause checklist, concurrent inference setup, banned legacy keys, known BUGs): read [diagnostics.md](diagnostics.md)
- **Selfplay throughput** (bottleneck diagnosis, workers / leaf batching, fp16, TensorRT, memory): read [performance.md](performance.md)
- **KataGo-style techniques** (playout cap randomization, FPU reduction, score / score-breakdown auxiliary heads, weight EMA): read [katago-techniques.md](katago-techniques.md)

Only read the file relevant to the current task.
