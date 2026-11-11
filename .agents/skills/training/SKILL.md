---
description: "Tune training hyperparameters, selfplay throughput and KataGo-style options; diagnose training issues"
---

# Training Tuning & Diagnostics

Read the sub-document matching your current task:

- **Launch / resume / stop a run** (GPU env setup, `gpu_env.sh`, nohup launch, monitoring, clean stop): read [launch.md](launch.md)
- **Hyperparameter selection** (c_puct, network depth/width, lr, Dirichlet, opponent pool by game type): read [hyperparams.md](hyperparams.md)
- **Diagnose issues** (symptom-to-cause checklist, concurrent inference setup, known BUGs): read [diagnostics.md](diagnostics.md)
- **Selfplay throughput** (bottleneck diagnosis, workers / leaf batching, fp16, TensorRT, memory): read [performance.md](performance.md)
- **KataGo-style techniques** (playout cap randomization, FPU reduction, score / opponent-policy auxiliary heads, weight EMA): read [katago-techniques.md](katago-techniques.md)

Only read the file relevant to the current task.
