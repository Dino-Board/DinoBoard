# KataGo-Style Training Techniques

Optional techniques adapted from KataGo (Wu 2019, "Accelerating Self-Play
Learning in Go") and Leela Zero. Every one is **off unless configured**, so
existing recipes behave exactly as before. Exact field syntax and validation
live in `game-config/reference.md`; this file covers what each does, when to
turn it on, and what it costs.

| Technique | Config | Where it acts | Default in code |
|-----------|--------|---------------|-----------------|
| Playout cap randomization | `training.playout_cap` | C++ selfplay | off |
| FPU reduction | MCTS profile `fpu_reduction` | C++ search (selfplay, eval, platform AI) | `null` (Q=0 for unvisited) |
| Score-margin auxiliary head | `training.score_head` | Python training only | off |
| Opponent-policy auxiliary head | `training.opponent_policy_head` | Python training only | off |
| Weight EMA for exported models | `training.ema` | Python export only | off |

`games/splendor/config/train_transformer_kata.json` enables all of them on
top of the Splendor transformer recipe and is the reference combination.

---

## Playout cap randomization

Each selfplay ply is a **full** search with probability `full_search_prob`
(root Dirichlet noise, sample kept for training) or otherwise a **fast**
search with `fast_simulations` and no noise, whose sample is dropped. Games
finish several times faster while every training sample still carries a
full-strength search target, and consecutive near-duplicate samples from one
game become rarer.

- Use it when selfplay throughput is at the evaluator ceiling
  ([performance.md](performance.md)).
- Starting point: full = 40% of your old simulations, fast = 1/5 of full,
  `full_search_prob` 0.25.
- Replaces the profile's `simulations` during selfplay; exclusive with
  `simulations_start`; needs `selfplay_concurrency.enabled`.

## FPU reduction (first-play urgency)

Unvisited children are scored with
`Q = Q(parent) − fpu_reduction × sqrt(sum of priors of visited children)`
instead of 0. With Q=0 an unvisited move looks like an even position, which
in a winning line makes the search waste visits on poor moves and in a losing
line makes it over-commit to the first ones it tried.

- Typical value 0.2–0.3 (KataGo uses ~0.2). Splendor `train.json` sets 0.2 in
  both `mcts.selfplay` and `mcts.eval`.
- It changes search behavior everywhere the profile is used, so eval win
  rates before and after enabling it are **not comparable** — compare
  checkpoints under the same setting.

## Score-margin auxiliary head

A small MLP on the PV net's shared embedding predicts each seat's final
score margin `(own score − best other score) / scale`, seat-rotated like the
value target, trained with MSE × `loss_weight`. It adds a dense learning
signal (how much you won by, not just whether) that shapes the trunk; the
head is not exported, so inference cost is unchanged.

- Requires the game's `GameBundle::final_score_extractor` (returns one score
  per seat from a terminal state; see `game-ai/registration.md`). Metadata
  reports it as `has_final_score_extractor`; enabling the head on a game
  without one raises.
- `scale` ≈ a typical winning margin's order of magnitude (Splendor: 15, the
  win threshold). `loss_weight` 0.5 is a reasonable start; KataGo keeps
  auxiliary losses well below the main value loss.
- Logged as `s_loss`.

## Opponent-policy auxiliary head

Another training-only head predicts the opponent's next move: the target is
the normalized root visit distribution of the next ply acted by a different
seat in the same game (fast-search plies included). Samples with no later
opponent ply get weight 0. Forcing the trunk to model opponent intent is
cheap and helps most in interactive games (racing for the same cards,
blocking).

- `loss_weight` 0.15 (KataGo's value).
- The target comes from the opponent's own search, which saw its private
  information. It is a training label only — inference never reads it — so
  it does not violate the hidden-information rule, in the same way the value
  target uses the final outcome.
- Logged as `o_loss`.

## Weight EMA

After every optimizer step an exponential moving average of the PV weights
is updated; the averaged net (not the raw weights) is what gets exported for
selfplay, eval and gating. Averaging smooths SGD noise and usually yields a
slightly stronger, steadier model. The raw weights keep training as before.

- `decay` 0.99–0.999. The effective window is about `1 / (1 − decay)`
  optimizer steps, i.e. `train_batches_per_step` updates per pipeline step:
  with 12 batches/step, 0.99 averages roughly the last 8 steps.
- Too high a decay makes exported models lag the trainer; selfplay data then
  comes from an older policy.

---

## Resume behavior

All three training-side options can be switched on or off when resuming:

- Heads and the EMA net are stored in `checkpoint.pt`
  (`score_head_state_dict`, `opponent_policy_head_state_dict`,
  `ema_state_dict`). A head enabled for the first time starts fresh; the EMA
  starts from the current weights.
- The PV optimizer has one **named** param group per module (`pv`,
  `score_head`, `opponent_policy_head`) and resume matches them by name: a
  newly enabled head gets fresh Adam moments, a disabled one is dropped.
- `fpu_reduction` and `playout_cap` are read from config every run.

Turning several on at once makes it impossible to attribute a change in
strength. To measure one technique, copy the run directory (`--resume`
requires `--output` to be the checkpoint's own directory), resume each copy
with configs that differ only in that option, and compare the resulting
models with `/eval-model` head-to-head rather than through the noisy
periodic eval.

Tests: `tests/framework/test_kata_features.py`, `tests/framework/test_playout_cap.py`.
