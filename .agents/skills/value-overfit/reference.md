# Value Overfitting in Self-Play: Detect, Fix, Verify

Lessons from Carcassonne and Harmonies (token transformer, KataGo options). In both, the
value head started memorising the outcomes of the games in the replay buffer and became
overconfident; eval vs the heuristic stayed high while gating stalled. Restarting with
less sample reuse fixed it, and head-to-head games against the old run's checkpoints
confirmed the new runs are much stronger.

## Symptoms

- Gating fails several evals in a row while eval vs the heuristic stays near saturation.
- Training `v_loss` keeps falling in pure self-play (it should stay roughly flat once
  the opponent improves as fast as the learner, see training DEC-001).
- The search's value estimates swing hard on small changes; `best_value` near +-1 in
  positions that are still open.

None of these proves overfitting. Gating can stall for other reasons, and a falling
`v_loss` can be real learning. Measure before changing the recipe.

## Measure: the value probe

Play fresh games the run has never trained on and compare the value head with the real
result there.

1. Self-play games with one checkpoint: fast search (about 150 simulations), no root
   noise (`dirichlet_alpha` 0), temperature decaying to about 0.1, enough games for tight
   intervals (400-640 games). `dinoboard_engine.run_selfplay_episode` returns every
   sample with its features, mover and final `z`.
2. For every sample: the raw value head (ONNX on the sample's features), and the
   result `z` (+-1, the training target).
3. A **scoreboard baseline**: logistic regression on the plain numbers the encoder hands
   the net (score margin, points still open, spaces left, who started), fitted per game
   phase with game-level cross-validation.
4. Report, with bootstrap intervals over games:
   - fresh-game MSE of the raw value against `z`: the same quantity as `v_loss`;
   - the same for the scoreboard baseline, and the difference raw minus scoreboard;
   - the **calibration slope**: `z` regressed on the net's value, per phase (bins of
     plies). 1 = calibrated, below 1 = overconfident (the net says 0.8 where 0.5 is true).

Reading it:

| Observation | Meaning |
|-------------|---------|
| fresh MSE about equal to train `v_loss` (or below it), slope about 1, raw below scoreboard | healthy: generalises and adds to the score |
| early-phase slope above 1 | cautious about openings: harmless, the opposite of the failure |
| train `v_loss` well below fresh MSE | memorising the buffer |
| mid-game slope 0.5-0.7 | overconfident |
| raw MSE at or above scoreboard | the value adds nothing a score count would not |

A flat `v_loss` around 0.7 that matches the fresh-game error is fine: Carcassonne v2 sat
there while getting much stronger. Outcomes are uncertain; the target is calibration,
not a low loss. Judge the fresh-game error against the scoreboard baseline of the same
checkpoint: as play gets stronger, games get closer and both errors rise together.

Costs: on the training box the probe slowed Harmonies training by about a fifth for an
hour. Run it where cores are free, with the same encoder code as the run (compare the
encoder sources' md5 if the trees differ).

## The knob that mattered: sample reuse

```
reuse         = train_batches_per_step * batch_size / new samples per step
draws/game    = train_batches_per_step * batch_size / episodes_per_step
buffer steps  = replay_buffer_size / new samples per step
```

New samples per step: watch how `samples=` in the step log grows before the buffer
fills (with `playout_cap` only full-search plies are recorded). Every sample of a game
carries the same `z`, so high reuse trains each game's one label many times.

| Run | batches x batch | reuse | buffer steps | other | value probe | outcome |
|-----|-----------------|-------|--------------|-------|-------------|---------|
| Harmonies, guided phase | 10 x 1024 | ~0.44 | ~55 | heuristic guidance | step 100: raw 0.69 vs scoreboard 0.76, slope ~0.9 | healthy |
| Harmonies, self-play | 10 x 1024 | ~1.9 | ~55 | opp head 0.15 | steps 200-1100: slope 0.55-0.73, train `v_loss` 0.55 -> 0.42 vs fresh 0.73-0.81 | gating failed six evals in a row |
| Harmonies v2 | 4 x 1024 | ~0.76 | ~185 | opp head off | steps 1000 / 2000 / 3300: raw - scoreboard -0.14 / -0.17 / -0.17, slope from ply 50 on 1.00-1.12, fresh 0.63-0.66 vs train 0.72-0.74 | still passing gating late in the run |
| Carcassonne old | 28 x 1024 | ~2.0 | ~21 | opp head 0.15 | step 775: raw - scoreboard +0.033, slope 0.55-0.65 | no new best after step 250 |
| Carcassonne v2 | 12 x 1024 | ~0.86 | ~105 | symmetry aug, opp head off | step 500: raw - scoreboard -0.043, slope 1.08-1.20, fresh 0.66 vs train 0.70 | stronger, see below |

Every data point so far fits "reuse near 2 -> overconfident, below 1 -> healthy". Several
knobs changed together (reuse, buffer, opponent head, symmetry), so this is the best
explanation, not an isolated measurement.

## Recipe

- Aim for reuse below 1 and a buffer of about 100 or more steps of data. Fewer
  `train_batches_per_step` is the direct lever; a larger `replay_buffer_size` keeps the
  window from shrinking to a few dozen steps of games.
- `symmetry_augmentation: true` when the encoder declares symmetries: each draw then
  sees a transformed sample, so the same game is not seen identically twice.
- Turn off `opponent_policy_head` (`"opponent_policy_head": null` in an override) when
  its loss sits at ln(number of choices) the whole run: it is not learning and only adds
  gradient noise to the shared trunk.
- Keep the guided phase: in Harmonies its checkpoints were healthy; overconfidence began
  when self-play data took over.
- Leave the MCTS settings, opening exploration and learning rate schedule alone while
  fixing this; change one family of knobs at a time so the probe can attribute.

## Restart, do not continue

An overconfident value head does not recover by training on: its confident targets feed
the search, which produces games that confirm it. Once the probe shows overconfidence:

1. Probe earlier checkpoints first (the run keeps `models/model_step_*.onnx`) and find
   the last healthy one.
2. Start the new run from that checkpoint's weights (`--resume` from a directory that
   holds only up to that ONNX: PV weights restored, optimizer and aux heads reset) or
   from scratch. Carcassonne v2 resumed from the old run's last best (step 250);
   Harmonies v2 started from scratch, though its step-100 checkpoint was healthy and
   would have saved the guided phase.
3. Put the new knobs in a `--config-override` JSON first; move them into the repo's
   train config once head-to-head games confirm the fix.

## Verify: head-to-head, not gating

Gating only compares with the previous best, so a steadily improving run can still fail
it often (both v2 runs passed less and less while getting much stronger). Play fixed
checkpoints against each other:

```bash
python scripts/eval_model.py --game <game> --model new.onnx --opponent old.onnx \
    --profile eval --paired-seats --games 200 --workers 100 --providers cuda,cpu --no-save -o /tmp/h2h
```

200 paired games give about +-5 points. Results:

| New | Old | New wins |
|-----|-----|----------|
| Harmonies v2 step 3300 | old run step 500 (its last best) | 92.5% |
| Harmonies v2 step 3300 | old run step 1200 (near its end) | 94.5% |
| Harmonies v2 step 1000 | old run step 500 | 84.75% |
| Carcassonne v2 step 1800 | old run step 250 (its last best, v2's start) | 80.25% |
| Carcassonne v2 step 1800 | old run step 1000 (its end) | 89.75% |
| Carcassonne v2 step 1000 | old run step 1000 (same step) | 84.5% |

## Known Issues

- **Probe on a busy box**: a head-to-head at `nice 19` on a box whose cores the training
  takes got about one core (about five hours per pairing). Run it where cores are free.
- **Different trees, different encoders**: the trees on two machines differed only in
  the encoder files; checkpoints would have been fed other features. Compare md5 of the
  encoder sources, or copy the run's tree, before evaluating its checkpoints elsewhere.
- **Restarting too early**: Harmonies was restarted before the early-checkpoint probes
  finished; they showed the guided phase was healthy, so a restart from step 100 would
  have kept that phase.
