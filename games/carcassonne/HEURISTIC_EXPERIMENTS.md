# Carcassonne Heuristic Experiments

This file records Carcassonne heuristic variants, tuning results, and retired
ideas. Keep strategy sources and general human advice in `STRATEGY_NOTES.md`;
keep benchmark history here so removed versions are not accidentally rebuilt.

## Post-Refactor Experiments (2026-06-22+)

After the phased-weights + final-scoring V head refactor (commit `13bc08c`),
all weights were re-designed to match the strategy guides ("opening = city/
cloister, endgame = farm warfare") rather than mechanically sampling the old
linear curves. Earlier sweep numbers in this file describe the pre-refactor
heuristic; they are not directly comparable to results below.

### Sanity: new sharp_pv_soft vs base (40 paired)

`sharp_pv_soft` redesign achieves 24W-14L-2D over `base` at 62.5% (seed
996958, sim 2000, world_cap 200). Confirms the phased-weights framework
beats a phase-flat baseline as expected.

- As seat0: 10-10 / 20 (50%)
- As seat1: 14-4-2 / 20 (75%)

Note the seat skew: sharp is much stronger when going second. Likely cause:
new sharp's stronger early-game city/cloister bias rewards reactive play
more than opening commitments, but at sim 2000 / wc 200 the sample is too
small to tell whether it's a real strategy issue or seat noise. Watch for
similar skew in candidate experiments below.

### First-mover seed log

Each row uses a freshly drawn seed; do not re-use across rounds.

| Date | Experiment | Seed | Games | Result |
|---|---|---|---|---|
| 2026-06-22 | sharp_pv_soft (post-refactor) vs base | 996958 | 40 paired | sharp 24-14-2 = 62.5% (seat skew) |
| 2026-06-22 | sharefix vs sharp_pv_soft | 807772 | 40 paired | sharefix 21-18-1 = 53.75% (seat 67/40) |
| 2026-06-22 | sharp_open_edge vs sharp_pv_soft | 566663 | 40 paired | open_edge 22-16-2 = 57.5% (seat 67/47) |
| 2026-06-22 | sharp_cloistercomplete vs sharp_pv_soft | 319101 | 40 paired | cc 17-23-0 = 42.5% (seat 50/35) |
| 2026-06-22 | sharp_farmpotential vs sharp_pv_soft | 806384 | 40 paired | fp 17-20-3 = 46.25% (seat 52/40) |

### First-round candidate screening summary

All four research profiles tested against new `sharp_pv_soft` at sim 2000 /
world_cap 200, 40 paired-seat games each, fresh seed per candidate. CI is
±15pp at this sample size — none of the candidates is statistically
significant on its own, but the directions are informative:

- **`sharp_open_edge`** (the only positive signal): 57.5%, balanced
  seats. Opens the `city/road_completion_bonus` + `*_open_penalty` +
  `opp_*_open_bonus` + `*_meeple_open_penalty` family. Most likely a
  real upgrade — the open-edge structural signal explicitly captures
  "completability" which the legacy ownership branches miss.
- **`sharefix`** (53.75% but seat-skewed): The marginal-evaluation
  path with `award_to_majority`-correct semantics edges out, but the
  67/40 seat skew suggests the marginal eval rewards an aggressive
  joining style that works only as seat 0. Worth a look at place-time
  weighting before discarding.
- **`sharp_farmpotential`** (46.25%): Adding `farm_potential` (incomplete-
  city contribution) hurts. Likely because the new V head already
  projects all incomplete cities via `apply_final_scoring`, so this
  rule-level addition double-counts.
- **`sharp_cloistercomplete`** (42.5%): Adding `tile_cloister_*`
  significantly hurts. Same root cause — V head already counts cloister
  occupancy in projection, so rewarding "tile that completes cloister"
  at the rule level pushes the heuristic to over-prioritize chapel
  neighbors at the expense of city/road completions that the V head
  was about to credit anyway.

**Lesson**: In the new framework with 100% final-scoring V head, "turn
on a feature that the engine already projects" pattern becomes
DOUBLE-COUNTING, not strengthening. Future feature additions should
either (a) sit OUTSIDE the V head's projection (open-edge structural
is one such — the V head doesn't model completability difficulty), or
(b) replace V projection of that feature, not augment it.

**Web binding**: `rule_mcts_web` (and the `web` selector) now resolves
to `kSharpOpenEdge` rather than `kSharpPvSoft`. Pending 80-game
re-confirmation but already deployed for play-testing.

## Active Variants

These names are intentionally the only MCTS-PV variants kept registered in
code and `config/heuristic.json`.

- `rule_mcts` / `rule_mcts_web` / `rule_mcts_sharp_pv_soft`: current default.
  Uses the sharp hand-written prior with a softer PV prior temperature
  (`0.78`), score scale `24`, and projected-score value blend `0.62`.
- `rule_mcts_base`: baseline rule weights and baseline PV settings. Keep only
  as the stable control arm for regression checks.
- `rule_mcts_sharp_cloister_soft_pv`: sharp default plus small tile-placement
  bonuses for helping occupied cloisters. Useful live candidate, but previous
  checks showed seat skew.
- `rule_mcts_sharp_reserve_soft_pv`: sharp cloister weights plus modest
  phase-dependent meeple reserve pressure. Useful live candidate, but not a
  clear default upgrade.
- `rule_mcts_sharp_cloister_reserve_soft_pv`: fusion of cloister help and
  light reserve pressure. Strong against base and current default in one
  random-stream follow-up, but lost to `rule_mcts_sharp_cloister_soft_pv`.

Current selector intent:

- `eval_vs_heuristic` uses `rule_mcts` for training/eval scale.
- `web` uses `rule_mcts_web` with the same profile and a larger simulation
  budget.

## Current Open Questions

- Does the phased PV value head overweight projected farms too early?
- Does farm tile-shaping in tile placement overfit local completed-city counts
  and distort otherwise good placements?
- Are early supply penalties too high, causing missed small city/cloister
  points?
- Is the prior temperature too sharp in the opening, reducing MCTS exploration
  of tactical joins and steals?
- Should final-score projection be split into completed score, incomplete
  feature estimate, and farm estimate instead of one blended margin?
- Can locked opponent meeples and completion difficulty be estimated cheaply
  enough to make blocking visible to the heuristic?

## Benchmark Method Notes

- Use paired seats: play each random seed twice, once with each candidate in
  each seat.
- Do not reuse the same fixed seed list across tuning rounds. Use a fresh
  random seed stream for every screening batch; only reuse a seed for its
  swapped-seat pair.
- Treat 20-game or smaller rows as screening only. Several candidates moved by
  15+ percentage points when retested on new seeds.
- For heuristic-vs-heuristic experiments, batching is unnecessary; plain
  parallel workers are faster and simpler.

## Sweep Notes

All runs below used heuristic-vs-heuristic MCTS-PV, `c_puct=1.4`, paired seats,
and parallel workers. Older rows were run at 1k simulations before the current
training-scale config moved to 2k simulations.

### Useful Findings

- `rule_mcts_sharp_pv_soft` remains the best default candidate so far. It beat
  `rule_mcts_base` 39-21-0 / 60 games on one seed set, then 32-25-3 / 60 on a
  later seed set. The second run is less flashy but still positive in both
  seats.
- `rule_mcts_sharp_reserve` beat `rule_mcts_sharp_pv_soft` directly
  33-24-3 / 60, but only scored 34-25-1 / 60 and 30-27-3 / 60 against
  `rule_mcts_base`. It looks like a style counter, not a clear default upgrade.
- `rule_mcts_sharp_cloister_soft_pv` scored 23-15-2 / 40 against base, but was
  highly seat-skewed: 15-4-1 as seat 0 and 8-11-1 as seat 1. Cloister tile
  help is useful but not sufficient by itself.
- `rule_mcts_sharp_reserve_soft_pv` scored 24-16-0 / 40 against base, more
  balanced than cloister-only. Soft PV helps the reserve rules, but the direct
  head-to-head still favors `rule_mcts_sharp_pv_soft` over the balanced fusion.

### Random-Stream Check

To avoid overfitting to a fixed benchmark seed list, candidates were screened
with fresh random seeds per candidate. Each seed was still played twice with
swapped seats to reduce first-player/deal variance, but seeds were not reused
across candidates.

- `rule_mcts_sharp_pv_soft` and `rule_mcts_sharp_cloister_soft_pv` were the
  best quick-screen candidates at 9-3 / 12 games each against
  `rule_mcts_base`, both with paired-seed score 3-0-3.
- `rule_mcts_sharp_reserve_soft_pv` was close behind at 8-3-1 / 12 games,
  paired-seed score 3-0-3.
- `rule_mcts_completion_lite`, the previous default candidate, fell to
  4-8 / 12 games and should not stay the default without stronger evidence.
- `rule_mcts_sharp_balanced` and plain `rule_mcts_sharp` both underperformed,
  suggesting the balanced farm retune and sharper PV prior are harmful in the
  current 1k-sim setup.

Follow-up random-stream checks found a triangle rather than a clean upgrade:

- `rule_mcts_sharp_cloister_reserve_soft_pv` beat `rule_mcts_base`
  18-6 / 24 games with balanced seats and paired-seed score 6-0-6.
- The same fusion beat `rule_mcts_sharp_pv_soft` 12-4 / 16 games, but lost to
  `rule_mcts_sharp_cloister_soft_pv` 6-10 / 16 games.
- `rule_mcts_sharp_cloister_soft_pv` then lost to `rule_mcts_sharp_pv_soft`
  10-14 / 24 games, with a weak seat 1 result.

Conclusion: keep `rule_mcts_sharp_cloister_reserve_soft_pv` and
`rule_mcts_sharp_cloister_soft_pv` as live candidates, but use
`rule_mcts_sharp_pv_soft` as the default until a candidate beats both base and
the current default on fresh random-stream checks without a seat skew.

## Retired Variants

These variants were removed from registration/config to keep the web/eval
surface small. Keep the notes so future tuning does not retread the same path.

- `rule_mcts_sharp`: sharper rule weights with sharper PV prior. It is the
  ancestor of the current default, but the later softer-PV version was stronger
  and the plain sharp profile underperformed in random-stream screening.
- `rule_mcts_meeple_saver`: heavy reserve penalties, lower PV projection
  weight. Screened weak at 6-14 / 20 because it skipped too many fast
  city/cloister points.
- `rule_mcts_farmer`: whole-profile farm aggression with higher farm weights
  and high PV projection weight. Looked excellent in one 20-game screen, then
  fell to 25-34-1 / 60 on fresh seeds. Lesson: strong farm bias must be
  phase-limited and validated for seat 1.
- `rule_mcts_phased_base_pv`: phased rule weights with baseline PV. Went
  7-13 / 20, suggesting the rule retune alone was not enough.
- `rule_mcts_phased_baseline_rules`: baseline rules with phased PV. Useful
  ablation only; not kept as a live candidate.
- `rule_mcts_phased_sharp_rules`: sharp rules with phased PV. Folded into
  later sharp/PV-soft experiments; not a distinct keeper.
- `rule_mcts_phased_no_farm_tile`: phased profile with farm tile-shaping
  removed. Recovered to 10-10 / 20 after `phased_base_pv` struggled, which
  implies farm tile-shaping was harmful in that blend, but it did not create a
  winning profile.
- `rule_mcts_cloister`: phased/no-farm-tile profile plus cloister tile help.
  Superseded by `rule_mcts_sharp_cloister_soft_pv`, which applies cloister help
  on the stronger sharp-soft base.
- `rule_mcts_reserve`: phased cloister profile plus strong reserve pressure.
  Superseded by the softer reserve variants; strong reserve pressure risks
  missing tactical points.
- `rule_mcts_farm_potential`: reserve-phased profile plus incomplete-city farm
  projection. Not a clear strength gain; farm projection should be revisited in
  a separated value head instead of folded into one global score.
- `rule_mcts_majority_projection`: phased profile with full-majority projection
  instead of owner-share projection. Roughly neutral. The idea may still matter
  for targeted join/steal tactics, but global full-award projection did not
  help.
- `rule_mcts_sharp_no_farm_tile`: sharp profile with farm tile-shaping removed.
  Kept only as an ablation during the farm-shaping investigation; no evidence
  it beats the sharp-soft default.
- `rule_mcts_sharp_cloister`: sharp profile plus cloister tile help with the
  sharper PV setup. Superseded by `rule_mcts_sharp_cloister_soft_pv`.
- `rule_mcts_sharp_reserve`: sharp cloister profile plus reserve pressure with
  the sharper PV setup. It beat `rule_mcts_sharp_pv_soft` 33-24-3 / 60, but was
  inconsistent against base, so keep the result as a style-counter warning
  rather than a default candidate.
- `rule_mcts_sharp_reserve_no_farm_tile`: reserve ablation with farm
  tile-shaping removed. No strong result recorded; remove until farm-shaping is
  redesigned more systematically.
- `rule_mcts_sharp_majority_projection`: sharp cloister profile plus global
  full-majority projection. Same lesson as the phased majority version:
  roughly neutral globally, potentially useful only as targeted tactics.
- `rule_mcts_fusion`: hand-merged opening/midgame/endgame weights from several
  promising ideas. Looked strong at 12-6-2 / 20, then fell to 29-30-1 / 60 on
  the same benchmark family. Lesson: do not trust attractive 20-game rows.
- `rule_mcts_sharp_pv_endgame`: sharper phase-dependent PV that ramps
  endgame projection weight to `1.00`. Removed because stronger endgame value
  blending did not clearly beat the softer default and risks overvaluing farms.
- `rule_mcts_sharp_balanced`: sharp-soft PV, light cloister help, modest
  early/mid meeple reserve, and delayed endgame farming. Beat base 25-13-2 / 40
  and 35-25-0 / 60, but lost to `rule_mcts_sharp_pv_soft` 27-31-2 / 60 and
  later underperformed in random-stream screening.
- `rule_mcts_sharp_balanced_endgame_pv`: balanced rules with endgame-heavy PV.
  Removed with the balanced family; no evidence it beats the default without
  overvaluing projected final scoring.
- `rule_mcts_sharp_pv_endgame_strong`: even stronger endgame projection
  (`value_projection_weight` up to `1.00`, smaller value scale). Removed
  because the stronger endgame value direction was not producing reliable
  strength gains.
- `rule_mcts_completion_aware`: balanced rules plus explicit completion/open
  feature terms and opponent-open bonuses. Too many moving parts for the signal
  we had; remove until completion difficulty can be measured more directly.
- `rule_mcts_completion_lite`: lighter completion/open feature terms. It was
  the previous default candidate, but fell to 4-8 / 12 against base on fresh
  random-stream seeds.

## Next Experiment Ideas

- Build features for locked opponent meeples and completion difficulty instead
  of only local feature ownership. Current weights see many blocks only through
  immediate feature delta.
- Split the PV value head into current score, incomplete feature estimate, and
  farm estimate. The single blended projected score makes it hard to tune farmer
  timing without hurting midgame tactics.
- Report paired seed swings, not only aggregate win rate, so seat and draw
  variance are visible during tuning.
