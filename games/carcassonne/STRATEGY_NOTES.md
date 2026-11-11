# Carcassonne Strategy Notes

This file records strategy notes for tuning the Carcassonne heuristic and
value head. It is intentionally practical: keep only ideas that can become
features, weights, or eval experiments.

## Local Game Shape

- Current implementation targets the base 2-player game.
- Tiles are represented as 24 types, `A` through `X`.
- The physical game has 72 tiles including the starting tile. Locally,
  `kDeckSlots = 71` and `kHistorySlots = 72`, because the `D` start tile is
  placed before the draw deck is built.
- Farmers are included in the rules and are scored only at final scoring.
- The web / AI representation uses normal tile orientation in the deck; legal
  placement rotations are actions, not pre-randomized deck state.

## Strategy Takeaways

### Meeple Economy

- A meeple on a short city, short road, or nearly surrounded cloister is usually
  stronger than a meeple locked for many turns in a large unfinished feature.
- Running out of meeples is a serious tactical failure: it prevents claiming
  surprise cloisters, small cities, and high-leverage endgame features.
- Passing on a meeple can be correct when the remaining supply is low or when
  the feature is unlikely to complete soon.
- Heuristic implication: supply penalty should be strong in the opening and
  midgame, then relax near the end when there are fewer future opportunities.

### Cities

- Small cities are efficient because they score quickly and return meeples.
- Large cities are high variance: they can be valuable, especially with shields,
  but an opponent can make completion awkward or impossible.
- Joining an opponent's city can be better than blocking when it creates shared
  full points or gives a majority swing.
- Heuristic implication: reward completing owned cities, reward small city
  meeple placements, penalize large incomplete city commitments unless there is
  clear completion potential or a majority/steal tactic.

### Roads

- Roads are often lower ceiling than cities or farms, but they are useful for
  quick points and meeple cycling.
- Long open roads can trap meeples. Short roads ending at junctions, cities, or
  cloisters are safer.
- Heuristic implication: roads should get modest positive prior when completion
  is near, but open road meeples should be conservative when supply is low.

### Cloisters

- Cloisters can be reliable 9-point targets when placed in dense areas.
- A cloister placed near existing tiles is much stronger than one isolated in
  open space.
- Heuristic implication: cloister placement value should scale with occupancy,
  and nearly complete cloisters should get strong meeple priority.

### Farms

- Farmers are a major endgame scoring source, but they never return to supply.
- Empty farms with no completed or likely cities are weak early commitments.
- Valuable farms are fields touching multiple completed or easy-to-complete
  cities, especially when they can be secured with one meeple or when the
  opponent's farm can be contested.
- Farm wars can consume too many meeples; this is good only when the expected
  endgame swing is large.
- Heuristic implication: farm meeple weights should be phase-dependent. Early
  farm commitments need a strong penalty unless the field already has real city
  value. Midgame can start valuing farm access. Endgame should use projected
  final scoring rather than only current scoreboard margin.

### Blocking And Sharing

- Carcassonne is not solitaire. Good tile placement can deny opponent
  completion, make a feature harder to finish, or connect into a feature to
  share/steal points.
- Blocking is more valuable when it locks an opponent meeple and the opponent
  is low on supply.
- Sharing is attractive when both players score full points but our alternative
  use of the tile/meeple is weak.
- Heuristic implication: current rule prior mostly scores local gain/loss; a
  stronger version should estimate opponent locked meeple value and completion
  difficulty, not only immediate feature ownership.

## Phase Plan For Heuristic Tuning

### Opening

- Prefer fast city/cloister points and meeple cycling.
- Avoid speculative farmers unless the field already touches completed cities or
  an unusually strong farm shape.
- Keep enough meeples in reserve.

### Midgame

- Balance immediate scoring with securing one meaningful farm.
- Contest opponent farms only when the projected field value is high.
- Prefer placements that keep the board compact enough to complete features.

### Endgame

- Switch value head from current score margin toward projected final score.
- Farmers, unfinished city/road/cloister points, and remaining meeple
  opportunities matter more than raw current score.
- Blocking opponent completions and preventing farm joins can be worth more
  than small immediate points.

## Research Findings (2026-06-22)

A focused audit of the rule heuristic + PV evaluator vs. STRATEGY_NOTES + the
rules engine surfaced one correctness bug and several disabled-but-wired
strategy levers. Recorded here so the next round of weight tuning has a
hypothesis list rather than a knob soup.

> **Note**: line numbers in this section reference the heuristic code AT THE
> TIME of the audit (commit `41545bb`). The heuristic was subsequently
> refactored into per-feature helpers and phased weights — see
> "Implementation status" below for what landed and where current code lives.
> Field names like `ownership_delta` / `owner_share` may not match current
> code; the named bugs and findings are still accurate descriptions of what
> the heuristic was doing pre-fix.

### Finding 1 — Tied-majority is mis-evaluated (bug, not a deliberate bias)

Rules: `award_to_majority` (`carcassonne_rules.cpp:271-283`) gives **full
points to every tied-best owner** — a 1-1 city of 8 effective points pays +8
to BOTH players. The heuristic's `ownership_delta` (`carcassonne_rule_heuristic.cpp:259-272`)
returns `points - points = 0` for ties, then multiplies that delta into
`owned_city_complete`/`owned_road_complete` etc. So the heuristic models
"contest into a tied majority" as zero gain when the actual game payout is
+full.

The PV evaluator has the symmetric error: `owner_share` (`carcassonne_heuristic_pv_evaluator.cpp:50-56`)
divides points by `owners[player]` count for tied owners. So a 1-0 city
projected as 8 to player; a 1-1 city projected as 4 to each. Both numbers
disagree with the rule's "8 to each tied owner" semantics.

This systematically discourages parasiting opponent's open cities/roads —
a strategically important play that the notes call out under "Blocking And
Sharing" — because the value head sees no margin gain.

Fix shape: rewrite `ownership_delta` to compute per-player awards
explicitly (`tied → full to each`, `loser → 0`, `winner → full`) and
return `me_award - opp_award`. Apply the same fix in `owner_share`.

### Finding 2 — Wired strategy levers, all zero in production

`RuleWeights` declares fields whose intent matches STRATEGY_NOTES but
which `sharp_weights()` (the production default) leaves at 0:

| Field (heuristic.cpp line) | Strategy intent |
|---|---|
| `tile_cloister_complete`, `tile_cloister_near` (78-79) | "Reward placing the tile that completes my own cloister" — only `sharp_cloister_*` enables |
| `farm_potential_base`, `farm_potential_progress` (80-81) | "Field touching multiple incomplete cities is a strong potential farm" |
| `city_completion_bonus`, `city_open_penalty`, `opp_city_open_bonus` (82-84) | "Open-edge count is the real completability signal, not just tile count" |
| `road_completion_bonus`, `road_open_penalty`, `opp_road_open_bonus` (85-87) | Same for roads — penalize trapped meeples on long-open roads |
| `city_meeple_open_penalty`, `road_meeple_open_penalty` (88-89) | "A meeple on an N-open feature is locked for ~N more turns" |

The fields exist; they just aren't turned on in sharp. Three of these
(`farm_potential_*`, `tile_cloister_*` outside cloister profile,
`*_open_penalty`) are the most direct expression of strategy themes that
the notes mark as important but the production heuristic is blind to.

### Finding 3 — `farm_supply` likely overweighted in sharp

`sharp_weights().farm_supply = 2.35` is multiplied by `supply_penalty`
(midgame ~1.0+) and subtracted from farm-meeple score. For a 2-completed-
city farm at progress 0.4: gain ≈ `farm_points_base + farm_points_progress *
0.4 = 0.32 + 0.42 = 0.74` × 6 = 4.4; loss ≈ `2.35 × 1.0 = 2.35` plus
`farm_empty_*` = 1.8 if completed_cities==0. Subtraction dominates more
often than the strategy notes seem to want. A sweep over
`farm_supply ∈ {1.4, 1.7, 2.0, 2.35}` would tell us whether 2.35 is
correctly tuned or merely conservative.

### Finding 4 — PV evaluator dead branch + cloister underprojection

In `projected_scores` (`carcassonne_heuristic_pv_evaluator.cpp:63-112`):

- The `info.complete` branch at line 85-89 is effectively dead. After a
  feature completes, `return_feature_meeples` clears the meeple owner on
  the cell, so `s.board_meeple_owner[cell] < 0` skips it at line 73-74.
  Worth a comment but not a bug.
- Cloister projection at line 100-102 uses **current** occupancy, not
  expected end-game occupancy. With deck remaining, a partially-filled
  cloister will likely fill more — the value head undervalues an early
  cloister meeple's expected payout.

Cheap fix: when projecting, scale `cloister_occupancy` by an expected-
fill factor `min(9, occupancy + remaining_neighbors_estimate)`, where
the estimate is `min(empty_neighbors, deck_count / k)` for some k.
Defer until we have evidence this matters in arena.

### Candidate features ranked by leverage / cost

1. **Tied-majority share fix** — `ownership_delta` + `owner_share`. Fixes a
   correctness bug; expected +3-5% winrate via the share/contest line.
2. **Open-edge terms in sharp** — flip `city_completion_bonus=0.6`,
   `city_meeple_open_penalty=0.4`, `road_open_penalty=0.25`,
   `road_meeple_open_penalty=0.35`. Zero new code.
3. **`tile_cloister_complete` in sharp** — set to 0.9, `tile_cloister_near`
   to 0.12. Production currently scores zero for "completing my own
   cloister with this tile", which is +9 raw value.
4. **Farm potential** — `farm_potential_base=0.10`,
   `farm_potential_progress=0.30`, plus an `incomplete_cities` term in
   place-action farm scoring at half weight of completed.
5. **`meeple_lockup_penalty`** (new term) — combine `open_edges + tiles*0.25`
   with `supply_penalty` in city/road meeple scoring. Captures "meeple
   turnover rate" the notes emphasize.

### Experiment plan (A/B with `eval_model.py`)

All comparisons use the heuristic-only path (no model) at sim 2000 wc 200,
400 games / 200 seeds-per-side, rule-vs-rule. Track winrate and average
margin.

```
# 1. Share fix isolated
python3 scripts/eval_model.py --game carcassonne \
  --left rule:rule_mcts_sharp_pv_soft --right rule:rule_mcts_sharefix \
  --games 400 --seed 42

# 2. Open-edge terms in sharp
python3 scripts/eval_model.py --game carcassonne \
  --left rule:rule_mcts_sharp_pv_soft --right rule:rule_mcts_sharp_open_edge \
  --games 400 --seed 42

# 3. Cloister-completion in sharp
python3 scripts/eval_model.py --game carcassonne \
  --left rule:rule_mcts_sharp_pv_soft --right rule:rule_mcts_sharp_cloistercomplete \
  --games 400 --seed 42

# 4. Farm potential
python3 scripts/eval_model.py --game carcassonne \
  --left rule:rule_mcts_sharp_pv_soft --right rule:rule_mcts_sharp_farmpotential \
  --games 400 --seed 42

# 5. farm_supply sweep
for fs in 1.4 1.7 2.0 2.35; do
  python3 scripts/eval_model.py --game carcassonne \
    --left rule:rule_mcts_sharp_farm$fs --right rule:rule_mcts_sharp_pv_soft \
    --games 200 --seed 42
done
```

Confirm any rule-level winner with PV-MCTS on top (the ranking can flip
when MCTS smooths over heuristic noise). If a rule-level winner doesn't
survive MCTS, the win came from prior shape rather than judgment quality
— prioritize the PV-evaluator-side fix (Finding 1) over rule weight changes.

### Implementation status (2026-06-22)

Wired into the codebase but NOT yet validated against production sharp:

- **`rule_mcts_sharefix`** — new profile `kSharpSharefix` in
  `carcassonne_rule_heuristic.cpp::sharp_sharefix_weights`. Replaces the
  old `owned_*_complete` / `my_*_*` / `opp_*_*` ownership branches with
  `marginal_feature_delta` that respects `award_to_majority` semantics
  (tied owners both get full points). Before-state regions are collected
  via `CarcassonneRules::feature_group_for_side` (newly exposed) and
  deduplicated by neighbor cell + group.
- **`rule_mcts_sharp_open_edge`** — new profile `kSharpOpenEdge`. Sets
  `city_completion_bonus=0.6`, `city_meeple_open_penalty=0.4`,
  `road_*` analogs, and `opp_*_open_bonus` so the open-edge structural
  signals fire in sharp.
- **`rule_mcts_sharp_cloistercomplete`** — new profile
  `kSharpCloisterComplete`. Sets `tile_cloister_complete=0.9`,
  `tile_cloister_near=0.12` on top of sharp.
- **`rule_mcts_sharp_farmpotential`** — new profile
  `kSharpFarmPotential`. Sets `farm_potential_base=0.10`,
  `farm_potential_progress=0.30`. (Note: place-time `incomplete_cities`
  contribution is part of a follow-up, not yet wired.)

PV evaluator fix (already in production for ALL profiles):

- `projected_owner_points` (`carcassonne_heuristic_pv_evaluator.cpp:50`)
  no longer divides by `owners[player]` count for ties. Now mirrors
  `award_to_majority` exactly. Affects every profile's value head — a
  uniform improvement, not gated on a profile flag.

To compare any new profile against production sharp, run:
```
python3 scripts/eval_model.py --game carcassonne \
  --left rule:rule_mcts_sharp_pv_soft --right rule:<new_name> \
  --games 400 --seed 42
```
Each new profile is registered with sim 2000 / world_cap 200 — same
budget as training selfplay, so winrates are comparable to what the
trained model would experience.

### Stack-up rules

When adding a new term, watch for these existing-term interactions:

- `meeple_lockup_penalty` ↔ `*_meeple_supply * supply_penalty` and
  `city_large_penalty`. Lockup uses open_edges; supply uses remaining; size
  uses tile count — three different signals overlap. Keep `lockup` modest
  and ablate against `*_meeple_supply` if both active.
- `farm_potential` ↔ `farm_empty_early_penalty`. If `incomplete_cities >= 1`,
  early-penalty already kicks in and potential cancels it. Gate
  `farm_potential` on `progress > farm_empty_until` or `completed_cities
  >= 1` until the regression is understood.
- Open-edge city/road bonuses ↔ `city_meeple_supply` / `road_meeple_supply`.
  Both penalize commitment. Sharp profile's supply weights are tuned
  without open-edge terms; turning open-edge on without re-tuning supply
  may double-discount large open features.
- `share_bonus` (post share-fix) ↔ `opp_city_*`/`opp_road_*` (which
  assume zero-sum). After the rewrite, those terms double-count opp
  ownership. Recompute as a single per-player award rather than
  patching on top.

## Sources

- Official rulebook PDF, for turn structure, feature scoring, and final scoring:
  https://cdn.svc.asmodee.net/production-zman/uploads/2024/10/Carcassonne-20th-Anniversary-Rulebook.pdf
- WikiCarpedia farmers page, for farmer placement and final-scoring details:
  https://wikicarpedia.com/car/The_Farmers
- Meeple Mountain strategy guide, for broad base-game strategic themes:
  https://www.meeplemountain.com/articles/carcassonne-strategy-guide/
- BoardGameGeek strategy thread, especially farmer timing and points-now versus
  points-later framing:
  https://boardgamegeek.com/thread/15879/strategy-guide
- Ticket to Carcassonne strategy guide, for tile efficiency, meeple reserves,
  and balancing immediate points with fields:
  https://tickettocarcassonne.com/carcassonne/
- STEM Geek strategy tips, for practical meeple reserve, small city, cloister,
  and field advice:
  https://stemgeek.com/carcassonne-strategy/
- My Board Game Guides strategy article, for score tracking, worker supply,
  tile tracking, and interaction/blocking framing:
  https://www.myboardgameguides.com/game-strategy/game-specific-strategy/carcassonne-strategies-and-tips/
- Chinese overview / strategy notes, useful for terminology and quick Chinese
  summaries:
  https://avalontactics.com/board-game-guides/carcassonne-gameplay-rules-strategy-guide
- Chinese 2-player focused PDF found during research:
  https://sirit.com.cn/sirit/uploads/2022/11/1760375348.pdf
