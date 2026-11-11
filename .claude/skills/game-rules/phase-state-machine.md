# Intra-Turn Phase State Machine — Pure Game-Side Recipe

How to model a game where a single turn has **multiple sequential sub-phases**
(declare → challenge → counter → resolve → exchange …) that the framework's
`current_player` rotation alone cannot express. Worked example: **Coup**
(`games/coup/`) — 11 stages.

## When this recipe fits

The framework gives you one built-in axis of progression: `current_player()`
rotates between top-level turns. That is enough for games where each turn is a
single decision (Love Letter: play a card, done). It is NOT enough when one
turn internally walks through a sub-protocol with its own actor changes and
legal-action sets:

- A claim can be challenged, then countered, then the counter challenged.
- A resolved challenge forces a reveal, then a redraw, then a lose-influence.
- An exchange draws cards, then asks the player to return some, in stages.

If your turn has sub-steps where **who acts** and **what they may do** both
change within the turn, model it with an explicit `stage` field. Do not try
to cram it into `current_player` overloading or implicit rules state.

## The pattern: one stage field, two switch projections

Declare a stage enum. Make it an `all_public` schema slot. Then project the
state machine twice: `legal_actions` switches on stage to return the legal
set, `do_action_fast_impl` switches on stage to execute and transition. These
two switches are **the same transition graph seen from two sides** — keep them
in lockstep.

### Schema

```cpp
enum class CoupStage : std::int8_t {
  kDeclareAction = 0, kChallengeAction = 1, kResolveChallengeAction = 2,
  kChooseLoseInfluence = 3, kCounterAction = 4, kChallengeCounter = 5,
  kResolveChallengeCounter = 6, kChooseLoseInfluenceCounter = 7,
  kExchangeReturn1 = 8, kExchangeReturn2 = 9, kLoseInfluenceFromAction = 10,
};

CoupStage stage = CoupStage::kDeclareAction;

// stage is all_public — it MUST be a schema slot.
viz::declare_field(schema, "stage", viz::all_public({}, NPlayers));
```

### legal_actions — stage dispatch (one side of the graph)

```cpp
switch (d.stage) {
  case CoupStage::kDeclareAction:        /* income/foreign aid/tax/steal/... */ break;
  case CoupStage::kChallengeAction:
  case CoupStage::kChallengeCounter:     actions = {kChallenge, kAllow}; break;
  case CoupStage::kResolveChallengeAction:
  case CoupStage::kResolveChallengeCounter: /* reveal slot 0 or 1 */        break;
  case CoupStage::kChooseLoseInfluence:
  case CoupStage::kChooseLoseInfluenceCounter:
  case CoupStage::kLoseInfluenceFromAction: /* lose slot 0 or 1 */          break;
  case CoupStage::kCounterAction:        /* block-or-allow, action-specific */ break;
  case CoupStage::kExchangeReturn1:
  case CoupStage::kExchangeReturn2:      /* return one of the drawn cards */ break;
}
```

### do_action_fast_impl — stage dispatch + transition (the other side)

```cpp
switch (d.stage) {
  case CoupStage::kDeclareAction:
    // some actions end the turn (income): advance_turn
    // some open a challenge window: d.stage = kChallengeAction
    // some skip straight to resolve: enter_counter_or_resolve
    break;
  case CoupStage::kChallengeAction:
    // challenge  -> d.stage = kResolveChallengeAction
    // allow      -> poll next challenger, or enter_counter_or_resolve
    break;
  // ... each case sets d.stage to exactly one successor
}
```

## Why `stage` MUST be an all_public schema slot

This is the invariant that prevents a desync class of bug. `stage` determines
`legal_actions`. If two session states agree on everything except `stage`,
they return different legal sets — but if `stage` is not in the hash, the DAG
treats them as the same node and reuses an edge that is illegal in one. The
symptom is identical to the harmonies `animal_deck` regression: an AI move
that is illegal in truth, surfacing mid-training.

Making `stage` `all_public` puts it in the hash, the snapshot, and the encoder
(usually a one-hot global feature). Session and truth then agree on the stage
byte-for-byte, and the perspective hash correctly distinguishes
"declare-action node" from "challenge-action node" even though `current_player`
may be the same seat.

## Encapsulate transitions, don't scatter them

Coup factors the repeated transition logic into three helpers rather than
re-writing the next-stage logic in every case:

| Helper | Effect |
|--------|--------|
| `advance_turn(d)` | reset stage to `kDeclareAction`, clear per-turn scratch (challenger, challenge flags, exchange_drawn), rotate `current_player` to the next alive seat |
| `resolve_action_effects(s, rng)` | apply the declared action's effect, then `advance_turn` (or route to exchange / lose-influence) |
| `enter_counter_or_resolve(s, rng)` | if the action is blockable, set `kCounterAction`; else `resolve_action_effects` |

Each `do_action_fast_impl` case sets `d.stage` to exactly one successor (or
calls a helper that does). There is no "fall through to an ambiguous state" —
every action leaves the machine in a well-defined stage. This is what makes
the 11-stage graph auditable: trace any case, see its single successor.

## `current_player` changes meaning per stage

Within a turn, `current_player` is NOT always the acting player — it is "the
seat that must act next in the current stage":

- `kDeclareAction`: the active player.
- `kChallengeAction` / `kCounterAction`: polled around the table
  (`challenge_check_index` walks alive seats excluding the active player).
- `kResolveChallengeAction`: snaps back to the active player (they reveal).
- `kChooseLoseInfluence`: the challenge loser.

This is fine — `current_player()` is the framework's single "who acts now"
signal, and the stage reinterprets it. Just make sure every stage transition
sets `current_player` to the correct seat for that stage, or `legal_actions`
will be computed for the wrong player. The `legal_actions` switch reads
`d.current_player` / `d.active_player` as the stage dictates.

## Worked example: Coup

11 stages model the full declare→challenge→counter→resolve→exchange
sub-protocol. Key transitions:

- `kDeclareAction` → `kChallengeAction` (claimable action) or `advance_turn`
  (income) or `kLoseInfluenceFromAction` (coup) or
  `enter_counter_or_resolve` (foreign aid).
- `kChallengeAction` → `kResolveChallengeAction` (someone challenges) or poll
  next seat / `enter_counter_or_resolve` (all allow).
- `kResolveChallengeAction` → `kChooseLoseInfluence` (claimant truth: redraw,
  challenger loses) or `advance_turn` (claimant bluff: claimant loses).
- `kExchangeReturn1` → `kExchangeReturn2` → `advance_turn`.

`stage` is one-hot encoded in the global feature block (kGlobalFeatures,
`stage OH(11)`), so the policy/value net sees the sub-phase directly. The
exchange-drawn cards are `all_hidden` with `reveal_slot_to` to the active
player during exchange — see `viz-tricks.md` for the reveal scoping.

## Invariants

- **`stage` is an `all_public` schema slot.** No exceptions. A hidden or
  off-schema stage = DAG reuse across different phases = illegal-move
  desync (same failure mode as harmonies `animal_deck`; see `state.md` Key
  invariants).
- **`legal_actions` and `do_action_fast_impl` switch on the same `stage` and
  describe the same graph.** Every action `legal_actions` returns for a stage
  must be handled by that stage's `do_action_fast_impl` case, and every case
  must set exactly one successor stage (directly or via a helper). A mismatch
  = an action that is legal but unhandled, or a transition to a stage with no
  legal actions.
- **Every stage transition sets `current_player` correctly for the new
  stage.** The stage reinterprets whose turn it is; leaving `current_player`
  stale makes `legal_actions` compute for the wrong seat.
- **Stage-internal scratch (challenger, challenge flags, exchange_drawn) is
  reset on `advance_turn`.** Per-turn state must not leak across turns —
  `advance_turn` is the single reset point.
- **`do_action_deterministic_impl` must replay the same stage transitions**
  (for tail-solver games). If your game has both a phase machine and a tail
  solver, the deterministic path walks the same graph suppressing randomness
  — see `tail-solve.md`.
- Prefer few stages. Coup's 11 is the high end; most turn-internal protocols
  need 2-4. If you find yourself adding a stage for every micro-step, check
  whether some are better modeled as a single stage with a richer action set.
