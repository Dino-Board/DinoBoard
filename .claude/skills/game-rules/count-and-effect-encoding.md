# Count Form for Multisets, Effect-Based Action IDs

How to represent **unordered multisets** (hands, markets, supplies) and how
to encode **actions on those multisets** so equivalent positions hash to the
same DAG node and the action space stays small. Worked example: **Jaipur**
(`games/jaipur/`).

## Why a recipe exists

Two failure modes recur whenever a game has hands or markets that don't care
about card order:

1. **Slot/positional state**: `hand[i] = card_id`, `market[i] = card_id`. Two
   permutations of the same hand hash differently → MCTS explores them as
   distinct nodes → no DAG reuse → simulations wasted on equivalent
   positions, and the encoder/policy head sees a meaningless slot ordering.
2. **Slot-indexed actions**: "play card from hand slot 2", "take market
   slot 0". The legal-action set then depends on the arbitrary slot
   ordering, the action space inflates with maximum hand/market size, and
   slot-2 of one canonical hand is not the same effect as slot-2 of another
   — the policy head can't generalize across permutations.

Both problems are fixed at the same time by the recipe below.

## Recipe

### State: count vectors, not slots

For any unordered collection (hand, market, supply, deck-of-fixed-types):

```cpp
// Hand as a count vector: hand_count[p][type] = number of cards of that type
std::array<std::array<int, kNumTypes>, kNumPlayers> hand_count{};
// Market the same way (extra slot for "wild" / camels / etc. if applicable):
std::array<int, kNumTypes + kExtras> market_count{};
```

Equivalent multisets hash identically because the bytes are identical. The
framework's per-perspective hash walker sees one canonical encoding.

**When to use slots instead.** If the cards' positions carry meaning (a
tableau where the leftmost is the next to discard, a row where adjacency
matters), keep slots — the order is part of the state. If you find yourself
sorting slots before hashing, that's a sign you should be using counts.

### Actions: effects, not slot indices

Encode actions by what they DO, indexed over the **type / effect space**, not
over hand/market positions. For Jaipur:

```cpp
// Idle phase actions:
//   0..5   sell good type g (sells ALL of type g)
//   6..11  take ONE card of good type g from market
//   12     take all camels
//   13     enter exchange
constexpr int kActSellBase   = 0;
constexpr int kActTakeBase   = 6;
constexpr int kActTakeCamels = 12;
```

A single action id describes a transformation on count vectors
(`hand_count[me][g] += 1`, `market_count[g] -= 1`), independent of any slot
order that doesn't exist anyway.

**Three independent action sub-spaces for multi-stage choices.** When a turn
has sub-phases that each select from the same domain (e.g. Jaipur's exchange
selects market goods, then camel count, then hand goods), give each phase
its own non-overlapping id range. **Do not reuse the idle "take type g" id
for "select market type g during exchange"** — the legal-actions set must be
phase-specific, and a shared id forces awkward gating logic. Jaipur:

```cpp
//   14..19 (exchange phase 1) select market good g to take
//   20     confirm market selection
//   21..28 (exchange phase 2) give K camels (K = action - 21)
//   29..34 (exchange phase 3) select hand good g to give
//   35     cancel exchange
```

The total action space stays small (Jaipur: 36, including a 2-card market of
6 types; the same game with slot-indexed actions over a max-7 hand and 5-slot
market would be triple that).

### `do_action_fast`: count arithmetic

```cpp
case Action::TakeOneOf(g): {
  --d.market_count[g];
  ++d.hand_count[me][g];
  break;
}
case Action::SellAllOf(g): {
  int n = d.hand_count[me][g];
  d.hand_count[me][g] = 0;
  // pay the top n tokens off pile g, award bonus token if n>=3, etc.
  break;
}
```

No "find the slot of the played card" step — there are no slots.

### `legal_actions`: derive from counts

```cpp
for (int g = 0; g < kNumTypes; ++g) {
  if (d.market_count[g] > 0 && hand_room_for_one(d, me))
    out.push_back(kActTakeBase + g);
  if (d.hand_count[me][g] >= sell_min(g))
    out.push_back(kActSellBase + g);
}
```

Legal-action membership is a property of the counts, not of any ordering.

## What this buys

| Benefit | Why |
|---------|-----|
| **DAG reuse** | Permutations of the same hand/market hash to one node; MCTS does not re-explore equivalents. |
| **Smaller action space** | One id per *effect*, not one per (slot × effect). Fewer logits, lower variance per training step. |
| **Encoder generalization** | The policy head sees `hand_count[g]` directly; learning "lots of cloth → sell cloth" doesn't depend on slot ordering. |
| **Simpler legality** | `legal_actions` is a function of counts; no slot-availability bookkeeping. |
| **Cleaner public events** | An exchange's "take/give types" is recoverable from the market `count` delta (`db→da`) — no need to track which slot moved where. |

## Invariants

- An unordered collection is **either** a count vector **or** a slot array,
  never both. Mixing them (e.g. counts in the schema + a parallel slot list
  for animation) means two sources of truth and a desync waiting to happen.
  Animations should re-derive slots from the count vector at render time.
- Action ids are by effect type. An id for "sell good 0" must mean the same
  transformation regardless of what the hand looks like.
- Multi-phase sub-protocols: distinct id ranges per phase. The phase tag
  (`exchange_phase` etc.) gates which range is legal; never overload an id
  across phases — see also `phase-state-machine.md`.
- If a hidden collection is owner-only, the **count form is what gets viz-
  masked**. The opponent's `hand_count` is hidden as a unit; the belief
  tracker resamples count vectors, not slot arrays. See
  `deck-maintenance.md` for the deck side of the same trick.
- Public events derived from count deltas: when emitting `goods_taken` /
  `goods_sold` / `exchange_resolved` for the belief tracker, derive the
  type list from `(d_after.market_count[g] - d_before.market_count[g])` —
  do **not** trust action-side scratch state (`exchange_take_count`) on the
  cancel path, because cancel rolls those scratch fields back without
  moving cards. (Jaipur regression: an early version emitted
  `exchange_resolved` on cancel from the scratch arrays and corrupted
  `net_in`. The fix is to read the public count delta.)

## Worked example: Jaipur

- Market: `market_count[7]` (6 goods + camels), 5 cards face-up filled by
  the deck.
- Hands: `hand_count[2][6]` (camels separate as `camel_count[2]`).
- Idle actions: 14 ids for the four legal effects (sell × 6, take × 6,
  take camels, enter exchange).
- Exchange: three independent sub-phases with non-overlapping id ranges
  (14..20 market, 21..28 camel count, 29..34 hand, 35 cancel) — total
  action space 36 vs. ~100+ for a slot-indexed encoding over a 5-slot
  market and 7-slot hand.

See `games/jaipur/jaipur_state.h` (constants + count fields) and
`games/jaipur/jaipur_rules.cpp` (`legal_actions` / `do_action_fast` count
arithmetic) for the full implementation. The exchange phase machine itself
follows `phase-state-machine.md`.
