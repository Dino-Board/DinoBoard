# Bag Draw with Per-Token Counts — Pure Game-Side Recipe

How to implement "draw a token from an opaque bag, maintain per-type counts"
when the bag is a **public, count-only** reservoir (no hidden ordering to
protect). Worked examples: **Azul** (`games/azul/`) and **Harmonies**
(`games/harmonies/`).

## When this recipe fits

- Tokens of the same type are interchangeable — only the **count per type**
  matters, not which specific token.
- The bag's remaining composition is **publicly derivable** (start total minus
  what's been placed/discarded), so there is no information to hide.
- Draws are physical randomness consumed inside `do_action_fast` (no belief
  sampling needed — the bag is public).

This includes games where token *types* are heterogeneous (Carcassonne tiles
A..X with different edges) as long as **multiple copies of the same type
exist and are interchangeable** and remaining counts are publicly derivable.
Carcassonne, dominoes, Catan resource decks all fit here.

If the bag's *contents* are secret (opponents must not learn what's left), do
NOT use this recipe — that's a hidden-deck case; see `deck-maintenance.md`.

## ⚠️ Hard rule: NEVER store the draw order in state

The bag's state is a count array, period. The order in which a future random
draw will resolve is a property of the **RNG**, not of the game state.

**Never do any of these:**

```cpp
// ❌ WRONG — pre-shuffled ordered deck as a state field
std::array<std::int8_t, kDeckSlots> tile_deck{};   // shuffled at reset, popped each draw
viz::declare_field(schema, "tile_deck",
                   viz::all_hidden({kDeckSlots}, NPlayers));   // even all_hidden is wrong
```

```cpp
// ❌ WRONG — folding a deck order into state_hash, hand-written or otherwise
StateHash64 state_hash() const override {
  ...
  for (auto v : tile_deck) hash_combine(h, v);    // poisons TT keys with hidden info
  ...
}
```

Two failure modes this triggers, in increasing subtlety:

1. **Direct leak via public viz.** An ordered deck declared `all_public`
   hands the AI the next K draws on a plate. Don't do this.
2. **Hash leak via hand-written `state_hash`.** Even with `viz::all_hidden`,
   the framework's `state_hash_for_perspective` correctly substitutes
   `kHiddenHashSentinel` for hidden slots — but legacy paths that call the
   game's hand-written `state_hash()` (e.g. `tail_solver`'s TT key) bypass
   that walker entirely. If the hand-written `state_hash()` mixes the deck
   array, two sims with the same observation history but different
   determinized deck orders produce different TT keys, so the search "feels"
   the hidden order even without seeing it. (Real bug found in Carcassonne;
   the fix is structural — drop the deck order from state, not patch the
   hash.)
3. **Sim divergence on supposedly-equal states.** Two sim clones of the
   same observed position will, with an ordered deck in state, hold
   different deck orderings and hash differently — splitting one DAG node
   into many. Sample-efficiency tanks even when nothing leaks observably.

The only safe shape is: **count array in state, RNG outside state, draw is
a weighted sample at the moment it happens.** The recipe below.

## The single rule: counts are the truth, viz is all_public

Store one int per type. Declare it `all_public`. Draw by sampling a type
weighted by remaining counts. There is no per-token identity, no order, no
viz trick — the count array is the entire state of the bag.

### Schema

```cpp
// kColors = number of token types
std::array<int, kColors> bag_counts{};        // remaining in bag
std::array<int, kColors> box_lid_counts{};    // discarded, refilled when bag empties (Azul)

viz::declare_field(schema, "bag_counts",
                   viz::all_public({kColors}, NPlayers));
viz::declare_field(schema, "box_lid_counts",
                   viz::all_public({kColors}, NPlayers));   // omit if no refill (Harmonies)
```

### Draw (weighted by remaining counts)

```cpp
int draw_one_token(std::mt19937_64& rng) {
  int total = 0;
  for (int c : bag_counts) total += c;
  if (total == 0) {
    // Azul: refill bag from the box lid, then continue.
    for (int c = 0; c < kColors; ++c) {
      bag_counts[c] = box_lid_counts[c];
      box_lid_counts[c] = 0;
      total += bag_counts[c];
    }
  }
  int pick = std::uniform_int_distribution<int>(0, total - 1)(rng);
  for (int c = 0; c < kColors; ++c) {
    if (pick < bag_counts[c]) { --bag_counts[c]; return c; }
    pick -= bag_counts[c];
  }
  // unreachable
}
```

The draw is a single RNG call that picks a type proportional to its count.
The RNG is the caller's (`do_action_fast`'s `rng`), consumed in-place —
exactly the framework's "physical randomness in `do_action_fast`" model
(ALGORITHM_OVERVIEW.md §7, fully-public + physical-random games).

## Why no belief / tracker / randomize_unseen

The bag is `all_public`, so the session state and the truth state agree on
`bag_counts` byte-for-byte (snapshot carries it wholesale). MCTS sims clone
the session state and call `draw_one_token` with a fresh `sim_rng` — each sim
draws a different type, and the DAG handles the divergence via hash
differences in the *result* (the placed token), not the bag. No chance node,
no `randomize_unseen`. This is the same reason Azul and Harmonies register
no `belief_tracker`.

## Worked examples

### Azul — bag with box-lid refill loop

20 of each of 5 colors start in the bag (`reset_with_seed`). Factories are
refilled by calling `draw_one_tile`. When the bag empties mid-refill, the box
lid (discarded tiles) is poured back in (`bag_counts[c] = box_lid_counts[c]`;
`box_lid_counts[c] = 0`) and drawing continues. Both arrays are
`all_public` schema fields — "tiles within a color are interchangeable; only
counts are real, and counts are publicly derivable" (azul_state.cpp schema
comment).

### Harmonies — pouch with no refill

`pouch_counts[kColors]` starts at a fixed total (sum=120), also `all_public`.
`draw_one_token` (harmonies_rules.cpp) is the same weighted-by-count sampler.
The difference from Azul: **no box-lid refill** — when the pouch can no longer
fill a slot (`total_pouch < kMaxTokensPerSlot`), the game ends instead. If
your bag is single-use (draws never return), Harmonies is the closer model; if
your bag cycles discarded tokens back in, Azul is.

## "But I want reproducible draws" — RNG belongs in the AI session, not state

A common reason people put a pre-shuffled `tile_deck` in state is "I want
the deck order to be reproducible across a replay or across sim clones."
That goal is correct; the implementation is wrong. The fix:

- The RNG (`std::mt19937_64& rng`) is the framework's `do_action_fast` arg.
  Each AI session / replay seeds its own RNG and feeds it into every
  `do_action_fast` call. Reproducibility comes from the seed, not from
  baking a frozen permutation into state.
- Different sims SHOULD draw different tiles from the same observed
  position — that's what makes ISMCTS / chance-node search work
  (cf. `randomize_unseen` for hidden-info games; Azul's bag is the
  fully-public analog). Freezing the order across sims would collapse the
  search into a single rollout.
- If you need a deterministic full game (e.g. for a unit test), seed the
  game's RNG via `reset_with_seed` and let `do_action_fast` consume it
  call by call. The state stays count-only; the determinism rides the RNG.

## Invariants

- `bag_counts` (and `box_lid_counts` if present) MUST be `all_public` schema
  slots. Making them hidden would force a `randomize_unseen` to invent bag
  composition — wrong tool, since the composition is public.
- The count array is the **only** bag state. Do not also store a per-token
  vector or an order; interchangeable tokens have no identity.
- The draw RNG is consumed inside `do_action_fast` (caller's `rng`), never on
  the state. State holds no RNG.
- `state_hash` MUST NOT include any draw-order array, even if that array is
  `all_hidden`. The schema-driven `state_hash_for_perspective` handles
  hidden slots correctly, but legacy callers of the game's hand-written
  `state_hash()` will mix the raw bytes — and a hidden-but-mixed array
  poisons TT keys exactly the same as a public leak. The structural fix is
  to not have the array at all.
- If you add a refill cycle, both the bag and the discard pile are
  `all_public`; the refill is a pure count move (`bag = lid; lid = 0`), no viz
  change.
