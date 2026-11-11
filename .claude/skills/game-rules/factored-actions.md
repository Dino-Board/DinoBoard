# Factored actions (compound options as atom sequences)

Worked example: **Ark Nova** (`games/arknova/arknova_actions.*`, `arknova_rules.*`).
Helpers: `engine/core/factored_action.h`.

## When to use

The native decision is a product of choices: card × building, shape × cell ×
rotation, "keep 4 of 8". Enumerating every option as one `ActionId` gives an
action space of tens of thousands with most ids never legal, a sparse policy
head, and branching factors in the thousands at the MCTS root.

Factor instead: each option becomes a short sequence of **atoms** from a small
fixed space, and the player picks one atom per ply. Search and the policy head
see the branching of one atom at a time.

## Pattern

1. **One mapper from option to atoms.** A single function turns a core option
   into its `AtomSequence`. Pointer atoms (hand card, display index, zoo card,
   map cell) point at a token the encoder emits, so the transformer can score
   them through pointer heads. Index unbounded collections by entity id (card
   id), not by slot, so they need no capacity.
2. **The prefix is public state.** Store the atoms chosen so far (`prefix`) in
   the state, inside a public viz field and the hash. The opponent sees the
   partial choice exactly as in the real game (a card is being played, it is
   not yet known where).
3. **`legal_actions` = `factored_next_atoms(options, prefix)`.** Build the
   option list, map it, run `check_factored_options` (non-empty, distinct,
   prefix-free), and return the next atoms.
4. **Apply when complete; skip forced atoms.** `do_action_fast` appends the
   atom, then loops `factored_extend_forced`: when the prefix spells out one
   option, apply it with the core `step`, clear the prefix, and continue while
   the next decision has a single continuation. Players are only asked where
   there is a real choice.
5. **Keep sequences prefix-free by construction.** Give every option of a
   decision the same shape (Ark Nova projects always end with `no_extra` or a
   zoo slot), or use distinct first atoms per shape.

## Sketch (rules file)

```cpp
bool settle_prefix(MyState& st, std::mt19937_64& rng) {
  Decision d = current_decision(st);            // options + their atom sequences
  int done = factored_extend_forced(d.atoms, st.prefix);
  if (done < 0) return false;                   // a real choice is pending
  core_step(st, d.options[done], rng);
  st.prefix.clear();
  refresh_viz(st);                              // viz writers stay in *_rules.cpp
  return true;
}
void apply_atom(MyState& st, ActionId a, std::mt19937_64& rng) {
  st.prefix.push_back(a);
  while (!st.terminal() && settle_prefix(st, rng)) {}
}
```

Call the same loop at the end of `reset_with_seed`, so the first decision
already has a real choice.

## Invariants

- The prefix only ever holds atoms of the *current* decision; it is empty
  whenever a new decision starts.
- Duplicate core options (same option listed twice) are deduped before
  mapping, otherwise `check_factored_options` throws.
- Unordered multi-picks ("discard 4 of 8") map to **ascending** atoms, one
  sequence per subset, never one per permutation.
- Placements identify a building by (lowest covered cell, rotation index), so
  the same footprint never gets two ids.
- Applying, forcing and the RNG used by the core step live in `*_rules.cpp`
  (viz writers and RNG rules apply there as in any game).

## Hidden information with a public prefix

A determinized world must keep the observed prefix legal. When the mover is
not the observer and the prefix is non-empty, the tracker resamples until
some option of the sampled world still starts with the prefix (Ark Nova:
`prefix_consistent`, at most 1000 tries, then throw). Cards the prefix names
are public, so place them first instead of hoping a sample draws them: the
opponent chose "play card 57 from hand", so the world's opponent hand holds
card 57.

An atom whose choice stays hidden (Ark Nova's Pouch tucks a hand card face
down) must not reach observers by id: override `IGameRules::public_action` to
map it to a never-legal stand-in atom. The framework passes that projection,
not the true action, to trackers, traces and AI views. Also check what the
hidden move does to *earlier* knowledge: after a face-down tuck the opponent
no longer knows which seen hand card went, so those cards become a public
"in the hand or tucked" set (Ark Nova `maybe_tucked`) that the encoder and
belief features expose and the sampler fills only into those two zones.

## Heuristics over atoms

A rule heuristic still thinks in core options. Score each option (searching
the core directly on a tracker-sampled world), then give each legal next atom
the best score among options below it (`ArknovaGreedyHeuristic`). Atoms whose
options were pruned get a score below every evaluated one.

## Testing

- Random play over atoms for many games: every atom returned by
  `legal_actions` must apply without throwing.
- Diff the core against a reference engine on core options, independent of
  the atom layer.
- Framework tests (`tests/framework`) cover masking and tracker invariants
  with the prefix in play.
