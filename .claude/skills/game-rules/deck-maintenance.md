# Hidden Deck Maintenance — Pure Game-Side Recipe

How to maintain a deck whose **contents are secret** (opponents must not learn
the order or remaining cards) but whose **size is public**. Worked examples:
**Splendor** (`games/splendor/`) and **Harmonies** (`games/harmonies/`).

## When this recipe fits

- Cards/tiles are drawn from a deck whose order and remaining contents are
  hidden from everyone (a shuffled face-down deck).
- The deck **size** is public (everyone knows how many cards remain), even
  though which cards they are is not.
- Opponents hold a belief about deck composition that must be sampled for
  ISMCTS determinization.

This is the hidden-information counterpart to `bag-draw.md`. If the reservoir
is public counts (Azul bag), use that recipe instead — it needs no belief.

## The two-slot split: hidden contents + public size

Store the deck contents hidden, but expose the size as a **separate
first-class public schema slot**. Do not let observers derive size from the
hidden deck vector — that is the bug this recipe exists to prevent.

### Schema

```cpp
// Splendor: three tiers of variable-length decks.
std::array<std::vector<std::int16_t>, 3> decks{};   // contents — NOT a schema slot
std::array<std::int16_t, 3> deck_sizes{};           // public size — IS a schema slot

// deck_sizes is all_public; decks contents are resampled by randomize_unseen.
viz::declare_field(schema, "deck_sizes",
                   viz::all_public({3}, NPlayers));
```

`decks[t]` itself is never declared as a schema field — its contents are
hidden and resampled per-sim. Only `deck_sizes[t]` rides the snapshot, and
`read/write_field_slot` for `deck_sizes` project to/from `decks[t].size()`.

## Why size must be a public slot (the core invariant)

If deck size is NOT a public schema field, an observer session can only learn
it by reverse-deriving from the hidden `decks` vector — which it does not
have. Two sims then disagree on `decks[t].empty()`, so the **same hash key**
maps to DAG nodes where "reserve-from-tier-t" is legal in one sim and illegal
in another. The bandit silently corrupts. The Splendor state.h comment states
it directly:

> reading tier off a stale `reserved` cid in a sim produces inconsistent deck
> sizes across sims and lets the same DAG node disagree on whether
> reserve-from-deck-tier-X is legal.

The fix is structural: `deck_sizes` is a public schema slot, rules maintain it
at every draw / refill / reserve site, and `randomize_unseen` produces decks
whose size is exactly `deck_sizes[t]`.

## randomize_unseen — resample hidden contents to match public size

The tracker reconstructs deck contents from the unseen pool, then trims to the
public size:

```
unseen_by_tier[t] = full_pool[t] − seen_cards − observer-known reserved
shuffle each tier
for each observer-hidden reserve: deal one card from its tier's unseen pool
  (tier taken from the PUBLIC reserved_tier slot, not the hidden reserved cid)
decks[t] = remaining unseen_by_tier[t]   // size == deck_sizes[t] by construction
```

The tier for a hidden reserve comes from the public `reserved_tier[p, slot]`
field, never from reverse-deriving the hidden card id. This keeps every sim's
deck size byte-identical for the same observation history, so the DAG merges
correctly.

## Worked examples

### Splendor — three tiers, belief-tracked

`decks[3]` (variable-length, hidden contents) + `deck_sizes[3]` (public).
Blind reserves (`reserved[p][3]`, `owner_only_first_axis` base = blind) are
revealed face-up via `viz::reveal_slot` when reserved from tableau, reset to
base on buy/discard. `reserved_tier[p][3]` is public so `randomize_unseen`
knows which tier a hidden reserve consumed. `SplendorBeliefTracker` does
uniform multiset sampling from the unseen pool (no belief net — Splendor's
hidden info is shallow).

### Harmonies — single animal deck, all_hidden + public count

`animal_deck[kAnimalCards]` is `all_hidden` (which specific animals remain);
`animal_deck_count` is `all_public` (how many remain). `apply_take_card`
(harmonies_rules.cpp) reads the hidden deck to draw — but only on the GT path.
On the session side the deck is resampled: `HarmoniesBeliefTracker::
randomize_unseen` reconstructs the unseen pool as `full − market − all hands`
(publicly derivable) and fills `animal_deck` to size `animal_deck_count`.

> Harmonies originally had `animal_deck` public and no tracker; making it
> hidden required adding the tracker + `install_event_protocol` (see the
> `create_game` guard and `registration.md`). The lesson: flipping a deck
> from public to hidden is not a one-line viz change — it activates the whole
> belief-sampling path.

## Invariants

- **Size is a public schema slot; contents are not.** Never derive size from
  the hidden deck vector on the observer side.
- `randomize_unseen` must produce decks whose size equals the public
  `deck_sizes[t]` exactly — same observation history ⇒ same sizes ⇒ same hash.
- Any field the deck-draw legality depends on (e.g. `reserved_tier`) must be
  public, so sims agree on legality. Do not reverse-derive it from hidden
  contents.
- `decks[t]` contents are read by rules **only on the GT path** (truth has the
  real deck). The session side reads resampled contents via `randomize_unseen`
  on a cloned world — never the truth deck.
- Rules maintain `deck_sizes` at every mutation site (draw / refill / reserve
  / buy). A size that drifts from the real vector length is a silent
  hash-corruption bug.
