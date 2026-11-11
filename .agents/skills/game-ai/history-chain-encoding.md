# History Chain Encoding — Feature Recipe

How to feed a **bounded window of per-round action history** into the policy/
value net when the game's decisions depend on the recent sequence of plays,
not just the current board. Worked example: **Maskmen** (`games/maskmen/`) —
current round + 8 rounds of history, each up to 6 plays.

This is a feature-encoding pattern on top of [encoder.md](encoder.md) (generic
encoding, perspective rotation, normalization) and [tracker.md](tracker.md)
(the tracker stores the history because it is public-derived memory). Read
those first.

## When this recipe fits

The current state alone does not determine the right move — the net needs to
see **what was played recently, by whom, in what order**. Examples:

- Trick-taking where "is this card known-stronger / unknown-relation" depends
  on the sequence of prior plays in the round (Maskmen).
- Bluff games where a claim's credibility depends on the claim/bid sequence.

If a single `last_play_*` scalar suffices (Love Letter: only the just-played
card matters), do NOT use a chain — encode the scalar. The chain is for games
whose decision needs a **sequence window**, not a single last-step.

## Where the history lives: the tracker, not the schema

The play history is **public-derived memory** — fully reconstructable from
the public event stream, not needed by the rules engine. Per tracker.md
"What goes on tracker vs schema," it belongs on the `IBeliefTracker`, not as
schema fields. Keeping it off the schema keeps the DAG hash narrow (history
does not bloat every node's hash) and avoids snapshot cost.

```cpp
struct MaskmenPlay { int actor; int type; int size; };

// On the tracker:
std::vector<MaskmenPlay> current_round_plays_;            // this round, append-only
std::vector<std::vector<MaskmenPlay>> history_plays_;     // past rounds, most-recent-first
```

The tracker rebuilds this from `observe_public_event` — it never reads truth.

## Tracker: append within a round, archive on round end

```cpp
void observe_public_event(int /*actor*/, ActionId /*action*/,
                          const std::vector<PublicEvent>& events) {
  for (const auto& ev : events) {
    if (ev.first == "played") {
      current_round_plays_.push_back({player, type, size});   // append in play order
    } else if (ev.first == "round_end") {
      history_plays_.insert(history_plays_.begin(),
                            std::move(current_round_plays_)); // newest round at front
      current_round_plays_.clear();
      if ((int)history_plays_.size() > kMaxHistoryRounds)
        history_plays_.pop_back();                            // drop oldest -> bounded window
    }
  }
}
```

Two storage invariants:
- `history_plays_` is kept **most-recent-first** (`insert(begin, …)`) so
  encoding reads it in natural order. Bound the window with `pop_back` —
  unbounded history grows memory and feature dim.
- `current_round_plays_` is **append-order** within the round; the encoder
  reverses it. Do not pre-reverse on storage — keep storage simple, reverse
  at encode time.

## Encoding: fixed-length slots, reverse order, padding

The net input is a **fixed-size** tensor. Three blocks, all most-recent-first:

```
1. Current round chain:  up to kMaxChainLen(6) plays, reversed, pad to 6
2. History chains:       up to kMaxHistoryRounds(8) rounds × 6 plays each,
                         each round reversed, rounds most-recent-first, pad
3. current_round_idx:    completed_rounds / 15.0   (a single normalized scalar)
```

```cpp
// Current round: reverse so most-recent play is first.
for (int i = cur_len - 1; i >= 0; --i)
  push_play(cur[i], perspective, /*is_first=*/i == 0, out);
push_padding(kMaxChainLen - cur_len, out);          // pad short rounds

// History: rounds are already most-recent-first in storage; reverse each round.
for (int r = 0; r < num_rounds; ++r) {
  for (int i = rlen - 1; i >= 0; --i)
    push_play(hist[r][i], perspective, i == 0, out);
  push_padding(kMaxChainLen - rlen, out);
}
push_padding((kMaxHistoryRounds - num_rounds) * kMaxChainLen, out);  // pad empty rounds
```

Why reverse (most-recent-first)? The recent plays matter most for the
decision; putting them at the start of the chain block keeps them adjacent in
the input and lets a feed-forward net weight them without a long lookback.

## Per-play encoding: type + size + rotated actor + first-in-round flag

Each play occupies a fixed `kChainPerPlay` slots:

```
type one-hot (kTypeCount)           // which card/token type was played
size / kMaxPlaySize                  // how many (normalized — see encoder.md)
actor one-hot, perspective-rotated   // (actor - perspective + N) % N  (see encoder.md)
is_first_play_in_round               // 1.0 if this was the round's opening play
```

- The actor is **perspective-rotated** (`(actor - perspective + N) % N`) so
  the net sees "me / next / …" regardless of absolute seat — same convention
  as encoder.md "Perspective rotation."
- The `is_first_play_in_round` flag marks round boundaries inside the chain,
  so the net can distinguish "opening play" from "follow-up" without a
  separate structure.

## Why fixed-length padding, not a recurrent head

The framework's encoder produces a **flat `vector<float>`** consumed by a
feed-forward policy/value net (no recurrence in the default architecture).
A bounded window + padding turns a variable-length history into a fixed-size
block the flat encoder can emit. This trades perfect long-range memory for
architectural simplicity. If a game truly needs unbounded history, that is a
recurrent/transformer-head design decision outside this recipe — but most
games' relevant history is short (Maskmen: 8 rounds), so the window suffices.

## Invariants

- **History is public-derived → lives on the tracker, not the schema.** It
  must rebuild from `observe_public_event` alone; never read truth. (tracker.md
  contract; `test_tracker_observation_only`.)
- **Bounded window.** `current_round_plays_` + `history_plays_` are capped
  (`kMaxChainLen`, `kMaxHistoryRounds`); overflow drops the oldest. Unbounded
  history = unbounded feature dim + memory leak.
- **Fixed-size encoding.** Every chain block pads to `kMaxChainLen`; empty
  rounds pad fully. `feature_dim` is a compile-time constant — do not make it
  depend on how many plays actually happened.
- **Reverse at encode, not at store.** Storage is append-order + most-recent-
  round-first; the encoder reverses each round. Mixing the two creates
  off-by-one ordering bugs that are invisible in metrics but corrupt the
  sequence the net sees.
- **Perspective-rotate the actor.** A play's actor must be encoded relative to
  the observer, or the same physical history looks different to each seat —
  breaking perspective-invariance (`test_tracker_perspective_invariance`).
- **Tracker-null safe.** When the encoder is called with no tracker (e.g.
  state-only encode), push zero padding for the whole chain block — never
  deref a null tracker. (Maskmen's `push_chain_blocks` does this.)
- **Shared between PV and belief extractors.** If both the policy/value
  feature extractor and the belief feature extractor need the history, factor
  the chain push into one helper (`push_chain_blocks`) and call it from both —
  two divergent encodings of the same history is a silent training bug.
