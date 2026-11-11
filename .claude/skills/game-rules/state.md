# IGameState Reference

## CloneableState CRTP

Every game state derives from `CloneableState<Derived>`, which auto-implements `clone_state()` and `copy_from()`:

```cpp
class MyGameState : public CloneableState<MyGameState> {
 public:
  // clone_state() and copy_from() are auto-generated
};
```

The default copy constructor must produce a correct deep copy. If you hold `shared_ptr<const T>` for COW data, the copy is shallow (shares the pointer) -- override `mask_all_hidden_slots` to detach before masking.

## Visibility Schema

Declared once per game in `<game>_state.cpp::schema()`. Static singleton pattern:

The schema is the single source of truth for both field shapes and base visibility masks. Framework walkers reuse it for perspective hashing, masked-state construction, and public snapshot I/O, so changing the schema changes both what slots exist and how hidden slots are masked.

```cpp
const viz::VisibilitySchema& MyGameState::schema() {
  static const viz::VisibilitySchema s = []() {
    viz::VisibilitySchema schema;
    schema.n_players = N;
    // Scalars: data_shape = {} (rank 0)
    viz::declare_field(schema, "current_player", viz::all_public({}, N));
    viz::declare_field(schema, "winner",         viz::all_public({}, N));
    viz::declare_field(schema, "terminal",       viz::all_public({}, N));
    // 1D array: data_shape = {len}
    viz::declare_field(schema, "scores",  viz::all_public({N}, N));
    viz::declare_field(schema, "board",   viz::all_public({BoardSize}, N));
    // Owner-only: first axis = owner, only owner sees own slots
    viz::declare_field(schema, "hand",    viz::owner_only_first_axis({N}, N));
    // All hidden: no viewer sees any slot
    viz::declare_field(schema, "deck",    viz::all_hidden({DeckSize}, N));
    return schema;
  }();
  return s;
}
```

### Base viz builder helpers

| Helper | Pattern | Use case |
|--------|---------|----------|
| `all_public(shape, N)` | viz = all 1s | scores, board, public counters |
| `owner_only_first_axis(shape, N)` | viz[owner,...,viewer]=1 iff owner==viewer | hands, private reserves |
| `all_hidden(shape, N)` | viz = all 0s | deck contents, set-aside cards |

`declare_field` enforces name uniqueness and viewer-count consistency. Declaration order defines `FieldId` values (0, 1, 2, ...).

## schema_ref() -- virtual accessor

```cpp
const viz::VisibilitySchema& schema_ref() const override {
  return MyGameState::schema();
}
```

Framework helpers (`make_masked_state`, hash walker, snapshot I/O) call this to drive the walker without knowing the concrete state type.

## reset_with_seed

```cpp
void MyGameState::reset_with_seed(std::uint64_t seed) {
  IGameState::reset_step_count_base();  // MUST be first line
  std::mt19937_64 rng(seed);            // one-shot RNG, dropped on return
  // Initialize all fields, shuffle deck, deal hands, etc.
  viz::init_viz(*this, schema());       // copies base_viz into state.viz_
}
```

- `reset_step_count_base()` zeroes the framework step counter (mandatory first line).
- `viz::init_viz(*this, schema())` initializes `state.viz_` from schema base_viz.
- RNG is local -- state holds no RNG.

## FieldId Enum and hash_field_slot

Define a FieldId enum matching schema declaration order, then switch-dispatch:

```cpp
// In <game>_state.cpp, anonymous namespace
namespace {
enum : viz::FieldId {
  kCurrentPlayer = 0,  // must match declare_field order
  kWinner,
  kTerminal,
  kScores,
  kBoard,
  kHand,
};
}

void MyGameState::hash_field_slot(
    Hasher& h, viz::FieldId fid,
    const std::vector<int>& idx) const {
  switch (fid) {
    case kCurrentPlayer: h.add(current_player_); return;
    case kWinner:        h.add(winner_ + 1); return;
    case kTerminal:      h.add(terminal ? 1 : 0); return;
    case kScores:
      h.add(scores[idx[0]] + 2); return;
    case kBoard:
      h.add(static_cast<int>(board[idx[0]]) + 2); return;
    case kHand:
      h.add(static_cast<int>(hand[idx[0]])); return;
  }
}
```

- Walker calls this for each visible slot. Hidden slots get `kHiddenHashSentinel` automatically.
- Do NOT re-consult `viz_` inside this method -- walker already filtered.
- `idx` covers data axes only (no viewer axis).

## hash_field_bulk / mask_field_bulk

Optional overrides for hot fields with many slots. The default walks per-slot and dispatches to `hash_field_slot` / `mask_field_slot`. Override when a tight inlined loop wins:

The example below is schematic: in real code, preserve the exact default protocol by mixing `fid`, every data-axis index, then either the visible value or `kHiddenHashSentinel`.

```cpp
void MyGameState::hash_field_bulk(
    Hasher& h, viz::FieldId fid,
    const viz::VizTensor& v, int perspective) const {
  if (fid == kBullheadPile) {
    // tight loop over the pile array, checking viz per slot
    for (int i = 0; i < PileSize; ++i) {
      h.add_position(fid, {i});
      size_t base = i * v.viewer_count();
      if (v.data[base + perspective])
        h.add(pile[i]);
      else
        h.add_sentinel();
    }
    return;
  }
  IGameState::hash_field_bulk(h, fid, v, perspective);  // default
}
```

## read_field_slot / write_field_slot -- snapshot wire I/O

```cpp
std::any MyGameState::read_field_slot(
    viz::FieldId fid, const std::vector<int>& idx) const {
  switch (fid) {
    case kScores: return std::any(scores[idx[0]]);
    case kBoard:  return std::any(static_cast<int>(board[idx[0]]));
    case kHand:   return std::any(static_cast<int>(hand[idx[0]]));
    // ...
  }
  return {};
}

void MyGameState::write_field_slot(
    viz::FieldId fid, const std::vector<int>& idx,
    const std::any& value) {
  switch (fid) {
    case kScores: scores[idx[0]] = std::any_cast<int>(value); return;
    case kBoard:  board[idx[0]] = static_cast<int8_t>(std::any_cast<int>(value)); return;
    case kHand:   hand[idx[0]] = static_cast<int8_t>(std::any_cast<int>(value)); return;
    // ...
  }
}
```

## mask_field_slot -- placeholder write for hidden slots

```cpp
void MyGameState::mask_field_slot(
    viz::FieldId fid, const std::vector<int>& idx) {
  switch (fid) {
    case kHand:
      hand[idx[0]] = kPlaceholderInt8; return;
    case kDeck:
      deck[idx[0]] = kPlaceholderInt8; return;
    // Fully-public fields never get called here
  }
}
```

## mask_all_hidden_slots -- masked clone hook

`make_masked_state(state, schema, perspective)` clones the state, then calls the cloned state's virtual `mask_all_hidden_slots(schema, perspective)`.

The default implementation walks every schema field and dispatches hidden slots to `mask_field_bulk`; the default `mask_field_bulk` then calls `mask_field_slot` once per hidden slot. Override `mask_all_hidden_slots` only when the state needs a one-time preparation step before placeholders are written, such as detaching COW/shared persistent data.

## public_snapshot / apply_public_snapshot -- observation wire

`viz::serialize_public_snapshot` writes the perspective's post-action observation into two halves:

- value half: for each visible schema slot, write `(idx_path, value)` under the field name using `read_field_slot`;
- viz half: write `__viz__`, a per-field flat slice of `viz_[field][..., perspective]`.

`viz::apply_public_snapshot` applies the viz half first by wholesale replacing the receiver's perspective slice, then writes the value half back via `write_field_slot`. Hidden slots are not refreshed from truth by this path; they remain structurally unreachable until search samples a world through the belief tracker.

Every field that can affect `state_hash_for_perspective(p)` must round-trip through this snapshot path for observer sessions, unless it is an explicitly skipped fixed-at-start field with a documented reason.

---

## Performance: hash_field_bulk / mask_field_bulk

The default `hash_field_bulk` / `mask_field_bulk` implementations forward per-slot to `hash_field_slot` / `mask_field_slot` with `std::vector<int>` index construction per slot. For games with many fields (Splendor 27, Coup 27, Azul 19), this overhead matters — MCTS calls hash every descent step.

**Override `hash_field_bulk` for hot fields** — replace the per-slot virtual dispatch with a tight inlined loop. See `games/6nimmit/6nimmit_state.cpp` for the canonical example: it overrides `hash_field_bulk` for `hand` and `bullhead_pile` (the two largest fields) with a direct loop that skips index vector construction.

Pattern:
```cpp
void hash_field_bulk(Hasher& h, viz::FieldId fid,
                     const viz::VizTensor& v, int player) const override {
  switch (fid) {
    case kFieldHand: {
      // Tight loop — no idx vector, no virtual dispatch per slot
      for (int i = 0; i < kHandSize; ++i) {
        h.add(static_cast<uint64_t>(fid));
        h.add(static_cast<uint64_t>(i));
        if (v.is_visible(i, player)) {
          h.add(static_cast<uint64_t>(data.hand[i]));
        } else {
          h.add(kHiddenHashSentinel);
        }
      }
      return;
    }
    default:
      IGameState::hash_field_bulk(h, fid, v, player);
  }
}
```

Priority: add to games with >15 schema fields (Splendor, Coup, Azul). Small games (TicTacToe 6 fields, Liar's Dice 6 fields) don't benefit enough.

## state_hash_for_perspective -- DAG node identity

`state_hash_for_perspective(player)` is framework-provided. It mixes `step_count_`, then walks the full schema in declaration order and row-major slot order:

- visible slot: mix field id, data-axis index, then call `hash_field_slot` / `hash_field_bulk` for the value;
- hidden slot: mix field id, data-axis index, then mix `kHiddenHashSentinel`.

Hashing the live state and hashing a masked clone must produce the same digest for the same perspective: hidden slots are hidden by the viz bit, not by the placeholder byte. The masked clone exists so encoders and evaluators cannot read truth; the hash walker already has its own structural hidden-sentinel path.

## UndoRecord pattern (for tail-solver undo)

```cpp
struct UndoRecord {
  ActionId action;
  int prev_player;
  int prev_winner;
  bool prev_terminal;
  // snapshot whatever do_action_deterministic mutates
};
std::vector<UndoRecord> undo_stack;  // member of state
```

Only `do_action_deterministic_impl` pushes to `undo_stack`. `do_action_fast_impl` must NOT push (it runs on the MCTS hot path, undo is never called).

## Key invariants

- Every piece of state in DAG node identity MUST be a schema slot. No off-schema escape hatch.
- **AI sessions reconstruct state from `public_snapshot` alone — and snapshot only carries schema-declared fields.** Any off-schema private member (a cached `last_*`, a derived counter, a Boolean flag the rules set in `do_action_fast`) is **uninitialized** in the session's state object: snapshot does not write it, `randomize_unseen` does not resample it. The moment `do_action_fast` / `legal_actions` / encoder reads it on the session side, it's UB. If a value matters at all to game logic, it must be a schema field (public or hidden); if it's purely public-derived memory that rules don't need, put it on the **tracker** (which has its own `pack_init_payload` / `observe_public_event` / `clone` lifecycle).
- The most common symptom of a schema-coverage gap is `legal_actions(ai_view)` computed on a divergent session board — returning an action that is illegal in truth, or an empty legal set. This is silent: it surfaces mid-training, not at startup. If a field changes `legal_actions` or game outcome and it is not a schema slot, this is almost certainly the cause. (The harmonies `632ebe2` regression hit this after `animal_deck` became hidden but the field round-trip was not fully wired.)
- Every piece of state that can change `legal_actions` for the acting player MUST be in that player's perspective hash scope (visible schema slot or structural sentinel). Same perspective hash implies same legal-action set.
- `step_count_` is framework-managed. Never touch it from game code.
- `viz_` is initialized by `viz::init_viz` and mutated only by rules via `reveal_slot` / `reveal_slot_to` / `reset_to_base`.
- `public_snapshot` must overwrite all public/hash-relevant observer state through `read_field_slot` / `write_field_slot` plus the `__viz__` slice.
- Encoders read masked clones; they should branch on placeholder sentinels, not query `viz_`.
- Default copy constructor must produce correct deep copy (CloneableState relies on it).

## winners() / losers() — Terminal Result Sets

The framework value model (`DefaultStateValueModel`) computes Z-values from two virtual methods:

```cpp
virtual std::vector<int> winners() const;  // players splitting +1
virtual std::vector<int> losers() const;   // players splitting -1
```

**Default behavior** (derived from `winner()`):
- `winners()` → `{winner()}` if winner >= 0, else `{}`
- `losers()` → all players except winner, else `{}`

This default gives **winner-takes-all**: 1 winner gets +1, N-1 losers split -1.

**Override for loser-pays games** (liarsdice, 6nimmit): override both methods so `losers()` returns the small loser set and `winners()` returns everyone else.

```cpp
// Example: liarsdice — 1 loser (wrong call), N-1 winners
std::vector<int> losers() const override {
  if (!data.terminal || data.winner < 0) return {};
  const int challenger = data.current_player_;
  const int bidder = (challenger + NPlayers - 1) % NPlayers;
  const int loser = (data.winner == challenger) ? bidder : challenger;
  return {loser};
}

std::vector<int> winners() const override {
  auto lo = losers();
  if (lo.empty()) return {};
  std::vector<int> result;
  for (int p = 0; p < NPlayers; ++p) {
    bool is_loser = false;
    for (int x : lo) { if (x == p) { is_loser = true; break; } }
    if (!is_loser) result.push_back(p);
  }
  return result;
}
```

**Value model formula** (`DefaultStateValueModel`):
- Each winner gets `+1 / |winners|`
- Each loser gets `-1 / |losers|`
- Players in neither set get `0`
- If all N players are in one set → all get `0`

**When to override**: any game where the natural payoff is "don't be the loser" rather than "be the winner". Set `"victory_mode": "loser_pays"` in `game.json` so the training pipeline uses the correct gating threshold variance.
