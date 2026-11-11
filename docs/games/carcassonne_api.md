# Carcassonne API

Carcassonne is a 2-player base-game implementation (72 tiles, 7 meeples each,
third-edition scoring: farms score 3 points per completed city they border,
each farm separately). There is no hidden information: the current tile and
the per-type counts of the tiles still in the bag are public, and the order of
future draws does not exist in the state (see "State Fields").

## Action Encoding

Tile placement actions are exactly `20 * 20 * 4 = 1600`:

```text
action = (row * 20 + col) * 4 + rotation
row = (action / 4) / 20
col = (action / 4) % 20
rotation = action % 4
```

Rows and columns are zero-based. Rotation is clockwise quarter-turns:
`0`, `1`, `2`, `3`.

**Only distinct rotations are actions.** A tile that looks the same after
fewer than four quarter turns offers only its lowest equivalent rotation:

| Tiles | Looks the same after | Legal rotations |
|-------|----------------------|-----------------|
| B, C, X | every quarter turn | `0` |
| F, G, H, U | a half turn | `0`, `1` |
| all others | a full turn | `0`..`3` |

So equal placements are one action and one state, and `board_rotation` of a
symmetric tile is always the lowest equivalent value. The repeated rotations
(e.g. tile U at rotation 2) are **rejected as illegal**, not silently
reduced: a client that lets the user turn the tile freely must send
`rotation % period` (the bundled web client does). `legal_actions` is always
the source of truth.

Meeple actions are a separate second phase after a tile is placed:

```text
1600      = skip meeple
1601-1604 = place on one of 4 road/city edge positions
1605-1612 = place farmer on one of 8 field positions
1613      = place on cloister center
```

Edge positions are top, right, bottom, left of the tile as it lies on the
board. Field positions are the eight half-edges, clockwise from the upper
half of the left edge: `TLL, TLT, TRT, TRR, BRR, BRB, BLB, BLL` (corner, then
the edge the half lies on). A city, road or farm that touches several
positions of the tile is offered once, at its lowest position.

The policy head therefore has 1614 total action IDs, but the board-placement
branch is still the fixed 1600-action region.

Only the tile-placement phase uses the 1600 board actions. The optional meeple
phase is a separate state stage and exposes only the skip/meeple action IDs.

## State Fields

Important serializer fields used by the web client:

- `rows`, `cols`: fixed at `20`.
- `stage`: `0` for tile placement, `1` for optional meeple placement.
- `board_tile`: length-400 array, `-1` for empty or tile type `0..23`.
- `board_rotation`: length-400 array, clockwise quarter-turns (always below
  the tile's rotation period, see above).
- `board_meeple_owner`: length-400 array, `-1` for no meeple or player `0..1`.
- `board_meeple_pos`: length-400 array, `-1` for no meeple, `0..3` for
  road/city edge positions, `4..11` for farmer positions, `12` for cloister.
- `final_meeple_scores`: length-400 array. Non-zero only on terminal states;
  each occupied meeple cell records the endgame score attributed to that meeple's
  terminal feature for board inspection. A feature with several meeples of
  the scoring player shows its points on one of them.
- `meeples_remaining`: supply count per player.
- `current_tile`: tile type to place, or `-1` after the deck is exhausted.
- `deck_count`, `tiles_remaining`: public count of face-down tiles.
- `remaining_tile_counts`: public multiset count after subtracting placed,
  current, and discarded unplaceable tiles.
- `discarded_tile_counts`: discarded unplaceable tile counts.
- `history_tile`, `history_rotation`, `history_row`, `history_col`: fixed
  length-72 public placement history. Slot 0 is the initial center tile; slots
  after the latest placed tile are `-1` for tile/row/col and `0` for rotation.
- `scores`: current score per player.
- `projected_final_scores`: each player's score if the game ended now
  (unfinished cities, roads, cloisters and farms by the end-of-game rules).
  During the meeple stage the features completed by the tile just placed
  count as completed, as they will be scored whichever meeple action follows.
  This is the projection the heuristic value function uses.
- `display_min_row`, `display_max_row`, `display_min_col`, `display_max_col`:
  auto-centered viewport window around the occupied map, expanded by one cell.

After every rules action, the engine recenters the occupied map inside the
fixed 20x20 coordinate space and shifts `board_*`, `history_row`,
`history_col`, `last_row`, `last_col`, and terminal meeple-score annotations
together. Clients should treat coordinates as canonical post-action coordinates
from the latest snapshot.

A client that writes a whole public snapshot itself
(`GameSession.apply_initial_observation`) may give a symmetric tile in any of
its equal rotations and a meeple on any edge / half-edge of its city, road or
farm. When the snapshot is stored, `board_rotation`, `history_rotation` and
`last_rotation` are reduced below the tile's rotation period and
`board_meeple_pos` is moved to the lowest position of the same feature, i.e.
the forms the rules themselves produce, so equivalent snapshots are the same
state (same hash, same network input). A meeple position that lies on no
feature of its tile is stored as given and belongs to no feature. Actions are
not relaxed: a meeple action must still be one of `legal_actions`, which names
each feature by its lowest position only.

The bag is count-only: `bag_counts[t]` (per-type remaining count) is an
`all_public` schema field — it is publicly derivable from initial counts
minus board placements minus the current draw minus discards, and there
is no hidden draw order to protect. The state holds no pre-shuffled deck
array, and no `belief_tracker` is registered (every Carcassonne field is
public). Each draw inside `do_action_fast` weighted-samples a tile type
from `bag_counts` using the caller's rng, decrementing that type's count.
See `.claude/skills/game-rules/bag-draw.md`.

## Rule Scope

The engine implements the base tile matching rule: a placed tile must share an
edge with an existing tile, and all touching city, road, and field edges must
match. If a drawn tile has no legal placement, it is discarded and another tile
is drawn.

Meeples can only be placed on the tile just placed. Connected roads, cities, and
farms that already contain any meeple are illegal targets. Completed roads,
cities, and cloisters score immediately and return their meeples; unfinished
roads, cities, cloisters, and farms score at game end using the base rules.

Farms are bordered by roads, cities and the edge of the map. In the tile
table (`carcassonne_tiles.cpp`) a field owns the half-edges through which it
continues onto the neighbouring tile; a half-edge on a city edge never
belongs to a field. The table is validated against this when it is built.

### Known limitation: the 20x20 window

The map lives in a fixed 20x20 coordinate space that is recentered after
every action. Real Carcassonne has no table edge; here a placement that would
make the map wider or taller than 20 tiles is illegal:

- at a width (or height) of 19 only one of the two ends can still grow, at 20
  neither can;
- a road or city running off the window can never be completed, and a
  cloister on the window border can never reach 9;
- a drawn tile whose only placements lie outside the window is discarded as
  unplaceable.

In 1500 heuristic games the map never exceeded 17; in 1500 uniformly random
games it reached 19 or 20 in 2% of the games. The token encoder
(`carcassonne_tok`) carries no absolute coordinates, so the network cannot
see the border.

## Network Feature Encoding

Two registered variants share rules and action space.

### `carcassonne` — flat history encoder (MLP), 3124 features

The net-MCTS encoder uses placement history rather than the full 20x20 board
grid. Each of the 72 history slots is 43-dimensional:

```text
24 tile-type one-hot
4 rotation one-hot
2 normalized coordinates
13 meeple-position values
```

Meeple-position values are perspective-relative: `1` for the acting player's
meeple at that position, `-1` for the opponent's meeple, and `0` for no meeple
or a returned meeple. Empty history slots are all zero. After the history block,
the encoder appends perspective-rotated current scores and meeple supplies
(`me_score`, `opp_score`, `me_meeples`, `opp_meeples`), then a 24-dimensional
one-hot for the current tile.

### `carcassonne_tok` — token encoder (transformer)

Tokens: one global token, one current-tile token, one token per placed tile,
one token per frontier cell (empty cell with at least one placed 4-neighbour),
in that order and only those present (7 to about 130; `feature_dim` 0).
Token types `0..3` in that order. Every token has 87 type-specific features
(unused tail is zero); "my"/"opp" are relative to the encoding perspective.

| | |
|---|---|
| Per token | `[1, entity, anchor, type, has_pos, x, y, 87 features]` = 94 |
| Entity id | `1 + tile type` on tile tokens and the current token, else `0` |
| Anchor id | `1 + cell` on tile and frontier tokens, `401` on the current token |
| Position | `x = column`, `y = row` on tile / frontier tokens and, in the meeple stage, on the current token (the cell of the tile just placed); the network sees positions only through its relative position bias, and the coordinate features below are always zero |

Policy binding (17 pointer heads per token, all by anchor): heads `0..3` on
a frontier token score placing the current tile on that cell at rotation
`0..3` (actions `cell * 4 + rotation`); heads `4..16` on the current token
score the 13 meeple positions (actions `1601..1613`); action `1600` (no
meeple) is scored by the global head.

**Global token** (features `0..36`)

| Index | Meaning |
|-------|---------|
| 0, 1, 2 | my score / 100, opp score / 100, (my - opp) / 50 |
| 3, 4, 5 | the same three for `projected_final_scores` |
| 6, 7 | my / opp meeples in supply, / 7 |
| 8 | tiles left in the bag / 71 |
| 9, 10 | stage is placement / meeple |
| 11 | the perspective player is the player to move |
| 12..35 | tiles of type A..X left in the bag, / 9 |
| 36 | tiles discarded as unplaceable / 5 |

**Current-tile token** (features `0..86`)

| Index | Meaning |
|-------|---------|
| 0, 1 | stage is placement / meeple |
| 2..5 | meeple stage: rotation one-hot of the tile just placed |
| 6, 7 | zero (cell coordinates; the position is in the header) |
| 8 | placement stage: legal placements / 40 |
| 9 + 6p + k | slot of meeple position `p` (0..3 sides, 4..11 field half-edges, 12 cloister), `k = 0..5` |

Slot of a side `p` (zero if that side is field): `k=0` a meeple may be placed
there, `1` it is a city (else a road), `2` tiles in the city / road / 10, `3`
its open edges / 10, `4` it is complete, `5` its shields / 5.
Slot of a half-edge (zero if it lies on a city edge): `k=0` a farmer may be
placed there, `1` farm id `(lowest half-edge of this farm on the tile + 1) / 8`
(equal for half-edges of the same farm), `2` unfinished cities the farm
borders / 5, `5` completed cities it borders / 5.
Cloister slot (zero unless the tile has one): `k=0` a monk may be placed,
`4` its 3x3 block is full, `5` occupied cells of the block / 9.

In the meeple stage the slots describe the tile just placed, in board
orientation, with the features as they are on the board. In the placement
stage they describe the tile to place **as if it stood alone and unrotated**
(every city / road has 1 tile and as many open edges as it has sides on the
tile, the cloister block holds 1 tile, nothing is legal): this is the tile's
geometry — terrain of each side, which sides and half-edges are connected,
shield, cloister — in the same terms the slots use later.

**Tile token** (features `0..76`)

| Index | Meaning |
|-------|---------|
| 0..3 | rotation one-hot |
| 4, 5 | zero (cell coordinates; the position is in the header) |
| 6 | most recently placed tile |
| 7, 8 | my / opp meeple stands on this tile |
| 9..12 | that meeple is on a city / road / farm / cloister |
| 13 + 8s + k | side `s`: `k=0` city, `1` road, `2` tiles of that city / road / 10, `3` its open edges / 10, `4` complete, `5` my meeples on it / 3, `6` opp meeples / 3, `7` shields / 5 (all zero for a field side) |
| 45 + 3h + k | half-edge `h`: `k=0` my farmers in that farm / 3, `1` opp farmers / 3, `2` completed cities it borders / 5 (zero on a city edge) |
| 69 | cloister tiles: occupied cells of the 3x3 block / 9 |
| 70 | occupied 4-neighbours / 4 |
| 71..76 | side pairs (0,1) (0,2) (0,3) (1,2) (1,3) (2,3): both sides are the same city, or the same road (also when they are separate on the tile and joined elsewhere) |

**Frontier token** (features `0..84`)

| Index | Meaning |
|-------|---------|
| 0, 1 | zero (cell coordinates; the position is in the header) |
| 2 + 8s + k | side `s`: `k=0..2` the neighbour there requires field / road / city (all zero: no neighbour), `3` tiles of the neighbouring city / road / 10, `4` its open edges / 10, `5` my meeples on it / 3, `6` opp meeples / 3, `7` its shields / 5 |
| 34 + 2h, 35 + 2h | my / opp farmers (/ 3) in the farm across half-edge `h` |
| 50 | occupied cells of the 3x3 block around the cell / 8 |
| 51..54 | placement stage: placing the current tile here at rotation `0..3` is legal |
| 55 | share of the tiles left in the bag that would fit here (any rotation) |
| 56 | no tile left in the bag fits here |
| 57 + 7r + k | placement stage, legal rotation `r`: what that placement does (below) |

Placement outcome `k` (computed by `CarcassonneRules::preview_placement`,
i.e. by the rules' own placement, collection and scoring code on a scratch
copy, before any meeple is added):

| k | Meaning |
|---|---------|
| 0 | features completed / 4: cities and roads of the placed tile that end up complete, plus cloisters whose 3x3 block it fills (those holding a monk, and its own) |
| 1, 2 | points I / the opponent score at the end of this turn from meeples already on the board, / 20 |
| 3 | best city of the tile the mover may claim: `0` none, `0.5` an open one, `1` one this placement completes |
| 4 | the same for roads |
| 5, 6 | my / opp meeples on the cities and roads the tile joins, / 3 |

Whether a monk can be placed does not depend on the cell (current-tile token:
cloister slot; global token: meeples in supply). Whether a farmer can be
placed is not given per rotation; the farmers across each half-edge are
(`34..49`), and the meeple stage has the exact answer.
