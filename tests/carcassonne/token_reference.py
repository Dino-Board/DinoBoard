"""Expected CarcassonneTokenEncoder features, recomputed from the public state
dict with the independent reference (carc_reference). Layout documented in
docs/games/carcassonne_api.md.

`expected_tokens(state, legal, me)` returns the type-specific feature blocks
(87 floats each) the encoder must produce, with absolute coordinates left at
zero as in `carcassonne_tok`.
"""
from __future__ import annotations

import numpy as np

from . import carc_reference as ref

NF = 87                          # type-specific features per token
GLOBAL, CURRENT, TILE, FRONTIER = range(4)
SLOTS = 9                        # current: 13 positions x 6
TILE_SIDE, TILE_FARM, TILE_CLOISTER, TILE_NEIGHBOURS, TILE_SAME = 13, 45, 69, 70, 71
FR_SIDE, FR_FARM, FR_OCCUPIED, FR_FITS, FR_BAG, FR_DEAD, FR_OUTCOME = 2, 34, 50, 51, 55, 56, 57
OUTCOME_DIM = 7
SIDE_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def position_slots(b: ref.Board, cell: int, legal) -> np.ndarray:
    """The 13 meeple-position slots of the tile at `cell`."""
    out = np.zeros(78)
    edge, city_of, road_of, farm_of, _ = b.view[cell]
    for s in range(4):
        if edge[s] == ref.FIELD:
            continue
        kind = "c" if edge[s] == ref.CITY else "r"
        tiles_n, open_n, complete, _, shields = b.stats(kind, cell, city_of[s] if kind == "c" else road_of[s])
        out[6 * s: 6 * s + 6] = [(ref.SIDE + s) in legal, kind == "c", tiles_n / 10, open_n / 10, complete,
                                 shields / 5]
    for h in range(8):
        if farm_of[h] < 0:
            continue
        done, still_open = b.farm_cities(b.root("f", cell, farm_of[h]))
        lowest = min(x for x in range(8) if farm_of[x] == farm_of[h])
        out[6 * (4 + h): 6 * (4 + h) + 6] = [(ref.FARM + h) in legal, (lowest + 1) / 8, still_open / 5, 0, 0, done / 5]
    if ref.TILES[b.bt[cell]]["chapel"]:
        occ = b.occupancy(cell)
        out[72:78] = [ref.CENTER in legal, 0, 0, 0, occ == 9, occ / 9]
    return out


def lone_tile_slots(ttype: int) -> np.ndarray:
    """Position slots of a tile type standing alone, unrotated."""
    bt, br, none = [-1] * 400, [0] * 400, [-1] * 400
    bt[210] = ttype
    return position_slots(ref.Board(bt, br, none, none), 210, set())


def expected_tokens(d: dict, legal, me: int) -> dict:
    """{"global": f, "current": (entity, cell or None, f), "tiles": {cell: (entity, f)},
    "frontier": {cell: f}} for state dict `d` seen by player `me`."""
    legal = set(legal)
    opp = 1 - me
    bt, br, ow, ps = d["board_tile"], d["board_rotation"], d["board_meeple_owner"], d["board_meeple_pos"]
    b = ref.Board(bt, br, ow, ps)
    stage = d["stage"]
    last = d["last_row"] * 20 + d["last_col"] if d["last_row"] >= 0 and d["last_col"] >= 0 else -1
    bag = d["remaining_tile_counts"]
    bag_total = sum(bag)

    g = np.zeros(NF)
    proj = ref.projected_scores(bt, br, ow, ps, d["scores"])
    g[0:3] = [d["scores"][me] / 100, d["scores"][opp] / 100, (d["scores"][me] - d["scores"][opp]) / 50]
    g[3:6] = [proj[me] / 100, proj[opp] / 100, (proj[me] - proj[opp]) / 50]
    g[6:8] = [d["meeples_remaining"][me] / 7, d["meeples_remaining"][opp] / 7]
    g[8] = bag_total / 71
    g[9], g[10], g[11] = stage == 0, stage == 1, d["current_player"] == me
    g[12:36] = [c / 9 for c in bag]
    g[36] = sum(d["discarded_tile_counts"]) / 5

    cur = np.zeros(NF)
    cur[0], cur[1] = stage == 0, stage == 1
    if stage == 0:
        cur_entity, cur_cell = 1 + d["current_tile"], None
        cur[8] = len(legal) / 40
        cur[SLOTS:SLOTS + 78] = lone_tile_slots(d["current_tile"])
    else:
        cur_entity, cur_cell = 1 + bt[last], last
        cur[2 + br[last]] = 1
        cur[SLOTS:SLOTS + 78] = position_slots(b, last, legal)

    tiles = {}
    for c in b.cells:
        r, q = divmod(c, 20)
        edge, city_of, road_of, farm_of, _ = b.view[c]
        f = np.zeros(NF)
        f[br[c]] = 1
        f[6] = c == last
        if ow[c] >= 0:
            f[7 if ow[c] == me else 8] = 1
            p = ps[c]
            f[12 if p == 12 else 11 if p >= 4 else 9 if edge[p] == ref.CITY else 10] = 1
        roots = [None] * 4
        for s in range(4):
            if edge[s] == ref.FIELD:
                continue
            kind = "c" if edge[s] == ref.CITY else "r"
            grp = city_of[s] if kind == "c" else road_of[s]
            roots[s] = b.root(kind, c, grp)
            tiles_n, open_n, complete, meeples, shields = b.stats(kind, c, grp)
            f[TILE_SIDE + 8 * s: TILE_SIDE + 8 * s + 8] = [kind == "c", kind == "r", tiles_n / 10, open_n / 10,
                                                          complete, meeples[me] / 3, meeples[opp] / 3, shields / 5]
        for h in range(8):
            if farm_of[h] < 0:
                continue
            root = b.root("f", c, farm_of[h])
            m = b.meeples.get(root, [0, 0])
            f[TILE_FARM + 3 * h: TILE_FARM + 3 * h + 3] = [m[me] / 3, m[opp] / 3, b.farm_cities(root)[0] / 5]
        if ref.TILES[bt[c]]["chapel"]:
            f[TILE_CLOISTER] = b.occupancy(c) / 9
        f[TILE_NEIGHBOURS] = sum(1 for s in range(4) if ref.inb(r + ref.DR[s], q + ref.DC[s])
                                 and bt[(r + ref.DR[s]) * 20 + q + ref.DC[s]] >= 0) / 4
        for i, (s1, s2) in enumerate(SIDE_PAIRS):
            f[TILE_SAME + i] = roots[s1] is not None and roots[s1] == roots[s2]
        tiles[c] = (1 + bt[c], f)

    frontier = {}
    for c in ref.frontier_cells(bt):
        r, q = divmod(c, 20)
        f = np.zeros(NF)
        need = ref.edge_needs(bt, br, c)
        for s in range(4):
            if need[s] < 0:
                continue
            f[FR_SIDE + 8 * s + need[s]] = 1
            if need[s] == ref.FIELD:
                continue
            n = (r + ref.DR[s]) * 20 + q + ref.DC[s]
            nedge, ncity, nroad, _, _ = b.view[n]
            o = (s + 2) % 4
            kind = "c" if nedge[o] == ref.CITY else "r"
            tiles_n, open_n, _, meeples, shields = b.stats(kind, n, ncity[o] if kind == "c" else nroad[o])
            f[FR_SIDE + 8 * s + 3: FR_SIDE + 8 * s + 8] = [tiles_n / 10, open_n / 10, meeples[me] / 3,
                                                          meeples[opp] / 3, shields / 5]
        for h in range(8):
            e = ref.HALF_EDGE[h]
            if need[e] < 0:
                continue
            n = (r + ref.DR[e]) * 20 + q + ref.DC[e]
            ng = b.view[n][3][ref.HALF_OPP[h]]
            if ng < 0:
                continue
            m = b.meeples.get(b.root("f", n, ng), [0, 0])
            f[FR_FARM + 2 * h], f[FR_FARM + 2 * h + 1] = m[me] / 3, m[opp] / 3
        f[FR_OCCUPIED] = b.occupancy(c) / 8
        if stage == 0:
            for rot in range(4):
                if c * 4 + rot not in legal:
                    continue
                f[FR_FITS + rot] = 1
                pv = ref.preview(bt, br, ow, ps, d["current_tile"], c, rot,
                                 d["meeples_remaining"][d["current_player"]])
                f[FR_OUTCOME + OUTCOME_DIM * rot: FR_OUTCOME + OUTCOME_DIM * (rot + 1)] = [
                    pv["completed"] / 4, pv["score_delta"][me] / 20, pv["score_delta"][opp] / 20,
                    pv["city_claim"] / 2, pv["road_claim"] / 2, pv["meeples"][me] / 3, pv["meeples"][opp] / 3]
        if bag_total > 0:
            fitting = sum(bag[t] for t in range(24) if bag[t] and any(ref.fits(need, t, rot) for rot in range(4)))
            f[FR_BAG] = fitting / bag_total
            f[FR_DEAD] = fitting == 0
        frontier[c] = f
    return {"global": g, "current": (cur_entity, cur_cell, cur), "tiles": tiles, "frontier": frontier}


def mismatches(got, want, tol: float = 2e-5):
    """Indices where two feature vectors differ (empty list = equal)."""
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    assert got.shape == want.shape, (got.shape, want.shape)
    return np.nonzero(np.abs(got - want) > tol)[0].tolist()


def differences(exp: dict, global_f, current, tiles: dict, frontier: dict) -> list:
    """Problems between expected tokens and the encoder's, given as its
    global features, (entity, features) of the current token,
    {cell: (entity, features)} of tile tokens and {cell: features} of
    frontier tokens."""
    out = []

    def cmp(name, got, want):
        bad = mismatches(got, want)
        if bad:
            out.append((name, bad, np.asarray(got)[bad].round(4).tolist(), np.asarray(want)[bad].round(4).tolist()))

    cmp("global", global_f, exp["global"])
    if current[0] != exp["current"][0]:
        out.append(("current entity", current[0], exp["current"][0]))
    cmp("current", current[1], exp["current"][2])
    if sorted(tiles) != sorted(exp["tiles"]) or sorted(frontier) != sorted(exp["frontier"]):
        out.append(("token cells", sorted(set(tiles) ^ set(exp["tiles"])), sorted(set(frontier) ^ set(exp["frontier"]))))
        return out
    for c, (entity, f) in tiles.items():
        if entity != exp["tiles"][c][0]:
            out.append(("tile entity", c, entity))
        cmp(f"tile@{c}", f, exp["tiles"][c][1])
    for c, f in frontier.items():
        cmp(f"frontier@{c}", f, exp["frontier"][c])
    return out
