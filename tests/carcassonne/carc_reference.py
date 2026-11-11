"""Independent reference for Carcassonne geometry, legality and scoring.

Shares no code with the engine: the tile table below was written from the
tile pictures / rulebook (RGG 3rd edition), connectivity is union-find
rather than the engine's breadth-first collection. Tests compare the engine
against it instead of restating the engine's own logic.

Conventions (same numbering as the engine's public action / state ids):
sides 0..3 = top, right, bottom, left; farm half-edges 0..7 = TLL, TLT, TRT,
TRR, BRR, BRB, BLB, BLL (named by corner, then by the edge they lie on);
rotation r = r clockwise quarter turns; meeple positions 0..3 sides, 4..11
half-edges, 12 cloister.
"""
from __future__ import annotations

ROWS = COLS = 20
T, R, B, L = 0, 1, 2, 3
TLL, TLT, TRT, TRR, BRR, BRB, BLB, BLL = range(8)
FIELD, ROAD, CITY = 0, 1, 2
CODES = "ABCDEFGHIJKLMNOPQRSTUVWX"
COUNTS = [2, 4, 1, 4, 5, 2, 1, 3, 2, 3, 3, 3, 2, 3, 2, 3, 1, 3, 2, 1, 8, 9, 4, 1]
PASS, SIDE, FARM, CENTER = 1600, 1601, 1605, 1613
DR = [-1, 0, 1, 0]
DC = [0, 1, 0, -1]
HALF_EDGE = [L, T, T, R, R, B, B, L]            # edge each half-edge lies on
HALF_OPP = [TRR, BLB, BRB, TLL, BLL, TRT, TLT, BRR]  # matching half across that edge
HALVES_OF_SIDE = [(TLT, TRT), (TRR, BRR), (BRB, BLB), (TLL, BLL)]
ALL8 = list(range(8))


def _tile(edges, cities=(), roads=(), farms=(), shield=False, chapel=False):
    return dict(edges=[{"f": FIELD, "r": ROAD, "c": CITY}[e] for e in edges],
                cities=[list(g) for g in cities], roads=[list(g) for g in roads],
                farms=[(list(h), list(cs)) for h, cs in farms], shield=shield, chapel=chapel)


# farms: (half-edges of the field, sides whose city segment the field touches)
_T = {
    "A": _tile("ffrf", roads=[[B]], farms=[(ALL8, [])], chapel=True),
    "B": _tile("ffff", farms=[(ALL8, [])], chapel=True),
    "C": _tile("cccc", cities=[[T, R, B, L]], shield=True),
    "D": _tile("crfr", cities=[[T]], roads=[[R, L]], farms=[([TLL, TRR], [T]), ([BRR, BRB, BLB, BLL], [])]),
    "E": _tile("cfff", cities=[[T]], farms=[([TLL, TRR, BRR, BRB, BLB, BLL], [T])]),
    "F": _tile("fcfc", cities=[[R, L]], farms=[([TLT, TRT], [R, L]), ([BRB, BLB], [R, L])], shield=True),
    "G": _tile("fcfc", cities=[[R, L]], farms=[([TLT, TRT], [R, L]), ([BRB, BLB], [R, L])]),
    "H": _tile("fcfc", cities=[[R], [L]], farms=[([TLT, TRT, BRB, BLB], [R, L])]),
    "I": _tile("ccff", cities=[[T], [R]], farms=[([TLL, BRB, BLB, BLL], [T, R])]),
    "J": _tile("crrf", cities=[[T]], roads=[[R, B]], farms=[([TLL, TRR, BLB, BLL], [T]), ([BRR, BRB], [])]),
    "K": _tile("cfrr", cities=[[T]], roads=[[B, L]], farms=[([TLL, TRR, BRB, BRR], [T]), ([BLL, BLB], [])]),
    "L": _tile("crrr", cities=[[T]], roads=[[R], [B], [L]],
               farms=[([TLL, TRR], [T]), ([BLL, BLB], []), ([BRB, BRR], [])]),
    "M": _tile("ccff", cities=[[T, R]], farms=[([TLL, BLB, BLL, BRB], [T, R])], shield=True),
    "N": _tile("ccff", cities=[[T, R]], farms=[([TLL, BLB, BLL, BRB], [T, R])]),
    "O": _tile("crrc", cities=[[T, L]], roads=[[R, B]], farms=[([BLB, TRR], [T, L]), ([BRR, BRB], [])],
               shield=True),
    "P": _tile("crrc", cities=[[T, L]], roads=[[R, B]], farms=[([BLB, TRR], [T, L]), ([BRR, BRB], [])]),
    "Q": _tile("ccfc", cities=[[T, R, L]], farms=[([BLB, BRB], [T, R, L])], shield=True),
    "R": _tile("ccfc", cities=[[T, R, L]], farms=[([BLB, BRB], [T, R, L])]),
    "S": _tile("ccrc", cities=[[T, R, L]], roads=[[B]], farms=[([BLB], [T, R, L]), ([BRB], [T, R, L])],
               shield=True),
    "T": _tile("ccrc", cities=[[T, R, L]], roads=[[B]], farms=[([BLB], [T, R, L]), ([BRB], [T, R, L])]),
    "U": _tile("rfrf", roads=[[T, B]], farms=[([TLL, TLT, BLB, BLL], []), ([TRR, TRT, BRR, BRB], [])]),
    "V": _tile("ffrr", roads=[[B, L]], farms=[([BLB, BLL], []), ([TLL, TLT, TRR, TRT, BRB, BRR], [])]),
    "W": _tile("frrr", roads=[[R], [B], [L]],
               farms=[([BLB, BLL], []), ([BRB, BRR], []), ([TLL, TLT, TRR, TRT], [])]),
    "X": _tile("rrrr", roads=[[T], [R], [B], [L]],
               farms=[([BLB, BLL], []), ([BRB, BRR], []), ([TLL, TLT], []), ([TRR, TRT], [])]),
}
TILES = [_T[c] for c in CODES]


def rot_view(ttype: int, rot: int):
    """(edge[4], city group of side[4], road group of side[4], farm group of
    half[8], city sides of each farm) of a tile type after `rot` turns."""
    d = TILES[ttype]
    edge = [d["edges"][(s - rot) % 4] for s in range(4)]
    city_of, road_of, farm_of = [-1] * 4, [-1] * 4, [-1] * 8
    for g, sides in enumerate(d["cities"]):
        for s in sides:
            city_of[(s + rot) % 4] = g
    for g, sides in enumerate(d["roads"]):
        for s in sides:
            road_of[(s + rot) % 4] = g
    farm_city = []
    for g, (halves, cs) in enumerate(d["farms"]):
        for h in halves:
            farm_of[(h + 2 * rot) % 8] = g
        farm_city.append([(s + rot) % 4 for s in cs])
    return edge, city_of, road_of, farm_of, farm_city


def shape(ttype: int, rot: int):
    """The rotated tile as a picture: independent of group numbering and of
    which rotation produced it."""
    edge, city_of, road_of, farm_of, farm_city = rot_view(ttype, rot)
    d = TILES[ttype]
    cities = frozenset(frozenset(s for s in range(4) if city_of[s] == g) for g in range(len(d["cities"])))
    roads = frozenset(frozenset(s for s in range(4) if road_of[s] == g) for g in range(len(d["roads"])))
    farms = frozenset((frozenset(h for h in range(8) if farm_of[h] == g), frozenset(farm_city[g]))
                      for g in range(len(d["farms"])))
    return ttype, tuple(edge), cities, roads, farms


def rotation_period(ttype: int) -> int:
    return next(p for p in (1, 2, 4) if shape(ttype, p % 4) == shape(ttype, 0))


def inb(r: int, q: int) -> bool:
    return 0 <= r < ROWS and 0 <= q < COLS


def edge_needs(board_tile, board_rot, cell):
    """Terrain a tile at `cell` must show per side (-1 = no neighbour)."""
    r, q = divmod(cell, COLS)
    need = [-1] * 4
    for s in range(4):
        nr, nq = r + DR[s], q + DC[s]
        if inb(nr, nq) and board_tile[nr * COLS + nq] >= 0:
            n = nr * COLS + nq
            need[s] = TILES[board_tile[n]]["edges"][((s + 2) % 4 - board_rot[n]) % 4]
    return need


def fits(need, ttype, rot) -> bool:
    return all(n < 0 or TILES[ttype]["edges"][(s - rot) % 4] == n for s, n in enumerate(need))


def frontier_cells(board_tile):
    return [c for c in range(ROWS * COLS) if board_tile[c] < 0 and any(
        inb(c // COLS + DR[s], c % COLS + DC[s]) and board_tile[(c // COLS + DR[s]) * COLS + c % COLS + DC[s]] >= 0
        for s in range(4))]


def legal_placements(board_tile, board_rot, ttype, *, all_rotations=False):
    """Action ids of the legal placements. A tile that looks the same after
    fewer than four quarter turns only offers its distinct rotations."""
    rots = range(4 if all_rotations else rotation_period(ttype))
    out = set()
    for c in frontier_cells(board_tile):
        need = edge_needs(board_tile, board_rot, c)
        out.update(c * 4 + rot for rot in rots if fits(need, ttype, rot))
    return out


class _UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


class Board:
    """Connected cities ("c"), roads ("r") and farms ("f") of a position.
    Nodes are (kind, cell, group index on that cell's tile)."""

    def __init__(self, board_tile, board_rot, owner, pos):
        self.bt, self.br = list(board_tile), list(board_rot)
        self.owner, self.pos = list(owner), list(pos)
        self.cells = [c for c in range(ROWS * COLS) if self.bt[c] >= 0]
        self.view = {c: rot_view(self.bt[c], self.br[c]) for c in self.cells}
        uf = self.uf = _UnionFind()
        open_list, off_list = [], []
        for c in self.cells:
            edge, city_of, road_of, farm_of, _ = self.view[c]
            r, q = divmod(c, COLS)
            for s in range(4):
                kind = edge[s]
                if kind == FIELD:
                    continue
                node = ("c" if kind == CITY else "r", c, city_of[s] if kind == CITY else road_of[s])
                uf.find(node)
                nr, nq = r + DR[s], q + DC[s]
                n = nr * COLS + nq
                if not inb(nr, nq) or self.bt[n] < 0:
                    (open_list if inb(nr, nq) else off_list).append(node)
                    continue
                nedge, ncity, nroad, _, _ = self.view[n]
                o = (s + 2) % 4
                assert nedge[o] == kind, "edge mismatch on board"
                uf.union(node, ("c" if kind == CITY else "r", n, ncity[o] if kind == CITY else nroad[o]))
            for h in range(8):
                g = farm_of[h]
                if g < 0:
                    continue
                node = ("f", c, g)
                uf.find(node)
                e = HALF_EDGE[h]
                nr, nq = r + DR[e], q + DC[e]
                n = nr * COLS + nq
                if not inb(nr, nq) or self.bt[n] < 0:
                    continue
                ng = self.view[n][3][HALF_OPP[h]]
                assert ng >= 0, "field half-edge faces a non-field half-edge"
                uf.union(node, ("f", n, ng))
        self.comp = {}
        for node in list(uf.p):
            self.comp.setdefault(uf.find(node), []).append(node)
        self.open_cnt = {}
        for n in open_list:
            self.open_cnt[uf.find(n)] = self.open_cnt.get(uf.find(n), 0) + 1
        # Incomplete: an edge towards an empty cell, or off the 20x20 window.
        self.incomplete = set(self.open_cnt) | {uf.find(n) for n in off_list}
        self.meeples = {}          # root -> [n0, n1]
        self.monks = []            # (cell, owner)
        for c in self.cells:
            if self.owner[c] < 0:
                continue
            node = self.meeple_node(c, self.pos[c])
            if node is None:
                self.monks.append((c, self.owner[c]))
                continue
            self.meeples.setdefault(uf.find(node), [0, 0])[self.owner[c]] += 1

    def meeple_node(self, c, p):
        edge, city_of, road_of, farm_of, _ = self.view[c]
        if p < 4:
            assert edge[p] != FIELD, "meeple on a field side"
            return ("c" if edge[p] == CITY else "r", c, city_of[p] if edge[p] == CITY else road_of[p])
        if p < 12:
            assert farm_of[p - 4] >= 0, "farmer on a half-edge without a field"
            return ("f", c, farm_of[p - 4])
        return None

    def root(self, kind, c, g):
        return self.uf.find((kind, c, g))

    def occupancy(self, c):
        r, q = divmod(c, COLS)
        return sum(1 for dr in (-1, 0, 1) for dq in (-1, 0, 1)
                   if inb(r + dr, q + dq) and self.bt[(r + dr) * COLS + q + dq] >= 0)

    def stats(self, kind, c, g):
        """(tiles, open edges, complete, meeples[2], shields) of a city / road."""
        root = self.root(kind, c, g)
        tiles_in = {n[1] for n in self.comp[root]}
        shields = sum(1 for x in tiles_in if TILES[self.bt[x]]["shield"]) if kind == "c" else 0
        return (len(tiles_in), self.open_cnt.get(root, 0), root not in self.incomplete,
                self.meeples.get(root, [0, 0]), shields)

    def points(self, root):
        tiles_n, _, complete, _, shields = self.stats(*self.comp[root][0])
        return (tiles_n + shields) * (2 if complete else 1) if self.comp[root][0][0] == "c" else tiles_n

    def farm_cities(self, root):
        """(complete, incomplete) distinct cities bordering the farm."""
        cities = set()
        for _, c, g in self.comp[root]:
            edge, city_of, _, _, farm_city = self.view[c]
            for s in farm_city[g]:
                assert edge[s] == CITY
                cities.add(self.root("c", c, city_of[s]))
        done = sum(1 for cr in cities if cr not in self.incomplete)
        return done, len(cities) - done

    def completion(self):
        """Mid-game scoring of everything complete that holds a meeple:
        (points[2], meeples returned[2])."""
        delta, returned = [0, 0], [0, 0]
        for root, cnt in self.meeples.items():
            if self.comp[root][0][0] == "f" or root in self.incomplete:
                continue
            for p in (0, 1):
                if cnt[p] == max(cnt):
                    delta[p] += self.points(root)
                returned[p] += cnt[p]
        for c, o in self.monks:
            if self.occupancy(c) == 9:
                delta[o] += 9
                returned[o] += 1
        return delta, returned

    def final(self):
        """End-of-game scoring of what is still on the board: (points[2],
        of which farm points[2])."""
        delta, farm = [0, 0], [0, 0]
        for root, cnt in self.meeples.items():
            kind = self.comp[root][0][0]
            if kind == "f":
                pts = 3 * self.farm_cities(root)[0]
            elif root in self.incomplete:
                pts = self.points(root)
            else:
                continue
            for p in (0, 1):
                if cnt[p] == max(cnt):
                    delta[p] += pts
                    if kind == "f":
                        farm[p] += pts
        for c, o in self.monks:
            if self.occupancy(c) < 9:
                delta[o] += self.occupancy(c)
        return delta, farm

    def legal_meeples(self, cell, meeples_left):
        """Action ids open to the mover after placing the tile at `cell`."""
        out = {PASS}
        if meeples_left <= 0 or self.owner[cell] >= 0:
            return out
        edge, city_of, road_of, farm_of, _ = self.view[cell]
        seen = set()
        for s in range(4):
            if edge[s] == FIELD:
                continue
            key = ("c" if edge[s] == CITY else "r", city_of[s] if edge[s] == CITY else road_of[s])
            if key not in seen and self.root(key[0], cell, key[1]) not in self.meeples:
                out.add(SIDE + s)
            seen.add(key)
        for h in range(8):
            g = farm_of[h]
            if g >= 0 and ("f", g) not in seen and self.root("f", cell, g) not in self.meeples:
                out.add(FARM + h)
            seen.add(("f", g))
        if TILES[self.bt[cell]]["chapel"]:
            out.add(CENTER)
        return out


def projected_scores(board_tile, board_rot, owner, pos, scores):
    """Scores if the game ended now: what is complete but still occupied is
    scored as complete, the rest by the end-of-game rules."""
    b = Board(board_tile, board_rot, owner, pos)
    done, _ = b.completion()
    rest, _ = b.final()
    return [scores[p] + done[p] + rest[p] for p in (0, 1)]


def preview(board_tile, board_rot, owner, pos, ttype, cell, rot, mover_meeples_left):
    """What placing `ttype` at (cell, rot) does before any meeple is added."""
    bt, br = list(board_tile), list(board_rot)
    bt[cell], br[cell] = ttype, rot
    b = Board(bt, br, owner, pos)
    edge, city_of, road_of, _, _ = b.view[cell]
    legal = b.legal_meeples(cell, mover_meeples_left)
    roots = {}
    claim = {"c": 0, "r": 0}
    for s in range(4):
        if edge[s] == FIELD:
            continue
        kind = "c" if edge[s] == CITY else "r"
        root = b.root(kind, cell, city_of[s] if kind == "c" else road_of[s])
        roots[root] = kind
        if SIDE + s in legal:
            claim[kind] = max(claim[kind], 1 if root in b.incomplete else 2)
    completed = sum(1 for root in roots if root not in b.incomplete)
    r, q = divmod(cell, COLS)
    for dr in (-1, 0, 1):
        for dq in (-1, 0, 1):
            c = (r + dr) * COLS + q + dq
            if not inb(r + dr, q + dq) or bt[c] < 0 or not TILES[bt[c]]["chapel"] or b.occupancy(c) != 9:
                continue
            if c == cell or (owner[c] >= 0 and pos[c] == 12):
                completed += 1
    return dict(completed=completed, score_delta=b.completion()[0],
                meeples=[sum(b.meeples.get(root, [0, 0])[p] for root in roots) for p in (0, 1)],
                city_claim=claim["c"], road_claim=claim["r"])
