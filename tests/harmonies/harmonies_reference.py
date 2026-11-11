"""Independent reference for the Harmonies rules facts and the token encoder.

Nothing here goes through the C++ tables: the stacking rules are written from
the rulebook (p.3, p.8), card patterns are read from animal_cards.json, the
board comes from the state's `board_columns`, and pattern fits, landscape
scores and every token feature are recomputed by brute force (e.g. "how much
closer does colour c on hex x bring this card" re-scores the whole modified
board instead of updating incrementally like the encoder).
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INF = math.inf
COLOURS = ["Blue", "Gray", "Brown", "Green", "Yellow", "Red"]  # colour id = index + 1
DIRS = [(0, -1), (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0)]   # N NE SE S SW NW

# Tower types as bottom-up colour stacks (harmonies_state.h numbering); the
# building (7) is Red on Brown, Gray or Red, kept with its base open.
STACKS = {0: [], 1: ["Blue"], 2: ["Gray"], 3: ["Brown"], 4: ["Green"], 5: ["Yellow"], 6: ["Red"],
          7: ["*", "Red"], 8: ["Brown", "Brown"], 9: ["Brown", "Green"], 10: ["Gray", "Gray"],
          11: ["Brown", "Brown", "Green"], 12: ["Gray", "Gray", "Gray"]}
HEIGHT = {t: len(s) for t, s in STACKS.items()}


def _stack_type(stack):
    if len(stack) == 2 and stack[1] == "Red" and stack[0] in ("Brown", "Gray", "Red"):
        return 7
    for t, s in STACKS.items():
        if s == stack:
            return t
    return None


def _legal_stack(stack):
    """Rulebook p.3: trees = Green on 0-2 Brown; buildings = Red on one Brown,
    Gray or Red; mountains = 1-3 Gray; anything else only alone."""
    if len(stack) == 1:
        return True
    if stack[-1] == "Green":
        return all(c == "Brown" for c in stack[:-1]) and len(stack) <= 3
    if stack[-1] == "Red":
        return len(stack) == 2 and stack[0] in ("Brown", "Gray", "Red")
    if stack[-1] == "Gray":
        return all(c == "Gray" for c in stack) and len(stack) <= 3
    if stack[-1] == "Brown":  # a trunk waiting for its leaves
        return stack == ["Brown", "Brown"]
    return False


@lru_cache(maxsize=None)
def transition(t: int, colour: int) -> int:
    """Tower type after stacking colour (1..6) on tower type t, -1 if illegal."""
    if t == 7:
        return -1
    stack = STACKS[t] + [COLOURS[colour - 1]]
    if not _legal_stack(stack):
        return -1
    nt = _stack_type(stack)
    return -1 if nt is None else nt


@lru_cache(maxsize=None)
def top_colour(t: int) -> int:
    return 0 if t == 0 else COLOURS.index(STACKS[t][-1]) + 1


@lru_cache(maxsize=None)
def distance(t: int, target: int) -> float:
    """Fewest tokens stacked on t to reach target."""
    frontier, seen, d = {t}, {t}, 0
    while frontier:
        if target in frontier:
            return d
        nxt = {transition(u, c) for u in frontier for c in range(1, 7)} - {-1} - seen
        seen |= nxt
        frontier, d = nxt, d + 1
    return INF


def _level_ok(req, actual):
    if req is None:
        return actual is None
    if actual is None:
        return False
    return actual in req if isinstance(req, list) else actual == req


def satisfies(t: int, levels) -> bool:
    stack = STACKS[t]
    bases = ["Brown", "Gray", "Red"] if t == 7 else [None]
    for base in bases:
        s = [base if c == "*" else c for c in stack] + [None] * (3 - len(stack))
        if all(_level_ok(r, a) for r, a in zip(levels, s)):
            return True
    return False


@lru_cache(maxsize=1)
def cards():
    data = json.loads((ROOT / "games/harmonies/animal_cards.json").read_text(encoding="utf-8"))["cards"]
    out = {}
    for c in data:
        pieces = [(tuple(p["offset"]), tuple(t for t in range(13) if satisfies(t, p["levels"])))
                  for p in c["pattern"]]
        out[c["id"]] = dict(pieces=pieces, anchor=c["anchor_index"], scores=c["scores"])
    return out


def piece_distance(t: int, targets) -> float:
    return min(distance(t, x) for x in targets)


def target_tower(targets) -> int:
    return min(targets, key=lambda x: (HEIGHT[x], x))


# ---- board geometry ----

class Geometry:
    def __init__(self, columns):
        self.axial = {}
        for q, col in enumerate(columns):
            for i, h in enumerate(col):
                self.axial[h] = (q, i - q // 2)
        self.n = len(self.axial)
        cell = {v: h for h, v in self.axial.items()}
        self.cell = cell
        self.nbrs = [[cell.get((self.axial[h][0] + dq, self.axial[h][1] + dr), -1) for dq, dr in DIRS]
                     for h in range(self.n)]

    def placements(self, cid):
        """(anchor hex, [hex per piece]) for every in-bounds rotation of the pattern."""
        card = cards()[cid]
        out = []
        for a in range(self.n):
            for k in range(6):
                def rot(v):
                    q, r = v
                    for _ in range(k):
                        q, r = -r, q + r
                    return q, r
                aq, ar = rot(card["pieces"][card["anchor"]][0])
                hexes = []
                for off, _ in card["pieces"]:
                    q, r = rot(off)
                    hexes.append(self.cell.get((self.axial[a][0] + q - aq, self.axial[a][1] + r - ar), -1))
                if -1 not in hexes:
                    out.append((a, hexes))
        return out


# ---- landscape scoring (rulebook p.8-9) ----

def groups(geo, tt, member):
    label = [-1] * geo.n
    sizes = []
    for h in range(geo.n):
        if label[h] >= 0 or not member(tt[h]):
            continue
        label[h] = len(sizes)
        todo, size = [h], 0
        while todo:
            u = todo.pop()
            size += 1
            for v in geo.nbrs[u]:
                if v >= 0 and label[v] < 0 and member(tt[v]):
                    label[v] = len(sizes)
                    todo.append(v)
        sizes.append(size)
    return label, sizes


MOUNTAIN = {2: 1, 10: 3, 12: 7}
TREE = {4: 1, 9: 3, 11: 7}


def cell_points(geo, tt, h):
    t = tt[h]
    if t in TREE:
        return TREE[t]
    if t in MOUNTAIN:
        return MOUNTAIN[t] if any(v >= 0 and tt[v] in MOUNTAIN for v in geo.nbrs[h]) else 0
    if t == 7:
        return 5 if len({top_colour(tt[v]) for v in geo.nbrs[h] if v >= 0} - {0}) >= 3 else 0
    return 0


def landscape(geo, tt, side):
    """(trees, mountains, fields, buildings, water)."""
    trees = sum(TREE.get(t, 0) for t in tt)
    mountains = sum(cell_points(geo, tt, h) for h in range(geo.n) if tt[h] in MOUNTAIN)
    buildings = sum(cell_points(geo, tt, h) for h in range(geo.n) if tt[h] == 7)
    fields = 5 * sum(1 for s in groups(geo, tt, lambda t: t == 5)[1] if s >= 2)
    if side == "B":
        water = 5 * max(1, len(groups(geo, tt, lambda t: t != 1)[1]))
    else:
        best = 0
        for src in range(geo.n):
            if tt[src] != 1:
                continue
            dist = {src: 0}
            todo = [src]
            while todo:
                u = todo.pop(0)
                for v in geo.nbrs[u]:
                    if v >= 0 and tt[v] == 1 and v not in dist:
                        dist[v] = dist[u] + 1
                        todo.append(v)
            best = max(best, max(dist.values()) + 1)
        water = [0, 0, 2, 5, 8, 11, 15][best] if best < 7 else 15 + 4 * (best - 6)
    return trees, mountains, fields, buildings, water


# ---- pattern fits ----

def card_fit(geo, tt, ha, cid):
    """Fewest missing tokens over placements whose cube hex is free and whose
    unsatisfied pieces can still be stacked to the requirement (INF if none);
    plus per placement its missing count."""
    card = cards()[cid]
    res = []
    for a, hexes in geo.placements(cid):
        if ha[a]:
            res.append((a, hexes, INF))
            continue
        m = 0
        for (off, targets), h in zip(card["pieces"], hexes):
            d = piece_distance(tt[h], targets)
            m += INF if (d > 0 and ha[h]) else d
        res.append((a, hexes, m))
    best = min((m for _, _, m in res), default=INF)
    return best, res


def fit_summary(geo, tt, ha, cid):
    best, res = card_fit(geo, tt, ha, cid)
    card = cards()[cid]
    part, missing_here, ready_at = [0] * geo.n, [0] * geo.n, [0] * geo.n
    for a, hexes, m in res:
        if m == 0:
            ready_at[a] = 1
        if best < INF and m == best:
            for (off, targets), h in zip(card["pieces"], hexes):
                part[h] = 1
                if piece_distance(tt[h], targets) > 0:
                    missing_here[h] = 1
    return dict(missing=best, part=part, missing_here=missing_here, ready_at=ready_at, ready=sum(ready_at))


def clip_missing(m):
    return 1.0 if m == INF else min(m, 6) / 6


# ---- expected tokens ----

def expected_tokens(d: dict, me: int) -> dict:
    side = d["board_side"]
    geo = Geometry(d["board_columns"])
    n = geo.n
    players = d["players"]
    nplayers = d["num_players"]
    seats = [(me + k) % nplayers for k in range(nplayers)]
    placing = d["current_player"] == me and not d["is_terminal"] and d["turn_started"]
    sel = d["selected_tokens"]
    tts = [p["tower_type"] for p in players]
    has = [p["has_animal"] for p in players]
    my_tt, my_ha = tts[me], has[me]

    def can_place(tt, ha, x, c):
        return not ha[x] and transition(tt[x], c) >= 0

    def deltas(tt, ha):
        base = sum(landscape(geo, tt, side))
        out = {}
        for x in range(n):
            for c in range(1, 7):
                if can_place(tt, ha, x, c):
                    t2 = list(tt)
                    t2[x] = transition(tt[x], c)
                    out[(x, c)] = sum(landscape(geo, t2, side)) - base
        return out

    dl = {p: deltas(tts[p], has[p]) for p in range(nplayers)}

    held = sorted([(cid, i) for i, cid in enumerate(players[me]["held_cards"]) if cid > 0])
    fits = [fit_summary(geo, my_tt, my_ha, cid) for cid, _ in held]
    after = []  # per card: {(x, c): missing after placing}
    for (cid, _), f in zip(held, fits):
        a = {}
        for x in range(n):
            for c in range(1, 7):
                if can_place(my_tt, my_ha, x, c):
                    t2 = list(my_tt)
                    t2[x] = transition(my_tt[x], c)
                    a[(x, c)] = card_fit(geo, t2, my_ha, cid)[0]
        after.append(a)
    adv = [any(f["missing"] < INF and a[(x, c)] < f["missing"] for f, a in zip(fits, after)
               for x in range(n) if (x, c) in a) for c in range(1, 7)]
    comp = [any(f["missing"] < INF and f["missing"] > 0 and a[(x, c)] == 0 for f, a in zip(fits, after)
                for x in range(n) if (x, c) in a) for c in range(1, 7)]

    exp = {}
    # global
    pouch = sum(d["pouch_counts"])
    g = dict(turn_started=d["turn_started"], took_card=d["took_card_this_turn"], end_triggered=d["end_triggered"],
             my_turn=d["current_player"] == me, pool=[v / 3 for v in sel], pool_total=sum(sel) / 3,
             pouch=[v / 23 for v in d["pouch_counts"]], pouch_total=pouch / 120, deck=d["animal_deck_count"] / 27,
             market_cards=sum(1 for c in d["animal_market"] if c > 0) / 5, refills_left=(pouch // 3) / 36,
             min_spaces=min(p["space_remaining"] for p in players) / n)
    exp["global"] = g

    def points_of(p):
        out = 0
        for cid, ci, cr in zip(p["held_cards"], p["cube_initial"], p["cube_remaining"]):
            if cid > 0:
                sc = cards()[cid]["scores"]
                placed = ci - cr
                out += sc[-1] - (sc[placed - 1] if placed else 0)
        return out

    exp["player"] = []
    for k, p in enumerate(seats):
        pl = players[p]
        others = [players[o]["score"] for o in range(nplayers) if o != p]
        bd = landscape(geo, tts[p], side)
        assert sum(bd) == pl["landscape_score"], "landscape score disagrees with the reference scorer"
        exp["player"].append(dict(
            seat=[1.0 if i == k else 0.0 for i in range(4)], to_move=d["current_player"] == p,
            first_player=d["first_player"] == p, animal_score=pl["animal_score"] / 50,
            landscape_score=pl["landscape_score"] / 50, total_score=pl["score"] / 100,
            margin=(pl["score"] - max(others)) / 50, trees=bd[0] / 20, mountains=bd[1] / 20, fields=bd[2] / 20,
            buildings=bd[3] / 20, water=bd[4] / 20,
            islands=len(groups(geo, tts[p], lambda t: t != 1)[1]) / 8, spaces=pl["space_remaining"] / n,
            cubes_placed=sum(has[p]) / n, cards_held=sum(1 for c in pl["held_cards"] if c > 0) / 4,
            card_points_open=points_of(pl) / 40))

    exp["slot"] = {}
    for s, counts in enumerate(d["central_board_counts"]):
        cand = [dl[me][(x, c)] for c in range(1, 7) if counts[c - 1] for x in range(n) if (x, c) in dl[me]]
        exp["slot"][s] = dict(tokens=[v / 3 for v in counts], total=sum(counts) / 3,
                              best_landscape=(max(cand) if cand else 0) / 10,
                              advancing=sum(counts[c - 1] for c in range(1, 7) if adv[c - 1]) / 3,
                              completing=sum(counts[c - 1] for c in range(1, 7) if comp[c - 1]) / 3)

    def card_static(cid):
        card = cards()[cid]
        need = [0.0] * 6
        habitat = [0.0] * 6
        for i, (off, targets) in enumerate(card["pieces"]):
            tower = target_tower(targets)
            stack = ["Red", "Red"] if tower == 7 else STACKS[tower]
            for col in stack:
                need[COLOURS.index(col)] += 0.25
            if i == card["anchor"]:
                habitat[top_colour(tower) - 1] = 1.0
        sc = card["scores"]
        return dict(pattern_size=len(card["pieces"]) / 4, need=need, habitat=habitat, cubes_total=len(sc) / 5,
                    full_points=sc[-1] / 20)

    def progress(cid, placed, remaining):
        sc = cards()[cid]["scores"]
        return dict(cubes_left=remaining / 5, points_now=(sc[placed - 1] if placed else 0) / 20,
                    next_points=(sc[placed] - (sc[placed - 1] if placed else 0)) / 10 if placed < len(sc) else 0.0)

    def fit_feats(f):
        return dict(missing=clip_missing(f["missing"]), no_fit=float(f["missing"] == INF), ready=f["ready"] / n)

    exp["market"] = {}
    for cid in d["animal_market"]:
        if cid <= 0:
            continue
        opp = min((card_fit(geo, tts[o], has[o], cid)[0] for o in range(nplayers) if o != me), default=INF)
        exp["market"][cid] = dict(market=1.0, **card_static(cid), **progress(cid, 0, len(cards()[cid]["scores"])),
                                  **fit_feats(fit_summary(geo, my_tt, my_ha, cid)), opp_missing=clip_missing(opp))
    exp["held"] = {}
    for k, p in enumerate(seats):
        pl = players[p]
        for cid, ci, cr in zip(pl["held_cards"], pl["cube_initial"], pl["cube_remaining"]):
            if cid <= 0:
                continue
            f = dict(**card_static(cid), **progress(cid, ci - cr, cr),
                     **fit_feats(fit_summary(geo, tts[p], has[p], cid)),
                     holder_seat=[1.0 if i == k else 0.0 for i in range(4)])
            f["my_hand" if k == 0 else "opp_hand"] = 1.0
            if k == 0:
                f["rank"] = [1.0 if i < len(held) and held[i][0] == cid else 0.0 for i in range(4)]
            exp["held"][(k, cid)] = f

    def landscape_cell(tt, ha, h):
        label, sizes = groups(geo, tt, lambda t: t != 1)
        flabel, fsizes = groups(geo, tt, lambda t: t == 5)
        nb = [v for v in geo.nbrs[h] if v >= 0]
        return dict(tower=[1.0 if t == tt[h] else 0.0 for t in range(13)], height=HEIGHT[tt[h]] / 3,
                    animal=float(ha[h]), points=cell_points(geo, tt, h) / 7,
                    field_group=(min(fsizes[flabel[h]], 5) / 5) if flabel[h] >= 0 else 0.0,
                    island=(sizes[label[h]] / n) if label[h] >= 0 else 0.0,
                    island_split=(len({label[v] for v in nb if label[v] >= 0}) / 3) if tt[h] == 1 else 0.0,
                    nb_colours=len({top_colour(tt[v]) for v in nb} - {0}) / 6,
                    nb_empty=sum(1 for v in nb if tt[v] == 0) / 6, edge=(6 - len(nb)) / 6)

    exp["cell"] = {}
    for h in range(n):
        f = landscape_cell(my_tt, my_ha, h)
        f["can_place"] = [float(can_place(my_tt, my_ha, h, c)) for c in range(1, 7)]
        f["legal"] = [float(can_place(my_tt, my_ha, h, c) and placing and sel[c - 1] > 0) for c in range(1, 7)]
        f["delta"] = [dl[me].get((h, c), 0) / 10 for c in range(1, 7)]
        part, miss, ready, advance = [0.0] * 4, [0.0] * 4, [0.0] * 4, [0.0] * 24
        for k, (fit, a) in enumerate(zip(fits, after)):
            part[k], miss[k], ready[k] = fit["part"][h], fit["missing_here"][h], fit["ready_at"][h]
            if fit["missing"] == INF:
                continue
            for c in range(1, 7):
                if (h, c) in a:
                    m2 = a[(h, c)]
                    m2 = fit["missing"] + 3 if m2 == INF else m2
                    advance[k * 6 + c - 1] = max(-3, min(3, fit["missing"] - m2)) / 3
        f.update(card_part=part, card_missing=miss, card_ready=ready, card_advance=advance)
        exp["cell"][h] = f

    exp["opp_cell"] = []
    for k, p in enumerate(seats[1:], start=1):
        for h in range(n):
            f = landscape_cell(tts[p], has[p], h)
            f["owner_seat"] = [1.0 if i == k - 1 else 0.0 for i in range(3)]
            f["delta"] = [dl[p].get((h, c), 0) / 10 for c in range(1, 7)]
            nbv = [0.0] * 48
            for dd, v in enumerate(geo.nbrs[h]):
                nbv[dd * 8 + (0 if v < 0 else 1 + top_colour(tts[p][v]))] = 1.0
            f["neighbour"] = nbv
            exp["opp_cell"].append(f)

    exp["cube"] = {}
    if placing:
        for k, ((cid, slot), fit) in enumerate(zip(held, fits)):
            remaining = players[me]["cube_remaining"][slot]
            placed = players[me]["cube_initial"][slot] - remaining
            sc = cards()[cid]["scores"]
            for h in range(n):
                if not fit["ready_at"][h]:
                    continue
                ha2 = list(my_ha)
                ha2[h] = 1
                blocks = sum(1 for j, (cj, _) in enumerate(held)
                             if j != k and card_fit(geo, my_tt, ha2, cj)[0] > fits[j]["missing"])
                exp["cube"][(h, cid)] = dict(rank=[1.0 if i == k else 0.0 for i in range(4)],
                                             points=(sc[placed] - (sc[placed - 1] if placed else 0)) / 10,
                                             completes=float(remaining == 1), cubes_left=(remaining - 1) / 5,
                                             blocks=blocks / 4)
    return exp


# ---- decoding the encoder's tokens ----

def layout_features(layout: dict) -> dict:
    return {t["name"]: {n: (o, w) for n, o, w in t["features"]} for t in layout["types"]}


def vectorize(feats: dict, spec: dict, width: int) -> np.ndarray:
    v = np.zeros(width, dtype=np.float64)
    for name, val in feats.items():
        o, w = spec[name]
        v[o:o + w] = np.asarray(val, dtype=np.float64).reshape(-1)[:w] if w > 1 else float(val)
    return v


def differences(x: np.ndarray, exp: dict, layout: dict, geo: Geometry, tol: float = 1e-5) -> list:
    """Compare decoded tokens with the expectation; returns a list of problems."""
    n_hexes = geo.n
    header = layout["header"]
    feats = layout_features(layout)
    names = [t["name"] for t in layout["types"]]
    width = x.shape[1] - header
    anchors = {a[0]: a[1] for a in layout["anchors"]}
    problems = []

    def cmp(tag, got, kind, want):
        v = vectorize(want, feats[kind], width)
        bad = np.flatnonzero(np.abs(got[header:] - v) > tol)
        if bad.size:
            inv = {o + i: f"{n}[{i}]" for n, (o, w) in feats[kind].items() for i in range(w)}
            problems.append(f"{tag}: " + ", ".join(f"{inv.get(int(b), b)} got {got[header + b]:.5f} want {v[b]:.5f}"
                                                   for b in bad[:4]))

    kinds = x[:, 3].astype(int)
    by = {name: x[kinds == i] for i, name in enumerate(names)}
    if len(by["global"]) != 1:
        problems.append("one global token expected")
    else:
        cmp("global", by["global"][0], "global", exp["global"])
    if len(by["player"]) != len(exp["player"]):
        problems.append("player token count")
    for i, (t, want) in enumerate(zip(by["player"], exp["player"])):
        cmp(f"player {i}", t, "player", want)
    slots = {int(t[2]) - anchors["slot"]: t for t in by["slot"]}
    if sorted(slots) != sorted(exp["slot"]):
        problems.append(f"slots {sorted(slots)}")
    for s, want in exp["slot"].items():
        if s in slots:
            cmp(f"slot {s}", slots[s], "slot", want)
    market = {int(t[1]): t for t in by["card"] if t[2] > 0}
    held = {}
    for t in by["card"]:
        if t[2] == 0:
            o, w = feats["card"]["holder_seat"]
            held[(int(np.argmax(t[header + o:header + o + w])), int(t[1]))] = t
    if sorted(market) != sorted(exp["market"]) or sorted(held) != sorted(exp["held"]):
        problems.append(f"cards market {sorted(market)} held {sorted(held)}")
    for cid, want in exp["market"].items():
        if cid in market:
            if int(market[cid][2]) != anchors["market_card"] + cid - 1:
                problems.append(f"market card {cid} anchor")
            cmp(f"market card {cid}", market[cid], "card", want)
    for key, want in exp["held"].items():
        if key in held:
            cmp(f"held card {key}", held[key], "card", want)
    cells = {int(t[2]) - anchors["cell"]: t for t in by["cell"]}
    if sorted(cells) != list(range(n_hexes)):
        problems.append("cell anchors")
    for h, want in exp["cell"].items():
        t = cells.get(h)
        if t is None:
            continue
        if not (t[4] == 1 and t[1] == 0 and (t[5], t[6]) == geo.axial[h]):
            problems.append(f"cell {h} header")
        cmp(f"cell {h}", t, "cell", want)
    # Opponent cells carry no anchor: compare as multisets (rows sorted on a
    # coarse key, then compared with the tolerance).
    def canonical(rows):
        rows = np.asarray(rows, dtype=np.float64).reshape(-1, width)
        return rows[np.lexsort(np.round(rows, 3).T[::-1])]
    want_opp = [vectorize(f, feats["opp_cell"], width) for f in exp["opp_cell"]]
    got_opp = by["opp_cell"][:, header:]
    if len(got_opp) != len(want_opp) or (
            want_opp and np.abs(canonical(got_opp) - canonical(want_opp)).max() > tol):
        problems.append("opponent cells differ")
    if len(by["opp_cell"]) and (by["opp_cell"][:, 1:3].any() or by["opp_cell"][:, 4:7].any()):
        problems.append("opponent cells must carry no entity, anchor or position")
    cubes = {}
    for t in by["cube"]:
        rel = int(t[2]) - anchors["cube"]
        cubes[(rel // 32, rel % 32 + 1)] = t
        if int(t[1]) != rel % 32 + 1 or t[4] != 1 or (t[5], t[6]) != geo.axial[rel // 32]:
            problems.append(f"cube token {rel} header")
    if sorted(cubes) != sorted(exp["cube"]):
        problems.append(f"cube tokens {sorted(cubes)} vs {sorted(exp['cube'])}")
    for key, want in exp["cube"].items():
        if key in cubes:
            cmp(f"cube {key}", cubes[key], "cube", want)
    return problems
