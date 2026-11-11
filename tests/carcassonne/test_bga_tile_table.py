"""The tile table pinned against a source that is not this codebase.

`bga_tile_types.json` is Board Game Arena's own `tile_types` table (the
geometry its Carcassonne implementation plays by) plus its base-game tile
counts, extracted unchanged from a live table capture kept in the owner's
bga-dinoboard project (tests/fixtures/carcassonne-bga-frame.json). Per tile
type it lists the city / road / field / abbey spots with the edges (1 up,
2 right, 3 bottom, 4 left) or border halves (10..17, clockwise from the left
half of the top edge) they reach, which cities each field borders, and
pennants.

Every BGA type must be exactly one of our 24 types in some rotation: same
edges, same city and road groups, same fields half-edge for half-edge, same
cities bordered by each field, same pennant / cloister, same number of
tiles. Checked for the independent reference table (carc_reference) and for
the table the engine itself publishes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import dinoboard_engine

from . import carc_reference as ref
from .test_rules_reference import _engine_tile_slots

BGA = json.loads((Path(__file__).with_name("bga_tile_types.json")).read_text(encoding="utf-8"))
BGA_TYPES = {int(k): v for k, v in BGA["tile_types"].items()}
BGA_COUNTS = {int(k): v for k, v in BGA["tile_counts"].items()}


def _spots(record, name):
    return record.get(name) or {}          # BGA writes an empty table as []


def bga_picture(record):
    """(edges, city groups, road groups, fields, pennant, cloister) of a BGA
    type in BGA's base orientation, in our side / half-edge numbering."""
    cities = {spot: frozenset(e - 1 for e in f["link"]) for spot, f in _spots(record, "city").items()}
    roads = {spot: frozenset(e - 1 for e in f["link"]) for spot, f in _spots(record, "road").items()}
    edges = tuple(ref.CITY if any(s in g for g in cities.values())
                  else ref.ROAD if any(s in g for g in roads.values()) else ref.FIELD for s in range(4))
    fields = frozenset(
        (frozenset((h - 9) % 8 for h in f["link"]), frozenset(cities[str(c)] for c in f["citylink"]))
        for f in _spots(record, "field").values())
    pennant = any(f.get("bonus") for f in _spots(record, "city").values())
    return edges, frozenset(cities.values()), frozenset(roads.values()), fields, pennant, bool(record.get("abbey"))


def rotated(picture, k):
    edges, cities, roads, fields, pennant, cloister = picture
    side = lambda group: frozenset((s + k) % 4 for s in group)
    return (tuple(edges[(s - k) % 4] for s in range(4)),
            frozenset(side(g) for g in cities), frozenset(side(g) for g in roads),
            frozenset((frozenset((h + 2 * k) % 8 for h in halves), frozenset(side(g) for g in border))
                      for halves, border in fields),
            pennant, cloister)


def reference_picture(t):
    edge, city_of, road_of, farm_of, farm_city = ref.rot_view(t, 0)
    d = ref.TILES[t]
    city = [frozenset(s for s in range(4) if city_of[s] == g) for g in range(len(d["cities"]))]
    road = [frozenset(s for s in range(4) if road_of[s] == g) for g in range(len(d["roads"]))]
    fields = frozenset(
        (frozenset(h for h in range(8) if farm_of[h] == g), frozenset(city[city_of[s]] for s in farm_city[g]))
        for g in range(len(d["farms"])))
    return tuple(edge), frozenset(city), frozenset(road), fields, d["shield"], d["chapel"]


def engine_picture(slots):
    """The same picture decoded from the geometry the engine publishes on the
    current-tile token (13 position slots x 6, tile alone and unrotated)."""
    s = slots.reshape(13, 6)
    edges = tuple(ref.CITY if s[side, 1] else ref.ROAD if s[side, 2] else ref.FIELD for side in range(4))

    def groups(terrain):
        # A slot carries the number of sides of its city / road (as open
        # edges of the lone tile); sides that share a count above 1 are one
        # group, which is unambiguous while a tile has at most one such group.
        sides = [side for side in range(4) if edges[side] == terrain]
        size = {side: int(round(s[side, 3] * 10)) for side in sides}
        multi = [side for side in sides if size[side] > 1]
        assert all(size[side] == len(multi) for side in multi), "two multi-side groups of one terrain"
        return [frozenset([side]) for side in sides if size[side] == 1] + ([frozenset(multi)] if multi else [])

    cities = groups(ref.CITY)
    farm_id = [int(round(s[4 + h, 1] * 8)) for h in range(8)]
    fields = set()
    for fid in sorted(set(farm_id) - {0}):
        halves = frozenset(h for h in range(8) if farm_id[h] == fid)
        bordered = {int(round(s[4 + h, 2] * 5)) for h in halves}
        assert len(bordered) == 1 and bordered <= {0, len(cities)}, "which cities a field borders is ambiguous"
        fields.add((halves, frozenset(cities) if bordered.pop() else frozenset()))
    pennant = any(s[side, 5] > 0 for side in range(4))
    return edges, frozenset(cities), frozenset(groups(ref.ROAD)), frozenset(fields), pennant, bool(s[12, 5] > 0)


def _match(picture_of_type):
    """BGA type -> (our type, rotation) with an identical picture."""
    out = {}
    for b, record in BGA_TYPES.items():
        hits = [(t, k) for t in range(24) for k in range(ref.rotation_period(t))
                if rotated(picture_of_type(t), k) == bga_picture(record)]
        assert len(hits) == 1, f"BGA type {b}: matches {[(ref.CODES[t], k) for t, k in hits]}"
        out[b] = hits[0]
    return out


def test_bga_table_is_the_base_game():
    assert sorted(BGA_TYPES) == list(range(1, 25))
    assert sum(BGA_COUNTS.values()) == 72
    for b, record in BGA_TYPES.items():
        edges, _, _, fields, _, _ = bga_picture(record)
        # The half-edge numbering is consistent with the edges: no field
        # reaches a city edge, every other half-edge is in exactly one field.
        halves = [h for group, _ in fields for h in group]
        assert sorted(halves) == [h for h in range(8) if edges[ref.HALF_EDGE[h]] != ref.CITY], f"BGA type {b}"


def test_reference_tile_table_matches_bga():
    match = _match(reference_picture)
    assert sorted(t for t, _ in match.values()) == list(range(24)), "not one BGA type per tile type"
    for b, (t, _) in match.items():
        assert BGA_COUNTS[b] == ref.COUNTS[t], f"tile {ref.CODES[t]} (BGA type {b}): count"


def test_engine_tile_table_matches_bga():
    slots = _engine_tile_slots()
    match = _match(lambda t: engine_picture(slots[t]))
    assert sorted(t for t, _ in match.values()) == list(range(24)), "not one BGA type per tile type"
    assert match == _match(reference_picture)
    state = dinoboard_engine.GameSession("carcassonne", seed=3).get_state_dict()
    counts = list(state["remaining_tile_counts"])
    counts[state["current_tile"]] += 1
    counts[state["board_tile"][10 * 20 + 10]] += 1      # the starting tile
    for b, (t, _) in match.items():
        assert BGA_COUNTS[b] == counts[t], f"tile {ref.CODES[t]} (BGA type {b}): count"


@pytest.mark.parametrize("bga_type,code", [(15, "D"), (22, "C"), (20, "B"), (21, "A"), (19, "X"), (16, "U"), (17, "V")])
def test_well_known_tiles_map_as_expected(bga_type, code):
    assert ref.CODES[_match(reference_picture)[bga_type][0]] == code
