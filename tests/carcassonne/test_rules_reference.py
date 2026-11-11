"""Carcassonne rules against the independent reference (carc_reference).

Covers the tile table itself (farm half-edges must agree with edge terrain,
for every tile and rotation), farm connectivity between every pair of
adjacent tiles, one action per geometrically distinct placement, and whole
games compared step by step (legal sets, scoring, meeple returns, final
scoring, winner, score projection).
"""
from __future__ import annotations

import ast
import functools
import random
import re
from pathlib import Path

import numpy as np
import pytest

import dinoboard_engine

from . import carc_reference as ref
from . import token_reference as tok
from .test_checklist import (
    COLS, MEEPLE_PASS_ACTION, STAGE_MEEPLE, _cell, _clear_board, _state)

GAME = "carcassonne"
TOK = "carcassonne_tok"
TOK_HEADER = 7


# ---------------------------------------------------------------------------
# Tile table
# ---------------------------------------------------------------------------

def _assert_farms_agree_with_edges(code, edge, farm_of, farm_city):
    """A farm is bordered by roads, cities and the edge of the map, and
    adjacent tiles always show the same terrain on their shared edge. So a
    half-edge on a city edge is never field (nothing to connect to), every
    other half-edge is field, and nothing can separate the two halves of a
    field edge."""
    for h in range(8):
        on_city_edge = edge[ref.HALF_EDGE[h]] == ref.CITY
        assert (farm_of[h] >= 0) != on_city_edge, f"tile {code}: half-edge {h} vs its edge terrain"
    for side, (h1, h2) in enumerate(ref.HALVES_OF_SIDE):
        if edge[side] == ref.FIELD:
            assert farm_of[h1] == farm_of[h2], f"tile {code}: field edge {side} split between two farms"
    for sides in farm_city:
        assert all(edge[s] == ref.CITY for s in sides), f"tile {code}: farm borders a city on a non-city edge"


def test_reference_tile_table_is_consistent_for_all_tiles_and_rotations():
    assert sum(ref.COUNTS) == 72 and len(ref.TILES) == 24
    for t, d in enumerate(ref.TILES):
        for rot in range(4):
            edge, city_of, road_of, farm_of, farm_city = ref.rot_view(t, rot)
            for s in range(4):
                assert (city_of[s] >= 0) == (edge[s] == ref.CITY)
                assert (road_of[s] >= 0) == (edge[s] == ref.ROAD)
            _assert_farms_agree_with_edges(ref.CODES[t], edge, farm_of, farm_city)
    assert [ref.rotation_period(t) for t in range(24)] == [
        4 if c not in "BCXFGHU" else 1 if c in "BCX" else 2 for c in ref.CODES]


@functools.lru_cache(maxsize=1)
def _engine_tile_slots():
    """Unrotated geometry of each tile type as the engine publishes it on the
    current-tile token of the opening position (drawn tile varies by seed)."""
    lay = dinoboard_engine.game_metadata(TOK)["token_layout"]
    slots = {}
    for seed in range(2000):
        x = np.asarray(dinoboard_engine.encode_state(TOK, seed)["features"],
                       dtype=np.float32).reshape(-1, lay["token_dim"])
        slots.setdefault(int(x[1, 1]) - 1, x[1, TOK_HEADER + tok.SLOTS:TOK_HEADER + tok.SLOTS + 78])
        if len(slots) == 24:
            return slots
    raise AssertionError(f"tile types never drawn first: {sorted(set(range(24)) - set(slots))}")


def test_engine_tile_table_farms_agree_with_edges_for_all_tiles_and_rotations():
    slots = _engine_tile_slots()
    for t in range(24):
        s = slots[t].reshape(13, 6)
        # Decode the engine's table: side terrain, farm membership of each
        # half-edge (halves of one farm share the code in column 1).
        edge0 = [ref.CITY if s[side, 1] else ref.ROAD if s[side, 2] else ref.FIELD for side in range(4)]
        farm0 = [int(round(s[4 + h, 1] * 8)) - 1 for h in range(8)]
        assert edge0 == ref.rot_view(t, 0)[0], f"tile {ref.CODES[t]}: edges"
        for rot in range(4):
            edge = [edge0[(side - rot) % 4] for side in range(4)]
            farm_of = [-1] * 8
            for h in range(8):
                farm_of[(h + 2 * rot) % 8] = farm0[h]
            _assert_farms_agree_with_edges(ref.CODES[t], edge, farm_of, [])
        # And the whole description equals the reference tile.
        assert tok.mismatches(slots[t], tok.lone_tile_slots(t)) == [], f"tile {ref.CODES[t]}"


def test_farm_links_between_every_pair_of_adjacent_tiles():
    """For every two tiles that may sit side by side (all rotations): put an
    opposing farmer on each half-edge position of the left tile in turn and
    compare the meeple options on the right tile with the reference. A farm
    half-edge on a city edge (the old tile D) shows up as a farm blocked
    through a city wall."""
    gs = dinoboard_engine.GameSession(GAME, seed=1)
    _clear_board(gs)
    left, right = _cell(10, 10), _cell(10, 11)
    gs._force_write_field("stage", [], STAGE_MEEPLE)
    gs._force_write_field("last_row", [], 10)
    gs._force_write_field("last_col", [], 11)
    gs._force_write_field("board_meeple_owner", [left], 1)
    pairs = 0
    for t1 in range(24):
        gs._force_write_field("board_tile", [left], t1)
        for r1 in range(ref.rotation_period(t1)):
            gs._force_write_field("board_rotation", [left], r1)
            edge1, _, _, farm1, _ = ref.rot_view(t1, r1)
            for t2 in range(24):
                for r2 in range(ref.rotation_period(t2)):
                    if ref.rot_view(t2, r2)[0][ref.L] != edge1[ref.R]:
                        continue
                    pairs += 1
                    gs._force_write_field("board_tile", [right], t2)
                    gs._force_write_field("board_rotation", [right], r2)
                    bt, br = [-1] * 400, [0] * 400
                    bt[left], br[left], bt[right], br[right] = t1, r1, t2, r2
                    for h in range(8):
                        gs._force_write_field("board_meeple_pos", [left], 4 + h)
                        owner, pos = [-1] * 400, [-1] * 400
                        if farm1[h] >= 0:      # elsewhere there is no field to stand on
                            owner[left], pos[left] = 1, 4 + h
                        want = ref.Board(bt, br, owner, pos).legal_meeples(right, 7)
                        assert set(gs.get_legal_actions()) == want, (
                            f"{ref.CODES[t1]} rot {r1} | {ref.CODES[t2]} rot {r2}, farmer on half-edge {h}")
    assert pairs > 1500


def test_web_client_tables_match_reference():
    """The web client keeps its own copy of the farm half-edge groups (to
    draw farmers) and of the rotation periods (to send the lowest equivalent
    rotation, the only one the engine accepts)."""
    js = (Path(__file__).resolve().parents[2] / "games/carcassonne/web/carcassonne.js").read_text(encoding="utf-8")
    farm_groups = ast.literal_eval(re.search(r"const FARM_GROUPS = (\[.*?\n\]);", js, re.S).group(1))
    assert len(farm_groups) == 24
    for t, groups in enumerate(farm_groups):
        assert [sorted(g) for g in groups] == [sorted(halves) for halves, _ in ref.TILES[t]["farms"]], ref.CODES[t]
    periods = {int(k): int(v) for k, v in re.findall(
        r"(\d+): (\d+)", re.search(r"const ROTATION_PERIOD = \{(.*?)\};", js).group(1))}
    assert periods == {t: ref.rotation_period(t) for t in range(24) if ref.rotation_period(t) != 4}


# ---------------------------------------------------------------------------
# One action per distinct placement
# ---------------------------------------------------------------------------

def _replay(seed, actions):
    gs = dinoboard_engine.GameSession(GAME, seed=seed)
    for a in actions:
        gs.apply_action(a)
    return gs


def _picture(state):
    """The position as drawn on the table: what lies where, and which tile
    was placed last (the one the meeple may go on)."""
    return (state["last_row"], state["last_col"],
            frozenset((c, ref.shape(t, state["board_rotation"][c]))
                      for c, t in enumerate(state["board_tile"]) if t >= 0))


@pytest.mark.parametrize("prefix_plies", [0, 28])
def test_legal_placements_equal_distinct_resulting_states_for_every_tile(prefix_plies):
    seed = 11
    gs = dinoboard_engine.GameSession(GAME, seed=seed)
    prefix = []
    for _ in range(prefix_plies):
        prefix.append(gs.get_heuristic_action("rule", 0.0)["action"])
        gs.apply_action(prefix[-1])
    base = _state(gs)
    assert base["stage"] == 0
    placed = 0
    for tile in range(24):
        gs = _replay(seed, prefix)
        gs._force_write_field("current_tile", [], tile)
        legal = gs.get_legal_actions()
        every_fit = ref.legal_placements(base["board_tile"], base["board_rotation"], tile, all_rotations=True)
        distinct = {(a // 4, ref.shape(tile, a % 4)) for a in every_fit}
        assert {(a // 4, ref.shape(tile, a % 4)) for a in legal} == distinct
        assert len(legal) == len(distinct)
        pictures = set()
        for action in legal:
            after = _replay(seed, prefix)
            after._force_write_field("current_tile", [], tile)
            after.apply_action(action)
            state = _state(after)
            last = state["last_row"] * COLS + state["last_col"]
            assert state["board_rotation"][last] < ref.rotation_period(tile)
            pictures.add(_picture(state))
        assert len(pictures) == len(legal), f"tile {ref.CODES[tile]}: two actions give the same board"
        placed += len(legal)
    assert placed > 24


# ---------------------------------------------------------------------------
# Whole games
# ---------------------------------------------------------------------------

def _check_game(seed, policy):
    gs = dinoboard_engine.GameSession(GAME, seed=seed)
    rng = random.Random(seed)
    d = _state(gs)
    plies = 0
    while not d["is_terminal"]:
        where = f"seed {seed} ply {plies}"
        legal = gs.get_legal_actions()
        bt, br, ow, ps = d["board_tile"], d["board_rotation"], d["board_meeple_owner"], d["board_meeple_pos"]
        assert d["projected_final_scores"] == ref.projected_scores(bt, br, ow, ps, d["scores"]), where
        assert all(br[c] < ref.rotation_period(t) for c, t in enumerate(bt) if t >= 0), where
        if d["stage"] == 0:
            assert set(legal) == ref.legal_placements(bt, br, d["current_tile"]), where
            assert len(set(legal)) == len(legal), where
        else:
            cell = d["last_row"] * COLS + d["last_col"]
            mover_left = d["meeples_remaining"][d["current_player"]]
            assert set(legal) == ref.Board(bt, br, ow, ps).legal_meeples(cell, mover_left), where
        action = gs.get_heuristic_action("rule", 0.0)["action"] if policy == "rule" else rng.choice(legal)
        before = d
        gs.apply_action(action)
        d = _state(gs)
        plies += 1
        if before["stage"] == 0:
            assert d["stage"] == STAGE_MEEPLE and d["scores"] == before["scores"], where
            assert d["current_player"] == before["current_player"], where
            continue
        # Turn end: score what the mover's tile (and meeple) completed.
        owner, pos = list(before["board_meeple_owner"]), list(before["board_meeple_pos"])
        left = list(before["meeples_remaining"])
        if action != MEEPLE_PASS_ACTION:
            cell = before["last_row"] * COLS + before["last_col"]
            owner[cell], pos[cell] = before["current_player"], action - ref.SIDE
            left[before["current_player"]] -= 1
        gained, returned = ref.Board(before["board_tile"], before["board_rotation"], owner, pos).completion()
        scores = [before["scores"][p] + gained[p] for p in (0, 1)]
        if d["is_terminal"]:
            final, _ = ref.Board(d["board_tile"], d["board_rotation"], d["board_meeple_owner"],
                                 d["board_meeple_pos"]).final()
            assert d["final_score_deltas"] == final, where
            scores = [scores[p] + final[p] for p in (0, 1)]
            winner = -1 if scores[0] == scores[1] else int(scores[1] > scores[0])
            assert d["winner"] == winner and d["shared_victory"] == (winner == -1), where
            assert sorted(gs.winners) == ([0, 1] if winner == -1 else [winner]), where
            assert d["current_tile"] == -1 and d["deck_count"] == 0, where
        else:
            assert d["current_player"] == 1 - before["current_player"] and d["stage"] == 0, where
        assert d["scores"] == scores, where
        assert d["meeples_remaining"] == [left[p] + returned[p] for p in (0, 1)], where
    return plies


@pytest.mark.parametrize("policy,seed", [("rule", 101), ("rule", 102), ("rule", 103),
                                         ("random", 201), ("random", 202), ("random", 203)])
def test_whole_game_matches_reference(policy, seed):
    assert _check_game(seed, policy) >= 130
