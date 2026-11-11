"""Snapshots written by another program (GameSession.apply_initial_observation).

A client that mirrors a real table does not replay actions; it writes the
whole public state. Such a snapshot may show a symmetric tile in any of its
equal rotations and a meeple on any edge / half-edge of its feature. Two
snapshots of the same physical position must be the same state to the
engine: same legal actions, same hash, same search (so the same network
inputs in every node), and the engine's own snapshots must pass unchanged.
"""
from __future__ import annotations

import random

import pytest

import dinoboard_engine

from conftest import get_test_model

from . import carc_reference as ref

GAMES = ["carcassonne", "carcassonne_tok"]


def _pairs(flat):
    return {tuple(flat[i]): flat[i + 1] for i in range(0, len(flat), 2)}


def _flat(d):
    out = []
    for k, v in d.items():
        out.extend([list(k), v])
    return out


def _rewrite(payload, kinds, rng):
    """The same position written differently: symmetric tiles turned by a
    multiple of their period ("R") and / or meeples moved to another edge or
    half-edge of the same city, road or farm on their tile ("M"). Returns
    (payload, number of slots changed)."""
    out = dict(payload)
    bt, br = _pairs(payload["board_tile"]), _pairs(payload["board_rotation"])
    changed = 0
    if "R" in kinds:
        rot = dict(br)
        for (c,), t in bt.items():
            if t >= 0 and ref.rotation_period(t) < 4:
                rot[(c,)] = rng.choice([r for r in range(4)
                                        if r % ref.rotation_period(t) == br[(c,)] and r != br[(c,)]])
                changed += 1
        out["board_rotation"] = _flat(rot)
        ht, hr = _pairs(payload["history_tile"]), _pairs(payload["history_rotation"])
        hrow, hcol = _pairs(payload["history_row"]), _pairs(payload["history_col"])
        out["history_rotation"] = _flat({k: rot[(hrow[k] * 20 + hcol[k],)] if ht[k] >= 0 else hr[k] for k in ht})
        row, col = _pairs(payload["last_row"])[()], _pairs(payload["last_col"])[()]
        if row >= 0 and col >= 0:
            out["last_rotation"] = [[], rot[(row * 20 + col,)]]
    if "M" in kinds:
        pos = _pairs(payload["board_meeple_pos"])
        for (c,), p in list(pos.items()):
            if not 0 <= p < 12:
                continue
            # Group membership by board position is the same for every equal rotation.
            edge, city_of, road_of, farm_of, _ = ref.rot_view(bt[(c,)], br[(c,)])
            if p < 4:
                group = city_of if edge[p] == ref.CITY else road_of
                others = [s for s in range(4) if edge[s] == edge[p] and group[s] == group[p] and s != p]
            else:
                others = [4 + h for h in range(8) if farm_of[h] == farm_of[p - 4] and 4 + h != p]
            if others:
                pos[(c,)] = rng.choice(others)
                changed += 1
        out["board_meeple_pos"] = _flat(pos)
    return out, changed


def _positions(game, seed, stops):
    """(seat, engine snapshot, truth state dict) at the given plies of one
    game mixing heuristic and random moves (meeples placed often)."""
    rng = random.Random(seed)
    gs = dinoboard_engine.GameSession(game, seed=seed)
    out = []
    for ply in range(max(stops) + 1):
        if gs.is_terminal:
            break
        if ply in stops:
            seat = gs.current_player
            out.append((seat, dict(gs.extract_initial_observation(seat)["public_snapshot"]), gs.get_state_dict()))
        legal = gs.get_legal_actions()
        if rng.random() < 0.5:
            action = gs.get_heuristic_action("rule", 0.0)["action"]
        else:
            placing = [a for a in legal if a != ref.PASS]
            action = rng.choice(placing if placing and legal[0] >= ref.PASS else legal)
        gs.apply_action(action)
    assert len(out) == len(stops)
    return out


def _load(game, seat, payload, model=""):
    gs = dinoboard_engine.GameSession(game, 4711, model, False, "")
    gs.apply_initial_observation(seat, {"public_snapshot": payload, "tracker_init": {}})
    return gs


STOPS = (17, 40, 63, 86, 109, 128)      # odd plies are meeple decisions


@pytest.mark.parametrize("game", GAMES)
def test_engine_snapshot_is_stored_unchanged(game):
    for seat, payload, truth in _positions(game, 31, STOPS):
        assert _load(game, seat, payload).get_state_dict() == truth


@pytest.mark.parametrize("game", GAMES)
def test_equivalent_snapshots_are_one_state(game):
    rng = random.Random(5)
    rewritten = {"R": 0, "M": 0}
    for seat, payload, truth in _positions(game, 32, STOPS):
        base = _load(game, seat, payload)
        for kinds in ("R", "M", "RM"):
            other, changed = _rewrite(payload, kinds, rng)
            if not changed:
                continue
            for k in kinds:
                rewritten[k] += 1
            gs = _load(game, seat, other)
            state = gs.get_state_dict()
            assert state == truth, f"{kinds}: stored state differs"
            for player in (0, 1):
                assert gs.state_hash_for_perspective(player) == base.state_hash_for_perspective(player), kinds
            assert gs.get_legal_actions() == base.get_legal_actions(), kinds
            # Stored in the forms the rules produce.
            for c, t in enumerate(state["board_tile"]):
                if t < 0:
                    continue
                assert state["board_rotation"][c] < ref.rotation_period(t)
                p = state["board_meeple_pos"][c]
                if 0 <= p < 12:
                    edge, city_of, road_of, farm_of, _ = ref.rot_view(t, state["board_rotation"][c])
                    if p < 4:
                        group = city_of if edge[p] == ref.CITY else road_of
                        assert p == min(s for s in range(4) if edge[s] == edge[p] and group[s] == group[p])
                    else:
                        assert p - 4 == min(h for h in range(8) if farm_of[h] == farm_of[p - 4])
    assert rewritten["R"] >= 4 and rewritten["M"] >= 4, rewritten


def _model(game, tmp_path):
    if game != "carcassonne_tok":
        return get_test_model(game)
    # A random relative position bias, so that the search depends on where the tokens are.
    import torch
    from training.model import create_model_from_config, export_onnx
    meta = dinoboard_engine.game_metadata(game)
    torch.manual_seed(11)
    net = create_model_from_config(
        {"type": "token_transformer", "d_model": 32, "num_layers": 1, "num_heads": 4, "ffn_dim": 64,
         "relative_position_radius": 4, "anchor_embedding": False},
        input_dim=0, policy_dim=meta["action_space"], num_players=2, token_layout=meta["token_layout"])
    with torch.no_grad():
        net.rel_bias.normal_(0.0, 1.0)
    path = tmp_path / "tok.onnx"
    export_onnx(net.eval(), path, 0)
    return str(path)


@pytest.mark.parametrize("game", GAMES)
def test_equivalent_snapshots_search_the_same(game, tmp_path):
    model = _model(game, tmp_path)
    rng = random.Random(6)

    def search(payload, seat):
        result = _load(game, seat, payload, model).get_ai_action(simulations=24, temperature=0.0)
        stats = result["stats"]
        return (result["action"], list(stats["root_actions"]), list(stats["root_action_visits"]),
                stats["best_value"], list(stats["root_values"]),
                sorted((int(a), tuple(v)) for a, v in stats["action_values"].items()))

    compared = 0
    for seat, payload, _ in _positions(game, 33, STOPS[1:5]):
        expected = search(payload, seat)
        other, changed = _rewrite(payload, "RM", rng)
        if changed:
            # Exact equality: any difference in a root prior or in the value
            # of an evaluated node would show in the visit counts or means.
            assert search(other, seat) == expected
            compared += 1
    assert compared >= 3


def test_bridge_style_meeple_positions_on_a_two_field_tile():
    """Tile D turned three times: the strip between city and road reaches the
    tile border at half-edges 1 (top) and 6 (bottom). Whichever one a client
    reports, the farmer is stored at the lower one, like the engine's own."""
    gs = dinoboard_engine.GameSession("carcassonne", seed=1)
    seat = gs.current_player
    payload = dict(gs.extract_initial_observation(seat)["public_snapshot"])
    cell = 10 * 20 + 10
    states = []
    for half in (1, 6):
        variant = dict(payload)
        variant["board_rotation"] = _flat({**_pairs(payload["board_rotation"]), (cell,): 3})
        variant["history_rotation"] = _flat({**_pairs(payload["history_rotation"]), (0,): 3})
        variant["board_meeple_owner"] = _flat({**_pairs(payload["board_meeple_owner"]), (cell,): 1})
        variant["board_meeple_pos"] = _flat({**_pairs(payload["board_meeple_pos"]), (cell,): 4 + half})
        states.append(_load("carcassonne", seat, variant))
    assert states[0].get_state_dict()["board_meeple_pos"][cell] == 4 + 1
    assert states[0].get_state_dict() == states[1].get_state_dict()
    assert states[0].state_hash_for_perspective(0) == states[1].state_hash_for_perspective(0)
