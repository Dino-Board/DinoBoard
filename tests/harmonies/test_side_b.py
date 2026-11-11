"""Personal board side B (default ids) next to side A (harmonies_a*).

Rulebook p.8 "Water – side B: the islands": each space or group of spaces
separated from each other by blue tokens is an island, 5 points each, and
there is always at least 1 island. The B board drawn there has columns of
4-3-4-3-4-3-4 hexes. Everything else scores as on side A.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest

import dinoboard_engine as engine

from . import harmonies_reference as ref

ROOT = Path(__file__).resolve().parents[2]
MODEL_A = ROOT / "games/harmonies/model/harmonies_a_2p.onnx"
BLUE, GRAY, YELLOW = 1, 2, 5


@pytest.mark.parametrize("game_id,side,columns", [
    ("harmonies", "B", [4, 3, 4, 3, 4, 3, 4]), ("harmonies_3p", "B", [4, 3, 4, 3, 4, 3, 4]),
    ("harmonies_tok", "B", [4, 3, 4, 3, 4, 3, 4]), ("harmonies_a", "A", [5, 4, 5, 4, 5]),
    ("harmonies_a_4p", "A", [5, 4, 5, 4, 5])])
def test_board_shape(game_id, side, columns):
    sd = engine.GameSession(game_id, seed=0).get_state_dict()
    assert sd["board_side"] == side
    assert [len(c) for c in sd["board_columns"]] == columns
    assert sum(sd["board_columns"], []) == list(range(sum(columns)))
    assert all(len(p["tower_type"]) == sum(columns) for p in sd["players"])
    assert sd["action_layout"]["end_turn"] == engine.game_metadata(game_id)["action_space"] - 1


def _landscape_after(game_id, board: dict, hex_, colour):
    """Player 0's landscape score after placing `colour` on `hex_` of a board
    whose other hexes are given as {hex: tower type} (the rest empty)."""
    gs = engine.GameSession(game_id, seed=1)
    for h, t in board.items():
        gs._force_write_field("tower_type", [0, h], t)
    gs._force_write_field("turn_started", [], True)
    for c in range(6):
        gs._force_write_field("selected_tokens", [c], 1 if c == colour - 1 else 0)
    gs.apply_action(37 + hex_ * 6 + colour - 1)
    sd = gs.get_state_dict()
    geo = ref.Geometry(sd["board_columns"])
    tt = sd["players"][0]["tower_type"]
    return sd["players"][0]["landscape_score"], ref.landscape(geo, tt, sd["board_side"]), geo


def test_side_b_islands_on_constructed_boards():
    geo = ref.Geometry(engine.GameSession("harmonies", seed=0).get_state_dict()["board_columns"])
    corner_nbrs = [v for v in geo.nbrs[0] if v >= 0]
    cases = [
        # (other hexes, placed hex, colour, islands)
        ({}, 5, YELLOW, 1),                                      # no blue: still 1 island
        ({}, 0, BLUE, 1),                                        # a blue corner separates nothing
        ({11: BLUE, 12: BLUE}, 13, BLUE, 2),                     # column 3 cuts the board in two
        ({h: BLUE for h in corner_nbrs[:-1]}, corner_nbrs[-1], BLUE, 2),  # hex 0 cut off
        ({4: BLUE, 5: BLUE, 6: BLUE, 18: BLUE, 19: BLUE}, 20, BLUE, 3),  # columns 1 and 5
        ({4: BLUE, 5: BLUE, 18: BLUE, 19: BLUE}, 20, BLUE, 2),  # column 1 has a gap at hex 6
        ({h: BLUE for h in range(1, 25)}, 0, BLUE, 1),           # no land left: at least 1 island
    ]
    for board, hex_, colour, islands in cases:
        score, parts, _ = _landscape_after("harmonies", board, hex_, colour)
        assert parts[4] == 5 * islands, (board, hex_)
        assert score == sum(parts) == 5 * islands, (board, hex_, score)


def test_side_b_islands_combine_with_other_landscapes():
    # Two mountains of height 1 side by side (1 + 1), a yellow pair (5),
    # column 3 blue (2 islands, 10).
    board = {0: GRAY, 1: GRAY, 7: YELLOW, 11: BLUE, 12: BLUE, 13: BLUE}
    score, parts, _ = _landscape_after("harmonies", board, 8, YELLOW)
    assert parts == (0, 2, 5, 0, 10) and score == 17
    # The same blue column on side A is a 3-long river (5 points); A has 23 hexes.
    score_a, parts_a, _ = _landscape_after("harmonies_a", {9: BLUE, 10: BLUE}, 11, BLUE)
    assert parts_a[4] == 5 and score_a == 5


@pytest.mark.parametrize("game_id", ["harmonies", "harmonies_3p", "harmonies_4p", "harmonies_a", "harmonies_a_3p"])
def test_landscape_scores_match_reference_in_played_games(game_id):
    """Every player's landscape score after every ply equals the independent
    scorer (both water rules), in greedy and random games."""
    checked = 0
    for seed in range(4):
        gs = engine.GameSession(game_id, seed=seed)
        rng = random.Random(seed)
        while not gs.is_terminal:
            a = gs.get_heuristic_action("greedy", 0.0)["action"] if seed % 2 else rng.choice(gs.get_legal_actions())
            gs.apply_action(a)
            sd = gs.get_state_dict()
            geo = ref.Geometry(sd["board_columns"])
            for p in sd["players"]:
                assert p["landscape_score"] == sum(ref.landscape(geo, p["tower_type"], sd["board_side"]))
                checked += 1
    assert checked > 500


@pytest.mark.parametrize("game_id", ["harmonies_3p", "harmonies_4p"])
@pytest.mark.parametrize("seed", [0, 5, 9])
def test_random_games_terminate(game_id, seed):
    gs = engine.GameSession(game_id, seed=seed)
    rng = random.Random(seed)
    for _ in range(2000):
        if gs.is_terminal:
            break
        gs.apply_action(rng.choice(gs.get_legal_actions()))
    assert gs.is_terminal
    sd = gs.get_state_dict()
    assert any(p["space_remaining"] <= 2 for p in sd["players"]) or sum(sd["pouch_counts"]) < 3


def test_side_a_model_is_only_for_side_a():
    """The shipped model (trained on side A) matches the side A encoder and
    plays harmonies_a_2p; the side B variants have other dimensions and no
    model file under their name, so nothing loads it there silently."""
    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(str(MODEL_A), providers=["CPUExecutionProvider"])
    meta_a = engine.game_metadata("harmonies_a_2p")
    assert sess.get_inputs()[0].shape[1] == meta_a["feature_dim"] == 1190
    assert meta_a["action_space"] == 912
    assert not (ROOT / "games/harmonies/model/harmonies_2p.onnx").exists()
    gs = engine.GameSession("harmonies_a_2p", seed=3, model_path=str(MODEL_A))
    assert gs.get_ai_action(simulations=8)["action"] in gs.get_legal_actions()
    meta_b = engine.game_metadata("harmonies_2p")
    assert (meta_b["feature_dim"], meta_b["action_space"]) == (1250, 988)
    with pytest.raises(Exception):
        engine.GameSession("harmonies_2p", seed=3, model_path=str(MODEL_A)).get_ai_action(simulations=8)
