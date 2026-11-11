"""W/L/D labels in games where there is no single winner.

In 3-4 player games several seats can share first place (or last). winner()
is then -1, and labelling such a game `draw = winner < 0` counted a candidate
that shared the win — or lost to tied leaders — as a draw. Labels follow the
seat's own terminal value instead (positive win, negative loss, zero draw);
2-player results are unchanged.
"""
import importlib.util

import dinoboard_engine as engine
import pytest

from conftest import PROJECT_ROOT, get_test_model
from training.pipeline import outcome_label


def _eval_model():
    spec = importlib.util.spec_from_file_location(
        "eval_model_script", PROJECT_ROOT / "scripts" / "eval_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_outcome_label_is_the_sign_of_the_seats_value():
    assert [outcome_label(v) for v in (1.0, 0.5, 1 / 3, 0.0, -0.5, -1.0)] == [
        "win", "win", "win", "draw", "loss", "loss"]


def test_eval_model_labels_shared_places_and_keeps_two_player_results():
    label = _eval_model().test_side_outcome
    shared_first = dict(winner=-1, draw=False, terminal_z_values=[0.5, 0.5, -1.0])
    assert [label(shared_first, s) for s in range(3)] == ["win", "win", "loss"]
    behind_tied = dict(winner=-1, draw=False, terminal_z_values=[-1.0, 0.5, 0.5])
    assert label(behind_tied, 0) == "loss"
    # A `winner` naming another seat of the shared win must not make it a loss.
    assert label(dict(winner=1, draw=False, terminal_z_values=[0.5, 0.5, -1.0]), 0) == "win"
    assert label(dict(winner=-1, draw=True, terminal_z_values=[0.0, 0.0, 0.0]), 1) == "draw"
    # 2 players: same labels as the winner / draw fields always gave.
    assert label(dict(winner=0, draw=False, terminal_z_values=[1.0, -1.0]), 0) == "win"
    assert label(dict(winner=0, draw=False, terminal_z_values=[1.0, -1.0]), 1) == "loss"
    assert label(dict(winner=-1, draw=True, terminal_z_values=[0.0, 0.0]), 0) == "draw"
    # Cut off at the ply cap (every seat -1, or no values): adjudicator's verdict.
    assert label(dict(winner=1, draw=False, terminal_z_values=[-1.0, -1.0]), 1) == "win"
    assert label(dict(winner=-1, draw=True, terminal_z_values=[]), 0) == "draw"


@pytest.mark.parametrize("batched", [False, True])
def test_eval_vs_heuristic_draw_flag_needs_equal_values(batched):
    game, seeds = "6nimmit_3p", list(range(1, 25))
    model = get_test_model(game)
    if batched:
        results = engine.run_concurrent_eval_vs_heuristic_batch(
            game_id=game, model_path=model, heuristic_selector="rule_v1",
            tasks=[dict(seed=s, model_is_player=s % 3) for s in seeds], num_threads=2,
            simulations=8, max_game_plies=600, batch_size=8, max_queue_size=64,
            intra_op_num_threads=1)["results"]
    else:
        results = [engine.run_constrained_eval_vs_heuristic(
            game_id=game, seed=s, model_path=model, simulations=8, model_is_player=s % 3,
            constrained=False, heuristic_selector="rule_v1", max_game_plies=600) for s in seeds]
    for r in results:
        z = [round(float(v), 4) for v in r["terminal_z_values"]]
        assert len(z) == 3 and abs(sum(z)) < 1e-3
        assert r["draw"] == (len(set(z)) == 1), r
    # 3 players: nobody ever gets 0 here, so no game is a draw for the candidate.
    assert all(outcome_label(r["terminal_z_values"][s % 3]) != "draw"
               for r, s in zip(results, seeds))


@pytest.mark.parametrize("game_id", ["tictactoe", "harmonies", "loveletter"])
def test_two_player_labels_match_the_winner_and_draw_fields(game_id):
    label = _eval_model().test_side_outcome
    selector = {"tictactoe": "rule", "harmonies": "rule", "loveletter": "rule_v1"}[game_id]
    model = get_test_model(game_id)
    for seed in range(1, 13):
        seat = seed % 2
        r = engine.run_constrained_eval_vs_heuristic(
            game_id=game_id, seed=seed, model_path=model, simulations=8, model_is_player=seat,
            constrained=False, heuristic_selector=selector, max_game_plies=600)
        assert r["draw"] == (r["winner"] < 0)
        old = "draw" if r["draw"] else ("win" if r["winner"] == seat else "loss")
        assert label(r, seat) == old
