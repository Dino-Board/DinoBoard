"""Eval vs heuristic with Splendor's belief tracker, model on either seat.

When the heuristic moves first its search must still get a tracker that has
seen the opening deal; otherwise randomize_unseen samples full decks and
throws on the deck_sizes mismatch.
"""
import dinoboard_engine
import pytest

from conftest import get_test_model


def test_concurrent_eval_with_heuristic_on_either_seat():
    tasks = [dict(seed=1000 + i, model_is_player=i % 2) for i in range(4)]
    out = dinoboard_engine.run_concurrent_eval_vs_heuristic_batch(
        game_id="splendor", model_path=get_test_model("splendor"),
        heuristic_selector="rule_v1", tasks=tasks, num_threads=2, simulations=16,
        max_game_plies=40, batch_size=16, max_queue_size=64, intra_op_num_threads=1)
    assert [r["total_plies"] > 0 for r in out["results"]] == [True] * 4


def test_concurrent_eval_vs_mcts_pv_heuristic_moving_first():
    out = dinoboard_engine.run_concurrent_eval_vs_heuristic_batch(
        game_id="splendor", model_path=get_test_model("splendor"),
        heuristic_selector="pv_512x4", tasks=[dict(seed=7, model_is_player=1)],
        num_threads=1, simulations=16, max_game_plies=12, batch_size=16,
        max_queue_size=64, intra_op_num_threads=1)
    assert out["results"][0]["total_plies"] > 0


@pytest.mark.parametrize("model_is_player", [0, 1])
def test_single_eval_with_heuristic_on_either_seat(model_is_player):
    out = dinoboard_engine.run_constrained_eval_vs_heuristic(
        game_id="splendor", seed=3, model_path=get_test_model("splendor"),
        simulations=16, model_is_player=model_is_player, constrained=False,
        heuristic_selector="rule_v1", max_game_plies=40)
    assert out["total_plies"] > 0
