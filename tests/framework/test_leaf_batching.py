"""Leaf batching (NetMctsConfig.leaf_batch_size > 1) through the concurrent bindings.

Every simulation must still back up exactly once (root visits == simulations),
for a hidden-info game with a belief tracker (splendor) and a fully public one
(quoridor), and the bindings must refuse a queue too small for
threads * leaf_batch_size instead of overflowing mid-run.
"""
import dinoboard_engine
import pytest

from conftest import get_test_model

SIMS = 48


@pytest.mark.parametrize("game_id", ["splendor", "quoridor"])
@pytest.mark.parametrize("leaf", [4, 16])
def test_selfplay_root_visits_equal_simulations(game_id, leaf):
    out = dinoboard_engine.run_concurrent_selfplay_step(
        game_id=game_id, seeds=[11, 12, 13], latest_model_path=get_test_model(game_id),
        pool_paths=[], num_self_threads=2, num_pool_threads=0,
        simulations=SIMS, max_game_plies=12, tail_solve_enabled=False,
        batch_size=32, max_queue_size=64, intra_op_num_threads=1,
        leaf_batch_size=leaf)
    checked = 0
    for ep in out["episodes"]:
        for s in ep["samples"]:
            if len(s["policy_action_ids"]) == 1:  # forced move: no search, one-hot
                assert s["simulations"] == 0 and list(s["policy_action_visits"]) == [1]
                continue
            assert s["simulations"] == SIMS
            assert sum(s["policy_action_visits"]) == SIMS, (
                f"{game_id} leaf={leaf}: root visits {sum(s['policy_action_visits'])} != {SIMS}")
            checked += 1
    assert checked > 0


def test_arena_with_leaf_batching_completes():
    model = get_test_model("splendor")
    seat = dict(model_path=model, simulations=SIMS, temperature=0.0, leaf_batch_size=8)
    tasks = [dict(seed=7 + i, seat_configs=[dict(seat), dict(seat, leaf_batch_size=1)])
             for i in range(2)]
    out = dinoboard_engine.run_concurrent_arena_batch(
        game_id="splendor", tasks=tasks, num_threads=2, max_game_plies=10,
        batch_size=16, max_queue_size=32, intra_op_num_threads=1)
    assert len(out["results"]) == 2


def test_selfplay_rejects_queue_smaller_than_inflight_leaves():
    with pytest.raises(ValueError, match="leaf_batch_size"):
        dinoboard_engine.run_concurrent_selfplay_step(
            game_id="splendor", seeds=[1], latest_model_path=get_test_model("splendor"),
            pool_paths=[], num_self_threads=4, num_pool_threads=0,
            simulations=8, max_game_plies=2, batch_size=8, max_queue_size=16,
            intra_op_num_threads=1, leaf_batch_size=8)


# splendor's pv_512x4 sets heuristic.json leaf_batch_size=8, so its shared
# batched evaluator needs threads * 8 queue slots even when the main search
# is unbatched.
def test_selfplay_rejects_queue_too_small_for_guidance_leaves():
    with pytest.raises(ValueError, match="guidance heuristic 'pv_512x4' leaf_batch_size"):
        dinoboard_engine.run_concurrent_selfplay_step(
            game_id="splendor", seeds=[1], latest_model_path=get_test_model("splendor"),
            pool_paths=[], num_self_threads=4, num_pool_threads=0,
            simulations=8, max_game_plies=2, batch_size=8, max_queue_size=16,
            intra_op_num_threads=1, leaf_batch_size=1,
            heuristic_guidance_ratio=1.0, heuristic_selector="pv_512x4")


def test_eval_rejects_queue_too_small_for_heuristic_leaves():
    with pytest.raises(ValueError, match="heuristic 'pv_512x4' leaf_batch_size"):
        dinoboard_engine.run_concurrent_eval_vs_heuristic_batch(
            game_id="splendor", model_path=get_test_model("splendor"),
            heuristic_selector="pv_512x4", tasks=[dict(seed=1, model_is_player=0)],
            num_threads=4, simulations=8, max_game_plies=2, batch_size=8,
            max_queue_size=16, intra_op_num_threads=1)


@pytest.mark.parametrize("devices, threads, providers, match", [
    ([0, 1], 3, ["cuda"], "multiple of the 2 devices"),
    ([0], 1, ["cpu"], "needs a 'cuda' or 'tensorrt' provider"),
    ([1, 1], 2, ["cuda"], "duplicate device id 1"),
])
def test_evaluator_rejects_bad_devices(devices, threads, providers, match):
    seat = dict(model_path=get_test_model("splendor"), simulations=4)
    with pytest.raises(RuntimeError, match=match):
        dinoboard_engine.run_concurrent_arena_batch(
            game_id="splendor", tasks=[dict(seed=1, seat_configs=[seat, seat])],
            num_threads=1, max_game_plies=2,
            num_inference_threads=threads, providers=providers, devices=devices)


def test_guided_selfplay_with_batched_guidance_completes():
    out = dinoboard_engine.run_concurrent_selfplay_step(
        game_id="splendor", seeds=[21, 22], latest_model_path=get_test_model("splendor"),
        pool_paths=[], num_self_threads=2, num_pool_threads=0,
        simulations=SIMS, max_game_plies=6, tail_solve_enabled=False,
        batch_size=32, max_queue_size=64, intra_op_num_threads=1, leaf_batch_size=8,
        heuristic_guidance_ratio=1.0, heuristic_selector="pv_512x4")
    assert sum(len(ep["samples"]) for ep in out["episodes"]) > 0
