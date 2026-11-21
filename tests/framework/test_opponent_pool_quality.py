"""Opponent pool sampling by quality scores (OpenAI Five, appendix N).

Each checkpoint enters at the top score; the latest model beating it in pool
games lowers its score, so the draw drifts towards the checkpoints the latest
model still loses to. Only the pool games' outcomes are used: no extra games.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import dinoboard_engine
from training import pipeline
from training.cli import find_game_config, find_heuristic_config, find_train_config
from training.opponent_pool import OpponentPool, PoolConfig, Selection, parse_pool_config

ROOT = Path(__file__).resolve().parents[2]


def _paths(*steps):
    return [(f"/m/model_step_{s:05d}.onnx", "") for s in steps]


def _ep(opp_step, latest_won, draw=False):
    return {"pool_opponent": f"/m/model_step_{opp_step:05d}.onnx", "latest_seat": 0,
            "winner": -1 if draw else (0 if latest_won else 1), "draw": draw}


def test_new_checkpoints_enter_at_the_top_score_and_deleted_ones_leave():
    pool = OpponentPool(PoolConfig(), {"scores": {"model_step_00010.onnx": -2.0,
                                                  "model_step_00020.onnx": -0.5,
                                                  "model_step_00005.onnx": -9.0}})
    pool.candidates(_paths(10, 20, 30), step=31)
    assert pool.scores == {"model_step_00010.onnx": -2.0, "model_step_00020.onnx": -0.5,
                           "model_step_00030.onnx": -0.5}


def test_window_keeps_recent_checkpoints_and_never_leaves_the_pool_empty():
    pool = OpponentPool(PoolConfig(window=0.5))
    assert [p for p, _ in pool.candidates(_paths(10, 40, 60, 90), step=100)] == [
        "/m/model_step_00060.onnx", "/m/model_step_00090.onnx"]
    assert len(OpponentPool(PoolConfig(window=0.01)).candidates(_paths(10, 20), step=100)) == 1


def test_update_lowers_beaten_opponents_only():
    """Win 1, draw 1/N, loss 0, scaled by lr * G / (N * inclusion)."""
    pool = OpponentPool(PoolConfig(learning_rate=0.1))
    pool.candidates(_paths(10, 20, 30, 40), step=41)
    sel = Selection(paths=_paths(10, 20, 30), pool_size=4,
                    inclusion={"model_step_00010.onnx": 0.5, "model_step_00020.onnx": 1.0,
                               "model_step_00030.onnx": 1.0})
    eps = [_ep(10, True), _ep(10, True), _ep(20, True, draw=True), _ep(20, True, draw=True),
           _ep(30, False), _ep(30, False), {"pool_opponent": None}]
    res = pool.update(sel, eps, num_players=2)
    assert res == {"model_step_00010.onnx": (2, 2.0), "model_step_00020.onnx": (2, 1.0),
                   "model_step_00030.onnx": (2, 0.0)}
    assert pool.scores["model_step_00010.onnx"] == pytest.approx(-0.1 * 6 * 1.0 / (4 * 0.5))
    assert pool.scores["model_step_00020.onnx"] == pytest.approx(-0.1 * 6 * 0.5 / 4)
    assert pool.scores["model_step_00030.onnx"] == 0.0
    assert pool.scores["model_step_00040.onnx"] == 0.0

    before = dict(pool.scores)
    pool.update(sel, eps, num_players=2, weight=0.0)
    assert pool.scores == before
    with pytest.raises(ValueError, match="did not draw"):
        pool.update(sel, [_ep(40, True)], num_players=2)


def test_uniform_mode_keeps_the_engine_draw_and_scores():
    pool = OpponentPool(PoolConfig(sampling="uniform"))
    sel = pool.select(_paths(10, 20, 30, 40, 50), step=51, rng=np.random.default_rng(0))
    assert len(sel.paths) == 5
    pool.update(sel, [_ep(10, True)], num_players=2)
    assert pool.scores["model_step_00010.onnx"] == 0.0


def test_draw_is_reproducible_and_takes_all_of_a_small_pool():
    pool = OpponentPool(PoolConfig(max_distinct=3))
    assert len(pool.select(_paths(10, 20), step=21, rng=np.random.default_rng(1)).paths) == 2
    paths = _paths(*range(10, 110, 10))
    a = pool.select(paths, step=111, rng=np.random.default_rng(7))
    b = pool.select(paths, step=111, rng=np.random.default_rng(7))
    assert a.paths == b.paths and 1 <= len(a.paths) <= 3


def test_outgrown_checkpoints_fade_from_the_draw():
    """The latest model always beats the five oldest checkpoints and never the
    newest: the newest ends up drawn most, the oldest little beyond the uniform mix."""
    pool = OpponentPool(PoolConfig(uniform_mix=0.1))
    paths = _paths(10, 20, 30, 40, 50, 60)
    rng = np.random.default_rng(3)
    for step in range(61, 261):
        sel = pool.select(paths, step, rng)
        eps = [_ep(int(Path(p).stem[-5:]), latest_won=not p.endswith("00060.onnx"))
               for p, _ in sel.paths for _ in range(10)]
        pool.update(sel, eps, num_players=2)
    p = pool.probabilities(pool.candidates(paths, 261))
    assert p[-1] > 0.85
    assert 0.1 / 6 <= p[0] < 0.025


@pytest.mark.parametrize("cfg, msg", [
    ({"opponent_pool_enabled": True, "opponent_pool_sampling": "elo"}, "one of"),
    ({"opponent_pool_enabled": True, "opponent_pool_window": 0.0}, "window"),
    ({"opponent_pool_enabled": True, "opponent_pool_uniform_mix": 1.5}, "uniform_mix"),
    ({"opponent_pool_sampling": "quality"}, "needs opponent_pool_enabled"),
])
def test_config_is_validated(cfg, msg):
    with pytest.raises(ValueError, match=msg):
        parse_pool_config(cfg, 3)


def test_engine_tags_each_pool_episode_with_its_opponent(tmp_path):
    src = ROOT / "games" / "splendor" / "model" / "splendor_2p.onnx"
    pool = []
    for step in (100, 200):
        shutil.copy(src, tmp_path / f"model_step_{step:05d}.onnx")
        pool.append(str(tmp_path / f"model_step_{step:05d}.onnx"))
    latest = tmp_path / "model_latest.onnx"
    shutil.copy(src, latest)
    out = dinoboard_engine.run_concurrent_selfplay_step(
        game_id="splendor", seeds=[1, 2, 3], latest_model_path=str(latest),
        pool_paths=pool, num_self_threads=0, num_pool_threads=1,
        opponent_pool_max_distinct=2, simulations=2, max_game_plies=4,
        batch_size=4, max_queue_size=8, intra_op_num_threads=1)
    assert all(ep["latest_seat"] >= 0 and ep["pool_opponent"] in pool for ep in out["episodes"])


def test_training_keeps_the_scores_across_resume(tmp_path):
    train_cfg, _ = find_train_config("tictactoe")
    train_cfg["heuristic"], _ = find_heuristic_config("tictactoe")
    train_cfg["network"] = {"type": "mlp", "hidden_layers": [16]}
    train_cfg["training"] = {
        **train_cfg["training"], "train_batches_per_step": 1,
        "opponent_pool_enabled": True, "opponent_pool_self_ratio": 0.5,
        "selfplay_concurrency": {
            "enabled": True, "leaf_batch_size": 1,
            "evaluator": {"batch_size": 8, "batch_timeout_us": 50, "max_queue_size": 64,
                          "intra_op_num_threads": 1, "num_inference_threads": 1}},
        "playout_cap": {"full_simulations": 8, "fast_simulations": 2, "full_search_prob": 0.5},
    }

    def train(steps, resume):
        pipeline.run_training_loop(
            game_id="tictactoe", game_config=find_game_config("tictactoe"),
            train_config=train_cfg, output_dir=tmp_path, steps=steps, episodes_per_step=6,
            eval_every=0, eval_games=2, eval_benchmarks=[], max_workers=2,
            batch_size=16, learning_rate=0.001, seed=5, save_every=1,
            resume_from=str(tmp_path) if resume else None)

    state = tmp_path / "models" / "opponent_pool.json"
    train(3, resume=False)
    scores = json.loads(state.read_text())["scores"]
    assert set(scores) == {"model_step_00001.onnx", "model_step_00002.onnx"}
    scores["model_step_00001.onnx"] = -7.0
    state.write_text(json.dumps({"scores": scores}))
    train(4, resume=True)
    after = json.loads(state.read_text())["scores"]
    assert after["model_step_00001.onnx"] <= -7.0
    assert "model_step_00003.onnx" in after
