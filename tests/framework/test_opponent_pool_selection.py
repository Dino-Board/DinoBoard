"""run_concurrent_selfplay_step must sample the opponent pool, not take a fixed prefix.

With more checkpoints than `opponent_pool_max_distinct`, different steps
(different `seeds[0]`) have to see different subsets of the pool; a fixed
prefix would pin every step to the oldest checkpoints.
"""
import shutil
from pathlib import Path

import dinoboard_engine
from training.pipeline import _collect_pool_paths

ROOT = Path(__file__).resolve().parents[2]


def _active_opponents(latest, pool, seed0):
    out = dinoboard_engine.run_concurrent_selfplay_step(
        game_id="splendor", seeds=[seed0, seed0 + 1], latest_model_path=str(latest),
        pool_paths=pool, num_self_threads=1, num_pool_threads=1,
        opponent_pool_max_distinct=2, simulations=2, max_game_plies=4,
        batch_size=4, max_queue_size=8, intra_op_num_threads=1)
    paths = out["stats"]["active_opp_paths"]
    assert len(paths) == 2
    return tuple(sorted(Path(p).name for p in paths))


def test_pool_selection_varies_with_seed(tmp_path):
    src = ROOT / "games" / "splendor" / "model" / "splendor_2p.onnx"
    models = tmp_path / "models"
    models.mkdir()
    for step in (100, 200, 300, 400, 500):
        shutil.copy(src, models / f"model_step_{step:05d}.onnx")
    latest = models / "model_latest.onnx"
    shutil.copy(src, latest)
    pool = [pv for pv, _ in _collect_pool_paths(models)]
    assert len(pool) == 5

    picks = {_active_opponents(latest, pool, seed0) for seed0 in range(1, 9)}
    assert len(picks) > 1, f"every step picked the same opponents: {picks}"
    assert _active_opponents(latest, pool, 777) == _active_opponents(latest, pool, 777)
