"""Checkpoints are replaced atomically.

`checkpoint.pt` is rewritten at the end of every step. Written in place, a
crash or kill in the middle of the write left a truncated file and `--resume`
could not load it. Files `--resume` depends on are now written to a temp file
in the same directory and renamed over the target once complete.

The interrupted save is simulated by a torch.save that writes a few bytes and
then fails, as a process dying mid-write would leave it.
"""
import json

import pytest
import torch

from training import pipeline
from training.cli import find_game_config, find_heuristic_config, find_train_config

CONCURRENCY = {
    "enabled": True, "leaf_batch_size": 1,
    "evaluator": {"batch_size": 8, "batch_timeout_us": 50, "max_queue_size": 64,
                  "intra_op_num_threads": 1, "num_inference_threads": 1},
}


def _train(out, steps, resume=False, eval_every=0):
    train_cfg, _ = find_train_config("tictactoe")
    train_cfg["heuristic"], _ = find_heuristic_config("tictactoe")
    train_cfg["network"] = {"type": "mlp", "hidden_layers": [32, 32]}
    train_cfg["training"] = {
        **train_cfg["training"], "train_batches_per_step": 1,
        "selfplay_concurrency": CONCURRENCY,
        "playout_cap": {"full_simulations": 16, "fast_simulations": 4, "full_search_prob": 0.5},
    }
    pipeline.run_training_loop(
        game_id="tictactoe", game_config=find_game_config("tictactoe"),
        train_config=train_cfg, output_dir=out, steps=steps, episodes_per_step=4,
        eval_every=eval_every, eval_games=2, eval_benchmarks=[], max_workers=1,
        batch_size=16, learning_rate=0.001, seed=9, save_every=1,
        resume_from=str(out) if resume else None)


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_atomic_write_replaces_only_on_success(tmp_path):
    target = tmp_path / "state.json"
    with pipeline.atomic_write(target) as tmp:
        assert tmp.parent == target.parent and tmp != target
        tmp.write_text(json.dumps({"v": 1}))
        assert not target.exists()  # nothing visible until the write completed
    assert json.loads(target.read_text()) == {"v": 1}

    with pytest.raises(RuntimeError, match="died"):
        with pipeline.atomic_write(target) as tmp:
            tmp.write_text('{"v": 2')  # partial content
            raise RuntimeError("died mid-write")
    assert json.loads(target.read_text()) == {"v": 1}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["state.json"]


def test_interrupted_checkpoint_save_leaves_a_resumable_checkpoint(tmp_path, monkeypatch):
    _train(tmp_path, steps=1)
    ckpt_path = tmp_path / "checkpoint.pt"
    before = _load(ckpt_path)
    assert before["step"] == 1

    real_save = torch.save

    def dying_save(obj, f, *args, **kwargs):
        if isinstance(obj, dict) and "optimizer_state_dict" in obj:
            with open(f, "wb") as fh:
                fh.write(b"PK\x03\x04 truncated")
            raise RuntimeError("killed mid-write")
        return real_save(obj, f, *args, **kwargs)

    monkeypatch.setattr(torch, "save", dying_save)
    with pytest.raises(RuntimeError, match="killed mid-write"):
        _train(tmp_path, steps=2, resume=True)
    monkeypatch.setattr(torch, "save", real_save)

    after = _load(ckpt_path)  # a truncated file would fail to load here
    assert after["step"] == 1
    assert set(after) == set(before)
    for key, value in before["model_state_dict"].items():
        assert torch.equal(after["model_state_dict"][key], value)

    _train(tmp_path, steps=2, resume=True)
    assert _load(ckpt_path)["step"] == 2
    # Step exports are complete files, and no temp file is left behind.
    assert (tmp_path / "models" / "model_step_00002.onnx").stat().st_size > 0
    assert not list(tmp_path.rglob("*.tmp"))


def test_interrupted_best_model_update_keeps_the_previous_best(tmp_path, monkeypatch):
    """model_best.onnx is what a resumed run gates against."""
    # Gating itself is not under test: every candidate is accepted as new best.
    monkeypatch.setattr(pipeline, "run_eval_batch", lambda *a, **k: {
        "wins": 2, "losses": 0, "draws": 0, "win_rate": 1.0})
    _train(tmp_path, steps=1, eval_every=1)
    best = tmp_path / "models" / "model_best.onnx"
    latest = tmp_path / "models" / "model_latest.onnx"
    step1 = best.read_bytes()
    assert step1 == latest.read_bytes()

    def dying_copy(src, dst, *args, **kwargs):
        with open(dst, "wb") as fh:
            fh.write(b"\x08 truncated")
        raise RuntimeError("killed mid-copy")

    monkeypatch.setattr(pipeline.shutil, "copy2", dying_copy)
    with pytest.raises(RuntimeError, match="killed mid-copy"):
        _train(tmp_path, steps=2, resume=True, eval_every=1)
    monkeypatch.undo()
    assert best.read_bytes() == step1

    monkeypatch.setattr(pipeline, "run_eval_batch", lambda *a, **k: {
        "wins": 2, "losses": 0, "draws": 0, "win_rate": 1.0})
    _train(tmp_path, steps=2, resume=True, eval_every=1)
    assert best.read_bytes() == latest.read_bytes() != step1
    assert not list(tmp_path.rglob("*.tmp"))
