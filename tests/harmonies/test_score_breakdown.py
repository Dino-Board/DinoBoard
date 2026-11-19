"""Harmonies' final score breakdown (trees, mountains, fields, buildings, water, animals per seat)
and the tok_v2_mcts teacher.

* Every variant declares 6 categories; an episode's breakdown sums to each seat's final score.
* tok_v2_mcts (when its model is present) plays legal moves in harmonies_tok.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import dinoboard_engine as engine
from conftest import get_test_model

ROOT = Path(__file__).resolve().parents[2]
TEACHER = ROOT / "games/harmonies/model/harmonies_tok_v2_teacher.onnx"
SIDE_A_TEACHER = ROOT / "games/harmonies/model/harmonies_a_2p.onnx"


@pytest.mark.parametrize("game", ["harmonies", "harmonies_a", "harmonies_tok", "harmonies_a_tok", "harmonies_3p"])
def test_breakdown_sums_to_final_scores(game, tmp_path_factory):
    meta = engine.game_metadata(game)
    assert meta["score_breakdown_categories"] == 6
    n = meta["num_players"]
    model = get_test_model(game, tmp_path_factory)
    for seed in (1, 2):
        ep = engine.run_selfplay_episode(game_id=game, seed=seed, model_path=model, simulations=2,
                                         max_game_plies=400)
        bd, final = ep["final_score_breakdown"], ep["final_scores"]
        assert len(bd) == 6 * n and all(v >= 0 for v in bd)
        for p in range(n):
            assert sum(bd[6 * p:6 * p + 6]) == pytest.approx(final[p])


def test_training_saves_and_restores_the_breakdown_head(tmp_path):
    import json

    import torch

    from training.cli import find_game_config, find_heuristic_config
    from training.pipeline import run_training_loop
    game = "harmonies_tok"
    cfg = find_game_config(game)
    cfg["max_game_plies"] = 40
    train_cfg = json.loads((ROOT / "games/harmonies/config/train_transformer.json").read_text(encoding="utf-8"))
    assert "score_breakdown_head" in train_cfg["training"]
    train_cfg["heuristic"], _ = find_heuristic_config(game)
    train_cfg["network"] = {**train_cfg["network"], "d_model": 32, "num_layers": 2, "num_heads": 4, "ffn_dim": 64}
    train_cfg["heuristic_guidance"] = {"name": "greedy", "hold_steps": 0, "steps": 0, "initial_ratio": 0.0,
                                       "temperature": 0.0}
    train_cfg["training"] = {**train_cfg["training"], "train_batches_per_step": 2, "inference_precision": "fp32",
                             "opponent_pool_enabled": False,
                             "playout_cap": {"full_simulations": 4, "fast_simulations": 2, "full_search_prob": 0.5}}
    kw = dict(game_id=game, game_config=cfg, train_config=train_cfg, output_dir=tmp_path, episodes_per_step=2,
              eval_every=0, eval_games=2, max_workers=1, batch_size=16, learning_rate=0.001, seed=5, save_every=0)
    run_training_loop(steps=2, **kw)
    ckpt = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    head = ckpt["score_breakdown_head_state_dict"]
    assert list(head.values())[-1].shape[0] == 2 * 6, "output: 6 categories x 2 seats"
    run_training_loop(steps=3, resume_from=str(tmp_path), **kw)
    resumed = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    assert resumed["step"] == 3 and "score_breakdown_head_state_dict" in resumed


@pytest.mark.skipif(not TEACHER.exists(), reason="the tok_v2 teacher model is not in games/harmonies/model")
def test_teacher_plays_legal_moves():
    gs = engine.GameSession("harmonies_tok", seed=7)
    for _ in range(3):
        r = gs.get_heuristic_action("tok_v2_mcts")
        assert r["action"] in gs.get_legal_actions()
        gs.apply_action(int(r["action"]))


@pytest.mark.skipif(not SIDE_A_TEACHER.exists(), reason="the side A MLP is not in games/harmonies/model")
def test_side_a_teacher_plays_legal_moves():
    """a_flat_mcts reads the state with the MLP encoding inside the token variant harmonies_a_tok."""
    gs = engine.GameSession("harmonies_a_tok", seed=7)
    for _ in range(12):
        r = gs.get_heuristic_action("a_flat_mcts")
        assert r["action"] in gs.get_legal_actions()
        gs.apply_action(int(r["action"]))
