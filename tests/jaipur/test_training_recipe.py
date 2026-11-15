"""Jaipur training recipe (games/jaipur/config/train.json).

* final_score_extractor reports each seat's full score (public rupees plus
  the face-down bonus tokens), which training.score_head needs.
* The recipe builds the token transformer and enables the KataGo-style
  options (playout cap, score head, opponent-policy head, EMA, FPU).
* Two steps of the real pipeline run with it (shrunk): step 1 played by the
  rule_mcts guide, step 2 by the network.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import dinoboard_engine as engine
from conftest import get_test_model

GAME = "jaipur"
ROOT = Path(__file__).resolve().parents[2]
TRAIN_CFG = ROOT / "games/jaipur/config/train.json"


def _train_cfg():
    return json.loads(TRAIN_CFG.read_text(encoding="utf-8"))


def test_final_scores_are_rupees_plus_bonus_tokens():
    assert engine.game_metadata(GAME)["has_final_score_extractor"]
    finished = 0
    for seed in range(6):
        ep = engine.run_heuristic_episode(game_id=GAME, seed=seed, temperature=1.0, max_game_plies=200,
                                          heuristic_selector="rule")
        scores = [int(v) for v in ep["final_scores"]]
        # Same seed and same moves on a session: the truth state at the end.
        replay = engine.GameSession(GAME, model_path="", seed=seed)
        for s in ep["samples"]:
            assert s["action_id"] in replay.get_all_legal_actions()
            replay.apply_action(s["action_id"])
        if not replay.is_terminal:
            continue
        finished += 1
        st = replay.get_state_dict()
        assert scores == [st[f"rupees_{p}"] + st[f"bonus_total_{p}"] for p in range(2)]
        assert st["winner"] == ep["winner"]
        if ep["winner"] >= 0 and scores[0] != scores[1]:
            assert scores[ep["winner"]] == max(scores)
    assert finished >= 4


def test_selfplay_reports_final_scores():
    ep = engine.run_selfplay_episode(game_id=GAME, seed=3, model_path=get_test_model(GAME), simulations=16,
                                     max_game_plies=200)
    assert len(ep["final_scores"]) == 2 and all(v >= 0 for v in ep["final_scores"])


def test_recipe_builds_model_and_enables_kata_options():
    from training import mcts_profile
    from training.cli import validate_guidance_until
    from training.model import TokenTransformerPVNet, create_model_from_config
    cfg = _train_cfg()
    validate_guidance_until(cfg["heuristic_guidance"], TRAIN_CFG)
    assert {"playout_cap", "score_head", "opponent_policy_head", "ema"} <= set(cfg["training"])
    for name in ("selfplay", "eval"):
        assert mcts_profile.resolve_profile(GAME, name).fpu_reduction == 0.2
    meta = engine.game_metadata(GAME)
    net = create_model_from_config(cfg["network"], input_dim=meta["feature_dim"], policy_dim=meta["action_space"],
                                   num_players=meta["num_players"], token_layout=meta["token_layout"])
    assert isinstance(net, TokenTransformerPVNet)


def test_training_loop_with_the_recipe(tmp_path, caplog):
    from training.cli import find_game_config, find_heuristic_config
    from training.pipeline import run_training_loop
    caplog.set_level(logging.INFO, logger="training.pipeline")
    cfg = find_game_config(GAME)
    train_cfg = _train_cfg()
    train_cfg["heuristic"], _ = find_heuristic_config(GAME)
    train_cfg["network"] = {**train_cfg["network"], "d_model": 32, "num_layers": 1, "ffn_dim": 64}
    train_cfg["heuristic_guidance"] = {**train_cfg["heuristic_guidance"], "hold_steps": 1, "steps": 2}
    train_cfg["training"] = {**train_cfg["training"], "train_batches_per_step": 2, "inference_precision": "fp32",
                             "playout_cap": {"full_simulations": 8, "fast_simulations": 4, "full_search_prob": 0.5}}
    run_training_loop(game_id=GAME, game_config=cfg, train_config=train_cfg, output_dir=tmp_path, steps=2,
                      episodes_per_step=2, eval_every=0, eval_games=2, max_workers=1, batch_size=16,
                      learning_rate=0.001, seed=5, save_every=0)
    steps = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Step ")]
    assert len(steps) == 2 and "h_ratio=1.00" in steps[0] and "h_ratio" not in steps[1]
    assert all("s_loss=" in s and "o_loss=" in s for s in steps)
    latest = tmp_path / "models" / "model_latest.onnx"
    assert latest.exists()
    ep = engine.run_selfplay_episode(game_id=GAME, seed=3, model_path=str(latest), simulations=8, max_game_plies=200)
    assert ep["samples"]
