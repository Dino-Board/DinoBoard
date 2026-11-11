"""`eval vs heuristic` states what each side searches with.

The candidate plays the game's eval MCTS profile; the opponent is whatever
heuristic.json's `eval_vs_heuristic` selector names — for splendor an MCTS
over a frozen network with its own, smaller budget and no tail solver. The
benchmark settings are deliberately left as they are; the training log (and
scripts/eval_model.py) now print both sides once so the win rate can be read
for what it is.
"""
import json

import dinoboard_engine as engine
import pytest

from conftest import PROJECT_ROOT
from training import pipeline
from training.cli import find_game_config, find_heuristic_config, find_train_config
from training.mcts_profile import resolve_profile


def test_heuristic_settings_report_the_instantiated_search():
    named = json.loads(
        (PROJECT_ROOT / "games" / "splendor" / "config" / "heuristic.json").read_text())["named"]
    pv = engine.heuristic_settings("splendor", "pv_512x4")
    assert pv["type"] == "mcts_pv"
    assert pv["simulations"] == named["pv_512x4"]["simulations"]
    assert pv["c_puct"] == pytest.approx(named["pv_512x4"]["c_puct"])
    assert pv["leaf_batch_size"] == named["pv_512x4"]["leaf_batch_size"]
    assert pv["model_path"].endswith(named["pv_512x4"]["model_path"])
    # Knobs heuristic.json cannot set: the heuristic's search runs without them.
    assert pv["tail_solve_enabled"] is False and pv["fpu_reduction"] is None
    assert pv["opponent_selection"] == "puct" and pv["world_cap"] == 0

    assert engine.heuristic_settings("splendor", "rule_v1") == {"name": "rule_v1", "type": "rule"}
    with pytest.raises(RuntimeError, match="not registered"):
        engine.heuristic_settings("splendor", "no_such_heuristic")


def test_splendor_eval_setup_shows_both_sides():
    profile = resolve_profile("splendor", "eval")
    setup = pipeline.eval_benchmark_setup("splendor", profile, "pv_512x4", 0.0, 8)
    cand, opp = setup["candidate"], setup["heuristic_opponent"]
    assert cand["simulations"] == profile.simulations
    assert cand["tail_solve_enabled"] == profile.tail_solve_enabled
    assert cand["fpu_reduction"] == profile.fpu_reduction
    assert opp["type"] == "mcts_pv" and opp["temperature"] == 0.0
    assert opp["simulations"] == engine.heuristic_settings("splendor", "pv_512x4")["simulations"]
    assert pipeline.eval_benchmark_setup(
        "splendor", profile, None, 0.0, 1)["heuristic_opponent"] is None


def test_training_logs_eval_setup_once_and_evals_with_it(tmp_path, caplog, monkeypatch):
    """The logged candidate settings are the ones the benchmark is called with."""
    calls = []

    def fake_eval_vs_heuristic(game_id, model_path, num_games, base_seed, simulations,
                               world_cap, constrained, heuristic_temperature, max_workers,
                               opponent_selection, **kwargs):
        calls.append(dict(kwargs, simulations=simulations, world_cap=world_cap,
                          opponent_selection=opponent_selection,
                          heuristic_temperature=heuristic_temperature))
        return {"wins": 1, "losses": 1, "draws": 0, "win_rate": 0.5}

    monkeypatch.setattr(pipeline, "run_eval_vs_heuristic", fake_eval_vs_heuristic)
    monkeypatch.setattr(pipeline, "run_eval_batch", lambda *a, **k: {
        "wins": 1, "losses": 1, "draws": 0, "win_rate": 0.5})
    caplog.set_level("INFO", logger="training.pipeline")

    train_cfg, _ = find_train_config("tictactoe")
    train_cfg["heuristic"], _ = find_heuristic_config("tictactoe")
    train_cfg["network"] = {"type": "mlp", "hidden_layers": [32, 32]}
    train_cfg["training"] = {
        **train_cfg["training"], "train_batches_per_step": 1,
        "selfplay_concurrency": {
            "enabled": True, "leaf_batch_size": 2,
            "evaluator": {"batch_size": 8, "batch_timeout_us": 50, "max_queue_size": 64,
                          "intra_op_num_threads": 1, "num_inference_threads": 1}},
        "playout_cap": {"full_simulations": 16, "fast_simulations": 4, "full_search_prob": 0.5},
    }
    pipeline.run_training_loop(
        game_id="tictactoe", game_config=find_game_config("tictactoe"),
        train_config=train_cfg, output_dir=tmp_path, steps=2, episodes_per_step=4,
        eval_every=1, eval_games=2, eval_benchmarks=["heuristic_free"], max_workers=1,
        batch_size=16, learning_rate=0.001, seed=4)

    messages = [r.getMessage() for r in caplog.records]
    cand_lines = [m for m in messages if "eval candidate" in m]
    opp_lines = [m for m in messages if "eval heuristic opponent" in m]
    assert len(cand_lines) == 1 and len(opp_lines) == 1  # once, not per eval round
    profile = resolve_profile("tictactoe", "eval")
    expected = pipeline.eval_benchmark_setup("tictactoe", profile, "rule", 0.0, 2)
    assert str(expected["candidate"]) in cand_lines[0]
    assert str(expected["heuristic_opponent"]) in opp_lines[0]
    assert expected["heuristic_opponent"] == {"name": "rule", "type": "rule", "temperature": 0.0}

    assert len(calls) == 2  # one heuristic benchmark per eval round
    for call in calls:
        used = {k: call[k] for k in expected["candidate"] if k != "leaf_batch_size"}
        assert used == {k: v for k, v in expected["candidate"].items() if k != "leaf_batch_size"}
        assert call["concurrent_cfg"]["leaf_batch_size"] == expected["candidate"]["leaf_batch_size"]
        assert call["heuristic_selector"] == expected["heuristic_opponent"]["name"]
        assert call["heuristic_temperature"] == expected["heuristic_opponent"]["temperature"]
