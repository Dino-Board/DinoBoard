"""The training log's `mcts/s` counts simulations that were actually run.

It used to be samples/s * configured simulations. Under playout cap only the
full-search plies are samples while the fast plies (most of them) still
search, so the figure was far too low; forced and tail-solved plies were
counted as full searches although they run none.

Expected counts are recomputed independently: FULL per full-search ply, FAST
per fast ply, 0 for plies with a single legal action (found by replaying the
episode on a GameSession).
"""
import re

import dinoboard_engine as engine
import pytest

from conftest import get_test_model
from training import pipeline
from training.cli import find_game_config, find_heuristic_config, find_train_config

FULL, FAST = 16, 4
CONCURRENCY = {
    "enabled": True, "leaf_batch_size": 1,
    "evaluator": {"batch_size": 8, "batch_timeout_us": 50, "max_queue_size": 64,
                  "intra_op_num_threads": 1, "num_inference_threads": 1},
}


def _expected_simulations(game_id, episodes, budget):
    total = 0
    for ep in episodes:
        gs = engine.GameSession(game_id, seed=1)  # tictactoe: no randomness, any seed replays
        for s in ep["samples"]:
            if len(gs.get_all_legal_actions()) > 1:
                total += budget(s)
            gs.apply_action(s["action_id"])
    return total


def test_executed_simulations_matches_per_ply_budgets():
    out = engine.run_concurrent_selfplay_step(
        game_id="tictactoe", seeds=list(range(50, 70)),
        latest_model_path=get_test_model("tictactoe"), num_self_threads=2,
        simulations=FULL, fast_simulations=FAST, full_search_prob=0.25,
        max_game_plies=9, batch_size=8, max_queue_size=64, intra_op_num_threads=1)
    episodes = out["episodes"]
    expected = _expected_simulations(
        "tictactoe", episodes, lambda s: FULL if s["full_search"] else FAST)
    assert pipeline.executed_simulations(episodes) == expected
    n_full = sum(s["full_search"] for ep in episodes for s in ep["samples"])
    assert expected != n_full * FULL  # what samples * sims would have reported


def test_executed_simulations_on_the_process_pool_binding():
    out = engine.run_selfplay_episodes_batch(
        game_id="tictactoe", seeds=[5, 6, 7],
        model_paths=[get_test_model("tictactoe")] * 2, simulations=FULL, max_game_plies=9)
    episodes = out["episodes"]
    assert pipeline.executed_simulations(episodes) == _expected_simulations(
        "tictactoe", episodes, lambda _s: FULL)


def test_guided_plies_report_the_simulations_run_inside_the_heuristic():
    """Every ply guided by splendor's mcts_pv heuristic: the selfplay search
    runs nothing, the heuristic runs its heuristic.json budget on each ply that
    has a choice. A rule heuristic searches nothing."""
    import json
    from conftest import PROJECT_ROOT
    budget = json.loads((PROJECT_ROOT / "games" / "splendor" / "config" / "heuristic.json")
                        .read_text())["named"]["pv_512x4"]["simulations"]

    def guided(selector):
        return engine.run_selfplay_episode(
            game_id="splendor", seed=3, model_path=get_test_model("splendor"), simulations=FULL,
            max_game_plies=6, heuristic_guidance_ratio=1.0, heuristic_temperature=0.0,
            heuristic_selector=selector, training_filter_ratio=0.0)

    ep = guided("pv_512x4")
    gs = engine.GameSession("splendor", seed=3)
    expected = 0
    for s in ep["samples"]:
        choice = len(gs.get_all_legal_actions()) > 1
        assert s["simulations"] == 0
        assert s["heuristic_simulations"] == (budget if choice else 0)
        expected += budget if choice else 0
        gs.apply_action(s["action_id"])
    assert expected > 0
    assert pipeline.heuristic_simulations([ep]) == expected
    assert pipeline.executed_simulations([ep]) == 0
    assert pipeline.heuristic_simulations([guided("rule_v1")]) == 0


def test_training_log_reports_executed_simulations(tmp_path, caplog, monkeypatch):
    captured = []
    real_selfplay = pipeline.run_selfplay_batch

    def recording_selfplay(*args, **kwargs):
        episodes, stats = real_selfplay(*args, **kwargs)
        captured.append(episodes)
        return episodes, stats

    monkeypatch.setattr(pipeline, "run_selfplay_batch", recording_selfplay)
    caplog.set_level("INFO", logger="training.pipeline")
    train_cfg, _ = find_train_config("tictactoe")
    train_cfg["heuristic"], _ = find_heuristic_config("tictactoe")
    train_cfg["network"] = {"type": "mlp", "hidden_layers": [32, 32]}
    train_cfg["training"] = {
        **train_cfg["training"], "train_batches_per_step": 1,
        "selfplay_concurrency": CONCURRENCY,
        "playout_cap": {"full_simulations": FULL, "fast_simulations": FAST,
                        "full_search_prob": 0.25},
    }
    pipeline.run_training_loop(
        game_id="tictactoe", game_config=find_game_config("tictactoe"),
        train_config=train_cfg, output_dir=tmp_path, steps=1, episodes_per_step=60,
        eval_every=0, eval_games=2, max_workers=2, batch_size=32, learning_rate=0.001, seed=3)

    (episodes,) = captured
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Step 1/1:")]
    assert len(lines) == 1
    # Same wording as before; scripts/plot_training_metrics.py keys on "Step N/M:".
    m = re.search(r"(\d+)smp/s, (\d+)mcts/s", lines[0])
    assert m, lines[0]
    sps, mps = int(m.group(1)), int(m.group(2))
    samples = int(re.search(r"samples=(\d+)", lines[0]).group(1))
    assert samples == sum(s["full_search"] for ep in episodes for s in ep["samples"])
    expected = _expected_simulations(
        "tictactoe", episodes, lambda s: FULL if s["full_search"] else FAST)
    # Both rates share the selfplay wall-clock, so their ratio is exact up to
    # the rounding of the two printed integers.
    assert mps / sps == pytest.approx(expected / samples, rel=0.1)
    assert expected / samples > 1.3 * FULL  # samples * sims is off by more than the tolerance
