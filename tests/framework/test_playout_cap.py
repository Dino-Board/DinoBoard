"""Playout cap randomization (SelfplayConfig.fast_simulations / full_search_prob).

Full-search plies search `simulations` and are training samples; fast plies
search `fast_simulations` and are emitted with full_search=False (episode
stats still see every ply), and the trainer drops them.
"""
import dinoboard_engine
import pytest

from conftest import get_test_model

FULL, FAST = 48, 8


def _step(game_id, **kw):
    args = dict(
        game_id=game_id, seeds=[21, 22, 23, 24], latest_model_path=get_test_model(game_id),
        pool_paths=[], num_self_threads=2, num_pool_threads=0,
        simulations=FULL, max_game_plies=24, tail_solve_enabled=False,
        batch_size=32, max_queue_size=64, intra_op_num_threads=1, leaf_batch_size=4)
    args.update(kw)
    return dinoboard_engine.run_concurrent_selfplay_step(**args)


@pytest.mark.parametrize("game_id", ["splendor", "quoridor"])
def test_fast_plies_use_fast_budget_and_are_flagged(game_id):
    out = _step(game_id, fast_simulations=FAST, full_search_prob=0.3)
    n_full = n_fast = 0
    for ep in out["episodes"]:
        assert len(ep["samples"]) == ep["total_plies"]
        for s in ep["samples"]:
            visits = sum(s["policy_action_visits"])
            n_full += s["full_search"]
            n_fast += not s["full_search"]
            if len(s["policy_action_ids"]) == 1:  # forced move: flagged as drawn, not searched
                assert s["simulations"] == 0 and visits == 1
            else:
                assert visits == s["simulations"] == (FULL if s["full_search"] else FAST)
    assert n_full > 0 and n_fast > n_full


def test_disabled_by_default_every_ply_full():
    out = _step("splendor")
    for ep in out["episodes"]:
        for s in ep["samples"]:
            forced = len(s["policy_action_ids"]) == 1  # no search, one-hot
            assert s["full_search"] and sum(s["policy_action_visits"]) == (1 if forced else FULL)


def test_same_seeds_same_split():
    a = _step("quoridor", fast_simulations=FAST, full_search_prob=0.5)
    b = _step("quoridor", fast_simulations=FAST, full_search_prob=0.5)
    split = lambda out: sorted(
        (ep["seed"] if "seed" in ep else i, tuple(s["full_search"] for s in ep["samples"]))
        for i, ep in enumerate(out["episodes"]))
    assert split(a) == split(b)


@pytest.mark.parametrize("kw, match", [
    (dict(fast_simulations=FULL, full_search_prob=0.5), "fast_simulations"),
    (dict(fast_simulations=FAST, full_search_prob=1.0), "full_search_prob"),
])
def test_rejects_invalid_config(kw, match):
    with pytest.raises(Exception, match=match):
        _step("quoridor", **kw)


def test_pipeline_rejects_playout_cap_with_simulations_ramp(tmp_path):
    from training.cli import find_game_config, find_heuristic_config, find_train_config
    from training.pipeline import run_training_loop
    cfg = find_game_config("tictactoe")
    train_cfg, _ = find_train_config("tictactoe")
    train_cfg["heuristic"], _ = find_heuristic_config("tictactoe")
    train_cfg["training"] = {
        **train_cfg.get("training", {}),
        "simulations_start": 10,
        "playout_cap": {"full_simulations": 16, "fast_simulations": 4, "full_search_prob": 0.25},
    }
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_training_loop(
            game_id="tictactoe", game_config=cfg, train_config=train_cfg,
            output_dir=tmp_path, steps=1, episodes_per_step=2, eval_every=0,
            eval_games=2, max_workers=1, batch_size=32, learning_rate=0.001, seed=1,
        )
