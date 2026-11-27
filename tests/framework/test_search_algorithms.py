"""Gumbel AlphaZero search, Ataraxos-style rollout search and PPO selfplay.

NetMctsConfig::algorithm selects the search (puct / gumbel / rollout); the
selfplay `search` dict additionally carries the policy_sampling switch that
PPO trains on. training.selfplay_search / training.algorithm='ppo' drive them
from train.json.
"""
import math

import dinoboard_engine
import numpy as np
import pytest
import torch

from conftest import get_test_model
from training.cli import find_game_config, find_heuristic_config, find_train_config
from training.pipeline import run_training_loop
from training.ppo import gae_targets, magnet_coef, parse_ppo_config

GUMBEL = {"algorithm": "gumbel", "gumbel_num_considered": 4, "gumbel_c_visit": 50.0, "gumbel_c_scale": 0.1}


def _arena(game_id, seats, n_games=2, max_plies=60):
    tasks = [{"seed": 300 + i, "seat_configs": seats} for i in range(n_games)]
    return dinoboard_engine.run_concurrent_arena_batch(
        game_id, tasks, 1, max_plies, 32, 50, 256, 1, 1, [], [], False)["results"]


def _seat(game_id, **kw):
    return {"model_path": get_test_model(game_id), "simulations": 16, "temperature": 0.0, **kw}


def _selfplay(game_id, search, **kw):
    args = dict(
        game_id=game_id, seeds=[41, 42, 43], latest_model_path=get_test_model(game_id),
        pool_paths=[], num_self_threads=2, num_pool_threads=0,
        simulations=16, max_game_plies=40, tail_solve_enabled=False,
        dirichlet_alpha=0.0, dirichlet_epsilon=0.0,
        batch_size=32, max_queue_size=64, intra_op_num_threads=1, leaf_batch_size=2,
        search=search)
    args.update(kw)
    return dinoboard_engine.run_concurrent_selfplay_step(**args)["episodes"]


# ---------------------------------------------------------------------------
# Sequential Halving schedule (mctx get_sequence_of_considered_visits)
# ---------------------------------------------------------------------------

def test_considered_visit_sequence_matches_mctx():
    seq = dinoboard_engine.gumbel_considered_visit_sequence
    assert seq(4, 16) == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert seq(16, 32) == [0] * 16 + [1] * 8 + [2] * 4 + [3] * 4
    assert seq(1, 5) == [0, 1, 2, 3, 4]
    assert seq(2, 6) == [0, 0, 1, 1, 2, 2]


# ---------------------------------------------------------------------------
# Gumbel search
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game_id", ["tictactoe", "splendor"])
def test_gumbel_selfplay_targets_are_distributions_over_root_actions(game_id):
    episodes = _selfplay(game_id, GUMBEL)
    searched = 0
    for ep in episodes:
        for s in ep["samples"]:
            assert s["action_id"] in s["policy_action_ids"]
            if len(s["policy_action_ids"]) == 1:
                continue
            searched += 1
            assert s["simulations"] == 16
            assert abs(sum(s["policy_action_visits"]) - 1_000_000) <= len(s["policy_action_ids"])
            assert s["has_policy_stats"]
    assert searched > 0


def test_gumbel_selfplay_is_seed_deterministic():
    a = _selfplay("tictactoe", GUMBEL, leaf_batch_size=1, num_self_threads=1)
    b = _selfplay("tictactoe", GUMBEL, leaf_batch_size=1, num_self_threads=1)
    key = lambda eps: sorted(tuple(s["action_id"] for s in ep["samples"]) for ep in eps)
    assert key(a) == key(b)


def test_gumbel_takes_an_immediate_tictactoe_win():
    # Tic-tac-toe ends in at most 9 plies; with 64 simulations over the 4
    # considered moves Sequential Halving reaches the terminal wins even
    # with a random network, so Gumbel never loses to a 1-simulation PUCT.
    gumbel = _seat("tictactoe", algorithm="gumbel", simulations=64, gumbel_num_considered=9)
    weak = _seat("tictactoe", simulations=1)
    results = _arena("tictactoe", [gumbel, weak], n_games=4) + \
        [dict(r, winner=1 - r["winner"] if r["winner"] >= 0 else -1)
         for r in _arena("tictactoe", [weak, gumbel], n_games=4)]
    assert all(r["winner"] != 1 for r in results)


@pytest.mark.parametrize("kw, match", [
    (dict(dirichlet_alpha=0.3, dirichlet_epsilon=0.25), "Dirichlet"),
    (dict(search=dict(GUMBEL, gumbel_c_scale=0.0)), "c_scale"),
    (dict(search={"algorithm": "rollout"}), "unknown key|test-time"),
    (dict(search={"algorithm": "gumbel", "time_limit_ms": 10.0}), "unknown key"),
])
def test_gumbel_selfplay_rejects_invalid_config(kw, match):
    search = kw.pop("search", GUMBEL)
    with pytest.raises((ValueError, RuntimeError), match=match):
        _selfplay("tictactoe", search, **kw)


def test_gumbel_rejects_time_limit_in_arena():
    seat = _seat("tictactoe", algorithm="gumbel", time_limit_ms=50.0)
    with pytest.raises((ValueError, RuntimeError), match="time_limit_ms"):
        _arena("tictactoe", [seat, seat], n_games=1)


def test_gumbel_large_leaf_batch_avoids_excessive_leaf_collisions():
    # On Splendor with leaf_batch_size >> num_considered, early flush on
    # pending-leaf collisions prevents stacking hundreds of duplicate backups
    # on unexpanded depth-1 nodes.
    seat = _seat("splendor_tok", algorithm="gumbel", simulations=256,
                 gumbel_num_considered=16, leaf_batch_size=128)
    r = _arena("splendor_tok", [seat, seat], n_games=1, max_plies=2)[0]
    for ps in r["ply_stats"]:
        assert ps["simulations_done"] == 256
        assert ps["network_evaluations"] >= 180


def test_gumbel_rescale_values_flag_accepted():
    for flag in (False, True):
        seat = _seat("tictactoe", algorithm="gumbel", simulations=32,
                     gumbel_num_considered=4, gumbel_rescale_values=flag)
        r = _arena("tictactoe", [seat, seat], n_games=1)[0]
        assert r["ply_stats"]


# ---------------------------------------------------------------------------
# Rollout search
# ---------------------------------------------------------------------------

def test_rollout_evaluation_count_and_budget():
    depth, per_action = 2, 3
    seat = _seat("tictactoe", algorithm="rollout", rollout_depth=depth,
                 rollout_rollouts_per_action=per_action, leaf_batch_size=64)
    (r,) = _arena("tictactoe", [seat, seat], n_games=1)
    first = r["ply_stats"][0]
    # Empty board: 9 candidates, none ends the game within 3 plies, so every
    # rollout evaluates depth + 1 positions; + 1 for the root.
    assert first["simulations_done"] == 9 * per_action
    assert first["network_evaluations"] == 9 * per_action * (depth + 1) + 1


def test_rollout_max_candidates_limits_rollouts():
    seat = _seat("splendor", algorithm="rollout", rollout_depth=1, rollout_rollouts_per_action=2,
                 rollout_max_candidates=3, leaf_batch_size=64)
    (r,) = _arena("splendor", [seat, seat], n_games=1, max_plies=6)
    assert all(ps["simulations_done"] in (0, 6) for ps in r["ply_stats"])
    assert any(ps["simulations_done"] == 6 for ps in r["ply_stats"])


def test_rollout_time_limit_runs_waves_until_deadline():
    seat = _seat("splendor", algorithm="rollout", rollout_depth=2, rollout_rollouts_per_action=100000,
                 leaf_batch_size=32, time_limit_ms=150.0)
    (r,) = _arena("splendor", [seat, seat], n_games=1, max_plies=4)
    for ps in r["ply_stats"]:
        if ps["simulations_done"] == 0:
            continue
        assert ps["search_ms"] < 150.0 * 1.5
        assert ps["simulations_done"] < 100000


def test_rollout_hidden_info_game_runs_and_stays_legal():
    seat = _seat("loveletter", algorithm="rollout", rollout_depth=4, rollout_rollouts_per_action=4,
                 leaf_batch_size=64)
    results = _arena("loveletter", [seat, _seat("loveletter")], n_games=3, max_plies=80)
    assert all(r["total_plies"] > 0 for r in results)


# ---------------------------------------------------------------------------
# Policy-sampling selfplay (PPO)
# ---------------------------------------------------------------------------

def test_policy_sampling_records_behavior_logp_and_value():
    episodes = _selfplay("splendor", {"policy_sampling": True, "policy_sampling_temperature": 1.0})
    for ep in episodes:
        assert len(ep["samples"]) == ep["total_plies"]
        for s in ep["samples"]:
            assert s["policy_sampled"] and s["full_search"] and s["simulations"] == 0
            assert s["behavior_logp"] <= 1e-6
            if len(s["policy_action_ids"]) == 1:
                assert abs(s["behavior_logp"]) < 1e-6
            assert -1.0 <= s["behavior_value"] <= 1.0
            assert s["action_id"] in s["policy_action_ids"]
            assert len(s["features"]) > 0


def test_policy_sampling_rejects_playout_cap():
    with pytest.raises((ValueError, RuntimeError), match="policy_sampling"):
        _selfplay("splendor", {"policy_sampling": True}, fast_simulations=4, full_search_prob=0.5)


def test_gae_targets():
    adv, ret = gae_targets([0.5, 0.2], 1.0, lam=1.0)
    assert adv == pytest.approx([0.5, 0.8]) and ret == pytest.approx([1.0, 1.0])
    adv, ret = gae_targets([0.5, 0.2], 1.0, lam=0.0)
    assert adv == pytest.approx([-0.3, 0.8]) and ret == pytest.approx([0.2, 1.0])
    adv, ret = gae_targets([0.1], -1.0, lam=0.95)
    assert adv == pytest.approx([-1.1]) and ret == pytest.approx([-1.0])


def _ppo_block(**kw):
    block = {"clip": 0.2, "epochs": 2, "minibatch_size": 64, "gae_lambda": 0.95,
             "value_loss_weight": 1.0, "entropy_coef": 0.01,
             "magnet": {"policy": "init", "coef_initial": 0.1, "coef_final": 0.01},
             "policy_temperature": 1.0, "target_kl": 0.0}
    block.update(kw)
    return block


def test_magnet_coef_decays_geometrically():
    cfg = parse_ppo_config({"algorithm": "ppo", "ppo": _ppo_block()})
    assert magnet_coef(cfg, 10, 10, 20) == pytest.approx(0.1)
    assert magnet_coef(cfg, 15, 10, 20) == pytest.approx(0.1 * math.sqrt(0.1))
    assert magnet_coef(cfg, 20, 10, 20) == pytest.approx(0.01)


@pytest.mark.parametrize("cfg, match", [
    ({"algorithm": "ppo"}, "ppo"),
    ({"algorithm": "ppo", "ppo": _ppo_block(clip=1.5)}, "clip"),
    ({"algorithm": "ppo", "ppo": _ppo_block(magnet={"policy": "x", "coef_initial": 0.1, "coef_final": 0.0})}, "magnet"),
    ({"algorithm": "a2c"}, "algorithm"),
    ({"ppo": _ppo_block()}, "not 'ppo'"),
])
def test_ppo_config_rejects_invalid(cfg, match):
    with pytest.raises((ValueError, KeyError), match=match):
        parse_ppo_config(cfg)


# ---------------------------------------------------------------------------
# Pipeline: rule_v1 warm start, then Gumbel / PPO
# ---------------------------------------------------------------------------

def _train_cfg(guided_steps, **training):
    cfg, _ = find_train_config("splendor")
    cfg["heuristic"], _ = find_heuristic_config("splendor")
    cfg["network"] = {"type": "mlp", "hidden_layers": [32, 32]}
    cfg["heuristic_guidance"] = {"name": "rule_v1", "hold_steps": guided_steps, "steps": guided_steps + 1,
                                 "initial_ratio": 1.0, "temperature": 0.5}
    cfg["training"] = {
        **cfg["training"], "train_batches_per_step": 2, "training_filter_steps": 0,
        "score_head": {"loss_weight": 0.5, "scale": 15.0},
        "selfplay_concurrency": {"enabled": True, "leaf_batch_size": 2, "evaluator": {
            "batch_size": 32, "batch_timeout_us": 50, "max_queue_size": 256,
            "intra_op_num_threads": 1, "num_inference_threads": 1, "providers": []}},
        **training,
    }
    cfg["training"].pop("simulations_start", None)
    return cfg


def _run(cfg, out, steps, resume=False):
    run_training_loop(
        game_id="splendor", game_config=find_game_config("splendor"), train_config=cfg,
        output_dir=out, steps=steps, episodes_per_step=4, eval_every=0, eval_games=2,
        max_workers=2, batch_size=32, learning_rate=0.001, seed=5,
        resume_from=str(out) if resume else None)
    return torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)


def test_pipeline_gumbel_after_warm_start(tmp_path, caplog):
    caplog.set_level("INFO", logger="training.pipeline")
    cfg = _train_cfg(1, selfplay_search=GUMBEL, mcts_selfplay_overrides={
        "simulations": 8, "dirichlet_alpha": 0.0, "dirichlet_total_alpha": 0.0, "dirichlet_shaped": False, "dirichlet_epsilon": 0.0, "root_policy_temperature": {"initial": 1.0, "final": 1.0, "halflife_plies": 0.0}})
    ckpt = _run(cfg, tmp_path, 2)
    assert ckpt["step"] == 2
    assert "selfplay_search: {'algorithm': 'gumbel'" in caplog.text
    assert "Step 2/2" in caplog.text and "prior_ent=" in caplog.text


def test_pipeline_gumbel_needs_noise_off(tmp_path):
    cfg = _train_cfg(0, selfplay_search=GUMBEL, mcts_selfplay_overrides={"simulations": 8})
    with pytest.raises((ValueError, RuntimeError), match="Dirichlet"):
        _run(cfg, tmp_path, 1)


def test_pipeline_ppo_after_warm_start_and_resume(tmp_path, caplog):
    caplog.set_level("INFO", logger="training.pipeline")
    cfg = _train_cfg(1, algorithm="ppo", ppo=_ppo_block())
    ckpt = _run(cfg, tmp_path, 3)
    assert ckpt["step"] == 3 and ckpt["ppo_state"]["start_step"] == 2
    assert ckpt["ppo_state"]["magnet_state_dict"] is not None
    assert caplog.text.count("ppo: ent=") == 2
    magnet = ckpt["ppo_state"]["magnet_state_dict"]
    ckpt = _run(cfg, tmp_path, 4, resume=True)
    assert ckpt["ppo_state"]["start_step"] == 2
    assert all(torch.equal(magnet[k], ckpt["ppo_state"]["magnet_state_dict"][k]) for k in magnet)
    assert any(not torch.equal(magnet[k], ckpt["model_state_dict"][k]) for k in magnet)


def test_pipeline_ppo_with_opening_exploration(tmp_path, caplog):
    caplog.set_level("INFO", logger="training.pipeline")
    cfg = _train_cfg(1, algorithm="ppo", ppo=_ppo_block(), opening_exploration={
        "episode_ratio": 1.0, "min_plies": 2, "max_plies": 6, "temperature": 1.0, "uniform_mix": 0.1})
    ckpt = _run(cfg, tmp_path, 3)
    assert ckpt["step"] == 3 and caplog.text.count("ppo: ent=") == 2
    assert "expl_eps=4" in caplog.text


@pytest.mark.parametrize("extra, match", [
    ({"ema": {"decay": 0.99}}, "ema"),
    ({"playout_cap": {"full_simulations": 8, "fast_simulations": 2, "full_search_prob": 0.5}}, "playout_cap"),
    ({"selfplay_search": GUMBEL}, "selfplay_search"),
])
def test_pipeline_ppo_rejects_incompatible_options(tmp_path, extra, match):
    cfg = _train_cfg(1, algorithm="ppo", ppo=_ppo_block(), **extra)
    with pytest.raises(ValueError, match=match):
        _run(cfg, tmp_path, 1)


MUON = {"type": "muon", "momentum": 0.95, "ns_steps": 5}


def _transformer(cfg):
    cfg["network"] = {"type": "token_transformer", "d_model": 32, "num_layers": 2, "num_heads": 2, "ffn_dim": 64}
    return cfg


def test_pipeline_muon_trains_and_resumes(tmp_path, caplog):
    caplog.set_level("INFO", logger="training.pipeline")
    cfg = _transformer(_train_cfg(1, optimizer=MUON))
    run_training_loop(
        game_id="splendor_tok", game_config=find_game_config("splendor_tok"), train_config=cfg,
        output_dir=tmp_path, steps=2, episodes_per_step=4, eval_every=0, eval_games=2,
        max_workers=2, batch_size=32, learning_rate=0.001, seed=5)
    ckpt = torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ckpt["optimizer_state_dict"]["type"] == "muon"
    assert ckpt["optimizer_state_dict"]["muon"]["state"]
    assert "optimizer: {'type': 'muon'" in caplog.text
    run_training_loop(
        game_id="splendor_tok", game_config=find_game_config("splendor_tok"), train_config=cfg,
        output_dir=tmp_path, steps=3, episodes_per_step=4, eval_every=0, eval_games=2,
        max_workers=2, batch_size=32, learning_rate=0.001, seed=5, resume_from=str(tmp_path))
    assert torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=False)["step"] == 3
    cfg["training"]["optimizer"] = {"type": "adamw"}
    with pytest.raises(ValueError, match="Muon"):
        run_training_loop(
            game_id="splendor_tok", game_config=find_game_config("splendor_tok"), train_config=cfg,
            output_dir=tmp_path, steps=4, episodes_per_step=4, eval_every=0, eval_games=2,
            max_workers=2, batch_size=32, learning_rate=0.001, seed=5, resume_from=str(tmp_path))


@pytest.mark.parametrize("opt, match", [
    ({"type": "muon"}, "needs exactly"),
    ({"type": "adamw", "momentum": 0.9}, "Muon settings"),
    ({"type": "sgd"}, "adamw' or 'muon"),
    (dict(MUON, momentum=1.0), "momentum"),
])
def test_optimizer_config_rejects_invalid(opt, match):
    from training.pipeline import parse_optimizer_config
    with pytest.raises(ValueError, match=match):
        parse_optimizer_config({"optimizer": opt})


def test_muon_needs_transformer_blocks(tmp_path):
    with pytest.raises(ValueError, match="transformer blocks"):
        _run(_train_cfg(1, optimizer=MUON), tmp_path, 1)


def test_pipeline_ppo_needs_abrupt_warm_start_end(tmp_path):
    cfg = _train_cfg(1, algorithm="ppo", ppo=_ppo_block())
    cfg["heuristic_guidance"]["steps"] = 3
    with pytest.raises(ValueError, match="steps == hold_steps"):
        _run(cfg, tmp_path, 1)



def test_game_session_get_ai_action_gumbel_and_rollout(tmp_path):
    model = get_test_model("tictactoe")
    for search_cfg, leaf in (
        ({"algorithm": "gumbel", "gumbel_num_considered": 4, "gumbel_c_scale": 0.1}, 16),
        ({"algorithm": "rollout", "rollout_depth": 4, "rollout_rollouts_per_action": 4,
          "rollout_max_candidates": 4, "rollout_mmd_eta": 10.0}, 128),
    ):
        sess = dinoboard_engine.GameSession("tictactoe", 7, model, False)
        res = sess.get_ai_action(simulations=16, temperature=0.0, leaf_batch_size=leaf, search=search_cfg)
        assert "action" in res and 0 <= res["action"] < 9
        assert res["stats"]["simulations"] > 0


def test_pipeline_ppo_with_midgame_start(tmp_path, caplog):
    caplog.set_level("INFO", logger="training.pipeline")
    cfg = _train_cfg(1, algorithm="ppo", ppo=_ppo_block(), midgame_start={
        "episode_ratio": 1.0, "min_plies": 2, "max_plies": 4,
        "temperature": 0.7, "fork_min_choices": 2, "fork_max_choices": 3})
    ckpt = _run(cfg, tmp_path, 3)
    assert ckpt["step"] == 3 and caplog.text.count("ppo: ent=") == 2
    assert "mid_eps=4" in caplog.text
