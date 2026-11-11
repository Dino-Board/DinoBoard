"""KataGo-style training options, all off unless configured:

- MCTS `fpu_reduction`: unvisited edges get Q = Q(node) - r * sqrt(visited
  prior mass) instead of 0.
- `training.score_head`: auxiliary per-seat final score-margin target from
  the game's `final_score_extractor`; training-only, never exported.
- `training.opponent_policy_head`: auxiliary target = search policy of the
  next ply acted by another seat; training-only, never exported.
- `training.ema`: exported models (selfplay / eval / gating) use an
  exponential moving average of the PV weights.
"""
import dinoboard_engine
import pytest
import torch

from conftest import get_test_model
from training import mcts_profile
from training.cli import find_game_config, find_heuristic_config, find_train_config
from training.model import AuxHead, PVNet, import_pv_onnx_state_dict
from training.pipeline import (
    build_pv_optimizer,
    ema_update,
    next_opponent_indices,
    run_training_loop,
    score_margin_targets,
)


def _episode(game_id, **kw):
    return dinoboard_engine.run_selfplay_episode(
        game_id=game_id, seed=7, model_path=get_test_model(game_id),
        simulations=64, max_game_plies=30, temperature=0.0,
        dirichlet_alpha=0.0, dirichlet_epsilon=0.0, **kw)


def _visits(ep):
    return [tuple(s["policy_action_visits"]) for s in ep["samples"]]


# ---------------------------------------------------------------------------
# FPU reduction
# ---------------------------------------------------------------------------

def test_fpu_unset_is_deterministic_legacy_search():
    assert _visits(_episode("quoridor")) == _visits(_episode("quoridor", fpu_reduction=None))


def test_fpu_reduction_changes_visit_distribution():
    legacy = _visits(_episode("quoridor"))
    assert _visits(_episode("quoridor", fpu_reduction=0.5)) != legacy


def test_fpu_reduction_rejects_negative():
    with pytest.raises(ValueError, match="fpu_reduction"):
        _episode("quoridor", fpu_reduction=-0.1)


def test_profile_fpu_reduction_validated_and_resolved():
    merged = dict(mcts_profile._BASE_DICT, fpu_reduction=-1.0)
    with pytest.raises(ValueError, match="fpu_reduction"):
        mcts_profile._validate_resolved("eval", "tictactoe", merged)
    assert mcts_profile.resolve_profile("tictactoe", "eval").fpu_reduction is None
    assert mcts_profile.resolve_profile("splendor", "selfplay").fpu_reduction == 0.2


# ---------------------------------------------------------------------------
# Score head
# ---------------------------------------------------------------------------

def test_splendor_episode_reports_final_scores():
    assert dinoboard_engine.game_metadata("splendor")["has_final_score_extractor"]
    assert not dinoboard_engine.game_metadata("tictactoe")["has_final_score_extractor"]
    ep = dinoboard_engine.run_heuristic_episode(
        game_id="splendor", seed=3, max_game_plies=400, heuristic_selector="rule_v1")
    scores = ep["final_scores"]
    assert len(scores) == 2 and all(s >= 0 for s in scores)
    if ep["winner"] >= 0:
        assert scores[ep["winner"]] >= 15
    assert "final_scores" not in dinoboard_engine.run_heuristic_episode(
        game_id="tictactoe", seed=3, heuristic_selector="rule")


def test_score_margin_targets_are_seat_rotated():
    assert score_margin_targets([15.0, 9.0], 0, 2, 15.0) == [6 / 15, -6 / 15]
    assert score_margin_targets([15.0, 9.0], 1, 2, 15.0) == [-6 / 15, 6 / 15]
    assert score_margin_targets([4.0, 10.0, 7.0], 2, 3, 1.0) == [-3.0, -6.0, 3.0]


def _splendor_train_cfg(**training):
    train_cfg, _ = find_train_config("splendor")
    train_cfg["heuristic"], _ = find_heuristic_config("splendor")
    train_cfg["network"] = {"type": "mlp", "hidden_layers": [32, 32]}
    train_cfg["heuristic_guidance"] = {
        "name": "rule_v1", "hold_steps": 0, "steps": 0, "initial_ratio": 0.0, "temperature": 0.0,
    }
    train_cfg["training"] = {
        **train_cfg["training"], "simulations_start": 8, "train_batches_per_step": 2,
        "training_filter_steps": 0, "selfplay_concurrency": {"enabled": False}, **training,
    }
    return train_cfg


def _run(game_id, train_cfg, out, steps, resume=False):
    run_training_loop(
        game_id=game_id, game_config=find_game_config(game_id), train_config=train_cfg,
        output_dir=out, steps=steps, episodes_per_step=2, eval_every=0, eval_games=2,
        max_workers=1, batch_size=32, learning_rate=0.001, seed=5,
        resume_from=str(out) if resume else None)
    return torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)


SCORE = {"score_head": {"loss_weight": 0.5, "scale": 15.0}}


def test_score_head_trains_and_survives_toggling_on_resume(tmp_path, caplog):
    caplog.set_level("INFO", logger="training.pipeline")
    ckpt = _run("splendor", _splendor_train_cfg(), tmp_path, 1)
    assert "score_head_state_dict" not in ckpt
    assert len(ckpt["optimizer_state_dict"]["param_groups"]) == 1

    ckpt = _run("splendor", _splendor_train_cfg(**SCORE), tmp_path, 2, resume=True)
    assert ckpt["step"] == 2 and "score_head_state_dict" in ckpt
    assert len(ckpt["optimizer_state_dict"]["param_groups"]) == 2
    head = ckpt["score_head_state_dict"]

    ckpt = _run("splendor", _splendor_train_cfg(**SCORE), tmp_path, 3, resume=True)
    assert any(not torch.equal(head[k], ckpt["score_head_state_dict"][k]) for k in head)

    ckpt = _run("splendor", _splendor_train_cfg(), tmp_path, 4, resume=True)
    assert ckpt["step"] == 4 and "score_head_state_dict" not in ckpt
    assert len(ckpt["optimizer_state_dict"]["param_groups"]) == 1
    assert "s_loss=" in caplog.text


@pytest.mark.parametrize("bad, match", [
    ({"score_head": {"loss_weight": 0.5}}, "exactly"),
    ({"score_head": {"loss_weight": 0.0, "scale": 15.0}}, "loss_weight"),
    ({"ema": {"decay": 1.0}}, "decay"),
    ({"opponent_policy_head": {"loss_weight": -1.0}}, "loss_weight"),
    ({"opponent_policy_head": {"weight": 0.15}}, "exactly"),
])
def test_rejects_invalid_blocks(tmp_path, bad, match):
    with pytest.raises(ValueError, match=match):
        _run("splendor", _splendor_train_cfg(**bad), tmp_path, 1)


def test_score_head_requires_final_score_extractor(tmp_path):
    train_cfg, _ = find_train_config("tictactoe")
    train_cfg["heuristic"], _ = find_heuristic_config("tictactoe")
    train_cfg["training"] = {**train_cfg["training"], **SCORE}
    with pytest.raises(ValueError, match="final_score_extractor"):
        _run("tictactoe", train_cfg, tmp_path, 1)


def _group_names(opt):
    return [g["name"] for g in opt.param_groups]


def _step(opt, modules):
    loss = sum(p.sum() for m in modules for p in m.parameters())
    opt.zero_grad()
    loss.backward()
    opt.step()


def test_build_pv_optimizer_matches_groups_by_name():
    net, score, opp = PVNet(4, 3, [8]), AuxHead(8, 2), AuxHead(8, 3)
    full = build_pv_optimizer(net, {"score_head": score, "opponent_policy_head": opp}, 1e-3, 0.0, None)
    _step(full, [net, score, opp])
    pv_moment = full.state_dict()["state"][0]["exp_avg"].clone()

    only_opp = build_pv_optimizer(net, {"opponent_policy_head": opp}, 1e-3, 0.0, full.state_dict())
    assert _group_names(only_opp) == ["pv", "opponent_policy_head"]
    assert torch.equal(only_opp.state[next(net.parameters())]["exp_avg"], pv_moment)
    assert only_opp.state[next(opp.parameters())]  # moments carried over by name

    back = build_pv_optimizer(net, {"score_head": score, "opponent_policy_head": opp},
                              1e-3, 0.0, only_opp.state_dict())
    assert sorted(_group_names(back)) == ["opponent_policy_head", "pv", "score_head"]
    assert not back.state[next(score.parameters())]  # re-enabled head starts fresh


def test_build_pv_optimizer_reads_legacy_unnamed_groups():
    net, score = PVNet(4, 3, [8]), AuxHead(8, 2)
    legacy = torch.optim.AdamW(net.parameters(), lr=1e-3)
    legacy.add_param_group({"params": list(score.parameters())})
    _step(legacy, [net, score])
    opt = build_pv_optimizer(net, {"score_head": score}, 1e-3, 0.0, legacy.state_dict())
    assert _group_names(opt) == ["pv", "score_head"]
    assert opt.state[next(score.parameters())]


# ---------------------------------------------------------------------------
# Opponent policy head
# ---------------------------------------------------------------------------

def test_next_opponent_indices():
    assert next_opponent_indices([0, 1, 0, 1]) == [1, 2, 3, None]
    assert next_opponent_indices([0, 0, 1, 1, 0]) == [2, 2, 4, 4, None]
    assert next_opponent_indices([0, 1, 2]) == [1, 2, None]
    assert next_opponent_indices([]) == []


OPP = {"opponent_policy_head": {"loss_weight": 0.15}}


def test_opponent_policy_head_trains_and_toggles_on_resume(tmp_path, caplog):
    caplog.set_level("INFO", logger="training.pipeline")
    ckpt = _run("splendor", _splendor_train_cfg(**SCORE), tmp_path, 1)
    assert "opponent_policy_head_state_dict" not in ckpt

    ckpt = _run("splendor", _splendor_train_cfg(**SCORE, **OPP), tmp_path, 2, resume=True)
    names = [g["name"] for g in ckpt["optimizer_state_dict"]["param_groups"]]
    assert sorted(names) == ["opponent_policy_head", "pv", "score_head"]
    head = ckpt["opponent_policy_head_state_dict"]
    assert head["net.2.weight"].shape[0] == dinoboard_engine.game_metadata("splendor")["action_space"]

    ckpt = _run("splendor", _splendor_train_cfg(**OPP), tmp_path, 3, resume=True)
    assert "score_head_state_dict" not in ckpt
    assert any(not torch.equal(head[k], ckpt["opponent_policy_head_state_dict"][k]) for k in head)
    assert "o_loss=" in caplog.text


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

def test_ema_update():
    net, ema = PVNet(4, 3, [8]), PVNet(4, 3, [8])
    with torch.no_grad():
        for p in net.parameters():
            p.fill_(1.0)
        for p in ema.parameters():
            p.fill_(0.0)
    ema_update(ema, net, 0.75)
    assert all(torch.allclose(p, torch.full_like(p, 0.25)) for p in ema.parameters())


def test_ema_weights_are_exported(tmp_path):
    train_cfg, _ = find_train_config("tictactoe")
    train_cfg["heuristic"], _ = find_heuristic_config("tictactoe")
    train_cfg["heuristic_guidance"] = {
        "name": "rule", "hold_steps": 0, "steps": 0, "initial_ratio": 0.0, "temperature": 0.0,
    }
    train_cfg["training"] = {**train_cfg["training"], "simulations_start": 10,
                             "training_filter_steps": 0, "ema": {"decay": 0.5}}
    ckpt = _run("tictactoe", train_cfg, tmp_path, 2)
    ema_sd, raw_sd = ckpt["ema_state_dict"], ckpt["model_state_dict"]
    assert any(not torch.equal(ema_sd[k], raw_sd[k]) for k in raw_sd)
    from training.model import create_model_from_config
    meta = dinoboard_engine.game_metadata("tictactoe")
    probe = create_model_from_config(
        train_cfg["network"], input_dim=meta["feature_dim"],
        policy_dim=meta["action_space"], num_players=meta["num_players"])
    exported = import_pv_onnx_state_dict(tmp_path / "models" / "model_latest.onnx", probe)
    assert all(torch.allclose(exported[k], ema_sd[k]) for k in ema_sd)
