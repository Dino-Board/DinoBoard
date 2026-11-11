"""Verify training config parameters actually reach the C++ engine.

Post-three-file migration (mcts.json baseline + train.json `mcts` profiles +
web.json profiles): this file covers two layers.

  - Engine-passthrough: the bindings honor the kwargs we send (tail_solve,
    heuristic guidance, training filter, simulations).
  - Resolver behavior: every (game, profile) combination resolves cleanly,
    illegal configs raise, and known-bad shapes are rejected.
"""
from __future__ import annotations

import json
from pathlib import Path

import dinoboard_engine
import pytest

from conftest import PROJECT_ROOT, enabled_games, get_test_model, load_game_config, run_short_selfplay
from training.mcts_profile import (
    clear_cache,
    max_game_plies,
    resolve_profile,
)


# ─── Engine-passthrough tests ────────────────────────────────────────────────


def test_tail_solve_disabled_produces_no_stats_quoridor():
    ep = dinoboard_engine.run_selfplay_episode(
        game_id="quoridor", seed=42, model_path=get_test_model("quoridor"), simulations=10,
        max_game_plies=200, tail_solve_enabled=False,
    )
    assert ep["tail_solve_attempts"] == 0


def test_tail_solve_enabled_has_valid_stats_quoridor():
    ep = dinoboard_engine.run_selfplay_episode(
        game_id="quoridor", seed=42, model_path=get_test_model("quoridor"), simulations=10,
        max_game_plies=200, tail_solve_enabled=True,
        tail_solve_depth_limit=3, tail_solve_node_budget=1000,
    )
    a = ep["tail_solve_attempts"]
    c = ep["tail_solve_completed"]
    s = ep["tail_solve_successes"]
    assert s <= c <= a, f"invariant violated: {s} <= {c} <= {a}"


def test_tail_solve_tiny_budget_has_more_exceeded():
    ep_tiny = dinoboard_engine.run_selfplay_episode(
        game_id="quoridor", seed=42, model_path=get_test_model("quoridor"), simulations=10,
        max_game_plies=50, tail_solve_enabled=True,
        tail_solve_depth_limit=3, tail_solve_node_budget=10,
    )
    ep_large = dinoboard_engine.run_selfplay_episode(
        game_id="quoridor", seed=42, model_path=get_test_model("quoridor"), simulations=10,
        max_game_plies=50, tail_solve_enabled=True,
        tail_solve_depth_limit=10, tail_solve_node_budget=200000,
    )
    tiny_ratio = ep_tiny["tail_solve_successes"] / max(1, ep_tiny["tail_solve_attempts"])
    large_ratio = ep_large["tail_solve_successes"] / max(1, ep_large["tail_solve_attempts"])
    assert large_ratio >= tiny_ratio


def test_heuristic_guidance_ratio_injects_heuristic_quoridor():
    ep_full = dinoboard_engine.run_selfplay_episode(
        game_id="quoridor", seed=100, model_path=get_test_model("quoridor"), simulations=10,
        max_game_plies=30, heuristic_guidance_ratio=1.0, heuristic_temperature=0.0,
        heuristic_selector="simple_max",
    )
    ep_none = dinoboard_engine.run_selfplay_episode(
        game_id="quoridor", seed=100, model_path=get_test_model("quoridor"), simulations=10,
        max_game_plies=30, heuristic_guidance_ratio=0.0,
    )
    full_visits = [sum(s["policy_action_visits"]) for s in ep_full["samples"]]
    none_visits = [sum(s["policy_action_visits"]) for s in ep_none["samples"]]
    avg_full = sum(full_visits) / max(1, len(full_visits))
    avg_none = sum(none_visits) / max(1, len(none_visits))
    assert avg_full != avg_none


def test_training_filter_ratio_restricts_policy_not_mask_quoridor():
    ep_filtered = dinoboard_engine.run_selfplay_episode(
        game_id="quoridor", seed=42, model_path=get_test_model("quoridor"), simulations=10,
        max_game_plies=30, training_filter_ratio=1.0,
    )
    ep_unfiltered = dinoboard_engine.run_selfplay_episode(
        game_id="quoridor", seed=42, model_path=get_test_model("quoridor"), simulations=10,
        max_game_plies=30, training_filter_ratio=0.0,
    )
    filtered_mask_sums = [sum(s["legal_mask"]) for s in ep_filtered["samples"]]
    unfiltered_mask_sums = [sum(s["legal_mask"]) for s in ep_unfiltered["samples"]]
    avg_filtered_mask = sum(filtered_mask_sums) / max(1, len(filtered_mask_sums))
    avg_unfiltered_mask = sum(unfiltered_mask_sums) / max(1, len(unfiltered_mask_sums))
    assert avg_filtered_mask >= avg_unfiltered_mask
    filtered_policy_counts = [len(s["policy_action_ids"]) for s in ep_filtered["samples"]]
    unfiltered_policy_counts = [len(s["policy_action_ids"]) for s in ep_unfiltered["samples"]]
    avg_filtered_policy = sum(filtered_policy_counts) / max(1, len(filtered_policy_counts))
    avg_unfiltered_policy = sum(unfiltered_policy_counts) / max(1, len(unfiltered_policy_counts))
    assert avg_filtered_policy < avg_unfiltered_policy


def test_max_game_plies_limits_episode(game_id, game_config):
    max_plies = 10
    ep = run_short_selfplay(game_id, max_game_plies=max_plies)
    assert ep["total_plies"] <= max_plies


def test_simulations_affects_visit_count():
    ep_low = dinoboard_engine.run_selfplay_episode(
        game_id="tictactoe", seed=42, model_path=get_test_model("tictactoe"), simulations=5, max_game_plies=9,
    )
    ep_high = dinoboard_engine.run_selfplay_episode(
        game_id="tictactoe", seed=42, model_path=get_test_model("tictactoe"), simulations=50, max_game_plies=9,
    )
    if not ep_low["samples"] or not ep_high["samples"]:
        pytest.skip("no samples")
    visits_low = sum(ep_low["samples"][0]["policy_action_visits"])
    visits_high = sum(ep_high["samples"][0]["policy_action_visits"])
    assert visits_high > visits_low


# ─── Resolver behavior tests ─────────────────────────────────────────────────


_PROFILE_NAMES = ["selfplay", "eval", "web_expert", "web_casual", "analysis"]


@pytest.mark.parametrize("game_id", enabled_games())
@pytest.mark.parametrize("profile_name", _PROFILE_NAMES)
def test_profile_resolves(game_id, profile_name):
    """Every (game, profile) combination must resolve to a complete MctsProfile."""
    p = resolve_profile(game_id, profile_name)
    assert p.simulations > 0
    assert p.opponent_selection in {"puct", "prior"}


@pytest.mark.parametrize("game_id", enabled_games())
def test_max_game_plies_top_level(game_id):
    """game.json must have a top-level `max_game_plies`."""
    n = max_game_plies(game_id)
    assert n > 0


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        resolve_profile("quoridor", "no_such_profile")


def test_simulations_passthrough_quoridor():
    """Profile.simulations should reach the engine via run_selfplay_episode."""
    p = resolve_profile("quoridor", "selfplay")
    ep = dinoboard_engine.run_selfplay_episode(
        game_id="quoridor", seed=1, model_path=get_test_model("quoridor"),
        simulations=p.simulations, max_game_plies=20,
    )
    if ep["samples"]:
        sample_visits = sum(ep["samples"][0]["policy_action_visits"])
        assert sample_visits > 0


_FULL_BASELINE = {
    "simulations": 100,
    "c_puct": 1.4,
    "temperature": 1.0,
    "temperature_schedule": {
        "enabled": False, "initial": 1.0, "final": 0.1, "decay_plies": 0,
    },
    "opponent_selection": "puct",
    "tail_solve_enabled": False,
    "tail_solve_depth_limit": 5,
    "tail_solve_node_budget": 1000,
    "ai_use_action_filter": False,
    "cover_root_edges": False,
}


def _write_fake_game(
    fake_root: Path,
    game_id: str,
    *,
    baseline: dict | None = None,
    train_mcts: dict | None = None,
    web_mcts: dict | None = None,
) -> None:
    """Write a minimal three-file game config for resolver tests.

    Defaults give a clean game; tests override the relevant block to inject
    the fault under test.
    """
    cfg_dir = fake_root / "games" / game_id / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "game.json").write_text(json.dumps({
        "game_id": game_id,
        "max_game_plies": 10,
    }))
    (cfg_dir / "mcts.json").write_text(json.dumps(
        baseline if baseline is not None else _FULL_BASELINE))
    (cfg_dir / "train.json").write_text(json.dumps({
        "mcts": train_mcts if train_mcts is not None else {
            "selfplay": {}, "eval": {},
        },
    }))
    (cfg_dir / "web.json").write_text(json.dumps(
        web_mcts if web_mcts is not None else {
            "web_expert": {}, "web_casual": {}, "analysis": {},
        }))


def test_unknown_field_in_baseline_raises(tmp_path, monkeypatch):
    """Typo in mcts.json baseline must surface, not silently apply BASE."""
    bad_baseline = dict(_FULL_BASELINE, tail_solve_start_ply=9)
    _write_fake_game(tmp_path, "fakegame", baseline=bad_baseline)

    from training import mcts_profile as mp
    monkeypatch.setattr(mp, "_PROJECT_ROOT", tmp_path)
    clear_cache()
    with pytest.raises(ValueError, match="unknown field"):
        mp.resolve_profile("fakegame", "selfplay")


def test_unknown_field_in_train_profile_raises(tmp_path, monkeypatch):
    """Typo in a train.json profile field name must surface, not silently
    apply baseline."""
    _write_fake_game(tmp_path, "fakegame", train_mcts={
        "selfplay": {"tail_solve_start_ply": 9},
        "eval": {},
    })

    from training import mcts_profile as mp
    monkeypatch.setattr(mp, "_PROJECT_ROOT", tmp_path)
    clear_cache()
    with pytest.raises(ValueError, match="unknown field"):
        mp.resolve_profile("fakegame", "selfplay")


def test_unknown_field_in_web_profile_raises(tmp_path, monkeypatch):
    """Typo in a web.json profile field name must surface."""
    _write_fake_game(tmp_path, "fakegame", web_mcts={
        "web_expert": {"tail_solve_start_ply": 9},
        "web_casual": {},
        "analysis": {},
    })

    from training import mcts_profile as mp
    monkeypatch.setattr(mp, "_PROJECT_ROOT", tmp_path)
    clear_cache()
    with pytest.raises(ValueError, match="unknown field"):
        mp.resolve_profile("fakegame", "web_expert")


def test_misplaced_train_profile_in_web_json_rejected(tmp_path, monkeypatch):
    """train profile names cannot live in web.json (and vice versa)."""
    _write_fake_game(tmp_path, "fakegame", web_mcts={
        "selfplay": {},
        "web_expert": {}, "web_casual": {}, "analysis": {},
    })

    from training import mcts_profile as mp
    monkeypatch.setattr(mp, "_PROJECT_ROOT", tmp_path)
    clear_cache()
    with pytest.raises(ValueError, match="don't belong in web.json"):
        mp.resolve_profile("fakegame", "web_expert")


def test_misplaced_web_profile_in_train_json_rejected(tmp_path, monkeypatch):
    _write_fake_game(tmp_path, "fakegame", train_mcts={
        "selfplay": {}, "eval": {}, "web_expert": {},
    })

    from training import mcts_profile as mp
    monkeypatch.setattr(mp, "_PROJECT_ROOT", tmp_path)
    clear_cache()
    with pytest.raises(ValueError, match="don't belong in train.json"):
        mp.resolve_profile("fakegame", "selfplay")


def test_tail_solve_enabled_without_trigger_raises(tmp_path, monkeypatch):
    """A baseline with tail_solve_enabled=true for a game without a registered
    trigger must be rejected. Tictactoe has no tail_solve_trigger, so we
    write a config that flips it on for tictactoe and expect ValueError."""
    bad_baseline = dict(_FULL_BASELINE, tail_solve_enabled=True)
    _write_fake_game(tmp_path, "tictactoe", baseline=bad_baseline)

    from training import mcts_profile as mp
    monkeypatch.setattr(mp, "_PROJECT_ROOT", tmp_path)
    clear_cache()
    with pytest.raises(ValueError, match="tail_solve"):
        mp.resolve_profile("tictactoe", "selfplay")


def test_temperature_schedule_flat_form_rejected(tmp_path, monkeypatch):
    """The flat `temperature_initial` field must be rejected — schedule
    must be the nested `temperature_schedule.{enabled,initial,final,decay_plies}`."""
    bad_baseline = dict(_FULL_BASELINE)
    bad_baseline.pop("temperature_schedule")
    bad_baseline["temperature_initial"] = 1.0
    _write_fake_game(tmp_path, "fakegame", baseline=bad_baseline)

    from training import mcts_profile as mp
    monkeypatch.setattr(mp, "_PROJECT_ROOT", tmp_path)
    clear_cache()
    with pytest.raises(ValueError):
        mp.resolve_profile("fakegame", "selfplay")


def test_unknown_profile_name_rejected(tmp_path, monkeypatch):
    """Asking the resolver for a non-canonical profile name raises before
    even touching files."""
    from training import mcts_profile as mp
    clear_cache()
    with pytest.raises(KeyError, match="unknown MCTS profile"):
        mp.resolve_profile("tictactoe", "no_such_profile")


def test_dirichlet_default_when_unspecified(tmp_path, monkeypatch):
    """Without an explicit dirichlet override anywhere, alpha/epsilon default
    to 0 (= noise off);on_first_n_plies sentinel -1 falls through to the
    game's max_game_plies (= "the whole game") so each game doesn't have to
    repeat that value in train.json."""
    _write_fake_game(tmp_path, "fakegame")

    from training import mcts_profile as mp
    monkeypatch.setattr(mp, "_PROJECT_ROOT", tmp_path)
    clear_cache()
    p = mp.resolve_profile("fakegame", "selfplay")
    assert p.dirichlet_alpha == 0.0
    assert p.dirichlet_epsilon == 0.0
    # fakegame has max_game_plies=10 (see _write_fake_game); sentinel resolved.
    assert p.dirichlet_on_first_n_plies == 10


def test_dirichlet_in_train_selfplay_only(tmp_path, monkeypatch):
    """Setting dirichlet on train.json's selfplay does not leak to eval/web."""
    _write_fake_game(tmp_path, "fakegame", train_mcts={
        "selfplay": {
            "dirichlet_alpha": 0.5,
            "dirichlet_epsilon": 0.25,
            "dirichlet_on_first_n_plies": 30,
        },
        "eval": {},
    })

    from training import mcts_profile as mp
    monkeypatch.setattr(mp, "_PROJECT_ROOT", tmp_path)
    clear_cache()
    sp = mp.resolve_profile("fakegame", "selfplay")
    eval_p = mp.resolve_profile("fakegame", "eval")
    web = mp.resolve_profile("fakegame", "web_expert")
    assert sp.dirichlet_epsilon == 0.25
    assert eval_p.dirichlet_epsilon == 0.0
    assert web.dirichlet_epsilon == 0.0
