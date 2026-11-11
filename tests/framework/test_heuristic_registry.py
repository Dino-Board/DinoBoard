"""Sanity-check the per-game heuristic registry + config wiring.

For every enabled game in `games/manifest.json`:
1. `games/<base>/config/heuristic.json` exists and parses.
2. Every selector's `name` is present in the `named` pool.
3. `IHeuristic::pick` returns non-empty action lists from the initial state
   for each named heuristic that doesn't require an external ONNX model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import dinoboard_engine as engine
from training.cli import find_heuristic_config, find_train_config


def _enabled_games() -> list[str]:
    manifest = json.loads((PROJECT_ROOT / "games" / "manifest.json").read_text())
    return [
        g["id"] for g in manifest["games"]
        if g.get("enabled", True)
    ]


@pytest.mark.parametrize("game_id", _enabled_games())
def test_heuristic_config_exists_and_parses(game_id):
    cfg, path = find_heuristic_config(game_id)
    assert path.exists(), f"{path} missing"
    assert cfg["named"], f"{path} 'named' is empty"
    for sel in ("eval_vs_heuristic", "web"):
        assert sel in cfg["selectors"], f"{path} missing selector {sel!r}"
        name = cfg["selectors"][sel]["name"]
        assert name in cfg["named"], (
            f"{path} selectors.{sel}.name = {name!r} not in named pool "
            f"{sorted(cfg['named'])}"
        )
    train_cfg, train_path = find_train_config(game_id)
    gs_name = train_cfg["heuristic_guidance"]["name"]
    assert gs_name in cfg["named"], (
        f"{train_path} heuristic_guidance.name = {gs_name!r} not in "
        f"{path} named pool {sorted(cfg['named'])}"
    )


@pytest.mark.parametrize("game_id", _enabled_games())
def test_default_selector_returns_legal_action(game_id):
    """The eval_vs_heuristic default heuristic must produce a legal action
    from the initial state. ONNX-backed heuristics (mcts_pv with
    model_path) run too if the file is on disk; skip only when missing.
    """
    cfg, _ = find_heuristic_config(game_id)
    # Variant id: framework tests use the canonical 2p variant exposed via
    # `engine.available_games()`; manifest ids are the base name.
    variants = [v for v in engine.available_games() if v == game_id or v.startswith(f"{game_id}_")]
    assert variants, f"no engine variant for game {game_id!r}"
    variant = variants[0]
    selector_name = cfg["selectors"]["eval_vs_heuristic"]["name"]
    named = cfg["named"][selector_name]
    if named.get("type") == "mcts_pv" and named.get("model_path"):
        model_path = Path(__file__).resolve().parents[2] / named["model_path"]
        if not model_path.exists():
            pytest.skip(
                f"{game_id}: eval selector points at {named['model_path']} "
                f"which is not on disk")
    gs = engine.GameSession(variant, seed=1, model_path="")
    result = gs.get_heuristic_action(selector_name)
    assert "action" in result
    legal = gs.get_legal_actions()
    assert result["action"] in legal, (
        f"{game_id}/{selector_name}: returned action {result['action']} "
        f"not in legal set"
    )
