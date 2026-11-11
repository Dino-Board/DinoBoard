"""Tests for the shared model path resolver in platform/model_paths.py."""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "platform"))

from model_paths import (  # noqa: E402
    base_game_id,
    expected_model_path,
    find_model_path,
    variant_model_name,
)


def test_base_game_id_strips_variant_suffix():
    assert base_game_id("azul") == "azul"
    assert base_game_id("azul_3p") == "azul"
    assert base_game_id("splendor_4p") == "splendor"


def test_base_game_id_keeps_unrelated_suffix():
    assert base_game_id("loveletter") == "loveletter"
    assert base_game_id("tic_tac_toe") == "tic_tac_toe"


def test_variant_model_name_default_is_2p():
    assert variant_model_name("azul") == "azul_2p"
    assert variant_model_name("splendor") == "splendor_2p"


def test_variant_model_name_keeps_explicit_variant():
    assert variant_model_name("azul_3p") == "azul_3p"
    assert variant_model_name("splendor_4p") == "splendor_4p"


def test_variant_ids_share_the_base_model_folder():
    assert base_game_id("splendor_tok") == "splendor"
    assert base_game_id("harmonies_a_3p") == "harmonies"
    assert variant_model_name("splendor_tok") == "splendor_tok_2p"
    assert variant_model_name("harmonies_a") == "harmonies_a_2p"
    p = expected_model_path("carcassonne_tok")
    assert p == _PROJECT_ROOT / "games" / "carcassonne" / "model" / "carcassonne_tok_2p.onnx"


def test_expected_model_path_canonical():
    p = expected_model_path("azul")
    assert p.name == "azul_2p.onnx"
    assert p.parent.name == "model"
    assert p.parent.parent.name == "azul"


def test_find_model_path_returns_empty_for_unknown():
    assert find_model_path("nonexistent_game_xyz") == ""


def test_find_model_path_resolves_a_real_deployed_model():
    """At least one shipped model should be on disk; verify resolver finds it."""
    candidates = ["tictactoe", "quoridor", "azul", "splendor", "loveletter", "coup"]
    found = [g for g in candidates if find_model_path(g)]
    assert found, "expected at least one deployed model.onnx — install set is broken"
    for game_id in found:
        path = find_model_path(game_id)
        assert path.endswith(".onnx")
        assert Path(path).exists()
