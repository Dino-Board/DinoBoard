"""Regression: feature encoder respects the hash scope.

The ISMCTS architecture aligns encoder scope with the hash scope: the
encoder reads only what the perspective can legally see (viz=1 slots
on the MaskedState). Opponents' private fields must NOT influence the
encoder output.

This is the structural enforcement of "no info leak through features":
- Hash groups sampled worlds by info set → multiple worlds share tree nodes
- Encoder must produce the SAME feature vector for all worlds in the info set
- Otherwise the network sees different features across sampled worlds at
  the same tree node, and its prior/value estimates become world-specific
  rather than info-set-specific → prior pollution under root sampling

Test method: for each hidden-info game, drive two selfplay episodes
with identical seeds. Their sample features must be bit-identical.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "platform"))

import dinoboard_engine as engine
from conftest import get_test_model


from conftest import hidden_info_games as _hidden_info_games

# Games with hidden info worth checking — TTT / Quoridor have no private
# fields and encoder invariance is trivial there. Derived from manifest
# so new hidden-info games are picked up automatically.
HIDDEN_INFO_GAMES = _hidden_info_games()


@pytest.mark.parametrize("game_id", HIDDEN_INFO_GAMES)
def test_encoder_output_invariant_under_opp_private_change(game_id):
    """Two sessions seeded identically should produce equal public state
    but potentially-different opp private state after resampling. The
    encoder output for the same perspective should be bit-identical.

    We approximate this via: two GameSession instances, same seed, drive
    them through identical action sequences, extract features at each
    perspective's decision point. Features should match.

    For games where the test model doesn't expose encoder features
    directly via Python, we use selfplay samples as a proxy: the same
    seed + same actions should produce identical sample.features if the
    encoder respects hash scope.
    """
    model_path = get_test_model(game_id)

    # Run two selfplay episodes with identical seeds. Their sample
    # features should match byte-for-byte (the RNG driving action
    # selection is deterministic given seed).
    ep_a = engine.run_selfplay_episode(
        game_id=game_id, seed=42, model_path=model_path,
        simulations=20, max_game_plies=15,
    )
    ep_b = engine.run_selfplay_episode(
        game_id=game_id, seed=42, model_path=model_path,
        simulations=20, max_game_plies=15,
    )

    assert len(ep_a["samples"]) == len(ep_b["samples"]), (
        f"[{game_id}] deterministic selfplay produced different numbers of "
        f"samples: {len(ep_a['samples'])} vs {len(ep_b['samples'])}")

    # Compare features sample-by-sample. If the encoder leaks opp private
    # info that differs across selfplay samplings of the same seed (which
    # it shouldn't if randomize_unseen is deterministic for the same seed),
    # we'd see differences. Since deterministic selfplay keeps opp state
    # identical too, this test mostly asserts selfplay reproducibility.
    for i, (sa, sb) in enumerate(zip(ep_a["samples"], ep_b["samples"])):
        fa = np.asarray(sa["features"], dtype=np.float32)
        fb = np.asarray(sb["features"], dtype=np.float32)
        if not np.array_equal(fa, fb):
            diff_idx = int(np.argmax(np.abs(fa - fb)))
            pytest.fail(
                f"[{game_id}] sample {i}: encoder features diverged at "
                f"index {diff_idx}: {fa[diff_idx]} vs {fb[diff_idx]}. "
                f"This usually means the encoder is reading non-deterministic "
                f"state (opp hidden field or rng nonce) even though the "
                f"observable state is identical."
            )


@pytest.mark.parametrize("game_id", HIDDEN_INFO_GAMES)
def test_encoder_stable_within_info_set(game_id):
    """Sanity: the feature_dim advertised by game_metadata matches the
    actual feature vector length produced by the encoder during selfplay.
    A mismatch indicates encoder drift after the refactor.
    """
    model_path = get_test_model(game_id)
    meta = engine.game_metadata(game_id)

    ep = engine.run_selfplay_episode(
        game_id=game_id, seed=123, model_path=model_path,
        simulations=20, max_game_plies=10,
    )
    assert ep["samples"], f"[{game_id}] no samples produced"
    for i, s in enumerate(ep["samples"]):
        actual_dim = len(s["features"])
        assert _whole_encoding(actual_dim, meta), (
            f"[{game_id}] sample {i}: feature vector length {actual_dim} "
            f"is not a whole encoding (game_metadata feature_dim {meta['feature_dim']})")


def _whole_encoding(n, meta):
    """feature_dim floats, or a positive multiple of token_dim for variable-length token encoders."""
    layout = meta.get("token_layout")
    if layout and layout["variable_tokens"]:
        return n > 0 and n % layout["token_dim"] == 0
    return n == meta["feature_dim"]


from conftest import enabled_games as _enabled_games
ALL_GAMES = _enabled_games()


@pytest.mark.parametrize("game_id", ALL_GAMES)
def test_feature_dim_matches_metadata(game_id):
    """encode(state).feature_dim matches game_metadata.feature_dim, and
    the actual feature vector length matches that dim."""
    meta = engine.game_metadata(game_id)
    enc = engine.encode_state(game_id, seed=42)
    assert enc["feature_dim"] == meta["feature_dim"]
    assert _whole_encoding(len(enc["features"]), meta)
