"""§A.c.7a structural test: encoder output never contains MaskedState
placeholder sentinels.

Background (CLAUDE.md "AI Pipeline Independence" + ALGORITHM_OVERVIEW
§2.5 / I13):

  IFeatureEncoder::encode materializes a MaskedState exactly once via
  make_masked_state(state, schema, perspective). For every slot whose
  viz[..., perspective] == 0, the masked clone overwrites the value
  with kPlaceholderInt32 / kPlaceholderInt8 / kPlaceholderBool. Game
  encoders MUST branch on the placeholder value (treating it as
  "hidden") rather than emitting it as a numeric feature.

  If a per-game `encode_features` forgets to handle a hidden slot —
  i.e. it reads the slot and dumps the raw int into the feature
  vector — the output will contain INT32_MIN (~ -2.147e9) or INT8_MIN
  (-128). That's a structural leak symptom: the network sees a
  sentinel value as if it were a real game-state feature.

This test guards the encoder contract end-to-end: across all games,
no feature value falls into the placeholder-sentinel range.

Note: this test is independent of §A.a session-state work — even
before per-seat session state landed, callers passed truth state into
encoder->encode and relied on make_masked_state to produce a
placeholder-only view. So it covers the framework-wide encoder lock
(applies to all six games, not just the four in-scope §A.a games).
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


# kPlaceholderInt32 = INT32_MIN, kPlaceholderInt8 = INT8_MIN. Encoder
# features are float32; if a placeholder int leaks through unmasked, it
# arrives at the float vector as ~ -2.147e9 (Int32 case) or -128.0 (Int8
# case). Real features in this codebase live in [0, ~10] (one-hot,
# small counts, normalized scores).
PLACEHOLDER_INT32 = float(np.iinfo(np.int32).min)
PLACEHOLDER_INT8 = -128.0


from conftest import enabled_games, hidden_info_games

ALL_GAMES = enabled_games()
_HIDDEN_INFO_GAMES = hidden_info_games()


@pytest.mark.parametrize("game_id", ALL_GAMES)
def test_encoder_features_have_no_placeholder_sentinels(game_id):
    """encode_state runs the encoder on a freshly-created state. The
    feature vector must not contain a placeholder sentinel value — that
    would mean a hidden slot leaked into features unmasked.
    """
    enc = engine.encode_state(game_id, seed=42)
    feats = np.asarray(enc["features"], dtype=np.float64)
    if feats.size == 0:
        return

    # Detect Int32 placeholder leak.
    bad32 = np.where(feats <= PLACEHOLDER_INT32 + 1)[0]
    assert bad32.size == 0, (
        f"[{game_id}] features contain Int32 placeholder sentinel at "
        f"indices {bad32.tolist()[:5]} (value {feats[bad32[0]]}). "
        f"An `encode_features` code path read a hidden slot without "
        f"checking for kPlaceholderInt32. The masked-state contract "
        f"requires hidden slots to be branched on the placeholder, "
        f"not emitted as raw features.")

    # Detect Int8 placeholder leak. -128 is well outside legitimate
    # encoder output ranges (one-hots / small counts / [0,1] norms).
    bad8 = np.where(np.isclose(feats, PLACEHOLDER_INT8, atol=1e-3))[0]
    assert bad8.size == 0, (
        f"[{game_id}] features contain Int8 placeholder sentinel "
        f"(-128) at indices {bad8.tolist()[:5]}. "
        f"An `encode_features` code path read a hidden int8 slot "
        f"without checking for kPlaceholderInt8.")


@pytest.mark.parametrize("game_id", _HIDDEN_INFO_GAMES)
def test_encoder_invariant_across_selfplay_plies(game_id):
    """Stronger version: drive selfplay to generate features at many
    plies and assert no placeholder sentinel ever surfaces. Catches
    encoder paths that only fire after the game enters a state where
    a particular hidden slot becomes meaningful (e.g. opponent has
    drawn their first card, blind-reserve slot is filled, court deck
    is non-empty)."""
    model_path = get_test_model(game_id)
    ep = engine.run_selfplay_episode(
        game_id=game_id, seed=42, model_path=model_path,
        simulations=10, max_game_plies=20,
    )

    for i, sample in enumerate(ep["samples"]):
        feats = np.asarray(sample["features"], dtype=np.float64)
        if feats.size == 0:
            continue
        bad32 = np.where(feats <= PLACEHOLDER_INT32 + 1)[0]
        assert bad32.size == 0, (
            f"[{game_id}] selfplay sample {i} (ply={sample['ply']}, "
            f"player={sample['player']}) features contain Int32 "
            f"placeholder sentinel at indices {bad32.tolist()[:5]}. "
            f"A hidden slot leaked into encoder output.")
        bad8 = np.where(np.isclose(feats, PLACEHOLDER_INT8, atol=1e-3))[0]
        assert bad8.size == 0, (
            f"[{game_id}] selfplay sample {i} features contain Int8 "
            f"placeholder sentinel (-128) at indices {bad8.tolist()[:5]}.")
