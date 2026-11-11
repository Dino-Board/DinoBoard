"""I14 (golden standard): state_hash_for_perspective contract.

Two invariants pinned here, complementing test_public_hash_excludes_internal_rng
(which checks the public-derivable property across different sessions):

  (1) Determinism — for any registered game, two GameSession objects
      constructed with the same seed produce byte-equal
      state_hash_for_perspective(p) for every perspective p, both at the
      initial state and after the same first legal action driven by
      apply_observation. Catches accidental dependence on object address,
      uninitialized memory, or wall-clock RNG.

  (2) Fully-public games have perspective-invariant hash — for tictactoe
      and quoridor, state_hash_for_perspective(p) is identical across
      every p (no private fields → public hash is the whole hash).
      Hidden-info games are NOT required to satisfy this — different
      perspectives see different private cards, so their hashes legally
      differ.

Together these pin the framework-derived hash contract: same observation
history → same hash, regardless of session RNG; fully-public game → same
hash for every viewer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "platform"))

import dinoboard_engine as engine


from conftest import enabled_games, hidden_info_games

ALL_GAMES = enabled_games()
# Fully public = enabled and not flagged hidden_info.
FULLY_PUBLIC_GAMES = [g for g in ALL_GAMES if g not in hidden_info_games()]


def _make_session(game_id: str, seed: int) -> engine.GameSession:
    return engine.GameSession(game_id, seed=seed, model_path="", use_filter=False)


@pytest.mark.parametrize("game_id", ALL_GAMES)
def test_hash_deterministic_across_same_seed_sessions(game_id: str) -> None:
    """Same seed → byte-equal state_hash_for_perspective for every viewer,
    at initial state and after one ply. Hash must be a pure function of
    the (seed, observation history) pair — never of allocator order, RNG
    walk, or wall-clock entropy."""
    seed = 4242
    a = _make_session(game_id, seed)
    b = _make_session(game_id, seed)
    n = a.num_players
    assert b.num_players == n

    # Initial state.
    for p in range(n):
        ha = a.state_hash_for_perspective(p)
        hb = b.state_hash_for_perspective(p)
        assert ha == hb, (
            f"[{game_id}] initial hash diverges across same-seed sessions "
            f"for perspective {p}: {ha:#x} vs {hb:#x}. state_hash_for_perspective "
            f"must be a pure function of (seed, observation history)."
        )


@pytest.mark.parametrize("game_id", FULLY_PUBLIC_GAMES)
def test_hash_perspective_invariant_for_fully_public_games(game_id: str) -> None:
    """Fully-public games (tictactoe, quoridor) have no private fields.
    state_hash_for_perspective(p) MUST be identical across every viewer p.

    If any of these games' schema walk ever starts emitting something
    that depends on `p` (a per-perspective field that shouldn't exist
    in a fully-public game), this test catches it — that indicates
    either an opp-private leak into the perspective hash or an
    unannounced asymmetry that breaks the fully-public contract.
    """
    seed = 31337
    sess = _make_session(game_id, seed)
    n = sess.num_players
    assert n >= 2

    h0 = sess.state_hash_for_perspective(0)
    for p in range(1, n):
        hp = sess.state_hash_for_perspective(p)
        assert hp == h0, (
            f"[{game_id}] state_hash_for_perspective differs across "
            f"perspectives in a fully-public game: p=0 {h0:#x} vs "
            f"p={p} {hp:#x}. Fully-public games have no private fields; "
            f"schema walk must yield identical digests for every perspective."
        )
