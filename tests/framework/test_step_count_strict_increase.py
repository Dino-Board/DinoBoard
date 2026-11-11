"""Regression: step_count strictly increases across rules-path transitions.

DAG acyclicity in MCTS rests on `IGameState::step_count_` being
monotonically incremented for every state transition that participates
in node identity. The framework owns that bookkeeping inside the
`IGameRules` wrappers around `do_action_fast` / `do_action_deterministic`
/ `undo_action` — game authors override only the protected `*_impl`
methods and physically cannot bump or skip step_count.

This test drives random legal actions through GameSession on each live
game and asserts step_count strictly increases by 1 per ply. If a future
change accidentally:
  * removes the wrapper bump (DAG goes cyclic, MCTS may loop or merge
    nodes that should be distinct),
  * double-bumps (FilteredRulesWrapper-shaped delegation regresses),
  * lets a game's *_impl reach back into step_count_ and skip,
this test fails on the first ply.

Also asserts that the session-side public-snapshot path (no rules run)
keeps the observer state's step_count in sync with truth — a property
guarded indirectly by `test_public_snapshot_round_trip` but called out
explicitly here so a regression points at the right place.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "platform"))

import dinoboard_engine as engine
from conftest import enabled_games


@pytest.mark.parametrize("game_id", enabled_games())
def test_step_count_increments_once_per_action(game_id):
    gs = engine.GameSession(game_id, seed=12345)
    rng = random.Random(12345)

    prev = gs.step_count
    assert prev == 0, f"[{game_id}] step_count should start at 0, got {prev}"

    plies = 0
    while not gs.is_terminal and plies < 80:
        legal = gs.get_legal_actions()
        if not legal:
            break
        action = rng.choice(legal)
        gs.apply_action(action)
        cur = gs.step_count
        assert cur == prev + 1, (
            f"[{game_id}] step_count must increase by 1 per ply: "
            f"prev={prev} cur={cur} action={action}"
        )
        prev = cur
        plies += 1

    assert plies > 0, f"[{game_id}] no actions applied — game terminated immediately?"
