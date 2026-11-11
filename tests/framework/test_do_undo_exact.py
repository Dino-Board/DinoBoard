"""Tail-solver contract: `do_action_deterministic` + `undo_action` restores
the state exactly.

`AlphaBetaTailSolver` searches on ONE state object and relies on undo to walk
back up. If an undo leaves anything behind — another player's board, a
refilled supply, a score — every sibling move after it is evaluated on a
corrupted position and the solver can report a "proven win" that is not one
(this happened in Azul: a round-ending pick was undone with the partial
record meant for ordinary picks).

For every enabled game that registers a tail solver, at every ply of random
playouts, `GameSession._debug_do_undo_roundtrip` runs, for EVERY legal
action, do + (a few nested do/undo) + undo on a clone of the truth state.
The clone must then equal an untouched clone in: the full state dict, the
legacy full hash (the solver's transposition key), every perspective hash,
and step_count.

`test_do_undo_consistency.py` only checks this indirectly (search
determinism); this file checks the state itself.
"""
from __future__ import annotations

import random

import pytest

import dinoboard_engine as engine
from conftest import enabled_games


def _tail_solver_games():
    return [g for g in enabled_games() if engine.game_metadata(g)["has_tail_solver"]]


@pytest.mark.parametrize("game_id", _tail_solver_games())
def test_do_deterministic_then_undo_restores_everything(game_id):
    rng = random.Random(2024)
    checked = 0
    for seed in range(4):
        gs = engine.GameSession(game_id, seed=seed)
        plies = 0
        while not gs.is_terminal and plies < 400:
            legal = gs.get_legal_actions()
            if not legal:
                break
            r = gs._debug_do_undo_roundtrip(nested=2)
            base = r["baseline"]
            for rt in r["roundtrips"]:
                checked += 1
                for key in ("state", "state_hash", "perspective_hashes", "step_count"):
                    assert rt[key] == base[key], (
                        f"[{game_id}] seed={seed} ply={plies}: "
                        f"do_action_deterministic + undo_action of action "
                        f"{rt['action']} did not restore '{key}'")
            gs.apply_action(rng.choice(legal))
            plies += 1
    assert checked > 0
