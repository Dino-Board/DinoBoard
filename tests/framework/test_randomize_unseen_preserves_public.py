"""`randomize_unseen` must not alter anything the observer can see.

Contract (engine/core/belief_tracker.h): the sampled world keeps every slot
with viz=1 for the observer — public fields and the observer's own private
fields — byte-equal; only viz=0 slots are (re)filled. The framework hash
`state_hash_for_perspective(observer)` mixes exactly the visible slots (hidden
ones contribute a constant sentinel), so it is a complete, game-agnostic
witness: the hash after sampling must equal the hash before.

A tracker that "repairs" a public size while rebuilding a hidden pile (Jaipur
used to enlarge the public bonus-pile sizes by the opponent's token count)
changes this hash. The consequence in search is a world whose public state
differs from the root's: wrong network inputs below the root and DAG nodes
that never merge with the real position.

Parametrized over every enabled game that registers a belief tracker.
Terminal positions are skipped: search never samples a terminal root, and
showdown-style reveals (Liar's Dice) may legitimately flip viz there.
"""
from __future__ import annotations

import pytest

import dinoboard_engine as engine
from conftest import games_with_tracker


@pytest.mark.parametrize("game_id", games_with_tracker())
def test_randomize_unseen_keeps_observer_hash(game_id):
    worlds = 0
    for seed in (1, 2, 3, 7, 42, 113, 555, 2024):
        for plies in (0, 4, 12, 30, 60, 110):
            r = engine.test_belief_tracker(
                game_id, seed=seed, plies=plies, randomize_trials=6)
            if r["original_state"]["is_terminal"]:
                continue
            for t, h in enumerate(r["trial_hashes"]):
                worlds += 1
                assert h == r["original_hash"], (
                    f"[{game_id}] seed={seed} plies={r['plies']} trial={t}: "
                    f"randomize_unseen changed a slot visible to the observer "
                    f"(perspective hash {r['original_hash']} -> {h})")
    assert worlds > 0, f"[{game_id}] no non-terminal position sampled"
