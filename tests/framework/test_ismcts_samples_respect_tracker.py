"""ISMCTS determinization samples must respect the observer's certain knowledge.

`belief_tracker.randomize_unseen(state, observer, rng)` is the engine's
root-determinization step: each MCTS world clones the AI's view and fills
the slots hidden from the observer.

The contract is: anything the observer definitely knows MUST be honored
exactly in every sampled world; slots it does not know are free to be
sampled. If a sample contradicts a known fact, the AI is searching trees
whose roots disagree with what it has legally observed — its policy and
value estimates become noise, and it can no longer exploit revealed
information (e.g. a Guard guess after a Priest reveal).

Where the knowledge lives is per game:
  * Love Letter — on the state's viz (rules reveal a hand slot to the seat
    that peeked; the tracker is stateless). The serializer exposes the viz
    row as `players[p]["hand_seen_by"]`. (This test used to look for a
    `known_hand` entry in tracker.serialize(), which no longer exists, and
    silently asserted nothing.)

Pattern for adding a hidden-info game:
1. Identify what the observer definitely knows (tracker.serialize() fields
   and/or viz-visible slots exposed by the state serializer).
2. Identify the matching fields in the sampled world's state dict.
3. Add a checker (snap, original_state, trial_state) -> number of known
   facts verified, asserting "known implies sample == truth".
4. Add the game to _CHECKERS.
"""
from __future__ import annotations

import dinoboard_engine
import pytest

OBSERVER = 0   # test_belief_tracker samples for seat 0


def _check_loveletter(snap: dict, original: dict, trial_state: dict) -> int:
    """Known to the observer: its own hand, the drawn card while it is the
    player to move, and every opponent hand whose slot it can see (Priest
    peek / Baron tie / King swap). Returns how many opponent hands were
    known (so the caller can prove the check is not vacuous)."""
    assert snap == {}, "LL tracker became stateful: extend this checker"
    players = original["players"]
    sampled = trial_state["players"]
    known_opponent_hands = 0
    for p, pl in enumerate(players):
        if not pl["alive"]:
            continue
        if OBSERVER not in pl["hand_seen_by"]:
            continue
        assert sampled[p]["hand"] == pl["hand"], (
            f"loveletter sample contradicts the observer's knowledge: seat "
            f"{OBSERVER} has seen that player {p} holds card {pl['hand']}, "
            f"but randomize_unseen sampled hand={sampled[p]['hand']}.")
        if p != OBSERVER:
            known_opponent_hands += 1
    if original["current_player"] == OBSERVER:
        assert trial_state["drawn_card"] == original["drawn_card"]
    # Public piles are never touched.
    for p, pl in enumerate(players):
        assert sampled[p]["discards"] == pl["discards"]
    assert trial_state["face_up_removed"] == original["face_up_removed"]
    return known_opponent_hands


_CHECKERS = {
    "loveletter": _check_loveletter,
    "loveletter_3p": _check_loveletter,
}

ISMCTS_SAMPLE_GAMES = list(_CHECKERS.keys())


@pytest.mark.parametrize("game_id", ISMCTS_SAMPLE_GAMES)
def test_ismcts_samples_respect_observer_knowledge(game_id: str):
    """Drive the game forward via random play, then call randomize_unseen
    multiple times and verify every sampled world honors what the observer
    definitely knows.

    `test_belief_tracker` exposes:
      - belief_snapshot: tracker.serialize() at the test position
      - original_state: the real state dict at that position
      - trial_states[t]: the full state dict after randomize_unseen on a
        clone, for trial t
    """
    positions = 0
    known_opponent_hands = 0
    for seed in range(120):
        for plies in (1, 2, 3, 4, 5, 6):
            r = dinoboard_engine.test_belief_tracker(
                game_id, seed=seed, plies=plies, randomize_trials=10)
            original = r["original_state"]
            if original["is_terminal"]:
                continue
            positions += 1
            assert len(r["trial_states"]) == 10
            checker = _CHECKERS[game_id]
            for t, ts in enumerate(r["trial_states"]):
                try:
                    known_opponent_hands += checker(r["belief_snapshot"], original, ts)
                except AssertionError as e:
                    raise AssertionError(
                        f"seed={seed} plies={plies} trial {t}: {e}") from e
    assert positions > 200
    # Non-vacuous: the sweep must contain positions where the observer
    # really knows an opponent's card.
    assert known_opponent_hands > 100, (
        f"[{game_id}] only {known_opponent_hands} known opponent hands in the sweep")
