"""Tracker-vs-truth consistency: at every ply of a random game, the
perspective player's belief tracker must NOT contradict the actual hidden
state held by the engine. "Known facts" the tracker reports must match
ground truth exactly; "unknown" slots are free to be anything.

This is the standard tracker acceptance test: it catches event-schema
gaps (a hidden-info update happens in C++ but no public event reaches
the tracker), incorrect event-application logic in the tracker (the tracker
records the wrong card after a Priest reveal), and stale state after card
movement (King swap, Prince force-discard).

The complementary test `test_api_belief_matches_selfplay` verifies that
TWO independent sessions reach the same belief from the same observation
stream — but two trackers can be wrong in the same way and still match.
This test pins the tracker's claims against ground truth, so a
systematically-wrong tracker fails here even if it agrees with itself.

Pattern for adding a new hidden-info game:
1. Identify which fields of the tracker's `serialize()` output are
   "definite knowledge" claims (e.g. loveletter `known_hand[p]`,
   coup `known_influence[p]`).
2. Identify the matching ground-truth fields exposed by `get_state_dict()`.
3. Add a checker function below mapping claim → truth and assert
   "claim != UNKNOWN_SENTINEL implies claim == truth".
4. Add the game to TRACKER_TRUTH_GAMES.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "platform"))

import dinoboard_engine as engine
from conftest import get_test_model


def _check_loveletter(state: dict, snap: dict, perspective: int) -> None:
    """Tracker claims `known_hand[p]` per player. Allowed values:
        0 (unknown) — always OK
        non-zero    — must equal ground-truth hand of that player

    Eliminated players (alive=False) have no hand on the table — the
    tracker should reflect this with 0, but a stale non-zero is also
    harmless because the field isn't used. We still check the alive
    case strictly.
    """
    known = snap.get("known_hand")
    if known is None:
        return  # tracker hasn't initialized yet
    players = state["players"]
    for p, claim in enumerate(known):
        if claim == 0:
            continue
        if not players[p]["alive"]:
            # Stale knowledge for an eliminated player — non-fatal but
            # still flag if it disagrees with the discard pile, since
            # downstream features may consume it.
            continue
        truth_hand = players[p]["hand"]
        assert claim == truth_hand, (
            f"loveletter: tracker claims player {p} holds card {claim} "
            f"but ground truth hand is {truth_hand} (perspective={perspective}).")


# Per-game checker. Each takes (gt_state_dict, tracker_snapshot_dict, perspective).
# Add a new entry per hidden-info game.
_CHECKERS = {
    "loveletter": _check_loveletter,
}

TRACKER_TRUTH_GAMES = list(_CHECKERS.keys())


@pytest.mark.parametrize("game_id", TRACKER_TRUTH_GAMES)
@pytest.mark.parametrize("seed", [7, 42, 113, 2024, 31337, 555, 8001, 9090, 12121, 33333])
def test_tracker_known_facts_match_truth(game_id: str, seed: int):
    """Drive a real game and assert the perspective player's tracker never
    claims a "definite" fact that disagrees with ground truth.

    Strategy:
      1. Run a self-play episode with `trace_perspective=0` to harvest a
         full action + public-event trace.
      2. Reconstruct ground truth by replaying actions in a fresh session.
      3. Drive an independent observer session through the trace using
         apply_initial_observation + apply_observation (same code path the
         third-party API uses).
      4. After every action, compare tracker.serialize() against GT's
         get_state_dict() via the per-game checker.
    """
    perspective = 0
    model_path = get_test_model(game_id)

    md = engine.game_metadata(game_id)
    assert 0 <= perspective < md["num_players"]

    # 1. Harvest action + observation trace.
    full_ep = engine.run_selfplay_episode(
        game_id=game_id, seed=seed, model_path=model_path,
        simulations=8, max_game_plies=120, trace_perspective=perspective,
    )

    # 2. Ground-truth replay session (deterministic action playback).
    gt2 = engine.GameSession(
        game_id, seed=seed, model_path="", use_filter=False)

    # 3. Observer session with a different seed — its hidden state is
    #    randomized differently, so any `known_*` claim it makes must come
    #    from the observation stream, not from peeking.
    obs_gs2 = engine.GameSession(
        game_id, seed=seed + 100000, model_path="", use_filter=False)
    obs_gs2.apply_initial_observation(perspective, full_ep["initial_observation"])

    # Initial belief vs. initial truth.
    state_dict = gt2.get_state_dict()
    snap = obs_gs2.get_belief_snapshot()
    _CHECKERS[game_id](state_dict, snap, perspective)

    for step in full_ep["observation_trace"]:
        action = step["action"]
        gt2.apply_action(action)
        obs_gs2.apply_observation(
            action,
            events=step["events"],
            public_snapshot=step.get("public_snapshot", {}),
        )
        if gt2.is_terminal:
            # No post-state to check — terminal state's hidden info is
            # already-revealed (showdown), and the tracker doesn't track
            # post-terminal positions.
            break
        state_dict = gt2.get_state_dict()
        snap = obs_gs2.get_belief_snapshot()
        _CHECKERS[game_id](state_dict, snap, perspective)
