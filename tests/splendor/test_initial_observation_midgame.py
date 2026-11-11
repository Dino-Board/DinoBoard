"""`initial_observation` must be a complete bootstrap at ANY ply.

External clients that only see the current position (a BGA/table adapter, a
judge adapter, a replay viewer) cannot stream the full action history. They
bootstrap a fresh AI session from a mid-game `extract_initial_observation`.

The framework contract (see `engine/core/belief_tracker.h`) is that the
payload is an authoritative restore: a session bootstrapped at ply N must end
up with the same public state and the same belief content as a session that
streamed every action from ply 0. For Splendor the tracker memory is the
seen-card multiset, so `pack_init_payload` must ship `seen_cards`; deriving it
from the current tableau alone drops every card that already left the pool
(bought / earlier flips), and `randomize_unseen` then samples decks larger
than the public `deck_sizes`.
"""
import random

import pytest

import dinoboard_engine

from conftest import get_test_model

GAME = "splendor_2p"


def _fresh_session(seed, model_path=""):
    return dinoboard_engine.GameSession(GAME, seed=seed, model_path=model_path)


def _bootstrap(session, gt, seat):
    session.apply_initial_observation(seat, gt.extract_initial_observation(seat))


def _play(gt, session, action, seat):
    trace = gt.apply_action_with_trace(action, seat)
    session.apply_observation(
        trace["action"], trace["events"], trace["public_snapshot"])
    return trace


@pytest.mark.parametrize("seed", [3, 17, 42])
def test_attach_at_ply_matches_streamed_session(seed):
    seat = 0
    gt = _fresh_session(seed)
    streamed = _fresh_session(seed + 1000)
    _bootstrap(streamed, gt, seat)

    rng = random.Random(seed)
    for ply in range(40):
        if gt.is_terminal:
            break
        legal = gt.get_legal_actions()
        if not legal:
            break
        _play(gt, streamed, rng.choice(legal), seat)

        if ply < 2:
            continue

        # Bootstrap a brand-new session from the CURRENT position only.
        # Legal actions are only meaningful for the seat to move: an AI view
        # masks other seats' blind reserves, so `legal_actions` for another
        # player would read placeholders. External clients decide only for
        # their own seat, which is exactly the perspective bootstrapped here.
        decide_seat = gt.current_player
        attached = _fresh_session(seed + 2000 + ply)
        _bootstrap(attached, gt, decide_seat)

        assert attached.get_legal_actions() == gt.get_legal_actions(), (
            f"seed={seed} ply={ply}: legal actions diverged after mid-game "
            f"bootstrap")
        assert attached.get_belief_snapshot() == streamed.get_belief_snapshot(), (
            f"seed={seed} ply={ply}: tracker memory diverged after mid-game "
            f"bootstrap (expected pack_init_payload to restore it)")


@pytest.mark.parametrize("seed", [7, 23])
def test_midgame_bootstrap_can_search(seed):
    """The real regression: randomize_unseen used to raise on a deck-size
    mismatch as soon as a card had left the tableau."""
    model = get_test_model("splendor")
    seat = 0
    gt = _fresh_session(seed)
    streamed = _fresh_session(seed + 1000)
    _bootstrap(streamed, gt, seat)

    rng = random.Random(seed)
    for _ in range(25):
        if gt.is_terminal:
            break
        legal = gt.get_legal_actions()
        if not legal:
            break
        _play(gt, streamed, rng.choice(legal), seat)

    # Search from a position that has definitely had cards leave the pool,
    # bootstrapping from the current position only. A mid-game bootstrap can
    # land on the opponent's turn, so bootstrap for whoever decides.
    decide_seat = gt.current_player
    assert decide_seat is not None
    attached = _fresh_session(seed + 2000, model_path=model)
    _bootstrap(attached, gt, decide_seat)
    result = attached.get_ai_action(simulations=64, temperature=0.0)
    assert result["action"] in gt.get_legal_actions()


def test_pack_init_payload_round_trips_through_init():
    """extract_initial_observation -> apply_initial_observation must restore
    the seen set the payload describes (not silently re-derive from tableau)."""
    gt = _fresh_session(123)
    init = gt.extract_initial_observation(0)
    payload = init["tracker_init"]
    assert "seen_cards" in payload
    # Opening tableau: 4 cards per tier are public, decks hold the rest.
    assert len(payload["seen_cards"]) == 12

    peer = _fresh_session(456)
    peer.apply_initial_observation(0, init)
    assert peer.get_belief_snapshot()["seen_cards"] == sorted(
        payload["seen_cards"])
