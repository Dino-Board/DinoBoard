"""The opponent's hand size is public and must survive the viewer state.

Jaipur and Maskmen pages draw the opponent's face-down hand from a hand-size
leaf of the state dict. The viewer state (`get_public_state_dict`) blanks
every leaf that depends on a slot hidden from the viewer, so that leaf has
to be computed from slots the viewer sees; otherwise the page shows an
empty hand for the opponent.
"""
from __future__ import annotations

import random

import pytest

import dinoboard_engine as engine


def _jaipur_sizes(state, seat):
    return state[f"hand_size_{seat}"]


def _maskmen_sizes(state, seat):
    return state["players"][seat]["hand_total"]


def _jaipur_truth(state, seat):
    return sum(state[f"hand_count_{seat}"])


def _maskmen_truth(state, seat):
    return sum(state["players"][seat]["hand_count"])


# Jaipur's hand_size_<p> is derived through the other seat's hand, so a
# viewer gets the opponent's size but not its own (its own hand is in
# hand_count_<viewer> anyway); only opponents are checked there.
CASES = [
    ("jaipur", 2, _jaipur_sizes, _jaipur_truth, True),
    ("maskmen_3p", 3, _maskmen_sizes, _maskmen_truth, False),
    ("maskmen_4p", 4, _maskmen_sizes, _maskmen_truth, False),
]


@pytest.mark.parametrize("game_id,n,served,truth,opponents_only", CASES,
                         ids=[c[0] for c in CASES])
def test_every_viewer_sees_every_hand_size(game_id, n, served, truth,
                                           opponents_only):
    rng = random.Random(11)
    positions = 0
    for game in range(4):
        gs = engine.GameSession(game_id, seed=100 + game, model_path="",
                                use_filter=False)
        while not gs.is_terminal and positions < 400:
            full = gs.get_state_dict()
            for viewer in range(n):
                view = gs.get_public_state_dict(viewer)
                for seat in range(n):
                    if opponents_only and seat == viewer:
                        continue
                    assert served(view, seat) == truth(full, seat), (
                        game_id, game, viewer, seat)
            positions += 1
            gs.apply_action(rng.choice(list(gs.get_legal_actions())))
    assert positions >= 100
