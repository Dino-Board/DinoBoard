"""6 nimmt!: every played card is reported once, and only once it lies face up.

The UI animates from `card_played` events. The seat that completes a round
has its card played and placed inside the same transition, so diffing the
pending slot before / after never sees it; the extractor takes that card
from the action. Expected values are recomputed here from the real game
(hands before / after), not read back from the engine's bookkeeping.
"""
from __future__ import annotations

import random

import pytest

import dinoboard_engine as engine

NUM_CARDS = 104


def _hands(sd):
    return [set(p["hand"]) for p in sd["players"]]


@pytest.mark.parametrize("game_id", ["6nimmit", "6nimmit_3p", "6nimmit_4p"])
def test_every_commit_yields_exactly_one_card_played_event(game_id):
    n = engine.game_metadata(game_id)["num_players"]
    for seed in range(6):
        gs = engine.GameSession(game_id, seed=seed, model_path="", use_filter=False)
        rng = random.Random(seed)
        committed = {p: [] for p in range(n)}      # truth: card leaving each hand
        reported = {p: [] for p in range(n)}       # card_played events
        while not gs.is_terminal:
            sd_before = gs.get_state_dict()
            actor = gs.current_player
            action = rng.choice(gs.get_legal_actions())
            trace = gs.apply_action_with_trace(action, 0)
            sd = gs.get_state_dict()
            if action < NUM_CARDS:
                committed[actor].append(action + 1)
                assert _hands(sd_before)[actor] - _hands(sd)[actor] == {action + 1}
            for ev in trace["events"]:
                assert ev["kind"] == "card_played"
                reported[ev["payload"]["player"]].append(ev["payload"]["card"])
        for p in range(n):
            assert len(committed[p]) == 10
            assert sorted(reported[p]) == sorted(committed[p]), (
                f"[{game_id}] seed={seed}: seat {p} played {sorted(committed[p])} "
                f"but card_played reported {sorted(reported[p])}")


@pytest.mark.parametrize("game_id", ["6nimmit", "6nimmit_3p"])
def test_card_played_is_emitted_only_once_the_card_is_face_up(game_id):
    """A card played face down produces no event; an event names a card only when it has been
    placed, which every seat can see: the events are the same for every observer, and the card
    is among the shown ones."""
    for seed in range(6):
        gs = engine.GameSession(game_id, seed=seed, model_path="", use_filter=False)
        rng = random.Random(seed)
        while not gs.is_terminal:
            sd_before = gs.get_state_dict()
            action = rng.choice(gs.get_legal_actions())
            leaves_round_open = (action < NUM_CARDS and
                                 sd_before["num_committed_this_round"] < sd_before["num_players"] - 1)
            trace = gs.apply_action_with_trace(action, rng.randrange(sd_before["num_players"]))
            sd = gs.get_state_dict()
            if leaves_round_open:
                assert trace["events"] == [] and trace["action"] == 108  # the hidden-commit action
            for ev in trace["events"]:
                assert ev["payload"]["card"] in sd["used"]
