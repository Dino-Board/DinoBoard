"""A resolved challenge is reported as truthful exactly when it was.

When a claim (or a block) is challenged, the challenged player reveals a
card. If it is the claimed role the claim was real: the card is shuffled
back and the event extractor emits `claim_resolved_truthful` (or
`block_resolved_truthful`), which the belief tracker reads as "this player
just held that role and redrew". If it is another card the player was
bluffing and loses it; no truthful event may be emitted then.

Ground truth here is the revealed card itself, read from the full state
before the reveal action, not anything the extractor computes.
"""
from __future__ import annotations

import random

import pytest

import dinoboard_engine as engine

REVEAL_SLOT0, REVEAL_SLOT1 = 23, 24
RESOLVE_CHALLENGE_ACTION, RESOLVE_CHALLENGE_COUNTER = 2, 6


def _resolutions(game_id, games, seed):
    """(stage, revealer, truthful, emitted kinds for the revealer) per resolved challenge."""
    rng = random.Random(seed)
    out = []
    for g in range(games):
        gs = engine.GameSession(game_id, seed=seed + g, model_path="", use_filter=False)
        while not gs.is_terminal:
            d = gs.get_state_dict()
            action = rng.choice(list(gs.get_legal_actions()))
            stage = d["stage"]
            pending = None
            if stage in (RESOLVE_CHALLENGE_ACTION, RESOLVE_CHALLENGE_COUNTER) and \
                    action in (REVEAL_SLOT0, REVEAL_SLOT1):
                if stage == RESOLVE_CHALLENGE_ACTION:
                    revealer, role = d["active_player"], d["claimed_character"]
                else:
                    revealer, role = d["blocker"], d["block_character"]
                card = d["players"][revealer]["influences"][action - REVEAL_SLOT0]["character"]
                pending = (stage, revealer, card == role)
            trace = gs.apply_action_with_trace(action, 0)
            if pending is not None:
                stage, revealer, truthful = pending
                kinds = {e["kind"] for e in trace["events"]
                         if revealer in (e["payload"].get("claimer"), e["payload"].get("blocker"))}
                out.append((stage, revealer, truthful, kinds, gs.is_terminal))
    return out


@pytest.mark.parametrize("game_id", ["coup", "coup_3p", "coup_4p"])
def test_truthful_events_match_the_revealed_card(game_id):
    res = _resolutions(game_id, games=60, seed=300)
    counts = {}
    for stage, revealer, truthful, kinds, terminal in res:
        event = ("claim_resolved_truthful" if stage == RESOLVE_CHALLENGE_ACTION
                 else "block_resolved_truthful")
        if terminal:
            continue
        assert (event in kinds) == truthful, (stage, revealer, truthful, kinds)
        counts[(stage, truthful)] = counts.get((stage, truthful), 0) + 1
    for stage in (RESOLVE_CHALLENGE_ACTION, RESOLVE_CHALLENGE_COUNTER):
        for truthful in (True, False):
            assert counts.get((stage, truthful), 0) >= 3, counts
