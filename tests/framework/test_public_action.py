"""IGameRules::public_action: what observers see of an action. Identity unless a game overrides it; an
override must map to a stand-in id that is never legal, so trackers and logs cannot confuse the two.

Games whose action id names something that stays hidden must override it — the id is forwarded to every
seat's tracker and travels on the observation wire:
  * coup     — an Exchange return names the character put back into the court deck;
  * 6nimmit  — a commit names the face-down card (until the round's last commit reveals them all);
  * arknova  — a Pouch tuck names the tucked hand card."""

import random

import pytest

import dinoboard_engine as engine
from conftest import FRAMEWORK_GAMES, get_test_model


def _observed(game_id, seed, max_plies):
    gs = engine.GameSession(game_id, seed=seed)
    rng = random.Random(seed)
    out = []
    for _ in range(max_plies):
        if gs.is_terminal:
            break
        actor = gs.current_player
        legal = gs.get_legal_actions()
        action = rng.choice(legal)
        trace = gs.apply_action_with_trace(action, (actor + 1) % gs.num_players)
        out.append((action, trace["action"], legal))
    return out


@pytest.mark.parametrize("game_id", FRAMEWORK_GAMES)
def test_default_projection_is_identity(game_id):
    for action, observed, _ in _observed(game_id, 7, 200):
        assert observed == action


@pytest.mark.parametrize("game_id", ["arknova", "coup", "coup_4p", "6nimmit", "6nimmit_4p"])
def test_overridden_projection_is_never_legal(game_id):
    plies = 3000 if game_id == "arknova" else 200
    projected = [(a, o, legal) for seed in range(3) for a, o, legal in _observed(game_id, seed, plies) if o != a]
    assert projected
    action_space = engine.game_metadata(game_id)["action_space"]
    for _, observed, legal in projected:
        assert observed not in legal
        if not game_id.startswith("arknova"):
            # Stand-in sits outside the policy head: it can never be a legal action or a policy target.
            assert observed == action_space


@pytest.mark.parametrize("game_id", ["coup", "coup_3p", "coup_4p"])
def test_coup_projects_exactly_the_exchange_returns(game_id):
    seen_return = 0
    for seed in range(20):
        for action, observed, _ in _observed(game_id, seed, 150):
            if 27 <= action <= 31:
                seen_return += 1
                assert observed == 32
            else:
                assert observed == action
    assert seen_return > 0


@pytest.mark.parametrize("game_id", ["6nimmit", "6nimmit_3p", "6nimmit_4p"])
def test_6nimmit_hides_commits_until_the_round_is_complete(game_id):
    n = engine.game_metadata(game_id)["num_players"]
    hidden = shown = 0
    for seed in range(6):
        gs = engine.GameSession(game_id, seed=seed)
        rng = random.Random(seed)
        committed = 0
        while not gs.is_terminal:
            actor = gs.current_player
            action = rng.choice(gs.get_legal_actions())
            trace = gs.apply_action_with_trace(action, (actor + 1) % n)
            if action >= 104:                       # SMALL_CHOOSE_ROW is public
                assert trace["action"] == action
                continue
            committed += 1
            if committed < n:
                # Face down: neither the id nor anything else in the observation names the card.
                hidden += 1
                assert trace["action"] == 108
                assert trace["events"] == []
                pending = trace["public_snapshot"]["pending_commits"]
                assert all(pending[k + 1] != action + 1 for k in range(0, len(pending), 2))
            else:
                # Last commit of the round: every card is turned face up in this very transition.
                shown += 1
                committed = 0
                assert trace["action"] == action
    assert hidden >= 6 * 10 * (n - 1) and shown == 6 * 10


def test_selfplay_trace_carries_the_projection():
    """The observation trace of a selfplay episode (the stream every seat's tracker is fed) holds the
    projected ids; the acting seat's own sample keeps the real action."""
    model = get_test_model("6nimmit")
    ep = engine.run_selfplay_episode(game_id="6nimmit", seed=11, model_path=model, simulations=4,
                                     max_game_plies=40, trace_perspective=1)
    first_commits = 0
    for sample, step in zip(ep["samples"], ep["observation_trace"]):
        real = sample["action_id"]
        if real < 104 and sample["player"] == 0:      # seat 0 commits first in a 2p round
            first_commits += 1
            assert step["action"] == 108
        else:
            assert step["action"] == real
    assert first_commits == 10
