"""GameSession.get_heuristic_action on an observation-driven (AI API) session.

Once a session is fed by apply_initial_observation / apply_observation, its
own state is the AI view and the per-seat views built in the constructor stop
advancing (get_ai_action already searched the right one). The heuristic path
kept reading the stale constructor view and returned actions that were illegal
in the real position.

Invariant: an API session that has observed the whole game picks, on its own
seat's turns, a legal action — the very action the truth session's view of
that seat yields (same seed, same information).
"""
import json
import random

import dinoboard_engine as engine
import pytest

from conftest import PROJECT_ROOT, games_with_snapshot


def _rule_heuristic(game_id):
    path = PROJECT_ROOT / "games" / game_id / "config" / "heuristic.json"
    named = json.loads(path.read_text(encoding="utf-8"))["named"]
    return next((name for name, node in named.items() if node["type"] == "rule"), None)


CASES = [
    (g, _rule_heuristic(g)) for g in games_with_snapshot()
    if engine.game_metadata(g)["has_public_state_applier"] and _rule_heuristic(g)
]


@pytest.mark.parametrize("game_id, heuristic", CASES)
def test_heuristic_action_follows_the_observed_position(game_id, heuristic):
    calls = 0
    for seat in range(engine.game_metadata(game_id)["num_players"]):
        truth = engine.GameSession(game_id, seed=11)
        api = engine.GameSession(game_id, seed=11)
        api.apply_initial_observation(seat, truth.extract_initial_observation(seat))
        rng = random.Random(100 + seat)
        for ply in range(300):
            if truth.is_terminal:
                break
            legal = truth.get_legal_actions()
            if truth.current_player == seat:
                picked = api.get_heuristic_action(heuristic)
                assert "action" in picked, f"[{game_id}] ply {ply}: no action returned"
                assert picked["action"] in legal, (
                    f"[{game_id}] seat {seat} ply {ply}: heuristic action "
                    f"{picked['action']} is illegal in the observed position")
                assert picked["action"] == truth.get_heuristic_action(heuristic)["action"], (
                    f"[{game_id}] seat {seat} ply {ply}: differs from the truth "
                    f"session's view of the same seat")
                calls += 1
            trace = truth.apply_action_with_trace(rng.choice(legal), seat)
            api.apply_observation(
                trace["action"], events=trace["events"],
                public_snapshot=trace["public_snapshot"])
    assert calls > 0
