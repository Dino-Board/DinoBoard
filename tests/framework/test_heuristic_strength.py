"""Heuristic strength regression: registered heuristics should
dominate a uniform-random picker.

Motivation: a registered heuristic is used for the web UI "heuristic"
difficulty and as a baseline in benchmark eval. If someone refactors
the heuristic and accidentally breaks scoring logic, it might still
produce legal actions but play no better than random — silently turning
the "heuristic" difficulty into a trivially-weak baseline. This test
catches that by running head-to-head selfplay where one side uses the
registered heuristic and the other picks uniformly.

For simple games (TicTacToe 9 moves, tree fully searchable), random vs
heuristic is close to 50/50. For LoveLetter/Coup we already trust the
registered picker. The HIGH-SIGNAL games are Azul, Splendor and
Quoridor where we have non-trivial heuristics.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import dinoboard_engine as engine


_HEURISTIC_NAME = {
    "azul": "rule_v1",
    "splendor": "rule_v1",
    "quoridor": "simple_max",
}


def _play_heuristic_vs_random(game_id: str, seed: int, heuristic_player: int) -> int:
    """Returns winner id, or -1 if draw/unfinished within 100 plies."""
    gs = engine.GameSession(game_id, seed=seed, model_path="")
    rng = random.Random(seed ^ 0xDEADBEEF)
    heuristic_name = _HEURISTIC_NAME[game_id]
    for _ in range(200):
        if gs.is_terminal:
            break
        if gs.current_player == heuristic_player:
            res = gs.get_heuristic_action(heuristic_name)
            action = res["action"]
        else:
            legal = gs.get_legal_actions()
            if not legal:
                break
            action = rng.choice(legal)
        gs.apply_action(action)
    state = gs.get_state_dict()
    return state.get("winner", -1)


@pytest.mark.parametrize("game_id,expected_winrate", [
    ("azul", 0.70),       # heuristic should win ≥70% vs random
    ("splendor", 0.70),
    ("quoridor", 0.90),   # quoridor heuristic is a real strategy, should dominate
])
def test_heuristic_beats_random(game_id, expected_winrate):
    """Heuristic_picker should beat uniform-random over many seeds.

    We run the heuristic as player 0 AND player 1 to eliminate seat bias.
    Combined win rate must exceed `expected_winrate`.
    """
    N_TRIALS = 30  # 15 games per seat
    heuristic_wins = 0
    decided = 0
    for i in range(N_TRIALS):
        seed = 1000 + i
        heuristic_player = i % 2
        winner = _play_heuristic_vs_random(game_id, seed, heuristic_player)
        if winner < 0:
            continue
        decided += 1
        if winner == heuristic_player:
            heuristic_wins += 1

    assert decided >= N_TRIALS // 2, (
        f"{game_id}: too few decided games ({decided}/{N_TRIALS}) — test unreliable"
    )
    winrate = heuristic_wins / decided
    heuristic_name = _HEURISTIC_NAME[game_id]
    assert winrate >= expected_winrate, (
        f"{game_id}: heuristic `{heuristic_name}` win rate vs uniform random = {winrate:.2f} "
        f"({heuristic_wins}/{decided}), expected ≥ {expected_winrate}. "
        f"Either the heuristic regressed or the test's expected floor is too high."
    )
