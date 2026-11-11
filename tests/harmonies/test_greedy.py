"""Harmonies greedy heuristic: terminates and beats random."""
from __future__ import annotations

import random

import dinoboard_engine as engine
import pytest

GAME = "harmonies"


def _play_one(seed: int, p0: str, p1: str) -> tuple[int, int]:
    """p0/p1 in {'greedy','random'}. Returns (winner, plies)."""
    gs = engine.GameSession(GAME, seed=seed, model_path="")
    rng = random.Random(seed ^ 0xABCDEF)
    for ply in range(3000):
        if gs.is_terminal:
            state = gs.get_state_dict()
            return state["winner"], ply
        side = p0 if gs.current_player == 0 else p1
        if side == "greedy":
            res = gs.get_heuristic_action("greedy")
            action = res["action"]
        else:
            legal = gs.get_legal_actions()
            assert legal, "no legal actions but not terminal"
            action = rng.choice(legal)
        gs.apply_action(action)
    pytest.fail(f"game did not terminate within 3000 plies (seed={seed})")


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_greedy_vs_greedy_terminates(seed):
    winner, plies = _play_one(seed, "greedy", "greedy")
    assert plies > 0
    assert winner in (-1, 0, 1)


def test_greedy_beats_random():
    """Greedy heuristic should beat uniform random; play both seats."""
    seeds = list(range(20))
    wins = 0
    games = 0
    for s in seeds:
        # Greedy as P0
        w, _ = _play_one(s, "greedy", "random")
        games += 1
        if w == 0:
            wins += 1
        # Greedy as P1
        w, _ = _play_one(s ^ 0x5A5A, "random", "greedy")
        games += 1
        if w == 1:
            wins += 1
    winrate = wins / games
    print(f"\ngreedy vs random: {wins}/{games} = {winrate:.2%}")
    assert winrate >= 0.70, f"greedy winrate {winrate:.2%} too low vs random"
