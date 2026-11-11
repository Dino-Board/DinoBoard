"""Regression: tail-solve adoption must populate stats.root_values.

When NetMcts adopts a ProvenWin from the tail solver, it short-circuits
the search and returns the proven action without running MCTS rollouts.
The early-return path used to leave `root_values` and `root_edge_values`
empty, which crashed the analysis pipeline (`_human_wr_from_stats`
indexes `root_values[human_player]`) and froze the expert-mode web
client end-to-end (the worker raised IndexError → phase=error → JS
poller waited 45s for done/idle).

Invariant: after `get_ai_action`, if `tail_solved` is True, then
`root_values` has length == num_players and is zero-sum, and
`action_values` contains the chosen action with the same vector.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "platform"))

import dinoboard_engine as engine

_AZUL_2P_MODEL = _PROJECT_ROOT / "games" / "azul" / "model" / "azul_2p.onnx"


@pytest.mark.skipif(not _AZUL_2P_MODEL.exists(), reason="azul 2p model missing")
def test_tail_solve_adoption_populates_root_values():
    # Tail-solve adoption depends on the trajectory reaching a position
    # that the bounded-budget solver actually solves. Different seeds yield
    # different trajectories; sweep a small set so this test is robust to
    # rng-path refactors that change which seed lands in a solvable shape.
    adopted = None
    actor = None
    for seed in (0xDEADBEEF, 0xBADF00D, 0x1337, 0xC0FFEE, 0x42, 0x999):
        gs = engine.GameSession("azul_2p", seed, str(_AZUL_2P_MODEL), True)
        gs.configure_tail_solve(True, 20, 1_000_000)
        while not gs.is_terminal:
            actor = gs.current_player
            res = gs.get_ai_action(50, 1.0)
            stats = res["stats"]
            if stats["tail_solved"]:
                adopted = (res, stats)
                break
            gs.apply_action(res["action"])
        if adopted is not None:
            break
    assert adopted is not None, "expected at least one tail-solve adoption in a full Azul 2p game"

    res, stats = adopted
    rv = list(stats["root_values"])
    assert len(rv) == 2, f"root_values length must equal num_players; got {rv}"
    # Zero-sum within float tolerance.
    assert abs(sum(rv)) < 1e-6, f"root_values must be zero-sum; got {rv} sum={sum(rv)}"
    # Acting player gets +1 (ProvenWin convention).
    assert rv[actor] == pytest.approx(1.0), f"actor seat {actor} should score +1; got {rv}"

    # action_values must contain the chosen action with the same vector.
    chosen = res["action"]
    av = stats["action_values"]
    assert chosen in av, f"chosen action {chosen} missing from action_values keys {list(av.keys())}"
    chosen_vec = list(av[chosen])
    assert chosen_vec == pytest.approx(rv), (
        f"action_values[{chosen}]={chosen_vec} should match root_values={rv}"
    )


if __name__ == "__main__":
    test_tail_solve_adoption_populates_root_values()
    print("tail-solve root_values regression test passed")
