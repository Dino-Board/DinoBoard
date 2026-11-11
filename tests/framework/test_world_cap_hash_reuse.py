"""Regression: framework `world_cap` actually amortizes sims across worlds.

`NetMctsConfig::world_cap` controls ISMCTS's K outer determinizations × M
inner sims-per-world split. With `world_cap=0` (legacy) every sim samples
its own hidden world and shares almost no descent path with its siblings —
the tree is wide and shallow. With `world_cap > 0` the K worlds are fixed
and sims 2..M of each world replay the same `randomize_unseen` outcome,
so PUCT gets to dive into the same subtree multiple times → deeper
descents and a flood of `dag_reuse_hits` (descent steps that hit an
existing hash→node instead of creating a fresh one).

This test is the plan §3.5 acceptance gate. We compare two get_ai_action
calls on the same 6nimmit position:

  A: world_cap=0  (K=simulations, M=1, every sim its own world)
  B: world_cap=K  (K≈sqrt(sims), M≈sqrt(sims))

and assert two structural signals:

  1. B reaches strictly deeper than A. Under K=0 sims at non-deterministic
     hidden-info games rarely converge to the same hash at depth >1 since
     each sim's randomize_unseen produces a different world; with K>0 the
     M inner sims share that world and accumulate visits inside the same
     subtree, so PUCT picks deeper.
  2. B's `dag_reuse_hits` is much higher than A's. This is the direct
     measurement of node sharing: under K=0 every descent steps into a
     fresh subtree (reuse near 0); under K>0 sims 2..M of each world
     traverse already-expanded nodes from sim 1.

Note we do NOT assert `expanded_nodes` drops with K>0. By construction
every sim expands exactly one leaf node (`expand_node` runs at the
unexpanded leaf, then break), so `expanded_nodes ≈ simulations + 1` for
every K — that's the engine's invariant, not a world_cap signal. The
amortization shows up in *where* those expansions happen (deeper, under
shared parents), which is what depth + reuse capture.

6nimmit is the canonical target for this test: it is the first game whose
training Phase 1 depends on world_cap working, and its branching factor
(≤10 hand cards × ≤4 row picks) is high enough that the depth/reuse
contrast between A and B is unambiguous in 200 sims.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "platform"))

import dinoboard_engine as engine
from tests.conftest import get_test_model


SIMS = 200


def _run(game_id: str, model_path: str, seed: int, world_cap: int) -> dict:
    gs = engine.GameSession(game_id, seed=seed, model_path=model_path,
                            use_filter=False)
    result = gs.get_ai_action(simulations=SIMS, world_cap=world_cap,
                              temperature=0.0)
    return result["stats"]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_world_cap_amortizes_sims_on_6nimmit(seed):
    game_id = "6nimmit"
    if game_id not in engine.available_games():
        pytest.skip(f"{game_id} not registered in this build")

    model_path = get_test_model(game_id)

    a = _run(game_id, model_path, seed, world_cap=0)
    K = int(round(SIMS ** 0.5))  # ≈14 for 200 sims
    b = _run(game_id, model_path, seed, world_cap=K)

    a_depth = a["max_depth_reached"]
    b_depth = b["max_depth_reached"]
    a_reuse = a["dag_reuse_hits"]
    b_reuse = b["dag_reuse_hits"]

    assert b_depth >= a_depth + 2, (
        f"world_cap regression: depth not amortized "
        f"(A.world_cap=0 depth={a_depth}, B.world_cap={K} depth={b_depth}). "
        f"Expected B at least 2 deeper than A.")
    # Under K=0 each sim expands its own subtree → reuse ≈ 0.
    # Under K=14 the M=14 sims of each world walk back through the world's
    # accumulated subtree. Reuse should jump by at least 100 — far above any
    # incidental transposition match.
    assert b_reuse >= a_reuse + 100, (
        f"world_cap regression: dag_reuse not amortized "
        f"(A.world_cap=0 reuse={a_reuse}, B.world_cap={K} reuse={b_reuse}). "
        f"Expected B reuse at least 100 above A.")
    assert a["simulations"] == SIMS
    assert b["simulations"] == SIMS


def test_world_cap_zero_is_legacy_default():
    """world_cap=0 must produce K=simulations, M=1 — i.e. legacy behavior.

    Sentinel: dag_reuse_hits at world_cap=0 should be tiny (incidental
    transpositions only) and definitely far below the K-cap regime.
    """
    game_id = "6nimmit"
    if game_id not in engine.available_games():
        pytest.skip(f"{game_id} not registered in this build")

    model_path = get_test_model(game_id)
    a = _run(game_id, model_path, seed=42, world_cap=0)
    assert a["simulations"] == SIMS
    assert a["max_depth_reached"] >= 1
    # In a non-trivial hidden-info game with K=0, dag reuse should be near
    # zero (a few incidental hash collisions at most).
    assert a["dag_reuse_hits"] < 20, (
        f"world_cap=0 should be K=simulations / M=1; saw dag_reuse_hits="
        f"{a['dag_reuse_hits']} which suggests M>1 leaked through.")
