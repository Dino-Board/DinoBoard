"""public_state_applier must be an exact inverse of the snapshot extractor
at the hash level.

For every game that registers a `public_state_applier`:

  truth state → extractor populates `public_snapshot` in trace step
  blank observer session + apply_initial_observation(perspective)
    + apply that snapshot via apply_public_snapshot
  → observer.state_hash_for_perspective(perspective)
    MUST equal truth.state_hash_for_perspective(perspective) for every
    perspective.

If this fails, the applier has drifted from `state_hash_for_perspective` and the
observer's public view no longer matches truth — which is the same
class of silent public-state drift as BUG-028.

Note: this test uses the direct `apply_public_snapshot` binding to
isolate the applier's correctness from `apply_observation`'s full flow.
End-to-end invocation through `apply_observation` is covered by
`test_api_belief_matches_selfplay` and `test_public_hash_excludes_internal_rng`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "platform"))

import dinoboard_engine as engine
from conftest import get_test_model


# Every hidden-info game registers a public_state_applier and has its
# extractor populate PublicEventTrace.public_snapshot. This list should
# match that set.
from conftest import games_with_snapshot
GAMES_WITH_APPLIER = games_with_snapshot()


def _truth_hashes_after_trace(game_id: str, seed: int, trace):
    """Rebuild truth state by replaying action_history directly."""
    gs = engine.GameSession(game_id, seed=seed, model_path="", use_filter=False)
    for step in trace:
        gs.apply_action(step["action"])
    num_players = gs.num_players
    return [gs.state_hash_for_perspective(p) for p in range(num_players)]


@pytest.mark.parametrize("game_id", GAMES_WITH_APPLIER)
def test_extractor_populates_public_snapshot(game_id):
    """Every trace step must carry a non-empty public_snapshot once the
    game is registered in GAMES_WITH_APPLIER."""
    model_path = get_test_model(game_id)
    ep = engine.run_selfplay_episode(
        game_id=game_id, seed=4242, model_path=model_path,
        simulations=10, max_game_plies=20, trace_perspective=0,
    )
    trace = ep.get("observation_trace") or []
    assert trace, f"[{game_id}] empty observation trace"
    for i, step in enumerate(trace):
        snap = step.get("public_snapshot")
        assert snap, (
            f"[{game_id}] trace step {i} has empty public_snapshot; "
            f"extractor is not populating PublicEventTrace.public_snapshot")


@pytest.mark.parametrize("game_id", GAMES_WITH_APPLIER)
def test_snapshot_applier_round_trip_at_each_ply(game_id):
    """For each ply N in the trace:
    - Build a fresh observer session with perspective=0
    - Replay trace[0..N-1] via apply_observation (gets to ply N's start)
    - Apply trace[N-1]'s snapshot directly via apply_public_snapshot
    - state_hash_for_perspective must match truth's hash at ply N
    """
    perspective = 0
    model_path = get_test_model(game_id)
    seed_truth = 4242
    ep = engine.run_selfplay_episode(
        game_id=game_id, seed=seed_truth, model_path=model_path,
        simulations=10, max_game_plies=20, trace_perspective=perspective,
    )
    trace = ep.get("observation_trace") or []
    assert trace, f"[{game_id}] empty observation trace"

    # Compute truth hashes at each ply by replaying action_history on a
    # fresh truth-seed session. We need ALL perspectives' hashes, but
    # snapshot is perspective-neutral (public fields only), so hash for
    # ANY perspective should match if applier restores public correctly.
    truth_snapshots = []
    truth_gs = engine.GameSession(
        game_id, seed=seed_truth, model_path="", use_filter=False)
    num_players = truth_gs.num_players
    # The trace carries what observers see of each action
    # (IGameRules::public_action — a stand-in id when the real one names a
    # hidden card); truth is replayed from the acting seats' own samples.
    real_actions = [s["action_id"] for s in ep["samples"]]
    assert len(real_actions) == len(trace)
    for action in real_actions:
        truth_gs.apply_action(action)
        truth_snapshots.append([
            truth_gs.state_hash_for_perspective(p) for p in range(num_players)
        ])

    # Build observer, replay up to each ply, then directly apply snapshot.
    # state_hash_for_perspective(p) = step_count + hash_public + hash_private(p).
    # Observer knows only ITS OWN perspective's private (from events +
    # tracker). Non-perspective players' "private" in observer is a sampled
    # value, not truth. So we can only round-trip-verify at p == perspective.
    # Round-trip of the PUBLIC hash alone is tested by comparing observer's
    # hash_for_perspective(perspective) to truth's, since step_count and
    # observer's own private are both truth-consistent.
    obs_gs = engine.GameSession(
        game_id, seed=9999, model_path="", use_filter=False)
    obs_gs.apply_initial_observation(perspective, ep["initial_observation"])
    for i, step in enumerate(trace):
        # apply_observation now auto-invokes the applier when snapshot is
        # supplied. We still do a redundant explicit apply_public_snapshot
        # to verify the applier is a correct inverse of the extractor
        # independently of apply_observation's full flow.
        obs_gs.apply_observation(
            step["action"], events=step["events"],
            public_snapshot=step["public_snapshot"])
        snap = step["public_snapshot"]
        obs_gs.apply_public_snapshot(snap, perspective)

        obs_hash = obs_gs.state_hash_for_perspective(perspective)
        assert obs_hash == truth_snapshots[i][perspective], (
            f"[{game_id}] ply {i}: observer hash_for_perspective({perspective}) "
            f"{obs_hash:#x} != truth hash {truth_snapshots[i][perspective]:#x} "
            f"after apply_public_snapshot. The applier is likely missing a "
            f"public field that state_hash_for_perspective reads.")
