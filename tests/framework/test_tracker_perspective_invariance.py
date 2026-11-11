"""Tracker perspective-invariance: per CLAUDE.md / belief_tracker.h, the
tracker's content is purely public-event-derived. Any two AI sessions
fed the same observation stream — regardless of which perspective they
were initialized for — must hold byte-equal `serialize()` at every ply.

This is what makes "clone the root tracker into each sim and feed
descent events with `perspective=root_player`" sound: tracker content
doesn't depend on perspective, so the choice is moot.

Currently real coverage = Splendor (LL serialize() is stateless and
trivially passes; Coup is temporarily unloaded — see
SIM_TRACKER_DESCENT_PLAN). When LL adds stateful tracker content or
Coup re-enters manifest, this test gains coverage with no edits.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "platform"))

import dinoboard_engine as engine
from conftest import games_with_capability, get_test_model


GAMES = games_with_capability("tracker")


@pytest.mark.parametrize("seed", [7, 42, 113, 2024, 31337])
@pytest.mark.parametrize("game_id", GAMES)
def test_tracker_serialize_is_perspective_invariant(game_id: str, seed: int):
    """Drive N parallel sessions (one per perspective) through the same
    selfplay episode. After every ply assert all N tracker.serialize()
    outputs are byte-equal.

    Each perspective gets its own observation trace because hidden-info
    games emit different `events` to different perspectives (private
    peeks, etc.). The action stream is identical across perspectives —
    we use that as the wall-clock to step all sessions in lockstep.
    """
    md = engine.game_metadata(game_id)
    n = md["num_players"]
    model_path = get_test_model(game_id)

    # 1. Harvest per-perspective observation traces.
    traces = []
    for p in range(n):
        ep = engine.run_selfplay_episode(
            game_id=game_id, seed=seed, model_path=model_path,
            simulations=8, max_game_plies=120, trace_perspective=p,
        )
        traces.append(ep)

    # 2. Build N observer sessions, each with a different seed (so any
    #    'serialize()' agreement is observation-derived, not seed-derived).
    sessions = []
    for p in range(n):
        gs = engine.GameSession(
            game_id, seed=seed + 1000 + p, model_path="", use_filter=False)
        gs.apply_initial_observation(p, traces[p]["initial_observation"])
        sessions.append(gs)

    # 3. Initial belief must already be perspective-invariant.
    init_snaps = [gs.get_belief_snapshot() for gs in sessions]
    for p in range(1, n):
        assert init_snaps[p] == init_snaps[0], (
            f"{game_id} seed={seed}: initial belief differs between "
            f"perspective 0 and {p}.\n  p0={init_snaps[0]}\n  p{p}={init_snaps[p]}")

    # 4. Step lockstep through the master action trace; after each ply
    #    every perspective's tracker must be byte-equal.
    master_trace = traces[0]["observation_trace"]
    for ply_idx, master_step in enumerate(master_trace):
        action = master_step["action"]
        for p in range(n):
            step_p = traces[p]["observation_trace"][ply_idx]
            assert step_p["action"] == action, (
                f"{game_id} seed={seed} ply={ply_idx}: action differs "
                f"between perspective traces (p0={action} vs p{p}={step_p['action']})")
            sessions[p].apply_observation(
                action,
                events=step_p["events"],
                public_snapshot=step_p.get("public_snapshot", {}),
            )
        snaps = [gs.get_belief_snapshot() for gs in sessions]
        for p in range(1, n):
            assert snaps[p] == snaps[0], (
                f"{game_id} seed={seed} ply={ply_idx}: tracker.serialize() "
                f"diverged between perspective 0 and {p}.\n"
                f"  action={action}\n  p0={snaps[0]}\n  p{p}={snaps[p]}")
