"""Sim-local tracker is maintained during MCTS descent.

Per SIM_TRACKER_DESCENT_PLAN: each MCTS sim clones the session-shared
tracker at sim entry, calls randomize_unseen, and during descent feeds
`observe_public_event` after every `do_action_fast` so deep nodes'
encoder reads see up-to-date public-derived state. This test asserts
that maintenance actually happens — i.e. the cloned sim_tracker
diverges from the session-shared root tracker as descent progresses.

Coverage = games_with_capability("tracker") MINUS LL (LL serialize() is
stateless so this test cannot detect descent-time changes there; LL's
maintenance is exercised indirectly via test_api_belief_matches_selfplay
and the sim's encoder reading the cloned tracker). Currently resolves
to Splendor only; Coup re-entering the manifest will gain coverage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "platform"))

import dinoboard_engine as engine
from conftest import games_with_capability, get_test_model


# Splendor's serialize() exposes seen_cards, which CHANGES on every deck
# flip during descent — perfect signal for "is sim_tracker actually
# being updated?". LL's serialize() is currently stateless and would
# pass trivially without proving anything; exclude until LL gains
# stateful tracker content. Same applies to 6nimmit — its tracker is
# stateless by design (all "belief" lives on state.bullhead_pile / table
# / committed; tracker.observe_public_event is a no-op). Jaipur's tracker
# is stateFUL: it maintains a per-player net-inflow bookkeeping (sold[],
# net_in[]) updated by observe_public_event on every goods-flow event, and
# serialize() exposes it — so this descent test applies. Ark Nova's tracker derives every unseen
# pool from public state and the observer's visible slots, so it holds nothing either.
_STATELESS_TRACKER_GAMES = {"loveletter", "6nimmit", "harmonies", "arknova"}
_STATEFUL_TRACKER_GAMES = [
    g for g in games_with_capability("tracker")
    if g not in _STATELESS_TRACKER_GAMES
]


@pytest.mark.parametrize("game_id", _STATEFUL_TRACKER_GAMES)
def test_sim_tracker_changes_during_descent(game_id: str):
    """Run MCTS with the on_sim_step debug callback active. Across all
    sims and all descent steps, at least some recorded sim_tracker
    snapshot must differ from the session-shared root snapshot — proving
    the cloned tracker is being fed events during descent rather than
    frozen at the root state.
    """
    md = engine.game_metadata(game_id)
    model_path = get_test_model(game_id)

    gs = engine.GameSession(
        game_id, seed=42, model_path=model_path, use_filter=False)

    # 64 sims × however many descent steps each takes — enough budget
    # for at least one sim to walk past descent step 0 in any tracker
    # game. A single descent step on Splendor flips a deck card →
    # seen_cards changes immediately → divergence on step 0 of sim 0.
    out = gs._debug_run_mcts_with_sim_traces(simulations=64)

    root = out["root_tracker"]
    traces = out["sim_traces"]
    assert traces, f"{game_id}: on_sim_step never fired — descent ran 0 steps?"

    # At least one sim_tracker snapshot must DIFFER from root. If every
    # snapshot equals root, the descent feed is broken (sim_tracker
    # frozen).
    diverged = any(t["tracker"] != root for t in traces)
    assert diverged, (
        f"{game_id}: every recorded sim_tracker snapshot ({len(traces)} "
        f"records) equals the root tracker — descent-time "
        f"observe_public_event is not running.\n  root={root}")


@pytest.mark.parametrize("game_id", _STATEFUL_TRACKER_GAMES)
def test_session_tracker_unaffected_by_search(game_id: str):
    """Companion guard: running search must NOT mutate the session's
    tracker. The sim_tracker is a sim-local clone; its updates die with
    the sim. cfg_.root_belief_tracker is read-only inside MCTS.
    """
    model_path = get_test_model(game_id)

    gs = engine.GameSession(
        game_id, seed=42, model_path=model_path, use_filter=False)

    before = gs.get_belief_snapshot()
    gs._debug_run_mcts_with_sim_traces(simulations=64)
    after = gs.get_belief_snapshot()

    assert before == after, (
        f"{game_id}: session-shared tracker mutated by MCTS search.\n"
        f"  before={before}\n  after={after}")
