"""Three-RNG independence contract (CLAUDE.md "Public ≠ ...", §B target state):

  There are exactly three RNGs at runtime:
    - gt_rng        — held by GT runner / API truth session, drives ground-truth
                      `do_action_fast` and initial deal.
    - session_rng   — held by each AI session. The session never calls
                      `randomize_unseen` or `do_action_fast`, so this RNG
                      is consumed only for internal seed derivations
                      (e.g. ply-indexed sub-seeds for the MCTS search).
    - sim_rng       — sim-local stack variable, derived from the MCTS root
                      `search_seed`, drives `randomize_unseen` at sim entry
                      AND every `do_action_fast` along descent.

  None is derived from another. None is shared. Each is independently seeded
  by its caller. Reseeding any one MUST NOT affect the observable behavior
  driven by the others.

This test pins three concrete invariants of that contract:

  (A) gt_rng ⊥ session_rng:
      Two API sessions seeded with DIFFERENT session seeds, fed the SAME
      observation trace produced by a single GT run, must produce identical
      `state_hash_for_perspective(perspective)` after every applied
      observation. (This overlaps test_public_hash_excludes_internal_rng but
      pins the directionality explicitly: GT seed is fixed, session seed
      varies.)

  (B) session_rng ⊥ sim_rng:
      Same observation history driven into the same session seed produces
      identical MCTS visit distributions across calls, even if some external
      RNG (here we approximate by varying the session seed across two
      sessions) has drawn different draws — because MCTS derives its
      `search_seed` from `(seed_, ply_count_)` deterministically. This
      manifests as: identical session seed + identical observation trace ⇒
      identical MCTS root visit vector at the perspective's first turn
      under temperature=0.

  (C) Determinism of session_rng given (seed, observation history):
      Two independently-constructed sessions with the SAME seed driven
      through the SAME observation history produce byte-equal *public*
      hash sequences. (Hidden-field bytes on the session are no longer
      freshened — see ALGORITHM_OVERVIEW.md §1.2 — but the public
      projection must remain seed-deterministic.)

Failure of any of (A) (B) (C) means an RNG is leaking into a path it should
not — the most common regression class is "search_seed accidentally derived
from session step_rng" or "session_rng accidentally seeded from gt_rng's
output", both of which cause silent search-strength / belief-sample
non-reproducibility.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "platform"))

import dinoboard_engine as engine
from conftest import get_test_model, games_with_capability


HIDDEN_INFO_GAMES = games_with_capability("hidden_info", "snapshot")


def _drive_session(game_id: str, seed: int, perspective: int, ep: dict):
    sess = engine.GameSession(
        game_id, seed=seed, model_path="", use_filter=False)
    sess.apply_initial_observation(perspective, ep["initial_observation"])
    hashes = [sess.state_hash_for_perspective(perspective)]
    for step in ep["observation_trace"]:
        sess.apply_observation(
            step["action"],
            events=step["events"],
            public_snapshot=step.get("public_snapshot", {}),
        )
        hashes.append(sess.state_hash_for_perspective(perspective))
    return sess, hashes


@pytest.mark.parametrize("game_id", HIDDEN_INFO_GAMES)
def test_session_seed_does_not_affect_public_hash(game_id):
    """(A) gt_rng ⊥ session_rng — public hash is a function of the
    observation history only, not of session_rng draws."""
    perspective = 0
    seed_truth = 4242

    model_path = get_test_model(game_id)
    ep = engine.run_selfplay_episode(
        game_id=game_id, seed=seed_truth, model_path=model_path,
        simulations=20, max_game_plies=40, trace_perspective=perspective,
    )
    assert ep.get("observation_trace"), f"{game_id}: empty trace"

    _, ha = _drive_session(game_id, 11111, perspective, ep)
    _, hb = _drive_session(game_id, 99999, perspective, ep)

    assert ha == hb, (
        f"[{game_id}] state_hash_for_perspective diverged between two API "
        f"sessions seeded differently but fed the same observation trace. "
        f"This means the public hash depends on session_rng draws — an "
        f"RNG-independence violation. See CLAUDE.md three-RNG contract.\n"
        f"  first divergence at index "
        f"{next((i for i, (a, b) in enumerate(zip(ha, hb)) if a != b), -1)}")


@pytest.mark.parametrize("game_id", HIDDEN_INFO_GAMES)
def test_same_seed_same_history_yields_identical_hidden(game_id):
    """(C) Determinism of session_rng under fixed (seed, observation
    history). Two independently constructed sessions with identical seed
    driven through identical observations must produce byte-equal hidden
    snapshots. If they don't, session_rng is being seeded from a moving
    target (clock, address, GT draws) instead of (seed_, ply_count_)."""
    perspective = 0
    seed_truth = 4242
    seed_session = 7777

    model_path = get_test_model(game_id)
    ep = engine.run_selfplay_episode(
        game_id=game_id, seed=seed_truth, model_path=model_path,
        simulations=20, max_game_plies=40, trace_perspective=perspective,
    )
    assert ep.get("observation_trace"), f"{game_id}: empty trace"

    sess_a, ha = _drive_session(game_id, seed_session, perspective, ep)
    sess_b, hb = _drive_session(game_id, seed_session, perspective, ep)

    assert ha == hb, (
        f"[{game_id}] same seed + same observation history produced "
        f"different public hashes across two sessions; session_rng is not "
        f"deterministic from (seed, ply_count_).")

    # Full state dicts (public + viz=0 leftovers) must agree byte-equal
    # — same seed + same observation history must produce byte-identical
    # session state, which means the deterministic seed flow into
    # reset_with_seed has not been polluted by an external RNG.
    assert sess_a.get_state_dict() == sess_b.get_state_dict(), (
        f"[{game_id}] same seed + same trace produced different state "
        f"dicts. The session is being seeded from a moving target "
        f"(clock, address, GT draws) instead of the explicit seed.")


@pytest.mark.parametrize("game_id", HIDDEN_INFO_GAMES)
def test_mcts_visits_reproducible_under_same_session_seed(game_id):
    """(B) session_rng ⊥ sim_rng — MCTS sim_rng is derived from
    (session.seed_, ply_count_), so two sessions with the same seed must
    produce identical MCTS root visit vectors at the same observation
    point. Drives just to the perspective's first decision and queries
    `get_ai_action` at temperature=0.

    If sim_rng leaked from session_rng's continuing draw counter (or from
    a wall-clock source), two independently-constructed sessions would
    produce different visit vectors despite identical observation history.
    """
    perspective = 0
    seed_truth = 4242
    seed_session = 13579

    model_path = get_test_model(game_id)
    ep = engine.run_selfplay_episode(
        game_id=game_id, seed=seed_truth, model_path=model_path,
        simulations=20, max_game_plies=40, trace_perspective=perspective,
    )
    trace = ep.get("observation_trace") or []
    if not trace:
        pytest.skip(f"{game_id}: empty trace")

    def _drive_until_perspective_turn():
        sess = engine.GameSession(
            game_id, seed=seed_session, model_path=model_path,
            use_filter=False)
        sess.apply_initial_observation(perspective, ep["initial_observation"])
        for step in trace:
            if (not sess.is_terminal
                    and sess.current_player == perspective):
                return sess
            sess.apply_observation(
                step["action"],
                events=step["events"],
                public_snapshot=step.get("public_snapshot", {}))
        return sess  # may be terminal — caller skips

    sess_a = _drive_until_perspective_turn()
    sess_b = _drive_until_perspective_turn()

    if sess_a.is_terminal or sess_a.current_player != perspective:
        pytest.skip(f"{game_id}: trace never reached perspective's turn")

    ra = sess_a.get_ai_action(simulations=40, temperature=0.0)
    rb = sess_b.get_ai_action(simulations=40, temperature=0.0)

    # Compare by the chosen action plus the simulations actually run.
    # Visit vectors aren't exposed directly by get_ai_action; the action +
    # simulations count + best_value form a fingerprint that diverges
    # under sim_rng leakage.
    sa = ra["stats"]
    sb = rb["stats"]
    assert ra["action"] == rb["action"], (
        f"[{game_id}] same-seed sessions chose different actions at the "
        f"same observation point: {ra['action']} vs {rb['action']}. "
        f"sim_rng appears to depend on something other than "
        f"(session.seed_, ply_count_).")
    assert sa["simulations"] == sb["simulations"]
    assert sa["best_value"] == pytest.approx(sb["best_value"], rel=0.0, abs=0.0), (
        f"[{game_id}] best_value diverged under identical seed + history: "
        f"{sa['best_value']} vs {sb['best_value']}.")
