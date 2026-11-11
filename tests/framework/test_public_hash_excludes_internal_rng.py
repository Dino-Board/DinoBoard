"""BUG-028 regression: state_hash_for_perspective must NOT hash internal RNG state.

Background (see CLAUDE.md "Public ≠ every state member that isn't a hand"
and KNOWN_ISSUES BUG-028):

  state_hash_for_perspective() defines what makes two states share the same MCTS
  DAG node. If a game accidentally hashes internal RNG salt, mt19937
  snapshots, deck-shuffle order, or pre-draw bag/box_lid vector ordering
  into the public hash, then two states that ARE the same public info set
  end up at different DAG nodes — search splits, transposition reuse
  collapses, and policy/value estimates become world-specific instead of
  info-set-specific. The symptom is silent: search "runs", but is
  mysteriously weak / inconsistent / different from the API path. Never
  a crash.

Test method (catches BUG-028 by construction):

  For each hidden-info game:
    1. Run a self-play episode under seed_truth, capturing the observation
       trace from `perspective`'s point of view.
    2. Build two API sessions with DIFFERENT seeds (seed_a, seed_b). Each
       internal RNG is therefore in a different state from the start.
    3. Drive both API sessions through the same observation history.
    4. Assert state_hash_for_perspective(perspective) is bit-equal between
       them after every applied observation.

  If state_hash_for_perspective hashes any internal RNG state — directly (rng_salt,
  mt19937 snapshot) or indirectly (vector order of cards in bag, deck,
  box_lid that no player can derive) — the two hashes diverge and the
  test fails immediately.

This is the strongest preventative we have for BUG-028: any new game that
slips a private/internal field into state_hash_for_perspective breaks CI.
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


# Games with hidden information AND a public_event_extractor (required to
# drive the API session via apply_observation). Fully-public games
# (tictactoe, quoridor) are trivially BUG-028-free by structure.
from conftest import games_with_capability
HIDDEN_INFO_GAMES = games_with_capability("hidden_info", "snapshot")


# Sweep parameters tuned so the test reliably triggers BUG-028-class drifts
# without ballooning CI runtime. Single-seed (the original form of this
# test) passed even on pre-fix engines that had a real drift bug, because
# whether `randomize_unseen` happens to assign the relevant unseen card to
# `decks[t]` vs an opp face-down reserve is itself random — so any check
# that hashes a single (seed_a, seed_b) pair is sampling from a Bernoulli
# distribution and may miss the divergence. The sweep makes the
# Bernoulli's failure rate `≈ p^N`, so realistic per-game drift rates
# (~2% per episode in the second-round Splendor bug) become reliable
# pytest failures (P[no drift in 30 episodes] ≈ 0.55, in 60 ≈ 0.30, in
# 120 ≈ 0.09 — we run 60 with model_path="" which is faster and gives
# us a reliable but not flaky safety net).
DRIFT_SWEEP_TRUTH_SEEDS = list(range(50, 50 + 60 * 7, 7))  # 60 episodes
DRIFT_SWEEP_MAX_PLIES = 60


@pytest.mark.parametrize("game_id", HIDDEN_INFO_GAMES)
def test_public_hash_invariant_under_internal_rng(game_id):
    """Sweep version of BUG-028 regression: across many self-play seeds,
    two API sessions seeded differently but fed the same observation
    history must produce identical state_hash_for_perspective.

    Why a sweep, not a single seed: the divergence trigger is conditional
    on `randomize_unseen` happening to assign a specific unseen card to
    one bucket vs another — itself random per (seed_a, seed_b). A single
    pair can hash-equal even on a buggy build by luck. We want CI to
    deterministically (or near-deterministically) catch any new game's
    `state_hash_for_perspective` slipping in unobservable randomness.

    Failure modes this catches by construction:
      - Hashing internal RNG state directly (rng_salt, mt19937 snapshot)
      - Hashing pre-draw deck/bag/box_lid vector ordering
      - Hashing a count/size that varies across `randomize_unseen` worlds
        but is NOT publicly derivable (e.g. `decks[t].size()` when cards
        can also live in opp face-down reserves — second-round Splendor
        bug, 2026-05-07)
    """
    perspective = 0
    seed_a = 9999
    seed_b = 314159  # different RNG state from seed_a

    model_path = get_test_model(game_id)
    drifts: list[tuple[int, int, str]] = []  # (seed_truth, ply, detail)

    for seed_truth in DRIFT_SWEEP_TRUTH_SEEDS:
        ep = engine.run_selfplay_episode(
            game_id=game_id, seed=seed_truth, model_path=model_path,
            simulations=20, max_game_plies=DRIFT_SWEEP_MAX_PLIES,
            trace_perspective=perspective,
        )
        trace = ep.get("observation_trace") or []
        if not trace:
            continue

        sess_a = engine.GameSession(
            game_id, seed=seed_a, model_path="", use_filter=False)
        sess_b = engine.GameSession(
            game_id, seed=seed_b, model_path="", use_filter=False)
        sess_a.apply_initial_observation(perspective, ep["initial_observation"])
        sess_b.apply_initial_observation(perspective, ep["initial_observation"])

        h_a = sess_a.state_hash_for_perspective(perspective)
        h_b = sess_b.state_hash_for_perspective(perspective)
        if h_a != h_b:
            drifts.append((seed_truth, -1, f"init: {h_a:#x} vs {h_b:#x}"))
            continue

        for step in trace:
            sess_a.apply_observation(
                step["action"], events=step["events"],
                public_snapshot=step.get("public_snapshot", {}))
            sess_b.apply_observation(
                step["action"], events=step["events"],
                public_snapshot=step.get("public_snapshot", {}))
            h_a = sess_a.state_hash_for_perspective(perspective)
            h_b = sess_b.state_hash_for_perspective(perspective)
            if h_a != h_b:
                event_kinds = [e["kind"] for e in step["events"]]
                drifts.append((
                    seed_truth, step["ply"],
                    f"act={step['action']} events={event_kinds} "
                    f"{h_a:#x} vs {h_b:#x}"))
                break

    assert not drifts, (
        f"[{game_id}] state_hash_for_perspective diverged across "
        f"{len(drifts)}/{len(DRIFT_SWEEP_TRUTH_SEEDS)} self-play seeds "
        f"under different API session RNG seeds.\n"
        f"First few drifts:\n  " +
        "\n  ".join(f"seed_truth={s} ply={p}: {d}" for s, p, d in drifts[:5]) +
        f"\n\nThis means state_hash_for_perspective(perspective={perspective}) "
        f"is hashing state that varies under "
        f"`randomize_unseen` but is NOT derivable from the observation "
        f"history. Common causes: (1) rng_salt / mt19937 snapshot; (2) "
        f"pre-draw deck/bag/box_lid vector ordering; (3) a size/count "
        f"that depends on the random partition of unseen cards across "
        f"deck vs opp face-down reserves (second-round Splendor bug, "
        f"2026-05-07). See BUG-028. Public hash MUST be a function of "
        f"the observation history alone."
    )


# Games where the opp information set is combinatorially too large for
# random worlds to collide at modest sim counts under world_cap=0
# (K=sims / M=1). 6 nimmt!: each opp holds 9 cards drawn from ~88 unseen
# → C(88,9) ≈ 10^12 possible randomize_unseen outputs, so two sims never
# produce the same opp-perspective node by chance when each sim picks its
# own world.
#
# For these games we run the behavioral test under world_cap > 0 so K
# fixed worlds × M sims-per-world force the M inner sims to share a
# descent prefix — reuse becomes inevitable by construction. Without
# this we'd have to rely solely on the structural guard
# (test_public_hash_invariant_under_internal_rng) and lose the
# behavioral signal for any future BUG-028-class regression that
# happens to slip past the structural sweep.
_LARGE_INFOSET_WORLD_CAP = {"6nimmit": 22}  # K≈sqrt(500), M≈23


@pytest.mark.parametrize("game_id", HIDDEN_INFO_GAMES)
def test_mcts_dag_reuses_under_different_sampled_worlds(game_id):
    """Behavioral angle on BUG-028: under correct hash scope, ISMCTS root
    determinization samples many worlds per simulation (different deck
    orderings, different opp hand assignments), but all worlds in the
    same information set hash to the same DAG node and share visits.
    This produces non-trivial `dag_reuse_hits`.

    If state_hash_for_perspective hashes internal RNG state, every sim's hash
    chain becomes unique (each randomize_unseen seed is different),
    DAG reuse collapses to ~0, and the search tree degenerates into N
    disjoint thin chains. This is the silent failure mode of BUG-028:
    search "runs" at the requested simulation count but each sim
    explores almost nothing.

    We assert reuse > 0 as a strict floor: any healthy hidden-info
    game with root determinization MUST produce at least one transposition
    hit at 500 sims. For large-infoset games (6 nimmt!) we use
    world_cap = K ≈ √sims so the M inner sims of each world share a
    descent prefix and reuse is forced to occur — without world_cap their
    info set is too big for any two world-1 sims to collide by luck.
    """
    world_cap = _LARGE_INFOSET_WORLD_CAP.get(game_id, 0)
    model_path = get_test_model(game_id)
    sess = engine.GameSession(
        game_id, seed=2026, model_path=model_path, use_filter=False)
    if sess.is_terminal:
        pytest.skip(f"[{game_id}] terminal at construction")
    r = sess.get_ai_action(
        simulations=500, world_cap=world_cap, temperature=0.0)
    stats = r["stats"]
    sims = stats.get("simulations", 0)
    reuse = stats.get("dag_reuse_hits", 0)
    assert sims > 0, f"[{game_id}] no simulations ran"
    assert reuse > 0, (
        f"[{game_id}] MCTS produced {sims} simulations (world_cap={world_cap}) "
        f"but dag_reuse_hits={reuse}. With root determinization on a hidden-info "
        f"game, every sim samples a different unseen world but they should "
        f"merge by information set in the DAG. Zero reuse means every sim "
        f"hashes uniquely — almost certainly because state_hash_for_perspective is "
        f"reading internal RNG state (rng_salt, deck/bag ordering, mt19937 "
        f"snapshot). See BUG-028. Effective search depth has collapsed to 1 "
        f"per sim — the AI is silently MUCH weaker than the simulation count "
        f"suggests."
    )


