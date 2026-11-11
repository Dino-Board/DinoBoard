"""ISMCTS info-leak regression: MCTS policy on the AI-API path stays
statistically close to the selfplay (truth-driven) path.

If GameSession MCTS secretly reads `bundle_->state`'s hidden fields
(opponent hand, blind-reserved card IDs, etc.), its policy output will
differ from an AI-API session that only knows what a real observer
would see — because the latter has no access to truth. This test
constructs both paths on top of the same observation history and
compares their MCTS policies at the perspective player's turn.

The match is statistical, not bit-exact: the two paths use independent
MCTS RNGs (selfplay rolls its own per-sim sim_rng, API session rolls
its own), so single-sim sample noise is expected. The thresholds —
argmax-mismatch rate <= 0.65, average TV distance <= 0.40 — bound the
noise loosely; a real info leak would push them well past these.

Currently parametrized only on Love Letter (bluff-heavy hidden hand,
LEAK_SENSITIVE_GAMES). Splendor is excluded due to a known
replay / `self_reserve_deck` interleave issue with `get_ai_action`.
The Web path is not directly covered — it shares the engine stack
with the API path but exercises a different binding entry point.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "platform"))

import dinoboard_engine as engine
from conftest import get_test_model


# Games whose MCTS policy depends on hidden info and therefore can leak.
# Splendor's event replay has a pre-existing issue unrelated to this
# refactor (self_reserve_deck payload tracking can fail when interleaved
# with get_ai_action calls); keep Splendor out until that's untangled.
LEAK_SENSITIVE_GAMES = ["loveletter"]


def _visits_to_distribution(action_ids, visits):
    """Normalize visit counts into a probability distribution keyed on
    action id. Ignores zero-visit actions so differences on never-
    considered actions don't inflate the total-variation distance."""
    total = sum(visits)
    if total <= 0:
        return {}
    return {a: v / total for a, v in zip(action_ids, visits) if v > 0}


def _tv_distance(p: dict, q: dict) -> float:
    """Total variation distance between two distributions over action IDs."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


@pytest.mark.parametrize("game_id", LEAK_SENSITIVE_GAMES)
def test_api_mcts_policy_matches_selfplay(game_id):
    """Aggregate across multiple seeds so a single short episode (Love
    Letter often ends in 4-5 plies, leaving only ~2 perspective decisions)
    can't dominate the mismatch rate. Across 8 seeds we typically see
    15-25 perspective plies — enough to drown out per-ply MCTS RNG ties."""
    perspective = 0
    seed_api = 9999
    sims = 50
    model_path = get_test_model(game_id)

    seeds_gt = [42, 100, 200, 300, 500, 700, 1000, 1234]

    comparisons = 0
    argmax_mismatches = 0
    tv_samples = []

    for seed_gt in seeds_gt:
        ep = engine.run_selfplay_episode(
            game_id=game_id,
            seed=seed_gt,
            model_path=model_path,
            simulations=sims,
            max_game_plies=40,
            temperature=0.0,
            trace_perspective=perspective,
        )
        trace = ep["observation_trace"]
        samples = ep["samples"]

        perspective_plies = [
            i for i, s in enumerate(samples) if s["player"] == perspective
        ]
        if not perspective_plies:
            continue

        api_gs = engine.GameSession(
            game_id, seed=seed_api, model_path=model_path, use_filter=False)
        api_gs.apply_initial_observation(perspective, ep["initial_observation"])

        ply_i = 0
        for step in trace:
            if ply_i in perspective_plies and samples[ply_i]["player"] == perspective:
                api_result = api_gs.get_ai_action(sims, 0.0)
                if "action" in api_result:
                    comparisons += 1
                    selfplay_argmax = samples[ply_i]["policy_action_ids"][
                        samples[ply_i]["policy_action_visits"].index(
                            max(samples[ply_i]["policy_action_visits"]))]
                    if api_result["action"] != selfplay_argmax:
                        argmax_mismatches += 1
                    sp_dist = _visits_to_distribution(
                        samples[ply_i]["policy_action_ids"],
                        samples[ply_i]["policy_action_visits"])
                    api_actions = api_result["stats"].get("root_actions", [])
                    api_visits = api_result["stats"].get("root_action_visits", [])
                    if api_actions and api_visits:
                        api_dist = _visits_to_distribution(api_actions, api_visits)
                        tv_samples.append(_tv_distance(sp_dist, api_dist))
            api_gs.apply_observation(
                step["action"],
                events=step["events"],
                public_snapshot=step.get("public_snapshot", {}),
            )
            ply_i += 1

    if comparisons == 0:
        pytest.skip(f"{game_id}: no perspective-acting plies to compare")

    mismatch_rate = argmax_mismatches / comparisons
    # With sims=50 and temperature=0, ISMCTS at the root determinizes a
    # different world per simulation. Across many low-priored actions in
    # Love Letter, argmax flips frequently between independent runs even
    # when the tracker is correct. Empirically (untrained models, 30
    # weight initializations × 8 seeds), mismatch rate is 21-54% with
    # mean ~36%. The threshold below is set with margin above that — a
    # real info leak (where one path sees truth) would push the rate
    # well above 70% because the cheating side gets consistent answers
    # while the observer side has to sample.
    assert mismatch_rate <= 0.65, (
        f"[{game_id}] argmax divergence rate {mismatch_rate:.1%} "
        f"({argmax_mismatches}/{comparisons}) — possible info leak: "
        f"GameSession's MCTS picks differ from API's MCTS on the same "
        f"observation history. Expected ≤ 65% under MCTS RNG jitter.")

    # Full distribution check — catches subtle skew that doesn't flip
    # the argmax. Average TV across all compared plies should be small;
    # a leak biasing priors or value backup in one path would push it up.
    if tv_samples:
        avg_tv = sum(tv_samples) / len(tv_samples)
        max_tv = max(tv_samples)
        assert avg_tv <= 0.40, (
            f"[{game_id}] average TV distance {avg_tv:.3f} across "
            f"{len(tv_samples)} plies exceeds 0.40 — MCTS visit "
            f"distribution differs materially between selfplay and API "
            f"paths. max_tv={max_tv:.3f}. Possible info leak.")
