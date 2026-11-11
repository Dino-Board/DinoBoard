"""A ply with exactly one legal action is played without simulations.

Every simulation of such a search would pass through the single root edge, so
the result is known up front: that action, visit distribution one-hot.
NetMcts::search_root returns it after the root evaluation; selfplay still emits
the sample (same ply / player / features / playout-cap flag), so the sample
stream stays aligned for every per-ply consumer.

Which plies are forced is recomputed independently here by replaying the
episode's actions on a GameSession with the same seed (same truth draws).
"""
import math
import random

import dinoboard_engine as engine
import pytest

from conftest import get_test_model

SIMS, FAST = 24, 6
# Both have forced plies in every game: a tile with one placement or a
# pass-only meeple decision (carcassonne), the last tile group of a round (azul).
GAMES = ["carcassonne", "azul"]
SEEDS = [301, 302]


def _replay(game_id, seed, samples):
    """(player to move, legal actions) before each ply of the episode."""
    gs = engine.GameSession(game_id, seed=seed)
    out = []
    for s in samples:
        legal = gs.get_all_legal_actions()
        assert s["action_id"] in legal, f"[{game_id}] ply {s['ply']}: illegal action in episode"
        out.append((gs.current_player, legal))
        gs.apply_action(s["action_id"])
    return out


def _check_episode(game_id, seed, ep, budget):
    """`budget(sample)` = simulations a searched ply of that kind should run."""
    samples = ep["samples"]
    assert len(samples) == ep["total_plies"]
    n_forced = n_searched = 0
    for i, (s, (player, legal)) in enumerate(zip(samples, _replay(game_id, seed, samples))):
        assert s["ply"] == i and s["player"] == player
        assert len(s["features"]) > 0 and len(s["z_values"]) == 2
        if len(legal) == 1:
            n_forced += 1
            assert s["simulations"] == 0, f"[{game_id}] forced ply {i} ran a search"
            assert s["action_id"] == legal[0]
            assert list(s["policy_action_ids"]) == legal
            assert list(s["policy_action_visits"]) == [1]
        else:
            n_searched += 1
            assert s["simulations"] == budget(s), f"[{game_id}] ply {i}"
            assert sum(s["policy_action_visits"]) == s["simulations"]
            assert set(s["policy_action_ids"]) == set(legal)
    return n_forced, n_searched


@pytest.mark.parametrize("game_id", GAMES)
def test_selfplay_forced_plies_run_no_simulations(game_id):
    forced = searched = 0
    for seed in SEEDS:
        ep = engine.run_selfplay_episode(
            game_id=game_id, seed=seed, model_path=get_test_model(game_id),
            simulations=SIMS, max_game_plies=400, training_filter_ratio=0.0)
        f, s = _check_episode(game_id, seed, ep, lambda _s: SIMS)
        forced += f
        searched += s
    assert forced > 0 and searched > 0, f"[{game_id}] forced={forced} searched={searched}"


@pytest.mark.parametrize("game_id", GAMES)
def test_playout_cap_flags_and_budgets_with_forced_plies(game_id):
    def step(full):
        return engine.run_concurrent_selfplay_step(
            game_id=game_id, seeds=SEEDS, latest_model_path=get_test_model(game_id),
            num_self_threads=1, simulations=full, fast_simulations=FAST,
            full_search_prob=0.3, max_game_plies=400, training_filter_ratio=0.0,
            batch_size=16, max_queue_size=64, intra_op_num_threads=1, leaf_batch_size=4)

    a, b = step(SIMS), step(SIMS // 2)
    forced = 0
    for seed, ep_a, ep_b in zip(SEEDS, a["episodes"], b["episodes"]):
        f, _ = _check_episode(
            game_id, seed, ep_a, lambda s: SIMS if s["full_search"] else FAST)
        forced += f
        # The full/fast coin flip of ply i depends only on (seed, i): a forced
        # ply consumes its flip like any other. The two runs play different
        # games (different budgets) yet must agree on the flags ply by ply.
        flags_a = [s["full_search"] for s in ep_a["samples"]]
        flags_b = [s["full_search"] for s in ep_b["samples"]]
        n = min(len(flags_a), len(flags_b))
        assert n > 20 and flags_a[:n] == flags_b[:n]
        assert [s["action_id"] for s in ep_a["samples"]] != [s["action_id"] for s in ep_b["samples"]]
    assert forced > 0


def _session_at_forced_position(game_id):
    gs = engine.GameSession(game_id, seed=7, model_path=get_test_model(game_id))
    rng = random.Random(7)
    while not gs.is_terminal:
        legal = gs.get_all_legal_actions()
        if len(legal) == 1:
            return gs, legal[0]
        gs.apply_action(rng.choice(legal))
    raise AssertionError(f"[{game_id}] no forced position found")


@pytest.mark.parametrize("game_id", GAMES)
def test_session_forced_move_returns_well_formed_stats(game_id):
    gs, only = _session_at_forced_position(game_id)
    res = gs.get_ai_action(simulations=50, temperature=0.0)
    stats = res["stats"]
    assert res["action"] == only
    assert stats["simulations"] == 0
    assert list(stats["root_actions"]) == [only] and list(stats["root_action_visits"]) == [1]
    # Value read-out consumers (web win-rate display) index these per seat.
    root_values = list(stats["root_values"])
    assert len(root_values) == gs.num_players
    assert all(math.isfinite(v) and -1.0 <= v <= 1.0 for v in root_values)
    assert list(stats["action_values"][only]) == root_values
    assert stats["best_value"] == root_values[gs.current_player]


@pytest.mark.parametrize("game_id", GAMES)
def test_analysis_mode_still_searches_forced_positions(game_id):
    gs, only = _session_at_forced_position(game_id)
    res = gs.get_ai_action(simulations=50, temperature=0.0, cover_root_edges=True)
    assert res["action"] == only
    assert res["stats"]["simulations"] == 50
    assert list(res["stats"]["root_action_visits"]) == [50]
