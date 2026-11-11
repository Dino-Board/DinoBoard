"""Phase 1: termination is winner-takes-all (first to empty hand wins)."""
import random
import dinoboard_engine
import pytest

from .test_state_basic import skip_if_unbuilt, VARIANTS


def _play_random_episode(variant, seed, max_plies=500):
    rng = random.Random(seed)
    gs = dinoboard_engine.GameSession(variant, seed=seed)
    plies = 0
    while not gs.is_terminal and plies < max_plies:
        la = gs.get_legal_actions()
        assert la
        gs.apply_action(rng.choice(la))
        plies += 1
    return gs


@skip_if_unbuilt
class TestTerminalReward:

    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 99, 123, 456])
    def test_random_episode_terminates_with_unique_winner(self, variant, seed):
        gs = _play_random_episode(variant, seed)
        assert gs.is_terminal, "random episode failed to terminate within bound"
        sd = gs.get_state_dict()
        w = sd["winner"]
        assert 0 <= w < sd["num_players"]
        # Winner has empty hand.
        assert sum(sd["players"][w]["hand_count"]) == 0
        # No legal actions in terminal.
        assert gs.get_legal_actions() == []

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_terminal_value_winner_takes_all(self, variant):
        # Aggregate over many random episodes — winners should be roughly
        # distributed across seats (no engine-side bias against any seat).
        # Loose bound: each seat wins at least once across 40 seeds.
        counts = {}
        for seed in range(40):
            gs = _play_random_episode(variant, seed)
            assert gs.is_terminal
            w = gs.get_state_dict()["winner"]
            counts[w] = counts.get(w, 0) + 1
        np = dinoboard_engine.GameSession(variant, seed=0).num_players
        for p in range(np):
            assert counts.get(p, 0) >= 1, \
                f"seat {p} never won in 40 random {variant} episodes — bias suspect: {counts}"
