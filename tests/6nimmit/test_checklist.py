"""6 nimmt! acceptance checklist (.claude/skills/game-test/checklist.md).

Two to four players, one hand: ten cards each from all 104, ten rounds; the
fewest bullheads win. The value of a game is its bullhead margin (the
others' average minus mine, /25, clipped to +-1), so a 66-point match is
played one hand at a time. Private hands, simultaneous face-down play. Token
encoder only; the belief tracker samples the other hands uniformly from the
unseen cards (no belief network). Heuristics `rule_v1` / `lkf_v1`. No tail
solver. An adjudicator gives a ply-capped game to the only player with the
fewest bullheads.

The rules themselves are covered by test_rules.py, the encoder by
test_token_encoder.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import dinoboard_engine
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from conftest import (assert_api_belief_matches_selfplay, get_test_model, load_game_config,  # noqa: E402
                      run_random_episode_states)

GAME = "6nimmit"
VARIANTS = {"6nimmit": 2, "6nimmit_3p": 3, "6nimmit_4p": 4}
MARGIN_SCALE = 25.0
CONFIG = load_game_config(GAME)
ACTION_SPACE = 108
MAX_PLIES = CONFIG["max_game_plies"]
TOKEN_DIM = dinoboard_engine.game_metadata(GAME)["token_layout"]["token_dim"]
variants = pytest.mark.parametrize("game", list(VARIANTS))


# 1. Registration & config
class TestRegistration:

    def test_registered(self):
        games = dinoboard_engine.available_games()
        assert set(VARIANTS) | {"6nimmit_2p"} <= set(games)
        assert not [g for g in games if g.startswith("6nimmit") and g not in set(VARIANTS) | {"6nimmit_2p"}]

    @variants
    def test_metadata_matches_config(self, game):
        cfg, meta = load_game_config(game), dinoboard_engine.game_metadata(game)
        assert meta["action_space"] == cfg["action_space"] == ACTION_SPACE
        assert meta["feature_dim"] == cfg["feature_dim"] == 0
        assert meta["num_players"] == cfg["num_players"] == VARIANTS[game]
        assert cfg["max_game_plies"] == MAX_PLIES

    def test_train_config_loads(self):
        from training.cli import find_train_config
        cfg, _ = find_train_config(GAME)
        assert cfg["network"]["type"] == "token_transformer"


# 2. GameSession basics
class TestGameSession:

    @variants
    def test_session_constructs(self, game):
        gs = dinoboard_engine.GameSession(game, seed=42)
        assert not gs.is_terminal and gs.num_players == VARIANTS[game]
        st = gs.get_state_dict()
        assert [p["hand_heads"] for p in st["players"]] == [0] * VARIANTS[game]
        dealt = [c for p in st["players"] for c in p["hand"]] + [row[0] for row in st["table"]]
        assert len(set(dealt)) == 4 + 10 * VARIANTS[game] and sorted(st["used"]) == sorted(dealt[-4:])

    @variants
    def test_initial_legal_actions_are_the_hand(self, game):
        gs = dinoboard_engine.GameSession(game, seed=42)
        st = gs.get_state_dict()
        hand = set(st["players"][st["current_player"]]["hand"])
        assert len(hand) == 10 and {a + 1 for a in gs.get_legal_actions()} == hand

    def test_action_info(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        for action in gs.get_legal_actions():
            info = gs.get_action_info(action)
            assert info["type"] == "commit" and info["card"] == action + 1
        assert gs.get_action_info(105) == {"action_id": 105, "type": "small_choose_row", "row": 1}
        assert gs.get_action_info(108)["hidden"] is True


# 3 / 4. Encoding
class TestEncoder:

    @variants
    def test_encode_state(self, game):
        info = dinoboard_engine.encode_state(game, seed=42)
        assert len(info["legal_mask"]) == ACTION_SPACE
        # global + players + 4 rows + 10 hand cards
        assert len(info["features"]) == (1 + VARIANTS[game] + 4 + 10) * TOKEN_DIM
        assert any(v != 0 for v in info["features"])

    def test_encoding_changes_with_the_state(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        before = list(gs._encode(0)["features"])
        gs.apply_action(gs.get_legal_actions()[0])
        assert list(gs._encode(0)["features"]) != before


def _assert_result(ep, n, finished=True):
    """The final scores are minus the bullheads taken; each seat's value is its margin. A game
    stopped by the ply cap is reported for the only player with the fewest bullheads (else
    nobody), and the framework trains every seat on a loss for it (anti-stalling)."""
    heads, z = [-s for s in ep["final_scores"]], ep["terminal_z_values"]
    best = [p for p in range(n) if heads[p] == min(heads)]
    assert ep["winner"] == (best[0] if len(best) == 1 else -1)
    if not finished:
        assert z == [-1.0] * n
        return
    for p in range(n):
        others = (sum(heads) - heads[p]) / (n - 1)
        assert z[p] == pytest.approx(max(-1.0, min(1.0, (others - heads[p]) / MARGIN_SCALE)))
    if all(abs(v) < 1 for v in z):
        assert abs(sum(z)) < 1e-5


# 5 / 7. Selfplay
class TestSelfplay:

    def _episode(self, seed, game=GAME, **kw):
        return dinoboard_engine.run_selfplay_episode(
            game_id=game, seed=seed, model_path=get_test_model(game), simulations=6, max_game_plies=MAX_PLIES, **kw)

    @variants
    def test_episode_is_a_whole_game(self, game):
        n = VARIANTS[game]
        ep = self._episode(42, game)
        stats, heads = ep["custom_stats"], [-s for s in ep["final_scores"]]
        assert ep["total_plies"] >= 10 * n
        assert len(heads) == n and sum(heads) > 0
        assert stats["fewest_heads"] == min(heads) and stats["most_heads"] == max(heads)
        _assert_result(ep, n)

    @variants
    def test_sample_integrity(self, game):
        ep = self._episode(43, game)
        _assert_result(ep, VARIANTS[game])
        for s in ep["samples"]:
            assert len(s["features"]) % TOKEN_DIM == 0 and len(s["features"]) > 0
            assert len(s["legal_mask"]) == ACTION_SPACE
            legal = {i for i, m in enumerate(s["legal_mask"]) if m > 0}
            visited = {a for a, v in zip(s["policy_action_ids"], s["policy_action_visits"]) if v > 0}
            assert visited and visited <= legal

    def test_determinism_same_seed(self):
        kw = dict(dirichlet_alpha=0.0, dirichlet_epsilon=0.0)
        ep1, ep2 = self._episode(12345, **kw), self._episode(12345, **kw)
        assert [s["action_id"] for s in ep1["samples"]] == [s["action_id"] for s in ep2["samples"]]
        assert ep1["winner"] == ep2["winner"]

    def test_seed_diversity(self):
        openings = {tuple(s["action_id"] for s in self._episode(seed)["samples"][:6]) for seed in range(8)}
        assert len(openings) >= 7

    @variants
    def test_ply_cap_goes_to_the_only_player_with_the_fewest_bullheads(self, game):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=game, seed=5, model_path=get_test_model(game), simulations=4, max_game_plies=12)
        assert ep["total_plies"] == 12
        _assert_result(ep, VARIANTS[game], finished=False)


# MCTS / arena
class TestMcts:

    @variants
    def test_apply_ai_action_advances_state(self, game):
        gs = dinoboard_engine.GameSession(game, seed=42, model_path=get_test_model(game))
        before = gs.get_state_dict()
        gs.apply_ai_action(simulations=10, temperature=0.0)
        assert gs.get_state_dict() != before

    def test_search_does_not_change_the_session(self):
        gs = dinoboard_engine.GameSession(GAME, seed=7, model_path=get_test_model(GAME))
        for _ in range(3):
            gs.apply_action(gs.get_legal_actions()[0])
        before = (gs.get_state_dict(), gs.get_legal_actions(), list(gs._encode(gs.current_player)["features"]))
        first = gs.get_ai_action(simulations=40, temperature=0.0)["action"]
        assert (gs.get_state_dict(), gs.get_legal_actions(), list(gs._encode(gs.current_player)["features"])) == before
        assert gs.get_ai_action(simulations=40, temperature=0.0)["action"] == first

    @variants
    def test_search_runs_to_the_end_of_the_game(self, game):
        """In the last rounds every simulation reaches the end of the game inside the search,
        where the margin is the value. The session must come out untouched and legal."""
        model = get_test_model(game)
        searched = 0
        for seed in range(21, 31):
            gs = dinoboard_engine.GameSession(game, seed=seed, model_path=model)
            while not gs.is_terminal:
                d = gs.get_state_dict()
                if d["current_round"] >= 8 and not d["small_phase_active"] and len(gs.get_legal_actions()) > 1:
                    before = gs.get_state_dict()
                    result = gs.get_ai_action(simulations=60, temperature=0.0)
                    assert gs.get_state_dict() == before and result["action"] in gs.get_legal_actions()
                    assert result["stats"]["simulations"] > 0
                    searched += 1
                    gs.apply_action(result["action"])
                else:
                    gs.apply_action(gs.get_heuristic_action("rule_v1", 0.0)["action"])
        assert searched >= 10

    @variants
    def test_arena_completes(self, game):
        n = VARIANTS[game]
        m = get_test_model(game)
        result = dinoboard_engine.run_arena_match(
            game_id=game, seed=42, model_paths=[m] * n, simulations_list=[6] * n, temperature=0.0)
        assert result["winner"] in range(-1, n) and result["total_plies"] >= 10 * n

    def test_temperature_adds_variety(self):
        model = get_test_model(GAME)
        actions = set()
        for seed in range(12):
            gs = dinoboard_engine.GameSession(GAME, seed=seed, model_path=model)
            actions.add(gs.get_ai_action(simulations=10, temperature=1.0)["action"])
        assert len(actions) > 1


# 8. Optional components
class TestComponents:

    @pytest.mark.parametrize("name", ["rule_v1", "lkf_v1"])
    @variants
    def test_heuristic_returns_legal_action(self, game, name):
        for seed in range(4):
            gs = dinoboard_engine.GameSession(game, seed=seed)
            for _ in range(8):
                action = gs.get_heuristic_action(name)["action"]
                assert action in gs.get_all_legal_actions()
                gs.apply_action(action)

    def test_no_tail_solver(self):
        with pytest.raises(RuntimeError, match="no tail_solver registered"):
            dinoboard_engine.tail_solve(game_id=GAME, seed=42, perspective_player=0, depth_limit=5,
                                        node_budget=10000)


# 10. AI API: a session fed only observations follows a whole game
PUBLIC_KEYS = ["table", "used", "current_round", "small_phase_active",
               "small_player", "is_terminal", "winner", "current_player"]


class TestApiSeparation:

    @pytest.mark.parametrize("game,perspective,seed", [
        ("6nimmit", 0, 42), ("6nimmit", 1, 43), ("6nimmit", 0, 44), ("6nimmit", 1, 45),
        ("6nimmit_3p", 0, 46), ("6nimmit_3p", 1, 47), ("6nimmit_3p", 2, 48),
        ("6nimmit_4p", 0, 49), ("6nimmit_4p", 2, 50), ("6nimmit_4p", 3, 51)])
    def test_api_session_follows_a_whole_game(self, game, perspective, seed):
        assert_api_belief_matches_selfplay(
            game, public_keys=PUBLIC_KEYS, perspective=perspective, seed_gt=seed, seed_ai=9000 + seed,
            plies=MAX_PLIES, simulations=4)

    @pytest.mark.parametrize("game,me", [("6nimmit", 1), ("6nimmit_3p", 2), ("6nimmit_4p", 1)])
    def test_api_session_sees_what_its_seat_sees_at_every_ply(self, game, me):
        """An observation-only session never runs the rules on hidden cards: the bullheads, the
        cards shown and whose turn it is reach it through the snapshot. (Random play: a seat
        choosing a row to take happens often, and the session must then know who is choosing.)"""
        import random
        pauses = 0
        for seed in range(77, 81):
            rng = random.Random(seed)
            gt = dinoboard_engine.GameSession(game, seed=seed, model_path="", use_filter=False)
            api = dinoboard_engine.GameSession(game, seed=5, model_path="", use_filter=False)
            api.apply_initial_observation(me, gt.extract_initial_observation(me))
            while not gt.is_terminal:
                action = rng.choice(gt.get_legal_actions())
                trace = gt.apply_action_with_trace(action, me)
                api.apply_observation(trace["action"], events=trace["events"],
                                      public_snapshot=trace["public_snapshot"])
                truth, seen = gt.get_state_dict(), api.get_state_dict()
                assert seen["players"][me]["hand"] == truth["players"][me]["hand"]
                assert all(seen[k] == truth[k] for k in PUBLIC_KEYS), [k for k in PUBLIC_KEYS if seen[k] != truth[k]]
                assert [p["hand_heads"] for p in seen["players"]] == [p["hand_heads"] for p in truth["players"]]
                assert api.state_hash_for_perspective(me) == gt.state_hash_for_perspective(me)
                assert api.is_terminal == gt.is_terminal
                if not gt.is_terminal:
                    assert api.current_player == gt.current_player
                    if gt.current_player == me:
                        assert api.get_legal_actions() == gt.get_legal_actions()
                        assert list(api._encode(me)["features"]) == list(gt._encode(me)["features"])
                pauses += truth["small_phase_active"]
        assert pauses >= 3


# 11. Rule invariants (random play; the full rules are in test_rules.py)
class TestRuleInvariants:

    @pytest.mark.parametrize("seed", list(range(3)))
    @variants
    def test_invariants_hold_along_random_episode(self, game, seed):
        n = VARIANTS[game]
        last = None
        for state in run_random_episode_states(game, seed=seed, max_plies=MAX_PLIES):
            assert 0 <= state["current_round"] <= 10
            assert all(p["hand_heads"] >= 0 for p in state["players"])
            if last is not None:
                assert all(p["hand_heads"] >= q["hand_heads"] for p, q in zip(state["players"], last["players"]))
                assert state["current_round"] >= last["current_round"]
            for row in state["table"]:
                cards = [c for c in row if c]
                assert 1 <= len(cards) <= 5 and row == cards + [0] * (5 - len(cards)) and cards == sorted(cards)
            assert 0 <= state["current_player"] < n
            if state["small_phase_active"]:
                assert state["small_player"] == state["current_player"]
            last = state
        assert last["is_terminal"] and last["current_round"] == 10
        assert sum(p["hand_heads"] for p in last["players"]) > 0
