"""Liar's Dice complete acceptance checklist.

Liar's Dice is a hidden-info game with:
  - belief_tracker (per-player owner-only dice; bids public)
  - public_event_extractor / applier — events stream is empty
    (`viz::no_events_extractor`); the wholesale snapshot half is the
    framework's `viz::serialize_public_snapshot` / `apply_public_snapshot`
  - per-face belief target per opponent: 6 faces × 6 count bins (n_f ∈ [0, 5]),
    flat one-hot length 36 per opp; joint 252-Σ=5 distribution is reconstructed
    at sample time as ∏_f p(n_f|f), renormalized on the 252 enumerated outcomes
  - multiplayer variants: 2p / 3p / 4p
  - single-round game: terminal on any challenge
  - no tail_solver, no training_filter, no adjudicator, no aux_scorer

Action layout (per variant):
  action_id ∈ [0, 5N*5)  → bid:  count = action_id // 5 + 1 (1..5N)
                                  face  = action_id  % 5 + 2 (2..6)
  action_id == 5N*5      → challenge

Rule invariants:
  1. Σ_f dice_count[p][f] == 5 for every player (single round, no dice loss)
  2. Bid strict monotonicity: each new bid strictly dominates the prior one
  3. Challenge only legal once a bid is on the table
  4. On terminal, winner is in [0, N) and all dice are revealed
"""
from __future__ import annotations

import sys
from pathlib import Path

import dinoboard_engine
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from conftest import get_test_model, load_game_config, run_random_episode_states

GAME = "liarsdice"
CONFIG = load_game_config(GAME)
ACTION_SPACE = CONFIG["action_space"]
FEATURE_DIM = CONFIG["feature_dim"]
VARIANTS = ["liarsdice", "liarsdice_3p", "liarsdice_4p"]


def _bid_strictly_greater(x_new, y_new, x_old, y_old):
    if x_new > x_old:
        return True
    if x_new == x_old and y_new > y_old:
        return True
    return False


# ---------------------------------------------------------------------------
# 1. Registration & config
# ---------------------------------------------------------------------------

class TestRegistration:

    def test_2p_registered(self):
        assert GAME in dinoboard_engine.available_games()

    def test_all_variants_registered(self):
        games = dinoboard_engine.available_games()
        for v in VARIANTS:
            assert v in games, f"{v} not registered"

    def test_metadata_matches_config(self):
        meta = dinoboard_engine.game_metadata(GAME)
        assert meta["action_space"] == ACTION_SPACE
        assert meta["feature_dim"] == FEATURE_DIM
        assert meta["num_players"] == 2

    def test_player_counts(self):
        assert dinoboard_engine.GameSession("liarsdice_3p", seed=42).num_players == 3
        assert dinoboard_engine.GameSession("liarsdice_4p", seed=42).num_players == 4

    def test_action_space_per_variant(self):
        # 5N * 5 + 1
        assert dinoboard_engine.game_metadata("liarsdice_2p")["action_space"] == 10 * 5 + 1
        assert dinoboard_engine.game_metadata("liarsdice_3p")["action_space"] == 15 * 5 + 1
        assert dinoboard_engine.game_metadata("liarsdice_4p")["action_space"] == 20 * 5 + 1

    def test_feature_dim_per_variant(self):
        # own dice(6) + cur_count(1) + cur_face one-hot(6) + N * 5 * 6 bid block
        assert dinoboard_engine.game_metadata("liarsdice_2p")["feature_dim"] == 13 + 2 * 5 * 6
        assert dinoboard_engine.game_metadata("liarsdice_3p")["feature_dim"] == 13 + 3 * 5 * 6
        assert dinoboard_engine.game_metadata("liarsdice_4p")["feature_dim"] == 13 + 4 * 5 * 6

    def test_belief_logit_count_per_variant(self):
        # Per-opp 6 faces × 6 count bins (n_f ∈ [0, 5]) = 36; total (N-1) * 36.
        assert dinoboard_engine.game_metadata("liarsdice_2p")["belief_logit_count"] == 1 * 36
        assert dinoboard_engine.game_metadata("liarsdice_3p")["belief_logit_count"] == 2 * 36
        assert dinoboard_engine.game_metadata("liarsdice_4p")["belief_logit_count"] == 3 * 36

    def test_belief_feature_dim_per_variant(self):
        # bid history block only: N * 5 * 6
        for v, n in [("liarsdice_2p", 2), ("liarsdice_3p", 3), ("liarsdice_4p", 4)]:
            assert dinoboard_engine.game_metadata(v)["belief_feature_dim"] == n * 5 * 6


# ---------------------------------------------------------------------------
# 2. GameSession basics
# ---------------------------------------------------------------------------

class TestGameSession:

    def test_session_constructs(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        assert not gs.is_terminal

    def test_legal_actions_in_range(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        legal = gs.get_legal_actions()
        assert len(legal) > 0
        assert all(0 <= a < ACTION_SPACE for a in legal)

    def test_first_move_excludes_challenge(self):
        # No prior bid → challenge illegal; every bid is legal.
        for variant in VARIANTS:
            gs = dinoboard_engine.GameSession(variant, seed=42)
            meta = dinoboard_engine.game_metadata(variant)
            n_bid_actions = meta["action_space"] - 1
            challenge_action = n_bid_actions
            legal = sorted(gs.get_legal_actions())
            assert challenge_action not in legal, (
                f"{variant}: challenge legal on first move")
            assert legal == list(range(n_bid_actions)), (
                f"{variant}: first-move legal != all bids")

    def test_challenge_legal_after_first_bid(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        gs.apply_action(0)  # bid X=1, Y=2
        legal = gs.get_legal_actions()
        challenge_action = ACTION_SPACE - 1
        assert challenge_action in legal

    def test_action_info(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        for action in gs.get_legal_actions()[:10]:
            info = gs.get_action_info(action)
            assert isinstance(info, dict) and len(info) > 0
            assert info["type"] in ("bid", "challenge")
            if info["type"] == "bid":
                assert 1 <= info["count"]
                assert 2 <= info["face"] <= 6

    def test_action_info_layout_bid(self):
        # action 0 → count=1, face=2;  action 1 → count=1, face=3;  ...
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        info0 = gs.get_action_info(0)
        assert info0 == {"action_id": 0, "type": "bid", "count": 1, "face": 2}
        info5 = gs.get_action_info(5)
        assert info5 == {"action_id": 5, "type": "bid", "count": 2, "face": 2}

    def test_action_info_challenge(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        info = gs.get_action_info(ACTION_SPACE - 1)
        assert info["type"] == "challenge"


# ---------------------------------------------------------------------------
# 3. Feature encoding
# ---------------------------------------------------------------------------

class TestEncoder:

    def test_encode_state_correct_dim(self):
        info = dinoboard_engine.encode_state(GAME, seed=42)
        assert len(info["features"]) == FEATURE_DIM
        assert len(info["legal_mask"]) == ACTION_SPACE

    def test_features_have_known_dim_after_play(self):
        gs = dinoboard_engine.GameSession(GAME, seed=300)
        if gs.is_terminal:
            return
        enc = dinoboard_engine.encode_state(GAME, seed=300)
        assert len(enc["features"]) == enc["feature_dim"]


# ---------------------------------------------------------------------------
# 4. Selfplay (2p, 3p, 4p)
# ---------------------------------------------------------------------------

class TestSelfplay:

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_episode_completes(self, variant):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=variant, seed=42, model_path=get_test_model(variant),
            simulations=10, max_game_plies=80,
        )
        assert ep["total_plies"] > 0

    def test_sample_integrity(self):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=GAME, seed=42, model_path=get_test_model(GAME),
            simulations=10, max_game_plies=80,
        )
        for s in ep["samples"]:
            assert len(s["features"]) == FEATURE_DIM
            assert len(s["legal_mask"]) == ACTION_SPACE
            legal_set = {i for i, m in enumerate(s["legal_mask"]) if m > 0}
            visited = {a for a, v in zip(s["policy_action_ids"], s["policy_action_visits"]) if v > 0}
            assert visited <= legal_set

    def test_determinism_same_seed(self):
        kw = dict(
            game_id=GAME, seed=12345, model_path=get_test_model(GAME),
            simulations=10, max_game_plies=80,
            dirichlet_alpha=0.0, dirichlet_epsilon=0.0,
        )
        ep1 = dinoboard_engine.run_selfplay_episode(**kw)
        ep2 = dinoboard_engine.run_selfplay_episode(**kw)
        assert ep1["winner"] == ep2["winner"]
        assert ep1["total_plies"] == ep2["total_plies"]


# ---------------------------------------------------------------------------
# 4b. Belief label structure (per-face one-hot, 6×6 = 36)
# ---------------------------------------------------------------------------

class TestBeliefLabel:
    """Belief target is per-face one-hot: 6 faces × 6 count bins = 36.
    Each opp row in hand_counts is a flat length-36 vector consisting of
    6 concatenated 6-way one-hots (slot f * 6 + n_f = 1). Σ_f n_f = 5.
    """

    @pytest.mark.parametrize("variant,n", [
        ("liarsdice_2p", 2), ("liarsdice_3p", 3), ("liarsdice_4p", 4)])
    def test_belief_label_per_face_one_hot(self, variant, n):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=variant, seed=42, model_path=get_test_model(variant),
            simulations=10, max_game_plies=80,
        )
        assert ep["belief_label_class_count"] == 36
        bs = ep["belief_samples"]
        assert len(bs) > 0, f"{variant}: no belief samples emitted"
        F, C = 6, 6
        for s in bs:
            rows = s["hand_counts"]
            assert len(rows) == n - 1, (
                f"{variant}: expected {n-1} opp rows, got {len(rows)}")
            for i, row in enumerate(rows):
                assert len(row) == 36, (
                    f"{variant}: opp {i} row length {len(row)} != 36")
                # Each face block must be a 6-way one-hot.
                total_dice = 0
                for f in range(F):
                    block = row[f * C : (f + 1) * C]
                    ones = sum(1 for v in block if v == 1)
                    others = sum(1 for v in block if v != 0 and v != 1)
                    assert others == 0, (
                        f"{variant}: opp {i} face {f} non-{{0,1}} entries")
                    assert ones == 1, (
                        f"{variant}: opp {i} face {f} not one-hot, ones={ones}")
                    # Recover n_f and accumulate.
                    n_f = next(c for c, v in enumerate(block) if v == 1)
                    total_dice += n_f
                assert total_dice == 5, (
                    f"{variant}: opp {i} dice sum {total_dice} != 5")


# ---------------------------------------------------------------------------
# 5. MCTS / arena
# ---------------------------------------------------------------------------

class TestMcts:

    def test_apply_ai_action_advances_state(self):
        m = get_test_model(GAME)
        gs = dinoboard_engine.GameSession(GAME, seed=42, model_path=m)
        before = gs.get_state_dict()
        gs.apply_ai_action(simulations=10, temperature=0.0)
        after = gs.get_state_dict()
        assert before != after

    def test_arena_completes(self):
        m = get_test_model(GAME)
        result = dinoboard_engine.run_arena_match(
            game_id=GAME, seed=42, model_paths=[m, m],
            simulations_list=[10, 10], temperature=0.0,
        )
        assert "winner" in result and result["total_plies"] > 0


# ---------------------------------------------------------------------------
# 6. Components NOT registered
# ---------------------------------------------------------------------------

class TestUnsupportedComponents:

    def test_no_tail_solver(self):
        with pytest.raises(RuntimeError, match="no tail_solver registered"):
            dinoboard_engine.tail_solve(
                game_id=GAME, seed=42, perspective_player=0,
                depth_limit=5, node_budget=10000,
            )


# ---------------------------------------------------------------------------
# 7. Rule invariants
# ---------------------------------------------------------------------------

def _assert_liarsdice_invariants(state: dict, *, prev_bid: tuple) -> tuple:
    """Per-ply assertions; returns the updated (prev_count, prev_face) tuple
    for the next call."""
    n = state["num_players"]

    # 1. current_player in range.
    cp = state["current_player"]
    assert 0 <= cp < n

    # 2. dice_count: every visible row sums to 5 (single-round, no dice loss);
    #    hidden rows (dice_count empty / not visible) are not asserted.
    for i, p in enumerate(state["players"]):
        dc = p["dice_count"]
        if dc and any(c >= 0 for c in dc):
            # All face counts non-negative.
            assert all(c >= 0 for c in dc), (
                f"player {i} dice_count has negatives: {dc}")
            assert sum(dc) == 5, (
                f"player {i} dice_count sums to {sum(dc)} != 5: {dc}")

    # 3. current_bid_count / current_bid_face consistency.
    cur_count = state["current_bid_count"]
    cur_face = state["current_bid_face"]
    if cur_count == 0:
        assert cur_face == 0, (
            f"cur_count=0 but cur_face={cur_face}")
    else:
        assert 1 <= cur_count <= 5 * n, (
            f"cur_count {cur_count} out of [1, {5 * n}]")
        assert 2 <= cur_face <= 6, (
            f"cur_face {cur_face} out of [2, 6]")

    # 4. Bid strict monotonicity: if a previous bid existed, this one must
    #    strictly dominate it (or the state is terminal post-challenge, in
    #    which case the bid stays at the last value but no further test
    #    applies).
    if prev_bid != (0, 0) and (cur_count, cur_face) != prev_bid and not state["is_terminal"]:
        assert _bid_strictly_greater(cur_count, cur_face, *prev_bid), (
            f"new bid ({cur_count}, {cur_face}) does not strictly dominate "
            f"prev ({prev_bid[0]}, {prev_bid[1]})")

    # 5. Terminal: winner in [0, N).
    if state["is_terminal"]:
        assert 0 <= state["winner"] < n, (
            f"terminal but winner={state['winner']}")
        # All dice revealed on terminal — owner-only viz flips public on
        # challenge resolution, so the GT serializer's `dice` list is
        # non-empty for every seat.
        for i, p in enumerate(state["players"]):
            assert p.get("dice"), (
                f"terminal but player {i} dice not revealed")

    return (cur_count, cur_face)


class TestRuleInvariants:

    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("seed", list(range(5)))
    def test_invariants_hold_along_random_episode(self, variant, seed):
        prev_bid = (0, 0)
        for state in run_random_episode_states(variant, seed=seed, max_plies=200):
            prev_bid = _assert_liarsdice_invariants(state, prev_bid=prev_bid)

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_episode_terminates_via_challenge(self, variant):
        for seed in range(5):
            states = list(run_random_episode_states(variant, seed=seed, max_plies=200))
            terminal = states[-1]
            assert terminal["is_terminal"], (
                f"{variant} seed={seed} did not reach terminal in 200 plies")
