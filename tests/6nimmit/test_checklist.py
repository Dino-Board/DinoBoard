"""6 nimmt! complete acceptance checklist.

6 nimmt! is a public-board / private-hand commit-and-reveal game with:
  - belief_tracker with per-player private hand (owner_only viz)
  - public_event_extractor / applier (events_only_diff is empty —
    `viz::no_events_extractor`; the wholesale snapshot half is the
    framework's `viz::serialize_public_snapshot` / `apply_public_snapshot`)
  - simultaneous commits realized as N sequential commit plies + one
    SMALL_CHOOSE_ROW ply per "card-too-small" event
  - multiplayer variants: 2p / 3p / 4p
  - 10 fixed rounds; deterministic post-processing of the ascending
    apply queue
  - heuristic `rule_v1` is the ported lyc_pbh_wzq.py decision tree
    (NOT uniform-random)
  - no tail_solver, no training_filter, no adjudicator, no aux_scorer

Rule invariants (104 distinct cards, head_value formula) live at the
bottom; per-card-type conservation collapses to "the 104-card multiset
is partitioned across {hands, table, bullhead piles, unseen pool}."
"""
from __future__ import annotations

import sys
from pathlib import Path

import dinoboard_engine
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from conftest import get_test_model, load_game_config, run_random_episode_states

GAME = "6nimmit"
CONFIG = load_game_config(GAME)
ACTION_SPACE = CONFIG["action_space"]
FEATURE_DIM = CONFIG["feature_dim"]
VARIANTS = ["6nimmit", "6nimmit_3p", "6nimmit_4p"]


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
        assert dinoboard_engine.GameSession("6nimmit_3p", seed=42).num_players == 3
        assert dinoboard_engine.GameSession("6nimmit_4p", seed=42).num_players == 4

    def test_feature_dim_per_variant(self):
        # 127 + N*107 (bullhead pile one-hot dropped → counts only;
        # pending_commits added as 1 hidden flag + 104 one-hot per seat)
        assert dinoboard_engine.game_metadata("6nimmit_3p")["feature_dim"] == 127 + 3 * 107
        assert dinoboard_engine.game_metadata("6nimmit_4p")["feature_dim"] == 127 + 4 * 107


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

    def test_initial_legal_actions_are_hand_commits(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        st = gs.get_state_dict()
        cp = st["current_player"]
        hand = set(st["players"][cp]["hand"])
        legal = set(gs.get_legal_actions())
        # action_id ∈ [0,103] is COMMIT_CARD with card = action_id+1.
        legal_cards = {a + 1 for a in legal}
        assert legal_cards == hand, (
            f"initial legal commits should equal hand, got {legal_cards} vs {hand}")

    def test_action_info(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        for action in gs.get_legal_actions()[:10]:
            info = gs.get_action_info(action)
            assert isinstance(info, dict) and len(info) > 0


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

    def test_seed_diversity(self):
        diff = 0
        for i in range(20):
            ep1 = dinoboard_engine.run_selfplay_episode(
                game_id=GAME, seed=i, model_path=get_test_model(GAME),
                simulations=10, max_game_plies=60,
            )
            ep2 = dinoboard_engine.run_selfplay_episode(
                game_id=GAME, seed=i + 10000, model_path=get_test_model(GAME),
                simulations=10, max_game_plies=60,
            )
            if [s["action_id"] for s in ep1["samples"]] != [s["action_id"] for s in ep2["samples"]]:
                diff += 1
        assert diff > 10, f"only {diff}/20 6nimmit games differed across distant seeds"


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

    def test_dag_reuse_hits_present(self):
        m = get_test_model(GAME)
        gs = dinoboard_engine.GameSession(GAME, seed=42, model_path=m)
        result = gs.get_ai_action(simulations=20, temperature=0.0)
        assert "dag_reuse_hits" in result["stats"]


# ---------------------------------------------------------------------------
# 6. Heuristic (ported lyc_pbh_wzq decision tree)
# ---------------------------------------------------------------------------

class TestHeuristic:

    def test_heuristic_returns_legal_action(self):
        for seed in range(20):
            gs = dinoboard_engine.GameSession(GAME, seed=seed)
            if gs.is_terminal:
                continue
            result = gs.get_heuristic_action("rule_v1")
            assert "action" in result
            assert result["action"] in gs.get_all_legal_actions()


# ---------------------------------------------------------------------------
# 7. Components NOT registered
# ---------------------------------------------------------------------------

class TestUnsupportedComponents:

    def test_no_tail_solver(self):
        with pytest.raises(RuntimeError, match="no tail_solver registered"):
            dinoboard_engine.tail_solve(
                game_id=GAME, seed=42, perspective_player=0,
                depth_limit=5, node_budget=10000,
            )


# ---------------------------------------------------------------------------
# 8. Temperature variety
# ---------------------------------------------------------------------------

class TestTemperature:

    def test_nonzero_temperature_adds_variety(self):
        actions = set()
        model = get_test_model(GAME)
        for seed in range(20):
            gs = dinoboard_engine.GameSession(GAME, seed=seed, model_path=model)
            if gs.is_terminal:
                continue
            result = gs.get_ai_action(simulations=10, temperature=1.0)
            if "action" in result:
                actions.add(result["action"])
        assert len(actions) > 1, "temperature=1.0 always picked the same action across seeds"


# ---------------------------------------------------------------------------
# 9. Round/termination invariant
# ---------------------------------------------------------------------------

class TestRoundProgression:
    """After every random episode, current_round must reach kNumRounds=10
    for terminal=True. apply_cursor must reset between rounds."""

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_terminal_at_round_10(self, variant):
        for seed in range(5):
            states = list(run_random_episode_states(variant, seed=seed, max_plies=200))
            terminal = states[-1]
            assert terminal["is_terminal"], (
                f"{variant} seed={seed} did not reach terminal in 200 plies")
            assert terminal["current_round"] == 10, (
                f"{variant} seed={seed} terminal at round={terminal['current_round']}")


# ---------------------------------------------------------------------------
# 10. Rule invariants (game-specific conservation laws)
# ---------------------------------------------------------------------------

# The 104-card deck is a fixed multiset {1, 2, ..., 104}. Each card appears
# in exactly one of: a player's hand, the table, a player's bullhead pile,
# the (private/observer-side) "unseen pool". The rule-invariant test
# asserts that {hand[*]} ∪ {table} ∪ {bullhead_pile[*]} is a set of
# DISTINCT cards drawn from {1..104}, each appearing at most once across
# all containers (since the unseen pool absorbs the rest).

_HEAD_VALUE_TABLE = {}
for _c in range(1, 105):
    _h = 0
    if _c % 5 == 0: _h += 2
    if _c % 10 == 0: _h += 1
    if _c % 11 == 0: _h += 5
    if _h == 0: _h = 1
    _HEAD_VALUE_TABLE[_c] = _h


def _assert_sixnimmit_invariants(state: dict) -> None:
    n = state["num_players"]
    # 1. Distinctness: every visible card appears at most once across all
    #    containers (hands + table + bullhead piles).
    seen: dict[int, str] = {}
    for i, p in enumerate(state["players"]):
        for c in p["hand"]:
            assert 1 <= c <= 104, f"player {i} hand has out-of-range card {c}"
            assert c not in seen, f"card {c} duplicated: {seen[c]} and player {i} hand"
            seen[c] = f"player {i} hand"
    for r, row in enumerate(state["table"]):
        for c in row:
            if c == 0:
                continue
            assert 1 <= c <= 104, f"table[{r}] has out-of-range card {c}"
            assert c not in seen, f"card {c} duplicated: {seen[c]} and table row {r}"
            seen[c] = f"table row {r}"
    for i, p in enumerate(state["players"]):
        for c in p["bullhead_pile"]:
            assert 1 <= c <= 104, f"player {i} bullhead has out-of-range card {c}"
            assert c not in seen, f"card {c} duplicated: {seen[c]} and player {i} bullhead"
            seen[c] = f"player {i} bullhead"

    # 2. bullhead_count must equal sum of head_value over each player's pile.
    for i, p in enumerate(state["players"]):
        expected = sum(_HEAD_VALUE_TABLE[c] for c in p["bullhead_pile"])
        assert p["bullhead_count"] == expected, (
            f"player {i} bullhead_count={p['bullhead_count']} != sum(head_value)={expected}")

    # 3. Round must be in [0, 10]; 10 implies terminal.
    rd = state["current_round"]
    assert 0 <= rd <= 10, f"current_round out of range: {rd}"
    if rd == 10:
        assert state["is_terminal"], f"current_round=10 but not terminal"

    # 4. Table rows: each row has length ≤ 5 (capacity); the visible
    #    representation may pad with 0s, count non-zero entries instead.
    for r, row in enumerate(state["table"]):
        non_zero = [c for c in row if c != 0]
        assert len(non_zero) <= 5, (
            f"table row {r} length {len(non_zero)} exceeds capacity 5")
        # Trailing zeros only — once a 0 appears, the rest must be 0.
        zero_started = False
        for c in row:
            if c == 0:
                zero_started = True
            else:
                assert not zero_started, f"table row {r} has 0 before card {c}"

    # 5. hand_count must equal len(hand).
    for i, p in enumerate(state["players"]):
        assert p["hand_count"] == len(p["hand"]), (
            f"player {i} hand_count={p['hand_count']} != len(hand)={len(p['hand'])}")
        # Hands are at most kHandSize=10, decreasing by ≤ 1 per round.
        assert p["hand_count"] <= 10

    # 6. current_player in range; on non-terminal, current_player < n.
    cp = state["current_player"]
    assert 0 <= cp < n
    if state["small_phase_active"]:
        assert state["small_player"] == cp, (
            f"small_phase_active but current_player={cp} != small_player={state['small_player']}")

    # 7. apply_cursor in [0, n].
    assert 0 <= state["apply_cursor"] <= n


class TestRuleInvariants:
    """Per-ply assertions on 6 nimmt!'s conservation laws."""

    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("seed", list(range(5)))
    def test_invariants_hold_along_random_episode(self, variant, seed):
        for state in run_random_episode_states(variant, seed=seed, max_plies=200):
            _assert_sixnimmit_invariants(state)
