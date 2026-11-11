"""Splendor complete acceptance checklist.

Splendor is a hidden-information game with:
  - belief_tracker (deck contents are hidden, but card pool is public)
  - tail_solver + tail_solve_trigger (any player >=10 points)
  - multiplayer variants: 2p / 3p / 4p
  - uniform-random heuristic (RandomHeuristic) (web 'heuristic' fallback)
  - no real heuristic, no training_filter, no adjudicator, no aux_scorer
"""
import dinoboard_engine
import pytest

from conftest import (
    assert_api_belief_matches_selfplay,
    get_test_model,
    load_game_config,
    run_random_episode_states,
)

GAME = "splendor"
CONFIG = load_game_config(GAME)
ACTION_SPACE = CONFIG["action_space"]
FEATURE_DIM = CONFIG["feature_dim"]
VARIANTS = ["splendor", "splendor_3p", "splendor_4p"]


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
        assert dinoboard_engine.GameSession("splendor_3p", seed=42).num_players == 3
        assert dinoboard_engine.GameSession("splendor_4p", seed=42).num_players == 4


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

    def test_action_info(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        for action in gs.get_legal_actions()[:20]:
            info = gs.get_action_info(action)
            assert isinstance(info, dict) and len(info) > 0

    def test_state_dict_has_basics(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        state = gs.get_state_dict()
        assert "current_player" in state
        assert "is_terminal" in state


# ---------------------------------------------------------------------------
# 3. Feature encoding
# ---------------------------------------------------------------------------

class TestEncoder:

    def test_encode_state_correct_dim(self):
        info = dinoboard_engine.encode_state(GAME, seed=42)
        assert len(info["features"]) == FEATURE_DIM
        assert len(info["legal_mask"]) == ACTION_SPACE


# ---------------------------------------------------------------------------
# 4. Selfplay (2p, 3p, 4p)
# ---------------------------------------------------------------------------

class TestSelfplay:

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_episode_completes(self, variant):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=variant, seed=42, model_path=get_test_model(variant),
            simulations=10, max_game_plies=50,
        )
        assert ep["total_plies"] > 0

    def test_sample_integrity(self):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=GAME, seed=42, model_path=get_test_model(GAME),
            simulations=10, max_game_plies=60,
        )
        for s in ep["samples"]:
            assert len(s["features"]) == FEATURE_DIM
            assert len(s["legal_mask"]) == ACTION_SPACE
            legal_set = {i for i, m in enumerate(s["legal_mask"]) if m > 0}
            visited = {a for a, v in zip(s["policy_action_ids"], s["policy_action_visits"]) if v > 0}
            assert visited <= legal_set, (
                f"ply {s['ply']}: visited illegal actions {visited - legal_set}"
            )

    def test_features_vary_across_plies(self):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=GAME, seed=42, model_path=get_test_model(GAME),
            simulations=10, max_game_plies=60,
        )
        if len(ep["samples"]) < 5:
            pytest.skip("too few samples")
        assert ep["samples"][0]["features"] != ep["samples"][4]["features"]

    def test_determinism_same_seed(self):
        kw = dict(
            game_id=GAME, seed=12345, model_path=get_test_model(GAME),
            simulations=10, max_game_plies=30,
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
                simulations=10, max_game_plies=30,
            )
            ep2 = dinoboard_engine.run_selfplay_episode(
                game_id=GAME, seed=i + 10000, model_path=get_test_model(GAME),
                simulations=10, max_game_plies=30,
            )
            if [s["action_id"] for s in ep1["samples"]] != [s["action_id"] for s in ep2["samples"]]:
                diff += 1
        assert diff > 10, f"only {diff}/20 splendor games differed across distant seeds"

    def test_high_sim_count_no_crash(self):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=GAME, seed=42, model_path=get_test_model(GAME),
            simulations=100, max_game_plies=30,
        )
        assert ep["total_plies"] > 0


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
# 6. Heuristic (uniform-random fallback)
# ---------------------------------------------------------------------------

class TestHeuristic:

    def test_heuristic_returns_legal_action(self):
        for seed in range(20):
            gs = dinoboard_engine.GameSession(GAME, seed=seed)
            result = gs.get_heuristic_action("rule_v1")
            assert "action" in result
            assert result["action"] in gs.get_all_legal_actions()


# ---------------------------------------------------------------------------
# 7. Tail solver
# ---------------------------------------------------------------------------

class TestTailSolver:
    """Splendor tail solver: AlphaBeta with stochastic_tail_solve_safe=True
    because SplendorRules::do_action_deterministic forces the tier-replenish
    step to skip drawing the hidden deck (forced_draw_override = -2). The
    solver only sees the current public state — never samples / branches over
    hidden information.
    """

    def test_tail_solve_api(self):
        r = dinoboard_engine.tail_solve(
            game_id=GAME, seed=42, perspective_player=0,
            depth_limit=2, node_budget=1000,
        )
        assert "value" in r and "budget_exceeded" in r

    def test_tail_solve_tiny_budget_exceeds(self):
        r = dinoboard_engine.tail_solve(
            game_id=GAME, seed=42, perspective_player=0,
            depth_limit=20, node_budget=5,
        )
        assert r["budget_exceeded"], "tiny budget should exceed"

    def test_configure_tail_solve_via_session(self):
        """Drive a session with tail_solve enabled — get_ai_action returns
        tail_solve stats fields whether or not the trigger fires."""
        m = get_test_model(GAME)
        gs = dinoboard_engine.GameSession(GAME, seed=42, model_path=m)
        gs.configure_tail_solve(True, 5, 1000)
        result = gs.get_ai_action(simulations=10, temperature=0.0)
        stats = result["stats"]
        assert "tail_solved" in stats and "tail_solve_value" in stats

    def test_configure_tail_solve_disabled_never_fires(self):
        m = get_test_model(GAME)
        gs = dinoboard_engine.GameSession(GAME, seed=42, model_path=m)
        gs.configure_tail_solve(False, 5, 1000)
        result = gs.get_ai_action(simulations=10, temperature=0.0)
        assert result["stats"]["tail_solved"] is False

    def test_selfplay_with_tail_solve_runs(self):
        """End-to-end: full selfplay episode with the tail solver wired the
        way mcts.json does. Tiny budget so it's fast — only assert the
        invariant successes <= completed <= attempts.
        """
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=GAME, seed=42, model_path=get_test_model(GAME),
            simulations=10, max_game_plies=200,
            tail_solve_enabled=True,
            tail_solve_depth_limit=2, tail_solve_node_budget=500,
        )
        a = ep["tail_solve_attempts"]
        c = ep["tail_solve_completed"]
        s = ep["tail_solve_successes"]
        assert s <= c <= a, f"invariant violated: {s} <= {c} <= {a}"


# ---------------------------------------------------------------------------
# 8. Hidden info: belief tracker on deck
# ---------------------------------------------------------------------------

class TestBeliefTracker:

    def test_randomized_decks_differ_from_original(self):
        r = dinoboard_engine.test_belief_tracker(GAME, seed=42, plies=20, randomize_trials=10)
        assert r["plies"] > 0
        orig = sorted(r["original_deck"])
        diffs = sum(1 for td in r["trial_decks"] if sorted(td) != orig)
        assert diffs > 0, "All trials matched original deck — tracker may be peeking"

    def test_tableau_cards_never_in_randomized_deck(self):
        r = dinoboard_engine.test_belief_tracker(GAME, seed=123, plies=15, randomize_trials=10)
        tab = set(r["tableau_cards"])
        for i, td in enumerate(r["trial_decks"]):
            overlap = set(td) & tab
            assert not overlap, f"Trial {i}: tableau cards {overlap} in randomized deck"

    def test_deck_size_preserved(self):
        r = dinoboard_engine.test_belief_tracker(GAME, seed=99, plies=15, randomize_trials=10)
        orig_len = len(r["original_deck"])
        for i, td in enumerate(r["trial_decks"]):
            assert len(td) == orig_len, (
                f"Trial {i}: deck size {len(td)} != original {orig_len}"
            )

    def test_randomization_has_variance(self):
        r = dinoboard_engine.test_belief_tracker(GAME, seed=77, plies=10, randomize_trials=10)
        unique = {tuple(td) for td in r["trial_decks"]}
        assert len(unique) > 1, "All trials produced identical decks — no randomization"

    def test_no_duplicate_cards(self):
        r = dinoboard_engine.test_belief_tracker(GAME, seed=55, plies=15, randomize_trials=5)
        for i, td in enumerate(r["trial_decks"]):
            assert len(td) == len(set(td)), f"Trial {i}: duplicate cards {td}"


# ---------------------------------------------------------------------------
# 9. AI API belief / public state / legal actions equivalence
# (independent-seed API session must match self-play tracker step-by-step)
# ---------------------------------------------------------------------------

class TestApiBeliefEquivalence:

    PUBLIC_KEYS = [
        "current_player", "is_terminal", "winner", "num_players",
        "bank", "tableau", "nobles", "players",
    ]

    def test_belief_matches_selfplay_under_independent_seed(self):
        assert_api_belief_matches_selfplay(GAME, self.PUBLIC_KEYS)


# ---------------------------------------------------------------------------
# 10. Rule invariants (game-specific conservation laws)
# ---------------------------------------------------------------------------

# Per-variant bank stocks: 4 colored gems in 2p, 5 in 3p, 7 in 4p; gold = 5 always.
_BANK_PER_COLOR = {2: 4, 3: 5, 4: 7}
_GOLD_TOTAL = 5


def _assert_splendor_invariants(state: dict) -> None:
    n = state["num_players"]
    bank = state["bank"]
    assert len(bank) == 6, f"bank must have 6 token types, got {len(bank)}"
    expected_per_color = _BANK_PER_COLOR[n]

    # Token conservation: bank + sum_p(player_gems) == initial supply.
    # Splendor never destroys or creates tokens — buys return them to the bank.
    for color in range(5):
        in_play = bank[color] + sum(p["gems"][color] for p in state["players"])
        assert in_play == expected_per_color, (
            f"color {color} token count broken: bank={bank[color]} "
            f"players={[p['gems'][color] for p in state['players']]} "
            f"total={in_play} expected={expected_per_color}")
    in_play_gold = bank[5] + sum(p["gems"][5] for p in state["players"])
    assert in_play_gold == _GOLD_TOTAL, \
        f"gold token count broken: total={in_play_gold} expected={_GOLD_TOTAL}"

    # Per-player limits.
    for i, p in enumerate(state["players"]):
        # Reserved slots ≤ 3.
        assert len(p["reserved"]) <= 3, \
            f"player {i} has {len(p['reserved'])} reserved cards (max 3)"
        # gems ≥ 0 each color, gold ≥ 0.
        for color, g in enumerate(p["gems"]):
            assert g >= 0, f"player {i} negative gems[{color}] = {g}"
        # bonuses ≥ 0 and == cards_count summed.
        for color, b in enumerate(p["bonuses"]):
            assert b >= 0, f"player {i} negative bonus[{color}] = {b}"
        bonus_total = sum(p["bonuses"])
        assert bonus_total == p["cards_count"], (
            f"player {i} cards_count {p['cards_count']} != sum(bonuses) {bonus_total}")
        # points ≥ 0.
        assert p["points"] >= 0, f"player {i} negative points {p['points']}"

    # Tableau width: each tier has at most 4 visible cards.
    for tier_idx, row in enumerate(state["tableau"]):
        assert len(row) <= 4, f"tier {tier_idx + 1} has {len(row)} cards (max 4)"

    # Nobles: at most 5 (which is per-rules max for 4p), points always 3 in
    # bookkeeping (no need to overconstrain — just sanity).
    assert len(state["nobles"]) <= 5, f"too many nobles: {len(state['nobles'])}"

    # current_player in range when not terminal.
    if not state["is_terminal"]:
        assert 0 <= state["current_player"] < n


class TestRuleInvariants:
    """Per-ply assertions on Splendor's conservation laws.

    Catches: token leaks (bank + player gems != initial), reserved slot
    overflow, bonus / cards_count drift, negative resources.
    """

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_invariants_hold_along_random_episode(self, seed):
        for state in run_random_episode_states(GAME, seed=seed, max_plies=200):
            _assert_splendor_invariants(state)
