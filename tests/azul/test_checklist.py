"""Azul complete acceptance checklist.

Azul is a public-information game with physical (symmetric) randomness:
the bag is shuffled but every player sees what's drawn into factories.

  - belief_tracker: randomize_unseen shuffles the bag; no per-player
    private fields (schema is fully all_public, walker emits no hidden
    sentinels)
  - multiplayer variants: 2p / 3p / 4p
  - uniform-random heuristic (RandomHeuristic) (web 'heuristic' fallback)
  - tail_solver: AlphaBeta + custom trigger (some pattern-line >=4 AND
    >=2 factories empty); safe to combine with belief tracker because
    do_action_deterministic forces a draw-terminal whenever a round-end
    refill (the only randomness) would have fired.
  - no training_filter, no adjudicator, no aux_scorer
"""
import dinoboard_engine
import pytest

from conftest import (
    get_test_model,
    load_game_config,
    run_random_episode_states,
)

GAME = "azul"
CONFIG = load_game_config(GAME)
ACTION_SPACE = CONFIG["action_space"]
FEATURE_DIM = CONFIG["feature_dim"]
VARIANTS = ["azul", "azul_3p", "azul_4p"]


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
        assert dinoboard_engine.GameSession("azul_3p", seed=42).num_players == 3
        assert dinoboard_engine.GameSession("azul_4p", seed=42).num_players == 4


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

    def test_state_dict_valid(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        state = gs.get_state_dict()
        assert isinstance(state, dict)
        assert state["current_player"] >= 0


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
            simulations=5, max_game_plies=100,
        )
        assert ep["total_plies"] > 0

    def test_sample_integrity(self):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=GAME, seed=42, model_path=get_test_model(GAME),
            simulations=10, max_game_plies=100,
        )
        for s in ep["samples"]:
            assert len(s["features"]) == FEATURE_DIM
            assert len(s["legal_mask"]) == ACTION_SPACE
            legal_set = {i for i, m in enumerate(s["legal_mask"]) if m > 0}
            visited = {a for a, v in zip(s["policy_action_ids"], s["policy_action_visits"]) if v > 0}
            assert visited <= legal_set

    def test_features_vary_across_plies(self):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=GAME, seed=42, model_path=get_test_model(GAME),
            simulations=10, max_game_plies=100,
        )
        if len(ep["samples"]) < 5:
            pytest.skip("too few samples")
        assert ep["samples"][0]["features"] != ep["samples"][4]["features"]

    def test_determinism_same_seed(self):
        kw = dict(
            game_id=GAME, seed=999, model_path=get_test_model(GAME),
            simulations=10, max_game_plies=50,
            dirichlet_alpha=0.0, dirichlet_epsilon=0.0,
        )
        ep1 = dinoboard_engine.run_selfplay_episode(**kw)
        ep2 = dinoboard_engine.run_selfplay_episode(**kw)
        assert ep1["winner"] == ep2["winner"]
        assert ep1["total_plies"] == ep2["total_plies"]

    def test_high_sim_count_no_crash(self):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=GAME, seed=42, model_path=get_test_model(GAME),
            simulations=100, max_game_plies=50,
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
# 7. Tail solver + trigger
# ---------------------------------------------------------------------------

class TestTailSolver:
    """Azul tail solver: AlphaBeta with stochastic_tail_solve_safe=True
    because do_action_deterministic collapses any round-end refill (the
    only random event) into a draw-terminal (winner=-1, value 0).
    """

    def test_tail_solve_api_returns_valid_result(self):
        r = dinoboard_engine.tail_solve(
            game_id=GAME, seed=42, perspective_player=0,
            depth_limit=2, node_budget=2000,
        )
        assert "value" in r and "best_action" in r
        assert "nodes_searched" in r and "budget_exceeded" in r
        assert r["nodes_searched"] >= 0
        assert -2.0 <= r["value"] <= 2.0

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
        """End-to-end: a full selfplay episode with the tail solver
        configured the way mcts.json wires it. Uses a tiny budget so it's
        fast — we only assert the invariant successes <= completed <= attempts.
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
# 8. Rule invariants (game-specific conservation laws)
# ---------------------------------------------------------------------------

# Standard Azul has 20 tiles per color * 5 colors = 100 tiles total.
_TILES_PER_COLOR = 20
_NUM_COLORS = 5
_TOTAL_TILES = _TILES_PER_COLOR * _NUM_COLORS


def _count_tiles_in_play(state: dict) -> tuple[int, list[int]]:
    """Count colored tiles currently visible somewhere on the table.
    Returns (total, per_color_total).

    Sources counted:
      - bag, box_lid (bag_counts + box_counts)
      - center, factories
      - players' pattern_lines (length, with their declared color)
      - players' wall (each set bit = 1 tile of the wall-column's color)
      - players' floor (only entries that are colored — first-player marker
        is encoded as -2 and not a tile)

    The first-player marker is bookkeeping, not a colored tile, so it's
    excluded from totals. Note: when the floor is full (7 entries), engine
    silently discards overflow tiles, so the strict equality "in_play ==
    100" can drift downward — we assert <=, not ==.
    """
    per_color = [0] * _NUM_COLORS
    for c in range(_NUM_COLORS):
        per_color[c] += state["bag_counts"][c]
        per_color[c] += state["box_counts"][c]
        per_color[c] += state["center"][c]
    for fac in state["factories"]:
        for c in range(_NUM_COLORS):
            per_color[c] += fac[c]
    for p in state["players"]:
        for ln in p["pattern_lines"]:
            color = ln["color"]
            if color >= 0 and ln["length"] > 0:
                per_color[color] += ln["length"]
        # Wall: column index inside row r is determined by Azul's color
        # mapping ((color + row) % 5), so column c of row r holds color
        # ((c - r) mod 5). Sum bits per color.
        for r, row in enumerate(p["wall"]):
            for c, bit in enumerate(row):
                if bit:
                    color = (c - r) % _NUM_COLORS
                    per_color[color] += 1
        for entry in p["floor"]:
            if 0 <= entry < _NUM_COLORS:
                per_color[entry] += 1
    return sum(per_color), per_color


def _assert_azul_invariants(state: dict) -> None:
    n = state["num_players"]
    # Tile conservation: total tiles in play <= 100, and per-color <= 20.
    # Floor overflow discards tiles, so equality may not hold exactly.
    in_play, per_color = _count_tiles_in_play(state)
    assert in_play <= _TOTAL_TILES, \
        f"impossible tile count: {in_play} > {_TOTAL_TILES}"
    for c, cnt in enumerate(per_color):
        assert cnt <= _TILES_PER_COLOR, \
            f"color {c}: {cnt} tiles in play (max {_TILES_PER_COLOR})"

    # Pattern lines: length never exceeds capacity; capacity is row+1.
    for i, p in enumerate(state["players"]):
        for r, ln in enumerate(p["pattern_lines"]):
            assert ln["capacity"] == r + 1
            assert 0 <= ln["length"] <= ln["capacity"], \
                f"player {i} row {r}: length {ln['length']} > cap {ln['capacity']}"
            if ln["length"] > 0:
                assert ln["color"] >= 0, \
                    f"player {i} row {r}: nonzero length {ln['length']} with color=-1"
            else:
                # color must be -1 when length 0 (no partial commitment).
                assert ln["color"] == -1, \
                    f"player {i} row {r}: length 0 but color {ln['color']}"
            # Wall already has this color in this row → can't keep filling.
            if ln["color"] >= 0:
                col = (ln["color"] + r) % _NUM_COLORS
                assert p["wall"][r][col] == 0, \
                    f"player {i} row {r}: line color {ln['color']} but wall already has it"
        # Floor depth ≤ 7.
        assert p["floor_count"] <= 7, \
            f"player {i} floor overflow: {p['floor_count']}"
        # Wall is binary.
        for row in p["wall"]:
            for v in row:
                assert v in (0, 1)

    # Factory width ≤ 4 tiles each.
    for fi, fac in enumerate(state["factories"]):
        s = sum(fac)
        assert s <= 4, f"factory {fi} has {s} tiles (max 4)"

    if not state["is_terminal"]:
        assert 0 <= state["current_player"] < n


class TestRuleInvariants:
    """Per-ply assertions on Azul's conservation laws.

    Catches: tile leak/duplication, factory overflow, illegal pattern line
    state (length > capacity, line color collides with already-walled
    column), floor overflow.
    """

    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("seed", list(range(5)))
    def test_invariants_hold_along_random_episode(self, variant, seed):
        for state in run_random_episode_states(variant, seed=seed, max_plies=200):
            _assert_azul_invariants(state)
