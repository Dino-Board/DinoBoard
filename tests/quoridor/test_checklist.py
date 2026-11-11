"""Quoridor complete acceptance checklist.

Quoridor is the deterministic full-information carrier for the framework.
Beyond the basics (state/rules/encoder), it exercises:
  - tail_solver + tail_solve_trigger (alpha-beta endgame solver, gated on
    short-shortest-path)
  - heuristics 命名池：simple_max（path-diff argmax baseline）+ rule_mcts_v1
    (B-flavor PV+MCTS，web hint 用，prior 来自 gorisanson-style PVEvaluator)
  - training_action_filter (cuts low-quality wall placements)
  - adjudicator (winner assignment when max_game_plies hit)
  - episode_stats_extractor (custom_stats in episode result)

A new game touching ANY of these optional components should mirror the
relevant tests in its own checklist; this file stays authoritative for
quoridor regardless of which features other games adopt.
"""
import dinoboard_engine
import pytest

from conftest import (
    get_test_model,
    load_game_config,
    run_random_episode_states,
)

GAME = "quoridor"
CONFIG = load_game_config(GAME)
ACTION_SPACE = CONFIG["action_space"]
FEATURE_DIM = CONFIG["feature_dim"]


# ---------------------------------------------------------------------------
# 1. Registration & config
# ---------------------------------------------------------------------------

class TestRegistration:

    def test_game_registered(self):
        assert GAME in dinoboard_engine.available_games()

    def test_metadata_matches_config(self):
        meta = dinoboard_engine.game_metadata(GAME)
        assert meta["action_space"] == ACTION_SPACE
        assert meta["feature_dim"] == FEATURE_DIM
        assert meta["num_players"] == 2


# ---------------------------------------------------------------------------
# 2. GameSession / action info
# ---------------------------------------------------------------------------

class TestGameSession:

    def test_session_constructs(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        assert gs.num_players == 2
        assert not gs.is_terminal

    def test_legal_actions_in_range(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        legal = gs.get_legal_actions()
        assert len(legal) > 0
        assert all(0 <= a < ACTION_SPACE for a in legal)

    def test_action_info_for_legal_actions(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        for action in gs.get_legal_actions()[:20]:
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


# ---------------------------------------------------------------------------
# 4. Selfplay
# ---------------------------------------------------------------------------

class TestSelfplay:

    def test_episode_completes(self):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=GAME, seed=42, model_path=get_test_model(GAME),
            simulations=10, max_game_plies=80,
        )
        assert ep["total_plies"] > 0

    def test_sample_integrity(self):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=GAME, seed=42, model_path=get_test_model(GAME),
            simulations=10, max_game_plies=50,
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
            simulations=10, max_game_plies=40,
            dirichlet_alpha=0.0, dirichlet_epsilon=0.0,
        )
        ep1 = dinoboard_engine.run_selfplay_episode(**kw)
        ep2 = dinoboard_engine.run_selfplay_episode(**kw)
        assert ep1["winner"] == ep2["winner"]
        assert ep1["total_plies"] == ep2["total_plies"]


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
# 6. Heuristic picker (real heuristic, must always be legal)
# ---------------------------------------------------------------------------

class TestHeuristic:

    def test_heuristic_returns_legal_action(self):
        for seed in range(50):
            gs = dinoboard_engine.GameSession(GAME, seed=seed)
            result = gs.get_heuristic_action("simple_max")
            assert "action" in result
            legal = gs.get_all_legal_actions()
            assert result["action"] in legal, (
                f"seed {seed}: heuristic action {result['action']} not legal"
            )


# ---------------------------------------------------------------------------
# 7. Tail solver + trigger
# ---------------------------------------------------------------------------

class TestTailSolver:

    def test_tail_solve_api_works(self):
        r = dinoboard_engine.tail_solve(
            game_id=GAME, seed=42, perspective_player=0,
            depth_limit=1, node_budget=100,
        )
        assert "value" in r and "best_action" in r

    def test_auxiliary_weight_runs_without_error(self):
        for w in (0.0, 0.01):
            ep = dinoboard_engine.run_selfplay_episode(
                game_id=GAME, seed=42, model_path=get_test_model(GAME),
                simulations=10, max_game_plies=200,
                tail_solve_enabled=True,
                tail_solve_depth_limit=3, tail_solve_node_budget=500,
                tail_solve_auxiliary_weight=w,
            )
            assert ep["total_plies"] > 0

    def test_configure_tail_solve_via_session(self):
        m = get_test_model(GAME)
        gs = dinoboard_engine.GameSession(GAME, seed=42, model_path=m)
        gs.configure_tail_solve(True, 5, 1000)
        result = gs.get_ai_action(simulations=10, temperature=0.0)
        stats = result["stats"]
        assert "tail_solved" in stats and "tail_solve_value" in stats

    def test_configure_tail_solve_disabled(self):
        m = get_test_model(GAME)
        gs = dinoboard_engine.GameSession(GAME, seed=42, model_path=m)
        gs.configure_tail_solve(False, 5, 1000)
        result = gs.get_ai_action(simulations=10, temperature=0.0)
        assert result["stats"]["tail_solved"] is False


# ---------------------------------------------------------------------------
# 8. Adjudicator (max_game_plies → winner assigned)
# ---------------------------------------------------------------------------

class TestAdjudicator:

    def test_adjudicated_z_is_valid(self):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=GAME, seed=42, model_path=get_test_model(GAME),
            simulations=10, max_game_plies=5,
        )
        if ep["total_plies"] >= 5 and not any(s.get("z_values") for s in ep["samples"]):
            for s in ep["samples"]:
                assert s["z"] in (-1.0, 0.0, 1.0), f"bad z={s['z']}"


# ---------------------------------------------------------------------------
# 9. Custom stats
# ---------------------------------------------------------------------------

class TestCustomStats:

    def test_custom_stats_dict(self):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=GAME, seed=42, model_path=get_test_model(GAME),
            simulations=10, max_game_plies=30,
        )
        assert isinstance(ep.get("custom_stats", {}), dict)


# ---------------------------------------------------------------------------
# 10. Training action filter (cuts low-quality walls)
# ---------------------------------------------------------------------------

class TestTrainingFilter:

    def test_filter_reduces_action_count(self):
        gs_f = dinoboard_engine.GameSession(GAME, seed=42, use_filter=True)
        gs_u = dinoboard_engine.GameSession(GAME, seed=42, use_filter=False)
        f = len(gs_f.get_legal_actions())
        u = len(gs_u.get_legal_actions())
        assert f < u, f"filter should reduce actions: {f} vs {u}"
        assert set(gs_f.get_legal_actions()).issubset(set(gs_u.get_legal_actions()))


# ---------------------------------------------------------------------------
# 11. Rule invariants (game-specific conservation laws)
# ---------------------------------------------------------------------------

def _bfs_reaches(start_row, start_col, goal_row, n, blocked):
    """Simplified reachability ignoring pawn jumps. Returns True if a path
    exists from start to any cell on goal_row, blocked according to
    `blocked(r, c, dr, dc) -> bool` which decides if (r,c)→(r+dr,c+dc) is blocked.
    Pawn jumps over the opponent are a *bonus* mobility — if we can reach
    the goal without any jump, the post-wall-placement legality also holds.
    """
    from collections import deque
    seen = {(start_row, start_col)}
    q = deque([(start_row, start_col)])
    while q:
        r, c = q.popleft()
        if r == goal_row:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n and 0 <= nc < n):
                continue
            if (nr, nc) in seen:
                continue
            if blocked(r, c, dr, dc):
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return False


def _make_blocker(h_walls, v_walls, n):
    """Translate wall lists into an edge-block test.
    Wall convention (matches engine): horizontal wall at (row, col) blocks
    (row, col)<->(row+1, col) and (row, col+1)<->(row+1, col+1). Vertical
    wall at (row, col) blocks (row, col)<->(row, col+1) and (row+1, col)<->
    (row+1, col+1).
    """
    h_set = {(w["row"], w["col"]) for w in h_walls}
    v_set = {(w["row"], w["col"]) for w in v_walls}

    def blocked(r, c, dr, dc):
        if dr == 1:  # moving south r->r+1 at col c
            if (r, c) in h_set or (r, c - 1) in h_set:
                return True
        elif dr == -1:  # north
            if (r - 1, c) in h_set or (r - 1, c - 1) in h_set:
                return True
        elif dc == 1:  # east c->c+1
            if (r, c) in v_set or (r - 1, c) in v_set:
                return True
        elif dc == -1:  # west
            if (r, c - 1) in v_set or (r - 1, c - 1) in v_set:
                return True
        return False
    return blocked


def _assert_quoridor_invariants(state: dict) -> None:
    n = state["board_size"]
    pawns = state["pawns"]
    walls_left = state["walls_remaining"]
    h_walls = state["horizontal_walls"]
    v_walls = state["vertical_walls"]

    # Two pawns on the board, each in bounds, not stacked.
    assert len(pawns) == len(walls_left) == 2
    positions = []
    for i, p in enumerate(pawns):
        assert p["player"] == i
        assert 0 <= p["row"] < n and 0 <= p["col"] < n, \
            f"pawn {i} off-board: {p}"
        positions.append((p["row"], p["col"]))
    if not state["is_terminal"]:
        assert positions[0] != positions[1], "pawns occupy same cell"

    # Wall counts within [0, 10]; total walls placed + remaining == 20 (per pair).
    for i, w in enumerate(walls_left):
        assert 0 <= w <= 10, f"player {i} walls_remaining out of range: {w}"
    placed = len(h_walls) + len(v_walls)
    total_remaining = sum(walls_left)
    assert placed + total_remaining == 20, \
        f"wall conservation broken: placed={placed} remaining={total_remaining}"

    # Walls in legal coordinate range (groove indices: 0..n-2 for both).
    for w in h_walls + v_walls:
        assert 0 <= w["row"] <= n - 2 and 0 <= w["col"] <= n - 2, \
            f"wall coord out of range: {w}"

    # Both players still have a path to their goal row (Quoridor's
    # canonical legality invariant — wall placements that would block any
    # pawn are illegal, so we should never observe a state violating this).
    if not state["is_terminal"]:
        block = _make_blocker(h_walls, v_walls, n)
        for i, p in enumerate(pawns):
            goal = state["goal_rows"][i]
            assert _bfs_reaches(p["row"], p["col"], goal, n, block), \
                f"player {i} has no path to goal row {goal}; walls={h_walls}+{v_walls}"


class TestRuleInvariants:
    """Per-ply assertions on conservation laws specific to Quoridor.

    Catches: wall over-placement, pawn off-board, illegal pawn stacking,
    and (most importantly) wall placements that would cut off a player —
    Quoridor's canonical no-fence-imprisonment rule.
    """

    @pytest.mark.parametrize("seed", list(range(20)))
    def test_invariants_hold_along_random_episode(self, seed):
        for state in run_random_episode_states(GAME, seed=seed, max_plies=120):
            _assert_quoridor_invariants(state)
