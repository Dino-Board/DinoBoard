import pytest

import dinoboard_engine

from conftest import get_test_model, load_game_config, run_random_episode_states

from . import carc_reference as ref

GAME = "carcassonne"
CONFIG = load_game_config(GAME)
ACTION_SPACE = CONFIG["action_space"]
FEATURE_DIM = CONFIG["feature_dim"]
ROWS = 20
COLS = 20
TILE_TYPES = 24
HISTORY_SLOTS = 72
HISTORY_FEATURE_DIM = 43
PLACE_ACTION_COUNT = ROWS * COLS * 4
MEEPLE_PASS_ACTION = PLACE_ACTION_COUNT
MEEPLE_SIDE_ACTION_BASE = MEEPLE_PASS_ACTION + 1
MEEPLE_FARM_ACTION_BASE = MEEPLE_SIDE_ACTION_BASE + 4
MEEPLE_CENTER_ACTION = MEEPLE_FARM_ACTION_BASE + 8
STAGE_PLACE_TILE = 0
STAGE_MEEPLE = 1
TILE_A_CHAPEL_ROAD = 0
TILE_B_CHAPEL = 1
TILE_C_FULL_CITY = 2
TILE_D_START = 3
TILE_E_CITY_TOP = 4
TILE_G_CITY_NARROW = 6
TILE_J_CITY_TOP_ROAD_BEND_RIGHT = 9
TILE_L_CITY_TOP_CROSSROADS = 11
TILE_N_CITY_DIAGONAL_TOP_RIGHT = 13
TILE_U_STRAIGHT_ROAD = 20
TILE_W_THREE_SPLIT_ROAD = 22
TILE_X_CROSSROADS = 23


def _state(gs):
    return gs.get_state_dict()


def _place_action(row, col, rotation):
    return (row * COLS + col) * 4 + rotation


def _position_from_meeple_action(action):
    if MEEPLE_SIDE_ACTION_BASE <= action < MEEPLE_SIDE_ACTION_BASE + 4:
        return action - MEEPLE_SIDE_ACTION_BASE
    if MEEPLE_FARM_ACTION_BASE <= action < MEEPLE_FARM_ACTION_BASE + 8:
        return 4 + action - MEEPLE_FARM_ACTION_BASE
    if action == MEEPLE_CENTER_ACTION:
        return 12
    raise ValueError(f"not a meeple action: {action}")


def _clear_board(gs):
    for cell in range(ROWS * COLS):
        gs._force_write_field("board_tile", [cell], -1)
        gs._force_write_field("board_rotation", [cell], 0)
        gs._force_write_field("board_meeple_owner", [cell], -1)
        gs._force_write_field("board_meeple_pos", [cell], -1)
        gs._force_write_field("final_meeple_scores", [cell], 0)
    for p in range(2):
        gs._force_write_field("scores", [p], 0)
        gs._force_write_field("final_score_deltas", [p], 0)
        gs._force_write_field("meeples_remaining", [p], 7)
    gs._force_write_field("current_player", [], 0)
    gs._force_write_field("winner", [], -1)
    gs._force_write_field("terminal", [], False)
    gs._force_write_field("shared_victory", [], False)
    gs._force_write_field("plies", [], 0)
    gs._force_write_field("stage", [], STAGE_PLACE_TILE)
    gs._force_write_field("last_row", [], -1)
    gs._force_write_field("last_col", [], -1)
    gs._force_write_field("last_rotation", [], 0)
    gs._force_write_field("last_score_delta", [], 0)
    for tile in range(TILE_TYPES):
        gs._force_write_field("discarded_tile_counts", [tile], 0)
        gs._force_write_field("bag_counts", [tile], 0)
    gs._force_write_field("deck_count", [], 0)
    for slot in range(HISTORY_SLOTS):
        gs._force_write_field("history_tile", [slot], -1)
        gs._force_write_field("history_rotation", [slot], 0)
        gs._force_write_field("history_row", [slot], -1)
        gs._force_write_field("history_col", [slot], -1)


def _arm_next_draw(gs):
    # Force the next draw_next_tile to weighted-sample TILE_B_CHAPEL by
    # making it the only tile left in the bag. (Bag is count-only — the
    # rng's weighted pick over a singleton multiset is deterministic.)
    for tile in range(TILE_TYPES):
        gs._force_write_field("bag_counts", [tile], 0)
    gs._force_write_field("bag_counts", [TILE_B_CHAPEL], 1)
    gs._force_write_field("deck_count", [], 1)


def _cell(row, col):
    return row * COLS + col


def _occupied_cells(state):
    return [idx for idx, tile in enumerate(state["board_tile"]) if tile >= 0]


def _occupied_bbox(state):
    cells = _occupied_cells(state)
    rows = [cell // COLS for cell in cells]
    cols = [cell % COLS for cell in cells]
    return min(rows), max(rows), min(cols), max(cols)


def _assert_board_centered(state):
    min_row, max_row, min_col, max_col = _occupied_bbox(state)
    assert min_row == (ROWS - (max_row - min_row + 1) + 1) // 2
    assert min_col == (COLS - (max_col - min_col + 1) + 1) // 2


def _meeple_cell(state, owner, pos):
    for cell, (cell_owner, cell_pos) in enumerate(
            zip(state["board_meeple_owner"], state["board_meeple_pos"])):
        if cell_owner == owner and cell_pos == pos:
            return cell
    raise AssertionError(f"meeple owner={owner} pos={pos} not found")


class TestRegistration:

    def test_game_registered(self):
        assert GAME in dinoboard_engine.available_games()

    def test_metadata_matches_config(self):
        meta = dinoboard_engine.game_metadata(GAME)
        assert PLACE_ACTION_COUNT == 1600
        assert meta["action_space"] == PLACE_ACTION_COUNT + 14
        assert meta["action_space"] == ACTION_SPACE
        assert meta["feature_dim"] == FEATURE_DIM
        assert meta["num_players"] == 2


class TestGameSession:

    def test_session_constructs(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        state = _state(gs)
        center = (ROWS // 2) * COLS + (COLS // 2)
        assert state["board_tile"][center] >= 0
        assert not state["is_terminal"]

    def test_legal_actions_are_tile_placements(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        legal = gs.get_legal_actions()
        assert legal
        assert all(0 <= a < PLACE_ACTION_COUNT for a in legal)
        for action in legal[:40]:
            cell = action // 4
            assert 0 <= cell < ROWS * COLS
            assert action % 4 in (0, 1, 2, 3)

    def test_tile_placement_actions_are_filtered_by_board_rules(self):
        gs = dinoboard_engine.GameSession(GAME, seed=2)
        legal = set(gs.get_legal_actions())
        center_row = ROWS // 2
        center_col = COLS // 2
        legal_cells = {action // 4 for action in legal}
        adjacent_cells = {
            (center_row - 1) * COLS + center_col,
            center_row * COLS + center_col + 1,
            (center_row + 1) * COLS + center_col,
            center_row * COLS + center_col - 1,
        }

        assert 0 < len(legal) < PLACE_ACTION_COUNT
        assert legal_cells <= adjacent_cells
        assert _place_action(0, 0, 0) not in legal
        assert _place_action(center_row, center_col, 0) not in legal

        with pytest.raises(ValueError, match="illegal carcassonne tile action"):
            gs.apply_action(_place_action(0, 0, 0))
        with pytest.raises(ValueError, match="illegal carcassonne tile action"):
            gs.apply_action(_place_action(center_row, center_col, 0))

    def test_touching_edges_must_match_terrain(self):
        gs = dinoboard_engine.GameSession(GAME, seed=2)
        legal = set(gs.get_legal_actions())
        action = _place_action(10, 9, 0)
        assert action not in legal
        with pytest.raises(ValueError, match="illegal carcassonne tile action"):
            gs.apply_action(action)

    def test_all_tile_actions_match_legal_filter(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        legal = set(gs.get_legal_actions())
        assert 0 < len(legal) < PLACE_ACTION_COUNT
        for action in range(PLACE_ACTION_COUNT):
            if action in legal:
                info = gs.get_action_info(action)
                assert info["type"] == "place_tile"
                continue
            with pytest.raises(ValueError, match="illegal carcassonne tile action"):
                clone = dinoboard_engine.GameSession(GAME, seed=42)
                clone.apply_action(action)

    def test_tile_phase_never_exposes_meeple_actions(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        legal = set(gs.get_legal_actions())
        assert legal
        assert all(action < PLACE_ACTION_COUNT for action in legal)
        for action in range(PLACE_ACTION_COUNT, ACTION_SPACE):
            assert action not in legal
            with pytest.raises(ValueError, match="illegal carcassonne tile action"):
                gs.apply_action(action)

    def test_action_info_for_each_legal_action(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        for action in gs.get_legal_actions()[:30]:
            info = gs.get_action_info(action)
            assert info["type"] == "place_tile"
            assert 0 <= info["row"] < ROWS
            assert 0 <= info["col"] < COLS
            assert 0 <= info["rotation"] < 4

    def test_place_then_pass_completes_turn(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        before = _state(gs)
        action = gs.get_legal_actions()[0]
        actor = before["current_player"]
        gs.apply_action(action)
        placed = _state(gs)
        cell = action // 4
        assert placed["board_tile"][cell] == before["current_tile"]
        assert placed["board_rotation"][cell] == action % 4
        assert placed["board_meeple_owner"][cell] == -1
        assert placed["board_meeple_pos"][cell] == -1
        assert placed["current_player"] == actor
        assert placed["stage"] == STAGE_MEEPLE
        assert placed["plies"] == before["plies"]
        assert MEEPLE_PASS_ACTION in gs.get_legal_actions()

        gs.apply_action(MEEPLE_PASS_ACTION)
        after = _state(gs)
        assert after["current_player"] == 1 - actor
        assert after["stage"] == STAGE_PLACE_TILE
        assert after["plies"] == before["plies"] + 1

    def test_board_recenters_after_each_action_and_shifts_history(self):
        gs = dinoboard_engine.GameSession(GAME, seed=1)
        _clear_board(gs)
        center = _cell(10, 10)
        gs._force_write_field("board_tile", [center], TILE_D_START)
        gs._force_write_field("history_tile", [0], TILE_D_START)
        gs._force_write_field("history_row", [0], 10)
        gs._force_write_field("history_col", [0], 10)
        gs._force_write_field("current_tile", [], TILE_A_CHAPEL_ROAD)
        _arm_next_draw(gs)

        gs.apply_action(_place_action(10, 11, 1))
        placed = _state(gs)
        _assert_board_centered(placed)
        assert placed["last_row"] == 10
        assert placed["last_col"] == 10
        assert placed["history_row"][:2] == [10, 10]
        assert placed["history_col"][:2] == [9, 10]

        gs.apply_action(MEEPLE_PASS_ACTION)
        after = _state(gs)
        _assert_board_centered(after)
        assert after["history_row"][:2] == [10, 10]
        assert after["history_col"][:2] == [9, 10]

    def test_meeple_actions_use_second_phase_ids(self):
        gs = dinoboard_engine.GameSession(GAME, seed=68)
        gs.apply_action(839)
        legal = gs.get_legal_actions()
        assert MEEPLE_PASS_ACTION in legal
        assert all(a == MEEPLE_PASS_ACTION or PLACE_ACTION_COUNT < a < ACTION_SPACE
                   for a in legal)
        info = gs.get_action_info(MEEPLE_PASS_ACTION)
        assert info["type"] == "meeple_pass"
        seen_kinds = set()
        for action in legal:
            if action == MEEPLE_PASS_ACTION:
                continue
            info = gs.get_action_info(action)
            assert info["type"] == "place_meeple"
            assert 0 <= info["position"] <= 12
            seen_kinds.add(info["kind"])
        assert {"feature", "farm", "cloister"} <= seen_kinds

    def test_meeple_can_only_be_placed_on_last_tile(self):
        gs = dinoboard_engine.GameSession(GAME, seed=68)
        gs.apply_action(839)
        state = _state(gs)
        legal = [a for a in gs.get_legal_actions() if a != MEEPLE_PASS_ACTION]
        assert legal

        action = legal[0]
        pos = _position_from_meeple_action(action)
        cell = state["last_row"] * COLS + state["last_col"]
        gs.apply_action(action)
        after = _state(gs)
        assert after["board_meeple_owner"][cell] == state["current_player"]
        assert after["board_meeple_pos"][cell] == pos
        assert sum(1 for owner in after["board_meeple_owner"] if owner >= 0) == 1

    def test_meeple_actions_are_not_legal_during_tile_phase(self):
        gs = dinoboard_engine.GameSession(GAME, seed=68)
        legal = set(gs.get_legal_actions())
        assert MEEPLE_PASS_ACTION not in legal
        assert all(a < PLACE_ACTION_COUNT for a in legal)
        with pytest.raises(ValueError, match="illegal carcassonne tile action"):
            gs.apply_action(MEEPLE_PASS_ACTION)

    def test_meeple_phase_never_exposes_tile_actions(self):
        gs = dinoboard_engine.GameSession(GAME, seed=68)
        gs.apply_action(839)
        legal = set(gs.get_legal_actions())
        assert legal
        assert all(action >= PLACE_ACTION_COUNT for action in legal)
        for action in range(PLACE_ACTION_COUNT):
            assert action not in legal
            with pytest.raises(ValueError, match="illegal carcassonne meeple action"):
                clone = dinoboard_engine.GameSession(GAME, seed=68)
                clone.apply_action(839)
                clone.apply_action(action)

    def test_farm_meeple_legality_crosses_matching_half_edges(self):
        gs = dinoboard_engine.GameSession(GAME, seed=1)
        _clear_board(gs)

        upper = 9 * COLS + 10
        lower = 10 * COLS + 10
        tile_s_city_bottom_road = 18
        tile_u_straight_road = 20
        brb_farm_position = 4 + 5

        gs._force_write_field("board_tile", [upper], tile_s_city_bottom_road)
        gs._force_write_field("board_meeple_owner", [upper], 1)
        gs._force_write_field("board_meeple_pos", [upper], brb_farm_position)
        gs._force_write_field("board_tile", [lower], tile_u_straight_road)
        gs._force_write_field("last_row", [], 10)
        gs._force_write_field("last_col", [], 10)
        gs._force_write_field("stage", [], STAGE_MEEPLE)
        gs._force_write_field("current_player", [], 0)

        legal = set(gs.get_legal_actions())
        assert MEEPLE_PASS_ACTION in legal
        assert MEEPLE_FARM_ACTION_BASE + 2 not in legal

    def test_same_empty_farm_is_compressed_to_one_action(self):
        gs = dinoboard_engine.GameSession(GAME, seed=1)
        _clear_board(gs)
        center = _cell(10, 10)
        gs._force_write_field("board_tile", [center], TILE_B_CHAPEL)
        gs._force_write_field("last_row", [], 10)
        gs._force_write_field("last_col", [], 10)
        gs._force_write_field("stage", [], STAGE_MEEPLE)
        gs._force_write_field("current_player", [], 0)

        legal = set(gs.get_legal_actions())
        farm_actions = {
            action for action in legal
            if MEEPLE_FARM_ACTION_BASE <= action < MEEPLE_CENTER_ACTION
        }
        assert farm_actions == {MEEPLE_FARM_ACTION_BASE}

    def test_same_road_feature_is_compressed_to_one_action(self):
        gs = dinoboard_engine.GameSession(GAME, seed=1)
        _clear_board(gs)
        center = _cell(10, 10)
        gs._force_write_field("board_tile", [center], TILE_D_START)
        gs._force_write_field("last_row", [], 10)
        gs._force_write_field("last_col", [], 10)
        gs._force_write_field("stage", [], STAGE_MEEPLE)
        gs._force_write_field("current_player", [], 0)

        legal = set(gs.get_legal_actions())
        assert MEEPLE_SIDE_ACTION_BASE + 1 in legal
        assert MEEPLE_SIDE_ACTION_BASE + 3 not in legal

    def test_three_split_road_exposes_three_road_meeple_actions(self):
        gs = dinoboard_engine.GameSession(GAME, seed=1)
        _clear_board(gs)
        center = _cell(10, 10)
        gs._force_write_field("board_tile", [center], TILE_W_THREE_SPLIT_ROAD)
        gs._force_write_field("last_row", [], 10)
        gs._force_write_field("last_col", [], 10)
        gs._force_write_field("stage", [], STAGE_MEEPLE)
        gs._force_write_field("current_player", [], 0)

        legal = set(gs.get_legal_actions())
        road_actions = {
            action for action in legal
            if MEEPLE_SIDE_ACTION_BASE <= action < MEEPLE_FARM_ACTION_BASE
        }
        assert road_actions == {
            MEEPLE_SIDE_ACTION_BASE + 1,
            MEEPLE_SIDE_ACTION_BASE + 2,
            MEEPLE_SIDE_ACTION_BASE + 3,
        }
        farm_actions = {
            action for action in legal
            if MEEPLE_FARM_ACTION_BASE <= action < MEEPLE_CENTER_ACTION
        }
        assert farm_actions == {
            MEEPLE_FARM_ACTION_BASE,
            MEEPLE_FARM_ACTION_BASE + 4,
            MEEPLE_FARM_ACTION_BASE + 6,
        }

    def test_city_top_crossroads_exposes_three_farm_meeple_actions(self):
        gs = dinoboard_engine.GameSession(GAME, seed=1)
        _clear_board(gs)
        center = _cell(10, 10)
        gs._force_write_field("board_tile", [center], TILE_L_CITY_TOP_CROSSROADS)
        gs._force_write_field("last_row", [], 10)
        gs._force_write_field("last_col", [], 10)
        gs._force_write_field("stage", [], STAGE_MEEPLE)
        gs._force_write_field("current_player", [], 0)

        legal = set(gs.get_legal_actions())
        feature_actions = {
            action for action in legal
            if MEEPLE_SIDE_ACTION_BASE <= action < MEEPLE_FARM_ACTION_BASE
        }
        assert feature_actions == {
            MEEPLE_SIDE_ACTION_BASE,
            MEEPLE_SIDE_ACTION_BASE + 1,
            MEEPLE_SIDE_ACTION_BASE + 2,
            MEEPLE_SIDE_ACTION_BASE + 3,
        }
        farm_actions = {
            action for action in legal
            if MEEPLE_FARM_ACTION_BASE <= action < MEEPLE_CENTER_ACTION
        }
        assert farm_actions == {
            MEEPLE_FARM_ACTION_BASE,
            MEEPLE_FARM_ACTION_BASE + 4,
            MEEPLE_FARM_ACTION_BASE + 6,
        }


class TestScoring:

    def test_completed_city_scores_and_returns_meeple(self):
        gs = dinoboard_engine.GameSession(GAME, seed=1)
        _clear_board(gs)
        center = _cell(10, 10)
        gs._force_write_field("board_tile", [center], TILE_D_START)
        gs._force_write_field("board_meeple_owner", [center], 0)
        gs._force_write_field("board_meeple_pos", [center], 0)
        gs._force_write_field("meeples_remaining", [0], 6)
        gs._force_write_field("current_tile", [], TILE_E_CITY_TOP)
        _arm_next_draw(gs)

        gs.apply_action(_place_action(9, 10, 2))
        gs.apply_action(MEEPLE_PASS_ACTION)
        state = _state(gs)
        assert state["scores"] == [4, 0]
        assert state["last_score_delta"] == 4
        assert state["meeples_remaining"][0] == 7
        assert state["board_meeple_owner"][center] == -1
        assert not state["is_terminal"]

    def test_tile_placement_clears_previous_score_delta(self):
        gs = dinoboard_engine.GameSession(GAME, seed=1)
        _clear_board(gs)
        center = _cell(10, 10)
        gs._force_write_field("board_tile", [center], TILE_D_START)
        gs._force_write_field("board_meeple_owner", [center], 0)
        gs._force_write_field("board_meeple_pos", [center], 0)
        gs._force_write_field("meeples_remaining", [0], 6)
        gs._force_write_field("current_tile", [], TILE_E_CITY_TOP)
        _arm_next_draw(gs)

        gs.apply_action(_place_action(9, 10, 2))
        gs.apply_action(MEEPLE_PASS_ACTION)
        scored = _state(gs)
        assert scored["scores"] == [4, 0]
        assert scored["last_score_delta"] == 4

        tile_action = gs.get_legal_actions()[0]
        assert tile_action < PLACE_ACTION_COUNT
        gs.apply_action(tile_action)
        placed = _state(gs)
        assert placed["stage"] == STAGE_MEEPLE
        assert placed["scores"] == [4, 0]
        assert placed["last_score_delta"] == 0

    def test_completed_road_scores_and_returns_meeple(self):
        gs = dinoboard_engine.GameSession(GAME, seed=1)
        _clear_board(gs)
        center = _cell(10, 10)
        right = _cell(10, 11)
        gs._force_write_field("board_tile", [center], TILE_D_START)
        gs._force_write_field("board_meeple_owner", [center], 0)
        gs._force_write_field("board_meeple_pos", [center], 1)
        gs._force_write_field("meeples_remaining", [0], 6)
        gs._force_write_field("board_tile", [right], TILE_A_CHAPEL_ROAD)
        gs._force_write_field("board_rotation", [right], 1)
        gs._force_write_field("current_tile", [], TILE_A_CHAPEL_ROAD)
        _arm_next_draw(gs)

        gs.apply_action(_place_action(10, 9, 3))
        gs.apply_action(MEEPLE_PASS_ACTION)
        state = _state(gs)
        assert state["scores"] == [3, 0]
        assert state["last_score_delta"] == 3
        assert state["meeples_remaining"][0] == 7
        assert state["board_meeple_owner"][center] == -1
        assert not state["is_terminal"]

    def test_completed_road_without_meeple_scores_nothing(self):
        gs = dinoboard_engine.GameSession(GAME, seed=1)
        _clear_board(gs)
        center = _cell(10, 10)
        gs._force_write_field("board_tile", [center], TILE_D_START)
        gs._force_write_field("current_tile", [], TILE_A_CHAPEL_ROAD)
        _arm_next_draw(gs)

        gs.apply_action(_place_action(10, 11, 1))
        gs.apply_action(MEEPLE_PASS_ACTION)
        state = _state(gs)
        assert state["scores"] == [0, 0]
        assert state["last_score_delta"] == 0
        assert state["meeples_remaining"] == [7, 7]
        assert not state["is_terminal"]

    def test_completed_cloister_scores_nine_and_returns_meeple(self):
        gs = dinoboard_engine.GameSession(GAME, seed=1)
        _clear_board(gs)
        center = _cell(10, 10)
        gs._force_write_field("board_tile", [center], TILE_B_CHAPEL)
        gs._force_write_field("board_meeple_owner", [center], 0)
        gs._force_write_field("board_meeple_pos", [center], 12)
        gs._force_write_field("meeples_remaining", [0], 6)
        for row in (9, 10, 11):
            for col in (9, 10, 11):
                if row == 10 and col == 10:
                    continue
                if row == 11 and col == 11:
                    continue
                gs._force_write_field("board_tile", [_cell(row, col)], TILE_B_CHAPEL)
        gs._force_write_field("current_tile", [], TILE_B_CHAPEL)
        _arm_next_draw(gs)

        gs.apply_action(_place_action(11, 11, 0))
        gs.apply_action(MEEPLE_PASS_ACTION)
        state = _state(gs)
        assert state["scores"] == [9, 0]
        assert state["last_score_delta"] == 9
        assert state["meeples_remaining"][0] == 7
        assert state["board_meeple_owner"][center] == -1
        assert not state["is_terminal"]

    def test_final_scoring_incomplete_city_scores_one_per_tile(self):
        gs = dinoboard_engine.GameSession(GAME, seed=1)
        _clear_board(gs)
        center = _cell(10, 10)
        gs._force_write_field("board_tile", [center], TILE_D_START)
        gs._force_write_field("board_meeple_owner", [center], 0)
        gs._force_write_field("board_meeple_pos", [center], 0)
        gs._force_write_field("current_tile", [], TILE_B_CHAPEL)
        gs._force_write_field("deck_count", [], 0)

        gs.apply_action(_place_action(11, 10, 0))
        gs.apply_action(MEEPLE_PASS_ACTION)
        state = _state(gs)
        assert state["is_terminal"]
        assert state["scores"] == [1, 0]
        assert state["final_score_deltas"] == [1, 0]
        assert state["winner"] == 0
        meeple_cell = _meeple_cell(state, 0, 0)
        assert state["final_meeple_scores"][meeple_cell] == 1
        assert sum(state["final_meeple_scores"]) == 1

    def test_final_scoring_farmer_scores_completed_adjacent_city(self):
        gs = dinoboard_engine.GameSession(GAME, seed=1)
        _clear_board(gs)
        center = _cell(10, 10)
        above = _cell(9, 10)
        gs._force_write_field("board_tile", [center], TILE_D_START)
        gs._force_write_field("board_meeple_owner", [center], 0)
        gs._force_write_field("board_meeple_pos", [center], 4)
        gs._force_write_field("board_tile", [above], TILE_E_CITY_TOP)
        gs._force_write_field("board_rotation", [above], 2)
        gs._force_write_field("current_tile", [], TILE_B_CHAPEL)
        gs._force_write_field("deck_count", [], 0)

        gs.apply_action(_place_action(11, 10, 0))
        gs.apply_action(MEEPLE_PASS_ACTION)
        state = _state(gs)
        assert state["is_terminal"]
        assert state["scores"] == [3, 0]
        assert state["final_score_deltas"] == [3, 0]
        assert state["winner"] == 0
        meeple_cell = _meeple_cell(state, 0, 4)
        assert state["final_meeple_scores"][meeple_cell] == 3
        assert sum(state["final_meeple_scores"]) == 3

    def test_final_meeple_scores_are_per_meeple_and_respect_majority(self):
        gs = dinoboard_engine.GameSession(GAME, seed=1)
        _clear_board(gs)
        left = _cell(10, 9)
        center = _cell(10, 10)
        right = _cell(10, 11)
        gs._force_write_field("board_tile", [left], TILE_N_CITY_DIAGONAL_TOP_RIGHT)
        gs._force_write_field("board_meeple_owner", [left], 0)
        gs._force_write_field("board_meeple_pos", [left], 1)
        gs._force_write_field("board_tile", [center], TILE_G_CITY_NARROW)
        gs._force_write_field("board_meeple_owner", [center], 1)
        gs._force_write_field("board_meeple_pos", [center], 3)
        gs._force_write_field("board_tile", [right], TILE_E_CITY_TOP)
        gs._force_write_field("board_rotation", [right], 3)
        gs._force_write_field("board_meeple_owner", [right], 0)
        gs._force_write_field("board_meeple_pos", [right], 3)
        gs._force_write_field("meeples_remaining", [0], 5)
        gs._force_write_field("meeples_remaining", [1], 6)
        gs._force_write_field("current_tile", [], TILE_B_CHAPEL)
        gs._force_write_field("deck_count", [], 0)

        gs.apply_action(_place_action(11, 10, 0))
        gs.apply_action(MEEPLE_PASS_ACTION)
        state = _state(gs)
        assert state["is_terminal"]
        assert state["scores"] == [3, 0]
        assert state["final_score_deltas"] == [3, 0]
        assert state["final_meeple_scores"][_meeple_cell(state, 0, 1)] == 3
        assert state["final_meeple_scores"][_meeple_cell(state, 0, 3)] == 0
        assert state["final_meeple_scores"][_meeple_cell(state, 1, 3)] == 0
        assert sum(state["final_meeple_scores"]) == 3

    def test_final_farm_scoring_ignores_farm_that_touches_no_city(self):
        # The road of D separates its two fields: the farmer above the road
        # borders the city, the one below (in the field shared with the
        # cloister tile) does not.
        gs = dinoboard_engine.GameSession(GAME, seed=1)
        _clear_board(gs)
        farm_one = _cell(10, 10)
        farm_two = _cell(11, 10)
        city = _cell(9, 10)
        gs._force_write_field("board_tile", [farm_one], TILE_D_START)
        gs._force_write_field("board_meeple_owner", [farm_one], 0)
        gs._force_write_field("board_meeple_pos", [farm_one], 4)
        gs._force_write_field("board_tile", [farm_two], TILE_B_CHAPEL)
        gs._force_write_field("board_meeple_owner", [farm_two], 0)
        gs._force_write_field("board_meeple_pos", [farm_two], 6)
        gs._force_write_field("board_tile", [city], TILE_E_CITY_TOP)
        gs._force_write_field("board_rotation", [city], 2)
        gs._force_write_field("meeples_remaining", [0], 5)
        gs._force_write_field("current_tile", [], TILE_B_CHAPEL)
        gs._force_write_field("deck_count", [], 0)

        gs.apply_action(_place_action(12, 10, 0))
        gs.apply_action(MEEPLE_PASS_ACTION)
        state = _state(gs)
        assert state["is_terminal"]
        assert state["scores"] == [3, 0]
        assert state["final_score_deltas"] == [3, 0]
        assert state["final_meeple_scores"][_meeple_cell(state, 0, 4)] == 3
        assert state["final_meeple_scores"][_meeple_cell(state, 0, 6)] == 0
        assert sum(state["final_meeple_scores"]) == 3

    def test_final_farm_scores_once_per_farm_and_shows_one_bubble_per_player(self):
        # One farm (D's strip above its road, continued by the upper field of
        # the road tile to its left) holds two farmers of player 0. It borders
        # one completed city: 3 points once, not once per farmer, and the
        # score bubble sits on exactly one of the two farmers. Player 1's
        # farmer is in a different farm (the field of the city cap) bordering
        # the same city, which is scored separately.
        gs = dinoboard_engine.GameSession(GAME, seed=1)
        _clear_board(gs)
        strip = _cell(10, 10)
        road = _cell(10, 9)
        cap = _cell(9, 10)
        farmer_tll = 4
        gs._force_write_field("board_tile", [strip], TILE_D_START)
        gs._force_write_field("board_meeple_owner", [strip], 0)
        gs._force_write_field("board_meeple_pos", [strip], farmer_tll)
        gs._force_write_field("board_tile", [road], TILE_U_STRAIGHT_ROAD)
        gs._force_write_field("board_rotation", [road], 1)
        gs._force_write_field("board_meeple_owner", [road], 0)
        gs._force_write_field("board_meeple_pos", [road], farmer_tll)
        gs._force_write_field("board_tile", [cap], TILE_E_CITY_TOP)
        gs._force_write_field("board_rotation", [cap], 2)
        gs._force_write_field("board_meeple_owner", [cap], 1)
        gs._force_write_field("board_meeple_pos", [cap], farmer_tll)
        gs._force_write_field("meeples_remaining", [0], 5)
        gs._force_write_field("meeples_remaining", [1], 6)
        gs._force_write_field("current_tile", [], TILE_B_CHAPEL)
        gs._force_write_field("deck_count", [], 0)

        before = _state(gs)
        board = ref.Board(before["board_tile"], before["board_rotation"],
                          before["board_meeple_owner"], before["board_meeple_pos"])
        assert board.root("f", strip, 0) == board.root("f", road, 0), "setup: one farm"
        assert board.root("f", strip, 0) != board.root("f", cap, 0), "setup: cap field is another farm"

        gs.apply_action(_place_action(11, 10, 0))
        gs.apply_action(MEEPLE_PASS_ACTION)
        state = _state(gs)
        assert state["is_terminal"]
        assert state["scores"] == [3, 3]
        assert state["final_score_deltas"] == [3, 3]
        assert state["winner"] == -1 and state["shared_victory"]
        bubbles = state["final_meeple_scores"]
        assert sorted([bubbles[strip], bubbles[road]]) == [0, 3]
        assert bubbles[cap] == 3
        assert sum(bubbles) == 6

    @pytest.mark.parametrize("owners,expected", [
        ((0, 0, 1), [3, 0]),      # 2 farmers against 1: the majority takes all
        ((0, None, 1), [3, 3]),   # 1 against 1: both score in full
        ((0, 1, 1), [0, 3]),
    ])
    def test_final_farm_scoring_majority_and_ties(self, owners, expected):
        # One farm along the upper side of a road three tiles long, bordering
        # the completed two-tile city above the D tile.
        gs = dinoboard_engine.GameSession(GAME, seed=1)
        _clear_board(gs)
        cells = (_cell(10, 10), _cell(10, 9), _cell(10, 8))
        for cell, owner in zip(cells, owners):
            gs._force_write_field("board_tile", [cell],
                                  TILE_D_START if cell == cells[0] else TILE_U_STRAIGHT_ROAD)
            gs._force_write_field("board_rotation", [cell], 0 if cell == cells[0] else 1)
            if owner is not None:
                gs._force_write_field("board_meeple_owner", [cell], owner)
                gs._force_write_field("board_meeple_pos", [cell], 4)
        gs._force_write_field("board_tile", [_cell(9, 10)], TILE_E_CITY_TOP)
        gs._force_write_field("board_rotation", [_cell(9, 10)], 2)
        gs._force_write_field("current_tile", [], TILE_B_CHAPEL)
        gs._force_write_field("deck_count", [], 0)
        gs.apply_action(_place_action(11, 10, 0))
        gs.apply_action(MEEPLE_PASS_ACTION)
        state = _state(gs)
        assert state["is_terminal"]
        assert state["final_score_deltas"] == expected


def _force_draw(gs, tile):
    gs._force_write_field("current_tile", [], tile)


class TestFarmsDoNotCrossCities:
    """Two D tiles whose city edges face each other close a two-tile city.
    The field strips between each tile's city and its road lie on opposite
    sides of that city and are different farms (rulebook: "Farms are
    bordered by roads, cities, and the edge of the area")."""

    def _two_d_tiles_with_farmer_on_upper_strip(self):
        gs = dinoboard_engine.GameSession(GAME, seed=7)
        _force_draw(gs, TILE_D_START)
        gs.apply_action(_place_action(9, 10, 2))
        farmer_brr = MEEPLE_FARM_ACTION_BASE + 4
        assert farmer_brr in gs.get_legal_actions()
        gs.apply_action(farmer_brr)
        return gs

    def test_farmer_may_claim_the_field_on_the_other_side_of_the_city(self):
        gs = self._two_d_tiles_with_farmer_on_upper_strip()
        _force_draw(gs, TILE_U_STRAIGHT_ROAD)
        gs.apply_action(_place_action(10, 9, 1))
        state = _state(gs)
        legal = set(gs.get_legal_actions())
        # The upper field of the road tile continues the START tile's strip;
        # player 0's farmer is beyond the city.
        assert MEEPLE_FARM_ACTION_BASE + 0 in legal
        cell = state["last_row"] * COLS + state["last_col"]
        board = ref.Board(state["board_tile"], state["board_rotation"],
                          state["board_meeple_owner"], state["board_meeple_pos"])
        assert legal == board.legal_meeples(cell, state["meeples_remaining"][1])

    def test_farm_scores_only_the_city_it_borders(self):
        gs = self._two_d_tiles_with_farmer_on_upper_strip()
        _force_draw(gs, TILE_U_STRAIGHT_ROAD)
        gs.apply_action(_place_action(10, 9, 1))
        gs.apply_action(MEEPLE_PASS_ACTION)
        _force_draw(gs, TILE_J_CITY_TOP_ROAD_BEND_RIGHT)
        gs.apply_action(_place_action(10, 8, 0))
        gs.apply_action(MEEPLE_PASS_ACTION)
        state = _state(gs)
        j_cell = state["board_tile"].index(TILE_J_CITY_TOP_ROAD_BEND_RIGHT)
        _force_draw(gs, TILE_E_CITY_TOP)
        gs.apply_action(_place_action(j_cell // COLS - 1, j_cell % COLS, 2))
        for tile in range(TILE_TYPES):
            gs._force_write_field("bag_counts", [tile], 0)
        gs._force_write_field("deck_count", [], 0)
        gs.apply_action(MEEPLE_PASS_ACTION)
        state = _state(gs)
        assert state["is_terminal"]
        # Two completed cities on the board: D+D and J+E. The farmer's strip
        # borders only D+D; the J+E city borders the start tile's strip.
        assert state["scores"] == [3, 0]
        assert state["final_score_deltas"] == [3, 0]


class TestSymmetricTiles:

    # (tile to place, lone tile already on the board, rotations offered)
    @pytest.mark.parametrize("tile,neighbour,rotations", [
        (TILE_B_CHAPEL, TILE_B_CHAPEL, [0]),            # same after every quarter turn
        (TILE_C_FULL_CITY, TILE_C_FULL_CITY, [0]),
        (TILE_X_CROSSROADS, TILE_X_CROSSROADS, [0]),
        (TILE_U_STRAIGHT_ROAD, TILE_B_CHAPEL, [0, 1]),  # same after a half turn
        (TILE_G_CITY_NARROW, TILE_B_CHAPEL, [0, 1]),
        (TILE_A_CHAPEL_ROAD, TILE_B_CHAPEL, [0, 1, 2, 3]),
    ])
    def test_only_distinct_rotations_are_actions(self, tile, neighbour, rotations):
        def position():
            gs = dinoboard_engine.GameSession(GAME, seed=5)
            _clear_board(gs)
            gs._force_write_field("board_tile", [_cell(10, 10)], neighbour)
            _force_draw(gs, tile)
            return gs

        legal = set(position().get_legal_actions())
        assert {a % 4 for a in legal} == set(rotations)
        state = _state(position())
        every_fit = ref.legal_placements(state["board_tile"], state["board_rotation"], tile,
                                         all_rotations=True)
        assert legal < every_fit or len(rotations) == 4
        # The repeats are not actions: an API client must send the lowest
        # equivalent rotation.
        for action in sorted(every_fit - legal):
            assert action % 4 not in rotations
            with pytest.raises(ValueError, match="illegal carcassonne tile action"):
                position().apply_action(action)
        # Nothing is lost: every geometrically distinct placement is offered.
        assert ({(a // 4, ref.shape(tile, a % 4)) for a in legal}
                == {(a // 4, ref.shape(tile, a % 4)) for a in every_fit})


class TestProjection:

    def test_projection_scores_what_the_pending_tile_completed(self):
        gs = dinoboard_engine.GameSession(GAME, seed=1)
        _clear_board(gs)
        center = _cell(10, 10)
        gs._force_write_field("board_tile", [center], TILE_D_START)
        gs._force_write_field("board_meeple_owner", [center], 0)
        gs._force_write_field("board_meeple_pos", [center], 0)
        gs._force_write_field("meeples_remaining", [0], 6)
        gs._force_write_field("current_tile", [], TILE_E_CITY_TOP)
        _arm_next_draw(gs)

        assert _state(gs)["projected_final_scores"] == [1, 0]
        gs.apply_action(_place_action(9, 10, 2))
        pending = _state(gs)
        assert pending["stage"] == STAGE_MEEPLE and pending["scores"] == [0, 0]
        # The two-tile city is complete and will pay 4 whatever the meeple
        # action is; it is not an unfinished city any more.
        assert pending["projected_final_scores"] == [4, 0]
        gs.apply_action(MEEPLE_PASS_ACTION)
        after = _state(gs)
        assert after["scores"] == [4, 0]
        assert after["projected_final_scores"] == [4, 0]


class TestEncoder:

    def test_encode_state_returns_correct_dim(self):
        info = dinoboard_engine.encode_state(GAME, seed=42)
        assert len(info["features"]) == FEATURE_DIM
        assert len(info["legal_mask"]) == ACTION_SPACE
        assert any(abs(x) > 1e-8 for x in info["features"])

    def test_encoder_history_layout_starts_with_initial_tile(self):
        info = dinoboard_engine.encode_state(GAME, seed=42)
        features = info["features"]
        assert FEATURE_DIM == HISTORY_SLOTS * HISTORY_FEATURE_DIM + 4 + TILE_TYPES

        first = features[:HISTORY_FEATURE_DIM]
        assert first[:TILE_TYPES] == [1.0 if i == TILE_D_START else 0.0
                                      for i in range(TILE_TYPES)]
        assert first[TILE_TYPES:TILE_TYPES + 4] == [1.0, 0.0, 0.0, 0.0]
        assert first[TILE_TYPES + 4:TILE_TYPES + 6] == pytest.approx([
            (10.0 - 9.5) / 9.5,
            (10.0 - 9.5) / 9.5,
        ])
        assert first[TILE_TYPES + 6:HISTORY_FEATURE_DIM] == [0.0] * 13

        current_tile_bits = features[-TILE_TYPES:]
        assert sum(current_tile_bits) == pytest.approx(1.0)

    def test_encoder_meeple_history_is_perspective_relative(self):
        seed = 68
        first_tile_action = 839
        gs = dinoboard_engine.GameSession(GAME, seed=seed)
        gs.apply_action(first_tile_action)
        meeple_action = next(
            action for action in gs.get_legal_actions()
            if action != MEEPLE_PASS_ACTION
        )
        meeple_pos = _position_from_meeple_action(meeple_action)
        events = [(0, first_tile_action), (0, meeple_action)]

        p0 = dinoboard_engine.encode_state_for_perspective(
            GAME, seed=seed, encode_player=0, tracker_perspective=-1,
            event_actions=events)
        p1 = dinoboard_engine.encode_state_for_perspective(
            GAME, seed=seed, encode_player=1, tracker_perspective=-1,
            event_actions=events)

        slot_offset = HISTORY_FEATURE_DIM
        meeple_offset = slot_offset + TILE_TYPES + 4 + 2
        p0_meeples = p0["features"][meeple_offset:meeple_offset + 13]
        p1_meeples = p1["features"][meeple_offset:meeple_offset + 13]
        assert p0_meeples[meeple_pos] == pytest.approx(1.0)
        assert p1_meeples[meeple_pos] == pytest.approx(-1.0)
        assert sum(abs(x) for i, x in enumerate(p0_meeples)
                   if i != meeple_pos) == pytest.approx(0.0)
        assert sum(abs(x) for i, x in enumerate(p1_meeples)
                   if i != meeple_pos) == pytest.approx(0.0)


class TestSelfplay:

    def test_episode_completes_or_reaches_limit(self):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=GAME,
            seed=42,
            model_path=get_test_model(GAME),
            simulations=5,
            max_game_plies=30,
        )
        assert ep["total_plies"] > 0
        assert ep["total_plies"] <= 30

    def test_sample_integrity(self):
        ep = dinoboard_engine.run_selfplay_episode(
            game_id=GAME,
            seed=7,
            model_path=get_test_model(GAME),
            simulations=5,
            max_game_plies=20,
        )
        assert ep["samples"]
        for sample in ep["samples"]:
            assert len(sample["features"]) == FEATURE_DIM
            assert len(sample["legal_mask"]) == ACTION_SPACE
            legal_set = {i for i, m in enumerate(sample["legal_mask"]) if m > 0}
            visited = {
                a for a, v in zip(
                    sample["policy_action_ids"], sample["policy_action_visits"])
                if v > 0
            }
            assert visited <= legal_set


class TestHeuristic:

    def test_heuristic_returns_legal_action(self):
        for seed in range(10):
            gs = dinoboard_engine.GameSession(GAME, seed=seed)
            result = gs.get_heuristic_action("rule", 0.0)
            assert result["action"] in gs.get_all_legal_actions()

    def test_rule_mcts_heuristic_returns_legal_action(self):
        for seed in range(3):
            gs = dinoboard_engine.GameSession(GAME, seed=seed)
            result = gs.get_heuristic_action("rule_mcts", 0.0)
            assert result["action"] in gs.get_all_legal_actions()

    def test_heuristic_episode_runs(self):
        ep = dinoboard_engine.run_heuristic_episode(
            game_id=GAME,
            seed=42,
            temperature=0.05,
            max_game_plies=30,
            heuristic_selector="rule",
        )
        assert ep["total_plies"] > 0
        assert ep["samples"]

    def test_rule_mcts_heuristic_episode_runs(self):
        ep = dinoboard_engine.run_heuristic_episode(
            game_id=GAME,
            seed=43,
            temperature=0.0,
            max_game_plies=12,
            heuristic_selector="rule_mcts",
        )
        assert ep["total_plies"] > 0
        assert ep["samples"]


class TestOptionalComponents:

    def test_no_tail_solver_registered(self):
        with pytest.raises(RuntimeError, match="no tail_solver registered"):
            dinoboard_engine.tail_solve(GAME, seed=42)


def _assert_carcassonne_invariants(state: dict) -> None:
    board = state["board_tile"]
    rotations = state["board_rotation"]
    owners = state["board_meeple_owner"]
    positions = state["board_meeple_pos"]
    counts = state["remaining_tile_counts"]
    discarded = state["discarded_tile_counts"]
    assert len(board) == ROWS * COLS
    assert len(rotations) == ROWS * COLS
    assert len(owners) == ROWS * COLS
    assert len(positions) == ROWS * COLS
    assert len(counts) == TILE_TYPES
    assert len(discarded) == TILE_TYPES
    assert all(-1 <= tile < TILE_TYPES for tile in board)
    assert all(0 <= rot < 4 for rot in rotations)
    assert all(owner in (-1, 0, 1) for owner in owners)
    assert all(-1 <= pos <= 12 for pos in positions)
    assert all(count >= 0 for count in counts)
    assert all(count >= 0 for count in discarded)
    placed = sum(1 for tile in board if tile >= 0)
    assert placed == state["plies"] + 1 + (
        1 if state["stage"] == STAGE_MEEPLE else 0)
    assert state["deck_count"] == sum(counts)
    assert placed + sum(counts) + sum(discarded) + (
        1 if state["current_tile"] >= 0 else 0) == 72
    for tile, owner, pos in zip(board, owners, positions):
        if tile < 0:
            assert owner == -1
            assert pos == -1
        if owner < 0:
            assert pos == -1
        else:
            assert pos >= 0


class TestRuleInvariants:

    def test_random_episode_invariants(self):
        for state in run_random_episode_states(GAME, seed=123, max_plies=60):
            _assert_carcassonne_invariants(state)
