"""Harmonies M1 acceptance: state + rules + token stacking only.

M1 scope: SELECT_SLOT / PLACE_TOKEN / END_TURN. No animal cards / cubes /
landscape scoring. The test suite verifies:
- registration + metadata
- initial state shape
- turn-state machine (SELECT_SLOT then PLACE_TOKEN until pool empty,
  END_TURN advances current_player)
- stacking transition table consistency
- random episode rule invariants (pouch + visible counts conserve, no
  illegal towers ever appear, has_animal stays 0 in M1)
"""
import dinoboard_engine
import pytest


GAME = "harmonies"
VARIANTS = ["harmonies", "harmonies_2p", "harmonies_3p", "harmonies_4p",
            "harmonies_a", "harmonies_a_2p", "harmonies_a_3p", "harmonies_a_4p"]
# Side B (25 hexes) on the default ids, side A (23 hexes) on harmonies_a*.
ACTION_SPACE = {"harmonies": 988, "harmonies_a": 912}


def _layout(gs):
    return gs.get_state_dict()["action_layout"]

# Tower types: 0=empty, 1..6=single colors (Blue..Red), 7=房2, 8=木木, 9=树2,
# 10=山2, 11=树3, 12=山3
TOWER_TYPES = list(range(13))


# ---------------------------------------------------------------------------
# 1. Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_variants_registered(self):
        games = dinoboard_engine.available_games()
        for v in VARIANTS:
            assert v in games

    @pytest.mark.parametrize("game_id", sorted(ACTION_SPACE))
    def test_metadata(self, game_id):
        meta = dinoboard_engine.game_metadata(game_id)
        assert meta["action_space"] == ACTION_SPACE[game_id]
        assert meta["num_players"] == 2

    @pytest.mark.parametrize("variant,expected_n", [
        ("harmonies_2p", 2), ("harmonies_3p", 3), ("harmonies_4p", 4),
        ("harmonies_a_2p", 2), ("harmonies_a_3p", 3), ("harmonies_a_4p", 4),
    ])
    def test_player_count(self, variant, expected_n):
        meta = dinoboard_engine.game_metadata(variant)
        assert meta["num_players"] == expected_n


# ---------------------------------------------------------------------------
# 2. Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def setup_method(self):
        self.gs = dinoboard_engine.GameSession(
            GAME, seed=42, model_path="", use_filter=False)
        self.s = self.gs.get_state_dict()

    def test_current_player_zero(self):
        assert self.s["current_player"] == 0

    def test_not_terminal(self):
        assert self.s["is_terminal"] is False
        assert self.s["winner"] == -1

    def test_turn_state_machine_initial(self):
        assert self.s["turn_started"] is False
        assert self.s["took_card_this_turn"] is False
        assert sum(self.s["selected_tokens"]) == 0

    def test_pouch_total_120_minus_central_board(self):
        # 120 starting tokens = 23+23+21+19+19+15.
        cb_total = sum(sum(slot) for slot in self.s["central_board_counts"])
        pouch_total = sum(self.s["pouch_counts"])
        assert pouch_total + cb_total == 120

    def test_central_board_filled(self):
        # 5 slots × 3 tokens each at start.
        for slot in self.s["central_board_counts"]:
            assert sum(slot) == 3

    def test_animal_market_distinct_5_cards(self):
        market = self.s["animal_market"]
        assert len(market) == 5
        assert all(1 <= c <= 32 for c in market)
        assert len(set(market)) == 5

    def test_animal_deck_27_remaining(self):
        deck = self.s["animal_deck"]
        # 32 - 5 in market = 27 still in draw pile.
        assert sum(deck) == 27

    def test_player_boards_empty(self):
        for p in self.s["players"]:
            assert len(p["tower_type"]) == 25
            assert all(t == 0 for t in p["tower_type"])
            assert all(a == 0 for a in p["has_animal"])
            assert p["space_remaining"] == 25
            # Side B: an empty board is one island, worth 5.
            assert p["score"] == p["landscape_score"] == 5

    def test_initial_legal_is_select_slot(self):
        legal = self.gs.get_legal_actions()
        # 5 SELECT_SLOT actions, ids 0..4.
        assert sorted(legal) == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# 3. Turn-state machine
# ---------------------------------------------------------------------------

class TestTurnStateMachine:
    def test_select_slot_drains_central_to_pool(self):
        gs = dinoboard_engine.GameSession(
            GAME, seed=42, model_path="", use_filter=False)
        s0 = gs.get_state_dict()
        slot = 0
        slot_counts = list(s0["central_board_counts"][slot])
        gs.apply_action(slot)  # SELECT_SLOT 0
        s1 = gs.get_state_dict()
        assert s1["turn_started"] is True
        # slot is now empty
        assert sum(s1["central_board_counts"][slot]) == 0
        # pool has those tokens
        assert list(s1["selected_tokens"]) == slot_counts
        # current player unchanged
        assert s1["current_player"] == 0

    def test_after_select_only_place_or_end(self):
        gs = dinoboard_engine.GameSession(
            GAME, seed=42, model_path="", use_filter=False)
        gs.apply_action(0)
        legal = gs.get_legal_actions()
        lay = _layout(gs)
        # After SELECT_SLOT, legal includes PLACE_TOKEN and may include
        # TAKE_CARD (5..36); never SELECT_SLOT or END_TURN.
        for a in legal:
            assert 5 <= a < lay["end_turn"]

    def test_end_turn_advances_player(self):
        gs = dinoboard_engine.GameSession(
            GAME, seed=42, model_path="", use_filter=False)
        gs.apply_action(0)  # select slot 0
        # Drive PLACE_TOKEN actions only (skip TAKE_CARD/PLACE_CUBE to keep
        # the test focused on the turn-state machine).
        while True:
            s = gs.get_state_dict()
            if sum(s["selected_tokens"]) == 0:
                break
            legal = [a for a in gs.get_legal_actions() if 37 <= a < _layout(gs)["place_cube"]]
            assert legal, "should have a legal place_token while pool has tokens"
            gs.apply_action(legal[0])
        # Now END_TURN must be among legal (other non-pool actions like
        # TAKE_CARD may also still be legal).
        legal = gs.get_legal_actions()
        end_turn = _layout(gs)["end_turn"]
        assert end_turn in legal
        gs.apply_action(end_turn)
        s2 = gs.get_state_dict()
        assert s2["current_player"] == 1
        assert s2["turn_started"] is False
        assert sum(s2["selected_tokens"]) == 0

    def test_end_turn_refills_central_board(self):
        gs = dinoboard_engine.GameSession(
            GAME, seed=42, model_path="", use_filter=False)
        gs.apply_action(0)
        while True:
            s = gs.get_state_dict()
            if sum(s["selected_tokens"]) == 0:
                break
            legal = gs.get_legal_actions()
            gs.apply_action(legal[0])
        gs.apply_action(_layout(gs)["end_turn"])
        s2 = gs.get_state_dict()
        # Slot 0 was emptied at the start of the turn; END_TURN should refill
        # it back to 3 tokens (assuming pouch has enough — should at start).
        assert sum(s2["central_board_counts"][0]) == 3


# ---------------------------------------------------------------------------
# 4. Tower stacking invariants (random play)
# ---------------------------------------------------------------------------

def _drive_random(gs, seed, max_plies=200):
    import random
    rng = random.Random(seed)
    yield gs.get_state_dict()
    for _ in range(max_plies):
        if gs.is_terminal:
            break
        legal = gs.get_legal_actions()
        if not legal:
            break
        gs.apply_action(rng.choice(legal))
        yield gs.get_state_dict()


class TestRandomPlayInvariants:
    @pytest.mark.parametrize("seed", [1, 7, 42, 123, 999])
    def test_tower_types_always_valid(self, seed):
        gs = dinoboard_engine.GameSession(
            GAME, seed=seed, model_path="", use_filter=False)
        for st in _drive_random(gs, seed=seed):
            for p in st["players"]:
                for t in p["tower_type"]:
                    assert t in TOWER_TYPES

    @pytest.mark.parametrize("seed", [1, 7, 42, 123, 999])
    def test_no_animals_until_taken(self, seed):
        # After M2 a player can take animal cards and place cubes. The
        # weaker invariant: has_animal[p][h]==1 implies player p has held
        # (or holds) some animal card. We just check has_animal is 0/1.
        gs = dinoboard_engine.GameSession(
            GAME, seed=seed, model_path="", use_filter=False)
        for st in _drive_random(gs, seed=seed):
            for p in st["players"]:
                for a in p["has_animal"]:
                    assert a in (0, 1)

    @pytest.mark.parametrize("seed", [1, 7, 42, 123, 999])
    def test_space_remaining_consistent_with_tower(self, seed):
        gs = dinoboard_engine.GameSession(
            GAME, seed=seed, model_path="", use_filter=False)
        for st in _drive_random(gs, seed=seed):
            for p in st["players"]:
                empties = sum(1 for t in p["tower_type"] if t == 0)
                assert p["space_remaining"] == empties

    @pytest.mark.parametrize("seed", [1, 7, 42])
    def test_token_conservation(self, seed):
        # Sum of pouch + central_board + selected_tokens + per-player layered
        # tokens (counted by tower height) == 120 at all times.
        # M1 has no animal cubes so counting tower height covers all placed
        # tokens. tower_height(t): 0->0, 1..6->1, 7..10->2, 11/12->3.
        height = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1,
                  7: 2, 8: 2, 9: 2, 10: 2, 11: 3, 12: 3}
        gs = dinoboard_engine.GameSession(
            GAME, seed=seed, model_path="", use_filter=False)
        for st in _drive_random(gs, seed=seed):
            total = sum(st["pouch_counts"])
            total += sum(sum(slot) for slot in st["central_board_counts"])
            total += sum(st["selected_tokens"])
            for p in st["players"]:
                total += sum(height[t] for t in p["tower_type"])
            assert total == 120, f"token count drifted to {total}"

    @pytest.mark.parametrize("seed", [1, 7, 42, 123, 999])
    def test_pool_always_drains_before_end_turn(self, seed):
        # When current player has turn_started=True and selected_tokens has
        # any tokens left, END_TURN must NOT be in legal_actions.
        gs = dinoboard_engine.GameSession(
            GAME, seed=seed, model_path="", use_filter=False)
        for st in _drive_random(gs, seed=seed):
            if gs.is_terminal:
                break
            legal = gs.get_legal_actions()
            if (st["turn_started"] and sum(st["selected_tokens"]) > 0):
                assert _layout(gs)["end_turn"] not in legal


# ---------------------------------------------------------------------------
# 5. Stacking transition rules
# ---------------------------------------------------------------------------

class TestStackingRules:
    def test_blue_yellow_no_stack(self):
        # Place Blue on empty -> single Blue (tower_type 1). Then no further
        # stacking is legal on that hex from any color.
        gs = dinoboard_engine.GameSession(
            GAME, seed=42, model_path="", use_filter=False)
        # Find any select_slot that contains a Blue (idx 0).
        s = gs.get_state_dict()
        slot = next(i for i, sl in enumerate(s["central_board_counts"])
                    if sl[0] > 0)
        gs.apply_action(slot)
        s = gs.get_state_dict()
        # PLACE_TOKEN(hex=0, color=Blue=1) -> action_id = 37 + 0*6 + 0 = 37
        assert 37 in gs.get_legal_actions()
        gs.apply_action(37)
        s = gs.get_state_dict()
        assert s["players"][0]["tower_type"][0] == 1  # single Blue
        # Now no PLACE_TOKEN on hex 0 should be legal — find any color from 0..5.
        for color_idx in range(6):
            assert (37 + 0 * 6 + color_idx) not in gs.get_legal_actions()
