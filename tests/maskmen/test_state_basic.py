"""Phase 1: basic state / registration / determinism sanity for maskmen.

Phase 1 only ships rules + state — no encoder, no tracker, no config files
yet. So these tests deliberately avoid load_game_config / feature_dim /
ONNX model paths; that all lands in Phase 2+.
"""
import dinoboard_engine
import pytest


GAME = "maskmen_3p"
VARIANTS = ["maskmen_3p", "maskmen_4p"]


def _available() -> bool:
    return GAME in dinoboard_engine.available_games()


skip_if_unbuilt = pytest.mark.skipif(
    not _available(), reason="maskmen not built (manifest disabled)")


@skip_if_unbuilt
class TestRegistration:

    def test_all_variants_registered(self):
        games = dinoboard_engine.available_games()
        for v in VARIANTS:
            assert v in games, f"{v} not registered"

    def test_player_counts(self):
        assert dinoboard_engine.GameSession("maskmen_3p", seed=42).num_players == 3
        assert dinoboard_engine.GameSession("maskmen_4p", seed=42).num_players == 4

    def test_2p_not_registered(self):
        # 2p was removed (degenerate at 2 players); make sure it really is gone.
        assert "maskmen_2p" not in dinoboard_engine.available_games()


@skip_if_unbuilt
class TestInitialState:

    @pytest.mark.parametrize("variant,np", [
        ("maskmen_3p", 3),
        ("maskmen_4p", 4),
    ])
    def test_initial_layout(self, variant, np):
        gs = dinoboard_engine.GameSession(variant, seed=7)
        sd = gs.get_state_dict()
        assert sd["current_player"] == 0, "first host must be seat 0"
        assert sd["is_first_play_of_round"] == 1
        assert sd["last_play_type"] == -1
        assert sd["last_play_size"] == 0
        assert sd["last_player_to_play"] == -1
        assert sd["winner"] == -1
        assert not sd["is_terminal"]
        assert sd["num_players"] == np
        # 6×6 closure starts empty.
        for row in sd["strength_above"]:
            assert all(v == 0 for v in row)
        # Each seat gets exactly kStartingHand = 15 cards across 6 types.
        for p in range(np):
            hc = sd["players"][p]["hand_count"]
            assert len(hc) == 6
            assert sum(hc) == 15

    def test_total_dealt_does_not_exceed_deck(self):
        # 60-card deck, 15 per player, surplus discarded.
        for variant, np in [("maskmen_3p", 3), ("maskmen_4p", 4)]:
            gs = dinoboard_engine.GameSession(variant, seed=99)
            sd = gs.get_state_dict()
            total = sum(sum(sd["players"][p]["hand_count"]) for p in range(np))
            assert total == 15 * np
            assert total <= 60


@skip_if_unbuilt
class TestDeterminism:

    def test_same_seed_same_initial_hands(self):
        a = dinoboard_engine.GameSession(GAME, seed=12345).get_state_dict()
        b = dinoboard_engine.GameSession(GAME, seed=12345).get_state_dict()
        for p in range(2):
            assert a["players"][p]["hand_count"] == b["players"][p]["hand_count"]

    def test_different_seed_different_hands(self):
        # Probabilistic but extremely safe — 60!/((10!)^6) is huge.
        a = dinoboard_engine.GameSession(GAME, seed=1).get_state_dict()
        b = dinoboard_engine.GameSession(GAME, seed=2).get_state_dict()
        assert a["players"][0]["hand_count"] != b["players"][0]["hand_count"] or \
               a["players"][1]["hand_count"] != b["players"][1]["hand_count"]


@skip_if_unbuilt
class TestActionInfo:

    def test_pass_action_id_is_6(self):
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        info = gs.get_action_info(6)
        assert info["type"] == "pass"

    def test_play_action_layout(self):
        # action_id == wrestler_type (0..5). Size is state-derived, not
        # part of the action.
        gs = dinoboard_engine.GameSession(GAME, seed=42)
        for t in range(6):
            info = gs.get_action_info(t)
            assert info["type"] == "play"
            assert info["wrestler_type"] == t
            assert "size" not in info
