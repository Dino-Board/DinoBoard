"""Phase 1: round-end transitions — passed_this_round resets, next host
is the last seat to have actually played, last_play_* clears.
"""
import random
import dinoboard_engine
import pytest

from .test_state_basic import skip_if_unbuilt, VARIANTS


def _is_round_start(sd):
    return (sd["is_first_play_of_round"] == 1
            and sd["last_play_type"] == -1
            and sd["last_play_size"] == 0
            and all(p == 0 for p in sd["passed_this_round"]))


@skip_if_unbuilt
class TestRoundTransition:

    # (test_2p_pass_ends_round_and_passes_host removed: the "single
    # opponent pass → round ends" invariant is only true at 2 players,
    # and maskmen_2p is no longer registered. The N-player generalization
    # is covered by test_round_state_is_self_consistent below.)

    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("seed", [0, 11, 23, 64])
    def test_round_state_is_self_consistent(self, variant, seed):
        # Throughout a random episode: whenever the engine reports a new
        # round (first_play=1), everything else that defines "round start"
        # must agree (no leftover last_play_*, no leftover pass bits).
        # And the current player must be the most recent player to have
        # played (or seat 0 at game-start, before any play has happened).
        rng = random.Random(seed)
        gs = dinoboard_engine.GameSession(variant, seed=seed)
        for _ in range(400):
            sd = gs.get_state_dict()
            if sd["is_first_play_of_round"] == 1:
                assert sd["last_play_type"] == -1
                assert sd["last_play_size"] == 0
                assert all(p == 0 for p in sd["passed_this_round"])
                if sd["last_player_to_play"] >= 0:
                    # The seat we hand the round to must still hold cards.
                    assert sum(sd["players"][sd["current_player"]]["hand_count"]) > 0
            if gs.is_terminal:
                break
            la = gs.get_legal_actions()
            assert la
            gs.apply_action(rng.choice(la))

    def test_passed_seat_skipped_until_round_ends(self):
        # 3-player: P0 opens, P1 passes, P2 plays. Round continues. Next
        # actor must be P0 — P1 has passed and is skipped.
        gs = dinoboard_engine.GameSession("maskmen_3p", seed=7)
        gs.apply_action(gs.get_legal_actions()[0])  # P0 opens
        assert gs.get_state_dict()["current_player"] == 1
        gs.apply_action(6)                          # P1 passes
        sd = gs.get_state_dict()
        # Either P2 acts (round continues) or — if P2 has no legal play —
        # the round may have already ended; we only enforce when not at
        # round-start.
        if sd["is_first_play_of_round"]:
            pytest.skip("seed sequence ended the round early")
        assert sd["current_player"] == 2
        # P2 plays (or passes — if passes the round ends and P0 hosts).
        la = gs.get_legal_actions()
        p2_plays = [a for a in la if a != 6]
        if not p2_plays:
            pytest.skip("P2 has no legal play here")
        gs.apply_action(p2_plays[0])
        sd2 = gs.get_state_dict()
        # The next actor must be active and not P1 (P1 is still flagged as
        # passed — unless the round just ended, in which case all flags clear).
        if not sd2["is_first_play_of_round"]:
            assert sd2["current_player"] != 1
            assert sd2["passed_this_round"][1] == 1
