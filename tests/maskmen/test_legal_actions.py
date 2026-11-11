"""Phase 1: legal_actions correctness against the rules in plan §1.4."""
import random
import dinoboard_engine
import pytest

from .test_state_basic import skip_if_unbuilt, VARIANTS


# Post-refactor action layout: 0..5 = play wrestler-type, 6 = pass.
# Size is fully determined by round state, not encoded in the action.
PASS = 6


def _decode(a):
    if a == PASS:
        return ("pass", None)
    return ("play", a)


@skip_if_unbuilt
class TestLegalActions:

    def test_first_play_no_pass(self):
        gs = dinoboard_engine.GameSession("maskmen_3p", seed=42)
        sd = gs.get_state_dict()
        assert sd["is_first_play_of_round"] == 1
        la = gs.get_legal_actions()
        assert PASS not in la, "host cannot pass the round opener"
        for a in la:
            assert 0 <= a < PASS

    def test_first_play_covers_every_held_type(self):
        gs = dinoboard_engine.GameSession("maskmen_3p", seed=42)
        sd = gs.get_state_dict()
        hc = sd["players"][0]["hand_count"]
        held = {t for t, c in enumerate(hc) if c >= 1}
        play_types = {a for a in gs.get_legal_actions() if a != PASS}
        assert play_types == held

    def test_pass_legal_after_opener(self):
        gs = dinoboard_engine.GameSession("maskmen_3p", seed=42)
        gs.apply_action(gs.get_legal_actions()[0])
        assert PASS in gs.get_legal_actions()

    def test_cannot_play_same_type_as_last_play(self):
        gs = dinoboard_engine.GameSession("maskmen_3p", seed=42)
        gs.apply_action(gs.get_legal_actions()[0])
        last_t = gs.get_state_dict()["last_play_type"]
        for a in gs.get_legal_actions():
            if a == PASS:
                continue
            assert a != last_t

    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("seed", [1, 4, 13, 77])
    def test_random_episode_actions_all_legal(self, variant, seed):
        rng = random.Random(seed)
        gs = dinoboard_engine.GameSession(variant, seed=seed)
        for _ in range(400):
            if gs.is_terminal:
                break
            la = gs.get_legal_actions()
            assert la
            gs.apply_action(rng.choice(la))

    def test_legal_play_obeys_a1_a2_filter(self):
        # After opener (no prior closure), unknown-relation only: each
        # legal play of type t against last_t must satisfy: t != last_t,
        # not known-weaker, and the seat holds at least last_size+1 of t
        # (since the closure is empty so far this is all A1).
        gs = dinoboard_engine.GameSession("maskmen_3p", seed=3)
        gs.apply_action(gs.get_legal_actions()[0])  # opener size 1
        sd = gs.get_state_dict()
        last_t = sd["last_play_type"]
        last_s = sd["last_play_size"]
        sa = sd["strength_above"]
        cur = sd["current_player"]
        hc = sd["players"][cur]["hand_count"]
        for a in gs.get_legal_actions():
            if a == PASS:
                continue
            t = a
            assert t != last_t
            assert sa[last_t][t] == 0
            if sa[t][last_t]:        # A2 — need last_s of them
                assert hc[t] >= last_s
            else:                     # A1 — need last_s + 1 of them
                need = last_s + 1
                assert hc[t] >= need
                assert need <= 3
