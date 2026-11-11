"""Phase 1: strength_above closure invariants across random episodes.

The closure lives in state and is the single source of truth for which
plays are legal. Every random episode must keep it acyclic (diagonal = 0,
no symmetric pair) and transitively closed.
"""
import random
import dinoboard_engine
import pytest

from .test_state_basic import skip_if_unbuilt, VARIANTS


def _closure_invariants(sa):
    n = len(sa)
    # diagonal zero
    for i in range(n):
        assert sa[i][i] == 0, f"self-loop at {i}: {sa}"
    # antisymmetry — never both directions
    for a in range(n):
        for b in range(a + 1, n):
            assert not (sa[a][b] and sa[b][a]), \
                f"symmetric edge {a}<->{b}: {sa}"
    # transitivity: a>b and b>c ⇒ a>c
    for a in range(n):
        for b in range(n):
            if not sa[a][b]:
                continue
            for c in range(n):
                if sa[b][c]:
                    assert sa[a][c], \
                        f"transitivity broken: {a}>{b}>{c} but not {a}>{c}: {sa}"


def _edge_count(sa):
    return sum(v for row in sa for v in row)


@skip_if_unbuilt
class TestStrengthClosure:

    @pytest.mark.parametrize("variant", VARIANTS)
    @pytest.mark.parametrize("seed", [0, 1, 2, 7, 42, 99])
    def test_closure_invariants_random_episode(self, variant, seed):
        rng = random.Random(seed)
        gs = dinoboard_engine.GameSession(variant, seed=seed)
        for _ in range(300):
            sd = gs.get_state_dict()
            _closure_invariants(sd["strength_above"])
            # 6 vertices → max edges in a strict total order = 15.
            assert _edge_count(sd["strength_above"]) <= 15
            if gs.is_terminal:
                break
            la = gs.get_legal_actions()
            assert la, "non-terminal state with no legal actions"
            gs.apply_action(rng.choice(la))

    def test_first_a1_creates_one_edge(self):
        # Seed picked so player 0 holds at least 2 distinct types — true for
        # almost any seed since each starts with 15/60 cards. We just verify
        # the first non-opener play installs exactly the (t, last_t) edge.
        gs = dinoboard_engine.GameSession("maskmen_3p", seed=5)
        sd0 = gs.get_state_dict()
        assert _edge_count(sd0["strength_above"]) == 0
        # Opener (size 1 of some type).
        opener = gs.get_legal_actions()[0]
        gs.apply_action(opener)
        sd1 = gs.get_state_dict()
        assert _edge_count(sd1["strength_above"]) == 0, \
            "opener must not create any strength edge"
        opener_type = sd1["last_play_type"]
        # Next legal play (skip pass) — first A1 in this episode.
        legal = [a for a in gs.get_legal_actions() if a != 18]
        if not legal:
            pytest.skip("seed has no A1 available after opener (rare)")
        a = legal[0]
        info = gs.get_action_info(a)
        gs.apply_action(a)
        sd2 = gs.get_state_dict()
        new_t = info["wrestler_type"]
        sa = sd2["strength_above"]
        # Exactly one new edge: new_t above opener_type (no prior closure
        # to chain through, since this is the first A1 in the game).
        assert sa[new_t][opener_type] == 1
        assert _edge_count(sa) == 1
