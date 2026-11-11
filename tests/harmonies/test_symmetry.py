"""Board symmetries: the group, rule invariance by experiment, and the
encoders' sample transforms (the single source of truth for training-time
augmentation, training/symmetry.py).

* The symmetry group of each board side is recomputed here from the board's
  columns (independently of scripts/gen_card_data.py) and must be what the
  encoders report: 4 elements (identity, 180 degrees, two mirrors) on both sides.
* Rules: for random games, a second session replays every action through
  symmetry g. Legal actions must correspond one-to-one at every ply, boards
  must be the g-image of each other, and scores / outcome must be equal.
* Encoders (side B MLP, side A MLP, side B tokens): encode(g s) must equal the
  encoder-declared transform of encode(s), legal masks included.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

import dinoboard_engine as engine
from training.symmetry import SymmetryAugmenter

DIRS = [(0, -1), (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0)]


def _axial(columns):
    return {h: (q, i - q // 2) for q, col in enumerate(columns) for i, h in enumerate(col)}


def _lattice_symmetries(columns):
    """Hex permutations of every rotation / reflection of the hex lattice (plus
    translation) that maps the board onto itself."""
    ax = _axial(columns)
    cells = {v: h for h, v in ax.items()}
    perms = []
    for mirror in (False, True):
        for k in range(6):
            def f(q, r):
                if mirror:
                    q, r = r, q
                for _ in range(k):
                    q, r = -r, q + r
                return q, r
            img = {h: f(*v) for h, v in ax.items()}
            for target in cells:
                tq, tr = target[0] - img[0][0], target[1] - img[0][1]
                moved = {h: (q + tq, r + tr) for h, (q, r) in img.items()}
                if set(moved.values()) == set(cells):
                    perms.append(tuple(cells[moved[h]] for h in sorted(ax)))
                    break
    return perms


def _hex_perm(sym, n_hexes):
    am = sym["action_map"]
    return tuple((am[37 + h * 6] - 37) // 6 for h in range(n_hexes))


@pytest.mark.parametrize("game_id,shape", [("harmonies", [4, 3, 4, 3, 4, 3, 4]),
                                           ("harmonies_a", [5, 4, 5, 4, 5])])
def test_symmetry_group_matches_independent_computation(game_id, shape):
    sd = engine.GameSession(game_id, seed=1).get_state_dict()
    columns = sd["board_columns"]
    assert [len(c) for c in columns] == shape
    n = sum(shape)
    expected = _lattice_symmetries(columns)
    assert len(expected) == 4
    for gid in (game_id, game_id + "_3p", "harmonies_tok" if game_id == "harmonies" else game_id + "_4p"):
        got = [_hex_perm(s, n) for s in engine.encoder_symmetries(gid)]
        assert got[0] == tuple(range(n)), "symmetry 0 must be the identity"
        assert sorted(got) == sorted(expected), gid


def _mapped(sym, actions):
    am = sym["action_map"]
    return sorted(am[a] for a in actions)


@pytest.mark.parametrize("game_id", ["harmonies", "harmonies_3p", "harmonies_a"])
def test_rules_are_invariant_under_board_symmetries(game_id):
    """A game replayed through symmetry g is the g-image of the original at
    every ply: legal actions correspond one-to-one, boards map hex by hex, and
    scores, winner and length are equal. Greedy and random movers mix so that
    animal cubes are placed often."""
    syms = engine.encoder_symmetries(game_id)
    n = len(syms[0]["action_map"])
    plies = cubes = 0
    for seed in range(6):
        for g in range(1, len(syms)):
            perm = _hex_perm(syms[g], (n - 38) // 38)
            a_gs = engine.GameSession(game_id, seed=seed)
            b_gs = engine.GameSession(game_id, seed=seed)
            rng = random.Random(seed * 10 + g)
            while not a_gs.is_terminal:
                legal = a_gs.get_legal_actions()
                assert _mapped(syms[g], legal) == sorted(b_gs.get_legal_actions())
                a = (a_gs.get_heuristic_action("greedy", 0.0)["action"] if rng.random() < 0.5
                     else rng.choice(legal))
                a_gs.apply_action(a)
                b_gs.apply_action(syms[g]["action_map"][a])
                plies += 1
                sa, sb = a_gs.get_state_dict(), b_gs.get_state_dict()
                for pa, pb in zip(sa["players"], sb["players"]):
                    for key in ("tower_type", "has_animal"):
                        assert [pb[key][perm[h]] for h in range(len(perm))] == pa[key]
                    for key in ("score", "animal_score", "landscape_score", "held_cards", "cube_remaining"):
                        assert pa[key] == pb[key], key
                cubes += 37 + 6 * len(perm) <= a < n - 1
            assert b_gs.is_terminal and a_gs.winner == b_gs.winner
    assert plies > 1000 and cubes > 20


def _tokens_as_set(x, token_dim):
    rows = np.asarray(x, dtype=np.float32).reshape(-1, token_dim)
    return sorted(map(tuple, rows.tolist()))


@pytest.mark.parametrize("game_id", ["harmonies", "harmonies_a", "harmonies_tok"])
def test_encoding_of_transformed_state_is_transformed_encoding(game_id):
    """encode(g s) == transform_g(encode(s)) and legal_mask(g s) ==
    transform_g(legal_mask(s)), for both perspectives, using the encoder's
    declared symmetries through the training-time augmenter."""
    meta = engine.game_metadata(game_id)
    lay = meta["token_layout"]
    syms = engine.encoder_symmetries(game_id)
    aug = SymmetryAugmenter(syms, feature_dim=meta["feature_dim"], action_space=meta["action_space"],
                            token_layout=lay)
    checked = 0
    for seed in range(4):
        for g in range(1, len(syms)):
            a_gs = engine.GameSession(game_id, seed=100 + seed)
            b_gs = engine.GameSession(game_id, seed=100 + seed)
            rng = random.Random(seed + 7 * g)
            ply = 0
            while not a_gs.is_terminal:
                if ply % 3 == 0:
                    for me in range(meta["num_players"]):
                        ea, eb = a_gs._encode(me), b_gs._encode(me)
                        sym = np.array([g])
                        xa = np.asarray([ea["features"]], dtype=np.float32)
                        got = aug.features(xa, sym)[0]
                        if lay:
                            assert _tokens_as_set(got, lay["token_dim"]) == _tokens_as_set(eb["features"], lay["token_dim"])
                        else:
                            np.testing.assert_array_equal(got, np.asarray(eb["features"], dtype=np.float32))
                        mask = aug.actions(np.asarray([ea["legal_mask"]], dtype=np.float32), sym)[0]
                        np.testing.assert_array_equal(mask, np.asarray(eb["legal_mask"], dtype=np.float32))
                        checked += 1
                legal = a_gs.get_legal_actions()
                a = (a_gs.get_heuristic_action("greedy", 0.0)["action"] if rng.random() < 0.5
                     else rng.choice(legal))
                a_gs.apply_action(a)
                b_gs.apply_action(syms[g]["action_map"][a])
                ply += 1
    assert checked > 300


def test_augmenter_maps_policy_mass_with_actions():
    """A one-hot policy on action a becomes a one-hot on action_map[a]."""
    meta = engine.game_metadata("harmonies_tok")
    syms = engine.encoder_symmetries("harmonies_tok")
    aug = SymmetryAugmenter(syms, feature_dim=0, action_space=meta["action_space"], token_layout=meta["token_layout"])
    rng = np.random.default_rng(0)
    acts = rng.integers(meta["action_space"], size=64)
    p = np.zeros((64, meta["action_space"]), dtype=np.float32)
    p[np.arange(64), acts] = 1.0
    sym = aug.sample(rng, 64)
    out = aug.actions(p, sym)
    for i in range(64):
        assert out[i, syms[sym[i]]["action_map"][acts[i]]] == 1.0 and out[i].sum() == 1.0
