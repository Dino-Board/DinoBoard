"""carcassonne_rtok: carcassonne_tok plus tokens for the scoring regions.

* Layout: carcassonne_tok's actions, anchors and heads; tokens 7 + 102 wide, a fifth token type.
* Without its region tokens and the features it appends to the global and frontier tokens, an
  rtok encoding is the tok encoding of the same position, for both perspectives.
* Region tokens come last, without anchor, entity or position, with a one-hot kind; the start
  tile alone is one city, one road and two farms.
* Quarter turns: encode(turned state) == transform(encode(state)).
"""
from __future__ import annotations

import numpy as np

import dinoboard_engine as engine

from .test_snapshot_equivalence import _load
from .test_symmetry import _positions, _turned

GAME, TOK = "carcassonne_rtok", "carcassonne_tok"
F = 7
TOK_DIM = F + 9 + 13 * 6
GLOBAL_FEATURES, FRONTIER_FEATURES, GLOBAL_EXTRA, FRONTIER_EXTRA, REGION_FEATURES = 37, 85, 12, 17, 30
DIM = F + FRONTIER_FEATURES + FRONTIER_EXTRA
GLOBAL, CURRENT, TILE, FRONTIER, REGION = range(5)


def _tokens(features, dim):
    return np.asarray(features, dtype=np.float32).reshape(-1, dim)


def test_layout_extends_tok():
    tok, rtok = engine.game_metadata(TOK)["token_layout"], engine.game_metadata(GAME)["token_layout"]
    assert tok["token_dim"] == TOK_DIM and rtok["token_dim"] == DIM
    assert tok["num_token_types"] == 4 and rtok["num_token_types"] == 5
    for key in ("variable_tokens", "token_positions", "num_entity_ids", "num_anchor_ids", "num_pointer_heads",
                "anchor_action", "action_token", "action_head"):
        assert rtok[key] == tok[key], key
    assert engine.encoder_sample_symmetry_count(GAME) == 4


def test_rtok_is_tok_plus_region_features():
    checked = regions_seen = 0
    for seed in (51, 52):
        for seat, payload in _positions(seed, every=4):
            for me in (0, 1):
                a = _tokens(_load(GAME, seat, payload)._encode(me)["features"], DIM)
                b = _tokens(_load(TOK, seat, payload)._encode(me)["features"], TOK_DIM)
                assert np.isfinite(a).all() and (a[:, 0] == 1).all()
                kind = a[:, 3].astype(int)
                rest, regions = a[kind != REGION], a[kind == REGION]
                assert (kind[len(rest):] == REGION).all(), "region tokens come last"
                assert len(rest) == len(b)
                for x, y in zip(rest, b):
                    t = int(y[3])
                    own = F + {GLOBAL: GLOBAL_FEATURES, FRONTIER: FRONTIER_FEATURES}.get(t, TOK_DIM - F)
                    np.testing.assert_array_equal(x[:own], y[:own])
                    assert not y[own:].any()
                    extra = {GLOBAL: GLOBAL_EXTRA, FRONTIER: FRONTIER_EXTRA}.get(t, 0)
                    assert not x[own + extra:].any()
                g = rest[0]
                assert g[3] == GLOBAL and g[F + GLOBAL_FEATURES + 11] > 0, "tiles on the board"
                assert len(regions) > 0 and not regions[:, [1, 2, 4, 5, 6]].any()
                assert (regions[:, F:F + 4].sum(1) == 1).all() and not regions[:, F + REGION_FEATURES:].any()
                regions_seen += len(regions)
                checked += 1
    assert checked > 40 and regions_seen > 5 * checked


def test_start_tile_regions():
    gs = engine.GameSession(GAME, seed=3)
    x = _tokens(gs._encode(gs.current_player)["features"], DIM)
    regions = x[x[:, 3] == REGION]
    kinds = sorted(int(np.argmax(r[F:F + 4])) for r in regions)
    assert kinds == [0, 1, 2, 2], "the start tile: one city, one road, two farms"


def test_encoding_of_turned_state_is_turned_encoding():
    checked = 0
    for seat, payload in _positions(61, every=6):
        base = _load(GAME, seat, payload)
        for g in (1, 2, 3):
            turned = _load(GAME, seat, _turned(payload, g))
            for me in (0, 1):
                a, b = base._encode(me), turned._encode(me)
                feats, out = engine.apply_sample_symmetries(
                    GAME, np.array([g]), np.asarray([a["features"]], dtype=np.float32),
                    [np.asarray([a["legal_mask"]], dtype=np.float32)])
                got = sorted(map(tuple, _tokens(feats[0], DIM).tolist()))
                want = sorted(map(tuple, _tokens(b["features"], DIM).tolist()))
                assert got == want, (g, me)
                np.testing.assert_array_equal(out[0][0], np.asarray(b["legal_mask"], dtype=np.float32))
                checked += 1
    assert checked > 50
