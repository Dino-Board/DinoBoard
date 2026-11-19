"""carcassonne_rtok: carcassonne_tok plus tokens for the scoring regions and their relations.

* Layout: carcassonne_tok's actions, anchors and heads; tokens 7 + 102 + 17 wide (the last 17 are
  token relations: a region's id and up to 16 linked ids); four more token types, one per region
  kind.
* Without its region tokens, the features it appends to the global and frontier tokens and the
  relation columns, an rtok encoding is the tok encoding of the same position, for both
  perspectives.
* Region tokens come last, without anchor, entity or position, with ids 1, 2, ... in order; the
  start tile alone is one city, one road and two farms.
* Relations agree with the features computed independently of them: a frontier cell links to
  exactly the regions its per-kind / per-leader counts and cloister counts describe, every region
  is linked from as many frontier cells as it can grow into, and a farm links to its incomplete
  adjacent cities.
* Quarter turns: encode(turned state) == transform(encode(state)), with region ids compared as
  the tokens they name (ids are names, the network does not depend on them).
"""
from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import pytest

import dinoboard_engine as engine

from .test_snapshot_equivalence import _load
from .test_symmetry import _positions, _turned

GAME, TOK = "carcassonne_rtok", "carcassonne_tok"
F = 7
TOK_DIM = F + 9 + 13 * 6
GLOBAL_FEATURES, FRONTIER_FEATURES, GLOBAL_EXTRA, FRONTIER_EXTRA, REGION_FEATURES = 37, 85, 12, 17, 30
LINKS = 16
KEY = F + FRONTIER_FEATURES + FRONTIER_EXTRA
DIM = KEY + 1 + LINKS
GLOBAL, CURRENT, TILE, FRONTIER, REGION = range(5)
CITY, ROAD, FARM, CLOISTER = range(4)
# Region token columns (after the header).
RG_MEEPLES, RG_CELLS, RG_FARM_CITIES = 7, 14, 21
# Frontier token columns after its own features.
FX_REGIONS, FX_CLOISTERS = 0, 11


def _tokens(features, dim):
    return np.asarray(features, dtype=np.float32).reshape(-1, dim)


def test_layout_extends_tok():
    tok, rtok = engine.game_metadata(TOK)["token_layout"], engine.game_metadata(GAME)["token_layout"]
    assert tok["token_dim"] == TOK_DIM and rtok["token_dim"] == DIM
    assert tok["num_token_types"] == 4 and rtok["num_token_types"] == 8
    for key in ("variable_tokens", "token_positions", "num_entity_ids", "num_anchor_ids", "num_pointer_heads",
                "anchor_action", "action_token", "action_head"):
        assert rtok[key] == tok[key], key
    assert tok["relation_key_column"] == -1
    assert (rtok["relation_key_column"], rtok["relation_link_column"], rtok["relation_link_count"],
            rtok["relation_max_id"]) == (KEY, KEY + 1, LINKS, 128)
    assert engine.encoder_sample_symmetry_count(GAME) == 4


def _leaders(mine, theirs):
    """Who leads a region, as the encoder's bits: me, them (both on a tie), nobody."""
    if mine == 0 and theirs == 0:
        return (False, False, True)
    return (mine >= theirs, theirs >= mine, False)


def _check_relations(a):
    kind = a[:, 3].astype(int)
    regions = a[kind >= REGION]
    ids = regions[:, KEY].astype(int)
    assert list(ids) == list(range(1, len(regions) + 1)), "region ids are 1, 2, ... in token order"
    by_id = {int(i): r for i, r in zip(ids, regions)}
    assert not a[kind < REGION, KEY].any(), "only region tokens have an id"
    links = {i: [int(v) for v in t[KEY + 1:] if v] for i, t in enumerate(a)}
    for i, t in enumerate(a):
        if kind[i] not in (FRONTIER, REGION + FARM):
            assert not links[i], f"token type {kind[i]} links"
        assert all(v in by_id for v in links[i]), "a link names no region"
        assert len(set(links[i])) == len(links[i]), "a region linked twice"
    linked_from = collections.Counter()
    max_links = 0
    for i in np.flatnonzero(kind == FRONTIER):
        x = a[i, F + FRONTIER_FEATURES:]
        named = [by_id[v] for v in links[i]]
        max_links = max(max_links, len(named))
        counts = np.zeros(9)
        cloisters = np.zeros(2)
        for r in named:
            k = int(r[3]) - REGION
            who = _leaders(round(r[F + RG_MEEPLES] * 3), round(r[F + RG_MEEPLES + 1] * 3))
            if k < CLOISTER:
                counts[3 * k:3 * k + 3] += np.array(who, dtype=float) * 0.25
            else:
                cloisters += np.array(who[:2], dtype=float) * 0.5
        np.testing.assert_allclose(counts, x[FX_REGIONS:FX_REGIONS + 9], atol=1e-6)
        np.testing.assert_allclose(cloisters, x[FX_CLOISTERS:FX_CLOISTERS + 2], atol=1e-6)
        linked_from.update(links[i])
    for v, r in by_id.items():
        assert round(r[F + RG_CELLS] * 6) == linked_from[v], "frontier cells it can grow into"
    for i in np.flatnonzero(kind == REGION + FARM):
        named = [by_id[v] for v in links[i]]
        assert all(int(r[3]) == REGION + CITY for r in named), "a farm links to cities only"
        assert len(named) == round(a[i, F + RG_FARM_CITIES + 1] * 5), "its incomplete adjacent cities"
    return max_links


def test_rtok_is_tok_plus_region_features_and_relations():
    checked = regions_seen = max_links = 0
    for seed in (51, 52):
        for seat, payload in _positions(seed, every=4):
            for me in (0, 1):
                a = _tokens(_load(GAME, seat, payload)._encode(me)["features"], DIM)
                b = _tokens(_load(TOK, seat, payload)._encode(me)["features"], TOK_DIM)
                assert np.isfinite(a).all() and (a[:, 0] == 1).all()
                kind = a[:, 3].astype(int)
                rest, regions = a[kind < REGION], a[kind >= REGION]
                assert (kind[len(rest):] >= REGION).all(), "region tokens come last"
                assert len(rest) == len(b)
                for x, y in zip(rest, b):
                    t = int(y[3])
                    own = F + {GLOBAL: GLOBAL_FEATURES, FRONTIER: FRONTIER_FEATURES}.get(t, TOK_DIM - F)
                    np.testing.assert_array_equal(x[:own], y[:own])
                    assert not y[own:].any()
                    extra = {GLOBAL: GLOBAL_EXTRA, FRONTIER: FRONTIER_EXTRA}.get(t, 0)
                    assert not x[own + extra:KEY].any()
                g = rest[0]
                assert g[3] == GLOBAL and g[F + GLOBAL_FEATURES + 11] > 0, "tiles on the board"
                assert len(regions) > 0 and not regions[:, [1, 2, 4, 5, 6]].any()
                assert (regions[:, F:F + 4].argmax(1) == regions[:, 3] - REGION).all(), "type = kind"
                assert (regions[:, F:F + 4].sum(1) == 1).all() and not regions[:, F + REGION_FEATURES:KEY].any()
                max_links = max(max_links, _check_relations(a))
                regions_seen += len(regions)
                checked += 1
    assert checked > 40 and regions_seen > 5 * checked
    assert 2 <= max_links < LINKS, f"a frontier cell links to {max_links} regions"


def test_start_tile_regions():
    gs = engine.GameSession(GAME, seed=3)
    x = _tokens(gs._encode(gs.current_player)["features"], DIM)
    kinds = sorted(int(t) - REGION for t in x[:, 3] if t >= REGION)
    assert kinds == [CITY, ROAD, FARM, FARM], "the start tile: one city, one road, two farms"


TEACHER = Path(__file__).resolve().parents[2] / "games/carcassonne/model/carcassonne_tok_2p.onnx"


@pytest.mark.skipif(not TEACHER.exists(), reason="the carcassonne_tok model is not in games/carcassonne/model")
def test_teacher_plays_legal_moves():
    """tok_v2_mcts reads the state with the carcassonne_tok encoding inside carcassonne_rtok."""
    gs = engine.GameSession(GAME, seed=7)
    for _ in range(6):
        r = gs.get_heuristic_action("tok_v2_mcts")
        assert r["action"] in gs.get_legal_actions()
        gs.apply_action(int(r["action"]))


def test_training_loop_with_relations(tmp_path):
    """Two CPU steps of the real pipeline with the rtok recipe (relation bias, compact fp16 replay,
    quarter-turn augmentation), guided by a rule heuristic; the exported model plays."""
    import json

    from training.cli import find_game_config, find_heuristic_config, validate_guidance_until
    from training.pipeline import run_training_loop
    path = Path(__file__).resolve().parents[2] / "games/carcassonne/config/train_transformer_rtok.json"
    train_cfg = json.loads(path.read_text(encoding="utf-8"))
    validate_guidance_until(train_cfg["heuristic_guidance"], path)
    assert train_cfg["heuristic_guidance"]["name"] == train_cfg["training"]["eval_heuristic"] == "tok_v2_mcts"
    cfg = find_game_config(GAME)
    cfg["max_game_plies"] = 24
    train_cfg["heuristic"], _ = find_heuristic_config(GAME)
    train_cfg["network"] = {**train_cfg["network"], "d_model": 32, "num_layers": 2, "num_heads": 4, "ffn_dim": 64}
    train_cfg["heuristic_guidance"] = {"name": "rule", "hold_steps": 1, "steps": 2, "initial_ratio": 1.0,
                                       "temperature": 0.0}
    train_cfg["training"] = {**train_cfg["training"], "train_batches_per_step": 2, "inference_precision": "fp32",
                             "eval_heuristic": "rule",
                             "playout_cap": {"full_simulations": 4, "fast_simulations": 2, "full_search_prob": 0.5}}
    run_training_loop(game_id=GAME, game_config=cfg, train_config=train_cfg, output_dir=tmp_path, steps=2,
                      episodes_per_step=2, eval_every=0, eval_games=2, max_workers=1, batch_size=16,
                      learning_rate=0.001, seed=5, save_every=0)
    latest = tmp_path / "models" / "model_latest.onnx"
    ep = engine.run_selfplay_episode(game_id=GAME, seed=3, model_path=str(latest), simulations=4, max_game_plies=12)
    assert ep["samples"]


def _structure(x):
    """The tokens with their region ids replaced by the tokens they name: equal for any id labels."""
    content = [tuple(t[:KEY].tolist()) for t in x]
    by_id = {int(t[KEY]): content[i] for i, t in enumerate(x) if t[KEY]}
    return sorted((content[i], tuple(sorted(by_id[int(v)] for v in t[KEY + 1:] if v))) for i, t in enumerate(x))


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
                assert _structure(_tokens(feats[0], DIM)) == _structure(_tokens(b["features"], DIM)), (g, me)
                np.testing.assert_array_equal(out[0][0], np.asarray(b["legal_mask"], dtype=np.float32))
                checked += 1
    assert checked > 50
