"""carcassonne_tok: CarcassonneTokenEncoder (variable-length) + relative-position TokenTransformerPVNet.

Covers the layout contract (every action bound by anchor, the current tile
anchored past the board cells), token well-formedness (positions match the
anchored cell, absolute coordinates zeroed), legality flags against the
rules, the end-of-game score projection, translation invariance of the
network on real states, ONNX / fp16 parity and weight round-trip,
rejected network configs, and concurrent selfplay in compact-sample mode.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import dinoboard_engine as engine
from training.model import TokenTransformerPVNet, create_model_from_config, export_onnx, padded_token_count

from . import token_reference as tok

GAME = "carcassonne_tok"
F = 7                       # features start after [valid, entity, anchor, type, has_pos, x, y]
TOKEN_FEATURES = 9 + 13 * 6  # the current-tile block, the longest
GLOBAL, CURRENT, TILE, FRONTIER = range(4)
COLS = 20
CURRENT_ANCHOR = 401
MEEPLE_PASS = 1600
MEEPLE_BASE = 1601
HEADS = 4 + 13
NET_CFG = {"type": "token_transformer", "d_model": 32, "num_layers": 2, "num_heads": 4, "ffn_dim": 64,
           "relative_position_radius": 4, "anchor_embedding": False}


def _meta():
    return engine.game_metadata(GAME)


def _net(seed=0, attention="matmul"):
    meta = _meta()
    torch.manual_seed(seed)
    net = create_model_from_config(
        {**NET_CFG, "attention": attention}, input_dim=meta["feature_dim"], policy_dim=meta["action_space"],
        num_players=meta["num_players"], token_layout=meta["token_layout"])
    assert isinstance(net, TokenTransformerPVNet)
    with torch.no_grad():
        net.rel_bias.normal_(0.0, 1.0)
    return net.eval()


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("carc_tok") / "tok.onnx"
    export_onnx(_net(seed=5, attention="fused"), path, 0)
    return str(path)


@pytest.fixture(scope="module")
def episodes(model_path):
    eps = [engine.run_selfplay_episode(game_id=GAME, seed=300 + i, model_path=model_path,
                                       simulations=4, max_game_plies=200) for i in range(2)]
    assert all(ep["samples"] for ep in eps)
    return eps


def _tokens(sample):
    return np.asarray(sample["features"], dtype=np.float32).reshape(-1, _meta()["token_layout"]["token_dim"])


def _batch(xs):
    n = padded_token_count(max(len(x) for x in xs))
    out = np.zeros((len(xs), n, xs[0].shape[1]), dtype=np.float32)
    for i, x in enumerate(xs):
        out[i, :len(x)] = x
    return torch.from_numpy(out.reshape(len(xs), -1))


def test_layout_contract():
    meta = _meta()
    lay = meta["token_layout"]
    assert meta["feature_dim"] == 0 and lay["variable_tokens"] and lay["token_positions"]
    assert lay["token_dim"] == F + TOKEN_FEATURES
    assert lay["num_anchor_ids"] == CURRENT_ANCHOR and lay["num_pointer_heads"] == HEADS
    assert set(lay["action_token"]) == {-1} and set(lay["action_head"]) == {-1}
    aa = lay["anchor_action"]
    for cell in (0, 37, 399):
        assert aa[cell * HEADS:(cell + 1) * HEADS] == [cell * 4 + h for h in range(4)] + [-1] * 13
    assert aa[(CURRENT_ANCHOR - 1) * HEADS:] == [-1] * 4 + [MEEPLE_BASE + p for p in range(13)]


def test_tokens_well_formed_and_flags_match_legal_actions(episodes):
    n_place = n_meeple = 0
    for ep in episodes:
        for s in ep["samples"]:
            x = _tokens(s)
            legal = set(np.nonzero(np.asarray(s["legal_mask"]))[0].tolist())
            kind = x[:, 3].astype(int)
            assert (x[:, 0] == 1).all() and np.isfinite(x).all() and np.abs(x[:, F:]).max() <= 2.0
            assert kind[0] == GLOBAL and kind[1] == CURRENT and set(kind[2:]) <= {TILE, FRONTIER}
            assert x[0, 2] == 0 and x[0, 4] == 0 and x[1, 2] == CURRENT_ANCHOR
            board = x[2:]
            cells = board[:, 2].astype(int) - 1
            assert cells.min() >= 0 and len(np.unique(cells)) == len(cells)
            assert (board[:, 4] == 1).all()
            np.testing.assert_array_equal(board[:, 5:7], np.stack([cells % COLS, cells // COLS], 1))
            assert not board[kind[2:] == TILE][:, F + 4:F + 6].any(), "tile coords zeroed"
            assert not board[kind[2:] == FRONTIER][:, F:F + 2].any(), "frontier coords zeroed"
            assert not x[1, F + 6:F + 8].any(), "current-tile coords zeroed"
            if x[0, F + 9] == 1:
                n_place += 1
                assert x[1, 4] == 0, "the tile to place has no position yet"
                enc = {int(t[2] - 1) * 4 + r for t in board[kind[2:] == FRONTIER]
                       for r in range(4) if t[F + tok.FR_FITS + r] == 1}
            else:
                n_meeple += 1
                last = board[(kind[2:] == TILE) & (board[:, F + 6] == 1)]
                assert len(last) == 1 and x[1, 4] == 1 and (x[1, 5:7] == last[0, 5:7]).all()
                enc = {MEEPLE_PASS} | {MEEPLE_BASE + p for p in range(13) if x[1, F + 9 + 6 * p] == 1}
            assert enc == legal, f"ply {s['ply']}: encoder/rules disagree on {sorted(enc ^ legal)[:8]}"
    assert n_place > 50 and n_meeple > 50


def test_every_feature_matches_independent_reference(episodes):
    """Replay the episodes on a GameSession (same seed, same draws) and
    recompute each sampled state's tokens from its state dict."""
    n_outcomes = 0
    for i, ep in enumerate(episodes):
        gs = engine.GameSession(GAME, seed=300 + i)
        for j, s in enumerate(ep["samples"]):
            if j % 4 == i:
                d = gs.get_state_dict()
                legal = gs.get_legal_actions()
                assert set(legal) == set(np.nonzero(np.asarray(s["legal_mask"]))[0].tolist()), "replay out of sync"
                x = _tokens(s)
                kind = x[:, 3].astype(int)
                rows = x[2:]
                exp = tok.expected_tokens(d, legal, s["player"])
                problems = tok.differences(
                    exp, x[0, F:], (int(x[1, 1]), x[1, F:]),
                    {int(t[2]) - 1: (int(t[1]), t[F:]) for t in rows[kind[2:] == TILE]},
                    {int(t[2]) - 1: t[F:] for t in rows[kind[2:] == FRONTIER]})
                assert not problems, f"episode {i} ply {s['ply']}: {problems[:3]}"
                n_outcomes += sum(int(f[tok.FR_FITS:tok.FR_FITS + 4].sum()) for f in exp["frontier"].values())
            gs.apply_action(s["action_id"])
    assert n_outcomes > 200, "placement outcomes were barely exercised"


def test_projection_matches_final_score_when_last_tile_passes(episodes):
    checked = 0
    for ep in episodes:
        last = ep["samples"][-1]
        x = _tokens(last)
        if x[0, F + 10] != 1 or last["action_id"] != MEEPLE_PASS:
            continue
        me = last["player"]
        got = np.rint(x[0, F + 3:F + 5] * 100).astype(int).tolist()
        final = [int(v) for v in ep["final_scores"]]
        assert got == [final[me], final[1 - me]]
        checked += 1
    if checked == 0:
        pytest.skip("no episode ended with a meeple pass")


def test_network_invariant_to_translation_and_token_order(episodes):
    net = _net(seed=1)
    rng = np.random.default_rng(0)
    for s in episodes[0]["samples"][::25]:
        x = _tokens(s)
        shifted = x.copy()
        shifted[x[:, 4] == 1, 5:7] += [3, -2]
        perm = np.concatenate([[0, 1], 2 + rng.permutation(len(x) - 2)])
        with torch.no_grad():
            pol, val = net(_batch([x, shifted, x[perm]]))
        for i in (1, 2):
            torch.testing.assert_close(val[i], val[0], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(pol[i], pol[0], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("attention", ["matmul", "fused"])
def test_onnx_matches_torch(tmp_path, episodes, attention):
    import onnxruntime as ort
    xs = [_tokens(s) for s in episodes[1]["samples"][::10]]
    feats = _batch(xs)
    net = _net(seed=2, attention=attention)
    path = tmp_path / "tok.onnx"
    export_onnx(net, path, 0)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    pol_o, val_o = sess.run(["policy", "value"], {"features": feats.numpy()})
    with torch.no_grad():
        pol_t, val_t = net(feats)
    np.testing.assert_allclose(pol_o, pol_t.numpy(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(val_o, val_t.numpy(), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("attention", ["matmul", "fused"])
def test_fp16_onnx_loads_and_matches(tmp_path, episodes, attention):
    import onnxruntime as ort
    from training.model import convert_onnx_fp16
    samples = episodes[1]["samples"][::10]
    feats = _batch([_tokens(s) for s in samples]).numpy()
    legal = np.stack([np.asarray(s["legal_mask"]) for s in samples]) > 0
    f32, f16 = tmp_path / "tok.onnx", tmp_path / "tok16.onnx"
    export_onnx(_net(seed=2, attention=attention), f32, 0)
    convert_onnx_fp16(f32, f16)
    a = ort.InferenceSession(str(f32), providers=["CPUExecutionProvider"]).run(None, {"features": feats})
    b = ort.InferenceSession(str(f16), providers=["CPUExecutionProvider"]).run(None, {"features": feats})
    assert np.abs(a[0] - b[0])[legal].max() < 1e-2
    assert np.abs(a[1] - b[1]).max() < 1e-2


@pytest.mark.parametrize("attention", ["matmul", "fused"])
def test_onnx_weights_import_back_exactly(tmp_path, attention):
    from training.model import import_pv_onnx_state_dict
    src = _net(seed=3, attention=attention)
    with torch.no_grad():
        for p in src.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    path = tmp_path / "tok.onnx"
    export_onnx(src, path, 0)
    dst = _net(seed=4, attention=attention)
    dst.load_state_dict(import_pv_onnx_state_dict(path, dst))
    for k, v in src.state_dict().items():
        assert torch.equal(v, dst.state_dict()[k]), k


def test_unknown_attention_is_rejected():
    meta = _meta()
    with pytest.raises(ValueError, match="attention must be one of"):
        create_model_from_config({**NET_CFG, "attention": "flash"}, input_dim=0,
                                 policy_dim=meta["action_space"], num_players=2,
                                 token_layout=meta["token_layout"])


def test_anchor_binding_conflicts_are_rejected():
    meta = _meta()
    lay = dict(meta["token_layout"])
    lay["anchor_action"] = list(lay["anchor_action"])
    lay["anchor_action"][4] = MEEPLE_BASE          # already bound to the current-tile anchor
    with pytest.raises(ValueError, match="bound more than once"):
        create_model_from_config(NET_CFG, input_dim=0, policy_dim=meta["action_space"],
                                 num_players=2, token_layout=lay)
    lay["anchor_action"] = lay["anchor_action"][:-1]
    with pytest.raises(ValueError, match="anchor_action has"):
        create_model_from_config(NET_CFG, input_dim=0, policy_dim=meta["action_space"],
                                 num_players=2, token_layout=lay)


def test_concurrent_selfplay_compact_samples(model_path):
    kwargs = dict(game_id=GAME, seeds=[7], latest_model_path=model_path, num_self_threads=1,
                  simulations=8, max_game_plies=12, batch_size=16, max_queue_size=64,
                  intra_op_num_threads=1, leaf_batch_size=2)
    lists = engine.run_concurrent_selfplay_step(**kwargs)["episodes"][0]["samples"]
    compact = engine.run_concurrent_selfplay_step(compact_samples=True, **kwargs)["episodes"][0]["samples"]
    assert len(lists) == len(compact) > 0
    for a, b in zip(lists, compact):
        assert isinstance(b["features"], np.ndarray) and b["features"].dtype == np.float32
        assert isinstance(b["legal_mask"], np.ndarray) and b["legal_mask"].dtype == np.uint8
        np.testing.assert_array_equal(np.asarray(a["features"], dtype=np.float32), b["features"])
        np.testing.assert_array_equal(np.asarray(a["legal_mask"]) != 0, b["legal_mask"] == 1)


@pytest.mark.parametrize("name", ["train_transformer.json", "train_transformer_guided.json"])
def test_train_config_builds_model(name):
    from training.cli import validate_guidance_until
    path = Path(__file__).resolve().parents[2] / "games/carcassonne/config" / name
    cfg = json.loads(path.read_text(encoding="utf-8"))
    validate_guidance_until(cfg["heuristic_guidance"], path)
    meta = _meta()
    net = create_model_from_config(cfg["network"], input_dim=0, policy_dim=meta["action_space"],
                                   num_players=meta["num_players"], token_layout=meta["token_layout"])
    assert net.token_positions and not hasattr(net, "anchor_embed")
