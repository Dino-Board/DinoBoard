"""splendor_tok: token encoder (SplendorTokenEncoder) + TokenTransformerPVNet.

Covers the layout contract the Python network relies on, agreement with the
flat encoder on shared facts, the hidden-reserve masking, slot-permutation
equivariance of the network, ONNX parity, and an end-to-end concurrent
selfplay step (including the MLP pv_512x4 guidance heuristic, which must keep
using the flat encoder inside the token variant).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import dinoboard_engine as engine
from conftest import get_test_model
from training.model import TokenTransformerPVNet, create_model_from_config, export_onnx

GAME = "splendor_tok"
FLAT = "splendor"
P = 2
TAKE_THREE_012 = 30      # kTakeThreeOffset for 2p, colors {0,1,2}
TAKE_THREE_013 = 31
RESERVE_DECK_T1 = 24     # kReserveDeckOffset, tier 1
CHOOSE_NOBLE = 60        # kChooseNobleOffset for 2p
NET_CFG = {"type": "token_transformer", "d_model": 64, "num_layers": 2, "num_heads": 4, "ffn_dim": 128}

TABLEAU_BASE = 1 + P
RESERVED_BASE = TABLEAU_BASE + 12
NOBLE_BASE = RESERVED_BASE + 3 * P
DECK_BASE = NOBLE_BASE + P + 1


def _meta():
    return engine.game_metadata(GAME)


def _tokens(feats, layout):
    return np.asarray(feats, dtype=np.float32).reshape(layout["num_tokens"], layout["token_dim"])


def _net(seed=0):
    meta = _meta()
    torch.manual_seed(seed)
    net = create_model_from_config(
        NET_CFG, input_dim=meta["feature_dim"], policy_dim=meta["action_space"],
        num_players=meta["num_players"], token_layout=meta["token_layout"])
    assert isinstance(net, TokenTransformerPVNet)
    return net.eval()


def test_layout_contract():
    meta = _meta()
    lay = meta["token_layout"]
    assert engine.game_metadata(FLAT)["token_layout"] is None
    assert lay["num_tokens"] == DECK_BASE + 3
    assert lay["num_tokens"] * lay["token_dim"] == meta["feature_dim"]
    assert meta["action_space"] == engine.game_metadata(FLAT)["action_space"]
    types = lay["token_types"]
    assert types[0] == 0 and types[1:1 + P] == [1] * P
    assert types[TABLEAU_BASE:NOBLE_BASE] == [2] * (12 + 3 * P)
    assert types[NOBLE_BASE:DECK_BASE] == [3] * (P + 1) and types[DECK_BASE:] == [4] * 3
    at, ah = lay["action_token"], lay["action_head"]
    for i in range(12):
        assert (at[i], ah[i]) == (TABLEAU_BASE + i, 0)            # buy face-up
        assert (at[12 + i], ah[12 + i]) == (TABLEAU_BASE + i, 1)  # reserve face-up
    for t in range(3):
        assert (at[24 + t], ah[24 + t]) == (DECK_BASE + t, 2)
        assert (at[27 + t], ah[27 + t]) == (RESERVED_BASE + t, 0)  # own reserves
    assert all(at[a] == -1 and ah[a] == -1 for a in range(30, CHOOSE_NOBLE))
    for i in range(P + 1):
        assert (at[CHOOSE_NOBLE + i], ah[CHOOSE_NOBLE + i]) == (NOBLE_BASE + i, 3)
    assert all(at[a] == -1 and ah[a] == -1 for a in range(CHOOSE_NOBLE + P + 1, len(at)))


@pytest.mark.parametrize("seed", [3, 7, 11])
def test_cards_and_gems_agree_with_flat_encoder(seed):
    lay = _meta()["token_layout"]
    events = [(0, TAKE_THREE_012), (1, TAKE_THREE_013)]
    for persp in range(P):
        tok = _tokens(engine.encode_state_for_perspective(GAME, seed, persp, -1, events)["features"], lay)
        flat = np.asarray(engine.encode_state_for_perspective(FLAT, seed, persp, -1, events)["features"],
                          dtype=np.float32)
        tableau_off = 6 + 15 * P + 6 * (P + 1)
        for i in range(12):
            blk = flat[tableau_off + 13 * i: tableau_off + 13 * (i + 1)]
            t = tok[TABLEAU_BASE + i]
            assert t[0] == blk[0]
            if blk[0] == 0:
                assert not t.any()
                continue
            f = t[2:]
            assert 1 <= t[1] <= 90
            assert f[0] == 1.0 and not f[1:5].any()
            tier = int(np.argmax(f[7:10])) + 1
            assert np.isclose(blk[1], tier / 3.0)
            assert np.isclose(blk[2], f[10])
            np.testing.assert_allclose(blk[3:8], f[11:16])
            np.testing.assert_allclose(blk[8:13], f[16:21])
        for r in range(P):
            np.testing.assert_allclose(flat[6 + 15 * r: 6 + 15 * r + 6], tok[1 + r][2 + 6: 2 + 12])
        np.testing.assert_allclose(flat[0:6], tok[0][2:8])


def test_opponent_blind_reserve_is_masked():
    lay = _meta()["token_layout"]
    events = [(0, TAKE_THREE_012), (1, RESERVE_DECK_T1)]
    # Perspective 0 sees seat 1's blind reserve as its relative seat-1 row.
    t = _tokens(engine.encode_state_for_perspective(GAME, 5, 0, -1, events)["features"], lay)[RESERVED_BASE + 3]
    expected = np.zeros(lay["token_dim"], dtype=np.float32)
    expected[0] = 1.0
    f = expected[2:]
    f[1 + 1] = 1.0   # owner: relative seat 1
    f[5] = 1.0       # face down
    f[6] = 1.0       # identity unknown
    f[7] = 1.0       # tier 1 (public)
    np.testing.assert_array_equal(t, expected)
    # The owner sees the real card in its own row.
    own = _tokens(engine.encode_state_for_perspective(GAME, 5, 1, -1, events)["features"], lay)[RESERVED_BASE]
    assert own[0] == 1.0 and 1 <= own[1] <= 90
    assert own[2 + 1] == 1.0 and own[2 + 5] == 1.0 and own[2 + 6] == 0.0 and own[2 + 7] == 1.0


def _selfplay_features(n_plies=40):
    ep = engine.run_selfplay_episode(
        game_id=GAME, seed=21, model_path=get_test_model(GAME),
        simulations=8, max_game_plies=n_plies)
    feats = np.asarray([s["features"] for s in ep["samples"]], dtype=np.float32)
    assert len(feats) > 10
    return feats


def test_selfplay_features_are_well_formed():
    lay = _meta()["token_layout"]
    feats = _selfplay_features()
    assert np.isfinite(feats).all()
    tok = feats.reshape(len(feats), lay["num_tokens"], lay["token_dim"])
    assert set(np.unique(tok[:, :, 0]).tolist()) <= {0.0, 1.0}
    ent = tok[:, :, 1]
    assert (ent == np.round(ent)).all() and ent.min() >= 0 and ent.max() < lay["num_entity_ids"]
    assert not tok[tok[:, :, 0] == 0].any(), "invalid tokens must be all-zero"
    assert np.abs(tok[:, :, 2:]).max() <= 2.0


def test_network_equivariant_to_interchangeable_slots():
    lay = _meta()["token_layout"]
    T, F = lay["num_tokens"], lay["token_dim"]
    feats = _selfplay_features()
    net = _net()
    rng = np.random.default_rng(0)
    A = _meta()["action_space"]
    for x in feats[::3]:
        tok = x.reshape(T, F)
        slot_perm = np.arange(T)          # new slot j holds old slot slot_perm[j]
        act_perm = np.arange(A)           # new action j corresponds to old action act_perm[j]
        for tier in range(3):
            p = rng.permutation(4)
            for j in range(4):
                slot_perm[TABLEAU_BASE + tier * 4 + j] = TABLEAU_BASE + tier * 4 + p[j]
                act_perm[tier * 4 + j] = tier * 4 + p[j]
                act_perm[12 + tier * 4 + j] = 12 + tier * 4 + p[j]
        p = rng.permutation(3)
        for j in range(3):
            slot_perm[RESERVED_BASE + j] = RESERVED_BASE + p[j]
            act_perm[27 + j] = 27 + p[j]
        p = rng.permutation(P + 1)
        for j in range(P + 1):
            slot_perm[NOBLE_BASE + j] = NOBLE_BASE + p[j]
            act_perm[CHOOSE_NOBLE + j] = CHOOSE_NOBLE + p[j]
        xp = tok[slot_perm].reshape(-1)
        with torch.no_grad():
            pol, val = net(torch.from_numpy(np.stack([x, xp])))
        torch.testing.assert_close(val[1], val[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(pol[1], pol[0][torch.from_numpy(act_perm)], atol=1e-5, rtol=1e-5)


def test_onnx_matches_torch(tmp_path):
    import onnxruntime as ort
    feats = _selfplay_features()
    net = _net(seed=1)
    path = tmp_path / "tok.onnx"
    export_onnx(net, path, _meta()["feature_dim"])
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    pol_o, val_o = sess.run(["policy", "value"], {"features": feats})
    with torch.no_grad():
        pol_t, val_t = net(torch.from_numpy(feats))
    np.testing.assert_allclose(pol_o, pol_t.numpy(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(val_o, val_t.numpy(), atol=1e-4, rtol=1e-4)


def test_onnx_weights_import_back_exactly(tmp_path):
    """--resume without checkpoint.pt rebuilds weights from ONNX; transformer
    projections export as anonymous (transposed / constant-folded) initializers."""
    from training.model import import_pv_onnx_state_dict
    src = _net(seed=3)
    # Trained weights: fresh LayerNorms are identical and the exporter would
    # merge their initializers.
    with torch.no_grad():
        for p in src.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    path = tmp_path / "tok.onnx"
    export_onnx(src, path, _meta()["feature_dim"])
    dst = _net(seed=4)
    dst.load_state_dict(import_pv_onnx_state_dict(path, dst))
    for k, v in src.state_dict().items():
        assert torch.equal(v, dst.state_dict()[k]), k


def test_transformer_requires_token_layout():
    meta = engine.game_metadata(FLAT)
    with pytest.raises(ValueError, match="token layout"):
        create_model_from_config(NET_CFG, input_dim=meta["feature_dim"], policy_dim=meta["action_space"],
                                 num_players=2, token_layout=meta["token_layout"])
    with pytest.raises(ValueError, match="missing"):
        create_model_from_config({"type": "token_transformer", "d_model": 64}, input_dim=1, policy_dim=1,
                                 num_players=2, token_layout=_meta()["token_layout"])


def test_concurrent_selfplay_with_transformer_and_mlp_guidance(tmp_path):
    path = tmp_path / "tok.onnx"
    export_onnx(_net(seed=2), path, _meta()["feature_dim"])
    out = engine.run_concurrent_selfplay_step(
        game_id=GAME, seeds=[1, 2], latest_model_path=str(path), num_self_threads=2,
        simulations=32, max_game_plies=6, batch_size=32, max_queue_size=64,
        intra_op_num_threads=1, leaf_batch_size=4)
    assert sum(len(ep["samples"]) for ep in out["episodes"]) > 0
    # pv_512x4 is an MLP over the flat encoder; it must not be fed tokens.
    out = engine.run_concurrent_selfplay_step(
        game_id=GAME, seeds=[3], latest_model_path=str(path), num_self_threads=1,
        simulations=16, max_game_plies=3, batch_size=32, max_queue_size=64,
        intra_op_num_threads=1, heuristic_guidance_ratio=1.0, heuristic_selector="pv_512x4")
    assert len(out["episodes"]) == 1
