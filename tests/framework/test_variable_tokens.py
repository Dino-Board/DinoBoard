"""Variable-length token transformer (TokenLayout.variable_tokens): outputs do not depend on padding,
the exported ONNX graph takes any token count and matches torch, and weights import back from ONNX.
With token_positions, attention depends on positions only through clipped offsets."""

from pathlib import Path

import numpy as np
import pytest
import torch

from training.model import (TokenTransformerPVNet, convert_onnx_fp16, export_onnx, import_pv_onnx_state_dict,
                            padded_token_count)

TOKEN_DIM = 4 + 6
NUM_TYPES = 3
NUM_ANCHORS = 5
HEADS = 2
POLICY = 1 + NUM_ANCHORS * HEADS  # action 0 global, then (anchor k, head h) -> 1 + (k-1)*HEADS + h


def layout():
    return {
        "num_tokens": 0, "token_dim": TOKEN_DIM, "num_token_types": NUM_TYPES, "num_entity_ids": 9,
        "num_pointer_heads": HEADS, "token_types": [], "action_token": [-1] * POLICY,
        "action_head": [-1] * POLICY, "num_anchor_ids": NUM_ANCHORS,
        "anchor_action": list(range(1, POLICY)), "variable_tokens": True,
    }


def net(attention="matmul"):
    torch.manual_seed(0)
    return TokenTransformerPVNet(layout(), POLICY, 2, d_model=32, num_layers=2, num_heads=4, ffn_dim=64,
                                 attention=attention).eval()


def tokens(n, seed):
    rng = np.random.default_rng(seed)
    t = np.zeros((n, TOKEN_DIM), dtype=np.float32)
    t[:, 0] = 1
    t[:, 1] = rng.integers(0, 9, n)
    t[:, 2] = rng.permutation(np.arange(1, NUM_ANCHORS + 1))[:n] if n <= NUM_ANCHORS else 0
    t[:, 3] = rng.integers(0, NUM_TYPES, n)
    t[:, 4:] = rng.standard_normal((n, TOKEN_DIM - 4))
    return t


def padded(t, n):
    out = np.zeros((n, TOKEN_DIM), dtype=np.float32)
    out[:len(t)] = t
    return out.reshape(1, -1)


def test_padded_token_count():
    assert [padded_token_count(n) for n in (1, 7, 8, 15, 16)] == [7, 7, 15, 15, 23]


def test_outputs_ignore_padding_and_anchor_scores_follow_tokens():
    m = net()
    t = tokens(4, 1)
    with torch.no_grad():
        outs = [m(torch.from_numpy(padded(t, n))) for n in (4, 7, 23)]
        swapped = m(torch.from_numpy(padded(t[[2, 0, 3, 1]], 7)))
    for p, v in outs[1:] + [swapped]:
        assert torch.allclose(p, outs[0][0], atol=1e-5)
        assert torch.allclose(v, outs[0][1], atol=1e-5)


def test_static_bindings_rejected():
    bad = layout()
    bad["action_token"][0], bad["action_head"][0] = 0, 0
    with pytest.raises(ValueError, match="anchor only"):
        TokenTransformerPVNet(bad, POLICY, 2, d_model=32, num_layers=1, num_heads=4, ffn_dim=64)


@pytest.mark.parametrize("attention", ["matmul", "fused"])
def test_onnx_any_length_matches_torch(tmp_path, attention):
    ort = pytest.importorskip("onnxruntime")
    m = net(attention)
    path = export_onnx(m, tmp_path / "m.onnx", 0)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    for n_tok, n_pad in ((3, 7), (9, 15), (20, 23)):
        x = np.concatenate([padded(tokens(min(n_tok, NUM_ANCHORS), s), n_pad) for s in (2, 3)])
        if n_tok > NUM_ANCHORS:
            extra = tokens(n_tok, 4)
            extra[:, 2] = 0
            x = np.concatenate([x, padded(extra, n_pad)])
        policy, value = sess.run(None, {"features": x})
        with torch.no_grad():
            tp, tv = m(torch.from_numpy(x))
        assert np.allclose(policy, tp.numpy(), atol=1e-4)
        assert np.allclose(value, tv.numpy(), atol=1e-4)


@pytest.mark.parametrize("attention", ["matmul", "fused"])
def test_fp16_copy_loads_and_matches(tmp_path, attention):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxconverter_common")
    m = net(attention)
    src = export_onnx(m, tmp_path / "m.onnx", 0)
    dst = convert_onnx_fp16(Path(src), tmp_path / "fp16" / "m.onnx")
    sess = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    x = np.concatenate([padded(tokens(4, s), 7) for s in (6, 7)])
    policy, value = sess.run(None, {"features": x})
    with torch.no_grad():
        tp, tv = m(torch.from_numpy(x))
    assert np.allclose(policy, tp.numpy(), atol=2e-2)
    assert np.allclose(value, tv.numpy(), atol=2e-2)


def test_weights_import_from_onnx(tmp_path):
    m = net("fused")
    with torch.no_grad():  # distinct values, as after training (the exporter merges equal constants)
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    path = export_onnx(m, tmp_path / "m.onnx", 0)
    fresh = TokenTransformerPVNet(layout(), POLICY, 2, d_model=32, num_layers=2, num_heads=4, ffn_dim=64,
                                  attention="fused")
    fresh.load_state_dict(import_pv_onnx_state_dict(tmp_path / "m.onnx", fresh))
    x = torch.from_numpy(padded(tokens(4, 5), 7))
    with torch.no_grad():
        assert torch.equal(fresh.eval()(x)[0], m(x)[0])
    assert path


POS_DIM = 7 + 6
RADIUS = 2


def pos_layout():
    return {**layout(), "token_dim": POS_DIM, "token_positions": True}


def pos_net(attention="matmul", anchor_embedding=False):
    torch.manual_seed(0)
    m = TokenTransformerPVNet(pos_layout(), POLICY, 2, d_model=32, num_layers=2, num_heads=4, ffn_dim=64,
                              attention=attention, relative_position_radius=RADIUS,
                              anchor_embedding=anchor_embedding)
    with torch.no_grad():  # a trained-looking table, so positions visibly matter
        m.rel_bias.normal_(0.0, 1.0)
    return m.eval()


def pos_tokens(n, seed, positioned=None):
    """Tokens on a grid; `positioned` (default: all but the last) get a position."""
    base = tokens(n, seed)
    rng = np.random.default_rng(seed + 100)
    t = np.zeros((n, POS_DIM), dtype=np.float32)
    t[:, :4] = base[:, :4]
    t[:, 7:] = base[:, 4:]
    positioned = n - 1 if positioned is None else positioned
    t[:positioned, 4] = 1
    t[:positioned, 5:7] = rng.integers(0, 5, (positioned, 2))
    return t


def pos_padded(t, n):
    out = np.zeros((n, POS_DIM), dtype=np.float32)
    out[:len(t)] = t
    return out.reshape(1, -1)


def run(m, *encodings):
    with torch.no_grad():
        return m(torch.from_numpy(np.concatenate(encodings)))


def test_relative_positions_are_translation_invariant_and_clipped():
    m = pos_net()
    t = pos_tokens(5, 1)
    shifted = t.copy()
    shifted[:4, 5:7] += [7, -3]
    moved = t.copy()
    moved[0, 5:7] += [1, 0]
    far, farther = t.copy(), t.copy()
    far[0, 5:7] = t[1:4, 5:7].max(0) + RADIUS
    farther[0, 5:7] = t[1:4, 5:7].max(0) + RADIUS + 5
    p, v = run(m, *(pos_padded(x, 7) for x in (t, shifted, moved, far, farther)))
    assert torch.allclose(p[1], p[0], atol=1e-5) and torch.allclose(v[1], v[0], atol=1e-5)
    assert not torch.allclose(p[2], p[0], atol=1e-3), "an offset change must change the output"
    assert torch.allclose(p[4], p[3], atol=1e-5), "offsets beyond the radius share a bucket"
    unpositioned = t.copy()
    unpositioned[4, 5:7] = [9, 9]  # ignored: token 4 has no position
    assert torch.allclose(run(m, pos_padded(unpositioned, 7))[0], p[0:1], atol=1e-5)


def test_relative_position_config_is_validated():
    kw = dict(d_model=32, num_layers=1, num_heads=4, ffn_dim=64)
    with pytest.raises(ValueError, match="relative_position_radius must be set exactly"):
        TokenTransformerPVNet(pos_layout(), POLICY, 2, **kw)
    with pytest.raises(ValueError, match="relative_position_radius must be set exactly"):
        TokenTransformerPVNet(layout(), POLICY, 2, relative_position_radius=2, **kw)
    with pytest.raises(ValueError, match="requires a variable_tokens layout"):
        TokenTransformerPVNet({**pos_layout(), "variable_tokens": False, "num_tokens": 1, "token_types": [0]},
                              POLICY, 2, relative_position_radius=2, **kw)
    no_anchor = {**layout(), "num_anchor_ids": 0, "anchor_action": [], "action_token": [-1], "action_head": [-1]}
    with pytest.raises(ValueError, match="anchor_embedding=false needs a layout with anchors"):
        TokenTransformerPVNet(no_anchor, 1, 2, anchor_embedding=False, **kw)
    assert not hasattr(pos_net(), "anchor_embed") and hasattr(pos_net(anchor_embedding=True), "anchor_embed")


@pytest.mark.parametrize("attention", ["matmul", "fused"])
def test_relative_positions_onnx_and_fp16_match_torch(tmp_path, attention):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxconverter_common")
    m = pos_net(attention)
    src = export_onnx(m, tmp_path / "m.onnx", 0)
    dst = convert_onnx_fp16(Path(src), tmp_path / "fp16" / "m.onnx")
    x = np.concatenate([pos_padded(pos_tokens(n, s), 7) for n, s in ((3, 2), (5, 3))])
    with torch.no_grad():
        tp, tv = m(torch.from_numpy(x))
    for path, atol in ((src, 1e-4), (dst, 5e-2)):
        policy, value = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"features": x})
        assert np.allclose(policy, tp.numpy(), atol=atol), path
        assert np.allclose(value, tv.numpy(), atol=atol), path


def test_relative_position_weights_import_from_onnx(tmp_path):
    m = pos_net("fused")
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    export_onnx(m, tmp_path / "m.onnx", 0)
    fresh = TokenTransformerPVNet(pos_layout(), POLICY, 2, d_model=32, num_layers=2, num_heads=4, ffn_dim=64,
                                  attention="fused", relative_position_radius=RADIUS, anchor_embedding=False)
    fresh.load_state_dict(import_pv_onnx_state_dict(tmp_path / "m.onnx", fresh))
    for k, v in m.state_dict().items():
        assert torch.equal(v, fresh.state_dict()[k]), k


def test_relative_position_bias_gradient_matches_embedding(monkeypatch):
    from training import model as model_mod
    m = pos_net().train()
    x = torch.from_numpy(np.concatenate([pos_padded(pos_tokens(n, s), 7) for n, s in ((5, 4), (3, 5))]))

    def table_grad():
        m.zero_grad()
        policy, value = m(x)
        (policy.square().sum() + value.sum()).backward()
        return m.rel_bias.grad.clone()

    grad = table_grad()
    assert grad.abs().sum() > 0
    monkeypatch.setattr(model_mod._BucketGather, "apply", torch.nn.functional.embedding)
    assert torch.allclose(grad, table_grad(), atol=1e-5)
