"""Token relations (TokenLayout.relation_*): attention adds a learned bias per (type, type, head) for
pairs of tokens one of which links to the other's key id.

* At initialization the bias is zero and the id columns are not features: outputs ignore them.
* Ids are names: relabelling them consistently changes nothing; a link from i to j acts as one
  from j to i; linking two tokens changes the output once the bias is trained.
* The exported ONNX graph (matmul / fused attention) and its fp16 copy match torch, and the
  weights import back from ONNX.
* Bad relation columns are rejected.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from training.model import TokenTransformerPVNet, convert_onnx_fp16, export_onnx, import_pv_onnx_state_dict

FEATS = 6
KEY = 7 + FEATS
LINKS = 3
TOKEN_DIM = KEY + 1 + LINKS
MAX_ID = 5
NUM_TYPES = 3
NUM_ANCHORS = 6
HEADS = 2
POLICY = 1 + NUM_ANCHORS * HEADS


def layout(**over):
    base = {
        "num_tokens": 0, "token_dim": TOKEN_DIM, "num_token_types": NUM_TYPES, "num_entity_ids": 9,
        "num_pointer_heads": HEADS, "token_types": [], "action_token": [-1] * POLICY,
        "action_head": [-1] * POLICY, "num_anchor_ids": NUM_ANCHORS,
        "anchor_action": list(range(1, POLICY)), "variable_tokens": True, "token_positions": True,
        "relation_key_column": KEY, "relation_link_column": KEY + 1, "relation_link_count": LINKS,
        "relation_max_id": MAX_ID,
    }
    return {**base, **over}


def net(attention="matmul", trained=True):
    torch.manual_seed(0)
    m = TokenTransformerPVNet(layout(), POLICY, 2, d_model=32, num_layers=2, num_heads=4, ffn_dim=64,
                              attention=attention, relative_position_radius=2, anchor_embedding=False)
    if trained:
        with torch.no_grad():
            m.relation_bias.normal_(0.0, 1.0)
    return m.eval()


def tokens(seed, n=6):
    """n tokens; 0..2 carry keys 1..3, token 3 links to keys 1 and 3, token 4 to key 2."""
    rng = np.random.default_rng(seed)
    t = np.zeros((n, TOKEN_DIM), dtype=np.float32)
    t[:, 0] = 1
    t[:, 1] = rng.integers(0, 9, n)
    t[:, 2] = rng.permutation(np.arange(1, NUM_ANCHORS + 1))[:n]
    t[:, 3] = rng.integers(0, NUM_TYPES, n)
    t[:, 4] = 1
    t[:, 5:7] = rng.integers(0, 4, (n, 2))
    t[:, 7:KEY] = rng.standard_normal((n, FEATS))
    t[0:3, KEY] = [1, 2, 3]
    t[3, KEY + 1:KEY + 3] = [1, 3]
    t[4, KEY + 1] = 2
    return t


def padded(t, n=7):
    out = np.zeros((n, TOKEN_DIM), dtype=np.float32)
    out[:len(t)] = t
    return out.reshape(1, -1)


def run(m, *encodings):
    with torch.no_grad():
        return m(torch.from_numpy(np.concatenate(encodings)))


def test_untrained_relations_and_id_columns_change_nothing():
    m = net(trained=False)
    t = tokens(1)
    no_ids = t.copy()
    no_ids[:, KEY:] = 0
    other = t.copy()
    other[:, KEY:] = np.random.default_rng(9).integers(0, MAX_ID + 1, (len(t), 1 + LINKS))
    p, v = run(m, *(padded(x) for x in (t, no_ids, other)))
    assert torch.allclose(p[1], p[0], atol=1e-6) and torch.allclose(p[2], p[0], atol=1e-6)
    assert torch.allclose(v[1], v[0], atol=1e-6) and torch.allclose(v[2], v[0], atol=1e-6)


def test_ids_are_names_and_links_are_symmetric():
    m = net()
    t = tokens(2)
    relabel = t.copy()
    perm = {0: 0, 1: 4, 2: 5, 3: 1}
    relabel[:, KEY:] = np.vectorize(perm.get)(t[:, KEY:].astype(int))
    # Token 4 linking to token 1's key 2 is the same relation as token 1 linking to token 4.
    reverse = t.copy()
    reverse[4, KEY + 1] = 0
    reverse[4, KEY] = 4
    reverse[1, KEY + 1] = 4
    unlinked = t.copy()
    unlinked[4, KEY + 1] = 0
    p, v = run(m, *(padded(x) for x in (t, relabel, reverse, unlinked)))
    assert torch.allclose(p[1], p[0], atol=1e-5) and torch.allclose(v[1], v[0], atol=1e-5)
    assert torch.allclose(p[2], p[0], atol=1e-5) and torch.allclose(v[2], v[0], atol=1e-5)
    assert not torch.allclose(p[3], p[0], atol=1e-3), "removing a link must change the output"


def test_bias_reaches_only_linked_pairs_by_type():
    m = net()
    t = tokens(3)
    kinds = torch.eye(NUM_TYPES)[torch.from_numpy(t[:, 3]).long()].unsqueeze(0)
    bias = m._relation_bias(torch.from_numpy(t).unsqueeze(0), kinds)[0]
    assert bias.shape == (4, len(t) + 1, len(t) + 1)
    w = m.relation_bias.detach().reshape(NUM_TYPES, NUM_TYPES, 4)
    ty = t[:, 3].astype(int)
    linked = {(3, 0), (3, 2), (4, 1)}
    for i in range(len(t)):
        for j in range(len(t)):
            expected = w[ty[i], ty[j]] if (i, j) in linked or (j, i) in linked else torch.zeros(4)
            assert torch.allclose(bias[:, 1 + i, 1 + j], expected, atol=1e-6), (i, j)
    assert bias[:, 0].abs().sum() == 0 and bias[:, :, 0].abs().sum() == 0, "CLS has no relations"


@pytest.mark.parametrize("attention", ["matmul", "fused"])
def test_onnx_and_fp16_match_torch(tmp_path, attention):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxconverter_common")
    m = net(attention)
    src = export_onnx(m, tmp_path / "m.onnx", 0)
    dst = convert_onnx_fp16(Path(src), tmp_path / "fp16" / "m.onnx")
    x = np.concatenate([padded(tokens(s, n)) for s, n in ((4, 6), (5, 5))])
    with torch.no_grad():
        tp, tv = m(torch.from_numpy(x))
    for path, atol in ((src, 1e-4), (dst, 5e-2)):
        policy, value = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"features": x})
        assert np.allclose(policy, tp.numpy(), atol=atol), path
        assert np.allclose(value, tv.numpy(), atol=atol), path


def test_weights_import_from_onnx(tmp_path):
    m = net("fused")
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    export_onnx(m, tmp_path / "m.onnx", 0)
    fresh = TokenTransformerPVNet(layout(), POLICY, 2, d_model=32, num_layers=2, num_heads=4, ffn_dim=64,
                                  attention="fused", relative_position_radius=2, anchor_embedding=False)
    fresh.load_state_dict(import_pv_onnx_state_dict(tmp_path / "m.onnx", fresh))
    for k, v in m.state_dict().items():
        assert torch.equal(v, fresh.state_dict()[k]), k


def test_bad_relation_columns_are_rejected():
    kw = dict(d_model=32, num_layers=1, num_heads=4, ffn_dim=64, relative_position_radius=2)
    for bad in (layout(relation_link_column=KEY), layout(relation_link_count=0), layout(relation_max_id=0),
                layout(relation_key_column=3), layout(relation_link_column=TOKEN_DIM - 1)):
        with pytest.raises(ValueError, match="bad relation columns"):
            TokenTransformerPVNet(bad, POLICY, 2, **kw)
    with pytest.raises(ValueError, match="need a variable_tokens layout"):
        TokenTransformerPVNet(layout(variable_tokens=False, token_positions=False, num_tokens=2, token_types=[0, 1]),
                              POLICY, 2, d_model=32, num_layers=1, num_heads=4, ffn_dim=64)
