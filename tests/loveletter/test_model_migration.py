"""games/loveletter/migrate_model_features.py — carrying nets trained with
the two dead encoder inputs over the encoder fix.

The tool must (a) leave the net's outputs unchanged when the old constant
inputs are replaced by the real ones, (b) be idempotent, (c) refuse models
it does not understand, and (d) have been applied to the shipped models.
"""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

import dinoboard_engine as engine
from conftest import get_test_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL = PROJECT_ROOT / "games" / "loveletter" / "migrate_model_features.py"
MODEL_DIR = PROJECT_ROOT / "games" / "loveletter" / "model"

_spec = importlib.util.spec_from_file_location("ll_migrate", TOOL)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

GAMES = {"loveletter": 2, "loveletter_3p": 3, "loveletter_4p": 4}


def _run(path, feats):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess.run(None, {"features": feats.astype(np.float32)})


def _real_features(game_id, count):
    """Encoder outputs at search roots of heuristic games (current encoder)."""
    rows = []
    seed = 0
    while len(rows) < count:
        ep = engine.run_heuristic_episode(
            game_id, seed=seed, temperature=1.0, max_game_plies=40, heuristic_selector="rule_v1")
        rows.extend(np.asarray(s["features"], dtype=np.float32) for s in ep["samples"])
        seed += 1
    return np.stack(rows[:count])


def _as_old_encoder(feats, n):
    """What the pre-fix encoder produced for the same positions."""
    old = feats.copy()
    old[:, 13 * n] = -64.0
    old[:, 3] = 0.0
    return old


@pytest.mark.parametrize("game_id", sorted(GAMES))
def test_migrated_net_reproduces_the_old_net_on_old_inputs(game_id, tmp_path):
    n = GAMES[game_id]
    original = Path(get_test_model(game_id))          # random PVNet, never migrated
    migrated = tmp_path / "migrated.onnx"
    assert tool.migrate(original, migrated) is True

    new_feats = _real_features(game_id, 400)
    assert len(set(new_feats[:, 13 * n])) > 3         # deck size really varies now
    if n == 2:
        assert new_feats[:, 3].max() == 1.0           # and the own-hand-seen bit fires
    old_feats = _as_old_encoder(new_feats, n)

    want_policy, want_value = _run(original, old_feats)
    got_policy, got_value = _run(migrated, new_feats)
    assert np.abs(got_policy - want_policy).max() < 1e-4
    assert np.abs(got_value - want_value).max() < 1e-5

    # Control: without migration the real inputs do change the old net's output.
    raw_policy, _ = _run(original, new_feats)
    assert np.abs(raw_policy - want_policy).max() > 1e-2


def test_migration_is_idempotent_and_tagged(tmp_path):
    original = Path(get_test_model("loveletter"))
    once = tmp_path / "once.onnx"
    assert tool.migrate(original, once) is True
    before = once.read_bytes()
    assert tool.migrate(once) is False                # second run: no-op
    assert once.read_bytes() == before
    props = {p.key: p.value for p in onnx.load(str(once)).metadata_props}
    assert props[tool.MIGRATION_KEY] == tool.MIGRATION_TAG


def test_only_the_two_columns_change(tmp_path):
    original = Path(get_test_model("loveletter_3p"))
    migrated = tmp_path / "m.onnx"
    tool.migrate(original, migrated)

    def tensors(path):
        return {i.name: onnx.numpy_helper.to_array(i) for i in onnx.load(str(path)).graph.initializer}

    a, b = tensors(original), tensors(migrated)
    assert a.keys() == b.keys()
    deck = tool.deck_size_index(99)
    for name in a:
        if name == "backbone.0.weight":
            keep = [c for c in range(99) if c not in (deck, 3)]
            assert np.array_equal(a[name][:, keep], b[name][:, keep])
            assert not b[name][:, [deck, 3]].any()
        elif name == "backbone.0.bias":
            assert np.allclose(b[name], a[name] - 64.0 * a["backbone.0.weight"][:, deck], atol=1e-5)
        else:
            assert np.array_equal(a[name], b[name])


def test_checkpoint_migration_matches_onnx_migration(tmp_path):
    from training.model import create_model_from_config

    cfg = {"feature_dim": 70, "action_space": 47, "num_players": 2}
    torch.manual_seed(5)
    net = create_model_from_config(cfg)
    ckpt = tmp_path / "checkpoint.pt"
    torch.save({"step": 3, "model_state_dict": net.state_dict(),
                "ema_state_dict": net.state_dict(), "optimizer_state_dict": {}}, ckpt)
    w0 = net.state_dict()["backbone.0.weight"].numpy().copy()
    b0 = net.state_dict()["backbone.0.bias"].numpy().copy()

    assert tool.migrate(ckpt) is True
    assert tool.migrate(ckpt) is False
    loaded = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    assert loaded["step"] == 3 and loaded[tool.MIGRATION_KEY] == tool.MIGRATION_TAG
    for key in ("model_state_dict", "ema_state_dict"):
        w = loaded[key]["backbone.0.weight"].numpy()
        b = loaded[key]["backbone.0.bias"].numpy()
        assert not w[:, [26, 3]].any()
        assert np.allclose(b, b0 - 64.0 * w0[:, 26], atol=1e-5)
        net.load_state_dict(loaded[key])              # still loads into the same architecture


def test_unknown_models_are_rejected(tmp_path):
    wrong_dim = Path(get_test_model("coup"))          # 143 features: not a Love Letter layout
    with pytest.raises(ValueError, match="Love Letter encoder layout"):
        tool.migrate(wrong_dim, tmp_path / "x.onnx")
    junk = tmp_path / "model.bin"
    junk.write_bytes(b"x")
    with pytest.raises(ValueError, match="expected an .onnx"):
        tool.migrate(junk)


@pytest.mark.parametrize("game_id", sorted(GAMES))
def test_shipped_models_are_migrated(game_id):
    """The models in games/loveletter/model were trained before the encoder
    fix; they must carry the migration tag (and so ignore the two inputs
    they never learned)."""
    path = MODEL_DIR / f"loveletter_{GAMES[game_id]}p.onnx"
    model = onnx.load(str(path))
    props = {p.key: p.value for p in model.metadata_props}
    assert props.get(tool.MIGRATION_KEY) == tool.MIGRATION_TAG, f"{path} was not migrated"
    w = next(onnx.numpy_helper.to_array(i) for i in model.graph.initializer if i.name == "backbone.0.weight")
    dim = engine.game_metadata(game_id)["feature_dim"]
    assert w.shape[1] == dim
    assert not w[:, [tool.deck_size_index(dim), 3]].any()


def test_tool_leaves_a_shipped_model_untouched(tmp_path):
    copy = tmp_path / "loveletter_2p.onnx"
    shutil.copy(MODEL_DIR / "loveletter_2p.onnx", copy)
    before = copy.read_bytes()
    assert tool.migrate(copy) is False
    assert copy.read_bytes() == before
