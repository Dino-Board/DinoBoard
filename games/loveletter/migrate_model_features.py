"""Carry a Love Letter policy/value net over the encoder fixes of two inputs.

Two inputs of `LoveLetterFeatureEncoder` used to be broken, so every net
trained before the fix only ever saw them as constants at its training
positions (search roots):

  * deck size (index 13*N): summed masked placeholders, i.e. the constant
    -64 instead of deck_size/16;
  * "my own hand has been seen" (index 3): always 0, because a seat's
    session never learned that an opponent had peeked.

The encoder now emits the real values at the same indices (feature dims are
unchanged). An old net has untrained weights on those two inputs, so feeding
it the real values would perturb it arbitrarily. This tool rewrites the first
layer so that the old net behaves exactly as it did at its training positions:

  * deck size: fold  W[:, deck] * (-64)  into the bias, then zero W[:, deck];
  * own-hand-seen: zero W[:, 3] (its input was 0, so the output is unchanged).

After migration the net ignores both inputs; a net trained from now on learns
to use them. Migrated files are tagged, so running the tool twice is a no-op.

Usage (from the repo root):
    python games/loveletter/migrate_model_features.py games/loveletter/model/*.onnx
    python games/loveletter/migrate_model_features.py runs/<run>/checkpoint.pt
    python games/loveletter/migrate_model_features.py model.onnx --out migrated.onnx

Supports the MLP nets this game ships (`training.model.PVNet`): an ONNX file
whose first node is a Gemm on the `features` input, or a training checkpoint
whose state dicts start with `backbone.0.weight` / `backbone.0.bias`. Anything
else is rejected rather than guessed at.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Layout of LoveLetterFeatureEncoder (loveletter_net_adapter.cpp): N per-player
# public blocks of 13, then the global block whose first entry is deck size.
PER_PLAYER_PUBLIC = 13
PER_PLAYER_PRIVATE = 16
GLOBAL_FEATURES = 12
OWN_HAND_SEEN_INDEX = 3            # perspective's block comes first
OLD_DECK_SIZE_CONSTANT = -64.0     # 8 placeholders of -128, divided by 16

MIGRATION_KEY = "loveletter_encoder_migration"
MIGRATION_TAG = "deck_size+own_hand_seen:v1"


def num_players_for(feature_dim: int) -> int:
    per_player = PER_PLAYER_PUBLIC + PER_PLAYER_PRIVATE
    n, rest = divmod(feature_dim - GLOBAL_FEATURES, per_player)
    if rest != 0 or not 2 <= n <= 4:
        raise ValueError(
            f"feature dim {feature_dim} is not a Love Letter encoder layout "
            f"(expected {per_player}*N + {GLOBAL_FEATURES} for N in 2..4)")
    return n


def deck_size_index(feature_dim: int) -> int:
    return PER_PLAYER_PUBLIC * num_players_for(feature_dim)


def fold_first_layer(weight: np.ndarray, bias: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """weight: [out, in] of the first layer; bias: [out]. Returns new copies."""
    if weight.ndim != 2 or bias.shape != (weight.shape[0],):
        raise ValueError(f"unexpected first-layer shapes {weight.shape} / {bias.shape}")
    deck = deck_size_index(weight.shape[1])
    w = weight.astype(np.float64, copy=True)
    b = bias.astype(np.float64, copy=True)
    b += w[:, deck] * OLD_DECK_SIZE_CONSTANT
    w[:, deck] = 0.0
    w[:, OWN_HAND_SEEN_INDEX] = 0.0
    return w.astype(weight.dtype), b.astype(bias.dtype)


def migrate_onnx(path: Path, out: Path) -> bool:
    """Returns False if `path` was already migrated (nothing written)."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    for prop in model.metadata_props:
        if prop.key == MIGRATION_KEY:
            if prop.value != MIGRATION_TAG:
                raise ValueError(f"{path}: unknown migration tag {prop.value!r}")
            return False

    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}
    input_names = [i.name for i in graph.input if i.name not in initializers]
    if len(input_names) != 1:
        raise ValueError(f"{path}: expected one graph input, found {input_names}")
    consumers = [n for n in graph.node if input_names[0] in n.input]
    if len(consumers) != 1 or consumers[0].op_type != "Gemm":
        raise ValueError(
            f"{path}: the features input must feed exactly one Gemm node "
            f"(found {[n.op_type for n in consumers]})")
    gemm = consumers[0]
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in gemm.attribute}
    if (gemm.input[0] != input_names[0] or len(gemm.input) != 3
            or attrs.get("alpha", 1.0) != 1.0 or attrs.get("beta", 1.0) != 1.0
            or attrs.get("transA", 0) != 0 or attrs.get("transB", 0) != 1):
        raise ValueError(f"{path}: unsupported first Gemm (inputs {list(gemm.input)}, attrs {attrs})")
    w_init = initializers[gemm.input[1]]
    b_init = initializers[gemm.input[2]]
    weight, bias = fold_first_layer(numpy_helper.to_array(w_init), numpy_helper.to_array(b_init))
    w_init.CopyFrom(numpy_helper.from_array(weight, w_init.name))
    b_init.CopyFrom(numpy_helper.from_array(bias, b_init.name))

    prop = model.metadata_props.add()
    prop.key = MIGRATION_KEY
    prop.value = MIGRATION_TAG
    onnx.checker.check_model(model)
    onnx.save(model, str(out))
    return True


def migrate_checkpoint(path: Path, out: Path) -> bool:
    """Returns False if `path` was already migrated (nothing written)."""
    import torch

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"{path}: not a training checkpoint dict")
    if MIGRATION_KEY in ckpt:
        if ckpt[MIGRATION_KEY] != MIGRATION_TAG:
            raise ValueError(f"{path}: unknown migration tag {ckpt[MIGRATION_KEY]!r}")
        return False
    touched = 0
    for key in ("model_state_dict", "ema_state_dict"):
        if key not in ckpt:
            continue
        sd = ckpt[key]
        if "backbone.0.weight" not in sd or "backbone.0.bias" not in sd:
            raise ValueError(
                f"{path}: {key} has no backbone.0.weight/bias — not an MLP "
                f"policy/value net this tool knows how to rewrite")
        weight, bias = fold_first_layer(
            sd["backbone.0.weight"].numpy(), sd["backbone.0.bias"].numpy())
        sd["backbone.0.weight"] = torch.from_numpy(weight)
        sd["backbone.0.bias"] = torch.from_numpy(bias)
        touched += 1
    if touched == 0:
        raise ValueError(f"{path}: no model_state_dict / ema_state_dict found")
    # The optimizer's running moments for the rewritten column are left as
    # they are: they only shape the first few updates of that column.
    ckpt[MIGRATION_KEY] = MIGRATION_TAG
    torch.save(ckpt, str(out))
    return True


def migrate(path: Path, out: Path | None = None) -> bool:
    out = out or path
    if path.suffix == ".onnx":
        return migrate_onnx(path, out)
    if path.suffix in (".pt", ".pth"):
        return migrate_checkpoint(path, out)
    raise ValueError(f"{path}: expected an .onnx model or a .pt checkpoint")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None,
                        help="write here instead of in place (single input only)")
    args = parser.parse_args(argv)
    if args.out is not None and len(args.paths) != 1:
        parser.error("--out needs exactly one input path")
    for path in args.paths:
        changed = migrate(path, args.out)
        print(f"{path}: {'migrated' if changed else 'already migrated, left untouched'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
