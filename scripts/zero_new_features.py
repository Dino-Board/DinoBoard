"""Make a token-transformer network trained before some features existed ignore them.

    python scripts/zero_new_features.py --game arknova \
        --features player:endgame_final,projected_vp ... src.onnx dst.onnx

The token transformer's first layer is one weight row per (token type, feature); the network was
trained with these features always 0, so zeroing their rows keeps its outputs as they were.
Feature offsets come from the engine's encoder layout. The result is checked: random values in
those features must not change the outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[1]


def zero_features(game: str, added: dict[str, list[str]], src: str, dst: str, check_plies: int = 12) -> None:
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    import dinoboard_engine as engine

    layout = json.loads(engine.GameSession(game, seed=0).get_state_dict()["encoder_layout"])
    type_ids = {t["name"]: i for i, t in enumerate(layout["types"])}
    rows = []  # (type id, feature row)
    for tname, names in added.items():
        feats = {n: (o, w) for n, o, w in layout["types"][type_ids[tname]]["features"]}
        for n in names:
            o, w = feats[n]
            rows += [(type_ids[tname], o + i) for i in range(w)]

    model = onnx.load(src)
    weights = [t for t in model.graph.initializer if t.name == "type_weight"]
    if len(weights) != 1:
        raise SystemExit("no single 'type_weight' initializer in " + src)
    w = numpy_helper.to_array(weights[0]).copy()
    for t, r in rows:
        w[t, r, :] = 0
    weights[0].CopyFrom(numpy_helper.from_array(w, "type_weight"))
    onnx.save(model, dst)

    # Check: the zeroed features no longer matter.
    dim = engine.game_metadata(game)["token_layout"]["token_dim"]
    header = layout["header"]
    gs = engine.GameSession(game, seed=5)
    for _ in range(check_plies):
        gs.apply_action(int(gs.get_heuristic_action("greedy")["action"]))
    x = np.asarray(gs._encode(gs.current_player)["features"], dtype=np.float32).reshape(-1, dim)
    n = -(-(len(x) + 1) // 8) * 8 - 1
    x = np.vstack([x, np.zeros((n - len(x), dim), np.float32)])
    sess = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    feed_type = np.float16 if "float16" in inp.type else np.float32
    rng = np.random.default_rng(0)
    y = x.copy()
    for t, r in rows:
        sel = (y[:, 3] == t) & (y[:, 0] == 1)
        y[sel, header + r] = rng.uniform(-2, 2, sel.sum())
    a = sess.run(None, {inp.name: x.reshape(1, -1).astype(feed_type)})
    b = sess.run(None, {inp.name: y.reshape(1, -1).astype(feed_type)})
    diff = max(float(np.abs(p.astype(np.float32) - q.astype(np.float32)).max()) for p, q in zip(a, b))
    print(f"zeroed {len(rows)} rows; outputs with random values there differ by {diff:.2e}")
    if diff > 1e-6:
        raise SystemExit("the zeroed features still change the outputs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--features", nargs="+", required=True, metavar="TYPE:NAME,NAME",
                    help="per token type, the features to zero")
    ap.add_argument("src")
    ap.add_argument("dst")
    args = ap.parse_args()
    added: dict[str, list[str]] = {}
    for spec in args.features:
        tname, _, names = spec.partition(":")
        added.setdefault(tname, []).extend(n for n in names.split(",") if n)
    zero_features(args.game, added, args.src, args.dst)


if __name__ == "__main__":
    main()
