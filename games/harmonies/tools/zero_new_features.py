"""Make a harmonies_tok network trained before the potential-score features ignore them.

    python games/harmonies/tools/zero_new_features.py model_best.onnx \
        games/harmonies/model/harmonies_tok_v2_teacher.onnx

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

ROOT = Path(__file__).resolve().parents[3]
ADDED = {"player": ["ready_points", "ready_cubes", "potential_total", "potential_margin"],
         "card": ["placeable"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    import dinoboard_engine as engine

    layout = json.loads(engine.GameSession("harmonies_tok", seed=0).get_state_dict()["encoder_layout"])
    type_ids = {t["name"]: i for i, t in enumerate(layout["types"])}
    rows = []  # (type id, feature row)
    for tname, names in ADDED.items():
        feats = {n: (o, w) for n, o, w in layout["types"][type_ids[tname]]["features"]}
        for n in names:
            o, w = feats[n]
            rows += [(type_ids[tname], o + i) for i in range(w)]

    model = onnx.load(args.src)
    weights = [t for t in model.graph.initializer if t.name == "type_weight"]
    if len(weights) != 1:
        raise SystemExit("no single 'type_weight' initializer in " + args.src)
    w = numpy_helper.to_array(weights[0]).copy()
    for t, r in rows:
        w[t, r, :] = 0
    weights[0].CopyFrom(numpy_helper.from_array(w, "type_weight"))
    onnx.save(model, args.dst)

    # Check: the zeroed features no longer matter.
    meta = engine.game_metadata("harmonies_tok")
    dim = meta["token_layout"]["token_dim"]
    header = layout["header"]
    gs = engine.GameSession("harmonies_tok", seed=5)
    for _ in range(12):
        gs.apply_action(int(gs.get_heuristic_action("greedy")["action"]))
    x = np.asarray(gs._encode(gs.current_player)["features"], dtype=np.float32).reshape(-1, dim)
    n = -(-(len(x) + 1) // 8) * 8 - 1
    x = np.vstack([x, np.zeros((n - len(x), dim), np.float32)])
    sess = ort.InferenceSession(args.dst, providers=["CPUExecutionProvider"])
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


if __name__ == "__main__":
    main()
