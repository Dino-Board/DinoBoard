"""Make a harmonies_tok network trained before the potential-score features ignore them.

    python games/harmonies/tools/zero_new_features.py model_latest.onnx \
        games/harmonies/model/harmonies_tok_v2_teacher.onnx

See scripts/zero_new_features.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from zero_new_features import zero_features  # noqa: E402

ADDED = {"player": ["ready_points", "ready_cubes", "potential_total", "potential_margin"],
         "card": ["placeable"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    args = ap.parse_args()
    zero_features("harmonies_tok", ADDED, args.src, args.dst)


if __name__ == "__main__":
    main()
