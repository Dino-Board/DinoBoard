#!/usr/bin/env python3
"""Plot game-specific custom metrics from a training run's train.log.

Usage:
  python3 scripts/plot_training_metrics.py [run_dir]

If run_dir is omitted, uses the DEFAULT_RUN below. Automatically discovers
all custom metric keys (anything after the standard fields in each log line)
and plots them on a single chart.
"""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_RUN = Path("runs/coup_20260603_130901")
WINDOW = 5

# Non-custom fields emitted by the training loop itself. Anything left after
# this filter is treated as a game-defined custom metric from
# episode_stats_extractor.
STANDARD_FIELDS = {
    "p_loss", "v_loss", "episodes", "samples", "p0", "p1", "p2", "p3",
    "draw", "plies", "cap_hits", "sims", "lr",
    "b_loss", "b_buf", "b_step", "time", "selfplay", "train",
    "h_ratio", "f_ratio", "ts",
}


def parse_metrics(log_path: Path) -> dict[str, list[tuple[int, float]]]:
    """Parse custom metrics from train.log. Returns {metric_name: [(step, value), ...]}."""
    step_re = re.compile(r"Step (\d+)/\d+:")
    kv_re = re.compile(r"(\w+)=([\d.]+)")

    metrics: dict[str, list[tuple[int, float]]] = {}
    with log_path.open() as f:
        for line in f:
            sm = step_re.search(line)
            if not sm:
                continue
            step = int(sm.group(1))
            for m in kv_re.finditer(line):
                key, val = m.group(1), float(m.group(2))
                if key in STANDARD_FIELDS:
                    continue
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append((step, val))
    return metrics


def smooth(y: np.ndarray, w: int) -> np.ndarray:
    if len(y) < w or w <= 1:
        return y
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="valid")


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RUN
    log_path = run_dir / "train.log"
    out_path = run_dir / "custom_metrics.png"

    if not log_path.exists():
        print(f"train.log not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    metrics = parse_metrics(log_path)
    if not metrics:
        print(f"no custom metrics found in {log_path}", file=sys.stderr)
        sys.exit(1)

    w = min(WINDOW, max(1, min(len(v) for v in metrics.values()) // 10))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, data) in enumerate(sorted(metrics.items())):
        steps = np.array([d[0] for d in data])
        values = np.array([d[1] for d in data])
        sx = steps[w - 1:] if w > 1 else steps
        sv = smooth(values, w)
        ax.plot(sx, sv, label=name, color=f"C{i}")

    ax.set_xlabel("training step")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1)
    game_name = run_dir.name.split("_")[0]
    total = sum(len(v) for v in metrics.values())
    ax.set_title(f"{game_name} custom metrics (window={w}) — {list(metrics.values())[0][-1][0]} steps")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved {out_path} ({len(metrics)} metrics, {total} data points, window={w})")


if __name__ == "__main__":
    main()
