#!/usr/bin/env python3
"""Plot game-specific custom metrics from a training run's train.log.

Usage:
  python3 scripts/plot_training_metrics.py [run_dir]
  python3 scripts/plot_training_metrics.py [run_dir] --fields prior_ent,prior_top1,search_ent,policy_kl

If run_dir is omitted, uses the DEFAULT_RUN below. Automatically discovers
all custom metric keys (anything after the standard fields in each log line)
and plots them on a single chart.

--fields plots the named train.log fields instead (standard ones included,
e.g. the search-diversity read-out above), auto-scaled, to
<run_dir>/field_metrics.png.
"""
import argparse
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
    # search-diversity read-out and opening-exploration counters
    "prior_ent", "prior_top1", "search_ent", "policy_kl", "expl_eps", "expl_plies",
}


def parse_metrics(log_path: Path, fields: list[str] | None = None) -> dict[str, list[tuple[int, float]]]:
    """Parse metrics from train.log. Returns {metric_name: [(step, value), ...]}.

    fields=None: the game's custom metrics (everything not in STANDARD_FIELDS).
    fields=[...]: exactly those fields."""
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
                if (key not in fields) if fields is not None else (key in STANDARD_FIELDS):
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
    parser = argparse.ArgumentParser(description="Plot metrics from a training run's train.log.")
    parser.add_argument("run_dir", nargs="?", default=str(DEFAULT_RUN))
    parser.add_argument(
        "--fields",
        help="comma-separated train.log fields to plot instead of the custom metrics, "
             "e.g. prior_ent,prior_top1,search_ent,policy_kl")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    fields = [f for f in args.fields.split(",") if f] if args.fields else None
    log_path = run_dir / "train.log"
    out_path = run_dir / ("field_metrics.png" if fields else "custom_metrics.png")

    if not log_path.exists():
        print(f"train.log not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    metrics = parse_metrics(log_path, fields)
    if fields:
        missing = [f for f in fields if f not in metrics]
        if missing:
            print(f"field(s) {missing} not found in {log_path}", file=sys.stderr)
            sys.exit(1)
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
    if fields:
        ax.set_ylabel("value")
    else:
        ax.set_ylabel("rate")
        ax.set_ylim(0, 1)
    game_name = run_dir.name.split("_")[0]
    total = sum(len(v) for v in metrics.values())
    kind = "fields" if fields else "custom metrics"
    ax.set_title(f"{game_name} {kind} (window={w}) — {list(metrics.values())[0][-1][0]} steps")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved {out_path} ({len(metrics)} metrics, {total} data points, window={w})")


if __name__ == "__main__":
    main()
