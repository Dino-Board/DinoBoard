"""Search-algorithm arena: calibrate per-move budgets and run round-robins.

Agents are named seat configs in a JSON file:

    {"puct_5s":    {"model_path": "m.onnx", "algorithm": "puct", "simulations": 9000,
                    "c_puct": 1.6, "fpu_reduction": 0.2, "leaf_batch_size": 32},
     "gumbel_5s":  {"model_path": "m.onnx", "algorithm": "gumbel", "simulations": 6000,
                    "gumbel_num_considered": 16, "leaf_batch_size": 32},
     "rollout_5s": {"model_path": "m.onnx", "algorithm": "rollout", "rollout_depth": 8,
                    "rollout_rollouts_per_action": 64, "leaf_batch_size": 2048}}

Any run_concurrent_arena_batch seat key is accepted (temperature defaults to 0).

calibrate: plays games one at a time (nothing else on the GPU, as in
deployment) and reports per-move search time, simulations / rollouts and
network evaluations, so each agent's fixed budget can be set to the median
time a real per-move limit allows.

    python scripts/search_arena.py calibrate --agents agents.json --names puct_5s --games 3

match: round-robin over the named agents; every seed is played twice with the
seats swapped. Prints the score matrix and Bradley-Terry Elo (anchor = first
name) with bootstrap 95% intervals, and writes everything to --out.

    python scripts/search_arena.py match --agents agents.json --names a b c \
        --games-per-pair 200 --threads 64 --out results.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

import dinoboard_engine

GAME = "splendor_tok"


def load_agents(path: str, names: list[str]) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        agents = json.load(f)
    missing = [n for n in names if n not in agents]
    if missing:
        raise SystemExit(f"agents not in {path}: {missing}")
    out = {}
    for n in names:
        seat = dict(agents[n])
        seat.setdefault("temperature", 0.0)
        out[n] = seat
    return out


def run_tasks(tasks, args, threads):
    return dinoboard_engine.run_concurrent_arena_batch(
        args.game, tasks, threads, args.max_plies, args.batch_size, args.batch_timeout_us,
        args.max_queue_size, 1, args.inference_threads, args.providers.split(","), [], False)["results"]


def calibrate(args):
    agents = load_agents(args.agents, args.names)
    for name, seat in agents.items():
        ms, sims, evals = [], [], []
        t0 = time.time()
        for g in range(args.games):
            (r,) = run_tasks([{"seed": args.seed + g, "seat_configs": [seat, seat]}], args, 1)
            for ps in r["ply_stats"]:
                if ps["simulations_done"] == 0:  # forced move or tail-solved
                    continue
                ms.append(ps["search_ms"])
                sims.append(ps["simulations_done"])
                evals.append(ps["network_evaluations"])
        q = lambda v, p: float(np.percentile(v, p))
        print(f"{name}: {len(ms)} searched moves in {time.time() - t0:.0f}s | search_ms median "
              f"{statistics.median(ms):.0f} p10 {q(ms, 10):.0f} p90 {q(ms, 90):.0f} | sims median "
              f"{statistics.median(sims):.0f} | evals median {statistics.median(evals):.0f} "
              f"({statistics.median(evals) / max(1e-9, statistics.median(ms)) * 1000:.0f}/s)", flush=True)


def bradley_terry(names, results, iters=2000):
    """Elo from pairwise game results [(i, j, score_i)], anchor names[0] = 0."""
    n = len(names)
    wins = np.zeros((n, n))
    for i, j, s in results:
        wins[i, j] += s
        wins[j, i] += 1.0 - s
    games = wins + wins.T
    r = np.ones(n)
    for _ in range(iters):
        new = np.empty(n)
        for i in range(n):
            denom = sum(games[i, j] / (r[i] + r[j]) for j in range(n) if j != i and games[i, j] > 0)
            new[i] = (wins[i].sum() + 0.5) / max(denom, 1e-12)
        new /= new[0]
        if np.max(np.abs(new - r)) < 1e-10:
            r = new
            break
        r = new
    return 400.0 * np.log10(r)


def match(args):
    agents = load_agents(args.agents, args.names)
    names = list(agents)
    n = len(names)
    out_path = Path(args.out) if args.out else None

    # Support resuming partial pair-by-pair runs from --out
    pair_records = {}
    if out_path and out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            if prev.get("names") == names and prev.get("games_per_pair") == args.games_per_pair:
                pair_records = dict(prev.get("pair_scores") or {})
        except Exception:
            pair_records = {}

    combos = list(itertools.combinations(enumerate(names), 2))
    t0 = time.time()
    for idx, ((ia, a), (ib, b)) in enumerate(combos, 1):
        key = f"{a}__vs__{b}"
        if key in pair_records and len(pair_records[key]) == args.games_per_pair:
            s_list = pair_records[key]
            print(f"[{idx}/{len(combos)}] {a} vs {b}: (cached) score {sum(s_list)/len(s_list):.3f}", flush=True)
            continue
        tasks, meta = [], []
        for g in range(args.games_per_pair // 2):
            seed = args.seed + g
            tasks.append({"seed": seed, "seat_configs": [agents[a], agents[b]]})
            meta.append(0)
            tasks.append({"seed": seed, "seat_configs": [agents[b], agents[a]]})
            meta.append(1)
        tp = time.time()
        results = run_tasks(tasks, args, args.threads)
        dt = time.time() - tp
        s_list = []
        w = l = d = 0
        for a_seat, r in zip(meta, results):
            z = r["terminal_z_values"]
            s_a = (z[a_seat] + 1.0) / 2.0 if z else 0.5
            s_list.append(s_a)
            if s_a > 0.75:
                w += 1
            elif s_a < 0.25:
                l += 1
            else:
                d += 1
        pair_records[key] = s_list
        pair_means = [(s_list[2 * k] + s_list[2 * k + 1]) / 2.0 for k in range(len(s_list) // 2)]
        mean_s = float(np.mean(pair_means))
        ci95 = float(1.96 * np.std(pair_means, ddof=1) / np.sqrt(len(pair_means))) if len(pair_means) > 1 else 0.0
        print(f"[{idx}/{len(combos)}] {a} vs {b}: score {mean_s:.3f} ± {ci95:.3f} "
              f"(W{w} L{l} D{d}), {dt / 60:.1f} min", flush=True)
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({
                "names": names, "agents": agents, "games_per_pair": args.games_per_pair,
                "pair_scores": pair_records, "partial": True,
                "seconds": time.time() - t0,
            }, indent=2))

    elapsed = time.time() - t0
    pairs = []
    score = np.zeros((n, n))
    count = np.zeros((n, n))
    for (ia, a), (ib, b) in combos:
        key = f"{a}__vs__{b}"
        for s_a in pair_records[key]:
            pairs.append((ia, ib, s_a))
            score[ia, ib] += s_a
            score[ib, ia] += 1.0 - s_a
            count[ia, ib] += 1
            count[ib, ia] += 1
    elo = bradley_terry(names, pairs)
    rng = np.random.default_rng(args.seed)
    boot = []
    for _ in range(args.bootstrap):
        sample = [pairs[k] for k in rng.integers(0, len(pairs), len(pairs))]
        boot.append(bradley_terry(names, sample, iters=500))
    boot = np.asarray(boot)
    lo, hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)

    width = max(len(n) for n in names)
    print(f"{len(pairs)} games in {elapsed:.0f}s", flush=True)
    print(" " * (width + 2) + "  ".join(f"{nm[:8]:>8}" for nm in names), flush=True)
    for i, nm in enumerate(names):
        cells = [f"{score[i, j] / count[i, j]:8.3f}" if count[i, j] else " " * 8 for j in range(len(names))]
        print(f"{nm:<{width}}  " + "  ".join(cells), flush=True)
    for i in np.argsort(-elo):
        print(f"  {names[i]:<{width}}  Elo {elo[i]:+7.1f}  [{lo[i]:+7.1f}, {hi[i]:+7.1f}]", flush=True)
    if out_path:
        out_path.write_text(json.dumps({
            "names": names, "agents": agents, "games_per_pair": args.games_per_pair,
            "score": score.tolist(), "count": count.tolist(),
            "elo": elo.tolist(), "elo_lo": lo.tolist(), "elo_hi": hi.tolist(),
            "pair_scores": pair_records, "partial": False,
            "seconds": elapsed,
        }, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["calibrate", "match"])
    p.add_argument("--agents", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--game", default=GAME)
    p.add_argument("--games", type=int, default=3, help="calibrate: games per agent")
    p.add_argument("--games-per-pair", type=int, default=200, help="match: games per pair (even)")
    p.add_argument("--threads", type=int, default=64)
    p.add_argument("--seed", type=int, default=777000)
    p.add_argument("--max-plies", type=int, default=160)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--providers", default="cuda,cpu")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--batch-timeout-us", type=int, default=50)
    p.add_argument("--max-queue-size", type=int, default=1 << 17)
    p.add_argument("--inference-threads", type=int, default=2)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    if args.mode == "match" and args.games_per_pair % 2:
        raise SystemExit("--games-per-pair must be even (seats are swapped per seed)")
    calibrate(args) if args.mode == "calibrate" else match(args)


if __name__ == "__main__":
    sys.exit(main())
