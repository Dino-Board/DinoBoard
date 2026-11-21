"""Which past checkpoints the latest model plays in the pool share of selfplay.

`uniform`: every checkpoint in the window is equally likely; the engine draws
`max_distinct` of them each step.

`quality` (OpenAI Five, Berner et al. 2019, appendix N): each checkpoint has a
score q and a step's opponents are drawn from softmax(q), mixed with
`uniform_mix` of the uniform distribution. A new checkpoint enters at the
highest score in the pool and loses score whenever the latest model beats it in
a pool game, so checkpoints the latest model has outgrown fade from the draw
while ones it still loses to keep their share. The pool games' own outcomes
drive this: no evaluation games are added.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SAMPLING_MODES = ("quality", "uniform")
_STEP_RE = re.compile(r"model_step_(\d+)\.onnx$")


@dataclass(frozen=True)
class PoolConfig:
    sampling: str = "quality"
    window: float = 1.0
    learning_rate: float = 0.01
    uniform_mix: float = 0.1
    max_distinct: int = 3


def parse_pool_config(train_cfg: dict, max_distinct: int) -> PoolConfig:
    """`training.opponent_pool_*` keys; only meaningful with the pool enabled."""
    keys = ("opponent_pool_sampling", "opponent_pool_window",
            "opponent_pool_quality_lr", "opponent_pool_uniform_mix")
    if not train_cfg.get("opponent_pool_enabled", False):
        given = [k for k in keys if k in train_cfg]
        if given:
            raise ValueError(f"training.{given[0]} needs opponent_pool_enabled: true")
    cfg = PoolConfig(
        sampling=str(train_cfg.get("opponent_pool_sampling", "quality")),
        window=float(train_cfg.get("opponent_pool_window", 1.0)),
        learning_rate=float(train_cfg.get("opponent_pool_quality_lr", 0.01)),
        uniform_mix=float(train_cfg.get("opponent_pool_uniform_mix", 0.1)),
        max_distinct=int(max_distinct),
    )
    if cfg.sampling not in SAMPLING_MODES:
        raise ValueError(f"training.opponent_pool_sampling must be one of {SAMPLING_MODES}, "
                         f"got {cfg.sampling!r}")
    if not 0.0 < cfg.window <= 1.0:
        raise ValueError(f"training.opponent_pool_window must be in (0, 1], got {cfg.window}")
    if cfg.learning_rate <= 0.0:
        raise ValueError(f"training.opponent_pool_quality_lr must be > 0, got {cfg.learning_rate}")
    if not 0.0 <= cfg.uniform_mix <= 1.0:
        raise ValueError(f"training.opponent_pool_uniform_mix must be in [0, 1], got {cfg.uniform_mix}")
    if cfg.max_distinct < 1:
        raise ValueError(f"opponent_pool_max_distinct must be >= 1, got {cfg.max_distinct}")
    return cfg


def checkpoint_step(path: str) -> int:
    m = _STEP_RE.search(Path(path).name)
    if m is None:
        raise ValueError(f"not a model_step_*.onnx checkpoint: {path}")
    return int(m.group(1))


@dataclass
class Selection:
    """A step's opponents. `inclusion[name]` is the chance the draw picked that
    checkpoint (quality mode), which the score update divides by."""
    paths: list[tuple[str, str]]
    pool_size: int
    probs: dict[str, float] = field(default_factory=dict)
    inclusion: dict[str, float] = field(default_factory=dict)


class OpponentPool:
    def __init__(self, cfg: PoolConfig, state: dict[str, Any] | None = None):
        self.cfg = cfg
        self.scores: dict[str, float] = dict((state or {}).get("scores", {}))

    def state(self) -> dict[str, Any]:
        return {"scores": self.scores}

    def candidates(self, pool_paths: list[tuple[str, str]], step: int) -> list[tuple[str, str]]:
        """Checkpoints inside the window, enrolling new ones at the top score."""
        present = {Path(pv).name for pv, _ in pool_paths}
        self.scores = {n: q for n, q in self.scores.items() if n in present}
        top = max(self.scores.values(), default=0.0)
        for pv, _ in pool_paths:
            self.scores.setdefault(Path(pv).name, top)
        oldest = (1.0 - self.cfg.window) * step
        kept = [pb for pb in pool_paths if checkpoint_step(pb[0]) >= oldest]
        return kept or pool_paths[-1:]

    def probabilities(self, cands: list[tuple[str, str]]) -> np.ndarray:
        q = np.array([self.scores[Path(pv).name] for pv, _ in cands], dtype=np.float64)
        p = np.exp(q - q.max())
        p /= p.sum()
        return (1.0 - self.cfg.uniform_mix) * p + self.cfg.uniform_mix / len(cands)

    def select(self, pool_paths: list[tuple[str, str]], step: int,
               rng: np.random.Generator) -> Selection:
        cands = self.candidates(pool_paths, step)
        k = self.cfg.max_distinct
        if self.cfg.sampling == "uniform" or not cands:
            return Selection(paths=cands, pool_size=len(cands))
        p = self.probabilities(cands)
        names = [Path(pv).name for pv, _ in cands]
        if len(cands) <= k:
            picked = list(range(len(cands)))
            inclusion = np.ones(len(cands))
        else:
            # k draws with replacement: inclusion = 1 - (1 - p)^k exactly.
            picked = sorted(set(rng.choice(len(cands), size=k, p=p).tolist()))
            inclusion = 1.0 - (1.0 - p) ** k
        return Selection(
            paths=[cands[i] for i in picked],
            pool_size=len(cands),
            probs={names[i]: float(p[i]) for i in range(len(cands))},
            inclusion={names[i]: float(inclusion[i]) for i in picked},
        )

    def update(self, selection: Selection, episodes: list[dict[str, Any]],
               weight: float = 1.0) -> dict[str, tuple[int, float]]:
        """Score the latest model's pool games and, in quality mode, lower each
        opponent by weight * lr * G * score_rate / (N * inclusion): G pool
        games, N checkpoints in the window. Its expectation over the draw
        equals the per-game rule q -= lr / (N p) on every win.

        A game scores (z + 1) / 2 from the latest seat's terminal value z: with
        two players 1 for a win, 0 for a loss, 1/2 for a draw; with more, it
        follows the game's own outcome (loser-pays games reward not losing).

        Returns {checkpoint name: (games, latest's score)}."""
        games: dict[str, int] = defaultdict(int)
        score: dict[str, float] = defaultdict(float)
        for ep in episodes:
            opp = ep.get("pool_opponent")
            if not opp:
                continue
            name = Path(opp).name
            games[name] += 1
            z = ep["terminal_z_values"]
            score[name] += 0.5 if not z else min(1.0, max(0.0, (z[ep["latest_seat"]] + 1.0) / 2.0))
        total = sum(games.values())
        if self.cfg.sampling == "quality" and weight > 0.0 and selection.pool_size > 0:
            for name, n in games.items():
                if name not in selection.inclusion:
                    raise ValueError(f"pool game against {name}, which this step did not draw")
                self.scores[name] -= (weight * self.cfg.learning_rate * total * (score[name] / n)
                                      / (selection.pool_size * selection.inclusion[name]))
        return {name: (games[name], score[name]) for name in sorted(games)}


def format_pool_step(selection: Selection, results: dict[str, tuple[int, float]]) -> str:
    """`step_01750 p=0.21 14/20` per opponent played this step."""
    parts = []
    for name, (n, s) in results.items():
        p = selection.probs.get(name)
        prob = f" p={p:.2f}" if p is not None else ""
        parts.append(f"step_{checkpoint_step(name):05d}{prob} {s:g}/{n}")
    return ", ".join(parts)
