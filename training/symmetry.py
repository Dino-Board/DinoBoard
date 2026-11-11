"""Board-symmetry data augmentation (train.json `training.symmetry_augmentation`).

The encoder is the single source of truth: IFeatureEncoder::symmetries()
(engine/core/feature_encoder.h, exported by `dinoboard_engine.encoder_symmetries`)
says, per symmetry, where every feature (flat encoders) or every anchor id,
position and feature column (variable token encoders) of a sample goes, and
where every action goes. This module turns that into gather tables once and
gives each sampled training example its own random symmetry at batch time, so
the replay buffer still holds every sample exactly once.
"""
from __future__ import annotations

import numpy as np

HEADER = 7  # variable token header: valid, entity, anchor, type, has position, x, y


def _gather_from_map(mapping, size: int, what: str) -> np.ndarray:
    """The inverse of a permutation given as old index -> new index, so that
    `new = old[gather]`."""
    m = np.asarray(mapping, dtype=np.int64)
    if m.shape != (size,) or not np.array_equal(np.sort(m), np.arange(size)):
        raise ValueError(f"symmetry {what} is not a permutation of {size} entries")
    gather = np.empty(size, dtype=np.int64)
    gather[m] = np.arange(size)
    return gather


class SymmetryAugmenter:
    """Applies the encoder's symmetry `sym[i]` to row i of a batch."""

    def __init__(self, symmetries: list[dict], *, feature_dim: int, action_space: int,
                 token_layout: dict | None):
        if len(symmetries) < 2:
            raise ValueError("symmetry augmentation needs an encoder with at least one non-identity symmetry")
        self.count = len(symmetries)
        self.action_gather = np.stack(
            [_gather_from_map(s["action_map"], action_space, "action_map") for s in symmetries])
        if not np.array_equal(self.action_gather[0], np.arange(action_space)):
            raise ValueError("symmetry 0 must be the identity")
        self.variable = bool(token_layout and token_layout["variable_tokens"])
        if not self.variable:
            self.feature_gather = np.stack(
                [_gather_from_map(s["feature_map"], feature_dim, "feature_map") for s in symmetries])
            return
        self.token_dim = int(token_layout["token_dim"])
        num_types = int(token_layout["num_token_types"])
        num_anchors = int(token_layout["num_anchor_ids"])
        self.anchor_map = np.stack([np.asarray(s["anchor_map"], dtype=np.int64) for s in symmetries])
        if self.anchor_map.shape[1] != num_anchors + 1 or (self.anchor_map[:, 0] != 0).any():
            raise ValueError(f"symmetry anchor_map must have num_anchor_ids + 1 = {num_anchors + 1} "
                             f"entries with anchor 0 fixed")
        for g in range(self.count):
            _gather_from_map(self.anchor_map[g], num_anchors + 1, "anchor_map")
        self.affine = np.asarray([s["position_affine"] for s in symmetries], dtype=np.float32)
        self.column_gather = np.tile(np.arange(self.token_dim), (self.count, num_types, 1))
        self.columns_fixed = np.ones(self.count, dtype=bool)
        for g, s in enumerate(symmetries):
            if len(s["column_map"]) not in (0, num_types):
                raise ValueError(f"symmetry column_map must list 0 or {num_types} token types")
            for t, cols in enumerate(s["column_map"]):
                if not cols:
                    continue
                gather = _gather_from_map(cols, self.token_dim, "column_map")
                if not np.array_equal(gather[:HEADER], np.arange(HEADER)):
                    raise ValueError("symmetry column_map must keep the token header in place")
                self.column_gather[g, t] = gather
                self.columns_fixed[g] &= bool(np.array_equal(gather, np.arange(self.token_dim)))

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.integers(self.count, size=n)

    def features(self, x: np.ndarray, sym: np.ndarray) -> np.ndarray:
        """`x` is [B, W]: flat features, or variable tokens padded with all-zero tokens."""
        out = x.copy()
        for g in range(1, self.count):
            rows = np.flatnonzero(sym == g)
            if rows.size:
                out[rows] = self._tokens(x[rows], g) if self.variable else x[rows][:, self.feature_gather[g]]
        return out

    def actions(self, p: np.ndarray, sym: np.ndarray) -> np.ndarray:
        """`p` is [B, action_space]: policy targets or legal masks."""
        out = p.copy()
        for g in range(1, self.count):
            rows = np.flatnonzero(sym == g)
            if rows.size:
                out[rows] = p[rows][:, self.action_gather[g]]
        return out

    def _tokens(self, x: np.ndarray, g: int) -> np.ndarray:
        tok = x.reshape(len(x), -1, self.token_dim)
        if not self.columns_fixed[g]:
            tok = np.take_along_axis(tok, self.column_gather[g][tok[:, :, 3].astype(np.int64)], axis=2)
        else:
            tok = tok.copy()
        tok[:, :, 2] = self.anchor_map[g][tok[:, :, 2].astype(np.int64)]
        a, b, c, d, e, f = self.affine[g]
        has = tok[:, :, 4] == 1
        qx, qy = tok[:, :, 5].astype(np.float32), tok[:, :, 6].astype(np.float32)
        tok[:, :, 5] = np.where(has, a * qx + b * qy + e, qx)
        tok[:, :, 6] = np.where(has, c * qx + d * qy + f, qy)
        return tok.reshape(len(x), -1)


def parse_symmetry_augmentation(train_cfg: dict, game_id: str, *, feature_dim: int, action_space: int,
                                token_layout: dict | None) -> SymmetryAugmenter | None:
    """training.symmetry_augmentation: true = a random board symmetry per
    sampled example (default false). Rejected for encoders without symmetries."""
    flag = train_cfg.get("symmetry_augmentation", False)
    if not isinstance(flag, bool):
        raise ValueError(f"training.symmetry_augmentation must be a bool, got {flag!r}")
    if not flag:
        return None
    import dinoboard_engine
    symmetries = dinoboard_engine.encoder_symmetries(game_id)
    if len(symmetries) < 2:
        raise ValueError(
            f"training.symmetry_augmentation is true but the encoder of {game_id!r} offers no board "
            f"symmetries (IFeatureEncoder::symmetries)")
    return SymmetryAugmenter(symmetries, feature_dim=feature_dim, action_space=action_space,
                             token_layout=token_layout)
