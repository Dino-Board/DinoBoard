"""Single source of truth for constructing engine.GameSession instances.

Both ai_service and game_service create GameSessions with similar shapes
(seed, model path, optional action filter, optional tail-solve, optional
action-history replay). Centralizing the construction eliminates ad-hoc
helpers like _apply_tail_solve_config and prevents call sites from
forgetting tail-solve / filter wiring (see BUG / OB-004).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import dinoboard_engine as engine

from model_paths import find_belief_model_path


@dataclass(frozen=True)
class SessionConfig:
    """Inputs needed to build a GameSession.

    `game_id` should be the variant id used by the engine (e.g. azul_3p),
    not the base game id. `model_path` is "" when running without a model.
    """

    game_id: str
    seed: int
    model_path: str
    use_action_filter: bool = False
    tail_solve_enabled: bool = False
    tail_solve_depth_limit: int = 0
    tail_solve_node_budget: int = 0
    tail_solve_auxiliary_weight: float = 0.0


def _from_session_dict(sess: dict, *, use_action_filter: bool) -> SessionConfig:
    """Build a SessionConfig from a game_service session dict.

    Centralizes the field-name knowledge so call sites in pipeline.py and
    sessions.py stay short. The caller picks `use_action_filter` because
    analysis/precompute paths force it False even when the live session
    has filtering enabled.
    """
    return SessionConfig(
        game_id=sess["actual_id"],
        seed=sess["seed"],
        model_path=sess["model_path"] if sess["use_model"] else "",
        use_action_filter=use_action_filter,
        tail_solve_enabled=bool(sess["tail_solve_enabled"]),
        tail_solve_depth_limit=int(sess["tail_solve_depth_limit"]),
        tail_solve_node_budget=int(sess["tail_solve_node_budget"]),
        tail_solve_auxiliary_weight=float(sess.get("tail_solve_auxiliary_weight", 0.0)),
    )


class SessionFactory:
    """Construct GameSession objects from a SessionConfig."""

    @staticmethod
    def create(
        cfg: SessionConfig,
        history: Optional[Sequence[int]] = None,
    ) -> engine.GameSession:
        # Belief-net path mirrors PV model. Only games registering a
        # belief_feature_extractor consult a learned posterior; for the
        # rest we leave the param "" so the bundle keeps its (no-op)
        # default. File missing for a game that DOES register an extractor
        # is a real configuration bug — let the C++ load throw.
        meta = engine.game_metadata(cfg.game_id)
        belief_path = (
            find_belief_model_path(cfg.game_id)
            if meta["has_belief_extractor"] else ""
        )
        gs = engine.GameSession(
            cfg.game_id, cfg.seed, cfg.model_path, cfg.use_action_filter,
            belief_path,
        )
        if cfg.tail_solve_enabled:
            gs.configure_tail_solve(
                True,
                cfg.tail_solve_depth_limit,
                cfg.tail_solve_node_budget,
                cfg.tail_solve_auxiliary_weight,
            )
        if history:
            for action_id in history:
                gs.apply_action(action_id)
        return gs

    @staticmethod
    def from_session_dict(
        sess: dict,
        *,
        use_action_filter: bool,
        history: Optional[Sequence[int]] = None,
    ) -> engine.GameSession:
        cfg = _from_session_dict(sess, use_action_filter=use_action_filter)
        return SessionFactory.create(cfg, history=history)
