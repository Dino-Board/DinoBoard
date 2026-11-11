"""In-memory game session management."""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "platform"))

import dinoboard_engine as engine
from model_paths import expected_model_path, find_model_path  # noqa: E402
from session_factory import SessionConfig, SessionFactory  # noqa: E402

from .visibility import viewer_legal_actions, viewer_state  # noqa: E402

from training.cli import find_heuristic_config  # noqa: E402
from training.mcts_profile import resolve_profile  # noqa: E402

logger = logging.getLogger("dinoboard.sessions")

# Session pool — guarded by _sessions_lock for safe concurrent reaper +
# request thread access. Reads via get_session() bump `last_active` so
# active sessions don't get evicted.
sessions: dict = {}
_sessions_lock = threading.Lock()

# Idle TTL: sessions untouched longer than this are evicted by the
# background reaper. Defaults to 1 hour. Set DINOBOARD_SESSION_TTL_SECONDS=0
# to disable eviction entirely (useful for local dev / tests).
SESSION_TTL_SECONDS = int(os.environ.get("DINOBOARD_SESSION_TTL_SECONDS", "3600"))
# How often the reaper wakes up. Defaults to 5 min — sweep cost is O(N)
# session dict scan, cheap enough that 5 min is fine even at 1000s of
# sessions. The actual eviction time is between TTL and TTL + interval.
SESSION_REAP_INTERVAL_SECONDS = int(
    os.environ.get("DINOBOARD_SESSION_REAP_INTERVAL_SECONDS", "300"))

# Map difficulty string to web profile name. "heuristic" stays out of MCTS
# entirely (no model, no profile).
_DIFFICULTY_TO_PROFILE = {
    "heuristic": None,
    "casual": "web_casual",
    "expert": "web_expert",
}


def load_game_configs() -> dict:
    configs = {}
    games_dir = PROJECT_ROOT / "games"
    for game_dir in sorted(games_dir.iterdir()):
        config_path = game_dir / "config" / "game.json"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
            configs[cfg["game_id"]] = cfg
    return configs


GAME_CONFIGS = load_game_configs()
VARIANT_TO_BASE = {
    vid: cfg["game_id"]
    for cfg in GAME_CONFIGS.values()
    for vid in cfg.get("variants", {})
}


def get_session(session_id: str) -> dict:
    with _sessions_lock:
        sess = sessions.get(session_id)
        if not sess:
            raise KeyError(f"session {session_id} not found")
        # Bump last_active inside the lock so a concurrent reaper sweep
        # can't evict between read and update.
        sess["last_active"] = time.time()
    return sess


def _make_pipeline_state() -> dict:
    return {
        "phase": "idle",
        "ai_action": None,
        "ai_action_info": None,
        "ai_stats": None,
        # Public events of the AI's last move as the human may observe them
        # (e.g. 6 nimmt!'s card_played once cards are face up).
        "ai_events": None,
        "analysis": None,
    }


def _make_precompute_state() -> dict:
    return {
        "ply_index": -1,
        "history_hash": "",
        "result": None,
    }


def create_session(
    game_id: str,
    seed: int,
    human_player: int,
    num_players: int,
    difficulty: str,
) -> tuple[str, dict]:
    if difficulty not in _DIFFICULTY_TO_PROFILE:
        raise ValueError(
            f"unknown difficulty {difficulty!r}, expected one of "
            f"{list(_DIFFICULTY_TO_PROFILE)}")
    profile_name = _DIFFICULTY_TO_PROFILE[difficulty]
    use_model = profile_name is not None

    if game_id not in GAME_CONFIGS:
        raise ValueError(f"unknown game_id {game_id!r}, not found in GAME_CONFIGS")

    # game.json `web_variants`: the registered variant web games use for a
    # player count (e.g. {"2": "splendor_tok"}), instead of <game>_<N>p / <game>.
    web_variant = GAME_CONFIGS[game_id].get("web_variants", {}).get(str(num_players))
    candidate = f"{game_id}_{num_players}p"
    if web_variant is not None:
        if web_variant not in engine.available_games():
            raise ValueError(
                f"{game_id!r} game.json web_variants names {web_variant!r} for "
                f"{num_players}p, which is not registered")
        actual_id = web_variant
    elif candidate in engine.available_games():
        actual_id = candidate
    elif game_id in engine.available_games():
        actual_id = game_id
    else:
        raise ValueError(
            f"requested {num_players}p variant for {game_id!r}, but neither "
            f"{candidate!r} nor {game_id!r} is registered"
        )

    model_path = ""
    if use_model:
        model_path = find_model_path(actual_id)
        if not model_path:
            raise FileNotFoundError(
                f"no trained model found for {actual_id}. "
                f"Expected at {expected_model_path(actual_id).relative_to(PROJECT_ROOT)}. "
                f"Copy from runs/<run_name>/models/model_best.onnx after training, "
                f"renaming to <variant>.onnx."
            )

    if use_model:
        game_victory_mode = GAME_CONFIGS[game_id].get("victory_mode", "winner_takes")
        live_profile = resolve_profile(game_id, profile_name)
        analysis_profile = resolve_profile(game_id, "analysis")
        effective_sims = live_profile.simulations
        effective_world_cap = live_profile.world_cap
        effective_c_puct = live_profile.c_puct
        effective_fpu = live_profile.fpu_reduction
        effective_temp = live_profile.temperature
        effective_opp_sel = live_profile.opponent_selection
        ai_use_filter = live_profile.ai_use_action_filter
        tail_solve_enabled = live_profile.tail_solve_enabled
        tail_solve_depth = live_profile.tail_solve_depth_limit
        tail_solve_budget = live_profile.tail_solve_node_budget
        tail_solve_auxiliary = live_profile.tail_solve_auxiliary_weight
        t_sched_enabled = live_profile.temperature_schedule_enabled
        t_initial = live_profile.temperature_initial
        t_final = live_profile.temperature_final
        t_decay = live_profile.temperature_decay_plies
        top_k = live_profile.top_k
        # Only `simulations` and `world_cap` are consumed from the analysis
        # profile. The analysis pipeline (pipeline.py) hard-codes
        # temperature=0.0, cover_root_edges=True, opponent_selection="puct" —
        # see the docstring on training/mcts_profile.py.
        analysis_sims = analysis_profile.simulations
        analysis_world_cap = analysis_profile.world_cap
    else:
        # Heuristic difficulty: no profile, no model, no MCTS knobs in play.
        game_victory_mode = GAME_CONFIGS[game_id].get("victory_mode", "winner_takes")
        effective_sims = 1
        effective_world_cap = 0
        effective_c_puct = 1.4
        effective_fpu = None
        effective_temp = 0.0
        effective_opp_sel = "puct"
        ai_use_filter = False
        tail_solve_enabled = False
        tail_solve_depth = 10
        tail_solve_budget = 200000
        tail_solve_auxiliary = 0.0
        analysis_sims = 0
        analysis_world_cap = 0
        t_sched_enabled = False
        t_initial = 0.0
        t_final = 0.0
        t_decay = 0
        top_k = 0

    # Live web session uses no filter (filter is for training only); analysis
    # / precompute paths construct their own isolated sessions in pipeline.py.
    gs = SessionFactory.create(SessionConfig(
        game_id=actual_id,
        seed=seed,
        model_path=model_path,
        use_action_filter=False,
        tail_solve_enabled=tail_solve_enabled,
        tail_solve_depth_limit=tail_solve_depth,
        tail_solve_node_budget=tail_solve_budget,
        tail_solve_auxiliary_weight=tail_solve_auxiliary,
    ))

    heuristic_cfg, _ = find_heuristic_config(actual_id)
    web_selector_cfg = heuristic_cfg["selectors"]["web"]
    web_heuristic_selector = web_selector_cfg["name"]
    web_heuristic_temperature = float(web_selector_cfg.get("temperature", 0.0))

    actual_num_players = gs.num_players
    if actual_num_players != num_players:
        raise ValueError(
            f"requested {num_players} players for {game_id!r}, but {actual_id!r} "
            f"created a {actual_num_players}p session"
        )
    ai_players = [p for p in range(actual_num_players) if p != human_player]

    session_id = uuid.uuid4().hex[:12]
    sess = {
        "game_session": gs,
        "human_player": human_player,
        "ai_players": ai_players,
        "ai_player": ai_players[0] if ai_players else -1,
        "game_id": game_id,
        "actual_id": actual_id,
        "victory_mode": game_victory_mode,
        "seed": seed,
        "difficulty": difficulty,
        "simulations": effective_sims,
        "world_cap": effective_world_cap,
        "c_puct": effective_c_puct,
        "fpu_reduction": effective_fpu,
        "temperature": effective_temp,
        "opponent_selection": effective_opp_sel,
        "analysis_simulations": analysis_sims,
        "analysis_world_cap": analysis_world_cap,
        "use_model": use_model,
        "model_path": model_path,
        "ai_use_filter": ai_use_filter,
        "tail_solve_enabled": tail_solve_enabled,
        "tail_solve_depth_limit": tail_solve_depth,
        "tail_solve_node_budget": tail_solve_budget,
        "heuristic_selector": web_heuristic_selector,
        "heuristic_temperature": web_heuristic_temperature,
        "temperature_schedule_enabled": t_sched_enabled,
        "temperature_initial": t_initial,
        "temperature_final": t_final,
        "temperature_decay_plies": t_decay,
        "top_k": top_k,
        "action_history": [],
        "replay_frames": [],
        "pipeline": _make_pipeline_state(),
        "precompute": _make_precompute_state(),
        "pipeline_lock": threading.Lock(),
        # Wallclock used by the idle-eviction reaper. get_session() bumps
        # this on every read. Sessions stale longer than SESSION_TTL_SECONDS
        # are dropped on the next sweep.
        "created_at": time.time(),
        "last_active": time.time(),
    }
    with _sessions_lock:
        sessions[session_id] = sess
    return session_id, sess


def reap_idle_sessions(now: Optional[float] = None) -> int:
    """Drop sessions whose last_active is older than SESSION_TTL_SECONDS.

    Returns the number evicted. Safe to call at any time.
    Disabled (always returns 0) when SESSION_TTL_SECONDS <= 0.
    """
    if SESSION_TTL_SECONDS <= 0:
        return 0
    if now is None:
        now = time.time()
    cutoff = now - SESSION_TTL_SECONDS
    # Two-phase: hold the lock briefly to gather victims + remove them
    # from the dict, then run side-effects (cancel pipeline, etc.) outside
    # the lock so a single slow shutdown doesn't block fresh requests.
    victims: list[tuple[str, dict]] = []
    with _sessions_lock:
        for sid, sess in list(sessions.items()):
            if sess.get("last_active", 0) < cutoff:
                victims.append((sid, sess))
                del sessions[sid]
    for sid, sess in victims:
        # cancel_pipeline lives in pipeline.py and would create a circular
        # import; do the minimum here — the worker checks pipeline phase
        # and bails. Lock acquire is best-effort: a worker holding it for
        # 12+ seconds is already stuck and the OS will reclaim its memory
        # when the process exits.
        try:
            sess["pipeline"]["phase"] = "done"
        except Exception:
            pass
    if victims:
        logger.info("reaped %d idle sessions (TTL=%ds)",
                    len(victims), SESSION_TTL_SECONDS)
    return len(victims)


def _session_reaper_loop(stop_event: threading.Event) -> None:
    """Background daemon: periodically calls reap_idle_sessions."""
    while not stop_event.is_set():
        # Wait first so we don't sweep immediately at startup; nothing's
        # idle in the first 5 minutes anyway.
        if stop_event.wait(SESSION_REAP_INTERVAL_SECONDS):
            break
        try:
            reap_idle_sessions()
        except Exception:
            logger.exception("session reaper sweep failed")


_REAPER_STOP = threading.Event()
_REAPER_THREAD: Optional[threading.Thread] = None


def start_session_reaper() -> None:
    """Start the background reaper thread. Idempotent."""
    global _REAPER_THREAD
    if SESSION_TTL_SECONDS <= 0:
        logger.info("session reaper disabled (SESSION_TTL_SECONDS<=0)")
        return
    if _REAPER_THREAD is not None and _REAPER_THREAD.is_alive():
        return
    _REAPER_STOP.clear()
    _REAPER_THREAD = threading.Thread(
        target=_session_reaper_loop, args=(_REAPER_STOP,),
        name="session-reaper", daemon=True)
    _REAPER_THREAD.start()
    logger.info(
        "session reaper started: TTL=%ds, sweep every %ds",
        SESSION_TTL_SECONDS, SESSION_REAP_INTERVAL_SECONDS)


def stop_session_reaper() -> None:
    _REAPER_STOP.set()


def rebuild_game_session(sess: dict) -> None:
    """Recreate GameSession and replay action_history. The live session is for
    humans, so use_filter is always False."""
    sess["game_session"] = SessionFactory.from_session_dict(
        sess,
        use_action_filter=False,
        history=sess["action_history"],
    )


def session_response(session_id: str, sess: dict, extra: Optional[dict] = None) -> dict:
    gs = sess["game_session"]
    current = gs.current_player
    human = sess["human_player"]
    # Everything below is what the human seat may see (visibility.py): the
    # state with hidden leaves withheld, and the options of the player to
    # move only when they do not depend on cards hidden from the human.
    legal, legal_hidden = viewer_legal_actions(gs, human)
    last_actor = None
    frames = sess.get("replay_frames")
    if frames and len(frames) > 0:
        last_actor = frames[-1].get("actor")

    result = {
        "session_id": session_id,
        "state": viewer_state(gs, human),
        "legal_actions": legal,
        # True: the player to move has options, but they are private to it
        # (an empty `legal_actions` then does NOT mean "no legal action").
        "legal_actions_hidden": legal_hidden,
        "current_player": current,
        "is_terminal": gs.is_terminal,
        "is_turn_start": gs.is_turn_start,
        "last_actor": last_actor,
        "winner": gs.winner,
        "winners": gs.winners if gs.is_terminal else [],
        "losers": gs.losers if gs.is_terminal else [],
        "victory_mode": sess.get("victory_mode", "winner_takes"),
        "num_players": gs.num_players,
        "human_player": sess["human_player"],
        "ai_players": sess["ai_players"],
        "ai_player": sess["ai_player"],
        "difficulty": sess["difficulty"],
    }
    if extra:
        result.update(extra)
    return result


def pipeline_mark_done_empty(sess: dict) -> None:
    p = sess["pipeline"]
    p["phase"] = "done"
    p["ai_action"] = None
    p["ai_action_info"] = None
    p["ai_stats"] = None
    p["ai_events"] = None
    p["analysis"] = None


def pipeline_reset(sess: dict) -> None:
    p = sess["pipeline"]
    p["phase"] = "idle"
    p["ai_action"] = None
    p["ai_action_info"] = None
    p["ai_stats"] = None
    p["ai_events"] = None
    p["analysis"] = None


def precompute_clear(sess: dict) -> None:
    pc = sess["precompute"]
    pc["ply_index"] = -1
    pc["history_hash"] = ""
    pc["result"] = None


def history_hash(action_history: list) -> str:
    return str(hash(tuple(action_history)))
