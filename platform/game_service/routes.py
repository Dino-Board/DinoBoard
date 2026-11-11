"""Game API routes."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import dinoboard_engine as engine

from .sessions import (
    GAME_CONFIGS,
    VARIANT_TO_BASE,
    SESSION_TTL_SECONDS,
    create_session,
    get_session,
    rebuild_game_session,
    session_response,
    sessions as _sessions,
    precompute_clear,
)
from .replay import make_replay_frame, build_frames_from_actions, full_replay_frames
from .visibility import actor_is_opaque
from .pipeline import (
    schedule_precompute,
    schedule_pipeline,
    schedule_pipeline_ai_only,
    cancel_pipeline,
    signal_cancel,
    _build_isolated_gs,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger("dinoboard.routes")

router = APIRouter(prefix="/api/games", tags=["games"])


class CreateGameRequest(BaseModel):
    game_id: str
    seed: int = 0
    human_player: int = 0
    num_players: int = 2
    difficulty: str = "expert"


class ActionRequest(BaseModel):
    action_id: int


@lru_cache(maxsize=1)
def _manifest_display_order() -> list[str]:
    """Display order = the order of `enabled` games in games/manifest.json.
    Single source of truth so adding/removing games happens in exactly one place."""
    manifest_path = PROJECT_ROOT / "games" / "manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    return [g["id"] for g in manifest["games"] if g.get("enabled", True)]


@router.get("/_admin/sessions")
def admin_sessions():
    """Diagnostic snapshot of in-memory sessions.

    No auth — intended for local debug only. Don't expose this in any
    production deployment without a gate (e.g. behind a Cloudflare
    Access policy or a separate localhost-only port).
    """
    import time
    now = time.time()
    rows = []
    for sid, sess in list(_sessions.items()):
        rows.append({
            "session_id": sid,
            "game_id": sess.get("game_id"),
            "actual_id": sess.get("actual_id"),
            "difficulty": sess.get("difficulty"),
            "plies": len(sess.get("action_history", [])),
            "is_terminal": sess["game_session"].is_terminal
                if "game_session" in sess else None,
            "created_age_s": round(now - sess.get("created_at", now), 1),
            "idle_age_s": round(now - sess.get("last_active", now), 1),
        })
    rows.sort(key=lambda r: r["idle_age_s"])
    return {
        "count": len(rows),
        "ttl_seconds": SESSION_TTL_SECONDS,
        "sessions": rows,
    }


@router.get("/available")
def available_games():
    display_order = _manifest_display_order()
    order_index = {gid: i for i, gid in enumerate(display_order)}
    # Filter to only games whose engine bundle is registered. A game may be
    # marked `enabled: false` in manifest (and thus not built) while still
    # having config files on disk; those games must not surface to the UI.
    registered = set(engine.available_games())
    registered |= {VARIANT_TO_BASE[v] for v in registered if v in VARIANT_TO_BASE}
    sorted_items = sorted(
        ((gid, cfg) for gid, cfg in GAME_CONFIGS.items() if gid in registered),
        key=lambda kv: (order_index.get(kv[0], len(display_order)), kv[0]),
    )
    games = []
    for gid, cfg in sorted_items:
        has_web = (PROJECT_ROOT / "games" / gid / "web" / "index.html").exists()
        games.append({
            "game_id": gid,
            "display_name": cfg.get("display_name", gid),
            "display_name_en": cfg.get("display_name_en", cfg.get("display_name", gid)),
            "players": cfg["players"],
            "has_web": has_web,
        })
    return {"games": games}


_SEED_SALT = secrets.token_bytes(16)


def _session_seed(client_seed: int) -> int:
    """Seed of a web game. The browser proposes one, but a client that knew
    the real seed could replay the game elsewhere (e.g. POST /api/replay/build)
    and read every hidden card — so it is mixed with a per-process secret.
    The result is logged for reproduction and returned by GET /replay once
    the game is over. DINOBOARD_WEB_CLIENT_SEED=1 (local debugging only)
    uses the client's seed as is."""
    if os.environ.get("DINOBOARD_WEB_CLIENT_SEED") == "1":
        return client_seed
    digest = hashlib.blake2b(str(client_seed).encode(), key=_SEED_SALT, digest_size=8).digest()
    return int.from_bytes(digest, "big") >> 11   # 53 bits: exact in JSON / JS


@router.post("")
def create_game(req: CreateGameRequest):
    try:
        seed = _session_seed(req.seed)
        session_id, sess = create_session(
            req.game_id, seed, req.human_player, req.num_players, req.difficulty)
        logger.info("web game %s: %s seed=%d", session_id, sess["actual_id"], seed)
    except Exception as e:
        raise HTTPException(400, str(e))

    sess["replay_frames"].append(make_replay_frame(sess, 0, "start", None, None))

    gs = sess["game_session"]
    if not gs.is_terminal and gs.current_player in sess["ai_players"]:
        pass
    elif req.difficulty == "expert":
        schedule_precompute(sess)

    return session_response(session_id, sess)


@router.get("/{session_id}")
def get_game_state(session_id: str):
    try:
        sess = get_session(session_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return session_response(session_id, sess)


@router.post("/{session_id}/action")
def apply_action(session_id: str, req: ActionRequest):
    """Human move. Returns immediately; pipeline runs in background."""
    try:
        sess = get_session(session_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    gs = sess["game_session"]
    if gs.is_terminal:
        raise HTTPException(400, "game is already over")
    # "Play for the opponent" only where the human sees everything that
    # opponent sees. Otherwise accepting / rejecting a move would tell the
    # human which moves the opponent's hidden cards allow.
    if actor_is_opaque(gs, sess["human_player"]):
        raise HTTPException(400, "cannot act for a player whose hidden information you cannot see")

    # Must wait for any running pipeline worker to finish before mutating
    # sess["game_session"] / action_history, otherwise the worker's late
    # writes race with the human move (duplicate AI moves, KeyError on
    # ai_move["action"], etc.). cancel_pipeline acquires the session's
    # pipeline_lock, which the worker holds for its entire run.
    cancel_pipeline(sess)

    action_info = gs.get_action_info(req.action_id)
    acting_player = gs.current_player

    try:
        # The human's own action is shown as is; `events` are the public
        # events of the move from the human's seat (cards of earlier,
        # hidden moves that this move turned face up, ...).
        trace = gs.apply_action_with_trace(req.action_id, sess["human_player"])
    except Exception as e:
        raise HTTPException(400, str(e))
    sess["action_history"].append(req.action_id)
    ply = len(sess["action_history"])
    sess["replay_frames"].append(
        make_replay_frame(sess, ply, f"player_{acting_player}", req.action_id, action_info,
                          analysis=None, events=trace["events"]))

    if not gs.is_terminal and gs.current_player in sess["ai_players"]:
        if sess["difficulty"] == "expert":
            schedule_pipeline(sess, session_id)
        else:
            schedule_pipeline_ai_only(sess, session_id)
    elif not gs.is_terminal and sess["difficulty"] == "expert":
        schedule_precompute(sess)
    elif gs.is_terminal and sess["difficulty"] == "expert":
        # Game just ended on the human's move (e.g. liarsdice "challenge").
        # No AI follow-up to schedule, but we still want analysis on this
        # final human frame so replay shows the recommended-move line on
        # the game-ending move. The pipeline worker handles ai_move=None
        # on terminal positions and routes the analysis to the most
        # recent human frame.
        schedule_pipeline(sess, session_id)

    return session_response(session_id, sess, {"action_info": action_info, "events": trace["events"]})


@router.post("/{session_id}/ai-action")
def trigger_ai_action(session_id: str):
    """Trigger AI-only pipeline (for AI-first games). Returns immediately."""
    try:
        sess = get_session(session_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    gs = sess["game_session"]
    if gs.is_terminal:
        raise HTTPException(400, "game is already over")
    if gs.current_player not in sess["ai_players"]:
        raise HTTPException(400, "not AI's turn")

    schedule_pipeline_ai_only(sess, session_id)
    return session_response(session_id, sess)


@router.get("/{session_id}/pipeline")
def get_pipeline_status(session_id: str):
    try:
        sess = get_session(session_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    pipe = sess["pipeline"]
    resp = {
        "phase": pipe["phase"],
        "ai_action": pipe["ai_action"],
        "ai_action_info": pipe["ai_action_info"],
        "ai_stats": pipe["ai_stats"],
        "ai_events": pipe["ai_events"],
        "analysis": pipe.get("analysis"),
    }
    if "error" in pipe:
        resp["error"] = pipe["error"]
    return resp


@router.post("/{session_id}/step-back")
def step_back(session_id: str):
    try:
        sess = get_session(session_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    history = sess["action_history"]
    if not history:
        raise HTTPException(400, "no moves to undo")

    cancel_pipeline(sess)

    if len(sess["replay_frames"]) > 1:
        sess["replay_frames"].pop()

    sess.get("ai_stats_full", {}).pop(len(history), None)
    history.pop()

    rebuild_game_session(sess)
    precompute_clear(sess)
    if sess["difficulty"] == "expert":
        schedule_precompute(sess)
    return session_response(session_id, sess)


@router.post("/{session_id}/ai-hint")
def ai_hint(session_id: str):
    import time

    try:
        sess = get_session(session_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    gs = sess["game_session"]
    if gs.is_terminal:
        raise HTTPException(400, "game is already over")
    # A hint is the best move for the player to move, computed from what
    # THAT player sees. For an opponent holding hidden cards it would give
    # them away.
    if actor_is_opaque(gs, sess["human_player"]):
        raise HTTPException(400, "no hint: the player to move holds hidden information")

    try:
        if sess["difficulty"] == "expert":
            pc = sess["precompute"]
            deadline = time.time() + 8.0
            while time.time() < deadline:
                if pc["result"] is not None:
                    break
                time.sleep(0.05)
            if pc["result"] is not None:
                hint = pc["result"]
            else:
                # Use isolated AI-session so the hint respects ai_use_action_filter.
                ai_gs = _build_isolated_gs(sess)
                hint = ai_gs.get_ai_action(
                    sess["simulations"],
                    0.0,
                    c_puct=sess["c_puct"],
                    world_cap=sess["world_cap"],
                    opponent_selection=sess["opponent_selection"],
                    top_k=sess["top_k"],
                    fpu_reduction=sess["fpu_reduction"],
                )
        elif sess["difficulty"] == "heuristic":
            hint = gs.get_heuristic_action(
                sess["heuristic_selector"],
                sess["heuristic_temperature"],
            )
        else:
            ai_gs = _build_isolated_gs(sess)
            simulations = sess["simulations"]
            hint = ai_gs.get_ai_action(
                simulations,
                0.0,
                c_puct=sess["c_puct"],
                world_cap=sess["world_cap"],
                opponent_selection=sess["opponent_selection"],
                top_k=sess["top_k"],
                fpu_reduction=sess["fpu_reduction"],
            )
    except Exception as e:
        raise HTTPException(400, str(e))

    result = {
        "action": hint["action"],
        "action_info": hint["action_info"],
    }
    if "stats" in hint:
        result["stats"] = hint["stats"]
    return result


@router.get("/{session_id}/replay")
def get_replay(session_id: str):
    try:
        sess = get_session(session_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    gs = sess["game_session"]
    human_p = sess["human_player"]
    players = {}
    for p in range(gs.num_players):
        if p == human_p:
            players[f"player_{p}"] = {"name": "玩家", "type": "human"}
        else:
            players[f"player_{p}"] = {"name": f"AI {p}", "type": "ai"}
    # Running game: the frames hold what the human may see, and the seed
    # (which determines every hidden card) stays on the server. Finished
    # game: everything, so the replay can show the opponents' side.
    finished = gs.is_terminal
    return {
        "game_id": sess["actual_id"],
        "seed": sess["seed"] if finished else None,
        "players": players,
        "frames": full_replay_frames(sess) if finished else sess["replay_frames"],
    }


replay_router = APIRouter(prefix="/api/replay", tags=["replay"])


class BuildReplayRequest(BaseModel):
    game_id: str
    seed: int
    action_history: list[int]


@replay_router.post("/build")
def build_replay(req: BuildReplayRequest):
    """Build frames from action_history (for replays that only store actions)."""
    try:
        frames = build_frames_from_actions(req.game_id, req.seed, req.action_history)
    except Exception as e:
        raise HTTPException(400, str(e))
    return {"frames": frames}


@replay_router.get("/file")
def get_replay_file(path: str):
    import json
    resolved = (PROJECT_ROOT / path).resolve()
    if not str(resolved).startswith(str(PROJECT_ROOT)):
        raise HTTPException(403, "path outside project root")
    if not resolved.exists():
        raise HTTPException(404, f"file not found: {path}")
    try:
        data = json.loads(resolved.read_text())
    except Exception as e:
        raise HTTPException(400, f"invalid JSON: {e}")
    if "frames" not in data and "action_history" not in data:
        raise HTTPException(400, "replay file must contain 'frames' or 'action_history'")
    return data
