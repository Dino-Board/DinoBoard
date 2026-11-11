"""Replay frame construction and serialization.

Unified replay JSON format:
{
  "game_id": str,
  "seed": int,
  "players": {
    "player_0": {"name": str, "type": "human"|"model"|"heuristic"|"ai"},
    "player_1": {"name": str, "type": ...},
    ...  # N players
  },
  "result": {"winner": int, "draw": bool, "total_plies": int},
  "config": { ... },             # optional match config
  "action_history": [int, ...],
  "frames": [ ... ]              # optional; rebuilt from action_history if absent
}

Actor convention: "player_0", "player_1", ..., "player_N-1", "start".
"""
from __future__ import annotations

from typing import Any, Optional

from .visibility import viewer_state


def make_replay_frame(
    sess: dict,
    ply_index: int,
    actor: str,
    action_id: Optional[int],
    action_info: Optional[dict],
    ai_stats: Optional[dict] = None,
    analysis: Optional[dict] = None,
    events: Optional[list] = None,
) -> dict:
    gs = sess["game_session"]
    # Frames of a running game are served to the human (GET /replay, the
    # replay panel during play): they hold what the human seat may see,
    # like every other response (visibility.py). full_replay_frames() below
    # restores the complete states once the game is over.
    frame = {
        "ply_index": ply_index,
        "actor": actor,
        "action_id": action_id,
        "action_info": action_info,
        "state": viewer_state(gs, sess.get("human_player", 0)),
        "current_player": gs.current_player,
        "is_terminal": gs.is_terminal,
        "winner": gs.winner,
        "ai_stats": ai_stats,
        "analysis": analysis,
    }
    if events is not None:
        # Public events of this move as the human seat observes them
        # (move frames of a live session only; absent on the start frame
        # and on frames rebuilt from an action history).
        frame["events"] = events
    return frame


# Search statistics keyed by root action. They spell out which action the
# AI preferred, so they must go when the action itself is hidden from the
# human; the value estimates (win-rate display) stay.
_PER_ACTION_STATS = ("action_values", "root_actions", "root_action_visits")


def public_ai_stats(ai_stats: Optional[dict]) -> Optional[dict]:
    if ai_stats is None:
        return None
    return {k: v for k, v in ai_stats.items() if k not in _PER_ACTION_STATS}


def full_replay_frames(sess: dict) -> list[dict]:
    """Frames of a FINISHED game with nothing withheld: complete states, the
    real action ids / infos and the AI's own search statistics, rebuilt by
    replaying the action history. Analysis and events are kept from the
    live frames. Only call this when the game is over."""
    live = sess["replay_frames"]
    history = sess["action_history"]
    if len(live) != len(history) + 1:
        return live
    truth = build_frames_from_actions(sess["actual_id"], sess["seed"], history)
    stats = sess.get("ai_stats_full", {})
    frames = []
    for i, (shown, real) in enumerate(zip(live, truth)):
        frame = dict(shown)
        frame["state"] = real["state"]
        frame["action_id"] = real["action_id"]
        frame["action_info"] = real["action_info"]
        if i in stats:
            frame["ai_stats"] = stats[i]
        frames.append(frame)
    return frames


def build_analysis_dict(
    best_wr: float,
    actual_wr: float,
    best_action: int,
    best_action_info: Optional[dict],
) -> dict:
    drop = max(0, (best_wr - actual_wr) * 100)
    return {
        "best_win_rate": round(best_wr, 4),
        "actual_win_rate": round(actual_wr, 4),
        "drop_score": round(drop, 1),
        "best_action": best_action,
        "best_action_info": best_action_info,
    }


def build_frames_from_actions(
    game_id: str,
    seed: int,
    action_history: list[int],
) -> list[dict]:
    """Replay action_history through a GameSession, producing frames."""
    from session_factory import SessionConfig, SessionFactory

    gs = SessionFactory.create(SessionConfig(
        game_id=game_id,
        seed=seed,
        model_path="",
        use_action_filter=False,
    ))
    frames = [{
        "ply_index": 0,
        "actor": "start",
        "action_id": None,
        "action_info": None,
        "state": gs.get_state_dict(),
        "current_player": gs.current_player,
        "is_terminal": gs.is_terminal,
        "winner": gs.winner,
        "ai_stats": None,
        "analysis": None,
    }]

    for i, action_id in enumerate(action_history):
        player = gs.current_player
        action_info = gs.get_action_info(action_id)
        gs.apply_action(action_id)
        frames.append({
            "ply_index": i + 1,
            "actor": f"player_{player}",
            "action_id": action_id,
            "action_info": action_info,
            "state": gs.get_state_dict(),
            "current_player": gs.current_player,
            "is_terminal": gs.is_terminal,
            "winner": gs.winner,
            "ai_stats": None,
            "analysis": None,
        })

    return frames


def build_replay_dict(
    game_id: str,
    seed: int,
    action_history: list[int],
    players: dict[str, dict[str, Any]],
    result: dict[str, Any],
    config: Optional[dict[str, Any]] = None,
) -> dict:
    """Build a standard replay dict (without frames — lightweight for storage)."""
    replay: dict[str, Any] = {
        "game_id": game_id,
        "seed": seed,
        "players": players,
        "result": result,
        "action_history": action_history,
    }
    if config:
        replay["config"] = config
    return replay
