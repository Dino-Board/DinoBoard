"""AI API routes. Contract: no game state crosses this boundary.

Endpoints:
  POST /ai/sessions              — create a session, returns session_id
  POST /ai/sessions/{id}/observe — record an action the AI observed
  POST /ai/sessions/{id}/decide  — ask the AI for its next action
  DELETE /ai/sessions/{id}       — close a session
  GET /ai/sessions/{id}          — status (actions observed, turn, terminal)

Snapshot-path games (splendor / loveletter / coup / 6nimmit / liarsdice
carry hidden info + trackers; azul is fully public but still uses the
snapshot protocol)
require the caller to pass `events` and `public_snapshot` on every
observe call so the AI's belief tracker (where present) stays consistent
with truth and the session's public state is rebuilt from the wire — the
action_id alone is not enough. Fully-public no-tracker games
(tictactoe / quoridor) only need `action_id`.

AI strength is server-controlled — `simulations` / `world_cap` / `c_puct` /
`temperature` / `opponent_selection` are not wire fields. The server resolves
them per-game via `resolve_profile(game_id, "web_expert")` against the unified
profile resolver (`mcts.json` baseline + `web.json::web_expert` delta); see
`training/mcts_profile.py` and `docs/guide/CONFIG_REFERENCE.md`.
`seed` is optional; omitting it lets the
server pick a fresh `secrets.randbits(64)` value, which is the default for
production clients. Tests may pass an explicit `seed` for reproducibility.
"""
from __future__ import annotations

import secrets
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .sessions import get_store

router = APIRouter(prefix="/ai/sessions", tags=["ai"])


class CreateSessionRequest(BaseModel):
    game_id: str = Field(..., description="e.g. 'quoridor', 'splendor'")
    seed: Optional[int] = Field(
        None,
        description="Optional RNG seed for the AI session's internal RNG, used "
                    "during MCTS sim-entry root determinization (sim_tracker."
                    "randomize_unseen on a clone). Independent from ground "
                    "truth — the session's own viz=0 slots are never freshened "
                    "outside sim entry, so seed choice does not affect public "
                    "state observed from snapshots. Omit to let the server "
                    "pick a secrets.randbits(64) value.",
    )
    my_seat: int = Field(..., description="Which player the AI is playing as (0-indexed)")
    initial_observation: Optional[dict] = Field(
        None,
        description="Perspective-specific facts known at game start. "
                    "**REQUIRED** for snapshot-path games (Azul, Splendor, "
                    "Love Letter, Coup, 6 nimmt!, Liar's Dice) — without it the AI session's own "
                    "seed-generated hidden state would silently diverge from "
                    "truth and the AI's first decision on its starting turn "
                    "would be made off the wrong starting hand / tableau. "
                    "Caller obtains it via `GameSession.extract_initial_"
                    "observation(seat)`. **Must be omitted** for fully-public "
                    "no-snapshot games (TicTacToe, Quoridor). The server "
                    "rejects mismatches with HTTP 400.",
    )


class CreateSessionResponse(BaseModel):
    session_id: str
    game_id: str
    num_players: int
    my_seat: int
    current_player: int
    is_terminal: bool


class PublicEvent(BaseModel):
    kind: str = Field(..., description="Game-specific event identifier "
                                       "(e.g. 'reveal_card', 'token_supply_delta').")
    payload: dict = Field(default_factory=dict)


class ObserveRequest(BaseModel):
    action_id: int = Field(..., description="The action that was just played by the "
                                            "current player. **Includes the AI's own "
                                            "moves**: /decide is non-mutating — it only "
                                            "returns the chosen action_id, it does not "
                                            "advance the session. The caller must apply "
                                            "the action to ground truth and then re-feed "
                                            "it here so the session sees its own move "
                                            "the same way it sees opponents'. Pass the "
                                            "action's public projection — the `action` "
                                            "field of `apply_action_with_trace` — not the "
                                            "raw id: where an id names something hidden "
                                            "(6 nimmt!'s face-down card, the character a "
                                            "Coup exchange returns) the projection is a "
                                            "stand-in outside the action space.")
    events: list[PublicEvent] = Field(
        default_factory=list,
        description="Public events emitted by this action (in producer order). "
                    "Required for snapshot-path games with a registered tracker "
                    "(Love Letter, Splendor, Coup, 6 nimmt!, Liar's Dice); pass `[]` for snapshot-path "
                    "games without a tracker (Azul) and for fully-public "
                    "no-snapshot games (TicTacToe, Quoridor).",
    )
    public_snapshot: dict = Field(
        default_factory=dict,
        description="Truth-side dump of the game's public fields after this "
                    "action. Required for snapshot-path games (Azul, Splendor, "
                    "Love Letter, Coup, 6 nimmt!, Liar's Dice); ignored for fully-public no-snapshot "
                    "games (TicTacToe, Quoridor) which advance via do_action_fast "
                    "on the AI-side seat state.",
    )


class ObserveResponse(BaseModel):
    actions_observed: int
    current_player: Optional[int]
    is_terminal: bool


class DecideResponse(BaseModel):
    action_id: int
    action_info: dict
    stats: dict
    current_player: Optional[int]
    is_terminal: bool


class StatusResponse(BaseModel):
    session_id: str
    closed: bool
    game_id: Optional[str] = None
    num_players: Optional[int] = None
    my_seat: Optional[int] = None
    is_terminal: Optional[bool]
    current_player: Optional[int]
    winner: Optional[int] = None
    actions_observed: int


@router.post("", response_model=CreateSessionResponse)
def create_session(req: CreateSessionRequest):
    seed = req.seed if req.seed is not None else secrets.randbits(64)
    try:
        sess = get_store().create(
            game_id=req.game_id,
            seed=seed,
            my_seat=req.my_seat,
            initial_observation=req.initial_observation,
        )
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))

    st = sess.status()
    return CreateSessionResponse(
        session_id=sess.session_id,
        game_id=sess.game_id,
        num_players=sess.num_players,
        my_seat=sess.my_seat,
        current_player=st["current_player"] if st["current_player"] is not None else -1,
        is_terminal=st["is_terminal"] if st["is_terminal"] is not None else False,
    )


@router.post("/{session_id}/observe", response_model=ObserveResponse)
def observe(session_id: str, req: ObserveRequest):
    try:
        sess = get_store().get(session_id)
    except KeyError as e:
        raise HTTPException(404, str(e))

    try:
        sess.observe(
            req.action_id,
            events=[e.model_dump() for e in req.events],
            public_snapshot=req.public_snapshot,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))

    st = sess.status()
    return ObserveResponse(
        actions_observed=st["actions_observed"],
        current_player=st["current_player"],
        is_terminal=st["is_terminal"] or False,
    )


@router.post("/{session_id}/decide", response_model=DecideResponse)
def decide(session_id: str):
    try:
        sess = get_store().get(session_id)
    except KeyError as e:
        raise HTTPException(404, str(e))

    try:
        result = sess.decide()
    except RuntimeError as e:
        raise HTTPException(409, str(e))

    st = sess.status()
    return DecideResponse(
        action_id=result["action_id"],
        action_info=result["action_info"],
        stats=result["stats"],
        current_player=st["current_player"],
        is_terminal=st["is_terminal"] or False,
    )


@router.get("/{session_id}", response_model=StatusResponse)
def get_status(session_id: str):
    try:
        sess = get_store().get(session_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return StatusResponse(**sess.status())


@router.delete("/{session_id}")
def close_session(session_id: str):
    get_store().close(session_id)
    return {"closed": True}
