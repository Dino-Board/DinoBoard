"""AI-only session: observation-in, action-out. No state crosses the API boundary."""
from __future__ import annotations

import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import dinoboard_engine as engine

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "platform"))

from model_paths import (  # noqa: E402
    base_game_id as _base_game_id,
    expected_model_path as _expected_model_path,
    find_model_path as _find_model_path,
)
from session_factory import SessionConfig, SessionFactory  # noqa: E402

from training.mcts_profile import resolve_profile  # noqa: E402


def _resolve_strength(
        game_id: str) -> tuple[int, int, float, float, str, bool, float, float, int, int, float | None]:
    """Read AI strength from the `web_expert` MCTS profile.

    AI API contract: strength = web_expert difficulty, never client-supplied.
    Resolver raises if the profile is missing or malformed.

    Returns (simulations, world_cap, c_puct, temperature, opponent_selection,
             schedule_enabled, temperature_initial, temperature_final,
             temperature_decay_plies, top_k, fpu_reduction).
    """
    base = _base_game_id(game_id)
    p = resolve_profile(base, "web_expert")
    return (p.simulations, p.world_cap, p.c_puct, p.temperature, p.opponent_selection,
            p.temperature_schedule_enabled,
            p.temperature_initial, p.temperature_final,
            p.temperature_decay_plies, p.top_k, p.fpu_reduction)


@dataclass
class AISession:
    """Observation-only AI session.

    Internally wraps a C++ GameSession; the internal GameSession is an
    implementation detail and is never exposed to callers. The API surface
    accepts only (game_id, seed, my_seat) at creation and action IDs as
    observations, returning only action IDs on decide.

    Thread-safety: each method acquires the session's lock; concurrent calls
    to the same session serialize. Different sessions are independent.
    """

    session_id: str
    game_id: str
    num_players: int
    my_seat: int
    simulations: int
    world_cap: int
    c_puct: float
    temperature: float
    opponent_selection: str
    has_public_state_applier: bool

    _gs: engine.GameSession = field(repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _action_count: int = 0
    _closed: bool = False
    temperature_schedule_enabled: bool = False
    temperature_initial: float = 0.0
    temperature_final: float = 0.0
    temperature_decay_plies: int = 0
    top_k: int = 0
    fpu_reduction: float | None = None

    def observe(
        self,
        action_id: int,
        events: Optional[list[dict]] = None,
        public_snapshot: Optional[dict] = None,
    ) -> None:
        """Record that a player played action_id.

        `action_id` is the public projection of the move (the trace's
        "action", IGameRules::public_action): for moves that hide a card it
        is a stand-in id, never the hidden choice itself.

        Caller must send observations in turn order — including for the AI's
        own moves. `decide()` is non-mutating: it picks an action and returns
        it, but does not advance the session. The caller then applies the
        action on its ground-truth game and forwards the resulting trace
        (events + snapshot) back via `observe()`. This mirrors how an external
        game server notifies the AI of every player's move and is the only
        path through which session state advances.

        For snapshot-path games (those with a registered public_state_applier
        — Splendor / Love Letter / Coup / 6 nimmt! / Liar's Dice carry hidden info + tracker; Azul is
        fully public but still travels the snapshot path), the caller MUST pass
        `events` and `public_snapshot` — without them the session's public
        state cannot be rebuilt from truth and (where a tracker exists) the
        belief tracker cannot stay consistent. A ValueError is raised in that
        case. For fully-public no-snapshot games (TicTacToe / Quoridor), these
        arguments must be omitted/empty — the session advances by replaying
        the action through `do_action_fast` on its seat state.
        """
        events = events or []
        public_snapshot = public_snapshot or {}
        with self._lock:
            if self._closed:
                raise RuntimeError(f"AISession {self.session_id} is closed")
            if self._gs.is_terminal:
                raise RuntimeError(
                    f"AISession {self.session_id}: game is already terminal, "
                    f"cannot observe more actions")

            if self.has_public_state_applier:
                if not public_snapshot:
                    raise ValueError(
                        f"AISession {self.session_id}: game {self.game_id!r} is a "
                        f"snapshot-path game; observe() requires public_snapshot "
                        f"(events may be empty for no-events games, but "
                        f"public_snapshot is always required). Caller passed "
                        f"action_id alone — the session cannot rebuild public "
                        f"state from that.")
                self._gs.apply_observation(
                    action_id,
                    events,
                    public_snapshot,
                )
            else:
                if events or public_snapshot:
                    raise ValueError(
                        f"AISession {self.session_id}: game {self.game_id!r} is a "
                        f"fully-public no-snapshot game; observe() must be "
                        f"called with action_id only (no events / snapshot) — "
                        f"the session advances via do_action_fast on its seat "
                        f"state.")
                legal = self._gs.get_legal_actions()
                if action_id not in legal:
                    raise ValueError(
                        f"AISession {self.session_id}: observed action {action_id} "
                        f"is not legal from current position. Legal: {legal}.")
                self._gs.apply_action(action_id)

            self._action_count += 1

    def decide(self) -> dict:
        """Ask the AI to pick an action at the current position.

        Returns {"action_id": int, "action_info": dict, "stats": dict}. This
        call is NON-MUTATING — the session's state is not advanced. The caller
        is expected to apply this action to its ground-truth game and then
        forward the resulting trace back via `observe()` (just like for any
        other player's move). This guarantees AI session state is rebuilt
        from the message stream only, never from speculative rule execution
        on a sampled-hidden world.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError(f"AISession {self.session_id} is closed")
            if self._gs.is_terminal:
                raise RuntimeError(
                    f"AISession {self.session_id}: game is already terminal")
            current = self._gs.current_player
            if current != self.my_seat:
                raise RuntimeError(
                    f"AISession {self.session_id}: it is player {current}'s "
                    f"turn, but this session is configured for seat "
                    f"{self.my_seat}. The caller must have missed an observe()."
                )
            kwargs = dict(
                simulations=self.simulations,
                world_cap=self.world_cap,
                c_puct=self.c_puct,
                temperature=self.temperature,
                opponent_selection=self.opponent_selection,
                top_k=self.top_k,
                fpu_reduction=self.fpu_reduction,
            )
            if self.temperature_schedule_enabled:
                kwargs["temperature_initial"] = self.temperature_initial
                kwargs["temperature_final"] = self.temperature_final
                kwargs["temperature_decay_plies"] = self.temperature_decay_plies
            result = self._gs.get_ai_action(**kwargs)
            action_id = result["action"]
            return {
                "action_id": action_id,
                "action_info": result["action_info"],
                "stats": result["stats"],
            }

    def status(self) -> dict:
        """Return lightweight status. Never returns game-state fields."""
        with self._lock:
            if self._closed:
                return {
                    "session_id": self.session_id,
                    "closed": True,
                    "is_terminal": None,
                    "current_player": None,
                    "winner": None,
                    "actions_observed": self._action_count,
                }
            return {
                "session_id": self.session_id,
                "closed": False,
                "game_id": self.game_id,
                "num_players": self.num_players,
                "my_seat": self.my_seat,
                "is_terminal": self._gs.is_terminal,
                "current_player": self._gs.current_player if not self._gs.is_terminal else None,
                "winner": self._gs.winner if self._gs.is_terminal else None,
                "actions_observed": self._action_count,
            }

    def close(self) -> None:
        with self._lock:
            self._closed = True


class SessionStore:
    """In-memory session registry. FastAPI instantiates one per process."""

    def __init__(self) -> None:
        self._sessions: dict[str, AISession] = {}
        self._lock = threading.Lock()

    def create(
        self,
        game_id: str,
        seed: int,
        my_seat: int,
        model_path_override: Optional[str] = None,
        initial_observation: Optional[dict] = None,
    ) -> AISession:
        if game_id not in engine.available_games():
            raise ValueError(
                f"unknown game_id {game_id!r}. Available: {engine.available_games()}")

        meta = engine.game_metadata(game_id)
        num_players = meta["num_players"]
        has_public_state_applier = bool(meta["has_public_state_applier"])
        if my_seat < 0 or my_seat >= num_players:
            raise ValueError(
                f"my_seat={my_seat} out of range for game with {num_players} players")

        if has_public_state_applier and initial_observation is None:
            # Snapshot-path games (Azul, Splendor, Love Letter, Coup, 6 nimmt!, Liar's Dice) carry
            # perspective-private starting facts that the AI session cannot
            # reconstruct from action_id alone. Without initial_observation
            # the session's own seed-generated hidden state would diverge
            # from truth — and would silently bias the AI's first decision
            # on its starting turn (e.g. Love Letter seat 0 deciding off the
            # wrong starting hand). Required, not optional.
            raise ValueError(
                f"game {game_id!r} is a snapshot-path game; "
                f"initial_observation is required (carries perspective's "
                f"starting facts — own hand for hidden-info games, "
                f"shared starting tableau for public-snapshot games).")
        if not has_public_state_applier and initial_observation:
            raise ValueError(
                f"game {game_id!r} is a fully-public no-snapshot game; "
                f"initial_observation must not be provided.")

        if model_path_override is not None:
            model_path = model_path_override
        else:
            model_path = _find_model_path(game_id)
        if not model_path:
            raise FileNotFoundError(
                f"no trained model found for {game_id}. "
                f"Expected at {_expected_model_path(game_id).relative_to(_PROJECT_ROOT)}.")

        gs = SessionFactory.create(SessionConfig(
            game_id=game_id,
            seed=seed,
            model_path=model_path,
            use_action_filter=False,
        ))

        if has_public_state_applier and initial_observation is not None:
            gs.apply_initial_observation(my_seat, initial_observation)

        (simulations, world_cap, c_puct, temperature, opponent_selection,
         t_sched_enabled, t_initial, t_final, t_decay,
         top_k, fpu_reduction) = _resolve_strength(game_id)

        session_id = uuid.uuid4().hex[:12]
        sess = AISession(
            session_id=session_id,
            game_id=game_id,
            num_players=num_players,
            my_seat=my_seat,
            simulations=simulations,
            world_cap=world_cap,
            c_puct=c_puct,
            temperature=temperature,
            opponent_selection=opponent_selection,
            has_public_state_applier=has_public_state_applier,
            temperature_schedule_enabled=t_sched_enabled,
            temperature_initial=t_initial,
            temperature_final=t_final,
            temperature_decay_plies=t_decay,
            top_k=top_k,
            fpu_reduction=fpu_reduction,
            _gs=gs,
        )
        with self._lock:
            self._sessions[session_id] = sess
        return sess

    def get(self, session_id: str) -> AISession:
        with self._lock:
            sess = self._sessions.get(session_id)
        if sess is None:
            raise KeyError(f"session {session_id} not found")
        return sess

    def close(self, session_id: str) -> None:
        with self._lock:
            sess = self._sessions.pop(session_id, None)
        if sess is not None:
            sess.close()

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)


# Process-global store. The platform app creates one and passes it to routes.
_STORE: Optional[SessionStore] = None


def get_store() -> SessionStore:
    global _STORE
    if _STORE is None:
        _STORE = SessionStore()
    return _STORE
