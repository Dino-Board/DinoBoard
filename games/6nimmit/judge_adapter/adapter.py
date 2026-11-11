#!/usr/bin/env python3
"""Stdin/stdout adapter that lets DinoBoard's AI play behind the 2017
6 nimmt! tournament judge (`games/6nimmit/参考/main.cpp`).

The judge fork+execl's this script every turn, writes the public game
history to its stdin, and reads a single int from its stdout. We
translate that wire format to/from the DinoBoard AI REST API
(`POST /ai/sessions[/{id}/observe|decide]`), keeping a long-lived AI
session on the server so model + tracker survive across forks.

Plan: docs/plans/6NIMMT_PLAN.md §1.8.

Two fork modes per the judge:
    judge → fork → execl(this, [no flag])     # commit phase: print 1..104
    judge → fork → execl(this, ["-small"])     # SMALL phase: print row 0..3

Server-seat-0 simplification (§1.8.0): the AI always sits at server seat
0 regardless of which `player_id` the judge assigned us. Judge seat
`(my_judge_id + s) mod N` maps to server seat `s`. The server-side AI
session sees a relabelled but otherwise identical game.

State persistence between forks:
  /tmp/6nimmit_judge_session_pid{ppid}_p{pid}.json holds enough to rebuild
  the side-GT replay and to know how many actions have already been
  /observe'd on the AI session — so partial-round progress (mid-round
  SMALL forks) is preserved without double-feeding actions.

Single-game (-s 1) only — long-game variants carry scores across
sub-games and stress score-aware policy that DinoBoard 6 nimmt! does
not currently train for.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Up three levels: games/6nimmit/judge_adapter/ -> games/6nimmit/ -> games/ -> <repo root>
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import dinoboard_engine as engine  # noqa: E402

from initial_observation import build_initial_observation  # noqa: E402


SERVER_URL = os.environ.get("DINOBOARD_AI_SERVER", "http://127.0.0.1:8765")

# One file per (judge process, judge player_id). The judge `-g K` knob
# runs multiple games in the same parent process, so we key on parent
# pid (judge) + player_id, not own pid (we are recreated per fork).
SESSION_FILE_FMT = "/tmp/6nimmit_judge_session_pid{ppid}_p{pid}.json"

# Hard-coded per plan §1.8 — DinoBoard ships only 1-104 / single-game
# 6 nimmt!, no Tactics / -c K<104 / -s 0 long-game variants.
SUPPORTED_NUM_CARDS = 104
NUM_ROUNDS = 10


def _log_path() -> str:
    # Adapter-side log (stdin / decisions / errors). One fork per ply, so
    # size stays small. Path overridable via env var so the judge's log
    # directory isn't polluted in tests. Resolved lazily so tests can
    # monkey-patch the env var after the module is imported.
    return os.environ.get("DINOBOARD_ADAPTER_LOG",
                           "/tmp/6nimmit_judge_adapter.log")


def _log(msg: str) -> None:
    try:
        with open(_log_path(), "a") as f:
            f.write(f"[pid={os.getpid()} ppid={os.getppid()} t={time.time():.3f}] {msg}\n")
    except Exception:
        pass


# --- stdin parsing --------------------------------------------------------

@dataclass
class JudgeStdin:
    player_id: int           # judge-side player_id (0..N-1)
    current_round: int       # judge's `current_round` value AT FORK TIME
    hand: list[int]          # 1..104, length = NUM_ROUNDS - current_round
    num_players: int
    num_cards: int
    initial_table: list[list[int]]   # [4][5] at game start
    history: list[dict]              # {"lastc": [N], "table_after": [4][5] | None}
    scores: list[int]
    is_small: bool                   # True if -small flag


def parse_judge_stdin(text: str, is_small: bool) -> JudgeStdin:
    """Parse judge stdin per `参考/main.cpp:200-211 + record_current() + update()`.

    Layout (commit fork):
        line:       player_id current_round
        line:       hand[NUM_ROUNDS-current_round] space-separated
        line:       num_players num_cards
        4 lines:    initial table (5 ints per row)
        for each k in 0..current_round-1:
            line:   lastc[N]
            4 lines: table_after_round_k
        line:       scores[N]

    Layout (SMALL fork): identical, except the LAST round in history has
    `lastc` only, no `table_after` — the round is mid-resolution. The
    judge's `update()` increments current_round before the apply loop, so
    `current_round` here is k+1 where round k is the in-flight round.
    """
    toks = text.split()
    i = 0
    player_id = int(toks[i]); i += 1
    current_round = int(toks[i]); i += 1
    hand_size = NUM_ROUNDS - current_round
    hand = [int(x) for x in toks[i:i + hand_size]]
    i += hand_size

    num_players = int(toks[i]); i += 1
    num_cards = int(toks[i]); i += 1

    initial_table = []
    for r in range(4):
        initial_table.append([int(x) for x in toks[i:i + 5]])
        i += 5

    full_rounds = current_round - 1 if is_small else current_round
    history = []
    for _ in range(full_rounds):
        lastc = [int(x) for x in toks[i:i + num_players]]
        i += num_players
        table_after = []
        for r in range(4):
            table_after.append([int(x) for x in toks[i:i + 5]])
            i += 5
        history.append({"lastc": lastc, "table_after": table_after})
    if is_small:
        # In-flight round: lastc only.
        in_flight_lastc = [int(x) for x in toks[i:i + num_players]]
        i += num_players
        history.append({"lastc": in_flight_lastc, "table_after": None})

    scores = [int(x) for x in toks[i:i + num_players]]
    i += num_players

    return JudgeStdin(player_id=player_id,
                      current_round=current_round,
                      hand=hand,
                      num_players=num_players,
                      num_cards=num_cards,
                      initial_table=initial_table,
                      history=history,
                      scores=scores,
                      is_small=is_small)


# --- HTTP shims (test-monkeypatchable) ------------------------------------

def _http_post(path: str, payload: dict, timeout: float = 30.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(SERVER_URL + path, data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _http_delete(path: str, timeout: float = 5.0) -> None:
    req = urllib.request.Request(SERVER_URL + path, method="DELETE")
    try:
        urllib.request.urlopen(req, timeout=timeout).read()
    except Exception:
        pass


def api_create_session(game_id: str, my_seat: int,
                        initial_observation: dict) -> str:
    payload = {"game_id": game_id, "my_seat": my_seat,
               "initial_observation": initial_observation}
    return _http_post("/ai/sessions", payload)["session_id"]


def api_observe(sid: str, action_id: int, events: list, public_snapshot: dict) -> dict:
    return _http_post(f"/ai/sessions/{sid}/observe", {
        "action_id": action_id,
        "events": events,
        "public_snapshot": public_snapshot,
    })


def api_decide(sid: str) -> dict:
    return _http_post(f"/ai/sessions/{sid}/decide", {})


def api_close(sid: str) -> None:
    _http_delete(f"/ai/sessions/{sid}")


# --- Saved state ----------------------------------------------------------

@dataclass
class SavedState:
    session_id: str
    my_judge_id: int
    num_players: int
    initial_hand: list[int]
    initial_row_heads: list[int]
    ai_action_count: int                    # /observe's already applied
    in_flight_smalls: list[int] = field(default_factory=list)


def _session_file_path(player_id: int) -> str:
    return SESSION_FILE_FMT.format(ppid=os.getppid(), pid=player_id)


def load_saved_state(player_id: int) -> Optional[SavedState]:
    path = _session_file_path(player_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        return SavedState(**d)
    except Exception:
        return None


def write_saved_state(player_id: int, st: SavedState) -> None:
    path = _session_file_path(player_id)
    with open(path, "w") as f:
        json.dump(st.__dict__, f)


def clear_saved_state(player_id: int) -> None:
    path = _session_file_path(player_id)
    if os.path.exists(path):
        try:
            os.unlink(path)
        except Exception:
            pass


# --- Side-GT driver -------------------------------------------------------

class SideGt:
    """Local replay used to derive (events, public_snapshot) traces per
    ply. Holds the full action log so we can rebuild a fresh clone for
    SMALL row inference (engine.GameSession does not expose a Python
    clone, and `do_action_fast` does not support undo — see CLAUDE.md).
    Re-replay is cheap (≤ NUM_ROUNDS × N actions per game)."""

    def __init__(self, num_players: int, hand: list[int], row_heads: list[int]):
        self._num_players = num_players
        self._hand = list(hand)
        self._row_heads = list(row_heads)
        self._actions: list[int] = []
        self._sess, _ = build_initial_observation(num_players, hand, row_heads, seed=0)

    @property
    def gs(self):
        return self._sess

    def apply(self, action_id: int) -> dict:
        """Apply on side GT and return the (events, public_snapshot)
        trace for perspective=0 (the AI's seat)."""
        trace = self._sess.apply_action_with_trace(action_id, 0)
        self._actions.append(action_id)
        return trace

    def clone_replay(self) -> "SideGt":
        c = SideGt(self._num_players, self._hand, self._row_heads)
        for a in self._actions:
            c._sess.apply_action_with_trace(a, 0)
            c._actions.append(a)
        return c


def _read_table(gs) -> list[list[int]]:
    sd = gs.get_state_dict()
    return [list(row) for row in sd["table"]]


def _table_matches(a: list[list[int]], b: list[list[int]]) -> bool:
    """Row-index-stable equality. The judge's `update_stack` only
    mutates the chosen `argmax`'s contents, never re-orders rows; our
    `run_apply_loop` does the same — so rows can be compared in place."""
    return a == b


def _find_small_path(branch: SideGt,
                      expected_table_after: list[list[int]]) -> Optional[list[int]]:
    """Return the ordered list of SMALL row choices that, applied in
    order on `branch`, land on `expected_table_after`. Returns None if
    no such sequence exists. Does not mutate `branch` (works on clones).

    Termination is guaranteed: each SMALL choice consumes one apply_queue
    entry; queue is bounded by N (≤ 4)."""
    if branch.gs.is_terminal:
        return ([] if _table_matches(_read_table(branch.gs),
                                       expected_table_after) else None)
    legal = branch.gs.get_legal_actions()
    if not legal or not (104 <= legal[0] <= 107):
        return ([] if _table_matches(_read_table(branch.gs),
                                       expected_table_after) else None)
    for r in range(4):
        sub = branch.clone_replay()
        try:
            sub.gs.apply_action_with_trace(104 + r, 0)
            sub._actions.append(104 + r)
        except Exception:
            continue
        rest = _find_small_path(sub, expected_table_after)
        if rest is not None:
            return [r] + rest
    return None


# --- Action-sequence reconstruction ---------------------------------------

def _round_action_sequence(side_gt: SideGt, lastc_judge_order: list[int],
                            my_judge_id: int, num_players: int,
                            expected_table_after: Optional[list[list[int]]],
                            in_flight_known_smalls: Optional[list[int]] = None
                            ) -> list[int]:
    """Apply one round on `side_gt` and return the action_id sequence
    used (commits + SMALL choices, in the order side_gt consumed them).

    `expected_table_after` is the ground-truth table after this round's
    full resolution; SMALL choices are inferred against it. Pass None
    iff this is an in-flight round (SMALL fork) — then we do not
    auto-resolve any SMALL pause; we instead apply
    `in_flight_known_smalls` row choices in order, stopping when those
    run out (caller will decide the next SMALL itself)."""
    actions: list[int] = []

    # Server seat order: server_seat s ↔ judge seat (my_judge_id + s) % N.
    server_commits = [
        lastc_judge_order[(my_judge_id + s) % num_players]
        for s in range(num_players)
    ]
    for s, card in enumerate(server_commits):
        actor = side_gt.gs.current_player
        if actor != s:
            raise RuntimeError(
                f"side_gt current_player={actor}, expected server seat {s}")
        action_id = card - 1
        side_gt.apply(action_id)
        actions.append(action_id)

    if expected_table_after is not None:
        path = _find_small_path(side_gt, expected_table_after)
        if path is None:
            raise RuntimeError(
                f"could not infer SMALL row sequence: side table = "
                f"{_read_table(side_gt.gs)}, expected = {expected_table_after}")
        for r in path:
            side_gt.apply(104 + r)
            actions.append(104 + r)
    else:
        for r in (in_flight_known_smalls or []):
            side_gt.apply(104 + r)
            actions.append(104 + r)

    return actions


# --- Main flow ------------------------------------------------------------

def _game_id(num_players: int) -> str:
    # 2p ships under the bare base id ("6nimmit"); 3p / 4p take the
    # explicit suffix. Mirrors what training emits in games/6nimmit/model/.
    if num_players == 2:
        return "6nimmit"
    return f"6nimmit_{num_players}p"


def _events_to_wire(events_list) -> list[dict]:
    out = []
    for e in events_list:
        out.append({"kind": e["kind"], "payload": dict(e["payload"])})
    return out


def _read_judge_stdin(stream, is_small: bool) -> str:
    """Read exactly as many lines as the judge protocol mandates, then
    stop. The 2017 judge does not close its write end of the pipe until
    the child has exited (`fclose(child_in)` happens after `waitpid` in
    `main.cpp`), so a naive `sys.stdin.read()` blocks until SIGALRM /
    SIGKILL. This routine instead consumes the exact framing — round
    count is known from the header so total line count is fully
    determined. Same approach the C++ trivial bot would take with
    line-based scanf."""
    out = []
    out.append(stream.readline())          # "player_id current_round"
    cr = int(out[0].split()[1])
    out.append(stream.readline())          # hand
    out.append(stream.readline())          # "num_players num_cards"
    for _ in range(4):                     # initial table (4 rows)
        out.append(stream.readline())
    full_rounds = cr - 1 if is_small else cr
    for _ in range(full_rounds):
        out.append(stream.readline())      # lastc[N]
        for _ in range(4):                 # table_after row
            out.append(stream.readline())
    if is_small:
        out.append(stream.readline())      # in-flight lastc[N]
    out.append(stream.readline())          # scores[N]
    return "".join(out)


def main(argv: list[str]) -> int:
    is_small = "-small" in argv

    text = _read_judge_stdin(sys.stdin, is_small)
    _log(f"stdin (small={is_small}): {text!r}")
    inp = parse_judge_stdin(text, is_small)

    if inp.num_cards != SUPPORTED_NUM_CARDS:
        _log(f"FATAL: -c {inp.num_cards} not supported (DinoBoard ships only "
             f"-c {SUPPORTED_NUM_CARDS})")
        return 1

    # New-game detection: commit fork at round 0 with a 10-card hand.
    # SMALL fork can never be a new game (round 0 cannot SMALL — there's
    # nothing in apply queue at the start of round 0).
    fresh_game = (not is_small and inp.current_round == 0 and len(inp.hand) == 10)

    saved = load_saved_state(inp.player_id)
    if fresh_game:
        if saved is not None:
            api_close(saved.session_id)
            clear_saved_state(inp.player_id)
            saved = None
    else:
        if saved is None:
            _log(f"FATAL: no session state for player_id={inp.player_id} "
                 f"but current_round={inp.current_round}, is_small={is_small}")
            return 1

    if fresh_game:
        initial_hand = list(inp.hand)
        initial_row_heads = [row[0] for row in inp.initial_table]
        my_judge_id = inp.player_id
        sid = api_create_session(
            _game_id(inp.num_players),
            my_seat=0,
            initial_observation=build_initial_observation(
                inp.num_players, initial_hand, initial_row_heads, seed=0)[1])
        ai_action_count = 0
        in_flight_smalls: list[int] = []
    else:
        initial_hand = saved.initial_hand
        initial_row_heads = saved.initial_row_heads
        my_judge_id = saved.my_judge_id
        sid = saved.session_id
        ai_action_count = saved.ai_action_count
        in_flight_smalls = list(saved.in_flight_smalls)

    # On a commit fork the previous in-flight round (if any) has fully
    # resolved — judge will have written its `table_after` to history.
    # The saved `in_flight_smalls` are now part of that round's history
    # and will be re-derived by replay; drop them.
    if not is_small:
        in_flight_smalls = []

    # Build the canonical action sequence the AI session SHOULD have
    # observed by the time we decide.
    side_gt = SideGt(inp.num_players, initial_hand, initial_row_heads)
    full_history = inp.history if not is_small else inp.history[:-1]
    target_actions: list[int] = []
    for h in full_history:
        target_actions.extend(_round_action_sequence(
            side_gt, h["lastc"], my_judge_id, inp.num_players,
            expected_table_after=h["table_after"]))
    if is_small:
        h = inp.history[-1]
        target_actions.extend(_round_action_sequence(
            side_gt, h["lastc"], my_judge_id, inp.num_players,
            expected_table_after=None,
            in_flight_known_smalls=in_flight_smalls))

    # /observe the deltas the AI session hasn't seen yet. We must rebuild
    # the side_gt walk from scratch (the loop above mutated it through
    # the full target sequence; we need the INTERMEDIATE traces).
    if ai_action_count < len(target_actions):
        replay = SideGt(inp.num_players, initial_hand, initial_row_heads)
        for idx, a in enumerate(target_actions):
            trace = replay.apply(a)
            if idx >= ai_action_count:
                api_observe(sid, a, _events_to_wire(trace["events"]),
                             trace["public_snapshot"])

    # Sanity: side_gt's legal actions should match what the judge expects.
    if side_gt.gs.is_terminal:
        _log("FATAL: side_gt terminal at decide time")
        return 1
    legal = side_gt.gs.get_legal_actions()
    if is_small:
        if not legal or not (104 <= legal[0] <= 107):
            _log(f"FATAL: -small fork but side_gt not paused at SMALL "
                 f"(legal={legal})")
            return 1
    else:
        if not legal or legal[0] >= 104:
            _log(f"FATAL: commit fork but side_gt not in commit phase "
                 f"(legal={legal})")
            return 1

    decide = api_decide(sid)
    action_id = decide["action_id"]

    if is_small:
        if not (104 <= action_id <= 107):
            _log(f"FATAL: judge requested SMALL but AI returned "
                 f"non-SMALL action {action_id}")
            return 1
        out = action_id - 104
    else:
        if not (0 <= action_id <= 103):
            _log(f"FATAL: judge requested commit but AI returned "
                 f"non-commit action {action_id}")
            return 1
        out = action_id + 1

    print(out)
    sys.stdout.flush()
    _log(f"stdout: {out} (action_id={action_id})")

    # Update saved state. We deliberately do NOT /observe the just-
    # decided action: it'll be rebuilt at the next fork from history
    # (commit) or from in_flight_smalls (SMALL).
    new_in_flight = list(in_flight_smalls)
    if is_small:
        new_in_flight.append(action_id - 104)
    else:
        new_in_flight = []
    write_saved_state(inp.player_id, SavedState(
        session_id=sid,
        my_judge_id=my_judge_id,
        num_players=inp.num_players,
        initial_hand=initial_hand,
        initial_row_heads=initial_row_heads,
        ai_action_count=len(target_actions),
        in_flight_smalls=new_in_flight,
    ))

    return 0


def cli_entry(argv: list[str]) -> int:
    """Top-level wrapper: catches+logs uncaught exceptions so the judge's
    SIGKILL/timeout fallback (random move) is the worst case rather than
    a Python traceback to stderr."""
    try:
        return main(argv)
    except Exception as e:
        _log(f"UNCAUGHT: {type(e).__name__}: {e}")
        import traceback
        _log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(cli_entry(sys.argv))
