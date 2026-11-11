"""The web platform must not send the browser anything the human seat cannot see.

While a game is running, every response of `platform/game_service` (state,
legal actions, pipeline status, hints, replay frames) has to be a function
of what the human seat can see under the game's visibility schema. The
frontend hiding a value is not enough: the value must not be in the JSON.

Each hidden-information game is played through the platform code path —
`create_session`, the `/action` route for the human, `_commit_ai_move` for
the AIs (random legal moves, so that challenges, reveals, blind reserves
and so on all occur), `GET /state`, `/pipeline`, `/replay` — for every
player count and every human seat. At every ply three independent checks
run:

  observer   A second engine session with a DIFFERENT seed is fed only the
             human seat's observation stream (initial observation + public
             snapshots), exactly what an external client would know. The
             state the platform serves from it must equal the state served
             from the real game. This catches hidden values wherever they
             live, including outside the visibility schema (a deck order).
  shuffle    Values hidden from the human are permuted inside the real
  revalue    game's state (shuffle) or replaced by other legal values
             (revalue), all served responses are recomputed, and they must
             be byte-identical. Then the real values are restored.

plus: the frames the replay panel gets during play equal the served state,
AI moves are reported through their public projection, and the options /
search statistics / hints of a player that holds hidden cards are withheld.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "platform"))

pytest.importorskip("fastapi")

import dinoboard_engine as engine                               # noqa: E402
from fastapi import HTTPException                               # noqa: E402
from game_service import pipeline, routes, sessions             # noqa: E402
from game_service.replay import make_replay_frame               # noqa: E402

MAX_PLIES = 70
MIN_POSITIONS = 24          # per (game, player count, human seat)
MAX_GAMES = 16              # short games (Love Letter, Liar's Dice) need several


# Games for which the observer check cannot be run, with the reason. The
# shuffle / revalue checks still run for them.
OBSERVER_CHECK_NOT_APPLICABLE = {
    # A card an opponent publicly took into its hand is listed by the
    # serializer (`public_hand`) from a slot of the owner-only `hand` field
    # plus the public "shown" mask; the viz of that slot is never opened to
    # the other seat, so an observer session never receives the card. Public
    # information that the observation stream does not carry — an Ark Nova
    # schema gap, not something the viewer state leaks.
    "arknova",
}


def _body(state):
    """State without the three header leaves. They are the IGameState
    accessors (public by contract; the platform also serves them at the
    top level of every response), and an observer session's own
    `current_player()` is not meaningful between its turns (6 nimmt!: a
    revealed card of an earlier round stays in the now-hidden slot)."""
    return {k: v for k, v in state.items()
            if k not in ("current_player", "is_terminal", "winner")}


def _hidden_info_bases():
    """Enabled games tagged `hidden_info` in games/manifest.json. (conftest's
    hidden_info_games() only lists ids registered verbatim, which drops games
    that register variants only, e.g. maskmen_3p / maskmen_4p.)"""
    manifest = json.loads((PROJECT_ROOT / "games" / "manifest.json").read_text(encoding="utf-8"))
    return [g["id"] for g in manifest["games"]
            if g.get("enabled", True) and "hidden_info" in g.get("capabilities", [])
            and g["id"] in sessions.GAME_CONFIGS]


def _cases():
    out = []
    for base in _hidden_info_bases():
        players = sessions.GAME_CONFIGS[base]["players"]
        for n in range(players["min"], players["max"] + 1):
            for seat in range(n):
                out.append((base, n, seat))
    return out


def _dump(obj) -> str:
    return json.dumps(obj, sort_keys=True)


def _diff(a, b, path="", out=None):
    """Paths at which two JSON values differ (for failure messages)."""
    out = [] if out is None else out
    if len(out) >= 12:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: {'missing' if k not in a else a[k]!r} / "
                           f"{'missing' if k not in b else b[k]!r}")
            else:
                _diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path}[{i}]", out)
    elif a != b or type(a) is not type(b):
        out.append(f"{path}: {a!r} / {b!r}")
    return out


def _fake_stats(n):
    return {
        "best_value": 0.25,
        "root_values": [0.25] + [-0.25] * (n - 1),
        "action_values": {"3": [0.1] * n},
        "root_actions": [3, 5],
        "root_action_visits": [10, 2],
        "tail_solved": False,
    }


@pytest.fixture(autouse=True)
def _no_background_workers(monkeypatch):
    """The test commits AI moves itself; the pipeline threads stay idle."""
    for name in ("schedule_pipeline", "schedule_pipeline_ai_only", "schedule_precompute"):
        monkeypatch.setattr(routes, name, lambda *a, **k: None)
    yield


def _observer(actual_id, truth, seat, seed):
    tw = engine.GameSession(actual_id, seed=seed + 7919, model_path="", use_filter=False)
    tw.apply_initial_observation(seat, truth.extract_initial_observation(seat))
    return tw


def _served_from(gs, sess):
    """What the platform would serve if `gs` were the session's game."""
    fake = {
        "game_session": gs,
        "human_player": sess["human_player"],
        "ai_players": sess["ai_players"],
        "ai_player": sess["ai_player"],
        "difficulty": sess["difficulty"],
        "replay_frames": [],
    }
    return sessions.session_response("observer", fake)


def _served(sid):
    return {
        "state": routes.get_game_state(sid),
        "pipeline": routes.get_pipeline_status(sid),
        "replay": routes.get_replay(sid),
    }


def _mover_holds_secrets(gs, base, mover, seat):
    """Expected opacity of the player to move, computed from the real game."""
    if mover == seat:
        return False
    if base == "splendor":
        # The only private thing a Splendor seat can hold is a face-down
        # reserved card; an empty reserve slot is no secret.
        return any(not card["visible"]
                   for card in gs.get_state_dict()["players"][mover]["reserved"])
    return gs.seat_sees_more_than(mover, seat)


def _groups(gs, seat, n):
    """Hidden slots grouped for the shuffle: a group may be permuted freely
    without touching anything public. Rank >= 2: slots sharing all but the
    last index (one owner's hand, one row). Rank <= 1: the whole field, but
    only when no seat at all sees any of its slots (a hidden deck) — a
    per-owner scalar must not move between owners."""
    hidden = [(name, tuple(idx)) for name, idx in gs._hidden_slots(seat)]
    nobody_sees = set(hidden)
    for q in range(n):
        nobody_sees &= {(name, tuple(idx)) for name, idx in gs._hidden_slots(q)}
    groups = {}
    for name, idx in hidden:
        if len(idx) >= 2:
            key = (name, idx[:-1])
        elif (name, idx) in nobody_sees:
            key = (name, ())
        else:
            key = (name, idx)
        groups.setdefault(key, []).append((name, list(idx)))
    return [(name, list(idx)) for name, idx in hidden], list(groups.values())


def _tamper(gs, seat, n, rng, mode):
    """Returns (number of slots changed, undo list)."""
    hidden, groups = _groups(gs, seat, n)
    undo = []
    changed = 0
    if mode == "shuffle":
        for slots in groups:
            # Negative values mark an empty slot — which slots are occupied
            # is public (a reserve count, an emptied exchange slot), so
            # those stay where they are.
            slots = [s for s in slots if gs._read_field(*s) >= 0]
            values = [gs._read_field(name, idx) for name, idx in slots]
            if len(set(map(repr, values))) < 2:
                continue
            perm = values[:]
            for _ in range(8):
                rng.shuffle(perm)
                if perm != values:
                    break
            for (name, idx), old, new in zip(slots, values, perm):
                if new != old:
                    gs._force_write_field(name, idx, new)
                    undo.append((name, idx, old))
                    changed += 1
    else:
        pool = {}
        for name, idx in hidden:
            v = gs._read_field(name, idx)
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                continue
            pool.setdefault(name, []).append((idx, v))
        for name, entries in pool.items():
            values = sorted({v for _, v in entries})
            for idx, v in entries:
                others = [x for x in values if x != v] or [v + 1]
                new = rng.choice(others)
                gs._force_write_field(name, idx, new)
                undo.append((name, idx, v))
                changed += 1
    return changed, undo


def _play(base, n, seat, seed, monkeypatch, counters):
    rng = random.Random(seed * 1009 + n * 31 + seat)
    sid, sess = sessions.create_session(
        game_id=base, seed=seed, human_player=seat, num_players=n, difficulty="heuristic")
    sess["replay_frames"].append(make_replay_frame(sess, 0, "start", None, None))
    gs = sess["game_session"]
    actual_id = sess["actual_id"]
    # A second copy of the real game: source of the observation stream and
    # of the expected public projection of every move.
    truth = engine.GameSession(actual_id, seed=seed, model_path="", use_filter=False)
    observer = _observer(actual_id, truth, seat, seed)
    where = f"[{actual_id} human={seat} seed={seed}]"

    for ply in range(MAX_PLIES):
        if gs.is_terminal:
            break
        served = _served(sid)
        state = served["state"]
        before = _dump(served)
        cp = gs.current_player
        at = f"{where} ply {ply} (player {cp} to move)"

        # ---- observer: same seat, other seed, public stream only ----
        if base not in OBSERVER_CHECK_NOT_APPLICABLE:
            seen = _served_from(observer, sess)
            real_state, seen_state = _body(state["state"]), _body(seen["state"])
            assert _dump(seen_state) == _dump(real_state), (
                f"{at}: the served state differs from what an observer of seat {seat} "
                f"(different seed, public observations only) is served — it contains "
                f"information the seat does not have. real / observer:\n  "
                + "\n  ".join(_diff(real_state, seen_state)))
            if not state["legal_actions_hidden"]:
                assert sorted(observer.get_legal_actions()) == sorted(state["legal_actions"]), (
                    f"{at}: legal actions were served although they depend on hidden values")
        counters["observer"] += 1

        # ---- what is served about the player to move ----
        opaque = _mover_holds_secrets(gs, base, cp, seat)
        assert state["legal_actions_hidden"] == opaque, (
            f"{at}: legal_actions_hidden={state['legal_actions_hidden']}, expected {opaque}")
        if opaque:
            assert state["legal_actions"] == []
            with pytest.raises(HTTPException):
                routes.ai_hint(sid)
            with pytest.raises(HTTPException):
                routes.apply_action(sid, routes.ActionRequest(action_id=gs.get_legal_actions()[0]))
            counters["opaque_turns"] += 1
        else:
            assert sorted(state["legal_actions"]) == sorted(gs.get_legal_actions())
        assert served["replay"]["seed"] is None, f"{at}: the seed is served during the game"
        assert _dump(served["replay"]["frames"][-1]["state"]) == _dump(state["state"]), (
            f"{at}: the last replay frame is not the served state")

        # ---- tamper: change what the seat cannot see, serve again ----
        truth_before = _dump(gs.get_state_dict())
        for mode in ("shuffle", "revalue"):
            changed, undo = _tamper(gs, seat, n, rng, mode)
            try:
                if changed:
                    after = _dump(_served(sid))
                    assert after == before, (
                        f"{at}: served responses changed after {changed} values hidden "
                        f"from seat {seat} were changed ({mode}). before / after:\n  "
                        + "\n  ".join(_diff(json.loads(before), json.loads(after))))
                    counters[mode] += changed
            finally:
                for name, idx, old in reversed(undo):
                    gs._force_write_field(name, idx, old)
            assert _dump(gs.get_state_dict()) == truth_before, "tamper was not undone"

        # ---- next move, through the platform ----
        legal = gs.get_legal_actions()
        action = rng.choice(legal)
        trace = truth.apply_action_with_trace(action, seat)
        observer.apply_observation(trace["action"], trace["events"], trace["public_snapshot"])
        if cp == seat:
            resp = routes.apply_action(sid, routes.ActionRequest(action_id=action))
            assert resp["events"] == trace["events"]
            if not gs.is_terminal:
                assert _dump(resp["state"]) == _dump(routes.get_game_state(sid)["state"])
        else:
            move = {"action": action, "action_info": gs.get_action_info(action),
                    "stats": _fake_stats(n)}
            pipeline._commit_ai_move(sess, sess["pipeline"], move, cp, None)
            sess["pipeline"]["phase"] = "done"
            pipe = routes.get_pipeline_status(sid)
            assert pipe["ai_action"] == trace["action"], f"{at}: AI action not projected"
            assert pipe["ai_action_info"] == truth.get_action_info(trace["action"])
            assert pipe["ai_events"] == trace["events"]
            if opaque:
                assert pipe["ai_stats"] is None, (
                    f"{at}: search statistics of a player holding hidden cards were served")
                counters["stats_withheld"] += 1
            frame = sess["replay_frames"][-1]
            assert frame["action_id"] == trace["action"] and frame["ai_stats"] == pipe["ai_stats"]
        assert _dump(gs.get_state_dict()) == _dump(truth.get_state_dict())

    if gs.is_terminal:
        # Game over: the replay may show everything, and names the seed.
        replay = routes.get_replay(sid)
        assert replay["seed"] == seed
        assert len(replay["frames"]) == len(sess["action_history"]) + 1
        assert _dump(replay["frames"][-1]["state"]) == _dump(gs.get_state_dict())
        assert [f["action_id"] for f in replay["frames"][1:]] == sess["action_history"]
        counters["finished"] += 1
    sessions.sessions.pop(sid, None)


@pytest.mark.parametrize("base,n,seat", _cases(),
                         ids=lambda v: str(v))
def test_no_response_depends_on_values_hidden_from_the_human(base, n, seat, monkeypatch):
    counters = {"observer": 0, "shuffle": 0, "revalue": 0, "opaque_turns": 0,
                "stats_withheld": 0, "finished": 0}
    for seed in range(3, 3 + MAX_GAMES):
        _play(base, n, seat, seed, monkeypatch, counters)
        if seed >= 4 and counters["observer"] >= MIN_POSITIONS:
            break
    assert counters["observer"] >= MIN_POSITIONS, f"too few positions checked: {counters}"
    assert counters["shuffle"] + counters["revalue"] > 0, f"nothing was tampered: {counters}"


def test_hidden_info_games_are_all_covered():
    """The matrix above is manifest-driven; this pins what it must contain."""
    covered = {base for base, _, _ in _cases()}
    assert {"splendor", "loveletter", "coup", "6nimmit", "liarsdice", "maskmen",
            "jaipur", "harmonies"} <= covered


def test_the_browser_cannot_choose_the_seed(monkeypatch):
    """A client that knew the seed could rebuild the whole game (deck order
    included) through POST /api/replay/build. The seed it proposes is mixed
    with a per-process secret, and GET /replay names it only after the end."""
    monkeypatch.delenv("DINOBOARD_WEB_CLIENT_SEED", raising=False)
    resp = routes.create_game(routes.CreateGameRequest(
        game_id="loveletter", seed=12345, human_player=0, num_players=2, difficulty="heuristic"))
    sess = sessions.get_session(resp["session_id"])
    assert sess["seed"] != 12345
    assert sess["seed"] == routes._session_seed(12345)          # stable within the process
    assert routes._session_seed(12345) != routes._session_seed(12346)
    assert 0 <= sess["seed"] < 2 ** 53                           # exact in JSON / JS
    assert routes.get_replay(resp["session_id"])["seed"] is None
    # What a client would get from replaying its own seed is a different deal.
    served_hand = resp["state"]["players"][0]["hand"]
    hands = set()
    for s in range(12340, 12350):
        built = routes.build_replay(routes.BuildReplayRequest(
            game_id="loveletter", seed=s, action_history=[]))
        hands.add(built["frames"][0]["state"]["players"][1]["hand"])
    assert served_hand in range(1, 9) and len(hands) > 1
    sessions.sessions.pop(resp["session_id"], None)

    monkeypatch.setenv("DINOBOARD_WEB_CLIENT_SEED", "1")         # local debugging switch
    assert routes._session_seed(12345) == 12345


def test_test_only_keys_never_reach_the_viewer_state():
    gs = engine.GameSession("splendor", seed=5, model_path="", use_filter=False)
    truth = gs.get_state_dict()
    assert "_test_all_deck_ids" in truth                         # tests keep the truth dump
    shown = gs.get_public_state_dict(0)
    assert not [k for k in shown if k.startswith("_")]
    assert set(shown) == {k for k in truth if not k.startswith("_")}
