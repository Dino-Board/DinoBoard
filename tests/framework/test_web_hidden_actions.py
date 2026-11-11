"""Web layer: an AI move whose action id names something hidden must reach
the browser only as its public projection.

`GET /api/games/{id}/pipeline` and the replay frames used to echo the AI's
real action id + info. For 6 nimmt! that is the face-down card (when the AI
commits before the human), for Coup the character an Exchange returns.
The pipeline now reports IGameRules::public_action's stand-in, withholds
the AI's search statistics (they are computed from the AI's own hidden
cards), and ships the move's public events so the frontend can still
animate the reveal. The real id and the statistics stay server-side
(action_history for step-back / rebuild, `ai_stats_full` for the replay of
the finished game).

The `state` of every response is covered by test_web_no_hidden_info.py.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "platform"))

pytest.importorskip("fastapi")

from game_service import pipeline, routes                      # noqa: E402
from game_service.replay import make_replay_frame, public_ai_stats  # noqa: E402
from game_service.sessions import create_session                # noqa: E402

FAKE_STATS = {
    "best_value": 0.25,
    "root_values": [0.25, -0.25],
    "action_values": {"3": [0.1, -0.1]},
    "root_actions": [3, 5],
    "root_action_visits": [10, 2],
    "tail_solved": False,
}


def _session(game_id, human_player):
    sid, sess = create_session(game_id=game_id, seed=11, human_player=human_player,
                               num_players=2, difficulty="heuristic")
    sess["replay_frames"].append(make_replay_frame(sess, 0, "start", None, None))
    return sid, sess


def _ai_plays(sess, action):
    gs = sess["game_session"]
    actor = gs.current_player
    assert actor in sess["ai_players"]
    move = {"action": action, "action_info": gs.get_action_info(action), "stats": dict(FAKE_STATS)}
    pipeline._commit_ai_move(sess, sess["pipeline"], move, actor, None)


def _wait_idle(sess):
    deadline = time.time() + 20
    while sess["pipeline"]["phase"] in ("queued", "analyzing", "ai_thinking") and time.time() < deadline:
        time.sleep(0.02)
    pipeline.cancel_pipeline(sess)


def test_public_ai_stats_drops_only_per_action_entries():
    out = public_ai_stats(FAKE_STATS)
    assert out == {"best_value": 0.25, "root_values": [0.25, -0.25], "tail_solved": False}
    assert public_ai_stats(None) is None


def test_6nimmit_ai_commit_before_the_human_is_not_shown():
    sid, sess = _session("6nimmit", human_player=1)      # AI is seat 0: commits first
    gs = sess["game_session"]
    # Highest cards on both sides: they chain onto rows (no SMALL pause), which
    # keeps the round's resolution inside the human's own request below.
    card = max(gs.get_state_dict()["players"][0]["hand"])
    _ai_plays(sess, card - 1)

    pipe = sess["pipeline"]
    assert sess["action_history"] == [card - 1]          # real id kept server-side
    assert pipe["ai_action"] == 108
    assert pipe["ai_action_info"] == {"action_id": 108, "type": "commit", "hidden": True}
    assert pipe["ai_events"] == []
    # The AI holds a hidden hand: nothing its search computed is served.
    assert pipe["ai_stats"] is None
    assert sess["ai_stats_full"] == {1: FAKE_STATS}       # kept for the post-game replay
    # What the polling endpoint and the live replay actually return.
    status = routes.get_pipeline_status(sid)
    assert status["ai_action"] == 108
    assert status["ai_action_info"] == pipe["ai_action_info"]
    assert status["ai_stats"] is None and status["ai_events"] == []
    frame = routes.get_replay(sid)["frames"][-1]
    assert frame["action_id"] == 108
    assert frame["action_info"] == pipe["ai_action_info"]
    assert frame["ai_stats"] is None
    assert frame["events"] == []
    # The served state does not contain the AI's cards or its committed card.
    served = routes.get_game_state(sid)["state"]
    assert served["players"][0]["hand"] == [] and served["players"][0]["hand_count"] == 9
    assert served["players"][0]["committed"] is True and served["players"][0]["pending_card"] == 0
    assert str(card) not in str(served["players"][0])

    # The human completes the round: its response carries the reveal.
    my_card = max(gs.get_state_dict()["players"][1]["hand"])
    resp = routes.apply_action(sid, routes.ActionRequest(action_id=my_card - 1))
    _wait_idle(sess)
    assert resp["action_info"]["card"] == my_card
    assert resp["state"]["current_round"] == 1, "seed no longer avoids a SMALL pause"
    table_cards = {c for row in resp["state"]["table"] for c in row if c > 0}
    revealed = {ev["payload"]["player"]: ev["payload"]["card"]
                for ev in resp["events"] if ev["kind"] == "card_played"}
    assert revealed == {0: card, 1: my_card}
    assert set(revealed.values()) <= table_cards | {
        c for pl in resp["state"]["players"] for c in pl["bullhead_pile"]}
    # A rebuilt session (step-back, isolated search) replays the real ids.
    rebuilt = pipeline._build_isolated_gs({**sess, "action_history": sess["action_history"][:2]})
    assert rebuilt.get_state_dict()["table"] == resp["state"]["table"]


def test_6nimmit_round_completing_ai_commit_is_shown():
    """Human in seat 0: the AI's commit completes the round and is revealed
    at once — the browser gets the card, as before. (The web UI seats the
    human last, so there every AI commit is of the hidden kind above.)"""
    sid, sess = _session("6nimmit", human_player=0)
    gs = sess["game_session"]
    my_card = max(gs.get_state_dict()["players"][0]["hand"])
    gs.apply_action(my_card - 1)
    sess["action_history"].append(my_card - 1)
    card = max(gs.get_state_dict()["players"][1]["hand"])
    _ai_plays(sess, card - 1)
    pipe = sess["pipeline"]
    assert pipe["ai_action"] == card - 1
    assert pipe["ai_action_info"]["card"] == card
    assert pipe["ai_stats"] is None                      # searched with its hidden hand
    shown = {ev["payload"]["player"]: ev["payload"]["card"] for ev in pipe["ai_events"]}
    shown.update({p: pl["pending_card"] for p, pl in enumerate(gs.get_state_dict()["players"])
                  if pl["pending_card"] > 0})
    assert shown == {0: my_card, 1: card}


def test_coup_ai_exchange_return_is_not_shown():
    for seed_offset in range(20):
        sid, sess = create_session(game_id="coup", seed=100 + seed_offset, human_player=1,
                                   num_players=2, difficulty="heuristic")
        sess["replay_frames"].append(make_replay_frame(sess, 0, "start", None, None))
        gs = sess["game_session"]
        _ai_plays(sess, 15)                                # Exchange: a public claim
        assert sess["pipeline"]["ai_action"] == 15
        assert sess["pipeline"]["ai_stats"] is None
        gs.apply_action(17)                                # human allows
        sess["action_history"].append(17)
        if gs.get_state_dict()["stage"] != 8:
            continue
        for _ in range(2):
            real = gs.get_legal_actions()[0]
            _ai_plays(sess, real)
            pipe = sess["pipeline"]
            assert sess["action_history"][-1] == real
            assert pipe["ai_action"] == 32
            assert pipe["ai_action_info"] == {"action_id": 32, "type": "return_card", "hidden": True}
            assert pipe["ai_stats"] is None
            frame = sess["replay_frames"][-1]
            assert frame["action_id"] == 32 and "character_name" not in frame["action_info"]
        return
    pytest.fail("no seed reached the exchange return stage")
