"""Coup Exchange: hidden-information regressions.

1. ExchangeReturn2 on a seat's session state. After the first return the
   emptied drawn slot is invisible to the exchanging player, so the session
   keeps the returned character there (snapshots only write visible slots).
   Rules used to read that stale value: on the session — i.e. inside the
   seat's own search — the second return produced a different hand than on
   the real state. Every option must now lead to the same hand on both.

2. Which character is returned is private: other seats must receive the
   stand-in action id, never the character.

3. Court-deck knowledge. A seat knows the cards it returned are in the
   deck (until someone draws unseen); a card shown to win a challenge is
   known to be in the deck or back in the same slot. Sampled worlds must
   honor this, checked here against an independent reconstruction from the
   seat's own before/after hands.
"""
from __future__ import annotations

import collections
import random

import pytest

import dinoboard_engine as engine
from conftest import get_test_model

GAME = "coup"
EXCHANGE, CHALLENGE, ALLOW = 15, 16, 17
RETURN_BASE = 27
HIDDEN_RETURN = 32
STAGE_RETURN1, STAGE_RETURN2 = 8, 9


def _pair(seed, seat=0, game=GAME, model=""):
    gt = engine.GameSession(game, seed=seed, model_path="", use_filter=False)
    ai = engine.GameSession(game, seed=seed + 99991, model_path=model, use_filter=False)
    ai.apply_initial_observation(seat, gt.extract_initial_observation(seat))
    return gt, ai


def _step(gt, ai, action, seat=0):
    trace = gt.apply_action_with_trace(action, seat)
    ai.apply_observation(trace["action"], trace["events"], trace["public_snapshot"])
    return trace


def _own_cards(session, seat=0):
    return sorted(i["character"] for i in session.get_state_dict()["players"][seat]["influences"])


def _to_return2(seed, rng):
    """Seat 0 exchanges (unchallenged) and makes a random first return.
    Returns (gt, ai, first_return) at the ExchangeReturn2 decision, or None."""
    gt, ai = _pair(seed)
    _step(gt, ai, EXCHANGE)
    _step(gt, ai, ALLOW)
    if gt.get_state_dict()["stage"] != STAGE_RETURN1:
        return None
    first = rng.choice(gt.get_legal_actions())
    _step(gt, ai, first)
    assert gt.get_state_dict()["stage"] == STAGE_RETURN2
    return gt, ai, first


def test_second_return_gives_same_hand_on_session_as_on_truth():
    options = 0
    stale_roots = 0
    for seed in range(250):
        rng = random.Random(seed)
        got = _to_return2(seed, rng)
        if got is None:
            continue
        gt, ai, first = got
        legal = gt.get_legal_actions()
        assert sorted(ai.get_legal_actions()) == sorted(legal)
        # The scenario is only meaningful if the session really holds a stale card.
        truth_drawn = gt.get_state_dict()["exchange_drawn"]
        session_drawn = ai.get_state_dict()["exchange_drawn"]
        if truth_drawn != session_drawn:
            stale_roots += 1
        for action in legal:
            rng2 = random.Random(seed)
            gt2, ai2, first2 = _to_return2(seed, rng2)
            assert first2 == first
            gt2.apply_action(action)
            ai2.apply_action(action)   # external-obs session: rules run on the session state
            options += 1
            assert _own_cards(ai2) == _own_cards(gt2), (
                f"seed={seed}: after returning {first - RETURN_BASE} then "
                f"{action - RETURN_BASE}, the session holds {_own_cards(ai2)} "
                f"but the real hand is {_own_cards(gt2)}")
            assert ai2.get_state_dict()["deck_size"] == gt2.get_state_dict()["deck_size"]
    assert options > 400
    assert stale_roots > 50, "sessions no longer carry a stale drawn slot — scenario lost"


def test_search_at_second_return_ignores_the_stale_drawn_slot():
    """Runs MCTS from ExchangeReturn2 roots. One session keeps the stale
    character in the emptied drawn slot (what really happens), its twin has
    that slot set to the value the real state holds there (-1). Same seed,
    so the two searches may only differ if a simulation reads the stale
    slot."""
    model = get_test_model(GAME)
    compared = 0
    for seed in range(60):
        rng = random.Random(seed)
        gt = engine.GameSession(GAME, seed=seed, model_path="", use_filter=False)
        twins = []
        for _ in range(2):
            ai = engine.GameSession(GAME, seed=seed + 77, model_path=model, use_filter=False)
            ai.apply_initial_observation(0, gt.extract_initial_observation(0))
            twins.append(ai)
        for action in (EXCHANGE, ALLOW):
            trace = gt.apply_action_with_trace(action, 0)
            for ai in twins:
                ai.apply_observation(trace["action"], trace["events"], trace["public_snapshot"])
        if gt.get_state_dict()["stage"] != STAGE_RETURN1:
            continue
        drawn = gt.get_state_dict()["exchange_drawn"]
        trace = gt.apply_action_with_trace(RETURN_BASE + drawn[0], 0)   # empties drawn slot 0
        for ai in twins:
            ai.apply_observation(trace["action"], trace["events"], trace["public_snapshot"])
        stale, clean = twins
        assert stale.get_state_dict()["exchange_drawn"][0] == drawn[0]   # stale byte is there
        clean._force_write_field("exchange_drawn", [0], -1)
        results = []
        for ai in (stale, clean):
            r = ai.get_ai_action(simulations=48, temperature=0.0)
            st = r["stats"]
            results.append((r["action"], tuple(st["root_actions"]),
                            tuple(st["root_action_visits"]), tuple(st["root_values"])))
        assert results[0] == results[1], (
            f"seed={seed}: MCTS at ExchangeReturn2 depends on the stale drawn slot\n"
            f"  stale: {results[0]}\n  clean: {results[1]}")
        compared += 1
    assert compared >= 40


def test_exchange_return_is_projected_for_observers():
    for seed in range(40):
        gt = engine.GameSession(GAME, seed=seed, model_path="", use_filter=False)
        gt.apply_action(EXCHANGE)
        gt.apply_action(ALLOW)
        if gt.get_state_dict()["stage"] != STAGE_RETURN1:
            continue
        for _ in range(2):
            legal = gt.get_legal_actions()
            action = legal[seed % len(legal)]
            trace = gt.apply_action_with_trace(action, 1)   # what seat 1 receives
            assert trace["action"] == HIDDEN_RETURN
            assert HIDDEN_RETURN not in legal
            info = gt.get_action_info(trace["action"])
            assert info["type"] == "return_card" and info["hidden"] is True
            assert "character" not in info and "character_name" not in info
            for ev in trace["events"]:
                assert "role" not in ev["payload"], ev


def test_selfplay_observation_trace_never_names_a_returned_card():
    model = get_test_model(GAME)
    returns = 0
    for seed in range(8):
        ep = engine.run_selfplay_episode(
            game_id=GAME, seed=seed, model_path=model, simulations=8,
            max_game_plies=60, trace_perspective=1)
        for sample, step in zip(ep["samples"], ep["observation_trace"]):
            real = sample["action_id"]
            if RETURN_BASE <= real < RETURN_BASE + 5:
                returns += 1
                assert step["action"] == HIDDEN_RETURN
            else:
                assert step["action"] == real
    assert returns > 0, "no exchange in the sampled episodes"


# ---------------------------------------------------------------------------
# Court-deck knowledge
# ---------------------------------------------------------------------------

def _pool_holds_known(sd, p, n):
    """deck_known[p] <= court deck + contents of p's flagged slots (+ the
    cards another seat has in flight during its own exchange)."""
    have = list(sd["deck_count"])
    taint = sd["players"][p]["deck_known_taint"]
    for q in range(n):
        for sl in range(2):
            if taint[q * 2 + sl]:
                c = sd["players"][q]["influences"][sl]["character"]
                if 0 <= c < 5:
                    have[c] += 1
    if sd["stage"] in (STAGE_RETURN1, STAGE_RETURN2) and sd["active_player"] != p:
        for c in sd["exchange_drawn"]:
            if 0 <= c < 5:
                have[c] += 1
    return all(k <= h for k, h in zip(sd["players"][p]["deck_known"], have))


def _card_totals(sd, n):
    tot = list(sd["deck_count"])
    for p in range(n):
        for i in sd["players"][p]["influences"]:
            if 0 <= i["character"] < 5:
                tot[i["character"]] += 1
    return tot


@pytest.mark.parametrize("game_id", ["coup", "coup_3p", "coup_4p"])
def test_deck_knowledge_is_never_wrong_on_the_real_state(game_id):
    """Soundness: whatever a seat believes for certain about the deck is
    true, at every ply of games rich in exchanges and challenges."""
    n = engine.game_metadata(game_id)["num_players"]
    with_knowledge = with_taint = 0
    for seed in range(120):
        gs = engine.GameSession(game_id, seed=seed, model_path="", use_filter=False)
        rng = random.Random(seed * 7 + n)
        steps = 0
        while not gs.is_terminal and steps < 150:
            legal = gs.get_legal_actions()
            pref = [a for a in legal if a in (EXCHANGE, CHALLENGE) or a >= 23]
            gs.apply_action(rng.choice(pref) if pref and rng.random() < 0.6 else rng.choice(legal))
            steps += 1
            sd = gs.get_state_dict()
            for p in range(n):
                with_knowledge += any(sd["players"][p]["deck_known"])
                with_taint += any(sd["players"][p]["deck_known_taint"])
                assert _pool_holds_known(sd, p, n), (
                    f"[{game_id}] seed={seed} step={steps}: seat {p} 'knows' "
                    f"{sd['players'][p]['deck_known']} but deck={sd['deck_count']}")
            if sd["stage"] not in (STAGE_RETURN1, STAGE_RETURN2):
                assert _card_totals(sd, n) == [3] * 5
    assert with_knowledge > 500 and with_taint > 100


def test_worlds_keep_returned_cards_out_of_opponent_hands():
    """Right after seat 0 completes an Exchange nobody else has touched the
    deck: the two cards it put back are there for certain. They are
    reconstructed here from seat 0's hand before the exchange, the two
    drawn cards and its hand afterwards — independently of the engine's
    bookkeeping — and no sampled world may give the opponent more copies
    of a character than that leaves."""
    cases = worlds = 0
    checked = []
    for seed in range(6000):
        r4 = engine.test_belief_tracker(GAME, seed=seed, plies=4, randomize_trials=12)
        o4 = r4["original_state"]
        if r4["belief_snapshot"]["exchange_complete_counts"][0] != 1:
            continue
        if o4["ply"] != 4 or o4["stage"] != 0 or o4["active_player"] != 1:
            continue
        o2 = engine.test_belief_tracker(GAME, seed=seed, plies=2, randomize_trials=1)["original_state"]
        if o2["stage"] != STAGE_RETURN1:
            continue
        before = [i["character"] for i in o2["players"][0]["influences"]]
        after = [i["character"] for i in o4["players"][0]["influences"]]
        returned = collections.Counter(before + list(o2["exchange_drawn"]))
        returned.subtract(collections.Counter(after))
        assert min(returned.values()) >= 0 and sum(returned.values()) == 2
        own = collections.Counter(after)
        cases += 1
        for ts in r4["trial_states"]:
            worlds += 1
            opp = collections.Counter(i["character"] for i in ts["players"][1]["influences"])
            for c in range(5):
                assert opp[c] <= 3 - own[c] - returned[c], (
                    f"seed={seed}: seat 0 holds {after}, returned {sorted(returned.elements())}, "
                    f"yet a sampled world deals the opponent {sorted(opp.elements())}")
        checked.append((o4, returned, r4["trial_states"]))
    # The engine's bookkeeping must agree with the reconstruction, and the
    # sampled decks must actually contain the returned cards.
    for o4, returned, trial_states in checked:
        assert o4["players"][0]["deck_known"] == [returned[c] for c in range(5)]
        for ts in trial_states:
            for c in range(5):
                assert ts["deck_count"][c] >= returned[c]
            assert _card_totals(ts, 2) == [3] * 5
            assert sum(ts["deck_count"]) == ts["deck_size"] == o4["deck_size"]
    assert cases > 300 and worlds > 3000


@pytest.mark.parametrize("game_id", ["coup", "coup_3p", "coup_4p"])
def test_sampled_worlds_honor_deck_knowledge_and_conserve_cards(game_id):
    n = engine.game_metadata(game_id)["num_players"]
    positions = worlds = tainted_positions = 0
    for seed in range(700):
        for plies in (4, 9, 16, 30):
            r = engine.test_belief_tracker(game_id, seed=seed, plies=plies, randomize_trials=5)
            o = r["original_state"]
            known = o["players"][0]["deck_known"]
            if o["is_terminal"] or not any(known):
                continue
            positions += 1
            tainted_positions += any(o["players"][0]["deck_known_taint"])
            for ts in r["trial_states"]:
                worlds += 1
                assert _pool_holds_known(ts, 0, n), (
                    f"[{game_id}] seed={seed} plies={plies}: world breaks what seat 0 knows")
                # Observer's own knowledge and cards are untouched ...
                assert ts["players"][0]["deck_known"] == known
                assert ts["players"][0]["influences"] == o["players"][0]["influences"]
                # ... the other seats' (hidden) knowledge rows are cleared ...
                for q in range(1, n):
                    assert not any(ts["players"][q]["deck_known"])
                    assert not any(ts["players"][q]["deck_known_taint"])
                # ... and cards are conserved.
                assert sum(ts["deck_count"]) == ts["deck_size"] == o["deck_size"]
                if ts["stage"] not in (STAGE_RETURN1, STAGE_RETURN2):
                    assert _card_totals(ts, n) == [3] * 5
    assert positions > 200 and tainted_positions > 20
