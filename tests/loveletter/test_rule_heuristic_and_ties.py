"""Love Letter: rule_v1 target decoding, and who wins a showdown tie.

rule_v1 read the target encoded in an action id as an offset from the
actor, while the rules encode the absolute seat. Every seat but the last
therefore scored "Prince on myself" as an attack on an opponent (seat 0
would discard its own Princess) and scored real attacks as illegal.

Showdown tie (deck empty, same highest card, same discard total):
`winner()` is -1, but only the tied players share the win. winners() must
list exactly them and losers() everyone else, so a player with a plainly
lower card — or one eliminated earlier — does not get a draw.
"""
from __future__ import annotations

import collections
import random

import pytest

import dinoboard_engine as engine

PRINCE = 37          # 37 + target seat
NAMES = ["", "Guard", "Priest", "Baron", "Handmaid", "Prince", "King", "Countess", "Princess"]


def _target_of(action):
    """Absolute target seat of a targeted action, per the action layout in
    loveletter_state.h (independent of the heuristic)."""
    if action < 28:
        return action // 7
    for base in (28, 32, 37, 41):
        if base <= action < base + 4:
            return action - base
    return None


@pytest.mark.parametrize("game_id", ["loveletter", "loveletter_3p", "loveletter_4p"])
def test_heuristic_never_princes_itself_when_an_opponent_can_be_hit(game_id):
    n = engine.game_metadata(game_id)["num_players"]
    decisions = 0
    for seed in range(400):
        gs = engine.GameSession(game_id, seed=seed, model_path="", use_filter=False)
        plies = 0
        while not gs.is_terminal and plies < 40:
            me = gs.current_player
            legal = gs.get_legal_actions()
            action = gs.get_heuristic_action("rule_v1", 0.0)["action"]
            opp_prince = [a for a in legal if PRINCE <= a < PRINCE + 4 and a != PRINCE + me]
            if opp_prince:
                decisions += 1
                assert action != PRINCE + me, (
                    f"[{game_id}] seed={seed}: seat {me} plays Prince on itself although "
                    f"{[a - PRINCE for a in opp_prince]} could be targeted (legal={legal})")
            gs.apply_action(action)
            plies += 1
    assert decisions > 100


def test_heuristic_attacks_from_every_seat():
    """With a Priest in hand and an unprotected opponent, peeking (2.5) is
    the top-scored option unless a Handmaid/Prince option ties or beats it;
    at minimum the chosen action must never be a self-targeted no-op while
    a real target exists — for every seat, not only the last one."""
    self_noop = collections.Counter()
    total = collections.Counter()
    for seed in range(300):
        gs = engine.GameSession("loveletter_3p", seed=seed, model_path="", use_filter=False)
        plies = 0
        while not gs.is_terminal and plies < 40:
            me = gs.current_player
            legal = gs.get_legal_actions()
            action = gs.get_heuristic_action("rule_v1", 0.0)["action"]
            targets = {_target_of(a) for a in legal} - {None}
            if any(t != me for t in targets):
                total[me] += 1
                if _target_of(action) == me:
                    self_noop[me] += 1
            gs.apply_action(action)
            plies += 1
    for seat in range(3):
        assert total[seat] > 100
        assert self_noop[seat] == 0, f"seat {seat} self-targeted {self_noop[seat]}/{total[seat]} times"


def test_mirror_match_is_seat_symmetric_2p():
    wins = collections.Counter()
    games = 1200
    for seed in range(games):
        ep = engine.run_heuristic_episode(
            "loveletter", seed=seed, temperature=0.0, max_game_plies=40,
            heuristic_selector="rule_v1")
        wins[ep["winner"]] += 1
    share = wins[0] / (wins[0] + wins[1])
    # Fair coin over ~1200 games: sd ~ 0.0145. The old decoding gave ~0.29.
    assert 0.44 < share < 0.56, f"seat 0 wins {share:.3f} of rule_v1 mirror games ({dict(wins)})"


# ---------------------------------------------------------------------------
# Showdown ties
# ---------------------------------------------------------------------------

def _expected_result(sd):
    """Winners per the rulebook, recomputed from the final state dict."""
    alive = [p for p, pl in enumerate(sd["players"]) if pl["alive"]]
    if len(alive) == 1:
        return alive
    key = {p: (sd["players"][p]["hand"], sum(sd["players"][p]["discards"])) for p in alive}
    best = max(key.values())
    return [p for p in alive if key[p] == best]


@pytest.mark.parametrize("game_id", ["loveletter", "loveletter_3p", "loveletter_4p"])
def test_winners_and_losers_match_the_rulebook(game_id):
    n = engine.game_metadata(game_id)["num_players"]
    showdowns = 0
    for seed in range(500):
        gs = engine.GameSession(game_id, seed=seed, model_path="", use_filter=False)
        rng = random.Random(seed)
        while not gs.is_terminal:
            gs.apply_action(rng.choice(gs.get_legal_actions()))
        sd = gs.get_state_dict()
        expected = _expected_result(sd)
        showdowns += sum(pl["alive"] for pl in sd["players"]) > 1
        if len(expected) == n:
            # Everyone tied: a draw, all values 0.
            assert gs.winner == -1 and sorted(gs.winners) == expected and gs.losers == []
            continue
        assert sorted(gs.winners) == expected
        assert sorted(gs.losers) == [p for p in range(n) if p not in expected]
        assert gs.winner == (expected[0] if len(expected) == 1 else -1)
    assert showdowns > 50


def _force_showdown(game_id, hands, discards, alive):
    """Build a finished round by writing the public result fields directly."""
    n = len(hands)
    gs = engine.GameSession(game_id, seed=1, model_path="", use_filter=False)
    for p in range(n):
        gs._force_write_field("alive", [p], 1 if alive[p] else 0)
        gs._force_write_field("hand", [p], hands[p] if alive[p] else 0)
        for c in range(1, 9):
            gs._force_write_field("discard_count", [p, c], discards[p].count(c))
    gs._force_write_field("winner", [], -1)
    gs._force_write_field("terminal", [], True)
    return gs


def test_partial_tie_pays_only_the_tied_players():
    # Seats 0 and 1 both hold a Baron with discards summing to 5; seat 2 holds a Guard.
    gs = _force_showdown("loveletter_3p", hands=[3, 3, 1],
                         discards=[[1, 4], [5], [2, 2]], alive=[True, True, True])
    assert gs.is_terminal and gs.winner == -1
    assert sorted(gs.winners) == [0, 1]
    assert gs.losers == [2]

    # An eliminated seat is a loser too, whatever it discarded.
    gs = _force_showdown("loveletter_4p", hands=[6, 0, 6, 2],
                         discards=[[1], [8, 8], [1], [1]], alive=[True, False, True, True])
    assert sorted(gs.winners) == [0, 2]
    assert sorted(gs.losers) == [1, 3]


def test_full_tie_is_a_draw():
    gs = _force_showdown("loveletter", hands=[4, 4], discards=[[1, 2], [3]], alive=[True, True])
    assert gs.winner == -1 and sorted(gs.winners) == [0, 1] and gs.losers == []
    gs = _force_showdown("loveletter_3p", hands=[2, 2, 2],
                         discards=[[1], [1], [1]], alive=[True, True, True])
    assert sorted(gs.winners) == [0, 1, 2] and gs.losers == []


def test_tie_break_by_discard_total_still_picks_one_winner():
    gs = _force_showdown("loveletter_3p", hands=[3, 3, 1],
                         discards=[[1, 4], [5, 1], [2]], alive=[True, True, True])
    # Not a tie at all: the rules would have set winner=1. With winner forced
    # to -1 the tied-for-first set is still just seat 1.
    assert gs.winners == [1] and sorted(gs.losers) == [0, 2]


def test_showdown_hands_are_face_up_for_every_seat():
    """winners() on a tie reads the hands; at a showdown they are revealed
    ("all players still in the round reveal their hands"), so any seat's
    view can evaluate it."""
    seen = 0
    for seed in range(300):
        gs = engine.GameSession("loveletter_3p", seed=seed, model_path="", use_filter=False)
        rng = random.Random(seed)
        while not gs.is_terminal:
            gs.apply_action(rng.choice(gs.get_legal_actions()))
        sd = gs.get_state_dict()
        alive = [p for p, pl in enumerate(sd["players"]) if pl["alive"]]
        if len(alive) < 2:
            continue
        seen += 1
        for p in alive:
            assert sd["players"][p]["hand_seen_by"] == [0, 1, 2]
    assert seen > 20
