"""6nimmit: the rules of a game (one hand), for two, three and four players.

* At every ply of whole games the engine's state equals a model of the game
  kept from the deal and the actions alone (nimmt_game_model.py): rows,
  bullheads, the cards shown, round, hands, whose turn, legal actions, end,
  winner and losers.
* The game ends with the tenth round; whoever took the fewest bullheads wins,
  the most lose.
* The other players' hands a search works with are drawn only from cards
  that have not been shown and are not the observer's own, and drawing them
  changes nothing the observer can see.
* The heuristics play whole games, and rule_v1 beats random play.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

import dinoboard_engine as engine

sys.path.insert(0, str(Path(__file__).parent))
import nimmt_game_model as gm  # noqa: E402

GAMES = list(gm.GAMES)


def _call(value):
    return value() if callable(value) else value


def _compare(d: dict, model: gm.Game, legal: list):
    n = model.n
    assert d["is_terminal"] == model.terminal and d["num_players"] == n
    assert [[c for c in row if c] for row in d["table"]] == model.table
    assert set(d["used"]) == model.gone
    assert [p["hand_heads"] for p in d["players"]] == model.heads
    assert d["current_round"] == model.round
    assert [p["hand"] for p in d["players"]] == model.hands
    if model.terminal:
        assert d["winner"] == model.winner and legal == [] and d["current_round"] == gm.HAND_SIZE
        return
    assert d["winner"] == -1
    assert d["small_phase_active"] == (model.chooser >= 0)
    assert d["small_player"] == model.chooser
    assert d["current_player"] == model.actor()
    assert sorted(legal) == sorted(model.legal())
    # What the table shows of the played cards: nothing while one is face down.
    assert [p["pending_card"] for p in d["players"]] == [model.face_up.get(p, 0) for p in range(n)]
    assert [p["committed"] for p in d["players"]] == [
        p in model.face_down or p in model.face_up or model.chooser >= 0 for p in range(n)]
    assert [p["hand_count"] for p in d["players"]] == [len(h) for h in model.hands]
    queue = sorted((c, p) for p, c in model.face_up.items())
    assert [c for c in d["apply_queue_card"] if c] == [c for c, _ in queue]


POLICIES = {"random": gm.random_policy, "rule_v1": gm.heuristic_policy("rule_v1"),
            "rule_v1_noisy": gm.heuristic_policy("rule_v1", epsilon=0.2)}


@pytest.mark.parametrize("game", GAMES)
@pytest.mark.parametrize("policy,seeds", [("random", range(60)), ("rule_v1", range(100, 160)),
                                          ("rule_v1_noisy", range(200, 260))])
def test_engine_follows_the_game_model_at_every_ply(game, policy, seeds):
    seen = dict(plies=0, games=0, choosing=0, takes=0, sole_winner=0, shared_result=0)
    for seed in seeds:
        heads_before = None
        for gs, model, d in gm.drive(engine, game, seed, POLICIES[policy]):
            _compare(d, model, gs.get_legal_actions())
            seen["plies"] += 1
            seen["choosing"] += model.chooser >= 0
            seen["takes"] += heads_before is not None and model.heads != heads_before
            heads_before = list(model.heads)
        assert model.terminal and model.round == gm.HAND_SIZE
        assert _call(gs.winner) == model.winner
        assert sorted(_call(gs.losers)) == model.losers and sorted(_call(gs.winners)) == model.winners
        seen["games"] += 1
        seen["sole_winner"] += model.winner >= 0
        seen["shared_result"] += not model.losers
    assert seen["plies"] > 1000, seen
    del seen["shared_result"]  # all level is rare
    assert all(v > 5 for v in seen.values()), seen


@pytest.mark.parametrize("game", GAMES)
def test_every_card_is_in_exactly_one_place(game):
    """Hands, rows, face-down cards and the shown cards never overlap, and all dealt cards are
    accounted for at every ply."""
    deal = gm.deal_size(gm.GAMES[game])
    for seed in range(10):
        for gs, model, d in gm.drive(engine, game, seed, gm.random_policy):
            on_table = [c for row in d["table"] for c in row if c]
            in_hands = [c for p in d["players"] for c in p["hand"]]
            assert len(set(on_table + in_hands)) == len(on_table) + len(in_hands)
            assert set(on_table) <= set(d["used"]) and not set(in_hands) & set(d["used"])
            assert len(d["used"]) + len(in_hands) + len(model.face_down) == deal


@pytest.mark.parametrize("game", GAMES)
@pytest.mark.parametrize("plies", [0, 1, 7, 13, 23, 31, 41])
def test_other_hands_are_sampled_only_from_unseen_cards(game, plies):
    n = gm.GAMES[game]
    covered, checked = set(), 0
    for seed in range(10):
        out = engine.test_belief_tracker(game_id=game, seed=seed, plies=plies, randomize_trials=30)
        truth = out["original_state"]
        if truth["is_terminal"]:
            continue
        me = truth["players"][0]
        # (My own face-down card, when I have played and others have not, is not in this dict;
        # test_sampling_never_returns_my_face_down_card covers it.)
        unseen = set(range(1, 105)) - set(truth["used"]) - set(me["hand"])
        face_down = not truth["small_phase_active"]  # a played card of this round is still face down
        for trial, h in zip(out["trial_states"], out["trial_hashes"]):
            assert h == out["original_hash"], "sampling changed something the observer can see"
            assert trial["players"][0]["hand"] == me["hand"] and trial["used"] == truth["used"]
            assert trial["table"] == truth["table"]
            everything = []
            for p in range(1, n):
                sampled = trial["players"][p]["hand"]
                assert set(sampled) <= unseen, f"sampled shown or own cards: {sorted(set(sampled) - unseen)}"
                # A face-down card is played again by the search, out of the sampled hand.
                held = truth["players"][p]["hand_count"] + (truth["players"][p]["committed"] and face_down)
                assert len(sampled) == held
                everything += sampled
                covered |= set(sampled)
            assert len(everything) == len(set(everything)), "one card in two hands"
            checked += 1
    if plies <= 13:
        assert checked and len(covered) > 50, "sampling should reach most of the deck"


@pytest.mark.parametrize("game", GAMES)
def test_sampling_never_returns_my_face_down_card(game):
    """After I play a card face down it is in no hand and not shown yet; it must not be dealt to
    another player."""
    checked = 0
    for seed in range(30):
        dealt = engine.GameSession(game, seed=seed).get_state_dict()["players"][0]["hand"]
        out = engine.test_belief_tracker(game_id=game, seed=seed, plies=1, randomize_trials=30)
        truth = out["original_state"]
        (played,) = set(dealt) - set(truth["players"][0]["hand"])
        assert truth["current_player"] == 1 and played not in truth["used"]
        for trial in out["trial_states"]:
            assert all(played not in p["hand"] for p in trial["players"][1:])
            checked += 1
    assert checked == 900


@pytest.mark.parametrize("game", GAMES)
@pytest.mark.parametrize("name", ["rule_v1", "lkf_v1"])
def test_heuristics_play_a_whole_game(game, name):
    for seed in range(3):
        plies = 0
        for gs, model, d in gm.drive(engine, game, seed, gm.heuristic_policy(name)):
            plies += 1
        assert model.terminal and plies > gm.HAND_SIZE * gm.GAMES[game]


@pytest.mark.parametrize("game", GAMES)
def test_rule_heuristic_beats_random_play(game):
    """rule_v1 is a real opponent for the warm start: against random cards it takes clearly
    fewer bullheads."""
    n = gm.GAMES[game]
    games, lost, margin = 120, 0, 0.0
    for seed in range(games):
        seat = seed % n

        def policy(gs, d, legal, rng, seat=seat):
            if d["current_player"] == seat:
                return gs.get_heuristic_action("rule_v1", 0.0)["action"]
            return rng.choice(legal)

        for gs, model, d in gm.drive(engine, game, seed, policy):
            pass
        lost += seat in model.losers
        margin += model.values()[seat] * gm.MARGIN_SCALE
    # Chance level: most bullheads in 1/n of the games, margin 0.
    assert lost <= games / n / 2, f"rule_v1 took the most bullheads in {lost}/{games}"
    assert margin / games > 4, f"rule_v1's average margin {margin / games:.1f} bullheads"


def test_rule_heuristic_takes_rows_knowing_the_other_card():
    """Two players: both cards are face up when a row has to be taken, so rule_v1 weighs each
    row against where the other card then lands (the bullheads it hands the opponent less the
    row's own), not against a guess at the other card."""
    def heads(row):
        return sum(gm.heads_of(c) for c in row)

    checked = 0
    for seed in range(80):
        for gs, model, d in gm.drive(engine, "6nimmit", seed, gm.heuristic_policy("rule_v1", 0.2)):
            if gs.is_terminal or not d["small_phase_active"]:
                continue
            me = d["small_player"]
            mine, other = (d["players"][p]["pending_card"] for p in (me, 1 - me))
            rows = [[c for c in row if c] for row in d["table"]]

            def score(r):
                after = [list(row) for row in rows]
                after[r] = [mine]
                below = [(row[-1], i) for i, row in enumerate(after) if row[-1] < other]
                lands = after[max(below)[1]] if below else []
                return (heads(lands) if len(lands) == gm.ROW_CAPACITY else 0) - heads(rows[r])

            chosen = gs.get_heuristic_action("rule_v1", 0.0)["action"] - 104
            best = max(score(r) for r in range(len(rows)))
            assert score(chosen) == best, (seed, rows, mine, other, chosen)
            checked += 1
    assert checked >= 50
