"""6 nimmt!: every played card must be recorded for every seat, and a card
that lies face up must never be reported as "unseen".

Two defects fixed together:
  * the seat that completes a round has its card applied inside the same
    transition as its commit, so the `card_played` event (diffing the
    pending slot before/after) never fired for it — the last seat's plays
    were only recorded in rounds with a SMALL pause;
  * belief-net input block [A] ("unseen pool") subtracted own hand and
    recorded plays but not the table or the bullhead piles, so the four
    starting row heads (and every unrecorded play) stayed marked unseen.

Expected values are recomputed here from the real game (hands before /
after, table, piles), not read back from the tracker.
"""
from __future__ import annotations

import random

import pytest

import dinoboard_engine as engine
from conftest import get_test_model

NUM_CARDS = 104


def _hands(sd):
    return [set(p["hand"]) for p in sd["players"]]


@pytest.mark.parametrize("game_id", ["6nimmit", "6nimmit_3p", "6nimmit_4p"])
def test_every_commit_yields_exactly_one_card_played_event(game_id):
    n = engine.game_metadata(game_id)["num_players"]
    for seed in range(8):
        gs = engine.GameSession(game_id, seed=seed, model_path="", use_filter=False)
        rng = random.Random(seed)
        committed = {p: [] for p in range(n)}      # truth: card leaving each hand
        reported = {p: [] for p in range(n)}       # card_played events
        while not gs.is_terminal:
            before = _hands(gs.get_state_dict())
            actor = gs.current_player
            action = rng.choice(gs.get_legal_actions())
            trace = gs.apply_action_with_trace(action, 0)
            after = _hands(gs.get_state_dict())
            if action < NUM_CARDS:
                gone = before[actor] - after[actor]
                assert gone == {action + 1}
                committed[actor].append(action + 1)
            for ev in trace["events"]:
                assert ev["kind"] == "card_played"
                reported[ev["payload"]["player"]].append(ev["payload"]["card"])
        for p in range(n):
            assert len(committed[p]) == 10
            assert sorted(reported[p]) == sorted(committed[p]), (
                f"[{game_id}] seed={seed}: seat {p} played {sorted(committed[p])} "
                f"but card_played reported {sorted(reported[p])}")
        # The tracker's memory (fed by those events) is complete and symmetric.
        played_flat = gs.get_belief_snapshot()["played_flat"]
        assert sorted(played_flat) == sorted(
            p * NUM_CARDS + c for p in range(n) for c in committed[p])


def test_card_played_is_emitted_only_once_the_card_is_face_up():
    """A commit that leaves the round open produces no event (the card is
    still face down); the event names a card only when it reaches the table."""
    for seed in range(8):
        gs = engine.GameSession("6nimmit_3p", seed=seed, model_path="", use_filter=False)
        rng = random.Random(seed)
        while not gs.is_terminal:
            action = rng.choice(gs.get_legal_actions())
            trace = gs.apply_action_with_trace(action, 0)
            sd = gs.get_state_dict()
            on_table = {c for row in sd["table"] for c in row if c > 0}
            in_piles = {c for p in sd["players"] for c in p["bullhead_pile"]}
            for ev in trace["events"]:
                assert ev["payload"]["card"] in on_table | in_piles


def test_belief_features_never_mark_a_public_card_unseen():
    """Block [A] of the belief-net input, checked at every ply of selfplay
    episodes against the real game replayed from the samples' actions."""
    model = get_test_model("6nimmit")
    checked = 0
    for seed in (3, 7, 19):
        ep = engine.run_selfplay_episode(
            game_id="6nimmit", seed=seed, model_path=model, simulations=4,
            max_game_plies=40, dirichlet_epsilon=0.0)
        gs = engine.GameSession("6nimmit", seed=seed, model_path="", use_filter=False)
        by_ply = {}
        for b in ep["belief_samples"]:
            by_ply.setdefault(b["ply"], {})[b["observer"]] = b["features"]
        committed = {0: set(), 1: set()}          # cards that left each hand
        for ply, sample in enumerate(ep["samples"]):
            sd = gs.get_state_dict()
            public = {c for row in sd["table"] for c in row if c > 0}
            for p in sd["players"]:
                public |= set(p["bullhead_pile"])
            hands = _hands(sd)
            # "Played" = committed AND already face up on the table / in a pile.
            played = {p: committed[p] & public for p in (0, 1)}
            for observer in (0, 1):
                feats = by_ply[ply][observer]
                assert len(feats) == 329          # dims unchanged (325 + 2N)
                unseen = {c for c in range(1, NUM_CARDS + 1) if feats[c - 1] > 0.5}
                own_played = {c for c in range(1, NUM_CARDS + 1) if feats[NUM_CARDS + c - 1] > 0.5}
                opp_played = {c for c in range(1, NUM_CARDS + 1) if feats[2 * NUM_CARDS + c - 1] > 0.5}
                assert not (unseen & public), (
                    f"seed={seed} ply={ply} observer={observer}: public cards "
                    f"{sorted(unseen & public)} are marked unseen")
                assert not (unseen & hands[observer])
                # Everything the opponent still holds must be in the pool.
                assert hands[1 - observer] <= unseen
                # [B] / [C]: cards each side has played so far (face up).
                assert own_played == played[observer]
                assert opp_played == played[1 - observer]
                checked += 1
            if sample["action_id"] < NUM_CARDS:
                committed[sample["player"]].add(sample["action_id"] + 1)
            gs.apply_action(sample["action_id"])
    assert checked > 120
