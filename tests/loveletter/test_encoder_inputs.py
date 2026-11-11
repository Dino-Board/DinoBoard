"""Love Letter encoder: the two inputs that used to be dead.

  * deck size (index 13*N) was the constant -64: the encoder summed the
    hidden per-type deck counts, which are placeholders on the masked state;
  * "my own hand has been seen" (index 3) was always 0 at a search root:
    the seat's session never learned that an opponent had peeked.

Expected values come from the real game replayed alongside (state dict),
not from the encoder. Feature dims must stay 70 / 99 / 128 — shipped
networks depend on the layout.
"""
from __future__ import annotations

import pytest

import dinoboard_engine as engine

PER_PLAYER_PUBLIC = 13
OWN_HAND_SEEN = 3
PRIEST = 28                      # 28 + target seat
DIMS = {"loveletter": 70, "loveletter_3p": 99, "loveletter_4p": 128}


def _episodes(game_id, seeds):
    """Yield (truth session positioned before the sample's action, sample)
    for heuristic episodes; features come from the acting seat's session."""
    for seed in seeds:
        ep = engine.run_heuristic_episode(
            game_id, seed=seed, temperature=1.0, max_game_plies=40,
            heuristic_selector="rule_v1")
        gs = engine.GameSession(game_id, seed=seed, model_path="", use_filter=False)
        prev = None
        for sample in ep["samples"]:
            yield gs, sample, prev
            prev = sample
            gs.apply_action(sample["action_id"])


@pytest.mark.parametrize("game_id", sorted(DIMS))
def test_feature_dim_unchanged(game_id):
    assert engine.game_metadata(game_id)["feature_dim"] == DIMS[game_id]
    assert len(engine.encode_state(game_id, seed=5)["features"]) == DIMS[game_id]


@pytest.mark.parametrize("game_id", sorted(DIMS))
def test_deck_size_feature_tracks_the_public_deck_size(game_id):
    n = engine.game_metadata(game_id)["num_players"]
    idx = PER_PLAYER_PUBLIC * n
    seen = set()
    for gs, sample, _ in _episodes(game_id, range(60)):
        deck_size = gs.get_state_dict()["deck_size"]
        seen.add(deck_size)
        assert sample["features"][idx] == pytest.approx(deck_size / 16.0), (
            f"[{game_id}] ply {sample['ply']}: deck holds {deck_size} cards, "
            f"feature says {sample['features'][idx]}")
    assert len(seen) >= 5, "deck size did not vary across the sampled plies"


def test_own_hand_seen_feature_is_set_after_an_opponent_peek_2p():
    """2p: right after the opponent plays Priest on me, I act with the very
    card it saw, so the flag must be 1."""
    peeks = 0
    for _, sample, prev in _episodes("loveletter", range(400)):
        me = sample["player"]
        if prev is not None and prev["player"] != me and prev["action_id"] == PRIEST + me:
            peeks += 1
            flag = sample["features"][OWN_HAND_SEEN]
            assert flag == 1.0, (
                f"ply {sample['ply']}: the opponent just played Priest on seat {me}, "
                f"but its own-hand-seen input is {flag}")
    assert peeks > 50


def test_own_hand_seen_feature_matches_truth_2p():
    """2p: the flag is 1 exactly when truth says the opponent still sees my
    hand (it goes back to 0 once the seen card has left the hand)."""
    flagged = total = 0
    for gs, sample, _ in _episodes("loveletter", range(400)):
        me = sample["player"]
        flag = sample["features"][OWN_HAND_SEEN]
        seen_by_other = any(v != me for v in gs.get_state_dict()["players"][me]["hand_seen_by"])
        assert flag == (1.0 if seen_by_other else 0.0)
        total += 1
        flagged += int(flag)
    assert 0 < flagged < total


@pytest.mark.parametrize("game_id", ["loveletter_3p", "loveletter_4p"])
def test_hand_seen_flags_with_three_plus_players(game_id):
    """3+ players: who has seen whose hand is not public (it would reveal
    whether a peeked card was the one just played), so a seat only knows
    its own column: the flag for another seat means "I have seen its hand",
    and the flag for my own hand stays 0."""
    n = engine.game_metadata(game_id)["num_players"]
    i_saw = 0
    for gs, sample, _ in _episodes(game_id, range(150)):
        me = sample["player"]
        players = gs.get_state_dict()["players"]
        assert sample["features"][OWN_HAND_SEEN] == 0.0
        for k in range(1, n):
            pid = (me + k) % n
            expect = 1.0 if (players[pid]["alive"] and me in players[pid]["hand_seen_by"]) else 0.0
            i_saw += int(expect)
            assert sample["features"][PER_PLAYER_PUBLIC * k + OWN_HAND_SEEN] == expect
    assert i_saw > 20
