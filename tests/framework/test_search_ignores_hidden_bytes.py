"""Search must not depend on the bytes a seat's session holds in slots the
seat cannot see.

A session only ever receives the slots visible to its seat; every other
slot keeps whatever was there before — the seed's own deal, a card that
has since been returned to a deck, or (in selfplay, where the per-seat
state starts as a clone of the real game) the TRUE hidden cards. The
framework's promise is that none of this can reach a decision: the hash
and the encoder see placeholders, and `randomize_unseen` overwrites every
hidden slot before a simulation runs rules on it.

This test checks the promise end to end, through real MCTS: two sessions
of the same seat with the same seed observe the same game; before each
decision one of them gets its hidden slots overwritten — once with the
true values of the other seats, once with random legal values. The two
searches must return bit-identical results (same seed => same search RNG,
so any difference comes from the tampered bytes).

It would have caught the Coup Exchange bug: after the first return the
emptied drawn slot still held the returned character on the session, and
the rules of the second return read it.
"""
from __future__ import annotations

import random

import pytest

import dinoboard_engine as engine
from conftest import get_test_model

VIZ_KEY = "__viz__"
ALL_VIEWERS_KEY = "__viz_all_viewers__"

# Per game: every field that is not all_public -> (data shape, low, high)
# of legal slot values. N is substituted with the player count.
HIDDEN_FIELDS = {
    "coup": {
        "influence": (("N", 2), 0, 4),
        "exchange_drawn": ((2,), -1, 4),
        "deck_count": ((5,), 0, 3),
        "deck_known": (("N", 5), 0, 2),
        "deck_known_taint": (("N", "N", 2), 0, 1),
    },
    "loveletter": {
        "hand": (("N",), 1, 8),
        "drawn_card": ((), 0, 8),
        "set_aside_card": ((), 1, 8),
        "deck_count": ((9,), 0, 2),
    },
    "6nimmit": {
        "hand": (("N", 104), 0, 1),
        "pending_commits": (("N",), 0, 104),
    },
}

GAMES = ["coup", "coup_3p", "loveletter", "loveletter_3p", "6nimmit", "6nimmit_3p"]
SIMULATIONS = 32


def _base(game_id):
    return game_id.split("_")[0]


def _unravel(flat, shape):
    idx = []
    for dim in reversed(shape):
        idx.append(flat % dim)
        flat //= dim
    return list(reversed(idx))


def _hidden_slots(snap, game_id, n):
    """(field, idx, low, high) for every slot hidden from the snapshot's seat."""
    out = []
    for name, (shape, low, high) in HIDDEN_FIELDS[_base(game_id)].items():
        if name not in snap[VIZ_KEY]:
            continue  # field unknown to this build of the game
        dims = tuple(n if d == "N" else d for d in shape)
        bits = snap[VIZ_KEY][name]
        for flat, bit in enumerate(bits):
            if not bit:
                out.append((name, _unravel(flat, dims), low, high))
    return out


def _tamper(session, gt, seat, game_id, n, rng, mode):
    """Overwrite the session's hidden slots. mode 'truth': the real values
    the other seats hold (random for slots nobody sees); mode 'garbage':
    random legal values everywhere."""
    hidden = _hidden_slots(gt.extract_initial_observation(seat)["public_snapshot"], game_id, n)
    known = {}
    if mode == "truth":
        for q in range(n):
            if q == seat:
                continue
            snap = gt.extract_initial_observation(q)["public_snapshot"]
            for name, pairs in snap.items():
                if name in (VIZ_KEY, ALL_VIEWERS_KEY):
                    continue
                for k in range(0, len(pairs), 2):
                    known[(name, tuple(pairs[k]))] = int(pairs[k + 1])
    for name, idx, low, high in hidden:
        value = known.get((name, tuple(idx)))
        if value is None:
            value = rng.randint(low, high)
        session._force_write_field(name, idx, value)
    return len(hidden)


def _search(session):
    r = session.get_ai_action(simulations=SIMULATIONS, temperature=0.0)
    st = r["stats"]
    return (r["action"], tuple(st["root_actions"]), tuple(st["root_action_visits"]),
            tuple(st["root_values"]), st["best_value"])


def _pick_gt_action(game_id, gt, seat, ai_action, rng):
    legal = gt.get_legal_actions()
    if _base(game_id) == "coup":
        if gt.current_player == seat:
            # Steer the tested seat into Exchange so both return stages
            # (incl. the second one, with a stale drawn slot) become roots.
            if 15 in legal and rng.random() < 0.7:
                return 15
        elif 17 in legal:
            return 17  # let claims through
    if gt.current_player == seat and rng.random() < 0.5:
        return ai_action
    return rng.choice(legal)


@pytest.mark.parametrize("mode", ["truth", "garbage"])
@pytest.mark.parametrize("game_id", GAMES)
def test_mcts_result_is_independent_of_hidden_session_bytes(game_id, mode):
    model = get_test_model(game_id)
    n = engine.game_metadata(game_id)["num_players"]
    roots = 0
    tampered_slots = 0
    stage9_roots = 0
    # Love Letter rounds last only a handful of plies: sample more of them.
    for seed in range(30 if _base(game_id) == "loveletter" else 6):
        rng = random.Random(seed * 977 + n)
        gt = engine.GameSession(game_id, seed=seed, model_path="", use_filter=False)
        seat = seed % n
        twins = []
        for _ in range(2):
            ai = engine.GameSession(game_id, seed=seed + 4242, model_path=model, use_filter=False)
            ai.apply_initial_observation(seat, gt.extract_initial_observation(seat))
            twins.append(ai)
        clean, dirty = twins
        steps = 0
        while not gt.is_terminal and steps < 80:
            ai_action = None
            if gt.current_player == seat:
                tampered_slots += _tamper(dirty, gt, seat, game_id, n, rng, mode)
                expected = _search(clean)
                got = _search(dirty)
                assert got == expected, (
                    f"[{game_id}/{mode}] seed={seed} step={steps}: search result "
                    f"changed when slots hidden from seat {seat} were overwritten.\n"
                    f"  clean : {expected}\n  dirty : {got}")
                assert sorted(clean.get_legal_actions()) == sorted(dirty.get_legal_actions())
                ai_action = expected[0]
                roots += 1
                if _base(game_id) == "coup" and gt.get_state_dict()["stage"] == 9:
                    stage9_roots += 1
            action = _pick_gt_action(game_id, gt, seat, ai_action, rng)
            trace = gt.apply_action_with_trace(action, seat)
            for ai in twins:
                ai.apply_observation(trace["action"], trace["events"], trace["public_snapshot"])
            steps += 1
    assert roots >= 12, f"[{game_id}] too few decisions compared ({roots})"
    assert tampered_slots > 0
    if _base(game_id) == "coup":
        assert stage9_roots >= 2, "no ExchangeReturn2 root was searched"
