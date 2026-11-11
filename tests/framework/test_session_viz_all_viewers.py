"""A seat's session must know what the OTHER seats can see.

MCTS rooted at seat p masks / hashes / encodes an opponent's decision
node with the opponent's column of the runtime viz tensor. The snapshot
wire used to carry only the receiver's own column, so in p's session the
other columns stayed at the schema base forever: inside p's search an
opponent could not see p's face-up Coup card, did not know it had peeked
p's Love Letter hand, etc.

Fields whose who-sees-what pattern is public are declared
`VizSync::kAllViewers`; their whole viz tensor rides the snapshot under
the optional `__viz_all_viewers__` key. This file pins:

  * at every search root of seat p, for every seat q, the session's
    hash-for-perspective-q equals truth's once the session's unknown slots
    hold the true values (i.e. the two states differ in nothing but values
    p cannot see) — an independent check that every viewer column matches;
  * a snapshot without the key (older sender / hand-built snapshot) still
    applies and behaves as before — and does NOT pass the check above,
    which is the negative control showing the check is not vacuous;
  * which fields are (not) synced: patterns that depend on hidden values
    must stay receiver-only.
"""
from __future__ import annotations

import random

import pytest

import dinoboard_engine as engine

ALL_VIEWERS_KEY = "__viz_all_viewers__"
VIZ_KEY = "__viz__"

# game -> fields expected under the all-viewers key (exactly).
SYNCED_FIELDS = {
    "coup": {"influence"},
    "coup_3p": {"influence"},
    "coup_4p": {"influence"},
    "loveletter": {"hand", "drawn_card"},
    # 3+ players: whether a peek survives a play is not public -> no "hand".
    "loveletter_3p": {"drawn_card"},
    "loveletter_4p": {"drawn_card"},
    "6nimmit": {"pending_commits"},
    "6nimmit_3p": {"pending_commits"},
    "6nimmit_4p": {"pending_commits"},
    # Splendor: who sees a reserve slot == its owner + the public
    # reserved_visible flag (kept in lockstep through buys / compaction).
    "splendor": {"reserved"},
    "splendor_3p": {"reserved"},
    "splendor_4p": {"reserved"},
    "splendor_tok": {"reserved"},
}

# Games whose every dynamic-viz field is synced, so the root check below
# must hold for every perspective.
FULLY_SYNCED_GAMES = ["coup", "coup_3p", "loveletter", "6nimmit", "6nimmit_3p",
                      "splendor", "splendor_3p", "splendor_4p", "splendor_tok"]


def _snapshot(gt, seat):
    return gt.extract_initial_observation(seat)["public_snapshot"]


def _new_session(game_id, gt, seat, seed, strip_key=False):
    obs = gt.extract_initial_observation(seat)
    if strip_key:
        obs["public_snapshot"].pop(ALL_VIEWERS_KEY, None)
    ai = engine.GameSession(game_id, seed=seed, model_path="", use_filter=False)
    ai.apply_initial_observation(seat, obs)
    return ai


def _write_true_values(session, gt, n):
    """Overwrite every slot some seat can see with its true value. Slots no
    seat sees never enter any perspective's hash."""
    for q in range(n):
        snap = _snapshot(gt, q)
        for name, pairs in snap.items():
            if name in (VIZ_KEY, ALL_VIEWERS_KEY):
                continue
            for k in range(0, len(pairs), 2):
                session._force_write_field(name, list(pairs[k]), pairs[k + 1])


def _pick_action(game_id, gt, rng):
    legal = gt.get_legal_actions()
    if game_id.startswith("coup"):
        # Favour challenges / reveals so cards actually turn face up.
        pref = [a for a in legal if a == 16 or a >= 23]
        if pref and rng.random() < 0.7:
            return rng.choice(pref)
    if game_id.startswith("loveletter"):
        pref = [a for a in legal if 28 <= a < 36 or 41 <= a < 45]  # Priest / Baron / King
        if pref and rng.random() < 0.8:
            return rng.choice(pref)
    if game_id.startswith("splendor"):
        pref = [a for a in legal if 12 <= a < 24]   # reserve a face-up card
        if pref and rng.random() < 0.3:
            return rng.choice(pref)
    return rng.choice(legal)


@pytest.mark.parametrize("game_id", FULLY_SYNCED_GAMES)
def test_session_matches_truth_for_every_perspective_at_roots(game_id):
    n = engine.game_metadata(game_id)["num_players"]
    roots = 0
    legacy_mismatch_roots = 0
    for seed in range(12):
        rng = random.Random(seed * 131 + n)
        gt = engine.GameSession(game_id, seed=seed, model_path="", use_filter=False)
        seat = seed % n
        ai = _new_session(game_id, gt, seat, seed + 5000)
        legacy = _new_session(game_id, gt, seat, seed + 5000, strip_key=True)
        steps = 0
        while not gt.is_terminal and steps < 120:
            if gt.current_player == seat:
                roots += 1
                _write_true_values(ai, gt, n)
                _write_true_values(legacy, gt, n)
                truth = [gt.state_hash_for_perspective(q) for q in range(n)]
                got = [ai.state_hash_for_perspective(q) for q in range(n)]
                assert got == truth, (
                    f"[{game_id}] seed={seed} step={steps}: seat {seat}'s session "
                    f"disagrees with truth about what other seats see "
                    f"(per-perspective hash match: "
                    f"{[a == b for a, b in zip(got, truth)]})")
                old = [legacy.state_hash_for_perspective(q) for q in range(n)]
                # The receiver's own column was always synced.
                assert old[seat] == truth[seat]
                if old != truth:
                    legacy_mismatch_roots += 1
            action = _pick_action(game_id, gt, rng)
            trace = gt.apply_action_with_trace(action, seat)
            snap = trace["public_snapshot"]
            ai.apply_observation(trace["action"], trace["events"], snap)
            old_snap = dict(snap)
            old_snap.pop(ALL_VIEWERS_KEY, None)
            legacy.apply_observation(trace["action"], trace["events"], old_snap)
            steps += 1
    assert roots >= 20, f"[{game_id}] too few roots exercised ({roots})"
    # Negative control: without the all-viewers payload the same check fails
    # at some roots, so the equality above is doing real work.
    assert legacy_mismatch_roots > 0, (
        f"[{game_id}] no root where the receiver-only sync was insufficient "
        f"— the scenario generator no longer exercises dynamic reveals")


@pytest.mark.parametrize("game_id", sorted(SYNCED_FIELDS))
def test_only_public_patterns_are_synced(game_id):
    gt = engine.GameSession(game_id, seed=3, model_path="", use_filter=False)
    snap = _snapshot(gt, 0)
    assert set(snap[ALL_VIEWERS_KEY]) == SYNCED_FIELDS[game_id]
    n = gt.num_players
    for name, tensor in snap[ALL_VIEWERS_KEY].items():
        own_slice = snap[VIZ_KEY][name]
        assert len(tensor) == len(own_slice) * n
        # The receiver's column of the full tensor is its `__viz__` slice.
        assert [tensor[k * n + 0] for k in range(len(own_slice))] == list(own_slice)


@pytest.mark.parametrize("game_id", ["liarsdice", "jaipur", "azul", "maskmen_3p"])
def test_games_without_synced_fields_keep_the_old_wire(game_id):
    """No kAllViewers field -> the snapshot is byte-for-byte what it was:
    hand-built snapshots of external clients keep matching it."""
    gt = engine.GameSession(game_id, seed=3, model_path="", use_filter=False)
    assert ALL_VIEWERS_KEY not in _snapshot(gt, 0)


def test_malformed_all_viewers_payload_is_rejected():
    gt = engine.GameSession("coup", seed=3, model_path="", use_filter=False)
    obs = gt.extract_initial_observation(0)
    good = obs["public_snapshot"][ALL_VIEWERS_KEY]["influence"]

    def apply(payload):
        o = gt.extract_initial_observation(0)
        o["public_snapshot"][ALL_VIEWERS_KEY] = payload
        ai = engine.GameSession("coup", seed=9, model_path="", use_filter=False)
        ai.apply_initial_observation(0, o)

    apply({"influence": list(good)})                       # sanity: accepted
    with pytest.raises(Exception, match="tensor length"):
        apply({"influence": list(good)[:-1]})
    # A field the schema keeps receiver-only must never be accepted here.
    with pytest.raises(Exception, match="kAllViewers"):
        apply({"exchange_drawn": [0, 0, 0, 0]})
    with pytest.raises(Exception, match="kAllViewers"):
        apply({"no_such_field": [0]})
