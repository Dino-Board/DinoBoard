"""I6 over-permission regression for Splendor.

The schema declares `reserved[player][slot]` with `viz::owner_only_first_axis`:
the first axis (player) gates visibility — only the owning player sees that
slot's card_id at base. Reserve-from-tableau actions reveal the slot publicly
via `viz::reveal_slot`; reserve-from-deck does NOT reveal (owner-only base
is correct), so opponents never learn the cid of a blind reserve.

`state_hash_for_perspective(0)` walks the schema and only mixes the slot
value for reserved[p][i] when `viz_["reserved"][p, i, 0] == 1`; hidden
slots get a fixed sentinel. For an opp's blind-reserved slot the gate
is 0 → my hash never observes the cid.

This test pins that property directly: drive two API sessions through the
SAME observation trace but with different session seeds (so `randomize_unseen`
samples a different cid into opp's blind-reserved slot in each). My
`state_hash_for_perspective(0)` must be byte-equal at every ply, AND the
trace must actually exercise an opp reserve-from-deck (otherwise the test
trivially passes without testing anything).

Failure here is the canonical info-leak signature for Splendor: my hash
is observing data that depends on opp's private reserved cid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "platform"))

import dinoboard_engine as engine
from conftest import get_test_model


# Splendor's "reserve from top of deck" action range: 3 actions, one per tier.
# Mirrored from games/splendor/splendor_state.h's SplendorConfig::kReserveDeckOffset.
# kBuyFaceupOffset = 0; kBuyFaceupCount = 12; → kReserveFaceupOffset = 12;
# kReserveFaceupCount = 12; → kReserveDeckOffset = 24, kReserveDeckCount = 3.
_RESERVE_DECK_OFFSET = 24
_RESERVE_DECK_COUNT = 3


def _is_opp_reserve_from_deck(actor: int, perspective: int, action: int) -> bool:
    if actor == perspective:
        return False
    return _RESERVE_DECK_OFFSET <= action < _RESERVE_DECK_OFFSET + _RESERVE_DECK_COUNT


def test_opp_blind_reserve_change_does_not_leak_to_my_hash():
    """I6: opp reserve-from-deck is owner-only at the schema. My
    `state_hash_for_perspective(0)` must NOT depend on what cid the opp
    blind-reserved.
    """
    perspective = 0
    game_id = "splendor"
    model_path = get_test_model(game_id)

    # We need a self-play trace that includes at least one opp
    # reserve-from-deck so the I6 invariant has something to bite on.
    # Sweep seeds until we find one — Splendor's MCTS doesn't always
    # pick reserve-from-deck under default settings, so a scan is
    # cheaper than scripting the action list.
    chosen_trace = None
    chosen_seed = None
    for seed in range(50, 50 + 60):
        ep = engine.run_selfplay_episode(
            game_id=game_id, seed=seed, model_path=model_path,
            simulations=20, max_game_plies=60,
            trace_perspective=perspective,
        )
        trace = ep.get("observation_trace") or []
        if any(_is_opp_reserve_from_deck(s["actor"], perspective, s["action"])
               for s in trace):
            chosen_trace = ep
            chosen_seed = seed
            break

    if chosen_trace is None:
        pytest.skip(
            "no self-play trace within 60 seeds exercised opp reserve-from-deck; "
            "I6 test needs a sweep widening")

    trace = chosen_trace["observation_trace"]

    # Two API sessions seeded differently. randomize_unseen at
    # apply_observation will populate opp's blind-reserved cid from
    # different sample distributions in each session (per CLAUDE.md
    # "AI Pipeline Independence from Game State", session hidden is a
    # belief sample, not truth).
    seed_a = 9_999
    seed_b = 314_159

    sess_a = engine.GameSession(
        game_id, seed=seed_a, model_path="", use_filter=False)
    sess_b = engine.GameSession(
        game_id, seed=seed_b, model_path="", use_filter=False)
    sess_a.apply_initial_observation(perspective, chosen_trace["initial_observation"])
    sess_b.apply_initial_observation(perspective, chosen_trace["initial_observation"])

    h_a = sess_a.state_hash_for_perspective(perspective)
    h_b = sess_b.state_hash_for_perspective(perspective)
    assert h_a == h_b, (
        f"initial state_hash_for_perspective({perspective}) diverges "
        f"between sessions seeded {seed_a} vs {seed_b}: "
        f"{h_a:#x} vs {h_b:#x}")

    saw_opp_blind_reserve = False
    for step in trace:
        sess_a.apply_observation(
            step["action"], events=step["events"],
            public_snapshot=step.get("public_snapshot", {}))
        sess_b.apply_observation(
            step["action"], events=step["events"],
            public_snapshot=step.get("public_snapshot", {}))
        h_a = sess_a.state_hash_for_perspective(perspective)
        h_b = sess_b.state_hash_for_perspective(perspective)
        if _is_opp_reserve_from_deck(step["actor"], perspective, step["action"]):
            saw_opp_blind_reserve = True
        assert h_a == h_b, (
            f"[seed_truth={chosen_seed} ply={step['ply']} actor={step['actor']} "
            f"action={step['action']}] state_hash_for_perspective({perspective}) "
            f"diverged: {h_a:#x} vs {h_b:#x}\n"
            f"  events={step['events']}\n"
            f"This means state_hash_for_perspective({perspective}) is "
            f"mixing a value that depends on data the schema marks as "
            f"owner-only for a different player. The most likely cause is a "
            f"`reserved` slot at {{p, i}} (p != {perspective}) being walked "
            f"despite viz_['reserved'][p, i, {perspective}] == 0, or "
            f"hash_field_slot('reserved', ...) consulting truth instead of "
            f"its argument indices.")

    assert saw_opp_blind_reserve, (
        "trace had opp reserve-from-deck somewhere by sweep selection but "
        "the per-step actor/action probe missed it; check actor field "
        "alignment against trace schema")
