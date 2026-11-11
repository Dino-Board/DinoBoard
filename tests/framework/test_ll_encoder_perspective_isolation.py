"""OB-002 regression: Love Letter encoder must not read opponent hand
knowledge from a tracker bound to a different perspective.

Architectural rule: per-perspective hand knowledge lives on state.viz_
(rules' reveal_slot_to writes), NOT on the tracker. When encoding for
seat S, the MaskedState built for S exposes only what S can legally
see; the tracker pointer is incidental for LL (perspective-agnostic
and stateless). A leak would mean the encoder is reading something
other than its own MaskedState — e.g. truth state directly, or another
perspective's tracker contents.

Test: drive a 3p Love Letter game where seat 0 plays Priest on seat 1
(seat 0's viz=1 for hand[1] now). Then encode for seat 2 using a
tracker bound to seat 0. The returned features for seat 2 must contain
NO bits derived from seat 0's knowledge of seat 1's hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "platform"))

import dinoboard_engine as engine


# Encoder layout for LoveLetter (NPlayers=3, kCardTypes=8):
#   per-player public block: 13 dims/player × NPlayers
#     (alive, protected, current_player, hand_exposed, 8 discard counts, discard_total)
#   global public: 12 dims (deck_size, ply, first_player, alive_count, 8 face_up_count)
#   per-player hand+drawn: 16 dims/player × NPlayers
#     For each pi in [0, NPlayers): pid = (perspective + pi) % NPlayers
#       dims [0..7]: hand one-hot (viz=1 slot → real, viz=0 → all-zero)
#       dims [8..15]: drawn_card one-hot (visible only to current_player)
#
# Hand-drawn block starts after the public prefix.
_NPLAYERS_3P = 3
_PUBLIC_PER_PLAYER = 13
_GLOBAL_PUBLIC = 12
_HAND_DIMS = 8
_DRAWN_DIMS = 8
_SLOT_DIMS = _HAND_DIMS + _DRAWN_DIMS
_HAND_BLOCK_OFFSET = _PUBLIC_PER_PLAYER * _NPLAYERS_3P + _GLOBAL_PUBLIC  # 51


def _find_priest_scenario():
    """Search seeds for one where seat 0's first action can be Priest on seat 1.
    kPriestOffset=28; Priest on seat 1 is action 29."""
    for seed in range(2000):
        gs = engine.GameSession("loveletter_3p", seed=seed, model_path="", use_filter=False)
        if gs.current_player != 0:
            continue
        legal = gs.get_legal_actions()
        if 29 not in legal:
            continue
        st = gs.get_state_dict()
        seat1_hand = st["players"][1]["hand"]
        return seed, 29, seat1_hand
    raise AssertionError("no seed in [0, 2000) gives Priest-on-seat-1 from seat 0")


def test_ob002_encoder_does_not_leak_priest_knowledge_to_third_party():
    """Priest reveal: tracker bound to seat 0 learns seat 1's hand.
    Encoder asked to produce features for seat 2 (with that same tracker)
    must NOT set any opp-hand bits for seat 1, because seat 2 doesn't
    legitimately know seat 1's hand — that's seat 0's private info."""
    seed, priest_action, seat1_hand = _find_priest_scenario()

    # Sanity: seat 1's hand is a real card we can look for.
    assert 1 <= seat1_hand <= 8

    # Encode seat 2's private with tracker bound to seat 0, after seat 0
    # plays Priest on seat 1.
    out = engine.encode_state_for_perspective(
        game_id="loveletter_3p",
        seed=seed,
        encode_player=2,
        tracker_perspective=0,
        event_actions=[(0, priest_action)],
    )
    features = out["features"]
    assert out["tracker_perspective"] == 0

    # Slot order for player=2: pi=0 → pid=2 (self), pi=1 → pid=0, pi=2 → pid=1.
    # Seat 1 is at slot pi=2, hand bits at offset _HAND_BLOCK_OFFSET + 2*16.
    seat1_slot_offset = _HAND_BLOCK_OFFSET + 2 * _SLOT_DIMS
    seat1_hand_bits = features[seat1_slot_offset : seat1_slot_offset + _HAND_DIMS]
    assert all(bit == 0.0 for bit in seat1_hand_bits), (
        f"OB-002 leak: seat 2's features include seat 1's hand bits "
        f"{list(seat1_hand_bits)}. Tracker bound to seat 0 (with known_hand[1]={seat1_hand}) "
        f"must not inject knowledge into a non-owner's encoding."
    )

    # And seat 0's slot (pi=1) should also be empty for player=2: seat 0
    # knows its own hand on its own state, but seat 2 doesn't.
    seat0_slot_offset = _HAND_BLOCK_OFFSET + 1 * _SLOT_DIMS
    seat0_hand_bits = features[seat0_slot_offset : seat0_slot_offset + _HAND_DIMS]
    assert all(bit == 0.0 for bit in seat0_hand_bits), (
        f"OB-002 leak: seat 2's features include seat 0's hand bits "
        f"{list(seat0_hand_bits)}."
    )


def test_ob002_perspective_player_sees_priest_knowledge():
    """Positive control: when encode_player matches tracker_perspective,
    the Priest reveal SHOULD show up in features. Otherwise the gate is
    over-eager and we've broken legitimate tracker use."""
    seed, priest_action, seat1_hand = _find_priest_scenario()

    out = engine.encode_state_for_perspective(
        game_id="loveletter_3p",
        seed=seed,
        encode_player=0,
        tracker_perspective=0,
        event_actions=[(0, priest_action)],
    )
    features = out["features"]
    assert out["tracker_perspective"] == 0

    # Slot order for player=0: pi=0 → pid=0 (self), pi=1 → pid=1, pi=2 → pid=2.
    # Seat 1 is at slot pi=1, hand bits at offset _HAND_BLOCK_OFFSET + 1*16.
    seat1_slot_offset = _HAND_BLOCK_OFFSET + 1 * _SLOT_DIMS
    seat1_hand_bits = list(features[seat1_slot_offset : seat1_slot_offset + _HAND_DIMS])
    # Card values are 1..8; one-hot bit index for card c is (c-1).
    expected_idx = seat1_hand - 1
    assert seat1_hand_bits[expected_idx] == 1.0, (
        f"Priest reveal lost: tracker bound to seat 0 should know seat 1's "
        f"hand={seat1_hand}, expected bit {expected_idx} set in features but got "
        f"{seat1_hand_bits}."
    )
    # Exactly one bit set.
    assert sum(b == 1.0 for b in seat1_hand_bits) == 1


if __name__ == "__main__":
    test_ob002_encoder_does_not_leak_priest_knowledge_to_third_party()
    test_ob002_perspective_player_sees_priest_knowledge()
    print("OB-002 tests passed")
