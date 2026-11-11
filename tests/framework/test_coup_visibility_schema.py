"""Pin coup's visibility schema surface.

Coup is a hidden-info game. Schema partitions:
  - all_public: stage / coins / alive / revealed-flags / current_player /
    declared_action / claimed_character / etc.
  - owner_only_first_axis: influence[N][2] (face-down hand cards).
  - all_hidden: exchange_drawn[2] (rules-side reveal_slot_to active_player
    on draw, reset_to_base on return).
  - court_deck: NOT a slot field. Variable-length vector; size is public,
    contents hidden, both already handled by state_hash_for_perspective /
    randomize_unseen.
"""
from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEADER = PROJECT_ROOT / "games" / "coup" / "coup_state.h"
SOURCE = PROJECT_ROOT / "games" / "coup" / "coup_state.cpp"


def _read(p: Path) -> str:
    assert p.exists(), f"{p} missing"
    return p.read_text(encoding="utf-8")


def test_header_declares_schema_accessor() -> None:
    text = _read(HEADER)
    assert '#include "../../engine/core/visibility_schema.h"' in text
    assert re.search(
        r"static\s+const\s+viz::VisibilitySchema\s*&\s*schema\s*\(\s*\)\s*;",
        text,
    ), "coup_state.h must declare `static const viz::VisibilitySchema& schema()`"


def test_source_partitions_fields_correctly() -> None:
    text = _read(SOURCE)
    assert '#include "../../engine/core/viz_runtime.h"' in text

    public_fields = (
        "current_player", "first_player", "winner", "terminal", "ply",
        "stage", "active_player", "declared_action", "action_target",
        "claimed_character", "challenger", "challenge_loser",
        "action_challenged", "action_challenge_succeeded", "blocker",
        "block_character", "counter_challenged",
        "counter_challenge_succeeded", "challenge_check_index",
        "exchange_held_count", "coins", "alive", "revealed",
    )
    for name in public_fields:
        assert re.search(
            rf'declare_field\(\s*schema\s*,\s*"{name}"\s*,\s*viz::all_public\(',
            text,
        ), f"schema() must declare '{name}' with viz::all_public(...)"

    # Hand cards: owner-only.
    assert re.search(
        r'declare_field\(\s*schema\s*,\s*"influence"\s*,\s*'
        r'viz::owner_only_first_axis\(',
        text,
    ), "schema() must declare 'influence' with owner_only_first_axis(...)"

    # Exchange-drawn: hidden base, rules will reveal_slot_to(active_player)
    # on draw later. Here we only pin the base.
    assert re.search(
        r'declare_field\(\s*schema\s*,\s*"exchange_drawn"\s*,\s*'
        r'viz::all_hidden\(',
        text,
    ), "schema() must declare 'exchange_drawn' with all_hidden(...)"

    # court_deck is variable-length and intentionally NOT in the schema.
    assert not re.search(
        r'declare_field\(\s*schema\s*,\s*"court_deck"\s*',
        text,
    ), (
        "court_deck must NOT be a schema slot — variable-length vector "
        "with hidden contents and public size; handled by "
        "state_hash_for_perspective / randomize_unseen instead"
    )


def test_reset_with_seed_wires_init_viz() -> None:
    text = _read(SOURCE)
    assert re.search(
        r"viz::init_viz\(\s*\*this\s*,\s*schema\(\s*\)\s*\)\s*;",
        text,
    ), "reset_with_seed must call viz::init_viz(*this, schema())"
