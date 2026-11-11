"""Pin splendor's visibility schema surface.

Splendor partitions:
  - all_public: bank, tableau, nobles, scores, bonuses, points,
    reserved_visible (the public face-up flag), counts, stage.
  - owner_only_first_axis: reserved[N][3] (card ids — face-down by
    default; rules reveal_slot when a reserve becomes face-up).
  - decks (variable-length per-tier vectors): NOT a schema slot.
    Public size + hidden contents already handled by state_hash_for_perspective
    + randomize_unseen.
"""
from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEADER = PROJECT_ROOT / "games" / "splendor" / "splendor_state.h"
SOURCE = PROJECT_ROOT / "games" / "splendor" / "splendor_state.cpp"


def _read(p: Path) -> str:
    assert p.exists(), f"{p} missing"
    return p.read_text(encoding="utf-8")


def test_header_declares_schema_accessor() -> None:
    text = _read(HEADER)
    assert '#include "../../engine/core/visibility_schema.h"' in text
    assert re.search(
        r"static\s+const\s+viz::VisibilitySchema\s*&\s*schema\s*\(\s*\)\s*;",
        text,
    ), "splendor_state.h must declare `static const viz::VisibilitySchema& schema()`"


def test_source_partitions_fields_correctly() -> None:
    text = _read(SOURCE)
    assert '#include "../../engine/core/viz_runtime.h"' in text

    public_fields = (
        "current_player", "first_player", "plies", "final_round_remaining",
        "stage", "pending_returns", "pending_nobles_size", "winner",
        "terminal", "shared_victory", "nobles_size", "pending_noble_slots",
        "scores", "bank", "player_points", "player_cards_count",
        "player_nobles_count", "reserved_size", "tableau_size", "nobles",
        "player_gems", "player_bonuses", "tableau", "reserved_visible",
        "reserved_tier",
    )
    for name in public_fields:
        assert re.search(
            rf'declare_field\(\s*schema\s*,\s*"{name}"\s*,\s*viz::all_public\(',
            text,
        ), f"schema() must declare '{name}' with viz::all_public(...)"

    assert re.search(
        r'declare_field\(\s*schema\s*,\s*"reserved"\s*,\s*'
        r'viz::owner_only_first_axis\(',
        text,
    ), "schema() must declare 'reserved' with owner_only_first_axis(...)"

    assert not re.search(
        r'declare_field\(\s*schema\s*,\s*"decks"\s*',
        text,
    ), (
        "decks must NOT be a schema slot — variable-length per-tier "
        "vectors with hidden contents and public size; handled by "
        "state_hash_for_perspective / randomize_unseen instead"
    )


def test_reset_with_seed_wires_init_viz() -> None:
    text = _read(SOURCE)
    assert re.search(
        r"viz::init_viz\(\s*\*this\s*,\s*schema\(\s*\)\s*\)\s*;",
        text,
    ), "reset_with_seed must call viz::init_viz(*this, schema())"
