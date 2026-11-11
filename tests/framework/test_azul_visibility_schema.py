"""Pin azul's visibility schema surface.

Azul partitions:
  - all_public: every game-facing field — factories, center, scores,
    each player's pattern lines / wall / floor / score, round meta.
    Azul has no hidden private state; the only hidden info is the
    bag/box-lid composition (multisets), and those are NOT slot fields.
  - bag, box_lid: variable-length vectors. Hidden contents, public
    size — handled by state_hash_for_perspective (multiset only) and
    randomize_unseen. NOT in the schema.
"""
from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEADER = PROJECT_ROOT / "games" / "azul" / "azul_state.h"
SOURCE = PROJECT_ROOT / "games" / "azul" / "azul_state.cpp"


def _read(p: Path) -> str:
    assert p.exists(), f"{p} missing"
    return p.read_text(encoding="utf-8")


def test_header_declares_schema_accessor() -> None:
    text = _read(HEADER)
    assert '#include "../../engine/core/visibility_schema.h"' in text
    assert re.search(
        r"static\s+const\s+viz::VisibilitySchema\s*&\s*schema\s*\(\s*\)\s*;",
        text,
    ), "azul_state.h must declare `static const viz::VisibilitySchema& schema()`"


def test_source_declares_all_public_fields() -> None:
    text = _read(SOURCE)
    assert '#include "../../engine/core/viz_runtime.h"' in text

    public_fields = (
        "current_player", "game_first_player", "first_player_next_round",
        "winner", "round_index", "terminal",
        "first_player_token_in_center", "shared_victory",
        "scores", "factories", "center",
        "player_line_len", "player_line_color", "player_wall_mask",
        "player_floor", "player_floor_count", "player_score",
    )
    for name in public_fields:
        assert re.search(
            rf'declare_field\(\s*schema\s*,\s*"{name}"\s*,\s*viz::all_public\(',
            text,
        ), f"schema() must declare '{name}' with viz::all_public(...)"

    # Azul has no per-seat private fields and no per-slot hidden fields.
    assert "all_hidden" not in text, (
        "azul has no per-slot hidden fields — bag/box_lid handled "
        "outside the schema as variable-length vectors"
    )
    assert "owner_only_first_axis" not in text, (
        "azul has no owner-private state — no field should declare "
        "owner_only_first_axis"
    )

    # Variable-length vectors must NOT be schema slots.
    for name in ("bag", "box_lid"):
        assert not re.search(
            rf'declare_field\(\s*schema\s*,\s*"{name}"\s*',
            text,
        ), (
            f"{name} must NOT be a schema slot — variable-length vector "
            "with hidden contents and public size; handled by "
            "state_hash_for_perspective (multiset) / randomize_unseen instead"
        )


def test_reset_with_seed_wires_init_viz() -> None:
    text = _read(SOURCE)
    assert re.search(
        r"viz::init_viz\(\s*\*this\s*,\s*schema\(\s*\)\s*\)\s*;",
        text,
    ), "reset_with_seed must call viz::init_viz(*this, schema())"
