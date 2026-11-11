"""Pin quoridor's visibility schema surface.

Quoridor is fully observable — pawn positions, wall placements, and
remaining-wall counts are all public. Same all_public-only pattern as
tictactoe; this file is the per-game pin so a future contributor can't
accidentally declare a field hidden when quoridor's rules don't permit
hidden info.
"""
from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEADER = PROJECT_ROOT / "games" / "quoridor" / "quoridor_state.h"
SOURCE = PROJECT_ROOT / "games" / "quoridor" / "quoridor_state.cpp"


def _read(p: Path) -> str:
    assert p.exists(), f"{p} missing"
    return p.read_text(encoding="utf-8")


def test_header_declares_schema_accessor() -> None:
    text = _read(HEADER)
    assert '#include "../../engine/core/visibility_schema.h"' in text
    assert re.search(
        r"static\s+const\s+viz::VisibilitySchema\s*&\s*schema\s*\(\s*\)\s*;",
        text,
    ), "quoridor_state.h must declare `static const viz::VisibilitySchema& schema()`"


def test_source_declares_all_public_fields() -> None:
    text = _read(SOURCE)
    assert '#include "../../engine/core/viz_runtime.h"' in text
    expected_fields = (
        "current_player",
        "winner",
        "terminal",
        "move_count",
        "scores",
        "pawn_row",
        "pawn_col",
        "walls_remaining",
        "h_walls",
        "v_walls",
    )
    for name in expected_fields:
        assert re.search(
            rf'declare_field\(\s*schema\s*,\s*"{name}"\s*,\s*viz::all_public\(',
            text,
        ), f"schema() must declare '{name}' with viz::all_public(...)"
    assert "all_hidden" not in text, (
        "quoridor is fully observable — schema() must not declare all_hidden"
    )
    assert "owner_only_first_axis" not in text, (
        "quoridor is fully observable — schema() must not declare owner_only_first_axis"
    )


def test_reset_with_seed_wires_init_viz() -> None:
    text = _read(SOURCE)
    assert re.search(
        r"viz::init_viz\(\s*\*this\s*,\s*schema\(\s*\)\s*\)\s*;",
        text,
    ), "reset_with_seed must call viz::init_viz(*this, schema())"
