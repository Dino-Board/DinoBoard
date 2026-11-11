"""Pin tictactoe's visibility schema surface.

  - the header exposes `schema()` (future-proofs against accidental
    rename / removal)
  - the .cpp's schema() body declares every persistent state field with
    `all_public` viz (tictactoe is fully observable — no field may declare
    `all_hidden` or `owner_only_first_axis`)
  - `reset_with_seed` wires `init_viz(*this, schema())` so a freshly reset
    state actually carries the schema's base_viz in its `viz_` map
"""
from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEADER = PROJECT_ROOT / "games" / "tictactoe" / "tictactoe_state.h"
SOURCE = PROJECT_ROOT / "games" / "tictactoe" / "tictactoe_state.cpp"


def _read(p: Path) -> str:
    assert p.exists(), f"{p} missing"
    return p.read_text(encoding="utf-8")


def test_header_declares_schema_accessor() -> None:
    text = _read(HEADER)
    assert '#include "../../engine/core/visibility_schema.h"' in text, (
        "tictactoe_state.h must include visibility_schema.h"
    )
    assert re.search(
        r"static\s+const\s+viz::VisibilitySchema\s*&\s*schema\s*\(\s*\)\s*;",
        text,
    ), "tictactoe_state.h must declare `static const viz::VisibilitySchema& schema()`"


def test_source_declares_all_public_fields() -> None:
    """Every persistent state field appears in schema() with all_public viz.
    Tic-Tac-Toe is fully observable — any all_hidden / owner_only_first_axis
    declaration here would be a bug."""
    text = _read(SOURCE)
    assert '#include "../../engine/core/viz_runtime.h"' in text, (
        "tictactoe_state.cpp must include viz_runtime.h"
    )
    # All persistent fields from tictactoe_state.h's struct body. Internal
    # framework state (rng_salt_, draw_nonce_, step_count_) is NOT declared
    # by per-game schemas — it lives behind the framework-internal field
    # convention.
    expected_fields = (
        "current_player",
        "winner",
        "terminal",
        "move_count",
        "scores",
        "board",
    )
    for name in expected_fields:
        assert re.search(
            rf'declare_field\(\s*schema\s*,\s*"{name}"\s*,\s*viz::all_public\(',
            text,
        ), f"schema() must declare '{name}' with viz::all_public(...)"

    # No hidden / owner-only viz in a fully-public game.
    assert "all_hidden" not in text, (
        "tictactoe is fully observable — schema() must not declare all_hidden"
    )
    assert "owner_only_first_axis" not in text, (
        "tictactoe is fully observable — schema() must not declare owner_only_first_axis"
    )


def test_reset_with_seed_wires_init_viz() -> None:
    """The schema must actually be applied to state.viz_ at reset time —
    a declared-but-never-installed schema would be silently ignored by
    every consumer. Pin the wiring."""
    text = _read(SOURCE)
    assert re.search(
        r"viz::init_viz\(\s*\*this\s*,\s*schema\(\s*\)\s*\)\s*;",
        text,
    ), "reset_with_seed must call viz::init_viz(*this, schema())"
