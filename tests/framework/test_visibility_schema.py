"""Surface tests for engine/core/visibility_schema.h + viz_runtime.h.

Static parse / inspection checks: the schema header declares the locked
vocabulary (`VizTensor`, `FieldDecl`, base-viz builders, `declare_field`),
viz_runtime.h re-exports the rules-side mutation primitives, and
IGameState carries the `viz_` member. Per-game schema details and
runtime semantics are exercised by `test_<game>_visibility_schema.py`
and the viz-protocol round-trip tests."""
from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_HEADER = PROJECT_ROOT / "engine" / "core" / "visibility_schema.h"
RUNTIME_HEADER = PROJECT_ROOT / "engine" / "core" / "viz_runtime.h"
GAME_INTERFACES = PROJECT_ROOT / "engine" / "core" / "game_interfaces.h"


def _read(p: Path) -> str:
    assert p.exists(), f"{p} missing"
    return p.read_text(encoding="utf-8")


def test_schema_header_has_locked_surface() -> None:
    """visibility_schema.h declares the locked schema vocabulary."""
    text = _read(SCHEMA_HEADER)
    # Core types.
    assert "struct VizTensor" in text
    assert "struct FieldDecl" in text
    assert "struct VisibilitySchema" in text
    # Static base-viz builders.
    for sym in ("all_public", "all_hidden", "owner_only_first_axis"):
        assert f"VizTensor {sym}(" in text or f" {sym}(" in text, f"missing {sym}"
    # Declaration helper.
    assert "declare_field" in text
    # Offset arithmetic helper used by runtime ops.
    assert "flat_offset_data_only" in text


def test_schema_header_has_no_overlay_residue() -> None:
    """The overlay closure design (kRevealWhen / OverlayKind /
    reveal_when / custom_overlay / FieldDecl::overlay) was rolled back
    to enforce I1 (rules are sole viz writer). Prevent accidental
    reintroduction."""
    text = _read(SCHEMA_HEADER)
    banned = (
        "OverlayKind",
        "kRevealWhen",
        "kRevealToHistory",
        "kDerivedSizeOf",
        "kCustom",
        "reveal_when(",
        "custom_overlay(",
        "FieldDecl::overlay",
    )
    found = [tok for tok in banned if tok in text]
    assert not found, (
        "visibility_schema.h still references overlay-design symbols: "
        + ", ".join(found)
    )


def test_runtime_header_has_helpers() -> None:
    """viz_runtime.h carries the rules-side mutation primitives + the
    SlotVisitor type alias. The `for_each_visible_slot` body lives in
    viz_walker.h; runtime only re-exports the callback type so encoder
    / snapshot headers can include the smaller header."""
    text = _read(RUNTIME_HEADER)
    for sym in ("init_viz", "reveal_slot", "reveal_slot_to",
                "reset_to_base", "swap_slot", "swap_slot_owned",
                "SlotVisitor"):
        assert sym in text, f"viz_runtime.h missing {sym}"


def test_igamestate_carries_viz_member() -> None:
    """game_interfaces.h's IGameState exposes `viz_` indexed by FieldId.

    The viz container was reshaped from `unordered_map<string, VizTensor>`
    to `vector<VizTensor>` so the schema-walker hot path (per-slot
    callbacks at every MCTS sim) can index by FieldId in O(1) instead of
    paying a string hash + map probe per field. String-keyed access is
    still supported for the rules-side reveal helpers via
    `viz::viz_get(state, name)`, which resolves name → FieldId against
    the schema once per call (slow path)."""
    text = _read(GAME_INTERFACES)
    assert re.search(
        r"std::vector<\s*viz::VizTensor\s*>\s*viz_",
        text,
    ), "IGameState must carry `std::vector<viz::VizTensor> viz_`"
    # Step-count base helper still in place. RNG helpers are not on
    # IGameState — the runner owns the rng.
    assert "reset_step_count_base" in text


def test_runtime_header_includes_full_igamestate() -> None:
    """viz_runtime.h pulls in game_interfaces.h (which carries the
    concrete IGameState). Schema header itself stays free of that
    include — keeps the include graph one-directional."""
    runtime = _read(RUNTIME_HEADER)
    schema = _read(SCHEMA_HEADER)
    assert '#include "game_interfaces.h"' in runtime
    assert '#include "game_interfaces.h"' not in schema, (
        "visibility_schema.h must NOT include game_interfaces.h — "
        "would create a cycle since game_interfaces.h includes the schema"
    )
