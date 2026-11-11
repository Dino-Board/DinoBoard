"""Lint (golden standard I3): every game's reset_with_seed override must
call IGameState::reset_step_count_base() as its first real statement.

Why first: reset_step_count_base resets the framework-managed step_count_
member which guards DAG acyclicity. If a game's body runs first and
mutates state before the base call, the step counter is out of sync with
the rest of the (zeroed) public fields, breaking the DAG invariant. RNG
does not live on IGameState; the runner owns it. So this rule reduces
to "step_count first."

Allowlist: empty — every shipped game must call the base.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GAMES_DIR = PROJECT_ROOT / "games"

# Games whose state .cpp file is exempt from this lint. Empty — every
# shipped game must call the base.
ALLOWLIST: set[str] = set()


def _state_cpp_files() -> list[Path]:
    return sorted(GAMES_DIR.glob("*/*_state.cpp"))


def _extract_reset_body(text: str, game_id: str) -> str | None:
    """Return the body (between the first { and matching }) of the first
    reset_with_seed override declaration in `text`, or None if not found.
    Robust to template prefixes and namespace qualifiers.
    """
    # Match `<...>::reset_with_seed(<args>) {` or `ClassName::reset_with_seed(<args>) {`
    sig = re.compile(
        r"(?:^|\n)\s*(?:template\s*<[^>]*>\s*)?"
        r"(?:void\s+)?[A-Za-z_][\w:<>,\s]*::reset_with_seed\s*\([^)]*\)\s*\{",
        re.MULTILINE,
    )
    m = sig.search(text)
    if not m:
        return None
    # Walk braces to find matching close.
    i = m.end() - 1  # position of opening {
    depth = 0
    start = None
    for j, ch in enumerate(text[i:], start=i):
        if ch == "{":
            depth += 1
            if start is None:
                start = j + 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:j]
    return None


@pytest.mark.parametrize("path", _state_cpp_files(), ids=lambda p: p.parent.name)
def test_reset_with_seed_first_line_calls_base(path: Path) -> None:
    """Every reset_with_seed body's first non-comment, non-blank line must
    be `IGameState::reset_step_count_base();`."""
    game_id = path.parent.name
    if game_id in ALLOWLIST:
        pytest.skip(f"{game_id} on allowlist; remove when migrated")

    text = path.read_text(encoding="utf-8")
    body = _extract_reset_body(text, game_id)
    if body is None:
        pytest.skip(f"{path.name} has no reset_with_seed override")

    # First non-blank, non-comment line.
    first_line = None
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        first_line = line
        break

    assert first_line is not None, f"{path.name}: empty reset_with_seed body"

    # Match either `IGameState::reset_step_count_base(...)` or, conceivably,
    # `this->reset_step_count_base(...)`. The first form is preferred (it
    # makes the base-class call site explicit).
    pattern = re.compile(
        r"^(?:IGameState\s*::|this\s*->)?\s*reset_step_count_base\s*\("
    )
    assert pattern.match(first_line), (
        f"{path.name}: first statement of reset_with_seed must be "
        f"`IGameState::reset_step_count_base();` (golden standard I3).\n"
        f"  got: {first_line!r}\n"
        f"  in: {path}"
    )
