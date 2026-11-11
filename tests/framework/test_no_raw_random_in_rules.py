"""Lint (golden standard I3, §2.3): rules code must not directly
construct std::random_device / std::mt19937{...} / std::shuffle / splitmix64
— all randomness must flow through IGameState::derive_rng(domain_tag).

Phase 2 wrap-up (this state): every shipped game has been migrated to
derive_rng. The allowlist is empty and the test runs in error mode for
all games. Any new file in games/<id>/*_rules.cpp or *_state.cpp
containing a banned pattern fails CI.

The lint only inspects *_rules.cpp / *_state.cpp under games/<id>/
since those are the files governed by §2.3. net_adapter.cpp,
register.cpp etc. are part of randomize_unseen / chance dispatch
which has its own contract (must accept an external rng) and is
deliberately out of scope.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GAMES_DIR = PROJECT_ROOT / "games"

# Banned patterns. `mt19937{...}` / `mt19937(...)` is the typical raw
# constructor smoke for "rolled own RNG plumbing". `random_device` is
# never legitimate inside rules. `std::shuffle` is banned because Phase 2
# treats deck/bag storage as multiset — randomness lives in per-draw
# index pick, not in a one-shot vector permutation.
#
# Distributions (`std::uniform_int_distribution` etc.) are NOT banned —
# they are the natural way to consume a derive_rng()-returned mt19937_64.
# A draw_one_tile that does `pick(rng)` against a freshly-derived rng is
# correct golden-standard code.
BANNED_PATTERNS: list[tuple[str, str]] = [
    ("std::random_device", r"std::random_device\b"),
    ("std::mt19937 constructor",
     r"std::mt19937(?:_64)?\s*[\{\(](?!\s*\)\s*;)"),
    ("std::default_random_engine", r"std::default_random_engine\b"),
    ("std::shuffle (raw)", r"std::shuffle\b"),
    ("splitmix64 (home-rolled rng)", r"\bsplitmix64\b"),
]

# Phase 2 complete: every shipped game now uses derive_rng. Keep the
# variable for forward compatibility — if a future game lands with an
# unmigrated rules path, allowlist it here while the migration PR is
# in flight.
ALLOWLISTED_GAMES: set[str] = set()


def _files_for(game_id: str) -> list[Path]:
    base = GAMES_DIR / game_id
    out: list[Path] = []
    out.extend(sorted(base.glob("*_rules.cpp")))
    out.extend(sorted(base.glob("*_state.cpp")))
    return out


def _strip_comments(text: str) -> str:
    """Drop // line comments and /* ... */ block comments before scanning,
    so banned identifiers mentioned only in docs don't trip the lint."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def _registered_game_dirs() -> list[str]:
    """Game dirs that contain at least one *_rules.cpp or *_state.cpp.
    Filters out reference / docs subdirs like games/6nimmit/参考."""
    out: list[str] = []
    for p in sorted(GAMES_DIR.iterdir()):
        if not p.is_dir():
            continue
        if any(p.glob("*_rules.cpp")) or any(p.glob("*_state.cpp")):
            out.append(p.name)
    return out


@pytest.mark.parametrize("game_id", _registered_game_dirs())
def test_no_raw_random_in_rules(game_id: str) -> None:
    if game_id in ALLOWLISTED_GAMES:
        pytest.skip(
            f"{game_id} not yet migrated to derive_rng; "
            f"remove from ALLOWLISTED_GAMES in Phase 2"
        )

    failures: list[str] = []
    for path in _files_for(game_id):
        body = _strip_comments(path.read_text(encoding="utf-8"))
        for label, pat in BANNED_PATTERNS:
            for m in re.finditer(pat, body):
                # Compute line number from the original (uncommented) text
                # for a useful error.
                line = body.count("\n", 0, m.start()) + 1
                failures.append(
                    f"  {path.relative_to(PROJECT_ROOT)}:{line}: "
                    f"banned `{label}` — use IGameState::derive_rng(domain_tag) instead"
                )

    assert not failures, (
        f"{game_id}: rules / state files contain raw random constructs "
        f"(golden standard §2.3 / I3):\n" + "\n".join(failures)
    )
