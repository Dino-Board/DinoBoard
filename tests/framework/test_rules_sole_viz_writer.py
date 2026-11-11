"""I1 (golden standard §2.2): rules are the sole writer of state.viz_.

The invariant: only `<game>_rules.cpp` (action handling) and
`<game>_state.cpp` (schema seeding via `viz::init_viz` inside
`reset_with_seed`) may mutate visibility state. Encoders, registers,
net_adapters, snapshot extractors, and tests must read viz_ but never
write it.

Why: if a second writer mutates viz_ outside rules, dynamic visibility
gets out of sync with what the action stream actually did. Hash /
encoder / snapshot all read viz_ — divergent writers produce a state
that hashes one way, encodes another, and serializes a third. Symptoms
are silent: search runs but DAG reuse collapses or features mismatch
the network's training distribution.

Banned tokens, scoped to games/<id>/*.cpp / *.h files:

  - viz::init_viz(...)         except in <game>_state.cpp (reset_with_seed)
  - viz::reveal_slot(...)      except in <game>_rules.cpp
  - viz::reveal_slot_to(...)   except in <game>_rules.cpp
  - viz::reset_to_base(...)    except in <game>_rules.cpp
  - direct map assignment to state.viz_[<name>] anywhere in games/

Rules-side bookkeeping is permitted to read viz_ (e.g. saving a
slot's pre-action visibility before reveal). The lint checks for
*writes*, not reads.

The lint is strict — any reveal_slot / reset_to_base / direct viz_
mutation outside the allowed file fails CI.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GAMES_DIR = PROJECT_ROOT / "games"


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def _registered_game_dirs() -> list[str]:
    out: list[str] = []
    for p in sorted(GAMES_DIR.iterdir()):
        if not p.is_dir():
            continue
        if any(p.glob("*_rules.cpp")) or any(p.glob("*_state.cpp")):
            out.append(p.name)
    return out


# Mutator → set of allowed filename suffixes.
#
# init_viz seeds viz_ from schema during reset_with_seed; lives in
# _state.cpp by convention. reveal_slot family mutates dynamic visibility
# inside do_action_fast / undo; lives in _rules.cpp.
ALLOWED_FILES: dict[str, tuple[str, ...]] = {
    "viz::init_viz": ("_state.cpp",),
    "viz::reveal_slot": ("_rules.cpp",),
    "viz::reveal_slot_to": ("_rules.cpp",),
    "viz::reset_to_base": ("_rules.cpp",),
}


def _is_allowed(call: str, path: Path) -> bool:
    suffixes = ALLOWED_FILES.get(call, ())
    return any(path.name.endswith(suf) for suf in suffixes)


@pytest.mark.parametrize("game_id", _registered_game_dirs())
def test_rules_sole_viz_writer(game_id: str) -> None:
    base = GAMES_DIR / game_id
    failures: list[str] = []

    for path in sorted(list(base.glob("*.cpp")) + list(base.glob("*.h"))):
        body = _strip_comments(path.read_text(encoding="utf-8"))

        for call in ALLOWED_FILES:
            # `re.escape(call) + r"\s*\("` naturally rejects
            # `viz::reveal_slot_to(` when scanning for `viz::reveal_slot`
            # — the `_to` between the identifier and `(` blocks the match.
            pattern = re.compile(re.escape(call) + r"\s*\(")
            for m in pattern.finditer(body):
                if not _is_allowed(call, path):
                    line = body.count("\n", 0, m.start()) + 1
                    failures.append(
                        f"  {path.relative_to(PROJECT_ROOT)}:{line}: "
                        f"`{call}` only allowed in {ALLOWED_FILES[call]}"
                    )

        # Direct map mutation: `state.viz_[...] = ...` or `s.viz_[...] = ...`
        # or `viz_[...] = ...` outside framework code (which is excluded by
        # the games/<id>/ scoping anyway).
        for m in re.finditer(r"\bviz_\s*\[[^\]]+\]\s*=", body):
            line = body.count("\n", 0, m.start()) + 1
            failures.append(
                f"  {path.relative_to(PROJECT_ROOT)}:{line}: "
                f"direct mutation of `viz_[...]` is forbidden — go through "
                f"viz::reveal_slot / viz::reset_to_base in rules"
            )

    assert not failures, (
        f"{game_id}: viz_ writer outside rules / state-reset code (golden "
        f"standard I1):\n" + "\n".join(failures)
    )
