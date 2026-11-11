# Documentation Update Guide

## What to update after code changes

| Change type | Update |
|-------------|--------|
| New game added | `games/manifest.json` (sources + capabilities), game's `docs/games/<game>_api.md` |
| New framework feature | `ALGORITHM_OVERVIEW.md` if it affects search/viz/protocol; relevant skill reference.md |
| Config field added/renamed | `.claude/skills/game-config/reference.md` |
| Web API changed | `.claude/skills/web-dev/api.md` |
| Training feature added | `.claude/skills/game-ai/training.md` or `.claude/skills/training/` |
| C++ interface changed | `.claude/skills/game-rules/` or `.claude/skills/game-ai/` (whichever interface) |
| Bug fixed | Nothing unless it reveals a new pitfall worth documenting in the relevant skill's "Known Issues" section |
| Test added/renamed | `.claude/skills/game-test/` if test patterns changed |

## Top-level docs (human-facing, committed to git)

| File | Purpose | Keep it |
|------|---------|---------|
| `README.md` / `README_CN.md` | External first impression | Under 150 lines, no implementation details |
| `FEATURES_OVERVIEW.md` | Developer capability map | Under 100 lines, link to skills for details |
| `ALGORITHM_OVERVIEW.md` | Algorithm introduction | Under 400 lines, concepts not code paths |
| `docs/FRAMEWORK_DESIGN_RATIONALE.md` | Design motivation | Only update if design philosophy changes |
| `docs/games/<game>_api.md` | Per-game API integration | Action encoding + events + snapshot fields |

## Format rules

- **English only** in skills (exception: i18n code examples can have Chinese strings)
- **No hardcoded counts** — never write "8 games", "19 skills", "3552 lines". These go stale. Say "games listed in manifest.json" instead.
- **No line number references** to source code — they go stale after any edit
- **No "last updated" dates** — git blame is the source of truth
- **Keep it DRY** — each fact lives in exactly one place. Other files link, not repeat.
- **Skill reference files: 80-250 lines** — if a reference.md exceeds 250 lines, split the skill into two

## What NOT to write

- Code patterns that can be derived by reading existing games (just say "mirror closest existing game")
- Git history / changelog entries (git log is canonical)
- Debugging solutions (the fix is in the code; commit message has context)
- Invariants that are already enforced by CI tests (just name the test)
- Temporary state or in-progress work
