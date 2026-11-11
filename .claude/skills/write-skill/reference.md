# How to Write a Claude Code Skill

Before writing or modifying a skill, read the official documentation at https://code.claude.com/docs/en/skills for the latest format. This reference summarizes the key patterns used in this project.

## Write ALL THREE mirrors (required)

Every skill in this project is mirrored to three locations so it loads in
Claude Code, the `.agents` runtime, and Cursor. When you create OR modify a
skill, apply the **same change to all three** — they are kept in sync by hand
(there is no generation script). Forgetting one means that tool silently runs
the stale version.

| Location | Format | Notes |
|----------|--------|-------|
| `.claude/skills/<name>/` | dir with `SKILL.md` + supporting `.md` | Canonical source. |
| `.agents/skills/<name>/` | dir, byte-identical to `.claude` copy | Just copy the files verbatim. |
| `.cursor/rules/<name>.mdc` | **flat** `.mdc` files | Cursor has no per-skill dir or on-demand file reads. |

### Mapping into `.cursor/rules/` (flat, prefix-named)

Cursor rules can't nest dirs or lazy-read sibling files, so each skill file
becomes its own flat `.mdc` with a name-prefixed convention:

| `.claude/skills/<name>/...` | `.cursor/rules/...` |
|------------------------------|----------------------|
| `SKILL.md` | `<name>.mdc` |
| `reference.md` | `<name>-reference.mdc` |
| `<topic>.md` | `<name>-<topic>.mdc` |

`.mdc` frontmatter differs from `SKILL.md`: it adds `globs:` (a list of path
globs that auto-attach the rule, or omit for none) and `alwaysApply: false`.
The body is the same content. Example header:

```markdown
---
description: One line describing the skill
globs:
  - "games/**/config/train.json"
alwaysApply: false
---
```

Because Cursor lists every `.mdc` independently (no routing), the SKILL.md
"read [topic.md]" routing has no `.mdc` equivalent — each topic just stands
alone as its own rule file.

## Directory structure

```
.claude/skills/<skill-name>/
  SKILL.md           # Required — short entry (5-10 lines)
  reference.md       # Detailed content Claude reads on demand
  other-topic.md     # Additional supporting files as needed
```

- Skill name = directory name directly under `.claude/skills/`
- One level only — no nesting skill dirs inside other skill dirs
- Supporting files (anything other than SKILL.md) are NOT auto-loaded; Claude reads them on demand via the Read tool

## SKILL.md format

```markdown
---
description: "One line — what this skill does and when to use it (max 1536 chars)"
---
Brief paragraph (1-3 sentences) explaining the scope.
Read reference.md for the full guide.
```

### Multi-document routing pattern

For a skill covering multiple sub-topics, SKILL.md routes to supporting files in the same directory:

```markdown
---
description: "Game rules — state, rules, and build integration"
---
Read the sub-document matching your current task:
- **IGameState**: read [state.md](state.md)
- **IGameRules**: read [rules.md](rules.md)
- **Build integration**: read [build.md](build.md)
Only read the file relevant to the current task.
```

This is the pattern used by game-rules, game-ai, game-test, web-dev, and training skills in this project. Supporting files load only when Claude explicitly reads them, keeping context small.

### Frontmatter fields (all optional except description)

| Field | Use |
|-------|-----|
| `description` | **Always set.** Claude uses this to decide whether to load the skill. Put key use case first (truncated at 1536 chars combined with `when_to_use`). |
| `when_to_use` | Extra trigger phrases, appended to description in listing |
| `disable-model-invocation` | `true` = user-only via `/name`, Claude can't auto-invoke |
| `user-invocable` | `false` = hidden from `/` menu, background knowledge only |
| `allowed-tools` | Tools permitted without prompting while skill is active |
| `context` | `fork` to run in a forked subagent context |
| `agent` | Which subagent to use when `context: fork` is set |
| `paths` | Glob patterns limiting when the skill activates |

## Reference file conventions

- **80-250 lines** — if it grows past 250, split into a separate supporting file or a new skill
- **English only** (exception: i18n code examples)
- **No hardcoded counts** — "games in manifest.json" not "8 games"
- **No source line numbers** — they go stale
- **Include code examples** from actual game implementations when showing interfaces
- **Include Known Issues** section at the end if relevant pitfalls exist (2-3 lines per issue: problem → cause → lesson)

## How Claude discovers skills

1. At session start, Claude sees ALL skill names + descriptions (always in context, ~1 line each)
2. Full SKILL.md body loads only when invoked (user types `/name` or Claude auto-matches)
3. Supporting files (reference.md etc.) load only when Claude explicitly reads them
4. Editing existing SKILL.md takes effect within the session; creating a NEW `.claude/skills/` directory requires restart

## When to create a new skill vs update existing

- **New skill**: the topic doesn't fit any existing skill's scope, AND it's something Claude will be asked to do repeatedly
- **Update existing**: the topic extends an existing skill without making any single file too large (>250 lines)
- **Don't create**: one-off tasks, ephemeral project state, things derivable from reading code
