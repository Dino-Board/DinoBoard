# Web Animation Reference

Source: `platform/static/general/animate.js`

---

## 1. How It Works

`describeTransition(prevState, newState, actionInfo, actionId)` returns a `step[]` array.

**Key mental model**: the engine executes all steps on the **OLD DOM** (still showing prevState), then the caller re-renders the entire board to newState. This means:

- All `from`/`to` selectors resolve against the old DOM
- After the animation chain finishes, everything snaps to newState — anything not animated just "pops"
- If describeTransition throws or any step fails, the engine silently falls back to instant re-render

Animations are cosmetic transitions — they never affect game state.

---

## 2. Step Types

### fly — move a floating sprite between two positions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| from | string\|Element | — | CSS selector or element (start position) |
| to | string\|Element | — | CSS selector or element (end position) |
| createElement | () => Element | — | Creates the visual flying sprite |
| duration | int | 350 | Flight duration ms |
| width | int | (auto) | **Always set explicitly** when `from` is a container bigger than the thing flying |
| height | int | (auto) | Same — explicit size prevents inheriting container dimensions |
| hideFrom | bool | false | Hide source element (`visibility:hidden`, preserves layout). Only when source has no meaningful empty state |
| onStart | (srcEl) => void | — | Mutate source DOM to "post-take" state. **Preferred over hideFrom** when the empty container background matters |
| onComplete | () => void | — | Update DOM after flight (e.g. decrement counter text) so later steps see correct state |

### flyGroup — parallel flights

`{ type: 'flyGroup', flights: [fly, fly, ...] }` — all flights run simultaneously, finishes when the slowest completes. Flights inside should use similar durations — mismatched speeds look jarring.

### group — parallel arbitrary steps

`{ type: 'group', children: [step, step, ...] }` — any mix of step types in parallel.

### popup — floating text that fades away

`{ type: 'popup', target, content, duration: 900 }` — anchored to target element, floats up and fades out. For score changes, status text ("+5", "BLOCKED").

Optional fields: `className` (extra CSS class on the popup element), `width` (default 60), `height` (default 28). Set explicit width/height for longer text content (e.g. Coup's action bubbles use `width: 220, height: 36`).

### reveal — blocking modal for private info

`{ type: 'reveal', title, body, buttonText: t('animate.reveal_btn_ack') }` — blocks animation chain until player acknowledges. For Priest peek, Baron compare, card draws. **Do not set `timeoutMs`** — forcing the player to see it is the point. (During replay autoplay, the engine auto-dismisses at 1800ms regardless.)

Optional field: `className` (extra CSS class on the reveal panel element).

### run — DOM mutation between animation phases

`{ type: 'run', fn: () => { ... } }` — synchronous DOM mutation callback. **Critical for multi-phase animations** where later flights need the DOM to reflect an intermediate state.

### fadeOut / highlight / pause

| Step | Fields | Description |
|------|--------|-------------|
| fadeOut | target, duration(350) | Fade element to transparent |
| highlight | target, className("anim-highlight"), duration(600) | Briefly add CSS class |
| pause | duration(200) | Explicit wait for timing control |

---

## 3. Maintaining Intermediate State Between Flights

A single action is often a chain of flights (buy card = gems fly to bank + card flies to player). Between steps, the DOM must show the correct intermediate state — otherwise later steps capture rects from wrong positions.

**Three mechanisms, in order of preference:**

1. **`onStart(srcEl)`** — called after rect capture, before flight. Mutate the source to show "post-take" state (e.g. show empty-slot ghost). The sprite flies away while the container remains visible with its empty background. **Use when the source has a meaningful empty state.**

2. **`hideFrom: true`** — hides the entire source element (`visibility:hidden`). Simple but hides empty-state backgrounds too. **Use only when "completely gone" is the correct visual.**

3. **`onComplete`** — called after flight finishes. Update text or counters so subsequent steps see correct numbers.

After ALL animations complete, the engine restores hidden elements and triggers a full re-render to newState.

---

## 4. Multi-Phase Choreography

Most animation bugs come from multi-phase sequences. The general pattern:

```
flyGroup (parallel flights)  →  pause  →  run (update DOM)  →  more flights  →  popup
```

### Example: 6 nimmt! card placement with row overflow

1. `flyGroup` — all row cards fly to player's penalty pile in parallel
2. `run` — clear row slots in DOM, update pile count, restore visibility
3. `pause` (250ms) — let player see the cleared row
4. `fly` — placed card flies into the now-empty row slot

Without the `run` between steps 1 and 4, step 4 would see the row still full from the old DOM.

### Example: Azul round-end settlement

1. `flyGroup` — take tiles from factory
2. `pause` (200ms) — settlement separator
3. Per filled row: `fly` (300ms) pattern→wall + `popup` (450ms) score + `pause` (40ms)
4. `group` — all floor penalty popups in parallel
5. `pause` (300ms) — bonus separator
6. `group` — bonus highlights in parallel
7. `group` — bonus score popups in parallel

---

## 5. Common Pitfalls

### Flying sprite inherits container size
**Symptom**: tile flies from factory but appears huge.
**Cause**: `width`/`height` omitted on fly step — auto-sized from the factory container rect instead of the tile.
**Fix**: always set explicit `width` and `height` when `from` is a container.

### Object disappears mid-chain, reappears on re-render
**Symptom**: step 1 flies object away with `hideFrom`, step 2 needs that object's position but it's gone.
**Cause**: `hideFrom` makes element invisible immediately — later steps can't find it.
**Fix**: use `onComplete` or a `run` step to maintain intermediate DOM state manually.

### Hidden element blocks underlying content
**Symptom**: tile covers a dashed outline. Tile flies away with `hideFrom`, but outline still invisible because the hidden tile occupies the space.
**Cause**: `visibility:hidden` preserves layout — the invisible element still blocks rendering of elements underneath.
**Fix**: use `onStart` instead — actually remove or restyle the element so underlying content shows.

### Stale selectors after run mutation
**Symptom**: `run` step clears a row, subsequent `fly` can't find `[data-row-slot="..."]`.
**Fix**: design `run` mutations to preserve selector targets, or capture elements before the mutation. Use `data-*` attributes as stable anchors.

### Parallel steps accidentally sequential
**Symptom**: two independent movements animate one after another with a 60ms gap.
**Cause**: steps in the array are sequential by default.
**Fix**: wrap in `flyGroup` (parallel flights) or `group` (any parallel mix).

---

## 6. `data-*` Selector Convention

`describeTransition` locates DOM elements via CSS selectors. Rendering functions must add `data-*` attributes.

**Hard requirement**: `data-player="<idx>"` on every player area — popup targeting depends on it.

Other names are game-specific — just keep them consistent between renderBoard and describeTransition.

---

## 7. Replay Mode

The engine accepts `options.autoplay`. During replay autoplay:

- `reveal` modals auto-dismiss at 1800ms (live play requires click)
- All other step types run identically to live play
- After each frame's animation finishes, the settled board is held for
  `REPLAY_AUTOPLAY_STEP_HOLD_MS` (1800ms, `STEP_GAP_MS` in `replay.js`) before
  the next frame starts. Frames with no animation (intra-turn sub-actions)
  get only this hold, so it must stay long enough to read the board.

**Game code must do nothing special for replay.** Never check "am I in replay?" inside describeTransition — the function should be pure. The engine handles replay differences automatically.

---

## 8. Timing Reference

### Framework constants

| Constant | Value |
|----------|-------|
| DEFAULT_DURATION | 350ms |
| STEP_GAP | 60ms |
| MAX_QUEUE_MS | 30,000ms |
| REPLAY_AUTOPLAY_REVEAL_MS | 1,800ms |
| REPLAY_AUTOPLAY_STEP_HOLD_MS | 1,800ms |

### Recommended durations by content type

| Content | Duration | Rationale |
|---------|----------|-----------|
| Token/gem fly | 350-400ms | Small, fast |
| Card fly | 450-600ms | Larger, need time to read |
| Dramatic reveal fly | 1,000-1,500ms | Player needs to process |
| Score popup | 450-900ms | Brief, anchored |
| Settlement inter-item pause | 40-100ms | Rapid but visible sequence |
| Phase separator pause | 200-300ms | Let player register phase change |

### Actual values in shipped games

| Game | Constant | Value |
|------|----------|-------|
| Azul | FLY_DURATION | 350ms |
| Azul | SETTLE_FLY_DURATION | 300ms |
| Azul | SETTLE_POPUP_DURATION | 450ms |
| Azul | SETTLE_INTER_TILE_PAUSE | 40ms |
| Azul | SETTLE_PAUSE | 200ms |
| Splendor | CARD_FLY_MS | 550ms |
| Splendor | TOKEN_FLY_MS | 400ms |

---

## 9. Minimum Checklist

Every describeTransition must satisfy:

1. Physical movement (take, place, move) → at least one `fly` / `flyGroup`
2. Non-physical action (declare, challenge) → at least one `popup`
3. Private info reveal (Priest peek, Baron compare) → `reveal`, NOT popup
4. `width`/`height` set explicitly on every fly whose `from` is a container
5. Multi-phase chains use `run` steps to maintain intermediate DOM state

---

## 10. Reference Implementations

| Game | File | Best for |
|------|------|----------|
| Azul | `games/azul/web/azul.js` | Multi-phase settlement animation (flyGroup → pause → per-tile fly+popup chain → floor penalty group → bonus highlights) |
| Splendor | `games/splendor/web/splendor.js` | Complex multi-step interactions (buy = gem flights + card flight in parallel; reserve = card + gold + deck refill) |
| 6 nimmt! | `games/6nimmit/web/6nimmit.js` | Simultaneous-action + run steps (reveal all commits → per-card apply with eat/append via `run` DOM mutations between flights) |
| Coup | `games/coup/web/coup.js` | Reveal modals + hidden-info handling (challenge-reveal modal, popup bubbles for info-only actions, reshuffle fly sequence) |
| Love Letter | `games/loveletter/web/loveletter.js` | Reveal modals + card play animation (Priest peek / Baron compare reveal steps, guard guess modal) |
