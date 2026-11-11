# Web Frontend Patterns

Common patterns, directory structure, and pre-ship checklist for game web frontends.

---

## i18n Setup

Every game frontend must use the platform i18n system. Import and register translations at the top of your game JS file.

```javascript
import { createApp } from '/static/general/app.js';
import { t, register } from '/static/general/i18n.js';

register({
  zh: {
    'mygame.action_take': '拿取',
    'mygame.action_pass': '跳过',
    'mygame.card_names': ['公爵', '刺客', '队长', '大使', '伯爵夫人'],
  },
  en: {
    'mygame.action_take': 'Take',
    'mygame.action_pass': 'Pass',
    'mygame.card_names': ['Duke', 'Assassin', 'Captain', 'Ambassador', 'Contessa'],
  },
});
```

- Key format: `<game_id>.<semantic_name>`
- Use `t('key')` for strings, `t('key', { param: value })` for interpolation (`{param}` in template).
- Use `tList('key')` for array-valued entries (color names, card names).
- Lookup order: current language -> 'en' fallback -> key itself (visible "missing" rather than blank).
- Language detection: localStorage `dinoboard.lang`, then `navigator.language`. Toggle triggers page reload.

---

## Allowed Imports

Game JS files may import from:

| Module | Exports |
|--------|---------|
| `/static/general/app.js` | `createApp` |
| `/static/general/i18n.js` | `t`, `tList`, `register`, `getLang`, `setLang` |
| `/static/general/api.js` | `apiPost`, `apiGet`, `API_BASE` (rarely needed directly) |

Do NOT import from `sidebar.js`, `animate.js`, `layout.js`, `info_panel.js`, `replay.js`, `modal.js`, `pipeline.js` -- these are internal to the platform.

---

## Seat Rotation

For multiplayer games, rotate the display so the human player always appears at the "bottom" or "self" position.

```javascript
function rotatedIndex(absoluteIdx, humanPlayer, numPlayers) {
  return (absoluteIdx - humanPlayer + numPlayers) % numPlayers;
}

// In renderBoard:
const myIdx = ctx.state.humanPlayer;
const numP = gameState.num_players || 2;
for (let abs = 0; abs < numP; abs++) {
  const display = rotatedIndex(abs, myIdx, numP);
  // display=0 is always "me", display=1 is "next clockwise", etc.
}
```

For `getPlayerSymbol`, return labels in absolute indexing (the platform handles rotation for display).

---

## Directory Structure

```
games/<id>/web/
  index.html      -- minimal HTML shell (loads game JS as module)
  <id>.js         -- main game frontend (imports createApp, registers i18n, calls createApp)
  styles.css      -- game-specific styles (loaded by index.html)
```

### index.html template

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="/static/common.css">
  <link rel="stylesheet" href="/static/general/layout.css">
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <script type="module" src="<id>.js"></script>
</body>
</html>
```

### Typical game JS template

```javascript
import { createApp } from '/static/general/app.js';
import { t, tList, register } from '/static/general/i18n.js';

register({ zh: { /* ... */ }, en: { /* ... */ } });

function renderBoard(container, gs, ctx) {
  if (!gs) { container.innerHTML = ''; return; }
  // Build/update DOM. Use ctx.canPlay to gate interactivity.
  // Use ctx.submitAction(actionId) on click handlers.
}

function describeTransition(prev, next, actionInfo, actionId) {
  const steps = [];
  // Build fly/popup/reveal steps comparing prev vs next state.
  return steps;
}

createApp({
  gameId: '<id>',
  players: { min: 2, max: 4 },
  renderBoard,
  describeTransition,
  getPlayerSymbol: (idx) => t(`<id>.player_${idx}`),
  formatOpponentMove: (info, actionId, actor) => { /* ... */ },
  formatSuggestedMove: (info, actionId, actor) => { /* ... */ },
});
```

---

## Pre-Ship Checklist (19 items)

### Functionality
1. [ ] `createApp` called with correct `gameId` matching manifest.json
2. [ ] `renderBoard` renders all game state; empty state handled (gs===null)
3. [ ] All legal actions reachable via natural interaction (no action-ID buttons)
4. [ ] `ctx.canPlay` gating: interactive elements disabled when not human's turn
5. [ ] `ctx.submitAction(actionId)` called with correct action ID from legal_actions
6. [ ] Multi-step actions (e.g. gem selection) tracked in local state, submitted as single actionId

### Animation
7. [ ] `describeTransition` defined and returns steps for every action type
8. [ ] fly steps have correct from/to selectors (test: elements exist in prevState's DOM)
9. [ ] hideFrom or onStart used appropriately per source element type
10. [ ] reveal modals used for private-info events (Priest peek, Baron compare)
11. [ ] popup used for ephemeral status (score change, block notification)

### Visual
12. [ ] Spatial anchoring: no layout shift when contents change
13. [ ] Current player indicator visible and prominent
14. [ ] Legal-action hover feedback present
15. [ ] Touch targets >= 44px on mobile
16. [ ] Works at mobile viewport (board scaled, no horizontal scroll)

### i18n
17. [ ] All user-visible strings use `t()` / `tList()`
18. [ ] Both zh and en translations registered
19. [ ] No hardcoded Chinese/English text in renderBoard

---

## Known UI BUGs

- **BUG-020**: Animation flyer sizing wrong under CSS transform:scale() -- fixed in animate.js (auto-detects scale from rendered rect vs offsetWidth).
- **BUG-035**: reveal modal click-to-dismiss blocks replay autoplay -- fixed with 1800ms auto-dismiss when `ctx.autoplay`.
- General: if animation selectors break after a DOM restructure, the engine catches the error and falls back to instant render. Fix the selectors rather than relying on this safety net.

---

## Reference Implementations

Study these shipped games when building a new frontend:

| Game | File | Best for |
|------|------|----------|
| Azul | `games/azul/web/azul.js` | Multi-phase settlement animation (action → round-end → bonus highlights) |
| Splendor | `games/splendor/web/splendor.js` | Complex multi-step interactions (buy = gems + card), stable screen-slot reconciliation |
| 6 nimmt! | `games/6nimmit/web/6nimmit.js` | Simultaneous-action + run steps, `hideSeatPicker` config |
| Coup | `games/coup/web/coup.js` | Reveal modals + hidden-info handling, `disableForce` + `showWinrateDefault: false` |
| Love Letter | `games/loveletter/web/loveletter.js` | Reveal modals + card play animation, guard guess modal, `revealHidden` usage |
