import { createApp } from '/static/general/app.js';
import { apiGet, API_BASE } from '/static/general/api.js';
import { t, register } from '/static/general/i18n.js';

register({
  zh: {
    'quo.title': '步步为营',
    'quo.intro': '先到达对侧底线获胜，可放墙但必须保留双方通路。',
    'quo.action_unknown': '动作 #{id}',
    'quo.action_start': '开局',
    'quo.move_to': '走子到 ({row}, {col})',
    'quo.place_h': '放横墙 @ ({row}, {col})',
    'quo.place_v': '放竖墙 @ ({row}, {col})',
    'quo.player_a': 'A（黑）',
    'quo.player_b': 'B（红）',
    'quo.walls_default': '墙剩余：--',
    'quo.walls_count': '墙剩余：A={a} / B={b}',
  },
  en: {
    'quo.title': 'Quoridor',
    'quo.intro': 'First to reach the opposite baseline wins. Place walls but always leave a path for both pawns.',
    'quo.action_unknown': 'Action #{id}',
    'quo.action_start': 'Game start',
    'quo.move_to': 'move to ({row}, {col})',
    'quo.place_h': 'place horizontal wall @ ({row}, {col})',
    'quo.place_v': 'place vertical wall @ ({row}, {col})',
    'quo.player_a': 'A (black)',
    'quo.player_b': 'B (red)',
    'quo.walls_default': 'Walls left: --',
    'quo.walls_count': 'Walls left: A={a} / B={b}',
  },
});

const BOARD_SIZE = 9;

function rcKey(r, c) { return r + ',' + c; }

function formatMove(info, actionId) {
  if (!info) {
    if (actionId !== null && actionId !== undefined) return t('quo.action_unknown', { id: actionId });
    return t('quo.action_start');
  }
  const type = String(info.type || '');
  const row = Number(info.row != null ? info.row : -1);
  const col = Number(info.col != null ? info.col : -1);
  if (type === 'move') return t('quo.move_to', { row: row + 1, col: col + 1 });
  if (type === 'hwall') return t('quo.place_h', { row: row + 1, col: col + 1 });
  if (type === 'vwall') return t('quo.place_v', { row: row + 1, col: col + 1 });
  if (actionId !== null && actionId !== undefined) return t('quo.action_unknown', { id: actionId });
  return t('quo.action_start');
}

function buildLegalMaps(legalActions) {
  const moves = new Map(), hwalls = new Map(), vwalls = new Map();
  if (!legalActions) return { moves, hwalls, vwalls };
  for (const actionId of legalActions) {
    if (actionId < 81) {
      moves.set(rcKey(Math.floor(actionId / 9), actionId % 9), actionId);
    } else if (actionId < 145) {
      const x = actionId - 81;
      hwalls.set(rcKey(Math.floor(x / 8), x % 8), actionId);
    } else if (actionId < 209) {
      const y = actionId - 145;
      vwalls.set(rcKey(Math.floor(y / 8), y % 8), actionId);
    }
  }
  return { moves, hwalls, vwalls };
}

function buildWallAnchorSets(st) {
  const hAnchors = new Set(), vAnchors = new Set();
  if (!st) return { hAnchors, vAnchors };
  for (const w of (st.horizontal_walls || [])) hAnchors.add(rcKey(w.row, w.col));
  for (const w of (st.vertical_walls || [])) vAnchors.add(rcKey(w.row, w.col));
  return { hAnchors, vAnchors };
}

function renderGridTo(boardEl, st, canPlay, legal, wallOwners) {
  boardEl.innerHTML = '';
  if (!st) return;

  const { hAnchors, vAnchors } = buildWallAnchorSets(st);
  const pawnByCell = new Map();
  for (const p of (st.pawns || [])) pawnByCell.set(rcKey(p.row, p.col), p.player);

  let submitFn = null;

  for (let gr = 0; gr < 17; gr++) {
    for (let gc = 0; gc < 17; gc++) {
      if (gr % 2 === 0 && gc % 2 === 0) {
        const row = gr / 2, col = gc / 2;
        const key = rcKey(row, col);
        // <div role="button"> instead of <button>: some Android
        // Chromium webviews give <button> an intrinsic vertical content
        // offset (a fraction of a pixel from UA baseline metrics) that
        // no amount of line-height/appearance reset can remove, pushing
        // any centered child visually below the cell's geometric center.
        const cell = document.createElement('div');
        cell.setAttribute('role', 'button');
        cell.setAttribute('tabindex', '-1');
        cell.className = 'board-cell';
        cell.setAttribute('data-cell', row + '-' + col);
        const actionId = legal.moves.get(key);
        if (canPlay && actionId != null) {
          cell.classList.add('legal');
          cell.dataset.action = actionId;
        }
        const pawn = pawnByCell.get(key);
        if (pawn != null) {
          const piece = document.createElement('div');
          piece.className = 'pawn ' + (pawn === 0 ? 'p0' : 'p1');
          piece.setAttribute('data-pawn', String(pawn));
          piece.textContent = pawn === 0 ? 'A' : 'B';
          cell.appendChild(piece);
        }
        boardEl.appendChild(cell);
      } else if (gr % 2 === 1 && gc % 2 === 0) {
        const hr = (gr - 1) / 2, hc = gc / 2;
        const key = rcKey(hr, hc), keyLeft = rcKey(hr, hc - 1);
        const el = document.createElement('div');
        el.setAttribute('role', 'button');
        el.setAttribute('tabindex', '-1');
        el.className = 'edge-slot edge-h';
        el.setAttribute('data-edge', 'h-' + hr + '-' + hc);
        if (hAnchors.has(key)) {
          el.classList.add('wall-anchor', 'wall-on');
          const owner = wallOwners && wallOwners.h.get(key);
          if (owner === 0) el.classList.add('wall-p0');
          else if (owner === 1) el.classList.add('wall-p1');
        } else if (hAnchors.has(keyLeft)) {
          el.classList.add('wall-tail');
        } else if (hc <= BOARD_SIZE - 1) {
          let actionId = legal.hwalls.get(key);
          if (actionId == null) actionId = legal.hwalls.get(keyLeft);
          if (canPlay && actionId != null) {
            el.classList.add('legal');
            el.dataset.action = actionId;
          } else {
            el.classList.add('edge-disabled');
          }
        } else {
          el.classList.add('edge-disabled');
        }
        boardEl.appendChild(el);
      } else if (gr % 2 === 0 && gc % 2 === 1) {
        const vr = gr / 2, vc = (gc - 1) / 2;
        const key = rcKey(vr, vc), keyUp = rcKey(vr - 1, vc);
        const el = document.createElement('div');
        el.setAttribute('role', 'button');
        el.setAttribute('tabindex', '-1');
        el.className = 'edge-slot edge-v';
        el.setAttribute('data-edge', 'v-' + vr + '-' + vc);
        if (vAnchors.has(key)) {
          el.classList.add('wall-anchor', 'wall-on');
          const owner = wallOwners && wallOwners.v.get(key);
          if (owner === 0) el.classList.add('wall-p0');
          else if (owner === 1) el.classList.add('wall-p1');
        } else if (vAnchors.has(keyUp)) {
          el.classList.add('wall-tail');
        } else if (vr <= BOARD_SIZE - 1) {
          let actionId = legal.vwalls.get(key);
          if (actionId == null) actionId = legal.vwalls.get(keyUp);
          if (canPlay && actionId != null) {
            el.classList.add('legal');
            el.dataset.action = actionId;
          } else {
            el.classList.add('edge-disabled');
          }
        } else {
          el.classList.add('edge-disabled');
        }
        boardEl.appendChild(el);
      } else {
        const cross = document.createElement('div');
        cross.className = 'board-cross';
        boardEl.appendChild(cross);
      }
    }
  }
}

let wallOwners = { h: new Map(), v: new Map() };

function buildWallOwnersFromFrames(frames, upToIndex) {
  const hOwners = new Map(), vOwners = new Map();
  const limit = upToIndex != null ? upToIndex + 1 : frames.length;
  for (let i = 0; i < limit; i++) {
    const f = frames[i];
    const info = f.action_info;
    if (!info || (info.type !== 'hwall' && info.type !== 'vwall')) continue;
    let actor = -1;
    if (f.actor === 'player_0') actor = 0;
    else if (f.actor === 'player_1') actor = 1;
    if (actor < 0) continue;
    const key = rcKey(info.row, info.col);
    if (info.type === 'hwall') hOwners.set(key, actor);
    else vOwners.set(key, actor);
  }
  return { h: hOwners, v: vOwners };
}

let replayFramesRef = null;

async function refreshWallOwners(sessionId, aiPlayer) {
  try {
    const data = await apiGet(API_BASE + '/' + sessionId + '/replay');
    wallOwners = buildWallOwnersFromFrames(data.frames, null);
  } catch (e) { /* keep previous */ }
}

function renderEmptyGrid(boardEl) {
  boardEl.innerHTML = '';
  for (let gr = 0; gr < 17; gr++) {
    for (let gc = 0; gc < 17; gc++) {
      if (gr % 2 === 0 && gc % 2 === 0) {
        const cell = document.createElement('div');
        cell.className = 'board-cell';
        boardEl.appendChild(cell);
      } else if (gr % 2 === 1 && gc % 2 === 1) {
        const cross = document.createElement('div');
        cross.className = 'board-cross';
        boardEl.appendChild(cross);
      } else {
        const el = document.createElement('div');
        el.className = 'edge-slot ' + (gr % 2 === 1 ? 'edge-h' : 'edge-v') + ' edge-disabled';
        boardEl.appendChild(el);
      }
    }
  }
}

// After a grid renders with a CSS-derived fractional `--slot` (common on
// mobile: `(100vw - 40px) / 17 ≈ 19.7px`), CSS Grid floors each rendered
// track to integer device pixels while `var(--slot)` keeps the fractional
// value. Any downstream `calc(var(--slot) * N)` then drifts out of sync
// with grid track boundaries — visibly enough on mobile to put legal-move
// dots on the wrong row. Fix: measure the actual floored track width
// post-render and write it back as the integer-pixel `--slot`, so
// `var(--slot)` and rendered tracks agree exactly.
function alignSlotToRenderedTrack(boardEl) {
  const firstCell = boardEl.querySelector('.board-cell');
  if (!firstCell) return;
  // Clear any previous inline override so the CSS-derived slot applies
  // at the current viewport. Accessing `offsetWidth` forces a reflow so
  // the subsequent measurement reads the re-derived track width.
  boardEl.style.removeProperty('--slot');
  // eslint-disable-next-line no-unused-expressions
  boardEl.offsetWidth;
  const rect = firstCell.getBoundingClientRect();
  if (rect.width <= 0) return;
  const px = Math.floor(rect.width);
  // Only override if the CSS-derived width was fractional; pinning to
  // an integer makes `var(--slot) * N` and grid tracks agree and stops
  // subpixel drift from misaligning legal-action highlights on mobile.
  if (Math.abs(rect.width - px) > 0.01) {
    boardEl.style.setProperty('--slot', px + 'px');
  }
}

let resizeAlignTimer = null;
window.addEventListener('resize', () => {
  if (resizeAlignTimer) cancelAnimationFrame(resizeAlignTimer);
  resizeAlignTimer = requestAnimationFrame(() => {
    const boardEl = document.querySelector('.quoridor-board-grid');
    if (boardEl) alignSlotToRenderedTrack(boardEl);
  });
});

function renderBoard(container, gameState, ctx) {
  container.innerHTML = '';

  const wrapper = document.createElement('div');
  wrapper.className = 'quoridor-board-wrap';

  const boardEl = document.createElement('div');
  boardEl.className = 'quoridor-board-grid';

  if (!gameState || !gameState.state) {
    renderEmptyGrid(boardEl);
    wrapper.appendChild(boardEl);
    container.appendChild(wrapper);
    requestAnimationFrame(() => alignSlotToRenderedTrack(boardEl));
    return;
  }

  const gs = gameState;
  const canPlay = ctx.canPlay;
  const legal = buildLegalMaps(gs.legal_actions);

  let owners = wallOwners;
  if (gs.ply_index != null && replayFramesRef) {
    owners = buildWallOwnersFromFrames(replayFramesRef, gs.ply_index);
  }

  renderGridTo(boardEl, gs.state, canPlay, legal, owners);

  boardEl.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-action]');
    if (btn) ctx.submitAction(Number(btn.dataset.action));
  });

  wrapper.appendChild(boardEl);
  container.appendChild(wrapper);
  requestAnimationFrame(() => alignSlotToRenderedTrack(boardEl));

  if (!gs.ply_index && ctx.state.sessionId) {
    refreshWallOwners(ctx.state.sessionId, ctx.state.aiPlayer);
  }
}

function describeTransition(prevState, newState, actionInfo, actionId) {
  if (!actionInfo || !prevState || !prevState.state) return null;

  const steps = [];
  const type = actionInfo.type;
  const actor = prevState.current_player;

  if (type === 'move') {
    const oldPawn = (prevState.state.pawns || []).find(p => p.player === actor);
    if (oldPawn) {
      // Resolve the live pawn size so the flyer is proportional to the
      // current --slot. If the lookup fails for any reason, fall back to
      // a reasonable default instead of letting animate.js pick the
      // (non-square) wrap rect.
      const livePawn = document.querySelector('[data-pawn="' + actor + '"]');
      const rect = livePawn ? livePawn.getBoundingClientRect() : null;
      const size = rect ? Math.round(Math.min(rect.width, rect.height)) : 28;
      steps.push({
        type: 'fly',
        from: '[data-pawn="' + actor + '"]',
        to: '[data-cell="' + actionInfo.row + '-' + actionInfo.col + '"]',
        width: size,
        height: size,
        createElement() {
          const el = document.createElement('div');
          el.className = 'anim-flying-token pawn-flying';
          el.style.background = actor === 0 ? '#1f2937' : '#b91c1c';
          el.style.color = '#fff';
          el.style.fontWeight = '700';
          el.textContent = actor === 0 ? 'A' : 'B';
          return el;
        },
        duration: 400,
        hideFrom: true,
      });
    }
  } else if (type === 'hwall' || type === 'vwall') {
    const dir = type === 'hwall' ? 'h' : 'v';
    steps.push({
      type: 'highlight',
      target: '[data-edge="' + dir + '-' + actionInfo.row + '-' + actionInfo.col + '"]',
      className: 'wall-placed-anim',
      duration: 600,
    });
  }

  return steps.length ? steps : null;
}

const wallsExtension = {
  render(el, gameState) {
    if (!gameState || !gameState.state) {
      el.textContent = t('quo.walls_default');
      return;
    }
    const wr = gameState.state.walls_remaining || [0, 0];
    el.textContent = t('quo.walls_count', { a: wr[0], b: wr[1] });
  }
};

createApp({
  gameId: 'quoridor',
  gameTitle: t('quo.title'),
  gameIntro: t('quo.intro'),
  players: { min: 2, max: 2 },
  renderBoard,
  describeTransition,
  formatOpponentMove: formatMove,
  formatSuggestedMove: formatMove,
  getPlayerSymbol: (humanPlayer) => humanPlayer === 0 ? t('quo.player_a') : t('quo.player_b'),
  onReplayFrames(frames) { replayFramesRef = frames; },
  onGameStart() { replayFramesRef = null; },
  extensions: [wallsExtension],
  difficulties: ['heuristic', 'casual', 'expert'],
  defaultDifficulty: 'expert',
  defaultSeat: 'random',
});
