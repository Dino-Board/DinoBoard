import { createApp } from '/static/general/app.js';
import { t, register } from '/static/general/i18n.js';

register({
  zh: {
    'carc.intro': '旋转当前板块，然后点击一个合法格子放置。',
    'carc.player_0': '蓝方',
    'carc.player_1': '红方',
    'carc.score': '分',
    'carc.meeples': '米宝',
    'carc.tiles_left': '牌堆',
    'carc.current_tile': '当前板块',
    'carc.rotate': '方向',
    'carc.rotate_ccw': '逆时针旋转',
    'carc.rotate_cw': '顺时针旋转',
    'carc.stage_place': '放置板块',
    'carc.stage_meeple': '放置米宝',
    'carc.status_place': '选择一个合法格子放置板块',
    'carc.status_meeple': '在放大板块上选择米宝位置，或跳过',
    'carc.pass_meeple': '跳过米宝',
    'carc.place_tile': '在 ({row}, {col}) 放置板块，旋转 {rot}',
    'carc.place_meeple': '放置米宝',
    'carc.meeple_pass': '不放米宝',
    'carc.unknown_action': '动作 #{id}',
    'carc.final_score': '+{score}',
  },
  en: {
    'carc.intro': 'Rotate the current tile, then click a legal square to place it.',
    'carc.player_0': 'Blue',
    'carc.player_1': 'Red',
    'carc.score': 'score',
    'carc.meeples': 'meeples',
    'carc.tiles_left': 'deck',
    'carc.current_tile': 'Current tile',
    'carc.rotate': 'Rotation',
    'carc.rotate_ccw': 'Rotate counterclockwise',
    'carc.rotate_cw': 'Rotate clockwise',
    'carc.stage_place': 'Place tile',
    'carc.stage_meeple': 'Place meeple',
    'carc.status_place': 'Choose a legal square for the tile',
    'carc.status_meeple': 'Choose a meeple position on the enlarged tile, or skip',
    'carc.pass_meeple': 'Skip meeple',
    'carc.place_tile': 'Place tile at ({row}, {col}), rotation {rot}',
    'carc.place_meeple': 'Place meeple',
    'carc.meeple_pass': 'No meeple',
    'carc.unknown_action': 'Action #{id}',
    'carc.final_score': '+{score}',
  },
});

const ROWS = 20;
const COLS = 20;
const PLACE_ACTION_COUNT = ROWS * COLS * 4;
const MEEPLE_PASS_ACTION = PLACE_ACTION_COUNT;
const MEEPLE_SIDE_ACTION_BASE = MEEPLE_PASS_ACTION + 1;
const MEEPLE_FARM_ACTION_BASE = MEEPLE_SIDE_ACTION_BASE + 4;
const MEEPLE_CENTER_ACTION = MEEPLE_FARM_ACTION_BASE + 8;
const TILE_CODES = 'ABCDEFGHIJKLMNOPQRSTUVWX'.split('');
const PLAYER_CLASS = ['p0', 'p1'];
const STAGE_PLACE = 0;
const STAGE_MEEPLE = 1;
const TILE_FLY_SIZE = 58;
const MEEPLE_FLY_SIZE = 24;

const FARM_GROUPS = [
  [[0, 1, 2, 3, 4, 5, 6, 7]],
  [[0, 1, 2, 3, 4, 5, 6, 7]],
  [],
  [[0, 3], [4, 5, 6, 7]],
  [[0, 3, 4, 5, 6, 7]],
  [[1, 2], [5, 6]],
  [[1, 2], [5, 6]],
  [[1, 2, 5, 6]],
  [[0, 5, 6, 7]],
  [[0, 3, 6, 7], [4, 5]],
  [[0, 3, 4, 5], [6, 7]],
  [[0, 3], [6, 7], [4, 5]],
  [[0, 5, 6, 7]],
  [[0, 5, 6, 7]],
  [[3, 6], [4, 5]],
  [[3, 6], [4, 5]],
  [[5, 6]],
  [[5, 6]],
  [[6], [5]],
  [[6], [5]],
  [[0, 1, 6, 7], [2, 3, 4, 5]],
  [[6, 7], [0, 1, 2, 3, 4, 5]],
  [[6, 7], [4, 5], [0, 1, 2, 3]],
  [[6, 7], [4, 5], [0, 1], [2, 3]],
];

const TILE_MEEPLE_ANCHORS = [
  {
    featureGroups: [{ sides: [2], point: [77.5, 81.3] }],
    farmGroups: [[17.9, 76.7]],
    center: [50, 43],
  },
  {
    featureGroups: [],
    farmGroups: [[15.8, 82.5]],
    center: [50, 53],
  },
  {
    featureGroups: [{ sides: [0, 1, 2, 3], point: [50, 51] }],
    farmGroups: [],
  },
  {
    featureGroups: [
      { sides: [0], point: [50, 17] },
      { sides: [1, 3], point: [50, 57] },
    ],
    farmGroups: [[15.8, 32.1], [22.5, 83.3]],
  },
  {
    featureGroups: [{ sides: [0], point: [50, 19] }],
    farmGroups: [[50, 67]],
  },
  {
    featureGroups: [{ sides: [1, 3], point: [50, 51] }],
    farmGroups: [[51.7, 12.9], [51.2, 91.7]],
  },
  {
    featureGroups: [{ sides: [1, 3], point: [50, 51] }],
    farmGroups: [[50.8, 12.9], [50.4, 92.5]],
  },
  {
    featureGroups: [
      { sides: [1], point: [85, 51] },
      { sides: [3], point: [15, 51] },
    ],
    farmGroups: [[50, 50]],
  },
  {
    featureGroups: [
      { sides: [0], point: [52.1, 15.8] },
      { sides: [1], point: [89.6, 54.6] },
    ],
    farmGroups: [[36, 66]],
  },
  {
    featureGroups: [
      { sides: [0], point: [50, 19] },
      { sides: [1, 2], point: [40.8, 62.5] },
    ],
    farmGroups: [[15.8, 47.1], [82, 82]],
  },
  {
    featureGroups: [
      { sides: [0], point: [50, 19] },
      { sides: [2, 3], point: [55, 55] },
    ],
    farmGroups: [[85, 50.4], [22.5, 74.6]],
  },
  {
    featureGroups: [
      { sides: [0], point: [49.6, 12.5] },
      { sides: [1], point: [86.7, 44.6] },
      { sides: [2], point: [46.7, 85] },
      { sides: [3], point: [13.3, 48.8] },
    ],
    farmGroups: [[50, 33], [14.2, 78.3], [82.1, 80]],
  },
  {
    featureGroups: [{ sides: [0, 1], point: [63, 36] }],
    farmGroups: [[31, 73]],
  },
  {
    featureGroups: [{ sides: [0, 1], point: [63, 36] }],
    farmGroups: [[31, 73]],
  },
  {
    featureGroups: [
      { sides: [0, 3], point: [35, 36] },
      { sides: [1, 2], point: [67, 65] },
    ],
    farmGroups: [[36.3, 80.8], [85.8, 81.7]],
  },
  {
    featureGroups: [
      { sides: [0, 3], point: [35, 36] },
      { sides: [1, 2], point: [72.5, 66.3] },
    ],
    farmGroups: [[37.1, 82.1], [87.5, 83.3]],
  },
  {
    featureGroups: [{ sides: [0, 1, 3], point: [50, 36] }],
    farmGroups: [[48.3, 85.8]],
  },
  {
    featureGroups: [{ sides: [0, 1, 3], point: [50, 36] }],
    farmGroups: [[50, 86.7]],
  },
  {
    featureGroups: [
      { sides: [0, 1, 3], point: [50, 36] },
      { sides: [2], point: [51.7, 83.8] },
    ],
    farmGroups: [[31.7, 88.8], [72.9, 89.6]],
  },
  {
    featureGroups: [
      { sides: [0, 1, 3], point: [50, 36] },
      { sides: [2], point: [47.5, 84.6] },
    ],
    farmGroups: [[22.5, 89.6], [74.6, 90.4]],
  },
  {
    featureGroups: [{ sides: [0, 2], point: [50, 51] }],
    farmGroups: [[19.2, 64.2], [76, 53]],
  },
  {
    featureGroups: [{ sides: [2, 3], point: [37, 67] }],
    farmGroups: [[14.6, 78.3], [67, 35]],
  },
  {
    featureGroups: [
      { sides: [1], point: [85, 56.7] },
      { sides: [2], point: [57.9, 87.5] },
      { sides: [3], point: [10.8, 51.7] },
    ],
    farmGroups: [[14.6, 87.1], [86.3, 86.3], [71.3, 17.5]],
  },
  {
    featureGroups: [
      { sides: [0], point: [50.4, 12.1] },
      { sides: [1], point: [86.3, 52.9] },
      { sides: [2], point: [50.8, 86.7] },
      { sides: [3], point: [11.7, 52.9] },
    ],
    farmGroups: [[13.8, 87.9], [84.6, 86.3], [14.6, 18.8], [83.3, 17.9]],
  },
];

let selectedRotation = 0;
let selectedRotationTile = null;

function assertAnchorTable() {
  if (TILE_MEEPLE_ANCHORS.length !== TILE_CODES.length) {
    throw new Error('carcassonne tile meeple anchor table count mismatch');
  }
  for (let tile = 0; tile < TILE_MEEPLE_ANCHORS.length; tile++) {
    const anchors = TILE_MEEPLE_ANCHORS[tile];
    if (!anchors || !Array.isArray(anchors.featureGroups) ||
        !Array.isArray(anchors.farmGroups)) {
      throw new Error(`carcassonne missing meeple anchors for tile ${tile}`);
    }
    const farmGroups = FARM_GROUPS[tile];
    if (!farmGroups) {
      throw new Error(`carcassonne missing farm groups for tile ${tile}`);
    }
    if (anchors.farmGroups.length !== farmGroups.length) {
      throw new Error(`carcassonne farm anchor count mismatch for tile ${tile}`);
    }
  }
}

assertAnchorTable();

function rotatePoint(point, rotation) {
  let x = point[0];
  let y = point[1];
  for (let i = 0; i < rotation; i++) {
    const nextX = 100 - y;
    y = x;
    x = nextX;
  }
  return [x, y];
}

function unrotatedSide(side, rotation) {
  return (side - rotation + 4) % 4;
}

function rotatedFarmerSide(farmerSide, rotation) {
  const turn = [2, 3, 4, 5, 6, 7, 0, 1];
  let out = farmerSide;
  for (let i = 0; i < rotation; i++) out = turn[out];
  return out;
}

function unrotatedFarmerSide(farmerSide, rotation) {
  return rotatedFarmerSide(farmerSide, (4 - rotation) % 4);
}

function rotatedFeatureSide(side, rotation) {
  return (side + rotation) % 4;
}

function tileHasChapel(tile) {
  return tile === 0 || tile === 1;
}

function featureTargetPoint(tile, rotation, side) {
  const anchors = TILE_MEEPLE_ANCHORS[tile];
  if (!anchors) {
    throw new Error(`carcassonne missing meeple anchors for tile ${tile}`);
  }
  const originalSide = unrotatedSide(side, rotation);
  for (const group of anchors.featureGroups) {
    if (group.sides.includes(originalSide)) {
      return rotatePoint(group.point, rotation);
    }
  }
  throw new Error(`carcassonne missing feature target for tile ${tile} side ${side}`);
}

function farmTargetPoint(tile, rotation, farmerSide) {
  const groups = FARM_GROUPS[tile];
  if (!groups) {
    throw new Error(`carcassonne missing farm groups for tile ${tile}`);
  }
  const anchors = TILE_MEEPLE_ANCHORS[tile];
  if (!anchors) {
    throw new Error(`carcassonne missing meeple anchors for tile ${tile}`);
  }
  const originalFarmerSide = unrotatedFarmerSide(farmerSide, rotation);
  for (let i = 0; i < groups.length; i++) {
    if (!groups[i].includes(originalFarmerSide)) continue;
    const point = anchors.farmGroups[i];
    if (!point) {
      throw new Error(`carcassonne missing farm anchor for tile ${tile} group ${i}`);
    }
    return rotatePoint(point, rotation);
  }
  throw new Error(`carcassonne missing farm target for tile ${tile} side ${farmerSide}`);
}

function meeplePoint(tile, rotation, pos) {
  if (pos >= 0 && pos < 4) {
    return featureTargetPoint(tile, rotation, pos);
  }
  if (pos >= 4 && pos < 12) {
    return farmTargetPoint(tile, rotation, pos - 4);
  }
  if (pos === 12) {
    const anchors = TILE_MEEPLE_ANCHORS[tile];
    if (!anchors || !anchors.center) {
      throw new Error(`carcassonne missing center target for tile ${tile}`);
    }
    return rotatePoint(anchors.center, rotation);
  }
  throw new Error(`carcassonne invalid meeple position ${pos}`);
}

function meepleKind(pos) {
  if (pos >= 0 && pos < 4) return 'feature';
  if (pos >= 4 && pos < 12) return 'farm';
  if (pos === 12) return 'center';
  throw new Error(`carcassonne invalid meeple position ${pos}`);
}

function tileUrl(tile) {
  if (tile == null || tile < 0) return '';
  return `assets/Base_Game_C2_Tile_${TILE_CODES[tile]}.png`;
}

function placeAction(row, col, rotation) {
  return (row * COLS + col) * 4 + rotation;
}

// Quarter turns after which a tile looks the same again (B, C, X: 1;
// F, G, H, U: 2). The engine only accepts the lowest equivalent rotation,
// so the rotation picked in the UI is reduced before it becomes an action.
const ROTATION_PERIOD = { 1: 1, 2: 1, 23: 1, 5: 2, 6: 2, 7: 2, 20: 2 };

function canonicalRotation(tile, rotation) {
  return rotation % (ROTATION_PERIOD[tile] || 4);
}

function meepleAction(pos) {
  if (pos >= 0 && pos < 4) return MEEPLE_SIDE_ACTION_BASE + pos;
  if (pos >= 4 && pos < 12) return MEEPLE_FARM_ACTION_BASE + pos - 4;
  if (pos === 12) return MEEPLE_CENTER_ACTION;
  return MEEPLE_PASS_ACTION;
}

function classifyAction(actionId) {
  if (actionId == null || actionId < 0) return null;
  if (actionId === MEEPLE_PASS_ACTION) return { type: 'meeple_pass' };
  if (actionId > MEEPLE_PASS_ACTION && actionId <= MEEPLE_CENTER_ACTION) {
    let position = 12;
    if (actionId < MEEPLE_SIDE_ACTION_BASE + 4) {
      position = actionId - MEEPLE_SIDE_ACTION_BASE;
    } else if (actionId < MEEPLE_FARM_ACTION_BASE + 8) {
      position = 4 + actionId - MEEPLE_FARM_ACTION_BASE;
    }
    return { type: 'place_meeple', position };
  }
  const cell = Math.floor(actionId / 4);
  return {
    type: 'place_tile',
    row: Math.floor(cell / COLS),
    col: cell % COLS,
    rotation: actionId % 4,
  };
}

function moveText(info, actionId) {
  const data = info && info.type ? info : classifyAction(actionId);
  if (!data) {
    return t('carc.unknown_action', { id: actionId });
  }
  if (data.type === 'meeple_pass') return t('carc.meeple_pass');
  if (data.type === 'place_meeple') return t('carc.place_meeple');
  if (data.type !== 'place_tile') return t('carc.unknown_action', { id: actionId });
  return t('carc.place_tile', {
    row: data.row ?? 0,
    col: data.col ?? 0,
    rot: `${(data.rotation ?? 0) * 90}°`,
  });
}

function makeTile(tile, rotation, className = '') {
  const el = document.createElement('div');
  el.className = `carc-tile ${className}`;
  if (tile >= 0) {
    const img = document.createElement('img');
    img.className = 'carc-tile-art';
    img.src = tileUrl(tile);
    img.alt = TILE_CODES[tile];
    img.style.transform = `rotate(${rotation * 90}deg)`;
    el.appendChild(img);
  }
  return el;
}

function cloneCurrentTileFlyer() {
  const current = document.querySelector('[data-anim="current-tile"]');
  if (!current) {
    throw new Error('carcassonne current tile animation source missing');
  }
  const flyer = current.cloneNode(true);
  flyer.classList.remove('preview');
  flyer.classList.add('flying');
  flyer.removeAttribute('data-anim');
  return flyer;
}

function meepleUrl(owner) {
  if (owner === 0) return 'assets/blue_meeple.png';
  if (owner === 1) return 'assets/red_meeple.png';
  throw new Error(`carcassonne invalid meeple owner ${owner}`);
}

function makeMeepleSprite(owner, className = '') {
  const img = document.createElement('img');
  img.className = `carc-meeple-sprite ${owner === 0 ? 'p0' : 'p1'} ${className}`.trim();
  img.src = meepleUrl(owner);
  img.alt = '';
  return img;
}

function appendMeepleAnchors(tileEl, cell, tile, rotation) {
  if (cell == null || tile < 0) return;
  const positions = [];
  const anchors = TILE_MEEPLE_ANCHORS[tile];
  if (!anchors) {
    throw new Error(`carcassonne missing meeple anchors for tile ${tile}`);
  }
  for (const group of anchors.featureGroups) {
    for (const side of group.sides) {
      const pos = rotatedFeatureSide(side, rotation);
      if (!positions.includes(pos)) positions.push(pos);
    }
  }
  if (tileHasChapel(tile)) positions.push(12);
  const groups = FARM_GROUPS[tile];
  if (!groups) {
    throw new Error(`carcassonne missing farm groups for tile ${tile}`);
  }
  for (const group of groups) {
    for (const farmerSide of group) {
      const pos = 4 + rotatedFarmerSide(farmerSide, rotation);
      if (!positions.includes(pos)) positions.push(pos);
    }
  }
  for (const pos of positions) {
    const anchor = document.createElement('span');
    anchor.className = 'carc-meeple-anchor';
    anchor.dataset.meepleAnchor = `${cell}-${pos}`;
    const point = meeplePoint(tile, rotation, pos);
    anchor.style.left = `${point[0]}%`;
    anchor.style.top = `${point[1]}%`;
    tileEl.appendChild(anchor);
  }
}

function renderMeeple(owner, pos, tile, rotation, cell = null) {
  const kindClass = pos >= 4 && pos < 12 ? ' farm-meeple' : '';
  const marker = document.createElement('span');
  marker.className = 'carc-meeple-wrap';
  const sprite = makeMeepleSprite(owner, `carc-meeple${kindClass}`);
  marker.appendChild(sprite);
  const point = meeplePoint(tile, rotation, pos);
  marker.style.left = `${point[0]}%`;
  marker.style.top = `${point[1]}%`;
  if (cell != null) {
    marker.dataset.meepleCell = String(cell);
  }
  return marker;
}

function appendFinalMeepleScore(layer, row, col, tile, rotation, pos, score) {
  const marker = document.createElement('span');
  marker.className = 'carc-final-meeple-score-marker';
  marker.style.gridRow = `${row + 1}`;
  marker.style.gridColumn = `${col + 1}`;
  const point = meeplePoint(tile, rotation, pos);
  const badge = document.createElement('span');
  badge.className = 'carc-final-meeple-score';
  badge.textContent = t('carc.final_score', { score });
  badge.style.left = `${point[0]}%`;
  badge.style.top = `${point[1]}%`;
  marker.appendChild(badge);
  layer.appendChild(marker);
}

function finalMeepleScore(game, cell) {
  if (!Array.isArray(game.final_meeple_scores)) {
    throw new Error('carcassonne missing final meeple score field');
  }
  if (cell < 0 || cell >= game.final_meeple_scores.length) {
    throw new Error(`carcassonne final meeple score cell out of range ${cell}`);
  }
  return game.final_meeple_scores[cell];
}

function renderMeepleTargets(tileEl, legalSet, ctx, tile, rotation) {
  for (let pos = 0; pos <= 12; pos++) {
    const action = meepleAction(pos);
    if (!legalSet.has(action)) continue;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `carc-meeple-target kind-${meepleKind(pos)} pos-${pos}`;
    btn.setAttribute('aria-label', t('carc.place_meeple'));
    const point = meeplePoint(tile, rotation, pos);
    btn.style.left = `${point[0]}%`;
    btn.style.top = `${point[1]}%`;
    btn.addEventListener('click', (ev) => {
      ev.stopPropagation();
      ctx.submitAction(action);
    });
    tileEl.appendChild(btn);
  }
}

function renderScorebar(root, gs) {
  const game = gs.state;
  const bar = document.createElement('div');
  bar.className = 'carc-scorebar';
  for (let p = 0; p < 2; p++) {
    const row = document.createElement('div');
    row.className = `carc-player-chip ${PLAYER_CLASS[p]} ${gs.current_player === p ? 'active' : ''}`;
    row.dataset.player = String(p);

    const name = document.createElement('span');
    name.textContent = p === 0 ? t('carc.player_0') : t('carc.player_1');
    const playerText = document.createElement('div');
    playerText.className = 'carc-player-text';
    const scoreLine = document.createElement('div');
    scoreLine.className = 'carc-score-line';
    const score = document.createElement('strong');
    if (gs.is_terminal) {
      if (!Array.isArray(game.final_score_deltas)) {
        throw new Error('carcassonne missing final score delta field');
      }
      const finalDelta = game.final_score_deltas[p];
      const beforeFinal = game.scores[p] - finalDelta;
      score.className = 'carc-final-score-breakdown';
      score.textContent = `${beforeFinal}+${finalDelta}`;
      scoreLine.append(score);
    } else {
      score.textContent = String(game.scores[p]);
      scoreLine.append(score);
    }
    const label = document.createElement('small');
    label.textContent = t('carc.score');
    scoreLine.append(label);
    playerText.append(name, scoreLine);

    const meeples = document.createElement('div');
    meeples.className = 'carc-meeple-supply';
    meeples.dataset.meepleSupply = String(p);
    const meepleIcon = makeMeepleSprite(p, 'supply-icon');
    meepleIcon.dataset.meepleSource = String(p);
    const meepleCount = document.createElement('span');
    meepleCount.dataset.meepleCount = String(p);
    meepleCount.textContent = `x${game.meeples_remaining[p]}`;
    const meepleLabel = document.createElement('small');
    meepleLabel.textContent = t('carc.meeples');
    meeples.append(meepleIcon, meepleCount, meepleLabel);

    row.append(playerText, meeples);
    bar.appendChild(row);
  }
  const deck = document.createElement('div');
  deck.className = 'carc-deck-count';
  deck.textContent = `${t('carc.tiles_left')}: ${game.tiles_remaining}`;
  bar.appendChild(deck);
  root.appendChild(bar);
}

function setSupplyCount(player, count) {
  const el = document.querySelector(`[data-meeple-count="${player}"]`);
  if (el) el.textContent = `x${count}`;
}

function makePlacedMeepleInDom(player, cell, pos, tile, rotation) {
  const tileEl = document.querySelector(`[data-anim="tile-${cell}"]`);
  if (!tileEl) return;
  const existing = tileEl.querySelector(`[data-meeple-cell="${cell}"]`);
  if (existing) existing.remove();
  const marker = renderMeeple(player, pos, tile, rotation, cell);
  marker.dataset.justPlaced = '1';
  tileEl.appendChild(marker);
}

function returnedMeeples(prev, next) {
  const prevOwners = prev?.board_meeple_owner || [];
  const nextOwners = next?.board_meeple_owner || [];
  const prevPos = prev?.board_meeple_pos || [];
  const prevTile = prev?.board_tile || [];
  const prevRot = prev?.board_rotation || [];
  const out = [];
  for (let cell = 0; cell < prevOwners.length; cell++) {
    const owner = prevOwners[cell];
    if (owner < 0 || nextOwners[cell] >= 0) continue;
    out.push({
      cell,
      owner,
      pos: prevPos[cell],
      tile: prevTile[cell],
      rotation: prevRot[cell] || 0,
    });
  }
  return out;
}

function scorePopupSteps(prev, next) {
  const steps = [];
  const before = prev?.scores || [];
  const after = next?.scores || [];
  for (let p = 0; p < Math.max(before.length, after.length); p++) {
    const gain = (after[p] || 0) - (before[p] || 0);
    if (gain <= 0) continue;
    steps.push({
      type: 'popup',
      target: `[data-player="${p}"]`,
      content: `+${gain}`,
      className: 'carc-popup',
      duration: 700,
      width: 64,
      height: 30,
    });
  }
  return steps;
}

function meepleReturnFlights(returns, nextMeeples) {
  return returns.map((ret) => ({
    from: `[data-meeple-cell="${ret.cell}"]`,
    to: `[data-meeple-source="${ret.owner}"]`,
    createElement: () => makeMeepleSprite(ret.owner, 'carc-meeple-flying'),
    width: MEEPLE_FLY_SIZE,
    height: MEEPLE_FLY_SIZE,
    duration: 360,
    onStart: (srcEl) => srcEl.remove(),
    onComplete: () => setSupplyCount(ret.owner, nextMeeples[ret.owner]),
  }));
}

function renderCurrentTile(root, gs, ctx, legalSet) {
  const game = gs.state;
  const panel = document.createElement('div');
  panel.className = 'carc-current';

  const title = document.createElement('div');
  title.className = 'carc-panel-title';
  title.textContent = game.stage === STAGE_MEEPLE
    ? t('carc.stage_meeple')
    : t('carc.current_tile');
  panel.appendChild(title);

  if (game.stage === STAGE_PLACE) {
    const tile = makeTile(game.current_tile, selectedRotation, 'preview');
    tile.dataset.anim = 'current-tile';
    panel.appendChild(tile);

    const rotWrap = document.createElement('div');
    rotWrap.className = 'carc-rotations';
    rotWrap.setAttribute('aria-label', t('carc.rotate'));
    for (const control of [
      {
        label: t('carc.rotate_ccw'),
        delta: 3,
        path: 'M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8 M3 3v5h5 M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16 M21 21v-5h-5',
      },
      {
        label: t('carc.rotate_cw'),
        delta: 1,
        path: 'M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8 M21 3v5h-5 M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16 M3 21v-5h5',
      },
    ]) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'carc-rotate-btn';
      btn.setAttribute('aria-label', control.label);
      btn.title = control.label;
      btn.innerHTML = `
        <svg class="carc-rotate-icon" viewBox="0 0 24 24" aria-hidden="true">
          <path d="${control.path}"></path>
        </svg>
      `;
      btn.addEventListener('click', () => {
        selectedRotation = (selectedRotation + control.delta) % 4;
        ctx.rerender();
      });
      rotWrap.appendChild(btn);
    }
    panel.appendChild(rotWrap);
  } else if (game.stage === STAGE_MEEPLE && game.last_row >= 0 && game.last_col >= 0) {
    const cell = game.last_row * COLS + game.last_col;
    const tile = makeTile(
      game.board_tile[cell],
      game.board_rotation[cell] || 0,
      'preview meeple-preview',
    );
    tile.dataset.anim = `meeple-preview-${cell}`;
    const owner = game.board_meeple_owner[cell];
    const pos = game.board_meeple_pos[cell];
    const placedTile = game.board_tile[cell];
    const placedRotation = game.board_rotation[cell] || 0;
    if (owner >= 0) tile.appendChild(
      renderMeeple(owner, pos, placedTile, placedRotation, cell));
    if (ctx.canPlay) {
      renderMeepleTargets(tile, legalSet, ctx, placedTile, placedRotation);
    }
    panel.appendChild(tile);
  }
  root.appendChild(panel);
}

function renderBoard(container, gs, ctx) {
  container.innerHTML = '';
  if (!gs) return;

  const game = gs.state;
  if (game.stage === STAGE_PLACE && game.current_tile !== selectedRotationTile) {
    selectedRotation = 0;
    selectedRotationTile = game.current_tile;
  }
  const legalSet = new Set(gs.legal_actions || []);
  const root = document.createElement('div');
  root.className = 'carc-root';
  renderScorebar(root, gs);

  const main = document.createElement('div');
  main.className = 'carc-main';

  const side = document.createElement('div');
  side.className = 'carc-side';
  renderCurrentTile(side, gs, ctx, legalSet);
  const status = document.createElement('div');
  status.className = 'carc-status';
  const statusLine = document.createElement('div');
  statusLine.textContent = game.stage === STAGE_MEEPLE
    ? t('carc.status_meeple')
    : t('carc.status_place');
  status.appendChild(statusLine);
  if (game.stage === STAGE_MEEPLE && ctx.canPlay && legalSet.has(MEEPLE_PASS_ACTION)) {
    const pass = document.createElement('button');
    pass.type = 'button';
    pass.className = 'carc-pass';
    pass.textContent = t('carc.pass_meeple');
    pass.addEventListener('click', () => ctx.submitAction(MEEPLE_PASS_ACTION));
    status.appendChild(pass);
  }
  side.appendChild(status);
  main.appendChild(side);

  const boardShell = document.createElement('div');
  boardShell.className = 'carc-board-shell';
  const grid = document.createElement('div');
  grid.className = 'carc-grid';
  const finalScoreLayer = document.createElement('div');
  finalScoreLayer.className = 'carc-final-score-layer';

  const minR = 0;
  const maxR = ROWS - 1;
  const minC = 0;
  const maxC = COLS - 1;
  grid.style.gridTemplateRows = `repeat(${maxR - minR + 1}, var(--carc-cell))`;
  grid.style.gridTemplateColumns = `repeat(${maxC - minC + 1}, var(--carc-cell))`;
  finalScoreLayer.style.gridTemplateRows = grid.style.gridTemplateRows;
  finalScoreLayer.style.gridTemplateColumns = grid.style.gridTemplateColumns;

  for (let row = minR; row <= maxR; row++) {
    for (let col = minC; col <= maxC; col++) {
      const cell = row * COLS + col;
      const tile = game.board_tile[cell];
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'carc-cell';
      btn.dataset.row = String(row);
      btn.dataset.col = String(col);
      btn.dataset.cell = String(cell);

      if (tile >= 0) {
        btn.classList.add('occupied');
        const tileEl = makeTile(tile, game.board_rotation[cell] || 0);
        tileEl.dataset.anim = `tile-${cell}`;
        appendMeepleAnchors(tileEl, cell, tile, game.board_rotation[cell] || 0);
        const owner = game.board_meeple_owner[cell];
        const pos = game.board_meeple_pos[cell];
        if (owner >= 0) {
          const meeple = renderMeeple(owner, pos, tile, game.board_rotation[cell] || 0, cell);
          if (gs.is_terminal) {
            appendFinalMeepleScore(
              finalScoreLayer,
              row - minR,
              col - minC,
              tile,
              game.board_rotation[cell] || 0,
              pos,
              finalMeepleScore(game, cell),
            );
          }
          tileEl.appendChild(meeple);
        }
        btn.appendChild(tileEl);
      } else {
        const action = placeAction(
          row, col, canonicalRotation(game.current_tile, selectedRotation));
        if (ctx.canPlay && game.stage === STAGE_PLACE && legalSet.has(action)) {
          btn.classList.add('legal');
          btn.appendChild(makeTile(game.current_tile, selectedRotation, 'ghost'));
          btn.addEventListener('click', () => ctx.submitAction(action));
        }
      }

      if (row === game.last_row && col === game.last_col) {
        btn.classList.add('last-placed');
      }
      grid.appendChild(btn);
    }
  }

  if (gs.is_terminal) {
    grid.appendChild(finalScoreLayer);
  }
  boardShell.appendChild(grid);
  main.appendChild(boardShell);
  root.appendChild(main);
  container.appendChild(root);
}

function describeTransition(prevState, nextState, actionInfo, actionId) {
  const info = actionInfo || classifyAction(actionId);
  if (!info) return [];
  if (!prevState || !prevState.state || !nextState || !nextState.state) return [];
  const prev = prevState.state;
  const next = nextState.state;
  if (info.type === 'meeple_pass' || info.type === 'place_meeple') {
    const actor = prevState.current_player;
    const cell = (prev.last_row * COLS) + prev.last_col;
    const steps = [];
    if (info.type === 'place_meeple') {
      const pos = info.position;
      const tile = prev.board_tile[cell];
      const rotation = prev.board_rotation[cell] || 0;
      steps.push({
        type: 'fly',
        from: `[data-meeple-source="${actor}"]`,
        to: `[data-meeple-anchor="${cell}-${pos}"]`,
        createElement: () => makeMeepleSprite(actor, 'carc-meeple-flying'),
        width: MEEPLE_FLY_SIZE,
        height: MEEPLE_FLY_SIZE,
        duration: 340,
        onStart: () => setSupplyCount(actor, prev.meeples_remaining[actor] - 1),
        onComplete: () => makePlacedMeepleInDom(actor, cell, pos, tile, rotation),
      });
    }

    const returnFlights = meepleReturnFlights(returnedMeeples(prev, next), next.meeples_remaining);
    if (info.type === 'place_meeple') {
      const pos = info.position;
      const newlyPlacedReturned = next.board_meeple_owner[cell] < 0;
      if (newlyPlacedReturned) {
        returnFlights.push({
          from: `[data-meeple-cell="${cell}"]`,
          to: `[data-meeple-source="${actor}"]`,
          createElement: () => makeMeepleSprite(actor, 'carc-meeple-flying'),
          width: MEEPLE_FLY_SIZE,
          height: MEEPLE_FLY_SIZE,
          duration: 360,
          onStart: (srcEl) => srcEl.remove(),
          onComplete: () => setSupplyCount(actor, next.meeples_remaining[actor]),
        });
      }
    }

    if (returnFlights.length) {
      steps.push({ type: 'flyGroup', flights: returnFlights });
    }
    steps.push(...scorePopupSteps(prev, next));
    return steps;
  }
  if (info.type !== 'place_tile') return [];
  const row = info.row;
  const col = info.col;
  const cell = row * COLS + col;
  const targetSelector = `[data-cell="${cell}"]`;
  const steps = [{
    type: 'fly',
    from: '[data-anim="current-tile"]',
    to: targetSelector,
    createElement: cloneCurrentTileFlyer,
    width: TILE_FLY_SIZE,
    height: TILE_FLY_SIZE,
    duration: 320,
    onStart: (srcEl) => {
      srcEl.style.visibility = 'hidden';
      const target = document.querySelector(targetSelector);
      const ghost = target ? target.querySelector('.carc-tile.ghost') : null;
      if (ghost) ghost.remove();
    },
  }];
  return steps;
}

createApp({
  gameId: 'carcassonne',
  players: { min: 2, max: 2 },
  difficulties: ['heuristic', 'casual', 'expert'],
  defaultDifficulty: 'heuristic',
  gameIntro: t('carc.intro'),
  renderBoard,
  describeTransition,
  getPlayerSymbol: (idx) => idx === 0 ? t('carc.player_0') : t('carc.player_1'),
  formatOpponentMove: moveText,
  formatSuggestedMove: moveText,
});
