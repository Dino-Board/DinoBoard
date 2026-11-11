import { createApp } from '/static/general/app.js';
import { t, tList, register } from '/static/general/i18n.js';

register({
  zh: {
    'harm.title': '自然和弦',
    'harm.intro': '从中央 5 个槽位选一个，把全部 token 取走。点击你的棋盘放置 token（按颜色一次一个）。可以拿动物卡，并在地形完成时放置动物方块。放地形、拿卡、放动物可穿插进行。',
    'harm.player_0': '玩家 0',
    'harm.player_1': '玩家 1',
    'harm.player_2': '玩家 2',
    'harm.player_3': '玩家 3',
    'harm.you': '我',
    'harm.opponent': '对手',
    'harm.score': '得分',
    'harm.animal_score': '动物',
    'harm.landscape_score': '地形',
    'harm.central': '中央槽位',
    'harm.market': '动物市场',
    'harm.held': '我的动物卡',
    'harm.hand': '本回合待放置 token',
    'harm.btn_end_turn': '结束回合',
    'harm.status_pick_slot': '请选择中央槽位',
    'harm.status_your_turn': '你的回合 — 放地形/拿卡/放动物',
    'harm.status_ai_thinking': 'AI 思考中…',
    'harm.status_win': '你赢了！',
    'harm.status_lose': '你输了。',
    'harm.status_draw': '平局。',
    'harm.empty_placeholder': '（空）',
    'harm.action_unknown': '动作 #{id}',
    'harm.act_select_slot': '选择槽位 {slot}',
    'harm.act_take_card': '拿动物卡 #{cid}',
    'harm.act_place_token': '在格子 {hex} 放置颜色 {color}',
    'harm.act_place_cube': '在格子 {hex} 放置动物方块（卡 #{cid}）',
    'harm.act_end_turn': '结束回合',
    'harm.cubes_remaining': '剩 {n}',
    'harm.card_names': ['', '蜜蜂', '青蛙', '北极狐', '魔鬼鱼', '瓢虫', '狐狸', '野猪', '翠鸟', '火烈鸟', '鹦鹉', '蝙蝠', '壁虎', '鳄鱼', '狼', '兔子', '熊', '羊驼', '乌鸦', '鹰', '水獭', '老鼠', '猴子', '黑豹', '刺猬', '猫鼬', '三文鱼', '浣熊', '鸭子', '企鹅', '孔雀', '考拉', '松鼠'],
  },
  en: {
    'harm.title': 'Harmonies',
    'harm.intro': 'Pick a central slot to take all its tokens, then place them on your board. You can also take an animal card and later place an animal cube when terrain matches.',
    'harm.player_0': 'Player 0',
    'harm.player_1': 'Player 1',
    'harm.player_2': 'Player 2',
    'harm.player_3': 'Player 3',
    'harm.you': 'You',
    'harm.opponent': 'Opponent',
    'harm.score': 'Score',
    'harm.animal_score': 'Animal',
    'harm.landscape_score': 'Landscape',
    'harm.central': 'Central slots',
    'harm.market': 'Animal market',
    'harm.held': 'My animal cards',
    'harm.hand': 'Tokens to place',
    'harm.btn_end_turn': 'End turn',
    'harm.status_pick_slot': 'Pick a central slot',
    'harm.status_your_turn': 'Your turn — place token / take card / place animal',
    'harm.status_ai_thinking': 'AI thinking…',
    'harm.status_win': 'You win!',
    'harm.status_lose': 'You lose.',
    'harm.status_draw': 'Draw.',
    'harm.empty_placeholder': '(empty)',
    'harm.action_unknown': 'Action #{id}',
    'harm.act_select_slot': 'Select slot {slot}',
    'harm.act_take_card': 'Take card #{cid}',
    'harm.act_place_token': 'Place color {color} at hex {hex}',
    'harm.act_place_cube': 'Place animal cube at hex {hex} (card #{cid})',
    'harm.act_end_turn': 'End turn',
    'harm.cubes_remaining': '{n} left',
    'harm.card_names': ['', 'Bee', 'Frog', 'Arctic Fox', 'Stingray', 'Ladybug', 'Fox', 'Boar', 'Kingfisher', 'Flamingo', 'Parrot', 'Bat', 'Gecko', 'Crocodile', 'Wolf', 'Rabbit', 'Marmot', 'Llama', 'Crow', 'Eagle', 'Otter', 'Mouse', 'Owl', 'Panther', 'Hedgehog', 'Meerkat', 'Salmon', 'Raccoon', 'Duck', 'Penguin', 'Peacock', 'Koala', 'Squirrel'],
  },
});

const N_COLORS = 6, N_CARDS = 32;

// Board geometry and action layout come with every state (`board_columns`,
// `action_layout`): hex ids per column, left to right, odd columns half a hex
// lower; the place-cube / end-turn ids depend on the board side's hex count.
// Initial values: side B (7 columns 4-3-4-3-4-3-4), the board "harmonies" plays.
let LAYOUT = { select_slot: 0, take_card: 5, place_token: 37, place_cube: 187, end_turn: 987 };
let COL_HEXES = [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9, 10], [11, 12, 13], [14, 15, 16, 17], [18, 19, 20], [21, 22, 23, 24]];

function adoptBoard(st) {
  if (st && st.action_layout) LAYOUT = st.action_layout;
  if (st && st.board_columns) COL_HEXES = st.board_columns;
}

function classifyAction(a) {
  if (a < LAYOUT.take_card) return { kind: 'select', slot: a - LAYOUT.select_slot };
  if (a < LAYOUT.place_token) return { kind: 'take', cid: a - LAYOUT.take_card + 1 };
  if (a < LAYOUT.place_cube) {
    const r = a - LAYOUT.place_token;
    return { kind: 'place_token', hex: Math.floor(r / N_COLORS), color: (r % N_COLORS) + 1 };
  }
  if (a < LAYOUT.end_turn) {
    const r = a - LAYOUT.place_cube;
    return { kind: 'place_cube', hex: Math.floor(r / N_CARDS), cid: (r % N_CARDS) + 1 };
  }
  return { kind: 'end_turn' };
}

function formatMove(info, actionId) {
  if (!info && (actionId == null)) return '';
  if (info) {
    const tp = String(info.type || '');
    if (tp === 'select_slot') return t('harm.act_select_slot', { slot: (info.slot ?? 0) + 1 });
    if (tp === 'take_card') return t('harm.act_take_card', { cid: info.card_id });
    if (tp === 'place_token') return t('harm.act_place_token', { hex: info.hex, color: info.color });
    if (tp === 'place_cube') return t('harm.act_place_cube', { hex: info.hex, cid: info.card_id });
    if (tp === 'end_turn') return t('harm.act_end_turn');
  }
  if (actionId == null) return '';
  const c = classifyAction(actionId);
  if (c.kind === 'select') return t('harm.act_select_slot', { slot: c.slot + 1 });
  if (c.kind === 'take') return t('harm.act_take_card', { cid: c.cid });
  if (c.kind === 'place_token') return t('harm.act_place_token', { hex: c.hex, color: c.color });
  if (c.kind === 'place_cube') return t('harm.act_place_cube', { hex: c.hex, cid: c.cid });
  if (c.kind === 'end_turn') return t('harm.act_end_turn');
  return t('harm.action_unknown', { id: actionId });
}

function cardName(cid) {
  const names = tList('harm.card_names');
  if (Array.isArray(names) && names[cid]) return names[cid];
  return '#' + cid;
}

// Local UI state for multi-step interactions.
let uiState = {
  pendingColor: null,
  activeCardId: null,
};

// Card data: CARD_DATA[card_id] = [scores[], pieces[], anchorIndex] where each piece is
// [dq, dr, color1, color2?, color3?] — bottom-up color stack.
// color: 1=Blue 2=Gray 3=Brown 4=Green 5=Yellow 6=Red.
const CARD_DATA = [
  null,
  [[8, 18], [[0, 0, 3, 4], [1, 0, 5], [0, 1, 5], [-1, 1, 5]], 0],
  [[2, 4, 6, 10, 15], [[0, 0, 4], [1, 0, 1]], 1],
  [[5, 10, 17], [[0, 0, 5], [1, 0, 3, 4], [-1, 1, 3, 4]], 0],
  [[4, 10, 16], [[0, 0, 1], [-1, 0, 2], [-1, 1, 2]], 0],
  [[2, 5, 8, 12, 17], [[0, 0, 5], [1, 0, 4]], 0],
  [[4, 9, 16], [[0, 0, 2], [0, -1, 5], [0, 1, 2]], 2],
  [[4, 8, 13], [[0, 0, 3, 4], [1, 0, 3, 6]], 0],
  [[5, 11, 18], [[0, 0, 3, 3, 4], [1, 0, 1], [-1, 1, 1]], 0],
  [[4, 10, 16], [[0, 0, 1], [-1, 1, 5], [-1, 0, 5]], 0],
  [[4, 9, 14], [[0, 0, 3, 4], [-1, 0, 1], [-1, 1, 1]], 0],
  [[3, 6, 10, 15], [[0, 0, 2], [1, 0, 3, 3, 4]], 0],
  [[5, 10, 16], [[0, 0, 5], [1, 0, 5], [-1, 0, 3, 6]], 2],
  [[4, 9, 15], [[0, 0, 1], [-1, 0, 1], [1, 0, 3, 3, 4]], 1],
  [[4, 10, 16], [[0, 0, 3, 3, 4], [-1, 0, 5], [-1, 1, 5]], 0],
  [[5, 10, 17], [[0, 0, 4], [-1, 0, 4], [1, 0, 3, 6]], 1],
  [[5, 11], [[0, 0, 4], [-1, 0, 2, 2], [-1, 1, 2, 2]], 0],
  [[5, 12], [[0, 0, 5], [0, -1, 5], [0, 1, 2, 2]], 1],
  [[4, 9], [[0, 0, 5], [1, 0, 3, 6], [-1, 1, 3, 6]], 0],
  [[5, 11], [[0, 0, 2, 2, 2], [1, 0, 5]], 0],
  [[5, 10, 16], [[0, 0, 4], [0, -1, 4], [0, 1, 1]], 2],
  [[5, 10, 17], [[0, 0, 3, 6], [1, 0, 5], [-1, 1, 5]], 0],
  [[5, 11], [[0, 0, 2, 2], [-1, 0, 1], [-1, 1, 1]], 0],
  [[5, 11], [[0, 0, 3, 4], [0, -1, 3, 4], [0, 1, 5]], 2],
  [[5, 12], [[0, 0, 3, 6], [-1, 0, 3, 4], [-1, 1, 3, 4]], 0],
  [[2, 5, 9, 14], [[0, 0, 2], [-1, 0, 5]], 0],
  [[3, 6, 10, 16], [[0, 0, 1], [0, 1, 2, 2, 2]], 0],
  [[6, 12], [[0, 0, 5], [1, 0, 1], [0, 1, 1], [-1, 1, 1]], 0],
  [[2, 4, 8, 13], [[0, 0, 1], [1, 0, 3, 6]], 0],
  [[4, 10, 16], [[0, 0, 2], [1, 0, 1], [-1, 1, 1]], 0],
  [[5, 10, 17], [[0, 0, 3, 6], [1, 0, 1], [-1, 1, 1]], 0],
  [[3, 6, 10, 15], [[0, 0, 3, 4], [1, 0, 4]], 0],
  [[4, 9, 15], [[0, 0, 3, 6], [-1, 0, 3, 3, 4]], 0],
];

// Render the interior of a card: name, checkbox scores, hex mini-map.
// cubesPlaced: how many cubes placed (0..scores.length); -1 = market (show all unchecked).
function renderCardContent(cid, cubesPlaced) {
  const frag = document.createDocumentFragment();
  const data = CARD_DATA[cid];
  if (!data) return frag;
  const [scores, pieces, anchorIndex] = data;

  // Name
  const nm = document.createElement('div');
  nm.className = 'harm-card-name';
  nm.textContent = cardName(cid);
  frag.appendChild(nm);

  // Score checkboxes: 3 per row, max 2 rows
  const scoreRow = document.createElement('div');
  scoreRow.className = 'harm-card-scores';
  for (let i = 0; i < scores.length; i++) {
    const cb = document.createElement('span');
    cb.className = 'harm-card-score-item';
    const checked = cubesPlaced >= 0 && i < cubesPlaced;
    cb.textContent = (checked ? '■' : '□') + scores[i];
    if (checked) cb.classList.add('done');
    scoreRow.appendChild(cb);
  }
  frag.appendChild(scoreRow);

  // Hex mini-map: axial (dq,dr) → pixel for flat-top hexes.
  // px = dq * 1.5 * R, py = (dq * sqrt3/2 + dr * sqrt3) * R
  // where R = S/2 is the hex radius.
  const S = 28; // hex cell size for mini-map
  const dotGap = 6;
  const R = S / 2;
  const SQ3 = Math.sqrt(3);

  // Compute pixel positions for all pieces to find bounding box.
  const pxArr = pieces.map(p => p[0] * 1.5 * R);
  const pyArr = pieces.map(p => (p[0] * SQ3 / 2 + p[1] * SQ3) * R);
  const minPx = Math.min(...pxArr);
  const maxPx = Math.max(...pxArr);
  const minPy = Math.min(...pyArr);
  const maxPy = Math.max(...pyArr);
  const mapW = maxPx - minPx + S;
  const mapH = maxPy - minPy + Math.ceil(S * 0.866);

  // Available card drawing area (card width 130px - padding 20px = 110px inner)
  const cardInnerW = 110;
  const cardMapMaxH = 90;

  const mapEl = document.createElement('div');
  mapEl.className = 'harm-card-map';
  mapEl.style.width = cardInnerW + 'px';
  mapEl.style.height = Math.min(Math.ceil(mapH), cardMapMaxH) + 'px';
  mapEl.style.position = 'relative';

  // Center offset: shift pieces so bounding box is centered in map area
  const clampedH = Math.min(Math.ceil(mapH), cardMapMaxH);
  const offsetX = (cardInnerW - mapW) / 2 - minPx;
  const offsetY = (clampedH - mapH) / 2 - minPy;

  for (let pi = 0; pi < pieces.length; pi++) {
    const piece = pieces[pi];
    const dq = piece[0], dr = piece[1];
    const colorStack = piece.slice(2); // bottom-up colors

    // Axial → pixel (flat-top hex)
    const px = dq * 1.5 * R + offsetX;
    const py = (dq * SQ3 / 2 + dr * SQ3) * R + offsetY;

    const cell = document.createElement('div');
    cell.className = 'harm-map-cell';
    cell.style.left = px + 'px';
    cell.style.top = py + 'px';
    cell.style.width = S + 'px';
    cell.style.height = Math.ceil(S * 0.866) + 'px';

    // Draw stacked dots bottom-up
    for (let lv = 0; lv < colorStack.length; lv++) {
      const dot = document.createElement('div');
      dot.className = 'harm-map-dot c' + colorStack[lv];
      dot.style.bottom = (lv * dotGap) + 'px';
      cell.appendChild(dot);
    }

    // Yellow dot marks the anchor hex (animal placement position)
    if (pi === anchorIndex) {
      const topLv = colorStack.length - 1;
      const animal = document.createElement('div');
      animal.className = 'harm-map-animal';
      animal.style.bottom = (topLv * dotGap + 5) + 'px';
      cell.appendChild(animal);
    }

    mapEl.appendChild(cell);
  }

  frag.appendChild(mapEl);
  return frag;
}

function buildLegalIndex(legal) {
  const idx = {
    selectSlots: new Set(),
    takeCards: new Set(),
    placeToken: new Map(),
    placeCube: new Map(),
    canEnd: false,
  };
  if (!legal) return idx;
  for (const a of legal) {
    const c = classifyAction(a);
    if (c.kind === 'select') idx.selectSlots.add(c.slot);
    else if (c.kind === 'take') idx.takeCards.add(c.cid);
    else if (c.kind === 'place_token') idx.placeToken.set(c.hex + ',' + c.color, a);
    else if (c.kind === 'place_cube') idx.placeCube.set(c.hex + ',' + c.cid, a);
    else if (c.kind === 'end_turn') idx.canEnd = true;
  }
  return idx;
}

// Tower type → full color stack (bottom-up). Used for hex rendering.
const TOWER_STACK = [
  [],           // 0 empty
  [1],          // 1 Blue
  [2],          // 2 Gray
  [3],          // 3 Brown
  [4],          // 4 Green
  [5],          // 5 Yellow
  [6],          // 6 Red
  [6, 6],       // 7 house2
  [3, 3],       // 8 wood2
  [3, 4],       // 9 tree2
  [2, 2],       // 10 mountain2
  [3, 3, 4],    // 11 tree3
  [2, 2, 2],    // 12 mountain3
];

function renderHex(hexId, towerType, hasAnimal) {
  const div = document.createElement('div');
  div.className = 'harm-hex';
  div.dataset.hex = String(hexId);
  const bg = document.createElement('div');
  bg.className = 'harm-hex-bg';
  div.appendChild(bg);
  const colors = TOWER_STACK[towerType] || [];
  const dotGap = 8;
  for (let lv = 0; lv < colors.length; lv++) {
    const dot = document.createElement('div');
    dot.className = 'harm-hex-dot c' + colors[lv];
    dot.style.bottom = (lv * dotGap) + 'px';
    div.appendChild(dot);
  }
  if (hasAnimal) {
    const an = document.createElement('div');
    an.className = 'harm-hex-animal';
    const topLv = colors.length - 1;
    an.style.bottom = (topLv * dotGap + 9) + 'px';
    div.appendChild(an);
  }
  return div;
}

function renderPlayerBoard(player, isMe, legalIndex, canPlay, ctx, seatIdx) {
  const wrap = document.createElement('div');
  wrap.className = 'harm-pboard';
  wrap.dataset.player = isMe ? 'me' : 'opp';
  wrap.dataset.seat = isMe ? 'me' : 'opp';

  const title = document.createElement('div');
  title.className = 'harm-pboard-title' + (isMe ? ' me' : '');
  // For self: "你"; for others: "玩家 N" (1-indexed), matching loveletter's
  // convention — opponents must be distinguishable at 3p/4p, not all "对手".
  const name = isMe ? t('harm.you') : t('harm.player_' + seatIdx);
  title.textContent = name
    + ' — ' + t('harm.score') + ': ' + (player.score ?? 0)
    + '  (' + t('harm.animal_score') + ' ' + (player.animal_score ?? 0)
    + ' / ' + t('harm.landscape_score') + ' ' + (player.landscape_score ?? 0) + ')';
  wrap.appendChild(title);

  const grid = document.createElement('div');
  grid.className = 'harm-hex-grid';
  grid.style.gridTemplateColumns = 'repeat(' + COL_HEXES.length + ', calc(var(--hex-size) * 0.75))';
  for (let q = 0; q < COL_HEXES.length; q++) {
    const col = document.createElement('div');
    col.className = 'harm-hex-col' + ((q % 2 === 1) ? ' offset' : '');
    for (const hexId of COL_HEXES[q]) {
      const tt = (player.tower_type || [])[hexId] || 0;
      const ha = (player.has_animal || [])[hexId] || 0;
      const hex = renderHex(hexId, tt, ha);
      if (isMe && canPlay) {
        if (uiState.pendingColor) {
          const aid = legalIndex.placeToken.get(hexId + ',' + uiState.pendingColor);
          if (aid != null) {
            hex.classList.add('legal');
            hex.dataset.action = aid;
          }
        }
        if (uiState.activeCardId) {
          const aid = legalIndex.placeCube.get(hexId + ',' + uiState.activeCardId);
          if (aid != null) {
            hex.classList.add('legal');
            hex.dataset.action = aid;
          }
        }
      }
      col.appendChild(hex);
    }
    grid.appendChild(col);
  }
  grid.addEventListener('click', (e) => {
    const cell = e.target.closest('[data-action]');
    if (!cell) return;
    const aid = Number(cell.dataset.action);
    const c = classifyAction(aid);
    if (c.kind === 'place_token') uiState.pendingColor = null;
    if (c.kind === 'place_cube') uiState.activeCardId = null;
    ctx.submitAction(aid);
  });
  wrap.appendChild(grid);

  // Animal cards for this player.
  wrap.appendChild(renderHeldCards(player, isMe, canPlay));

  return wrap;
}

function renderHand(selectedTokens, legalIndex, canPlay) {
  const wrap = document.createElement('div');
  wrap.className = 'harm-hand';
  if (!selectedTokens) return wrap;
  let any = false;
  for (let c = 1; c <= N_COLORS; c++) {
    const cnt = selectedTokens[c - 1] || 0;
    for (let i = 0; i < cnt; i++) {
      any = true;
      const tok = document.createElement('div');
      tok.className = 'harm-hand-token c' + c;
      if (canPlay && uiState.pendingColor === c) tok.classList.add('active');
      tok.addEventListener('click', () => {
        uiState.pendingColor = (uiState.pendingColor === c) ? null : c;
        window.dispatchEvent(new CustomEvent('harm-rerender'));
      });
      wrap.appendChild(tok);
    }
  }
  if (!any) {
    const ph = document.createElement('span');
    ph.className = 'harm-placeholder';
    ph.textContent = t('harm.empty_placeholder');
    wrap.appendChild(ph);
  }
  return wrap;
}

function renderCentralSlots(state, legalIndex, canPlay, ctx) {
  const wrap = document.createElement('div');
  wrap.className = 'harm-central';
  const counts = state.central_board_counts || [];
  for (let s = 0; s < counts.length; s++) {
    const slot = document.createElement('div');
    slot.className = 'harm-slot';
    slot.dataset.slot = String(s);
    if (canPlay && legalIndex.selectSlots.has(s)) {
      slot.classList.add('legal');
      slot.dataset.action = String(LAYOUT.select_slot + s);
    } else {
      slot.classList.add('disabled');
    }
    const tokRow = document.createElement('div');
    tokRow.className = 'harm-slot-tokens';
    let totalTokens = 0;
    const row = counts[s] || [];
    let tokenIdx = 0;
    for (let c = 1; c <= N_COLORS; c++) {
      const n = row[c - 1] || 0;
      totalTokens += n;
      for (let i = 0; i < n; i++) {
        const m = document.createElement('div');
        m.className = 'harm-token-mini c' + c;
        m.dataset.slotToken = s + '-' + tokenIdx;
        tokenIdx++;
        tokRow.appendChild(m);
      }
    }
    slot.appendChild(tokRow);
    wrap.appendChild(slot);
  }
  wrap.addEventListener('click', (e) => {
    const slot = e.target.closest('[data-action]');
    if (!slot) return;
    ctx.submitAction(Number(slot.dataset.action));
  });
  return wrap;
}

function renderMarket(state, legalIndex, canPlay, ctx) {
  const wrap = document.createElement('div');
  wrap.className = 'harm-market';
  const market = state.animal_market || [];
  for (const cid of market) {
    const card = document.createElement('div');
    card.className = 'harm-card';
    card.dataset.card = String(cid);
    if (cid === 0) {
      card.classList.add('empty');
      card.textContent = t('harm.empty_placeholder');
    } else {
      if (canPlay && legalIndex.takeCards.has(cid)) {
        card.classList.add('legal');
        card.dataset.action = String(LAYOUT.take_card + cid - 1);
      }
      card.appendChild(renderCardContent(cid, -1));
    }
    wrap.appendChild(card);
  }
  wrap.addEventListener('click', (e) => {
    const card = e.target.closest('[data-action]');
    if (!card) return;
    ctx.submitAction(Number(card.dataset.action));
  });
  return wrap;
}

function renderHeldCards(player, isMe, canPlay) {
  const wrap = document.createElement('div');
  wrap.className = 'harm-hand-cards';
  const held = player.held_cards || [];
  const remaining = player.cube_remaining || [];
  for (let i = 0; i < held.length; i++) {
    const cid = held[i];
    const card = document.createElement('div');
    card.className = 'harm-held-card';
    card.dataset.heldCard = String(cid);
    if (!cid) {
      card.classList.add('empty');
      card.textContent = t('harm.empty_placeholder');
      wrap.appendChild(card);
      continue;
    }
    if (isMe && canPlay && uiState.activeCardId === cid) card.classList.add('active');
    const data = CARD_DATA[cid];
    const totalCubes = data ? data[0].length : 0;
    const cubesPlaced = totalCubes - remaining[i];
    card.appendChild(renderCardContent(cid, cubesPlaced));
    if (isMe && canPlay) {
      card.style.cursor = 'pointer';
      card.addEventListener('click', () => {
        uiState.activeCardId = (uiState.activeCardId === cid) ? null : cid;
        window.dispatchEvent(new CustomEvent('harm-rerender'));
      });
    }
    wrap.appendChild(card);
  }
  return wrap;
}

function renderActions(legalIndex, canPlay, ctx) {
  const wrap = document.createElement('div');
  wrap.className = 'harm-actions';
  const endBtn = document.createElement('button');
  endBtn.className = 'harm-btn primary';
  endBtn.textContent = t('harm.btn_end_turn');
  endBtn.disabled = !(canPlay && legalIndex.canEnd);
  endBtn.addEventListener('click', () => {
    if (canPlay && legalIndex.canEnd) {
      uiState.pendingColor = null;
      uiState.activeCardId = null;
      ctx.submitAction(LAYOUT.end_turn);
    }
  });
  wrap.appendChild(endBtn);
  return wrap;
}

let lastCtx = null;
window.addEventListener('harm-rerender', () => {
  if (lastCtx && lastCtx.rerender) lastCtx.rerender();
});

function renderBoard(container, gameState, ctx) {
  lastCtx = ctx;
  container.innerHTML = '';
  const root = document.createElement('div');
  root.className = 'harm-root';

  if (!gameState || !gameState.state) {
    container.appendChild(root);
    return;
  }
  const st = gameState.state;
  adoptBoard(st);
  const canPlay = ctx.canPlay;
  const legal = buildLegalIndex(gameState.legal_actions);
  const myIdx = ctx.state.humanPlayer ?? 0;
  const numP = gameState.num_players || 2;

  // Status banner.
  const status = document.createElement('div');
  status.className = 'harm-banner';
  if (gameState.is_terminal) {
    const w = gameState.winner;
    status.textContent = (w === myIdx) ? t('harm.status_win')
      : (w === -1) ? t('harm.status_draw') : t('harm.status_lose');
  } else if (canPlay) {
    status.classList.add('my-turn');
    if (!st.turn_started) status.textContent = t('harm.status_pick_slot');
    else status.textContent = t('harm.status_your_turn');
  } else {
    status.textContent = t('harm.status_ai_thinking');
  }
  root.appendChild(status);

  // Animal market.
  const mt = document.createElement('div');
  mt.className = 'harm-section-title';
  mt.textContent = t('harm.market');
  root.appendChild(mt);
  root.appendChild(renderMarket(st, legal, canPlay, ctx));

  // Central board.
  const ct = document.createElement('div');
  ct.className = 'harm-section-title';
  ct.textContent = t('harm.central');
  root.appendChild(ct);
  root.appendChild(renderCentralSlots(st, legal, canPlay, ctx));

  // Selected tokens hand.
  const ht = document.createElement('div');
  ht.className = 'harm-section-title';
  ht.textContent = t('harm.hand');
  root.appendChild(ht);
  root.appendChild(renderHand(st.selected_tokens, legal, canPlay));

  // Player boards — perspective-rotated: "me" first, then opponents clockwise.
  const players = st.players || [];
  const boards = document.createElement('div');
  boards.className = 'harm-boards';
  for (let k = 0; k < numP; k++) {
    const pIdx = (myIdx + k) % numP;
    if (players[pIdx]) {
      const pb = renderPlayerBoard(players[pIdx], k === 0, legal, canPlay, ctx, pIdx);
      pb.dataset.player = String(pIdx);
      boards.appendChild(pb);
    }
  }
  root.appendChild(boards);

  // Action buttons.
  root.appendChild(renderActions(legal, canPlay, ctx));

  container.appendChild(root);
}

// describeTransition: animate token/cube placements and card takes.
function describeTransition(prevState, newState, actionInfo, actionId) {
  if (!actionInfo || !prevState) return null;
  const steps = [];
  const tp = String(actionInfo.type || '');
  // Determine which player acted, to target the correct board.
  const prev = prevState.state || prevState;
  const actor = prevState.current_player;
  const boardSel = '[data-player="' + actor + '"]';

  if (tp === 'select_slot') {
    const slotIdx = actionInfo.slot ?? 0;
    const colorMap = { 1: '#3b82f6', 2: '#94a3b8', 3: '#a16207', 4: '#22c55e', 5: '#facc15', 6: '#dc2626' };
    const slotCounts = (prev.central_board_counts || [])[slotIdx] || [];
    const flights = [];
    let tokenIdx = 0;
    for (let c = 1; c <= N_COLORS; c++) {
      const n = slotCounts[c - 1] || 0;
      for (let i = 0; i < n; i++) {
        const bg = colorMap[c] || '#ccc';
        const fromSel = '[data-slot-token="' + slotIdx + '-' + tokenIdx + '"]';
        flights.push({
          from: fromSel,
          to: '.harm-hand',
          createElement() {
            const d = document.createElement('div');
            d.style.cssText = 'width:20px;height:20px;border-radius:50%;border:1px solid #00000033;background:' + bg + ';';
            return d;
          },
          width: 20,
          height: 20,
          duration: 500,
          hideFrom: true,
        });
        tokenIdx++;
      }
    }
    if (flights.length) {
      steps.push({ type: 'flyGroup', flights });
    }
  } else if (tp === 'place_token') {
    const hex = actionInfo.hex;
    const color = actionInfo.color || 1;
    const colorMap = { 1: '#3b82f6', 2: '#94a3b8', 3: '#a16207', 4: '#22c55e', 5: '#facc15', 6: '#dc2626' };
    steps.push({
      type: 'fly',
      from: '.harm-hand',
      to: boardSel + ' [data-hex="' + hex + '"]',
      createElement() {
        const d = document.createElement('div');
        d.style.cssText = 'width:20px;height:20px;border-radius:50%;border:1px solid #00000033;background:' + (colorMap[color] || '#ccc') + ';';
        return d;
      },
      width: 20,
      height: 20,
      duration: 450,
      onStart(srcEl) {
        const tok = srcEl.querySelector('.harm-hand-token.c' + color);
        if (tok) tok.style.visibility = 'hidden';
      },
    });
  } else if (tp === 'place_cube') {
    const hex = actionInfo.hex;
    const cid = actionInfo.card_id;
    const cardSel = boardSel + ' [data-held-card="' + cid + '"]';
    steps.push({
      type: 'fly',
      from: cardSel,
      to: boardSel + ' [data-hex="' + hex + '"]',
      createElement() {
        const d = document.createElement('div');
        d.style.cssText = 'width:12px;height:12px;background:#facc15;border:1.5px solid #a16207;border-radius:50%;';
        return d;
      },
      width: 15,
      height: 15,
      duration: 500,
    });
  } else if (tp === 'take_card') {
    const cid = actionInfo.card_id;
    steps.push({
      type: 'fly',
      from: '[data-card="' + cid + '"]',
      to: boardSel + ' .harm-hand-cards',
      createElement() {
        const d = document.createElement('div');
        d.style.cssText = 'width:80px;height:40px;background:#fff;border:2px solid #16a34a;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:600;';
        d.textContent = cardName(cid);
        return d;
      },
      width: 80,
      height: 40,
      duration: 600,
      hideFrom: true,
    });
  } else if (tp === 'end_turn') {
    steps.push({
      type: 'pause',
      duration: 200,
    });
  }

  return steps.length ? steps : null;
}

createApp({
  gameId: 'harmonies',
  gameTitle: t('harm.title'),
  gameIntro: t('harm.intro'),
  players: { min: 2, max: 4 },
  renderBoard,
  describeTransition,
  formatOpponentMove: formatMove,
  formatSuggestedMove: formatMove,
  getPlayerSymbol: (idx) => t('harm.player_' + idx),
  difficulties: ['heuristic', 'casual', 'expert'],
  defaultDifficulty: 'heuristic',
  defaultSeat: 'random',
  showWinrateDefault: true,
  onGameStart() {
    uiState.pendingColor = null;
    uiState.activeCardId = null;
  },
});
