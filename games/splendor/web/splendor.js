import { createApp } from '/static/general/app.js';
import { t, register, tList } from '/static/general/i18n.js';

register({
  zh: {
    'spl.title': '璀璨宝石',
    'spl.intro': '拿宝石、买卡、抢贵族，先到15分触发终局',
    'spl.color_white': '白',
    'spl.color_blue': '蓝',
    'spl.color_green': '绿',
    'spl.color_red': '红',
    'spl.color_black': '黑',
    'spl.color_gold': '金',
    'spl.action_unknown': '动作 #{id}',
    'spl.action_start': '开局',
    'spl.move.buy_faceup': '购买 T{tier} #{slot}',
    'spl.move.reserve_faceup': '保留明牌 T{tier} #{slot}',
    'spl.move.reserve_deck': '保留牌堆 T{tier}',
    'spl.move.buy_reserved': '购买保留牌 #{slot}',
    'spl.move.take_three': '拿三色：{cs}',
    'spl.move.take_two_diff': '拿两色：{cs}',
    'spl.move.take_one': '拿一枚：{c}',
    'spl.move.take_two_same': '拿两枚同色：{c}',
    'spl.move.choose_noble': '选择贵族 #{slot}',
    'spl.move.return_token': '返还：{c}',
    'spl.move.pass': '跳过回合',
    'spl.bank_title': '银行宝石',
    'spl.tableau_title': '发展卡',
    'spl.tableau_full': '桌面开发卡（T3/T2/T1）',
    'spl.nobles_title': '贵族',
    'spl.turn_n': '第{n}回合',
    'spl.return_mode': '返还模式：点击宝石堆返还 1 枚。',
    'spl.noble_mode': '贵族选择模式：点击可选贵族完成本回合。',
    'spl.not_your_turn': '当前不是你的回合。',
    'spl.no_take': '当前无法拿宝石。',
    'spl.pick_hint': '操作：点击宝石堆选择，三色点三次/同色点两次，然后点确认。',
    'spl.picked': '已选择：{cs}（可确认或取消）',
    'spl.confirm_take': '确认拿宝石',
    'spl.cancel_pick': '取消选择',
    'spl.pass_btn': '跳过回合',
    'spl.pending_returns': '当前需返还 {n} 枚宝石',
    'spl.pending_noble': '当前需要选择一位贵族。',
    'spl.tier_label': 'Tier {n}',
    'spl.deck_btn': '保留牌堆 ({n})',
    'spl.btn_buy': '购买',
    'spl.btn_reserve': '保留',
    'spl.btn_op': '操作',
    'spl.noble_3pt': '贵族 3分',
    'spl.btn_choose_noble': '选择贵族',
    'spl.points_suffix': '分',
    'spl.player_n': '玩家{n}',
    'spl.player_self_suffix': '（你）',
    'spl.player_current_suffix': ' · 当前',
    'spl.stat_meta': '分:{points}  卡:{cards}  贵族:{nobles}',
    'spl.gems_label': '宝石',
    'spl.gems_total': '总数:{n}',
    'spl.bonus_label': '发展卡',
    'spl.reserve_label': '保留卡 (最多 3 张)',
    'spl.reserve_empty': '空',
    'spl.reserve_hidden': '暗保留 T{tier}',
    'spl.btn_buy_reserved': '购买保留',
  },
  en: {
    'spl.title': 'Splendor',
    'spl.intro': 'Take gems, buy cards, attract nobles. First to 15 points triggers the endgame.',
    'spl.color_white': 'White',
    'spl.color_blue': 'Blue',
    'spl.color_green': 'Green',
    'spl.color_red': 'Red',
    'spl.color_black': 'Black',
    'spl.color_gold': 'Gold',
    'spl.action_unknown': 'Action #{id}',
    'spl.action_start': 'Game start',
    'spl.move.buy_faceup': 'Buy T{tier} #{slot}',
    'spl.move.reserve_faceup': 'Reserve face-up T{tier} #{slot}',
    'spl.move.reserve_deck': 'Reserve top of T{tier} deck',
    'spl.move.buy_reserved': 'Buy reserved #{slot}',
    'spl.move.take_three': 'Take 3: {cs}',
    'spl.move.take_two_diff': 'Take 2 diff: {cs}',
    'spl.move.take_one': 'Take 1: {c}',
    'spl.move.take_two_same': 'Take 2 same: {c}',
    'spl.move.choose_noble': 'Choose noble #{slot}',
    'spl.move.return_token': 'Return: {c}',
    'spl.move.pass': 'Pass',
    'spl.bank_title': 'Bank gems',
    'spl.tableau_title': 'Development cards',
    'spl.tableau_full': 'Tableau (T3/T2/T1)',
    'spl.nobles_title': 'Nobles',
    'spl.turn_n': 'Turn {n}',
    'spl.return_mode': 'Return mode: click a stack to return 1 token.',
    'spl.noble_mode': 'Noble pick: click an available noble to finish the turn.',
    'spl.not_your_turn': "Not your turn.",
    'spl.no_take': 'Cannot take gems right now.',
    'spl.pick_hint': 'Tap stacks to pick (3 different / 2 same), then confirm.',
    'spl.picked': 'Picked: {cs} (confirm or cancel)',
    'spl.confirm_take': 'Confirm take',
    'spl.cancel_pick': 'Cancel pick',
    'spl.pass_btn': 'Pass',
    'spl.pending_returns': 'Need to return {n} tokens',
    'spl.pending_noble': 'Choose a noble to continue.',
    'spl.tier_label': 'Tier {n}',
    'spl.deck_btn': 'Reserve deck ({n})',
    'spl.btn_buy': 'Buy',
    'spl.btn_reserve': 'Reserve',
    'spl.btn_op': 'Op',
    'spl.noble_3pt': 'Noble 3pt',
    'spl.btn_choose_noble': 'Take noble',
    'spl.points_suffix': 'pt',
    'spl.player_n': 'Player {n}',
    'spl.player_self_suffix': ' (you)',
    'spl.player_current_suffix': ' · current',
    'spl.stat_meta': 'pts:{points}  cards:{cards}  nobles:{nobles}',
    'spl.gems_label': 'Gems',
    'spl.gems_total': 'total:{n}',
    'spl.bonus_label': 'Bonuses',
    'spl.reserve_label': 'Reserved (max 3)',
    'spl.reserve_empty': 'empty',
    'spl.reserve_hidden': 'Hidden T{tier}',
    'spl.btn_buy_reserved': 'Buy reserved',
  },
});

const COLOR_NAMES = ['white', 'blue', 'green', 'red', 'black', 'gold'];
const COLOR_KEYS = ['spl.color_white','spl.color_blue','spl.color_green','spl.color_red','spl.color_black','spl.color_gold'];
function colorLabel(c) { return t(COLOR_KEYS[c]); }
const TAKE_THREE_COMBOS = [
  [0,1,2],[0,1,3],[0,1,4],[0,2,3],[0,2,4],[0,3,4],
  [1,2,3],[1,2,4],[1,3,4],[2,3,4]
];
const TAKE_TWO_DIFF_COMBOS = [
  [0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]
];

const BUY_FACEUP = 0, RESERVE_FACEUP = 12, RESERVE_DECK = 24, BUY_RESERVED = 27;
const TAKE_THREE = 30, TAKE_TWO_DIFF = 40, TAKE_ONE = 50, TAKE_TWO_SAME = 55;
const CHOOSE_NOBLE = 60, RETURN_TOKEN = 63, PASS_ACTION = 69;

let gemPick = [];
let currentCtx = null;
// Latest gameState seen by renderBoard, kept so formatSuggestedMove
// (smart hint) can look up the card the AI is recommending and append
// its gem-color details for debugging.
let currentGameState = null;

// --- Stable screen slots (frontend-only) ---
//
// The C++ engine ships tableau[t] as a packed-prefix array (length 0..4,
// no internal -1s). Action IDs `buy_faceup` / `reserve_faceup` index
// into that packed array. Visually we want each card to keep its
// screen position across plies — when card X is taken from packed slot
// 1, the deck-flipped replacement should land on the same SCREEN slot
// that X just vacated, not jump to the end.
//
// We solve this entirely on the frontend by maintaining a per-tier
// screen layout: an array of 4 cells, each either a card object (from
// the latest packed JSON) or null. Reconciliation per tier:
//   - cards present in cache AND in new packed: keep their screen cell
//   - cards in cache but not in new packed: cell becomes null (taken)
//   - cards new in packed: fill the first null cell (the just-vacated
//     position, which is exactly what we want for the refill animation)
//
// Mappings the renderer needs:
//   screenSlot → card object (for rendering + data-tableau anchor)
//   screenSlot → packedSlot (for action IDs)
//   packedSlot → screenSlot (for animations targeting a packed-indexed
//     action like buy_faceup, reserve_faceup, or a smart-hint suggestion)
//
// Cache survives across renders. Reset on game start via resetSplendorBoardState.
let screenLayoutCache = [null, null, null];   // per tier, length-4 arrays of {card, packedSlot} | null
let lastScreenLayout = [null, null, null];    // snapshot taken BEFORE the current render's reconcile, used by describeTransition

function emptyTierLayout() {
  return [null, null, null, null];
}

function packedToCardIds(tier) {
  const ids = [];
  for (let s = 0; s < tier.length; s++) {
    const c = tier[s];
    const id = (c && c.card_id != null) ? c.card_id : -1;
    if (id >= 0) ids.push({ id, card: c, packedSlot: s });
  }
  return ids;
}

// Given a previous screen layout (length-4, cells are {card, packedSlot} | null)
// and a new packed-tier from the JSON, return a new screen layout that
// preserves positions of surviving cards and fills new cards into the
// first empty cells. If `prev` is null, use identity mapping (packed
// slot s → screen slot s).
function reconcileTierLayout(prev, packedTier) {
  const newCards = packedToCardIds(packedTier);  // [{id, card, packedSlot}]
  const next = emptyTierLayout();
  const remaining = new Map();  // id → {card, packedSlot}
  for (const e of newCards) remaining.set(e.id, e);

  if (prev) {
    for (let s = 0; s < 4; s++) {
      const cell = prev[s];
      if (!cell || cell.card == null) continue;
      const prevId = (cell.card.card_id != null) ? cell.card.card_id : -1;
      if (prevId < 0) continue;
      if (remaining.has(prevId)) {
        // Card survives — keep it at its old screen slot, refresh
        // packedSlot (could have shifted due to other tier changes).
        next[s] = remaining.get(prevId);
        remaining.delete(prevId);
      }
    }
  }

  // New (unrecognized) cards fill the first empty screen cells in
  // packed order. When prev is null this collapses to identity.
  let cursor = 0;
  for (const e of newCards) {
    if (!remaining.has(e.id)) continue;
    while (cursor < 4 && next[cursor] != null) cursor++;
    if (cursor >= 4) break;
    next[cursor] = e;
    cursor++;
  }
  return next;
}

function reconcileScreenLayout(tableau) {
  const out = [null, null, null];
  for (let t = 0; t < 3; t++) {
    out[t] = reconcileTierLayout(screenLayoutCache[t], tableau[t] || []);
  }
  return out;
}

// Look up screen slot containing the card at packed slot `packedSlot`
// in the CURRENT (post-reconcile) layout. Returns -1 if not found.
function packedSlotToScreen(layout, tier, packedSlot) {
  const row = layout[tier];
  if (!row) return -1;
  for (let s = 0; s < 4; s++) {
    const cell = row[s];
    if (cell && cell.packedSlot === packedSlot) return s;
  }
  return -1;
}

// Resolve the screen slot for a buy_faceup / reserve_faceup action,
// given the layout the user is looking at NOW (pre-action). Used by
// describeTransition to anchor "card flies away from screen slot S".
//
// Call timing matters: describeTransition is invoked BEFORE
// renderBoard(displayFrame), so screenLayoutCache still reflects the
// previous render's output — i.e. the pre-action board the user is
// staring at. lastScreenLayout (set inside renderBoard) is one render
// older and not what we want here.
function packedSlotToScreenInPrev(tier, packedSlot) {
  return packedSlotToScreen(screenLayoutCache, tier, packedSlot);
}

// Reset cache when a new game starts or replay jumps. Caller passes
// the JSON tableau to seed identity mapping.
function resetScreenLayoutFor(tableau) {
  screenLayoutCache = [null, null, null];
  lastScreenLayout = [null, null, null];
  if (tableau) {
    screenLayoutCache = reconcileScreenLayout(tableau);
    lastScreenLayout = screenLayoutCache.map(r => (r ? r.slice() : null));
  }
}

function getLegalSet(gameState) {
  if (!gameState) return new Set();
  return new Set(gameState.legal_actions || []);
}


function getStage(gameState) {
  return gameState && gameState.state ? (gameState.state.stage || 0) : 0;
}

// Append the recommended card's gem color so the user can spot
// misaligned tableau slots immediately. Example: " [颜色=红]".
function formatCardDebug(card) {
  if (!card || card.card_id == null || card.card_id < 0) return '';
  if (card.bonus == null) return '';
  return ` [颜色=${colorLabel(card.bonus)}]`;
}

// info.slot is a PACKED-array slot (the action ID's payload).
// gs.state.tableau is also packed-prefix from the engine, so direct
// indexing is correct here.
function lookupTableauCard(info) {
  const gs = currentGameState;
  if (!gs || !gs.state) return null;
  const tier = info && info.tier;
  const slot = info && info.slot;
  if (tier == null || slot == null) return null;
  const row = (gs.state.tableau || [])[tier] || [];
  return row[slot] || null;
}

function lookupReservedCard(info, actorIdx) {
  const gs = currentGameState;
  if (!gs || !gs.state) return null;
  const slot = info && info.slot;
  if (slot == null) return null;
  const players = gs.state.players || [];
  const p = players[actorIdx];
  if (!p) return null;
  const reserved = p.reserved || [];
  return reserved[slot] || null;
}

// Translate engine's packed slot (the value carried in action IDs)
// into the SCREEN slot the user actually sees in the rendered tableau.
// screenLayoutCache reflects the most recent render's reconciled layout.
function packedToScreenForDisplay(tier, packedSlot) {
  if (tier == null || packedSlot == null) return packedSlot;
  const screen = packedSlotToScreen(screenLayoutCache, tier, packedSlot);
  return screen >= 0 ? screen : packedSlot;
}

function formatActionInfo(info, actionId) {
  if (!info || !info.type) return t('spl.action_unknown', { id: actionId });
  switch (info.type) {
    case 'buy_faceup': return t('spl.move.buy_faceup', { tier: info.tier + 1, slot: packedToScreenForDisplay(info.tier, info.slot) + 1 });
    case 'reserve_faceup': return t('spl.move.reserve_faceup', { tier: info.tier + 1, slot: packedToScreenForDisplay(info.tier, info.slot) + 1 });
    case 'reserve_deck': return t('spl.move.reserve_deck', { tier: info.tier + 1 });
    case 'buy_reserved': return t('spl.move.buy_reserved', { slot: info.slot + 1 });
    case 'take_three': return t('spl.move.take_three', { cs: (info.colors || []).map(c => colorLabel(c)).join('+') });
    case 'take_two_different': return t('spl.move.take_two_diff', { cs: (info.colors || []).map(c => colorLabel(c)).join('+') });
    case 'take_one': return t('spl.move.take_one', { c: colorLabel(info.color) });
    case 'take_two_same': return t('spl.move.take_two_same', { c: colorLabel(info.color) });
    case 'choose_noble': return t('spl.move.choose_noble', { slot: info.noble_slot + 1 });
    case 'return_token': return t('spl.move.return_token', { c: colorLabel(info.token) });
    case 'pass': return t('spl.move.pass');
    default: return t('spl.action_unknown', { id: actionId });
  }
}

// --- Gem picking logic ---

function resolveTakeActionId(colors, legalSet) {
  if (!colors || colors.length === 0) return null;
  if (colors.length === 2 && colors[0] === colors[1]) {
    const aid = TAKE_TWO_SAME + colors[0];
    return legalSet.has(aid) ? aid : null;
  }
  const uniq = Array.from(new Set(colors));
  if (uniq.length !== colors.length) return null;
  if (uniq.length === 1) { const aid = TAKE_ONE + uniq[0]; return legalSet.has(aid) ? aid : null; }
  if (uniq.length === 2) return resolveCombo(TAKE_TWO_DIFF_COMBOS, TAKE_TWO_DIFF, uniq, legalSet);
  if (uniq.length === 3) return resolveCombo(TAKE_THREE_COMBOS, TAKE_THREE, uniq, legalSet);
  return null;
}

function resolveCombo(combos, offset, colors, legalSet) {
  const sorted = colors.slice().sort((a, b) => a - b);
  for (let i = 0; i < combos.length; i++) {
    const combo = combos[i];
    if (combo.length !== sorted.length) continue;
    if (combo.every((v, j) => v === sorted[j])) {
      const aid = offset + i;
      return legalSet.has(aid) ? aid : null;
    }
  }
  return null;
}

function hasAnyTakeAction(legalSet) {
  for (let a = TAKE_THREE; a < CHOOSE_NOBLE; a++) {
    if (legalSet.has(a)) return true;
  }
  return false;
}

function hasTakeExtension(colors, legalSet) {
  if (!colors || colors.length === 0 || colors.length > 3) return false;
  if (resolveTakeActionId(colors, legalSet) != null) return true;
  if (colors.length === 1) {
    const first = colors[0];
    if (resolveTakeActionId([first, first], legalSet) != null) return true;
    for (let c = 0; c < 5; c++) {
      if (c === first) continue;
      if (resolveTakeActionId([first, c], legalSet) != null) return true;
      for (let d = c + 1; d < 5; d++) {
        if (d === first) continue;
        if (resolveTakeActionId([first, c, d], legalSet) != null) return true;
      }
    }
    return false;
  }
  if (colors.length === 2 && colors[0] !== colors[1]) {
    for (let c = 0; c < 5; c++) {
      if (colors.includes(c)) continue;
      if (resolveTakeActionId([colors[0], colors[1], c], legalSet) != null) return true;
    }
    return false;
  }
  return false;
}

function canStartWithColor(c, legalSet) {
  return resolveTakeActionId([c], legalSet) != null || hasTakeExtension([c], legalSet);
}

function canAppendColor(nextColor, legalSet) {
  if (gemPick.length >= 3) return false;
  if (gemPick.length === 0) return canStartWithColor(nextColor, legalSet);
  const next = gemPick.concat([nextColor]);
  const uniqSet = new Set(next);
  if (uniqSet.size !== next.length && !(next.length === 2 && next[0] === next[1])) return false;
  return hasTakeExtension(next, legalSet);
}

// --- Rendering helpers ---

function tokenChip(colorIdx, count, options) {
  options = options || {};
  const el = document.createElement('button');
  el.type = 'button';
  el.className = 'token-chip ' + COLOR_NAMES[colorIdx];
  if (options.selected) el.classList.add('selected');
  el.textContent = colorLabel(colorIdx) + ':' + count;
  if (options.onClick && !options.disabled) {
    el.classList.add('clickable');
    el.addEventListener('click', options.onClick);
  } else {
    el.disabled = true;
  }
  return el;
}

function renderDevCard(card, actions) {
  const root = document.createElement('div');
  root.className = 'dev-card';
  if (!card) { root.classList.add('dev-card-placeholder'); return root; }

  const points = card.points || 0;
  const bonus = card.bonus != null ? card.bonus : 0;
  const tier = card.tier || 0;

  const head = document.createElement('div');
  head.className = 'dev-card-head';
  const left = document.createElement('span');
  left.className = 'dev-card-head-left';
  left.innerHTML = '<span>T' + tier + '</span><span class="bonus-inline ' + COLOR_NAMES[Math.max(0, Math.min(4, bonus))] + '">' + colorLabel(Math.max(0, Math.min(4, bonus))) + '</span>';
  const right = document.createElement('span');
  right.textContent = points + t('spl.points_suffix');
  head.append(left, right);
  root.appendChild(head);

  const costRow = document.createElement('div');
  costRow.className = 'cost-row';
  const cost = card.cost || [];
  for (let i = 0; i < 5; i++) {
    const v = cost[i] || 0;
    if (v <= 0) continue;
    const pill = document.createElement('span');
    pill.className = 'cost-gem ' + COLOR_NAMES[i];
    pill.textContent = String(v);
    costRow.appendChild(pill);
  }
  root.appendChild(costRow);

  actions = actions || [];
  if (actions.length > 0) {
    if (actions.some(a => !a.disabled && a.onClick)) root.classList.add('clickable');
    const actionRow = document.createElement('div');
    actionRow.className = 'card-op-row';
    for (const act of actions) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'card-op-btn';
      btn.textContent = act.label || t('spl.btn_op');
      btn.disabled = !!act.disabled;
      if (!btn.disabled && act.onClick) {
        btn.addEventListener('click', (ev) => { ev.stopPropagation(); act.onClick(); });
      }
      actionRow.appendChild(btn);
    }
    root.appendChild(actionRow);
  }
  return root;
}

// --- Main board render ---

function renderEmptyBoard(container) {
  const wrap = document.createElement('div');
  wrap.className = 'splendor-board';

  const bankSection = document.createElement('div');
  bankSection.className = 'bank-area';
  bankSection.innerHTML = '<h3>' + t('spl.bank_title') + '</h3>';
  const bankRow = document.createElement('div');
  bankRow.className = 'bank-row';
  for (let c = 0; c < 6; c++) bankRow.appendChild(tokenChip(c, 0, {}));
  bankSection.appendChild(bankRow);
  wrap.appendChild(bankSection);

  const tableau = document.createElement('div');
  tableau.className = 'tableau-wrap';
  tableau.innerHTML = '<h3>' + t('spl.tableau_title') + '</h3>';
  const grid = document.createElement('div');
  grid.className = 'tableau-grid-layout';
  for (let tier = 2; tier >= 0; tier--) {
    const line = document.createElement('div');
    line.className = 'tableau-tier-line';
    const cardRow = document.createElement('div');
    cardRow.className = 'card-row';
    for (let i = 0; i < 4; i++) cardRow.appendChild(renderDevCard(null));
    line.appendChild(cardRow);
    grid.appendChild(line);
  }
  tableau.appendChild(grid);
  wrap.appendChild(tableau);

  const nobles = document.createElement('div');
  nobles.className = 'nobles-area';
  nobles.innerHTML = '<h3>' + t('spl.nobles_title') + '</h3><div class="nobles-row"></div>';
  wrap.appendChild(nobles);

  container.appendChild(wrap);
}

function renderBoard(container, gameState, ctx) {
  currentCtx = ctx;
  currentGameState = gameState;
  container.innerHTML = '';

  if (!gameState || !gameState.state) {
    renderEmptyBoard(container);
    return;
  }

  const game = gameState.state;
  const legalSet = getLegalSet(gameState);
  const playing = ctx.canPlay;
  const stage = getStage(gameState);
  const inReturn = stage === 1;
  const inNoble = stage === 2;

  if (!playing || inReturn || inNoble) gemPick = [];

  const wrap = document.createElement('div');
  wrap.className = 'splendor-board';

  // Turn badge
  const turnBadge = document.createElement('div');
  turnBadge.className = 'turn-badge';
  const plies = game.plies || 0;
  turnBadge.textContent = t('spl.turn_n', { n: Math.floor(plies / 2) + 1 });

  // Bank
  const bankSection = document.createElement('div');
  bankSection.className = 'bank-area';
  const bankHead = document.createElement('div');
  bankHead.className = 'bank-head-line';
  bankHead.innerHTML = '<h3>' + t('spl.bank_title') + '</h3>';
  bankHead.appendChild(turnBadge);
  bankSection.appendChild(bankHead);

  const bankRow = document.createElement('div');
  bankRow.className = 'bank-row';
  const bank = game.bank || [0, 0, 0, 0, 0, 0];
  for (let c = 0; c < 6; c++) {
    const selectable = c < 5 && playing && !inNoble &&
      (inReturn ? legalSet.has(RETURN_TOKEN + c) : canStartWithColor(c, legalSet));
    const selected = gemPick.includes(c);
    const chip = tokenChip(c, bank[c], {
      onClick: selectable ? (() => { const cc = c; return () => onBankGemClick(cc, gameState, ctx); })() : null,
      selected,
    });
    chip.dataset.bankGem = c;
    bankRow.appendChild(chip);
  }
  bankSection.appendChild(bankRow);

  const pickStatus = document.createElement('div');
  pickStatus.className = 'muted inline-title';
  if (inReturn) pickStatus.textContent = t('spl.return_mode');
  else if (inNoble) pickStatus.textContent = t('spl.noble_mode');
  else if (!playing) pickStatus.textContent = t('spl.not_your_turn');
  else if (!hasAnyTakeAction(legalSet)) pickStatus.textContent = t('spl.no_take');
  else if (gemPick.length === 0) pickStatus.textContent = t('spl.pick_hint');
  else pickStatus.textContent = t('spl.picked', { cs: gemPick.map(c => colorLabel(c)).join('+') });
  bankSection.appendChild(pickStatus);

  const opsRow = document.createElement('div');
  opsRow.className = 'inline-ops';
  const takeAid = resolveTakeActionId(gemPick, legalSet);

  const confirmBtn = document.createElement('button');
  confirmBtn.type = 'button';
  confirmBtn.className = 'inline-op-btn';
  confirmBtn.textContent = t('spl.confirm_take');
  confirmBtn.disabled = !(playing && !inReturn && !inNoble && takeAid != null);
  confirmBtn.addEventListener('click', () => { if (takeAid != null) ctx.submitAction(takeAid); });

  const cancelBtn = document.createElement('button');
  cancelBtn.type = 'button';
  cancelBtn.className = 'inline-op-btn';
  cancelBtn.textContent = t('spl.cancel_pick');
  cancelBtn.disabled = gemPick.length === 0 || inNoble;
  cancelBtn.addEventListener('click', () => { gemPick = []; ctx.rerender(); });

  const passBtn = document.createElement('button');
  passBtn.type = 'button';
  passBtn.className = 'inline-op-btn';
  passBtn.textContent = t('spl.pass_btn');
  passBtn.disabled = !(playing && legalSet.has(PASS_ACTION) && gemPick.length === 0);
  passBtn.addEventListener('click', () => ctx.submitAction(PASS_ACTION));

  opsRow.append(confirmBtn, cancelBtn, passBtn);
  bankSection.appendChild(opsRow);

  const pendingMsg = document.createElement('div');
  pendingMsg.className = 'muted';
  if (inReturn) pendingMsg.textContent = t('spl.pending_returns', { n: game.pending_returns || 0 });
  else if (inNoble) pendingMsg.textContent = t('spl.pending_noble');
  bankSection.appendChild(pendingMsg);

  wrap.appendChild(bankSection);

  // Nobles
  const noblesArea = document.createElement('div');
  noblesArea.className = 'nobles-area';
  noblesArea.innerHTML = '<h3>' + t('spl.nobles_title') + '</h3>';
  const noblesRow = document.createElement('div');
  noblesRow.className = 'nobles-row';
  const nobles = game.nobles || [];
  for (let ni = 0; ni < nobles.length; ni++) {
    const noble = nobles[ni];
    const card = document.createElement('div');
    card.className = 'noble-card';
    card.dataset.noble = ni;
    const selectable = playing && inNoble && legalSet.has(CHOOSE_NOBLE + ni);
    if (selectable) {
      card.classList.add('clickable');
      card.addEventListener('click', ((slot) => () => ctx.submitAction(CHOOSE_NOBLE + slot))(ni));
    }
    card.innerHTML = '<div class="noble-title">' + t('spl.noble_3pt') + '</div>';
    const reqRow = document.createElement('div');
    reqRow.className = 'cost-row';
    const req = noble.requirements || [];
    for (let c = 0; c < 5; c++) {
      const v = req[c] || 0;
      if (v <= 0) continue;
      const pill = document.createElement('span');
      pill.className = 'cost-gem ' + COLOR_NAMES[c];
      pill.textContent = String(v);
      reqRow.appendChild(pill);
    }
    card.appendChild(reqRow);
    if (selectable) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'card-op-btn';
      btn.textContent = t('spl.btn_choose_noble');
      btn.addEventListener('click', (ev) => { ev.stopPropagation(); ctx.submitAction(CHOOSE_NOBLE + ni); });
      card.appendChild(btn);
    }
    noblesRow.appendChild(card);
  }
  noblesArea.appendChild(noblesRow);
  wrap.appendChild(noblesArea);

  // Tableau
  const tableauWrap = document.createElement('div');
  tableauWrap.className = 'tableau-wrap';
  tableauWrap.innerHTML = '<h3>' + t('spl.tableau_full') + '</h3>';
  const tableauLayout = document.createElement('div');
  tableauLayout.className = 'tableau-grid-layout';
  const tableau = game.tableau || [[], [], []];
  const deckSizes = game.deck_sizes || [0, 0, 0];

  // Reconcile the engine's packed tableau against the previous screen
  // layout: surviving cards keep their screen slot, taken slots stay
  // empty until filled by the next render's incoming card. Snapshot the
  // pre-render layout into lastScreenLayout BEFORE updating cache, so
  // describeTransition can resolve packed-slot → screen-slot for the
  // pre-action DOM that's still on screen during animation.
  lastScreenLayout = screenLayoutCache.map(r => (r ? r.slice() : null));
  const screenLayout = reconcileScreenLayout(tableau);
  screenLayoutCache = screenLayout;

  for (let ti = 2; ti >= 0; ti--) {
    const line = document.createElement('div');
    line.className = 'tableau-tier-line';

    const reserveDeckAid = RESERVE_DECK + ti;
    const deckBtn = document.createElement('button');
    deckBtn.type = 'button';
    deckBtn.className = 'deck-op-btn deck-op-btn-left';
    deckBtn.textContent = t('spl.deck_btn', { n: deckSizes[ti] });
    deckBtn.disabled = !(playing && legalSet.has(reserveDeckAid));
    deckBtn.dataset.deck = ti;
    deckBtn.addEventListener('click', () => ctx.submitAction(reserveDeckAid));
    line.appendChild(deckBtn);

    const tier = document.createElement('div');
    tier.className = 'tableau-tier';
    tier.innerHTML = '<div class="tableau-tier-title">' + t('spl.tier_label', { n: ti + 1 }) + '</div>';
    const row = document.createElement('div');
    row.className = 'card-row';
    // Iterate the 4 stable screen slots. Empty slots (taken cards
    // whose deck didn't refill yet) render as placeholders. Action
    // IDs come from the cell's packedSlot — the engine's action ID
    // semantics are unchanged.
    const tierLayout = screenLayout[ti] || emptyTierLayout();
    for (let s = 0; s < 4; s++) {
      const cell = tierLayout[s];
      const card = cell ? cell.card : null;
      const isEmpty = !card || card.card_id == null || card.card_id < 0;
      const packedSlot = cell ? cell.packedSlot : -1;
      const buyAid = packedSlot >= 0 ? BUY_FACEUP + ti * 4 + packedSlot : -1;
      const resAid = packedSlot >= 0 ? RESERVE_FACEUP + ti * 4 + packedSlot : -1;
      const buyOk = !isEmpty && playing && buyAid >= 0 && legalSet.has(buyAid);
      const resOk = !isEmpty && playing && resAid >= 0 && legalSet.has(resAid);
      const cardEl = isEmpty ? renderDevCard(null) : renderDevCard(card, [
        { label: t('spl.btn_buy'), disabled: !buyOk, onClick: buyOk ? () => ctx.submitAction(buyAid) : null },
        { label: t('spl.btn_reserve'), disabled: !resOk, onClick: resOk ? () => ctx.submitAction(resAid) : null },
      ]);
      // data-tableau anchors are KEYED BY SCREEN SLOT — animations
      // and CSS layout depend on stable screen positions.
      cardEl.dataset.tableau = ti + '-' + s;
      row.appendChild(cardEl);
    }
    tier.appendChild(row);
    line.appendChild(tier);
    tableauLayout.appendChild(line);
  }
  tableauWrap.appendChild(tableauLayout);
  wrap.appendChild(tableauWrap);

  container.appendChild(wrap);
}

function onBankGemClick(colorIdx, gameState, ctx) {
  const legalSet = getLegalSet(gameState);
  const stage = getStage(gameState);
  if (stage === 1) {
    const returnAid = RETURN_TOKEN + colorIdx;
    if (legalSet.has(returnAid)) ctx.submitAction(returnAid);
    return;
  }
  if (stage === 2) return;
  if (!hasAnyTakeAction(legalSet)) return;
  if (!canAppendColor(colorIdx, legalSet)) return;
  gemPick = gemPick.concat([colorIdx]);
  ctx.rerender();
}

// --- Player area ---

function renderPlayerArea(container, gameState, ctx) {
  container.innerHTML = '';
  if (!gameState || !gameState.state) {
    const outer = document.createElement('div');
    outer.className = 'splendor-players-wrap';
    const wrap = document.createElement('div');
    wrap.className = 'players-wrap';
    for (let i = 0; i < 2; i++) {
      const card = document.createElement('div');
      card.className = 'player-card';
      card.innerHTML = '<div class="player-title"><span>' + t('spl.player_n', { n: i }) + '</span></div>';
      wrap.appendChild(card);
    }
    outer.appendChild(wrap);
    container.appendChild(outer);
    return;
  }

  const game = gameState.state;
  const appState = ctx.state;
  const legalSet = getLegalSet(gameState);
  const playing = ctx.canPlay;
  const players = game.players || [];
  const cp = gameState.current_player;

  const outer = document.createElement('div');
  outer.className = 'splendor-players-wrap';

  const wrap = document.createElement('div');
  wrap.className = 'players-wrap';

  const numPlayers = gameState.num_players || 2;
  if (numPlayers > 2) wrap.classList.add('players-multi');

  for (let p = 0; p < numPlayers; p++) {
    const pd = players[p] || {};
    const isHuman = p === appState.humanPlayer;
    const isCurrent = p === cp;

    const card = document.createElement('div');
    card.className = 'player-card' + (isCurrent ? ' active-turn' : '');
    card.dataset.player = p;

    const title = document.createElement('div');
    title.className = 'player-title';
    title.innerHTML = '<span>' + t('spl.player_n', { n: p }) + (isHuman ? t('spl.player_self_suffix') : '') + (isCurrent ? t('spl.player_current_suffix') : '') + '</span>' +
      '<span class="player-title-meta">' + t('spl.stat_meta', { points: pd.points || 0, cards: pd.cards_count || 0, nobles: pd.nobles_count || 0 }) + '</span>';
    card.appendChild(title);

    // Gems
    const gemsGroup = document.createElement('div');
    gemsGroup.className = 'player-stat-group';
    const gems = pd.gems || [];
    const gemsTotal = gems.reduce((s, v) => s + (v || 0), 0);
    gemsGroup.innerHTML = '<div class="player-stat-label">' + t('spl.gems_label') + ' <span class="player-stat-extra">' + t('spl.gems_total', { n: gemsTotal }) + '</span></div>';
    const gemsRow = document.createElement('div');
    gemsRow.className = 'token-row';
    for (let c = 0; c < 6; c++) {
      const chip = tokenChip(c, gems[c] || 0);
      chip.dataset.playerGem = p + '-' + c;
      gemsRow.appendChild(chip);
    }
    gemsGroup.appendChild(gemsRow);
    card.appendChild(gemsGroup);

    // Bonuses
    const bonusGroup = document.createElement('div');
    bonusGroup.className = 'player-stat-group';
    bonusGroup.innerHTML = '<div class="player-stat-label">' + t('spl.bonus_label') + '</div>';
    const bonusRow = document.createElement('div');
    bonusRow.className = 'bonus-row';
    const bonuses = pd.bonuses || [];
    for (let c = 0; c < 5; c++) {
      const chip = document.createElement('span');
      chip.className = 'bonus-rect ' + COLOR_NAMES[c];
      chip.textContent = colorLabel(c) + ':' + (bonuses[c] || 0);
      bonusRow.appendChild(chip);
    }
    bonusGroup.appendChild(bonusRow);
    card.appendChild(bonusGroup);

    // Reserved — always render 3 fixed slots so the card height doesn't jump
    // when the player reserves a card mid-game.
    const reserveGroup = document.createElement('div');
    reserveGroup.className = 'player-stat-group reserve-group';
    reserveGroup.innerHTML = '<div class="player-stat-label">' + t('spl.reserve_label') + '</div>';
    const reserveRow = document.createElement('div');
    reserveRow.className = 'reserve-row';
    const reserved = pd.reserved || [];
    for (let ri = 0; ri < 3; ri++) {
      const item = reserved[ri];
      if (!item) {
        const empty = document.createElement('div');
        empty.className = 'reserve-slot-empty';
        empty.dataset.reserved = p + '-' + ri;
        empty.textContent = t('spl.reserve_empty');
        reserveRow.appendChild(empty);
        continue;
      }
      if (!item.visible && !isHuman && !ctx.revealHidden) {
        // Face-down reserve (drawn from deck) — kept hidden even when
        // playing for the AI seat via 替对手落子, since we mustn't reveal
        // its identity. Buy_reserved on a hidden slot is therefore not
        // exposed; the human can still play other actions for the AI.
        // Replay-mode reveal-hidden bypasses this gate so the user can
        // see the AI's face-down picks during review.
        const hidden = document.createElement('div');
        hidden.className = 'reserve-hidden';
        hidden.dataset.reserved = p + '-' + ri;
        hidden.textContent = t('spl.reserve_hidden', { tier: item.tier || '?' });
        reserveRow.appendChild(hidden);
        continue;
      }
      const buyResAid = BUY_RESERVED + ri;
      // canBuy: own seat normally, OR AI seat during 替对手落子 (force mode
      // makes ctx.canPlay true while current_player is the AI seat).
      // The visibility gate above already ensures we never offer a buy
      // for a face-down reserve.
      const isActingSeat = playing && p === cp;
      const canBuy = isActingSeat && legalSet.has(buyResAid);
      const actions = canBuy ? [{ label: t('spl.btn_buy_reserved'), onClick: () => ctx.submitAction(buyResAid) }] : [];
      const resCardEl = renderDevCard(item, actions);
      resCardEl.dataset.reserved = p + '-' + ri;
      reserveRow.appendChild(resCardEl);
    }
    reserveGroup.appendChild(reserveRow);
    card.appendChild(reserveGroup);

    wrap.appendChild(card);
  }

  outer.appendChild(wrap);
  container.appendChild(outer);
}

function formatMove(actionInfo, actionId) {
  if (!actionInfo && (actionId === null || actionId === undefined)) return t('spl.action_start');
  return formatActionInfo(actionInfo, actionId);
}

// Smart-hint formatter — appends the recommended card's gem details
// (bonus color, cost colors, points) when the suggestion is a card
// action, so the user can spot misaligned tableau slots immediately.
function formatSuggestion(actionInfo, actionId, actorIdx) {
  const base = formatMove(actionInfo, actionId);
  if (!actionInfo || !actionInfo.type) return base;
  let card = null;
  if (actionInfo.type === 'buy_faceup' || actionInfo.type === 'reserve_faceup') {
    card = lookupTableauCard(actionInfo);
  } else if (actionInfo.type === 'buy_reserved') {
    card = lookupReservedCard(actionInfo, actorIdx);
  }
  if (!card) return base;
  return base + formatCardDebug(card);
}

// --- Animation ---

function makeFlyingToken(colorIdx) {
  const el = document.createElement('div');
  el.className = 'anim-flying-token ' + COLOR_NAMES[colorIdx];
  el.textContent = colorLabel(colorIdx);
  return el;
}

// Clone a live card DOM element (dev-card / noble-card) so the flyer
// shows the full card visual instead of a blank rectangle. Strips
// interactive affordances (buttons, hover hooks) so the cloned sprite
// is purely decorative mid-flight.
function cloneCardEl(selector) {
  const src = document.querySelector(selector);
  if (!src) return null;
  const clone = src.cloneNode(true);
  clone.querySelectorAll('button').forEach(b => b.remove());
  clone.classList.remove('clickable');
  clone.style.pointerEvents = 'none';
  clone.style.margin = '0';
  return clone;
}

const CARD_FLY_MS = 550;
const TOKEN_FLY_MS = 400;

function makeHiddenCardFlyer(tier) {
  const el = document.createElement('div');
  el.className = 'dev-card dev-card-hidden';
  el.textContent = 'T' + ((tier || 0) + 1);
  return el;
}

// After a reserve fly finishes, the destination slot in the DOM is still
// the "empty" placeholder — and the flyer sprite is removed by animate.js.
// If a *subsequent* animation step plays (e.g. the deck→tableau refill),
// the user sees a gap where the just-reserved card should be. Paint the
// slot into a persistent static representation so the intermediate state
// survives until the caller re-renders the real DOM. See §3 "中间状态维护"
// in docs/guide/WEB_DESIGN_PRINCIPLES.md.
function paintReserveSlotFromCard(selector, sourceCardSelector) {
  const slot = document.querySelector(selector);
  if (!slot) return;
  const src = document.querySelector(sourceCardSelector);
  if (!src) return;
  // Replace the empty placeholder class so the slot renders as a card
  // (.reserve-slot-empty has dashed borders + "空" text that would show
  // through otherwise). Copy the source card's innerHTML — contents are
  // the visible card face — and strip interactive affordances.
  slot.className = 'dev-card';
  slot.innerHTML = src.innerHTML;
  slot.querySelectorAll('button').forEach(b => b.remove());
  slot.classList.remove('clickable');
  slot.style.pointerEvents = 'none';
}

function paintReserveSlotHidden(selector, tier) {
  const slot = document.querySelector(selector);
  if (!slot) return;
  slot.className = 'dev-card dev-card-hidden';
  slot.textContent = 'T' + ((tier || 0) + 1);
}

function refillStepsFromDeck(prev, next) {
  // When a tableau slot is taken (buy_faceup / reserve_faceup), the
  // engine deals a replacement from the deck. The packed JSON shifts
  // entries so we can't read positions directly — find the new card_id
  // (in next, not in prev), then look up which SCREEN slot it lands on
  // by simulating a reconcile against the same prev screen layout the
  // current render started from (lastScreenLayout).
  const refills = [];
  const prevDecks = prev.deck_sizes || [];
  const nextDecks = next.deck_sizes || [];
  for (let t = 0; t < 3; t++) {
    const drew = (prevDecks[t] || 0) - (nextDecks[t] || 0);
    if (drew <= 0) continue;
    const prevRow = (prev.tableau && prev.tableau[t]) || [];
    const nextRow = (next.tableau && next.tableau[t]) || [];
    const prevIds = new Set(prevRow
      .map(c => (c && c.card_id != null) ? c.card_id : -1)
      .filter(id => id >= 0));
    let newCardId = -1;
    for (const c of nextRow) {
      const id = (c && c.card_id != null) ? c.card_id : -1;
      if (id >= 0 && !prevIds.has(id)) { newCardId = id; break; }
    }
    if (newCardId < 0) continue;
    // Predict the post-reconcile screen layout by running the same
    // reconciliation the renderer will do next, against the layout
    // currently on screen (screenLayoutCache — describeTransition runs
    // before the next render, so cache still holds prev's layout).
    const predicted = reconcileTierLayout(screenLayoutCache[t], nextRow);
    let screenSlot = -1;
    for (let s = 0; s < 4; s++) {
      const cell = predicted[s];
      if (cell && cell.card && cell.card.card_id === newCardId) { screenSlot = s; break; }
    }
    if (screenSlot < 0) continue;
    refills.push({
      from: `[data-deck="${t}"]`,
      to: `[data-tableau="${t}-${screenSlot}"]`,
      createElement: () => makeHiddenCardFlyer(t),
      duration: CARD_FLY_MS,
    });
  }
  return refills;
}

function describeTransition(prevState, newState, actionInfo, actionId) {
  if (!actionInfo || !actionInfo.type) return null;
  if (!prevState || !prevState.state || !newState || !newState.state) return null;

  const steps = [];
  const actor = prevState.current_player;
  const prev = prevState.state;
  const next = newState.state;

  switch (actionInfo.type) {
    case 'take_three':
    case 'take_two_different': {
      const colors = actionInfo.colors || [];
      const flights = colors.map(c => ({
        from: `[data-bank-gem="${c}"]`,
        to: `[data-player-gem="${actor}-${c}"]`,
        createElement: () => makeFlyingToken(c),
        duration: TOKEN_FLY_MS,
      }));
      if (flights.length) steps.push({ type: 'flyGroup', flights });
      break;
    }

    case 'take_one': {
      const c = actionInfo.color;
      if (c != null) {
        steps.push({
          type: 'fly',
          from: `[data-bank-gem="${c}"]`,
          to: `[data-player-gem="${actor}-${c}"]`,
          createElement: () => makeFlyingToken(c),
          duration: TOKEN_FLY_MS,
        });
      }
      break;
    }

    case 'take_two_same': {
      const c = actionInfo.color;
      if (c != null) {
        const flights = [0, 1].map(() => ({
          from: `[data-bank-gem="${c}"]`,
          to: `[data-player-gem="${actor}-${c}"]`,
          createElement: () => makeFlyingToken(c),
          duration: TOKEN_FLY_MS,
        }));
        steps.push({ type: 'flyGroup', flights });
      }
      break;
    }

    case 'buy_faceup': {
      const gemFlights = gemReturnFlights(prev, next, actor);
      // The action's `slot` is a packed-array index against `prev`.
      // Translate to the SCREEN slot rendered for the pre-action board
      // (lastScreenLayout, captured at the top of the current render).
      const screenSlot = packedSlotToScreenInPrev(actionInfo.tier, actionInfo.slot);
      const fromSel = `[data-tableau="${actionInfo.tier}-${screenSlot}"]`;
      const cardFlight = {
        from: fromSel,
        to: `[data-player="${actor}"] .bonus-row`,
        createElement: () => cloneCardEl(fromSel),
        hideFrom: true,
        duration: CARD_FLY_MS,
      };
      steps.push({ type: 'flyGroup', flights: [...gemFlights, cardFlight] });
      // After the taken slot is hidden, fly a replacement from the deck
      // into the same slot so players see the refill explicitly.
      const refills = refillStepsFromDeck(prev, next);
      if (refills.length) steps.push({ type: 'flyGroup', flights: refills });
      break;
    }

    case 'buy_reserved': {
      const gemFlights = gemReturnFlights(prev, next, actor);
      const fromSel = `[data-reserved="${actor}-${actionInfo.slot}"]`;
      const cardFlight = {
        from: fromSel,
        to: `[data-player="${actor}"] .bonus-row`,
        createElement: () => cloneCardEl(fromSel),
        hideFrom: true,
        duration: CARD_FLY_MS,
      };
      steps.push({ type: 'flyGroup', flights: [...gemFlights, cardFlight] });
      break;
    }

    case 'reserve_faceup': {
      // Find the empty reserve slot that will receive the card in the
      // next state. Reserved cards are appended, so the target slot
      // index is the count before reserving.
      const prevReserved = (prev.players[actor].reserved || []).length;
      const screenSlot = packedSlotToScreenInPrev(actionInfo.tier, actionInfo.slot);
      const fromSel = `[data-tableau="${actionInfo.tier}-${screenSlot}"]`;
      const toSel = `[data-player="${actor}"] [data-reserved="${actor}-${prevReserved}"]`;
      // Keep a pre-flight clone around so that after the tableau source is
      // hidden we can still reconstruct the reserved card into the empty
      // slot during onComplete (prevents the "reserved card disappears
      // mid-deck-refill" bug — see §3 中间状态维护 in web-design-principles).
      const preservedCardSel = fromSel;
      const cardFlight = {
        from: fromSel,
        to: toSel,
        createElement: () => cloneCardEl(fromSel),
        hideFrom: true,
        duration: CARD_FLY_MS,
        onComplete: () => paintReserveSlotFromCard(toSel, preservedCardSel),
      };
      const flights = [cardFlight];
      const prevGold = (prev.players[actor].gems || [])[5] || 0;
      const nextGold = (next.players[actor].gems || [])[5] || 0;
      if (nextGold > prevGold) {
        flights.push({
          from: '[data-bank-gem="5"]',
          to: `[data-player-gem="${actor}-5"]`,
          createElement: () => makeFlyingToken(5),
          duration: TOKEN_FLY_MS,
        });
      }
      steps.push({ type: 'flyGroup', flights });
      // Refill the vacated tableau slot from the deck.
      const refills = refillStepsFromDeck(prev, next);
      if (refills.length) steps.push({ type: 'flyGroup', flights: refills });
      break;
    }

    case 'reserve_deck': {
      const prevReserved = (prev.players[actor].reserved || []).length;
      const toSel = `[data-player="${actor}"] [data-reserved="${actor}-${prevReserved}"]`;
      const tier = actionInfo.tier;
      const flights = [{
        from: `[data-deck="${actionInfo.tier}"]`,
        to: toSel,
        createElement: () => makeHiddenCardFlyer(tier),
        duration: CARD_FLY_MS,
        // Keep a hidden-card placeholder in the reserve slot so later
        // flights (e.g. gold coin from the bank) don't leave a visual gap
        // where the reserved card should be. Opponents' reserves from the
        // deck are rendered as `.reserve-hidden` in steady state, but for
        // the intermediate animation state using the same hidden-card
        // visual is fine — re-render replaces it anyway.
        onComplete: () => paintReserveSlotHidden(toSel, tier),
      }];
      const prevGold = (prev.players[actor].gems || [])[5] || 0;
      const nextGold = (next.players[actor].gems || [])[5] || 0;
      if (nextGold > prevGold) {
        flights.push({
          from: '[data-bank-gem="5"]',
          to: `[data-player-gem="${actor}-5"]`,
          createElement: () => makeFlyingToken(5),
          duration: TOKEN_FLY_MS,
        });
      }
      steps.push({ type: 'flyGroup', flights });
      break;
    }

    case 'choose_noble': {
      const fromSel = `[data-noble="${actionInfo.noble_slot}"]`;
      steps.push({
        type: 'fly',
        from: fromSel,
        to: `[data-player="${actor}"]`,
        createElement: () => cloneCardEl(fromSel),
        hideFrom: true,
        duration: CARD_FLY_MS,
      });
      break;
    }

    case 'return_token': {
      const c = actionInfo.token;
      if (c != null) {
        steps.push({
          type: 'fly',
          from: `[data-player-gem="${actor}-${c}"]`,
          to: `[data-bank-gem="${c}"]`,
          createElement: () => makeFlyingToken(c),
          duration: TOKEN_FLY_MS,
        });
      }
      break;
    }
  }

  return steps.length ? steps : null;
}

function gemReturnFlights(prev, next, actor) {
  const flights = [];
  const prevGems = prev.players[actor].gems || [];
  const nextGems = next.players[actor].gems || [];
  for (let c = 0; c < 6; c++) {
    const spent = (prevGems[c] || 0) - (nextGems[c] || 0);
    for (let i = 0; i < spent; i++) {
      flights.push({
        from: `[data-player-gem="${actor}-${c}"]`,
        to: `[data-bank-gem="${c}"]`,
        createElement: () => makeFlyingToken(c),
        duration: TOKEN_FLY_MS,
      });
    }
  }
  return flights;
}

function gemReturnSteps(steps, prev, next, actor) {
  const prevGems = prev.players[actor].gems || [];
  const nextGems = next.players[actor].gems || [];
  for (let c = 0; c < 6; c++) {
    const spent = (prevGems[c] || 0) - (nextGems[c] || 0);
    if (spent > 0) {
      steps.push({
        type: 'fly',
        from: `[data-player-gem="${actor}-${c}"]`,
        to: `[data-bank-gem="${c}"]`,
        createElement: () => makeFlyingToken(c),
        duration: TOKEN_FLY_MS,
      });
    }
  }
}

createApp({
  gameId: 'splendor',
  gameTitle: t('spl.title'),
  gameIntro: t('spl.intro'),
  players: { min: 2, max: 4 },
  renderBoard,
  renderPlayerArea,
  describeTransition,
  formatOpponentMove: formatMove,
  formatSuggestedMove: formatSuggestion,
  getPlayerSymbol: (humanPlayer) => t('spl.player_n', { n: humanPlayer }),
  difficulties: ['heuristic', 'casual', 'expert'],
  defaultDifficulty: 'expert',
});
