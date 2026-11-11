import { createApp } from '/static/general/app.js';
import { t, register } from '/static/general/i18n.js';

register({
  zh: {
    'sn.title': '牛头王',
    'sn.intro': '点击手牌出牌；若你的牌比所有行尾都小，需选择一行吃下',
    'sn.player_n': '玩家{n}',
    'sn.player_self_suffix': '（你）',
    'sn.player_score_suffix': ' · 牛头：{n}',
    'sn.action_unknown': '动作 #{id}',
    'sn.action_start': '开局',
    'sn.action_commit': '出 {card}',
    'sn.action_commit_hidden': '出牌',
    'sn.action_small': '吃第{row}行',
    'sn.action_invalid': '无效动作',
    'sn.round_label': '第 {n} / 10 轮',
    'sn.round_default': '第 -- / 10 轮',
    'sn.committed_label': '已出 {n} / {total}',
    'sn.small_prompt': '你的牌比所有行尾都小 — 选择一行吃下',
    'sn.small_other_prompt': '玩家{n} 必须选择吃下的行',
    'sn.row_n': '第{n}行',
    'sn.row_label_short': '行 {n}',
    'sn.bullhead_short': '牛头 ×{n}',
    'sn.head_label': '牛',
    'sn.commit_area_label': '本轮出牌',
    'sn.bullhead_pile_label': '牛头堆 ×{n}',
  },
  en: {
    'sn.title': '6 nimmt!',
    'sn.intro': 'Click a hand card to play; if smaller than every row tail, pick a row to eat.',
    'sn.player_n': 'Player {n}',
    'sn.player_self_suffix': ' (you)',
    'sn.player_score_suffix': ' · bullheads: {n}',
    'sn.action_unknown': 'Action #{id}',
    'sn.action_start': 'Game start',
    'sn.action_commit': 'play {card}',
    'sn.action_commit_hidden': 'played a card',
    'sn.action_small': 'eat row {row}',
    'sn.action_invalid': 'invalid action',
    'sn.round_label': 'Round {n} / 10',
    'sn.round_default': 'Round -- / 10',
    'sn.committed_label': 'Committed {n} / {total}',
    'sn.small_prompt': 'Your card is below every row tail — pick a row to eat.',
    'sn.small_other_prompt': 'Player {n} must pick a row to eat.',
    'sn.row_n': 'Row {n}',
    'sn.row_label_short': 'Row {n}',
    'sn.bullhead_short': 'Heads x{n}',
    'sn.head_label': 'Bull',
    'sn.commit_area_label': 'This round',
    'sn.bullhead_pile_label': 'Bullheads x{n}',
  },
});

const NUM_CARDS = 104;
const NUM_ROWS = 4;
const ROW_CAPACITY = 5;
const COMMIT_OFFSET = 0;
const SMALL_OFFSET = NUM_CARDS;

function headValue(card) {
  if (card < 1 || card > NUM_CARDS) return 0;
  let h = 0;
  if (card % 5 === 0) h += 2;
  if (card % 10 === 0) h += 1;
  if (card % 11 === 0) h += 5;
  if (h === 0) h = 1;
  return h;
}

function commitAction(card) { return COMMIT_OFFSET + (card - 1); }
function smallRowAction(row) { return SMALL_OFFSET + row; }

// What the server shows for an opponent's commit that is still face down
// (kHiddenCommitAction): the card id only arrives once it is revealed.
const HIDDEN_COMMIT = NUM_CARDS + NUM_ROWS;

function classifyAction(actionId) {
  if (actionId == null) return null;
  if (actionId === HIDDEN_COMMIT) return { type: 'commit', hidden: true };
  if (actionId >= COMMIT_OFFSET && actionId < COMMIT_OFFSET + NUM_CARDS) {
    return { type: 'commit', card: actionId - COMMIT_OFFSET + 1 };
  }
  if (actionId >= SMALL_OFFSET && actionId < SMALL_OFFSET + NUM_ROWS) {
    return { type: 'small_choose_row', row: actionId - SMALL_OFFSET };
  }
  return null;
}

function formatMove(info, actionId) {
  let kind = info;
  if (!kind) kind = classifyAction(actionId);
  if (!kind) {
    if (actionId !== null && actionId !== undefined) return t('sn.action_unknown', { id: actionId });
    return t('sn.action_start');
  }
  const type = String(kind.type || '');
  if (type === 'commit') return t('sn.action_commit_hidden');
  if (type === 'small_choose_row') return t('sn.action_small', { row: kind.row + 1 });
  if (actionId !== null && actionId !== undefined) return t('sn.action_unknown', { id: actionId });
  return t('sn.action_start');
}

function makeCardEl(card, opts) {
  opts = opts || {};
  const el = document.createElement(opts.clickable ? 'button' : 'div');
  if (opts.clickable) el.type = 'button';
  el.className = 'sn-card';
  if (opts.clickable) el.classList.add('sn-card-btn');
  if (opts.legal) el.classList.add('sn-card-legal');
  if (opts.dimmed) el.classList.add('sn-card-dimmed');
  if (opts.faceDown) el.classList.add('sn-card-back');
  if (opts.mini) el.classList.add('sn-card-mini');
  if (card == null || card <= 0) {
    el.classList.add('sn-card-empty');
    return el;
  }
  const num = document.createElement('span');
  num.className = 'sn-card-num';
  num.textContent = String(card);
  el.appendChild(num);
  const heads = headValue(card);
  const badge = document.createElement('span');
  badge.className = 'sn-card-heads sn-heads-' + heads;
  badge.textContent = '×' + heads;
  el.appendChild(badge);
  if (opts.onClick) el.addEventListener('click', opts.onClick);
  return el;
}

function makeFlyingCard(card) {
  const el = document.createElement('div');
  el.className = 'sn-card sn-card-flying';
  if (card == null || card <= 0) {
    el.classList.add('sn-card-back');
    return el;
  }
  const num = document.createElement('span');
  num.className = 'sn-card-num';
  num.textContent = String(card);
  el.appendChild(num);
  const heads = headValue(card);
  const badge = document.createElement('span');
  badge.className = 'sn-card-heads sn-heads-' + heads;
  badge.textContent = '×' + heads;
  el.appendChild(badge);
  return el;
}

function bullheadCount(pd) {
  return pd && pd.bullhead_count != null ? pd.bullhead_count : 0;
}

function bullheadCards(pd) {
  if (!pd || !Array.isArray(pd.bullhead_pile)) return [];
  return pd.bullhead_pile.slice().sort((a, b) => a - b);
}

// Render the persistent commit area at the top of the board.
// Always 4 fixed slots so the container doesn't shift.
function renderCommitArea(container, st, numPlayers) {
  const wrap = document.createElement('div');
  wrap.className = 'sn-commit-area';

  const label = document.createElement('div');
  label.className = 'sn-commit-area-label';
  label.textContent = t('sn.commit_area_label');
  wrap.appendChild(label);

  const slotsWrap = document.createElement('div');
  slotsWrap.className = 'sn-commit-slots';

  // Pre-render N fixed slots (hidden seats unused).
  const ac = (st.apply_queue_card) || [];
  const ap = (st.apply_queue_player) || [];
  const cursor = st.apply_cursor != null ? st.apply_cursor : 0;
  // Build per-player committed map (seat -> {card?, applied?, position?}).
  const seatCard = new Array(numPlayers).fill(null);
  const seatApplied = new Array(numPlayers).fill(false);
  for (let i = 0; i < ac.length; i++) {
    const c = ac[i];
    const p = ap[i];
    if (p < 0 || p >= numPlayers) continue;
    if (c > 0) {
      seatCard[p] = c;
      seatApplied[p] = i < cursor;  // already-applied slot means card cleared earlier
    } else if (i < cursor) {
      // Slot has been applied — apply_queue_card was reset to 0 after applying.
      // We don't know the original card from state alone, so just mark applied.
      seatApplied[p] = true;
    }
  }
  // committed flags also tell us "this seat played but card not yet flipped"
  // (commit phase). The serializer puts this per-player as
  // st.players[p].committed (boolean), so build an array view.
  const stPlayers = (st.players) || [];
  const committed = stPlayers.map(pd => !!(pd && pd.committed));

  for (let p = 0; p < numPlayers; p++) {
    const slot = document.createElement('div');
    slot.className = 'sn-commit-slot';
    slot.setAttribute('data-commit-slot', String(p));

    const tag = document.createElement('div');
    tag.className = 'sn-commit-tag-mini';
    tag.textContent = t('sn.player_n', { n: p });
    slot.appendChild(tag);

    if (seatCard[p] != null && !seatApplied[p]) {
      // Card visible in apply queue (post-reveal, pre-apply).
      slot.appendChild(makeCardEl(seatCard[p], {}));
    } else if (seatApplied[p]) {
      // Already applied — fade slot, leave empty (card is now on table or in pile).
      slot.classList.add('applied');
      slot.appendChild(makeCardEl(null, {}));
    } else if (committed[p]) {
      // Commit phase, face-down.
      slot.classList.add('face-down');
      const fd = document.createElement('div');
      fd.className = 'sn-card sn-card-back';
      slot.appendChild(fd);
    } else {
      slot.classList.add('empty');
      slot.appendChild(makeCardEl(null, {}));
    }
    slotsWrap.appendChild(slot);
  }

  wrap.appendChild(slotsWrap);
  container.appendChild(wrap);
}

function renderTable(container, st, ctx, smallTargets) {
  const wrap = document.createElement('div');
  wrap.className = 'sn-table';

  const rows = (st && st.table) || [];
  for (let r = 0; r < NUM_ROWS; r++) {
    const rowEl = document.createElement('div');
    rowEl.className = 'sn-row';
    rowEl.setAttribute('data-row', String(r));

    const label = document.createElement('div');
    label.className = 'sn-row-label';
    label.textContent = t('sn.row_label_short', { n: r + 1 });
    rowEl.appendChild(label);

    const slotsWrap = document.createElement('div');
    slotsWrap.className = 'sn-row-slots';
    const cards = rows[r] || [];
    let rowHeads = 0;
    let rowLen = 0;
    for (let s = 0; s < ROW_CAPACITY; s++) {
      const c = cards[s] || 0;
      const slot = document.createElement('div');
      slot.className = 'sn-slot';
      slot.setAttribute('data-row-slot', r + '-' + s);
      if (c > 0) {
        slot.appendChild(makeCardEl(c, {}));
        rowHeads += headValue(c);
        rowLen++;
      } else {
        slot.classList.add('empty');
        slot.appendChild(makeCardEl(null, {}));
      }
      slotsWrap.appendChild(slot);
    }
    rowEl.appendChild(slotsWrap);

    const headsEl = document.createElement('div');
    headsEl.className = 'sn-row-heads';
    headsEl.textContent = t('sn.bullhead_short', { n: rowHeads });
    rowEl.appendChild(headsEl);

    const isTarget = smallTargets && smallTargets.has(r);
    if (isTarget) {
      rowEl.classList.add('sn-row-eat-target');
      rowEl.addEventListener('click', () => ctx.submitAction(smallRowAction(r)));
    }

    if (rowLen === ROW_CAPACITY) rowEl.classList.add('sn-row-full');

    wrap.appendChild(rowEl);
  }
  container.appendChild(wrap);
}

function renderBoard(container, gameState, ctx) {
  container.innerHTML = '';

  const root = document.createElement('div');
  root.className = 'sn-board-root';

  if (!gameState || !gameState.state) {
    // Empty / pre-game placeholder. Keep dimensions stable so the board
    // looks the same before the first game starts as during play.
    const placeholderPlayers = 2;
    renderCommitArea(root, {}, placeholderPlayers);
    const empty = document.createElement('div');
    empty.className = 'sn-table sn-table-empty';
    for (let r = 0; r < NUM_ROWS; r++) {
      const rowEl = document.createElement('div');
      rowEl.className = 'sn-row';
      const label = document.createElement('div');
      label.className = 'sn-row-label';
      label.textContent = t('sn.row_label_short', { n: r + 1 });
      rowEl.appendChild(label);
      const slotsWrap = document.createElement('div');
      slotsWrap.className = 'sn-row-slots';
      for (let s = 0; s < ROW_CAPACITY; s++) {
        const slot = document.createElement('div');
        slot.className = 'sn-slot empty';
        slot.appendChild(makeCardEl(null, {}));
        slotsWrap.appendChild(slot);
      }
      rowEl.appendChild(slotsWrap);
      empty.appendChild(rowEl);
    }
    root.appendChild(empty);
    container.appendChild(root);
    return;
  }

  const st = gameState.state;
  const numPlayers = gameState.num_players || 2;
  const canPlay = ctx.canPlay;
  const legalSet = new Set(gameState.legal_actions || []);

  // SMALL phase target rows for the acting player.
  const smallTargets = new Set();
  if (canPlay && st.small_phase_active) {
    for (const a of legalSet) {
      if (a >= SMALL_OFFSET && a < SMALL_OFFSET + NUM_ROWS) {
        smallTargets.add(a - SMALL_OFFSET);
      }
    }
  }

  renderCommitArea(root, st, numPlayers);
  renderTable(root, st, ctx, smallTargets);

  container.appendChild(root);
}

function renderPlayerArea(container, gameState, ctx) {
  container.innerHTML = '';
  if (!gameState || !gameState.state) return;

  const st = gameState.state;
  const numPlayers = gameState.num_players || 2;
  const players = st.players || [];
  const appState = ctx.state;
  const humanPlayer = appState.humanPlayer;
  const canPlay = ctx.canPlay;
  const legalSet = new Set(gameState.legal_actions || []);
  const inSmall = !!st.small_phase_active;

  const wrap = document.createElement('div');
  wrap.className = 'sn-players';
  if (numPlayers > 2) wrap.classList.add('sn-players-multi');

  for (let pi = 0; pi < numPlayers; pi++) {
    const pd = players[pi] || {};
    const isHuman = pi === humanPlayer;
    const isCurrent = gameState.current_player === pi;

    const seatEl = document.createElement('div');
    seatEl.className = 'sn-seat' + (isCurrent ? ' active-turn' : '');
    seatEl.setAttribute('data-player', String(pi));
    if (isHuman) seatEl.setAttribute('data-opponent', '0'); else seatEl.setAttribute('data-opponent', '1');

    const headerEl = document.createElement('div');
    headerEl.className = 'sn-seat-header';
    const titleEl = document.createElement('div');
    titleEl.className = 'sn-seat-title';
    const headsCount = bullheadCount(pd);
    titleEl.textContent = t('sn.player_n', { n: pi })
      + (isHuman ? t('sn.player_self_suffix') : '')
      + t('sn.player_score_suffix', { n: headsCount });
    headerEl.appendChild(titleEl);
    seatEl.appendChild(headerEl);

    // Hand row — only the human's hand shows numbers.
    const handEl = document.createElement('div');
    handEl.className = 'sn-hand';
    handEl.setAttribute('data-hand', String(pi));

    // In replay mode with reveal toggled, opponent hands render
     // face-up (truth-bearing — replay drives a single GT session).
    if (isHuman || ctx.revealHidden) {
      const hand = (pd.hand || []).slice().sort((a, b) => a - b);
      const canCommit = canPlay && isCurrent && !inSmall;
      for (const c of hand) {
        const aid = commitAction(c);
        const isLegal = canCommit && legalSet.has(aid);
        const cardEl = makeCardEl(c, {
          clickable: isLegal,
          legal: isLegal,
          dimmed: !isLegal && canCommit,
          onClick: isLegal ? () => ctx.submitAction(aid) : null,
        });
        cardEl.setAttribute('data-card', String(c));
        handEl.appendChild(cardEl);
      }
      if (!hand.length) {
        const placeholder = document.createElement('div');
        placeholder.className = 'sn-hand-empty';
        placeholder.textContent = '—';
        handEl.appendChild(placeholder);
      }
    } else {
      const cnt = pd.hand_count || 0;
      for (let i = 0; i < cnt; i++) {
        const back = document.createElement('div');
        // Use the same full-size sn-card backing as the human's hand so
        // when an opponent commits a card, the flight source/sink stays
        // at the same scale as the rest of the table — no jarring
        // 32×44 → 60×84 jump mid-animation.
        back.className = 'sn-card sn-card-back';
        handEl.appendChild(back);
      }
      if (cnt === 0) {
        const placeholder = document.createElement('div');
        placeholder.className = 'sn-hand-empty';
        placeholder.textContent = '—';
        handEl.appendChild(placeholder);
      }
    }
    seatEl.appendChild(handEl);

    // Bullhead pile row — visible to everyone.
    const pileWrap = document.createElement('div');
    pileWrap.className = 'sn-pile';
    pileWrap.setAttribute('data-pile', String(pi));
    const pileLabel = document.createElement('div');
    pileLabel.className = 'sn-pile-label';
    pileLabel.textContent = t('sn.bullhead_pile_label', { n: headsCount });
    pileWrap.appendChild(pileLabel);
    const pileCards = document.createElement('div');
    pileCards.className = 'sn-pile-cards';
    const eaten = bullheadCards(pd);
    for (const c of eaten) {
      const mini = makeCardEl(c, { mini: true });
      mini.setAttribute('data-pile-card', String(c));
      pileCards.appendChild(mini);
    }
    if (!eaten.length) {
      const empty = document.createElement('div');
      empty.className = 'sn-pile-empty';
      empty.textContent = '—';
      pileCards.appendChild(empty);
    }
    pileWrap.appendChild(pileCards);
    seatEl.appendChild(pileWrap);

    // SMALL phase prompt.
    if (inSmall && st.small_player === pi) {
      const promptEl = document.createElement('div');
      promptEl.className = 'sn-small-prompt';
      promptEl.textContent = isHuman
        ? t('sn.small_prompt')
        : t('sn.small_other_prompt', { n: pi });
      seatEl.appendChild(promptEl);
    }

    wrap.appendChild(seatEl);
  }

  container.appendChild(wrap);
}

// ====== Animation logic ======
//
// Two animation modes:
//
// 1) Mid-round commit (not the last seat): card flies from hand to its
//    own commit slot, lands face-down (other seats can see it's there).
//    DOM state after step: hand minus card; commit slot face-down.
//
// 2) Final commit (last seat to commit) — server has already applied
//    apply_queue_card and run_apply_loop. We need to:
//    a) Animate this seat's card flying to its commit slot face-down
//       (matches the others which already are face-down).
//    b) Reveal: flip all N face-down to face-up by replacing slot DOM.
//    c) For each (card, player) in ascending order:
//       - either fly card to row tail (no eat), or
//       - eat row: fly each row card to player's bullhead pile, then fly
//         placed card to row[0].
//    Each substep mutates the live DOM (run step) so the next fly's
//    "from" position reflects the current intermediate state.
//
//    SMALL phase pause: if mid-apply the next card has no valid row,
//    server stops at small_phase_active=true. We just animate up to the
//    last applied entry (apply_cursor) and the human/AI then picks a row
//    in a separate transition (handled by 'small_choose_row' below).

// Snapshot of "what the table looks like after applying entries [0, k)"
// using the prev table + the apply queue. Used to drive intermediate
// animations and DOM mutations.
function simulateApply(prevTable, queue, k) {
  // Deep copy table.
  const tbl = prevTable.map(row => row.slice());
  const eats = [];  // {idx, row, eatenCards, player, placed}
  for (let i = 0; i < k; i++) {
    const { card, player } = queue[i];
    if (!card) continue;
    let target = -1, bestTail = -1;
    for (let r = 0; r < NUM_ROWS; r++) {
      let tail = 0;
      for (let s = 0; s < ROW_CAPACITY; s++) {
        if (tbl[r][s] > 0) tail = tbl[r][s];
      }
      if (tail > 0 && tail < card && tail > bestTail) {
        bestTail = tail;
        target = r;
      }
    }
    if (target < 0) {
      // SMALL pause point — stop simulation.
      eats.push({ idx: i, row: -1, eatenCards: [], player, placed: card, smallStop: true });
      return { table: tbl, eats };
    }
    let len = 0;
    for (let s = 0; s < ROW_CAPACITY; s++) if (tbl[target][s] > 0) len = s + 1;
    if (len >= ROW_CAPACITY) {
      const eaten = [];
      for (let s = 0; s < ROW_CAPACITY; s++) {
        if (tbl[target][s] > 0) eaten.push(tbl[target][s]);
        tbl[target][s] = 0;
      }
      tbl[target][0] = card;
      eats.push({ idx: i, row: target, eatenCards: eaten, player, placed: card });
    } else {
      tbl[target][len] = card;
      eats.push({ idx: i, row: target, eatenCards: [], player, placed: card });
    }
  }
  return { table: tbl, eats };
}

// Mutate a DOM table element to reflect a new table snapshot (cell value array).
function paintTableSnapshot(boardRoot, tbl) {
  for (let r = 0; r < NUM_ROWS; r++) {
    for (let s = 0; s < ROW_CAPACITY; s++) {
      const slot = boardRoot.querySelector('[data-row-slot="' + r + '-' + s + '"]');
      if (!slot) continue;
      slot.innerHTML = '';
      slot.classList.remove('empty');
      const c = tbl[r][s];
      if (c > 0) {
        slot.appendChild(makeCardEl(c, {}));
      } else {
        slot.classList.add('empty');
        slot.appendChild(makeCardEl(null, {}));
      }
    }
    // Refresh row heads count.
    const rowEl = document.querySelector('[data-row="' + r + '"]');
    if (rowEl) {
      let rowHeads = 0;
      let rowLen = 0;
      for (let s = 0; s < ROW_CAPACITY; s++) {
        const c = tbl[r][s];
        if (c > 0) { rowHeads += headValue(c); rowLen++; }
      }
      const headsEl = rowEl.querySelector('.sn-row-heads');
      if (headsEl) headsEl.textContent = t('sn.bullhead_short', { n: rowHeads });
      rowEl.classList.toggle('sn-row-full', rowLen === ROW_CAPACITY);
    }
  }
}

function flipCommitSlotToCard(seat, card) {
  const slot = document.querySelector('[data-commit-slot="' + seat + '"]');
  if (!slot) return;
  // Preserve the seat-tag if present.
  const tag = slot.querySelector('.sn-commit-tag-mini');
  slot.innerHTML = '';
  slot.classList.remove('empty', 'face-down', 'applied');
  if (tag) slot.appendChild(tag);
  if (card != null && card > 0) {
    slot.appendChild(makeCardEl(card, {}));
  }
}

function clearCommitSlot(seat, markApplied) {
  const slot = document.querySelector('[data-commit-slot="' + seat + '"]');
  if (!slot) return;
  const tag = slot.querySelector('.sn-commit-tag-mini');
  slot.innerHTML = '';
  slot.classList.remove('face-down');
  if (markApplied) slot.classList.add('applied');
  else slot.classList.add('empty');
  if (tag) slot.appendChild(tag);
  slot.appendChild(makeCardEl(null, {}));
}

// Client-side round log: server clears apply_queue_card[i] after applying
// each entry, so prev/new state alone can't reconstruct the apply order
// during a multi-card resolution. We accumulate (seat -> card) as commits
// fly in, and consume the snapshot on the final commit.
const _roundCommitLog = {
  round: -1,
  commits: {},  // seat -> card
};

function _logCommit(round, seat, card) {
  if (_roundCommitLog.round !== round) {
    _roundCommitLog.round = round;
    _roundCommitLog.commits = {};
  }
  _roundCommitLog.commits[seat] = card;
}

function _consumeRoundCommits(round) {
  if (_roundCommitLog.round !== round) return null;
  const entries = Object.entries(_roundCommitLog.commits)
    .map(([seat, card]) => ({ seat: Number(seat), card }));
  // Reset for the next round.
  _roundCommitLog.round = -1;
  _roundCommitLog.commits = {};
  return entries;
}

// Cards of this round that the new frame shows face up: seat -> card.
// Sources are public by construction — `card_played` events of the
// transition (cards applied to the table) and pending cards the serializer
// exposes once the round is sealed (SMALL pause). An opponent's earlier
// commit reaches the browser only as a hidden stand-in, so this is where
// its card is learned.
function revealedCommits(newFrame) {
  const out = {};
  const events = (newFrame && newFrame.events) || [];
  for (const ev of events) {
    if (ev && ev.kind === 'card_played' && ev.payload) {
      out[ev.payload.player] = ev.payload.card;
    }
  }
  const players = (newFrame && newFrame.state && newFrame.state.players) || [];
  players.forEach((pd, seat) => {
    if (pd && pd.pending_card > 0) out[seat] = pd.pending_card;
  });
  return out;
}

function describeTransition(prevFrame, newFrame, actionInfo, actionId) {
  if (!prevFrame || !prevFrame.state || !newFrame || !newFrame.state) return null;
  const info = actionInfo || classifyAction(actionId);
  if (!info) return null;
  const type = String(info.type || '');
  const actor = prevFrame.current_player;
  const prevSt = prevFrame.state;
  const newSt = newFrame.state;
  const numPlayers = prevFrame.num_players || 2;
  const steps = [];

  if (type === 'small_choose_row') {
    // Drive the small eat AND every remaining apply-queue entry. Server
    // pauses on the small entry (apply_queue_card[cursor] still set), then
    // resumes draining queue[cursor+1..N-1] after the row is chosen — the
    // remaining commits silently disappear without the per-card flights
    // unless we animate them here.
    const cursor = prevSt.apply_cursor || 0;
    const ac = prevSt.apply_queue_card || [];
    const ap = prevSt.apply_queue_player || [];
    const N = ac.length;
    const smallRow = info.row;
    const smallPlayer = prevSt.small_player;
    const smallCard = ac[cursor] || 0;

    // Build the queue covering [cursor..N-1] for simulateApply purposes.
    const tailQueue = [];
    for (let i = cursor + 1; i < N; i++) {
      const c = ac[i];
      if (c > 0) tailQueue.push({ idx: i, card: c, player: ap[i] });
    }

    // --- Step 1: small-entry eat ---
    const smallEaten = [];
    for (let s = 0; s < ROW_CAPACITY; s++) {
      const c = (prevSt.table || [])[smallRow] && prevSt.table[smallRow][s];
      if (c > 0) smallEaten.push(c);
    }
    steps.push({
      type: 'highlight',
      target: '[data-row="' + smallRow + '"]',
      className: 'sn-row-eat-flash',
      duration: 600,
    });
    if (smallEaten.length) {
      const flights = smallEaten.map((c, k) => ({
        from: '[data-row-slot="' + smallRow + '-' + k + '"]',
        to: '[data-pile="' + smallPlayer + '"]',
        createElement: () => makeFlyingCard(c),
        duration: 450,
        width: 60, height: 84,
        hideFrom: true,
      }));
      steps.push({ type: 'flyGroup', flights });
      // Materialize pile + clear row + restore visibility, same trick as
      // the eat branch in the final-commit path.
      const eatenLocal = smallEaten.slice();
      steps.push({
        type: 'run',
        fn: () => {
          const pileEl = document.querySelector('[data-pile="' + smallPlayer + '"] .sn-pile-cards');
          if (pileEl) {
            const ph = pileEl.querySelector('.sn-pile-empty');
            if (ph) ph.remove();
            for (const c of eatenLocal) {
              const mini = makeCardEl(c, { mini: true });
              mini.setAttribute('data-pile-card', String(c));
              pileEl.appendChild(mini);
            }
          }
          const labelEl = document.querySelector('[data-pile="' + smallPlayer + '"] .sn-pile-label');
          if (labelEl) {
            const cur = parseInt((labelEl.textContent.match(/\d+/) || ['0'])[0], 10) || 0;
            const inc = eatenLocal.reduce((a, c) => a + headValue(c), 0);
            labelEl.textContent = t('sn.bullhead_pile_label', { n: cur + inc });
          }
          for (let s = 0; s < ROW_CAPACITY; s++) {
            const slot = document.querySelector('[data-row-slot="' + smallRow + '-' + s + '"]');
            if (!slot) continue;
            slot.innerHTML = '';
            slot.classList.add('empty');
            slot.style.visibility = '';
            slot.appendChild(makeCardEl(null, {}));
          }
          const rowEl = document.querySelector('[data-row="' + smallRow + '"]');
          if (rowEl) {
            const headsEl = rowEl.querySelector('.sn-row-heads');
            if (headsEl) headsEl.textContent = t('sn.bullhead_short', { n: 0 });
            rowEl.classList.remove('sn-row-full');
          }
        },
      });
      steps.push({ type: 'pause', duration: 250 });
    }
    // Placed card flies from commit slot into row[0].
    steps.push({
      type: 'fly',
      from: '[data-commit-slot="' + smallPlayer + '"]',
      to: '[data-row-slot="' + smallRow + '-0"]',
      createElement: () => makeFlyingCard(smallCard),
      duration: 500,
      width: 60, height: 84,
      hideFrom: true,
    });

    // Build the post-small table snapshot — same content as
    // simulateApply on a "queue with one fake entry forcing eat at smallRow".
    const tableAfterSmall = (prevSt.table || []).map(row => row.slice());
    for (let s = 0; s < ROW_CAPACITY; s++) tableAfterSmall[smallRow][s] = 0;
    tableAfterSmall[smallRow][0] = smallCard;
    steps.push({
      type: 'run',
      fn: () => {
        paintTableSnapshot(document.querySelector('.sn-board-root'), tableAfterSmall);
        clearCommitSlot(smallPlayer, true);
      },
    });
    steps.push({ type: 'pause', duration: 500 });

    // --- Step 2: continue with the remaining queue entries [cursor+1..N) ---
    if (tailQueue.length) {
      const tailSim = simulateApply(tableAfterSmall, tailQueue, tailQueue.length);
      for (let k = 0; k < tailSim.eats.length; k++) {
        const e = tailSim.eats[k];
        if (e.smallStop) break;  // another nested small pause — server will re-pause

        if (e.eatenCards.length) {
          steps.push({
            type: 'highlight',
            target: '[data-row="' + e.row + '"]',
            className: 'sn-row-eat-flash',
            duration: 700,
          });
          const flights = e.eatenCards.map((c, slotIdx) => ({
            from: '[data-row-slot="' + e.row + '-' + slotIdx + '"]',
            to: '[data-pile="' + e.player + '"]',
            createElement: () => makeFlyingCard(c),
            duration: 650,
            width: 60, height: 84,
            hideFrom: true,
          }));
          steps.push({ type: 'flyGroup', flights });
          const eatenCardsLocal = e.eatenCards.slice();
          const playerLocal = e.player;
          const rowLocal = e.row;
          steps.push({
            type: 'run',
            fn: () => {
              const pileEl = document.querySelector('[data-pile="' + playerLocal + '"] .sn-pile-cards');
              if (pileEl) {
                const ph = pileEl.querySelector('.sn-pile-empty');
                if (ph) ph.remove();
                for (const c of eatenCardsLocal) {
                  const mini = makeCardEl(c, { mini: true });
                  mini.setAttribute('data-pile-card', String(c));
                  pileEl.appendChild(mini);
                }
              }
              const labelEl = document.querySelector('[data-pile="' + playerLocal + '"] .sn-pile-label');
              if (labelEl) {
                const cur = parseInt((labelEl.textContent.match(/\d+/) || ['0'])[0], 10) || 0;
                const inc = eatenCardsLocal.reduce((a, c) => a + headValue(c), 0);
                labelEl.textContent = t('sn.bullhead_pile_label', { n: cur + inc });
              }
              for (let s = 0; s < ROW_CAPACITY; s++) {
                const slot = document.querySelector('[data-row-slot="' + rowLocal + '-' + s + '"]');
                if (!slot) continue;
                slot.innerHTML = '';
                slot.classList.add('empty');
                slot.style.visibility = '';
                slot.appendChild(makeCardEl(null, {}));
              }
              const rowEl = document.querySelector('[data-row="' + rowLocal + '"]');
              if (rowEl) {
                const headsEl = rowEl.querySelector('.sn-row-heads');
                if (headsEl) headsEl.textContent = t('sn.bullhead_short', { n: 0 });
                rowEl.classList.remove('sn-row-full');
              }
            },
          });
          steps.push({ type: 'pause', duration: 250 });
          steps.push({
            type: 'fly',
            from: '[data-commit-slot="' + e.player + '"]',
            to: '[data-row-slot="' + e.row + '-0"]',
            createElement: () => makeFlyingCard(e.placed),
            duration: 700,
            width: 60, height: 84,
            hideFrom: true,
          });
        } else {
          const before = simulateApply(tableAfterSmall, tailQueue, k).table;
          let len = 0;
          for (let s = 0; s < ROW_CAPACITY; s++) if (before[e.row][s] > 0) len = s + 1;
          steps.push({
            type: 'fly',
            from: '[data-commit-slot="' + e.player + '"]',
            to: '[data-row-slot="' + e.row + '-' + len + '"]',
            createElement: () => makeFlyingCard(e.placed),
            duration: 700,
            width: 60, height: 84,
            hideFrom: true,
          });
        }
        const snapshotAfter = simulateApply(tableAfterSmall, tailQueue, k + 1).table;
        steps.push({
          type: 'run',
          fn: () => {
            paintTableSnapshot(document.querySelector('.sn-board-root'), snapshotAfter);
            clearCommitSlot(e.player, true);
          },
        });
        steps.push({ type: 'pause', duration: 500 });
      }
    }
    return steps;
  }

  if (type !== 'commit') return null;

  // A face-down opponent commit carries no card (hidden stand-in); the
  // commit that completes the round does, and so does the human's own.
  const revealed = revealedCommits(newFrame);
  const card = info.card != null ? info.card
    : (revealed[actor] != null ? revealed[actor] : null);
  const round = (prevSt.current_round != null ? prevSt.current_round : 0);
  if (card != null) _logCommit(round, actor, card);

  // `committed` is per-player on st.players[p], not a top-level array.
  const prevPlayers = prevSt.players || [];
  const seatCommitsBefore = prevPlayers.reduce((a, pd) => a + (pd && pd.committed ? 1 : 0), 0);
  const isFinalCommit = (seatCommitsBefore + 1) >= numPlayers;

  if (!isFinalCommit) {
    // Fly a face-down card from actor's hand to their commit slot.
    // - Self: source is the specific card in own hand (face-up flying).
    // - Opponent: source is anywhere in their hand row (face-down flying);
    //   we can't address a specific card because their hand cards are
    //   rendered as face-down placeholders without data-card.
    const isSelf = actor === prevFrame.human_player;
    steps.push({
      type: 'fly',
      from: isSelf
        ? '[data-hand="' + actor + '"] [data-card="' + card + '"]'
        : '[data-hand="' + actor + '"]',
      to: '[data-commit-slot="' + actor + '"]',
      createElement: () => isSelf ? makeFlyingCard(card) : makeFlyingCard(null),
      duration: 500,
      width: 60, height: 84,
      // Self: hide the specific source card so it appears to lift off.
      // Opponent: source is the whole hand row (no per-card data attr on
      // face-down minis); hiding it would blank every opponent card during
      // flight. The next render naturally drops one card via hand_count--.
      hideFrom: isSelf,
      onComplete: () => {
        // Mark slot face-down so the next render-equivalent mid-state is correct.
        const slot = document.querySelector('[data-commit-slot="' + actor + '"]');
        if (!slot) return;
        const tag = slot.querySelector('.sn-commit-tag-mini');
        slot.innerHTML = '';
        slot.classList.remove('empty', 'applied');
        slot.classList.add('face-down');
        if (tag) slot.appendChild(tag);
        const fd = document.createElement('div');
        fd.className = 'sn-card sn-card-back';
        slot.appendChild(fd);
      },
    });
    return steps;
  }

  // Final commit — drive the whole reveal-and-apply animation.
  // Build apply queue from newSt (server already applied). Note newSt's
  // apply_queue_card slots are 0 for already-applied entries; we
  // reconstruct the original order by combining apply_queue_player +
  // committed cards from prevSt.pending_commits + actor's just-played card.
  // Simpler: pending_commits is owner_only — only own card visible. Use
  // the new apply_queue_player slots and walk newSt+prev.bullhead/table
  // to recover cards. But this is fragile. Easier: rebuild from sources
  // we know:
  //   - Actor's card: `card` (just played)
  //   - Other players' cards: from newSt.apply_queue_card (for slots that
  //     haven't been applied yet) + bullhead diff vs prev for applied
  //     ones. But we don't know which seat ate what without rerunning
  //     the apply logic.
  //
  // Trick: the C++ side always sets apply_queue_card slot = 0 *after*
  // applying that index. So if newSt is mid-loop (small_phase pause),
  // apply_queue_card[apply_cursor..N-1] still hold the unprocessed cards
  // and apply_queue_player matches them. For applied entries (idx < cursor)
  // we have a problem — the card got cleared. But the only way the loop
  // would have already processed entries before the human sees it is the
  // entire sequence ran; in that case apply_cursor == N and all slots
  // are 0.
  //
  // Strategy:
  //   Case A: SMALL pause — newSt has small_phase_active=true.
  //     apply_queue_card[cursor..N-1] are the *still pending* cards in
  //     ascending-card order. We must still know what cards 0..cursor-1
  //     were so we can animate them. Rebuild from prev+apply diff:
  //     compute set S = sorted({prev.pending_commits[me] | me visible} ∪
  //     {card} ∪ infer from new bullhead diff). Too fragile.
  //
  //   Simpler & robust: at the moment of final commit, we know:
  //     - newSt.apply_queue_player[i] is set for all i in [0, N) (server
  //       built the full queue before running the loop).
  //     - newSt.apply_queue_card[i] is non-zero only for unapplied
  //       entries.
  //     For applied entries, we infer the played card from its effect:
  //       * If that seat's bullhead_pile got new cards vs prev, the
  //         placed card is the one currently at that row[0] in newSt
  //         (the "eat" outcome) OR sitting at row[len-1] (the "append"
  //         outcome). We can locate it by diff.
  //     This is still fragile.
  //
  //   Cleanest path: ask backend for the played-cards-this-round somehow.
  //   But we don't have that API yet, so we use a server-supplied hint
  //   in newFrame.action_info.commits if present (we don't add it now).
  //
  //   For v1 we use a heuristic: animate only the actor's own play into
  //   commit slot, then fade-flip all face-downs, then immediately render
  //   the new state without per-card flight. This is still better than
  //   the old behavior because reveal is at least visible.

  // Step 1: actor's card flies face-up to its commit slot. The other
  // seats are already showing face-down placeholders; landing this one
  // face-up and revealing the rest in the same beat (below) skips the
  // information-free "all face-down hold" — the human sees their card
  // arrive, then everyone flips together.
  const isSelfFinal = actor === prevFrame.human_player;
  steps.push({
    type: 'fly',
    from: isSelfFinal
      ? '[data-hand="' + actor + '"] [data-card="' + card + '"]'
      : '[data-hand="' + actor + '"]',
    to: '[data-commit-slot="' + actor + '"]',
    createElement: () => makeFlyingCard(card),
    duration: 500,
    width: 60, height: 84,
    hideFrom: isSelfFinal,
    onComplete: () => {
      flipCommitSlotToCard(actor, card);
    },
  });

  // Reconstruct the queue using the client-side round log (we recorded
  // every commit as it flew in, so we have all N seats' cards even after
  // the server cleared apply_queue_card[i] mid-loop).
  // Seats whose commit arrived hidden are filled in from what this
  // transition turned face up.
  const bySeat = {};
  for (const e of (_consumeRoundCommits(round) || [])) bySeat[e.seat] = e.card;
  for (const seat of Object.keys(revealed)) bySeat[seat] = revealed[seat];
  const logged = Object.entries(bySeat).map(([seat, c]) => ({ seat: Number(seat), card: c }));
  const queue = [];
  if (logged && logged.length === numPlayers) {
    // Sort ascending by card — same rule the server uses in
    // seal_pending_into_queue.
    const sorted = logged.slice().sort((a, b) => a.card - b.card);
    for (let i = 0; i < sorted.length; i++) {
      queue.push({ idx: i, card: sorted[i].card, player: sorted[i].seat });
    }
  } else {
    // Fallback: state-only reconstruction. apply_queue_card is 0 for
    // already-applied entries, so unapplied cards are recoverable but
    // applied ones aren't.
    const ac = (newSt.apply_queue_card || []);
    const ap = (newSt.apply_queue_player || []);
    for (let i = 0; i < numPlayers; i++) {
      const p = ap[i];
      if (p == null) continue;
      let c = ac[i];
      if (c <= 0) c = null;
      queue.push({ idx: i, card: c, player: p });
    }
  }

  const allKnown = queue.length === numPlayers && queue.every(q => q.card != null);

  steps.push({
    type: 'run',
    fn: () => {
      // Reveal: replace all face-down slots with face-up cards (using
      // queue's known cards). For unknown ones (already applied), leave
      // them in current state (they'll be cleared/applied later).
      for (const q of queue) {
        if (q.card != null) {
          flipCommitSlotToCard(q.player, q.card);
        }
      }
    },
  });
  // Long beat after reveal — let the human read every committed card and
  // mentally sort them before any apply animation starts.
  steps.push({ type: 'pause', duration: 1540 });

  if (!allKnown) {
    // Fallback: just paint the final table, no per-card animation.
    steps.push({
      type: 'run',
      fn: () => {
        paintTableSnapshot(document.querySelector('.sn-board-root'), newSt.table || []);
        for (const q of queue) clearCommitSlot(q.player, true);
      },
    });
    return steps;
  }

  // Step 3: per-card apply animation.
  const sim = simulateApply(prevSt.table || [], queue, queue.length);
  for (let k = 0; k < sim.eats.length; k++) {
    const e = sim.eats[k];
    if (e.smallStop) break;  // SMALL pause stops here

    if (e.eatenCards.length) {
      // Eat: highlight target row, fly row cards to pile, then placed card.
      steps.push({
        type: 'highlight',
        target: '[data-row="' + e.row + '"]',
        className: 'sn-row-eat-flash',
        duration: 700,
      });
      const flights = e.eatenCards.map((c, slotIdx) => ({
        from: '[data-row-slot="' + e.row + '-' + slotIdx + '"]',
        to: '[data-pile="' + e.player + '"]',
        createElement: () => makeFlyingCard(c),
        duration: 650,
        width: 60, height: 84,
        hideFrom: true,
      }));
      steps.push({ type: 'flyGroup', flights });
      // Materialize eaten cards into the pile and clear the row NOW —
      // before the placed-card flight. Otherwise the flyers vanish at
      // landing and the eaten cards "disappear" for ~1s while the placed
      // card is still in flight.
      const eatenCardsLocal = e.eatenCards.slice();
      const playerLocal = e.player;
      const rowLocal = e.row;
      steps.push({
        type: 'run',
        fn: () => {
          const pileEl = document.querySelector('[data-pile="' + playerLocal + '"] .sn-pile-cards');
          if (pileEl) {
            const ph = pileEl.querySelector('.sn-pile-empty');
            if (ph) ph.remove();
            for (const c of eatenCardsLocal) {
              const mini = makeCardEl(c, { mini: true });
              mini.setAttribute('data-pile-card', String(c));
              pileEl.appendChild(mini);
            }
          }
          const labelEl = document.querySelector('[data-pile="' + playerLocal + '"] .sn-pile-label');
          if (labelEl) {
            const cur = parseInt((labelEl.textContent.match(/\d+/) || ['0'])[0], 10) || 0;
            const inc = eatenCardsLocal.reduce((a, c) => a + headValue(c), 0);
            labelEl.textContent = t('sn.bullhead_pile_label', { n: cur + inc });
          }
          // Clear the eaten row so the placed card has empty slots to land into.
          // Also restore visibility — flyGroup's hideFrom set the slot
          // elements to visibility:hidden, and the framework only restores
          // them at the end of the entire transition. Without this the
          // placed card flies into an invisible slot.
          for (let s = 0; s < ROW_CAPACITY; s++) {
            const slot = document.querySelector('[data-row-slot="' + rowLocal + '-' + s + '"]');
            if (!slot) continue;
            slot.innerHTML = '';
            slot.classList.add('empty');
            slot.style.visibility = '';
            slot.appendChild(makeCardEl(null, {}));
          }
          const rowEl = document.querySelector('[data-row="' + rowLocal + '"]');
          if (rowEl) {
            const headsEl = rowEl.querySelector('.sn-row-heads');
            if (headsEl) headsEl.textContent = t('sn.bullhead_short', { n: 0 });
            rowEl.classList.remove('sn-row-full');
          }
        },
      });
      steps.push({ type: 'pause', duration: 250 });
      steps.push({
        type: 'fly',
        from: '[data-commit-slot="' + e.player + '"]',
        to: '[data-row-slot="' + e.row + '-0"]',
        createElement: () => makeFlyingCard(e.placed),
        duration: 700,
        width: 60, height: 84,
        hideFrom: true,
      });
    } else {
      // Append: card flies from commit slot to row tail.
      const targetSlot = (() => {
        // After [0..k) applied, what's the next free slot in target row?
        let len = 0;
        // Use current table (pre this k application).
        const before = simulateApply(prevSt.table || [], queue, k).table;
        for (let s = 0; s < ROW_CAPACITY; s++) if (before[e.row][s] > 0) len = s + 1;
        return len;
      })();
      steps.push({
        type: 'fly',
        from: '[data-commit-slot="' + e.player + '"]',
        to: '[data-row-slot="' + e.row + '-' + targetSlot + '"]',
        createElement: () => makeFlyingCard(e.placed),
        duration: 700,
        width: 60, height: 84,
        hideFrom: true,
      });
    }
    // After the placed-card flight lands, paint the table to its post-k
    // snapshot and clear the commit slot. Eat-side pile/row mutations
    // already happened above (between flyGroup and placed flight) so we
    // don't redo them here.
    const snapshotAfter = simulateApply(prevSt.table || [], queue, k + 1).table;
    steps.push({
      type: 'run',
      fn: () => {
        paintTableSnapshot(document.querySelector('.sn-board-root'), snapshotAfter);
        clearCommitSlot(e.player, true);
      },
    });
    // Pause between cards so the human can see each step's outcome.
    steps.push({ type: 'pause', duration: 500 });
  }

  return steps;
}

const roundExtension = {
  render(el, gameState) {
    if (!gameState || !gameState.state) {
      el.textContent = t('sn.round_default');
      return;
    }
    const st = gameState.state;
    const round = (st.current_round != null ? st.current_round : 0) + 1;
    const committed = st.num_committed_this_round || 0;
    const total = gameState.num_players || 2;
    el.textContent = t('sn.round_label', { n: Math.min(round, 10) })
      + ' · ' + t('sn.committed_label', { n: committed, total });
  },
};

createApp({
  gameId: '6nimmit',
  gameTitle: t('sn.title'),
  gameIntro: t('sn.intro'),
  players: { min: 2, max: 4 },
  // Simultaneous-action game — seat order has no meaning. Hide the seat
  // picker; the framework defaults the human to seat 0.
  hideSeatPicker: true,
  // The opponents' hands are hidden: there is nothing to pick for them.
  disableForce: true,
  renderBoard,
  renderPlayerArea,
  describeTransition,
  formatOpponentMove: formatMove,
  formatSuggestedMove: formatMove,
  getPlayerSymbol: (humanPlayer) => t('sn.player_n', { n: humanPlayer }),
  extensions: [roundExtension],
  difficulties: ['heuristic', 'casual', 'expert'],
  defaultDifficulty: 'expert',
  showWinrateDefault: false,
});
