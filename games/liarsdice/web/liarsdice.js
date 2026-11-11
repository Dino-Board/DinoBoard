import { createApp } from '/static/general/app.js';
import { t, register } from '/static/general/i18n.js';

register({
  zh: {
    'ld.title': '大话骰',
    'ld.intro': '吹牛 / 开盅。点 +/− 调整出价，"出价"提交，"开！"挑战。',
    'ld.player_n': '玩家{n}',
    'ld.player_self_suffix': '（你）',
    'ld.current_bid_label': '当前出价',
    'ld.current_bid_empty': '等待第一次出价',
    'ld.bid_join': '个',
    'ld.draft_legal': '合法出价',
    'ld.draft_illegal': '此出价不合法',
    'ld.btn_bid': '出价',
    'ld.btn_challenge': '开！',
    'ld.opp_cup': '对手骰盅',
    'ld.self_cup': '你的骰',
    'ld.spectator': '等待 AI 行动…',
    'ld.terminal_status': '对局结束',
    'ld.action_unknown': '动作 #{id}',
    'ld.action_start': '开局',
    'ld.action_bid': '叫 {count} 个 {face}',
    'ld.action_challenge': '开！',
    'ld.loser_you': '你输了！',
    'ld.loser_you_text': '',
    'ld.loser_other': '{label} 输了',
    'ld.loser_other_text': '',
  },
  en: {
    'ld.title': "Liar's Dice",
    'ld.intro': 'Bid count×face or call "challenge". Use ± to set draft, then submit.',
    'ld.player_n': 'Player {n}',
    'ld.player_self_suffix': ' (you)',
    'ld.current_bid_label': 'Current bid',
    'ld.current_bid_empty': 'No bid yet — make the first call',
    'ld.bid_join': '×',
    'ld.draft_legal': 'Bid is legal',
    'ld.draft_illegal': 'Bid is not legal',
    'ld.btn_bid': 'Bid',
    'ld.btn_challenge': 'Challenge!',
    'ld.opp_cup': "Opponent's cup",
    'ld.self_cup': 'Your dice',
    'ld.spectator': 'Waiting for AI…',
    'ld.terminal_status': 'Game over',
    'ld.action_unknown': 'Action #{id}',
    'ld.action_start': 'Game start',
    'ld.action_bid': 'bid {count} × {face}',
    'ld.action_challenge': 'Challenge!',
    'ld.loser_you': 'You lost!',
    'ld.loser_you_text': '',
    'ld.loser_other': '{label} lost',
    'ld.loser_other_text': '',
  },
});

const DICE_PER_PLAYER = 5;
const FACE_COUNT = 6;
const MIN_BID_FACE = 2;

// Action layout: [0, 5N*5) → bid; 5N*5 → challenge.
// X = action_id // 5 + 1 (1..5N);  Y = action_id % 5 + 2 (2..6).
function totalDice(numPlayers) { return DICE_PER_PLAYER * numPlayers; }
function challengeAction(numPlayers) { return totalDice(numPlayers) * 5; }
function encodeBid(x, y) { return (x - 1) * 5 + (y - 2); }
function decodeBid(actionId) {
  return { count: Math.floor(actionId / 5) + 1, face: (actionId % 5) + 2 };
}
function classifyAction(actionId, numPlayers) {
  if (actionId == null) return null;
  const ch = challengeAction(numPlayers);
  if (actionId === ch) return { type: 'challenge' };
  if (actionId >= 0 && actionId < ch) {
    return Object.assign({ type: 'bid' }, decodeBid(actionId));
  }
  return null;
}

function bidStrictlyGreater(xn, yn, xo, yo) {
  if (xn > xo) return true;
  if (xn === xo && yn > yo) return true;
  return false;
}

function formatMove(info, actionId, _actor) {
  if (!info) {
    if (actionId == null) return t('ld.action_start');
    return t('ld.action_unknown', { id: actionId });
  }
  if (info.type === 'challenge') return t('ld.action_challenge');
  if (info.type === 'bid') return t('ld.action_bid', { count: info.count, face: info.face });
  if (actionId != null) return t('ld.action_unknown', { id: actionId });
  return t('ld.action_start');
}

// Per-session draft state. Lives in module scope because createApp doesn't
// give the renderBoard callback a place to keep mutable game-side state.
let draft = { count: 1, face: MIN_BID_FACE };

function clampDraft(numPlayers, prevBid) {
  const td = totalDice(numPlayers);
  if (draft.count < 1) draft.count = 1;
  if (draft.count > td) draft.count = td;
  if (draft.face < MIN_BID_FACE) draft.face = MIN_BID_FACE;
  if (draft.face > FACE_COUNT) draft.face = FACE_COUNT;
  // If a current bid exists and the draft is not strictly greater, advance
  // it to the minimum strict raise (same count + face+1, or count+1, face=2).
  if (prevBid && prevBid.count > 0) {
    if (!bidStrictlyGreater(draft.count, draft.face, prevBid.count, prevBid.face)) {
      if (prevBid.face < FACE_COUNT) {
        draft.count = prevBid.count;
        draft.face = prevBid.face + 1;
      } else {
        draft.count = prevBid.count + 1;
        draft.face = MIN_BID_FACE;
      }
      if (draft.count > td) draft.count = td;
    }
  }
}

function makeDieEl(face, opts) {
  opts = opts || {};
  const el = document.createElement('div');
  el.className = 'ld-die';
  if (opts.back) {
    el.classList.add('ld-die-back');
    el.textContent = '?';
    return el;
  }
  if (face === 1) el.classList.add('ld-die-wild');
  if (opts.match) el.classList.add('ld-die-match');
  if (opts.revealed) el.classList.add('ld-die-revealed');
  el.textContent = face != null ? String(face) : '?';
  return el;
}

function diceListFromCount(dc) {
  const out = [];
  if (!Array.isArray(dc)) return out;
  for (let f = 0; f < dc.length && f < FACE_COUNT; f++) {
    const n = Math.max(0, Number(dc[f]) || 0);
    for (let k = 0; k < n; k++) out.push(f + 1);
  }
  return out.sort((a, b) => a - b);
}

function renderCurrentBid(parent, st) {
  const wrap = document.createElement('div');
  wrap.className = 'ld-current-bid';
  const label = document.createElement('div');
  label.className = 'ld-current-bid-label';
  label.textContent = t('ld.current_bid_label');
  wrap.appendChild(label);

  const value = document.createElement('div');
  value.className = 'ld-current-bid-value';
  const cnt = (st && st.current_bid_count) || 0;
  const face = (st && st.current_bid_face) || 0;
  if (cnt === 0 || face === 0) {
    wrap.classList.add('ld-current-bid-empty');
    value.textContent = t('ld.current_bid_empty');
  } else {
    const num = document.createElement('span');
    num.textContent = String(cnt);
    value.appendChild(num);
    const join = document.createElement('span');
    join.textContent = t('ld.bid_join');
    value.appendChild(join);
    value.appendChild(makeDieEl(face, { match: true }));
  }
  wrap.appendChild(value);
  parent.appendChild(wrap);
}

function renderOpponents(parent, st, ctx, numPlayers, currentMatchFace) {
  const wrap = document.createElement('div');
  wrap.className = 'ld-opponents';
  // Seating order, top → bottom: 下家, 下下家, ..., 上家. Self is rendered
  // below, so the cup directly above self is the upstream (上家) and the
  // top of the list is the immediate downstream (下家) — the natural
  // around-the-table reading order from the human player's seat.
  const opps = [];
  const human = (ctx.state && ctx.state.humanPlayer != null) ? ctx.state.humanPlayer : 0;
  for (let k = 1; k < numPlayers; k++) {
    opps.push((human + k) % numPlayers);
  }

  const players = (st && st.players) || [];
  const isTerminal = !!(st && st.is_terminal);
  const cur = st ? st.current_player : -1;

  for (const p of opps) {
    const pd = players[p] || {};
    const cup = document.createElement('div');
    cup.className = 'ld-cup';
    cup.setAttribute('data-seat', String(p));
    if (cur === p && !isTerminal) cup.classList.add('ld-cup-current');

    const tag = document.createElement('div');
    tag.className = 'ld-cup-tag';
    if (cur === p && !isTerminal) tag.classList.add('ld-cup-tag-current');
    tag.textContent = t('ld.player_n', { n: p });
    cup.appendChild(tag);

    const row = document.createElement('div');
    row.className = 'ld-dice-row';
    // Opponent dice are face-down ("?") until someone calls challenge —
    // i.e. only revealed on the terminal frame. The GT-side serializer
    // returns truth (`dice_count` is populated even mid-game), so we
    // ignore it and gate reveal solely on isTerminal.
    // In replay mode with the "reveal hidden" toggle on, also show
    // opponent dice on every frame (the replay reconstructs from a
    // single GT session, so dice_count is truth-bearing). But the
    // terminal flip animation only fires on isTerminal — replay-only
    // see-through must NOT trigger the 480ms flip every frame, or the
    // dice spin nonstop as the playback advances.
    const reveal = isTerminal || ctx.revealHidden;
    const flip = isTerminal;
    const dice = reveal && Array.isArray(pd.dice_count)
        ? diceListFromCount(pd.dice_count) : [];
    if (reveal && dice.length > 0) {
      for (const f of dice) {
        row.appendChild(makeDieEl(f, {
          match: currentMatchFace != null && (f === 1 || f === currentMatchFace),
          revealed: flip,
        }));
      }
    } else {
      // Face-down DICE_PER_PLAYER cups.
      for (let k = 0; k < DICE_PER_PLAYER; k++) {
        row.appendChild(makeDieEl(null, { back: true }));
      }
    }
    cup.appendChild(row);
    wrap.appendChild(cup);
  }
  parent.appendChild(wrap);
}

function renderSelf(parent, st, ctx, numPlayers, currentMatchFace) {
  const wrap = document.createElement('div');
  wrap.className = 'ld-self';
  const human = (ctx.state && ctx.state.humanPlayer != null) ? ctx.state.humanPlayer : 0;
  const cur = st ? st.current_player : -1;
  const isTerminal = !!(st && st.is_terminal);
  if (cur === human && !isTerminal) wrap.classList.add('cup-current');

  const tag = document.createElement('div');
  tag.className = 'ld-self-tag';
  tag.textContent = t('ld.player_n', { n: human }) + t('ld.player_self_suffix');
  wrap.appendChild(tag);

  const row = document.createElement('div');
  row.className = 'ld-dice-row';
  const players = (st && st.players) || [];
  const pd = players[human] || {};
  const dice = Array.isArray(pd.dice_count) ? diceListFromCount(pd.dice_count) : [];
  if (dice.length > 0) {
    for (const f of dice) {
      row.appendChild(makeDieEl(f, {
        match: currentMatchFace != null && (f === 1 || f === currentMatchFace),
      }));
    }
  } else {
    for (let k = 0; k < DICE_PER_PLAYER; k++) {
      row.appendChild(makeDieEl(null, { back: true }));
    }
  }
  wrap.appendChild(row);
  parent.appendChild(wrap);
}

function renderControls(parent, st, ctx, numPlayers) {
  const wrap = document.createElement('div');
  wrap.className = 'ld-controls';

  const human = (ctx.state && ctx.state.humanPlayer != null) ? ctx.state.humanPlayer : 0;
  const cur = st ? st.current_player : -1;
  const isTerminal = !!(st && st.is_terminal);
  const myTurn = !isTerminal && cur === human && ctx.canPlay;

  // Per WEB_DESIGN_PRINCIPLES §2 (spatial anchoring) + checklist
  // "动作按钮栏尺寸和槽位固定": don't swap the controls block out for
  // a "waiting for AI" placeholder on opponent turns — the steppers /
  // bid / challenge buttons stay in place; we just disable them so
  // nothing can be submitted out of turn.
  const td = totalDice(numPlayers);
  const prevBid = st && st.current_bid_count > 0
      ? { count: st.current_bid_count, face: st.current_bid_face } : null;
  clampDraft(numPlayers, prevBid);

  const legalSet = new Set(myTurn ? (ctx.legalActions || []) : []);
  const draftAction = encodeBid(draft.count, draft.face);
  const bidLegal = myTurn && legalSet.has(draftAction);
  const challengeLegal = myTurn && legalSet.has(challengeAction(numPlayers));

  const row = document.createElement('div');
  row.className = 'ld-controls-row';

  // X stepper (count)
  const xWrap = document.createElement('div');
  xWrap.className = 'ld-stepper';
  const xVal = document.createElement('div');
  xVal.className = 'ld-stepper-value';
  xVal.textContent = String(draft.count);
  const xBtns = document.createElement('div');
  xBtns.className = 'ld-stepper-buttons';
  const xUp = document.createElement('button');
  xUp.type = 'button'; xUp.textContent = '+';
  xUp.disabled = !myTurn || draft.count >= td;
  xUp.addEventListener('click', () => { draft.count += 1; ctx.rerender(); });
  const xDn = document.createElement('button');
  xDn.type = 'button'; xDn.textContent = '−';
  xDn.disabled = !myTurn || draft.count <= 1;
  xDn.addEventListener('click', () => { draft.count -= 1; ctx.rerender(); });
  xBtns.appendChild(xUp);
  xBtns.appendChild(xVal);
  xBtns.appendChild(xDn);
  xWrap.appendChild(xBtns);
  row.appendChild(xWrap);

  // Times symbol between count and face.
  const times = document.createElement('div');
  times.className = 'ld-draft-times';
  times.textContent = t('ld.bid_join');
  row.appendChild(times);

  // Y stepper (face)
  const yWrap = document.createElement('div');
  yWrap.className = 'ld-stepper';
  const yBtns = document.createElement('div');
  yBtns.className = 'ld-stepper-buttons';
  const yUp = document.createElement('button');
  yUp.type = 'button'; yUp.textContent = '+';
  yUp.disabled = !myTurn || draft.face >= FACE_COUNT;
  yUp.addEventListener('click', () => { draft.face += 1; ctx.rerender(); });
  const yVal = document.createElement('div');
  yVal.className = 'ld-stepper-value';
  yVal.textContent = String(draft.face);
  const yDn = document.createElement('button');
  yDn.type = 'button'; yDn.textContent = '−';
  yDn.disabled = !myTurn || draft.face <= MIN_BID_FACE;
  yDn.addEventListener('click', () => { draft.face -= 1; ctx.rerender(); });
  yBtns.appendChild(yUp);
  yBtns.appendChild(yVal);
  yBtns.appendChild(yDn);
  yWrap.appendChild(yBtns);
  row.appendChild(yWrap);

  wrap.appendChild(row);

  // Status line stays in place every frame — including terminal — so the
  // controls block doesn't shrink when the game ends. Per
  // WEB_DESIGN_PRINCIPLES §2 (spatial anchoring): same panel, same height,
  // contents change.
  const status = document.createElement('div');
  if (isTerminal) {
    status.className = 'ld-draft-status';
    status.textContent = t('ld.terminal_status');
  } else if (!myTurn) {
    status.className = 'ld-draft-status';
    status.textContent = t('ld.spectator');
  } else {
    status.className = 'ld-draft-status ' + (bidLegal ? 'legal' : 'illegal');
    status.textContent = bidLegal ? t('ld.draft_legal') : t('ld.draft_illegal');
  }
  wrap.appendChild(status);

  const btnRow = document.createElement('div');
  btnRow.className = 'ld-buttons';

  const bidBtn = document.createElement('button');
  bidBtn.type = 'button';
  bidBtn.className = 'ld-btn ld-btn-primary';
  bidBtn.textContent = t('ld.btn_bid');
  bidBtn.disabled = !bidLegal;
  bidBtn.addEventListener('click', () => {
    if (bidLegal) ctx.submitAction(draftAction);
  });
  btnRow.appendChild(bidBtn);

  const challengeBtn = document.createElement('button');
  challengeBtn.type = 'button';
  challengeBtn.className = 'ld-btn ld-btn-danger';
  challengeBtn.textContent = t('ld.btn_challenge');
  challengeBtn.disabled = !challengeLegal;
  challengeBtn.addEventListener('click', () => {
    if (challengeLegal) ctx.submitAction(challengeAction(numPlayers));
  });
  btnRow.appendChild(challengeBtn);

  wrap.appendChild(btnRow);
  parent.appendChild(wrap);
}

function renderBoard(container, gameState, ctx) {
  container.innerHTML = '';
  const root = document.createElement('div');
  root.className = 'ld-root';

  const st = gameState && gameState.state ? gameState.state : null;
  const numPlayers = st ? (st.num_players || 2) : (ctx.numPlayers || 2);

  // Re-derive a `legalActions` shortcut on ctx for the controls renderer.
  // The general layer exposes `gameState.legal_actions` when it's the
  // human's turn. ctx.rerender() routes back through the framework's
  // render() which calls this function with the latest gameState.
  const legalActions = (gameState && gameState.legal_actions) || [];
  const ctx2 = Object.assign({}, ctx, {
    legalActions,
    numPlayers,
  });

  const matchFace = st && st.current_bid_face > 0 ? st.current_bid_face : null;

  renderCurrentBid(root, st);
  renderOpponents(root, st, ctx2, numPlayers, matchFace);
  renderSelf(root, st, ctx2, numPlayers, matchFace);
  renderControls(root, st, ctx2, numPlayers);

  container.appendChild(root);
}

function describeTransition(prevState, newState, actionInfo, actionId) {
  // Bids: just a re-render (the current-bid plate updates). Challenge:
  // flip every face-down die — handled by .ld-die-revealed which is added
  // unconditionally on the terminal frame so the keyframe runs once.
  if (!actionInfo) return null;
  if (actionInfo.type === 'challenge') {
    return [{ type: 'highlight', target: '.ld-current-bid', className: 'anim-highlight', duration: 400 }];
  }
  return null;
}

createApp({
  gameId: 'liarsdice',
  gameTitle: t('ld.title'),
  gameIntro: t('ld.intro'),
  players: { min: 2, max: 4 },
  renderBoard,
  describeTransition,
  formatOpponentMove: formatMove,
  formatSuggestedMove: formatMove,
  getPlayerSymbol: (p) => t('ld.player_n', { n: p }),
  aiStepPause: 1500,
  difficulties: ['heuristic', 'casual', 'expert'],
  defaultDifficulty: 'expert',
  defaultSeat: 'random',
  // Hidden-info game (own dice owner-only): showing winrate or smart-hint
  // would leak truth via root values.
  disableForce: true,
  showWinrateDefault: false,
  onGameStart: () => { draft = { count: 1, face: MIN_BID_FACE }; },
});
