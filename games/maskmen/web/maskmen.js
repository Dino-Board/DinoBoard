import { createApp } from '/static/general/app.js';
import { t, tList, register } from '/static/general/i18n.js';

register({
  zh: {
    'mm.title': '面具人',
    'mm.intro': '直接点你想出的角色 — 出多少张由当前线索决定（开局1张；已知更强=同张；未知关系=上张+1）。先出完牌赢。',
    'mm.player_n': '玩家{n}',
    'mm.player_self_suffix': '（你）',
    'mm.type_names': ['红', '橙', '绿', '青', '蓝', '粉'],
    'mm.last_play_label': '当前线索',
    'mm.last_play_empty': '本轮等待坐庄出牌（必须出 1 张）',
    'mm.times': '×',
    'mm.chain_title': '本小局出牌链',
    'mm.strength_title': '强度关系（拓扑序：左弱 → 右强；箭头指向更强的一方）',
    'mm.strength_empty': '尚未建立任何强弱关系',
    'mm.opp_passed_pill': '已过',
    'mm.opp_count': '剩余 {n}',
    'mm.self_tag': '你的手牌（点击直接出牌）',
    'mm.btn_pass': '过',
    'mm.status_pick_action': '点击一个角色出牌，或选择「过」',
    'mm.status_must_open': '你是本轮坐庄 — 必须出 1 张（不能过）',
    'mm.spectator': '等待对手行动…',
    'mm.terminal_status': '对局结束',
    'mm.action_unknown': '动作 #{id}',
    'mm.action_start': '开局',
    'mm.action_pass': '过',
    'mm.action_play': '出 {type}',
  },
  en: {
    'mm.title': 'Maskmen',
    'mm.intro': 'Click the wrestler you want to play — the count is fixed by the current line (opener=1; known stronger=same; unknown=last+1). First to empty hand wins.',
    'mm.player_n': 'Player {n}',
    'mm.player_self_suffix': ' (you)',
    'mm.type_names': ['Red', 'Orange', 'Green', 'Cyan', 'Blue', 'Pink'],
    'mm.last_play_label': 'Current line',
    'mm.last_play_empty': 'Round host must open — exactly 1 card',
    'mm.times': '×',
    'mm.chain_title': 'This round so far',
    'mm.strength_title': 'Strength (topo order: weaker ← → stronger; arrows point to the stronger side)',
    'mm.strength_empty': 'No strength relations known yet',
    'mm.opp_passed_pill': 'PASSED',
    'mm.opp_count': '{n} left',
    'mm.self_tag': 'Your hand (click to play)',
    'mm.btn_pass': 'Pass',
    'mm.status_pick_action': 'Click a wrestler to play, or Pass',
    'mm.status_must_open': "You're the round host — must play 1 card (Pass disabled)",
    'mm.spectator': 'Waiting for opponent…',
    'mm.terminal_status': 'Game over',
    'mm.action_unknown': 'Action #{id}',
    'mm.action_start': 'Game start',
    'mm.action_pass': 'Pass',
    'mm.action_play': 'play {type}',
  },
});

const TYPE_COUNT = 6;
const PASS_ACTION = 6;

// Per-round play log: a list of {seat, type, size} since the last
// round-start (the engine clears last_play_* on round-end and we mirror
// that here by detecting first_play_of_round transitions). Used by the
// "this round" chain renderer.
let roundChain = [];
let prevIsFirstPlay = 1;

function typeName(t_) {
  const arr = tList('mm.type_names');
  return (arr && arr[t_]) ? arr[t_] : `T${t_}`;
}

// Size implied for a play of `typ` at the current state. After the
// action lands, the resulting state's last_play_size is what was just
// played, so this is also how we read it back for history rendering.
function impliedSize(st, typ) {
  if (!st) return 1;
  if (st.is_first_play_of_round === 1) return 1;
  const sa = st.strength_above || [];
  if (sa[typ] && sa[typ][st.last_play_type]) return st.last_play_size;       // A2
  return st.last_play_size + 1;                                              // A1
}

function formatMove(info, actionId, _actor) {
  if (actionId == null) return t('mm.action_start');
  if (actionId === PASS_ACTION) return t('mm.action_pass');
  if (actionId >= 0 && actionId < PASS_ACTION) {
    return t('mm.action_play', { type: typeName(actionId) });
  }
  return t('mm.action_unknown', { id: actionId });
}

function makeMask(typ, opts) {
  opts = opts || {};
  const el = document.createElement('div');
  el.className = 'mm-mask mm-mask-t' + typ;
  if (opts.small) el.classList.add('mm-mask-small');
  if (opts.tiny) el.classList.add('mm-mask-tiny');
  const name = typeName(typ);
  el.textContent = name ? name.slice(0, 1) : String(typ);
  el.title = name;
  return el;
}

function renderStatus(parent, st) {
  const wrap = document.createElement('div');
  wrap.className = 'mm-status';
  const label = document.createElement('div');
  label.className = 'mm-status-label';
  label.textContent = t('mm.last_play_label');
  wrap.appendChild(label);

  const line = document.createElement('div');
  line.className = 'mm-status-line';
  const isFirst = !st || st.is_first_play_of_round === 1;
  if (isFirst || st.last_play_type < 0) {
    wrap.classList.add('mm-status-empty');
    line.textContent = t('mm.last_play_empty');
  } else {
    line.appendChild(makeMask(st.last_play_type));
    const times = document.createElement('span');
    times.className = 'mm-times';
    times.textContent = t('mm.times');
    line.appendChild(times);
    const sz = document.createElement('span');
    sz.textContent = String(st.last_play_size);
    line.appendChild(sz);
  }
  wrap.appendChild(line);
  parent.appendChild(wrap);
}

// Render this round's play log as a left-to-right chain of (seat, mask, ×size).
// Resets whenever the engine signals a new round.
function renderChain(parent, st) {
  const wrap = document.createElement('div');
  wrap.className = 'mm-chain';
  const title = document.createElement('div');
  title.className = 'mm-chain-title';
  title.textContent = t('mm.chain_title');
  wrap.appendChild(title);

  const body = document.createElement('div');
  body.className = 'mm-chain-body';
  if (roundChain.length === 0) {
    const empty = document.createElement('span');
    empty.className = 'mm-chain-empty';
    empty.textContent = t('mm.last_play_empty');
    body.appendChild(empty);
  } else {
    for (let i = 0; i < roundChain.length; i++) {
      const item = roundChain[i];
      if (i > 0) {
        const arr = document.createElement('span');
        arr.className = 'mm-chain-arrow';
        arr.textContent = '→';
        body.appendChild(arr);
      }
      const cell = document.createElement('div');
      cell.className = 'mm-chain-cell';
      const seat = document.createElement('div');
      seat.className = 'mm-chain-seat';
      seat.textContent = 'P' + item.seat;
      cell.appendChild(seat);
      cell.appendChild(makeMask(item.type, { small: true }));
      const sz = document.createElement('div');
      sz.className = 'mm-chain-size';
      sz.textContent = '× ' + item.size;
      cell.appendChild(sz);
      body.appendChild(cell);
    }
  }
  wrap.appendChild(body);
  parent.appendChild(wrap);
}

// Topological-sort the strength closure into ranks (left = weaker,
// right = stronger; rank = longest chain of "beats it" predecessors).
// Lays out nodes with deterministic vertical slots so each rank's
// vertical position is stable, then draws the cover relation as SVG
// arrows.
function renderStrength(parent, st) {
  const wrap = document.createElement('div');
  wrap.className = 'mm-strength';
  const title = document.createElement('div');
  title.className = 'mm-strength-title';
  title.textContent = t('mm.strength_title');
  wrap.appendChild(title);

  const body = document.createElement('div');
  body.className = 'mm-strength-body';
  const sa = (st && st.strength_above) || [];

  const hasEdge = (a, b) => !!(sa[a] && sa[a][b]);
  // Involved = any node touched by an edge.
  const involved = new Set();
  for (let a = 0; a < TYPE_COUNT; a++) {
    for (let b = 0; b < TYPE_COUNT; b++) {
      if (hasEdge(a, b)) { involved.add(a); involved.add(b); }
    }
  }

  if (involved.size === 0) {
    const empty = document.createElement('div');
    empty.className = 'mm-strength-empty';
    empty.textContent = t('mm.strength_empty');
    body.appendChild(empty);
    wrap.appendChild(body);
    parent.appendChild(wrap);
    return;
  }

  // Cover relation = direct dominance (no intermediate).
  const isCover = new Set();
  const coverKey = (a, b) => a + '>' + b;
  for (const a of involved) {
    for (const b of involved) {
      if (a === b || !hasEdge(a, b)) continue;
      let cover = true;
      for (const c of involved) {
        if (c === a || c === b) continue;
        if (hasEdge(a, c) && hasEdge(c, b)) { cover = false; break; }
      }
      if (cover) isCover.add(coverKey(a, b));
    }
  }

  // Rank = longest path of "is beaten by" arrows ending here. That puts
  // weakest at rank 0 (left) and strongest at the highest rank (right) —
  // matches the user's spec "左是小的右是大的". Recursion is safe: the
  // closure is acyclic.
  const types = Array.from(involved).sort((a, b) => a - b);
  const rank = new Map();
  function rankOf(u) {
    if (rank.has(u)) return rank.get(u);
    let r = 0;
    // u is at least one step beyond anything that u beats.
    for (const v of types) {
      if (hasEdge(u, v)) r = Math.max(r, rankOf(v) + 1);
    }
    rank.set(u, r);
    return r;
  }
  for (const t_ of types) rankOf(t_);

  const maxRank = Math.max.apply(null, Array.from(rank.values()));
  const cols = [];
  for (let i = 0; i <= maxRank; i++) cols.push([]);
  for (const t_ of types) cols[rank.get(t_)].push(t_);
  for (const col of cols) col.sort((a, b) => a - b);

  // Layout constants — must match CSS visual mask size.
  const COL_W = 70;          // horizontal slot per rank
  const ROW_H = 50;          // vertical slot per node within rank
  const NODE_R = 18;         // mask radius for arrow endpoint trim
  const PAD_X = 24;
  const PAD_Y = 16;

  // Cover-edge predecessor list per node (in left-to-right traversal,
  // a node's "predecessor" is whatever it directly beats — that node
  // sits one column to its left).
  const preds = new Map();
  for (const k of isCover) {
    const [as, bs] = k.split('>');
    const a = parseInt(as, 10);  // stronger (right column)
    const b = parseInt(bs, 10);  // weaker (left column)
    if (!preds.has(a)) preds.set(a, []);
    preds.get(a).push(b);
  }

  // Slot assignment by barycenter median, sweeping left-to-right.
  // For each node, pick the y-slot of the *upper* of its predecessors
  // (median, ties broken upward). That way a 2-into-1 merge lands on
  // one branch's row — the other branch becomes the visible bend.
  // Resulting layout for the diamond case 红→橙, 粉→橙, 橙→青:
  //   row 0: 红 → 橙 → 青     (straight main line)
  //   row 1: 粉 ↗              (tributary bending up)
  const slot = new Map();
  for (let i = 0; i < cols[0].length; i++) slot.set(cols[0][i], i);
  for (let c = 1; c <= maxRank; c++) {
    const items = cols[c].slice();
    const desired = new Map();
    for (const v of items) {
      const ps = (preds.get(v) || []).filter(p => slot.has(p));
      if (ps.length === 0) {
        desired.set(v, 0);
      } else {
        const ys = ps.map(p => slot.get(p)).sort((x, y) => x - y);
        // median, ties → upper (lower index)
        desired.set(v, ys[Math.floor((ys.length - 1) / 2)]);
      }
    }
    items.sort((a, b) => desired.get(a) - desired.get(b) || a - b);
    const used = new Set();
    for (const v of items) {
      let want = desired.get(v);
      while (used.has(want)) want += 1;
      used.add(want);
      slot.set(v, want);
    }
  }

  const allSlots = Array.from(slot.values());
  const minSlot = Math.min.apply(null, allSlots);
  const maxSlot = Math.max.apply(null, allSlots);
  const slotsHigh = maxSlot - minSlot + 1;
  const width = PAD_X * 2 + (maxRank + 1) * COL_W;
  const height = PAD_Y * 2 + slotsHigh * ROW_H;

  const pos = new Map();
  for (const t_ of types) {
    const x = PAD_X + (rank.get(t_) || 0) * COL_W + COL_W / 2;
    const y = PAD_Y + (slot.get(t_) - minSlot) * ROW_H + ROW_H / 2;
    pos.set(t_, { x, y });
  }

  const svgNS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(svgNS, 'svg');
  svg.setAttribute('width', String(width));
  svg.setAttribute('height', String(height));
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.classList.add('mm-strength-svg');

  // Arrowhead marker.
  const defs = document.createElementNS(svgNS, 'defs');
  const marker = document.createElementNS(svgNS, 'marker');
  marker.setAttribute('id', 'mm-arrowhead');
  marker.setAttribute('viewBox', '0 0 10 10');
  marker.setAttribute('refX', '9');
  marker.setAttribute('refY', '5');
  marker.setAttribute('markerWidth', '7');
  marker.setAttribute('markerHeight', '7');
  marker.setAttribute('orient', 'auto-start-reverse');
  const arrow = document.createElementNS(svgNS, 'path');
  arrow.setAttribute('d', 'M 0 0 L 10 5 L 0 10 z');
  arrow.setAttribute('fill', '#475569');
  marker.appendChild(arrow);
  defs.appendChild(marker);
  svg.appendChild(defs);

  // Draw cover edges. Edge a→b means a beats b; in our layout that's
  // strong→weak, i.e. right→left. We *draw* arrows pointing toward the
  // *stronger* side (so the arrow tip rests at a, tail at b), which is
  // the convention the user asked for ("箭头指向更强的一方").
  for (const k of isCover) {
    const [as, bs] = k.split('>');
    const a = parseInt(as, 10);  // stronger
    const b = parseInt(bs, 10);  // weaker
    const pa = pos.get(a);
    const pb = pos.get(b);
    if (!pa || !pb) continue;
    // Trim from b toward a so the arrow doesn't sit under the mask.
    const dx = pa.x - pb.x;
    const dy = pa.y - pb.y;
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    const ux = dx / len;
    const uy = dy / len;
    const x1 = pb.x + ux * NODE_R;
    const y1 = pb.y + uy * NODE_R;
    const x2 = pa.x - ux * NODE_R;
    const y2 = pa.y - uy * NODE_R;
    const line = document.createElementNS(svgNS, 'line');
    line.setAttribute('x1', String(x1));
    line.setAttribute('y1', String(y1));
    line.setAttribute('x2', String(x2));
    line.setAttribute('y2', String(y2));
    line.setAttribute('stroke', '#475569');
    line.setAttribute('stroke-width', '1.8');
    line.setAttribute('marker-end', 'url(#mm-arrowhead)');
    svg.appendChild(line);
  }

  // Draw nodes as circles + text (foreignObject would let us reuse
  // makeMask, but circles render more crisply at small sizes).
  const PALETTE = ['#ef4444', '#f59e0b', '#16a34a', '#06b6d4', '#6366f1', '#db2777'];
  for (const [typ, p] of pos.entries()) {
    const g = document.createElementNS(svgNS, 'g');
    const circle = document.createElementNS(svgNS, 'circle');
    circle.setAttribute('cx', String(p.x));
    circle.setAttribute('cy', String(p.y));
    circle.setAttribute('r', String(NODE_R));
    circle.setAttribute('fill', PALETTE[typ] || '#888');
    circle.setAttribute('stroke', 'rgba(0,0,0,0.18)');
    circle.setAttribute('stroke-width', '2');
    g.appendChild(circle);
    const text = document.createElementNS(svgNS, 'text');
    text.setAttribute('x', String(p.x));
    text.setAttribute('y', String(p.y + 1));
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('dominant-baseline', 'middle');
    text.setAttribute('fill', '#fff');
    text.setAttribute('font-weight', '800');
    text.setAttribute('font-size', '14');
    text.setAttribute('style', 'text-shadow: 0 1px 2px rgba(0,0,0,0.45);');
    const name = typeName(typ);
    text.textContent = name ? name.slice(0, 1) : String(typ);
    g.appendChild(text);
    svg.appendChild(g);
  }

  body.appendChild(svg);
  wrap.appendChild(body);
  parent.appendChild(wrap);
}

function rotatedOpponents(human, numP) {
  const out = [];
  for (let k = 1; k < numP; k++) out.push((human + k) % numP);
  return out;
}

function renderOpponents(parent, st, ctx, numPlayers) {
  const wrap = document.createElement('div');
  wrap.className = 'mm-opponents';
  const human = (ctx.state && ctx.state.humanPlayer != null) ? ctx.state.humanPlayer : 0;
  const players = (st && st.players) || [];
  const cur = st ? st.current_player : -1;
  const isTerminal = !!(st && st.is_terminal);
  const passed = (st && st.passed_this_round) || [];

  for (const p of rotatedOpponents(human, numPlayers)) {
    const pd = players[p] || {};
    const row = document.createElement('div');
    row.className = 'mm-opp';
    row.setAttribute('data-seat', String(p));
    if (cur === p && !isTerminal) row.classList.add('mm-opp-current');
    if (passed[p] === 1 && !isTerminal) row.classList.add('mm-opp-passed');

    const tag = document.createElement('div');
    tag.className = 'mm-opp-tag';
    tag.textContent = t('mm.player_n', { n: p });
    if (passed[p] === 1 && !isTerminal) {
      const pill = document.createElement('span');
      pill.className = 'mm-opp-pass-pill';
      pill.textContent = t('mm.opp_passed_pill');
      tag.appendChild(pill);
    }
    row.appendChild(tag);

    let count = 0;
    if (typeof pd.hand_total === 'number' && pd.hand_total >= 0) count = pd.hand_total;
    else if (Array.isArray(pd.hand_count)) {
      let any = true;
      let s = 0;
      for (const v of pd.hand_count) { if (v < 0) { any = false; break; } s += v; }
      if (any) count = s;
    }

    const cnt = document.createElement('div');
    cnt.className = 'mm-opp-count';
    cnt.textContent = t('mm.opp_count', { n: count });
    row.appendChild(cnt);

    wrap.appendChild(row);
  }
  parent.appendChild(wrap);
}

function renderSelf(parent, st, ctx, numPlayers) {
  const wrap = document.createElement('div');
  wrap.className = 'mm-self';
  const human = (ctx.state && ctx.state.humanPlayer != null) ? ctx.state.humanPlayer : 0;
  const cur = st ? st.current_player : -1;
  const isTerminal = !!(st && st.is_terminal);
  const myTurn = !isTerminal && cur === human && ctx.canPlay;
  if (myTurn) wrap.classList.add('mm-self-current');

  const tag = document.createElement('div');
  tag.className = 'mm-self-tag';
  const left = document.createElement('span');
  left.textContent = t('mm.player_n', { n: human }) + t('mm.player_self_suffix');
  tag.appendChild(left);
  const hint = document.createElement('span');
  hint.className = 'mm-self-hint';
  hint.textContent = t('mm.self_tag');
  tag.appendChild(hint);
  wrap.appendChild(tag);

  const legalSet = new Set(ctx.legalActions || []);

  const stacks = document.createElement('div');
  stacks.className = 'mm-self-stacks';
  const players = (st && st.players) || [];
  const my = players[human] || {};
  const myHand = my.hand_count || [0, 0, 0, 0, 0, 0];
  const sa = (st && st.strength_above) || [];
  const last_t = st && !st.is_first_play_of_round ? st.last_play_type : -1;

  for (let typ = 0; typ < TYPE_COUNT; typ++) {
    const stack = document.createElement('div');
    stack.className = 'mm-stack';
    stack.setAttribute('data-type', String(typ));
    const cnt = myHand[typ] || 0;
    const playable = myTurn && legalSet.has(typ);
    if (cnt <= 0) stack.setAttribute('data-empty', '1');
    if (!myTurn) stack.setAttribute('data-disabled', '1');
    if (myTurn && cnt > 0 && !playable) stack.setAttribute('data-illegal', '1');

    stack.appendChild(makeMask(typ));

    const cntEl = document.createElement('div');
    cntEl.className = 'mm-stack-count';
    cntEl.textContent = String(cnt);
    stack.appendChild(cntEl);

    if (playable) {
      stack.addEventListener('click', () => ctx.submitAction(typ));
    }
    stacks.appendChild(stack);
  }
  wrap.appendChild(stacks);
  parent.appendChild(wrap);
}

function renderControls(parent, st, ctx) {
  const wrap = document.createElement('div');
  wrap.className = 'mm-controls';
  const human = (ctx.state && ctx.state.humanPlayer != null) ? ctx.state.humanPlayer : 0;
  const cur = st ? st.current_player : -1;
  const isTerminal = !!(st && st.is_terminal);
  const myTurn = !isTerminal && cur === human && ctx.canPlay;
  const legalSet = new Set(ctx.legalActions || []);

  const row = document.createElement('div');
  row.className = 'mm-controls-row';

  const passBtn = document.createElement('button');
  passBtn.type = 'button';
  passBtn.className = 'mm-btn mm-btn-danger';
  passBtn.textContent = t('mm.btn_pass');
  const canPass = myTurn && legalSet.has(PASS_ACTION);
  passBtn.disabled = !canPass;
  passBtn.addEventListener('click', () => {
    if (!canPass) return;
    ctx.submitAction(PASS_ACTION);
  });
  row.appendChild(passBtn);

  wrap.appendChild(row);

  const status = document.createElement('div');
  status.className = 'mm-status-line';
  if (isTerminal) {
    status.textContent = t('mm.terminal_status');
    status.classList.add('spectator');
  } else if (!myTurn) {
    status.textContent = t('mm.spectator');
    status.classList.add('spectator');
  } else if (st && st.is_first_play_of_round === 1 && !canPass) {
    status.textContent = t('mm.status_must_open');
  } else {
    status.textContent = t('mm.status_pick_action');
  }
  wrap.appendChild(status);

  parent.appendChild(wrap);
}

function renderBoard(container, gameState, ctx) {
  container.innerHTML = '';
  const root = document.createElement('div');
  root.className = 'mm-root';

  const st = gameState && gameState.state ? gameState.state : null;
  const numPlayers = st ? (st.num_players || 2) : (ctx.numPlayers || 2);

  // Maintain the per-round chain. The engine clears last_play_* and
  // sets is_first_play_of_round=1 at round-end; we mirror that here.
  // Within a round, append every new (last_play_type, last_play_size,
  // last_player_to_play) we observe.
  if (st) {
    if (st.is_first_play_of_round === 1 && prevIsFirstPlay !== 1) {
      // Round just ended (engine reset its scalars). Clear our log too.
      roundChain = [];
    }
    if (st.is_first_play_of_round !== 1 && st.last_play_type >= 0) {
      const tail = roundChain[roundChain.length - 1];
      const sig = st.last_player_to_play + ':' + st.last_play_type + ':' + st.last_play_size;
      const tailSig = tail ? (tail.seat + ':' + tail.type + ':' + tail.size) : '';
      if (sig !== tailSig) {
        roundChain.push({
          seat: st.last_player_to_play,
          type: st.last_play_type,
          size: st.last_play_size,
        });
      }
    }
    prevIsFirstPlay = st.is_first_play_of_round;
  }

  const legalActions = (gameState && gameState.legal_actions) || [];
  const ctx2 = Object.assign({}, ctx, { legalActions, numPlayers });

  renderStatus(root, st);
  renderChain(root, st);
  renderStrength(root, st);
  renderOpponents(root, st, ctx2, numPlayers);
  renderSelf(root, st, ctx2, numPlayers);
  renderControls(root, st, ctx2);

  container.appendChild(root);
}

function describeTransition(_prev, _next, _info, actionId) {
  if (actionId == null) return null;
  return [{ type: 'highlight', target: '.mm-status', className: 'anim-highlight', duration: 400 }];
}

createApp({
  gameId: 'maskmen',
  gameTitle: t('mm.title'),
  gameIntro: t('mm.intro'),
  players: { min: 3, max: 4 },
  renderBoard,
  describeTransition,
  formatOpponentMove: formatMove,
  formatSuggestedMove: formatMove,
  getPlayerSymbol: (p) => t('mm.player_n', { n: p }),
  aiStepPause: 900,
  difficulties: ['heuristic', 'casual', 'expert'],
  defaultDifficulty: 'expert',
  defaultNumPlayers: 3,
  defaultSeat: 'random',
  disableForce: true,
  showWinrateDefault: false,
  onGameStart: () => { roundChain = []; prevIsFirstPlay = 1; },
});
