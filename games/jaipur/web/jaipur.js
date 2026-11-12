import { createApp } from '/static/general/app.js';
import { apiPost, API_BASE } from '/static/general/api.js';

// Count-based Jaipur frontend, played like BGA: click cards to select them
// (market goods, market camels, hand cards, own camels), then press the one
// button that the selection allows — take, take camels, sell ×k or trade.
// The selection lives only on the page. A trade is submitted as its atom
// sequence (take atoms by ascending type, then give atoms, camels last;
// see jaipur_state.h); the server only sees legal atoms.
const GOOD_ICON = ['💎','💰','🥈','🌶','🧵','🧶'];
const GOOD_LABEL = ['钻石','黄金','白银','香料','布匹','皮革'];
const GOOD_CSS = ['good-diamond','good-gold','good-silver','good-spice','good-cloth','good-leather'];
const CAMEL = 6;

// Action ids (must match jaipur_state.h)
const A_TAKE = 0, A_TAKE_CAMELS = 6, A_SELL = 7, A_TRADE_TAKE = 49, A_TRADE_GIVE = 79;
const sellAction = (g, k) => A_SELL + g * 7 + (k - 1);
const tradeTakeAction = (g, k) => A_TRADE_TAKE + g * 5 + (k - 1);
const tradeGiveAction = (x, k) => A_TRADE_GIVE + x * 5 + (k - 1);
const MIN_SELL = [2, 2, 2, 1, 1, 1];

const sum = (a) => a.reduce((x, y) => x + y, 0);
function cardIcon(c){return c===0?'🐫':GOOD_ICON[c-1];}
function goodName(x){return x===CAMEL?'🐫骆驼':GOOD_ICON[x]+GOOD_LABEL[x];}

// ---- Page-side selection, reset whenever the position changes ----
let sel = null, selKey = null;
function freshSelection() { return { mkt: [0,0,0,0,0,0,0], hand: [0,0,0,0,0,0], herd: 0 }; }
function selectionFor(gs, ctx) {
  const st = gs.state || {};
  const key = `${ctx.state.sessionId}|${st.plies}|${gs.current_player}|${(st.trade_take||[]).join(',')}|${(st.trade_give||[]).join(',')}`;
  if (key !== selKey) { selKey = key; sel = freshSelection(); }
  return sel;
}

// What the current selection means: {atoms, label} for a submittable move,
// or {hint} explaining what is missing. Mirrors the rulebook; the engine's
// legal set is the final check (every atom is checked before submitting).
function planMove(s, st, my, legal) {
  const hand = st['hand_count_'+my] || [0,0,0,0,0,0];
  const herd = st['camel_count_'+my] || 0;
  const mk = sum(s.mkt.slice(0, 6)), mc = s.mkt[CAMEL], hk = sum(s.hand), hc = s.herd;
  if (mk + mc + hk + hc === 0)
    return { hint: '点击市场的牌拿取或交换；点击手牌出售；交换时可点自己的骆驼支付' };
  if (mc > 0) {
    if (mk + hk + hc > 0) return { hint: '拿骆驼必须单独进行（一次拿走市场上全部骆驼）' };
    return legal.has(A_TAKE_CAMELS) ? { atoms: [A_TAKE_CAMELS], label: `拿走全部骆驼 ×${mc}` } : { hint: '不能拿骆驼' };
  }
  if (mk === 0) {
    if (hc > 0) return { hint: '骆驼只能在交换时付出：再选市场上要换来的货物' };
    const types = [0,1,2,3,4,5].filter(g => s.hand[g] > 0);
    if (types.length > 1) return { hint: '一次只能出售一种货物' };
    const g = types[0], k = s.hand[g];
    if (k < MIN_SELL[g]) return { hint: `${GOOD_LABEL[g]}至少要卖 ${MIN_SELL[g]} 张` };
    const a = sellAction(g, k);
    return legal.has(a) ? { atoms: [a], label: `出售 ${GOOD_ICON[g]}${GOOD_LABEL[g]} ×${k}` } : { hint: '不能出售' };
  }
  if (mk === 1 && hk + hc === 0) {
    const g = s.mkt.findIndex((n, i) => i < CAMEL && n > 0);
    if (sum(hand) >= 7) return { hint: '手牌已满 7 张，不能再拿货物' };
    return legal.has(A_TAKE + g) ? { atoms: [A_TAKE + g], label: `拿取 ${GOOD_ICON[g]}${GOOD_LABEL[g]}` } : { hint: '不能拿取' };
  }
  // Trade: at least 2 market goods for as many hand goods / own camels.
  if (mk < 2) return { hint: '交换至少要换 2 张：再选一张市场货物' };
  const give = hk + hc;
  if (give < mk) return { hint: `交换：还需选 ${mk - give} 张手牌或骆驼付出` };
  if (give > mk) return { hint: `交换：付出的牌多了 ${give - mk} 张` };
  for (let g = 0; g < 6; ++g)
    if (s.mkt[g] > 0 && s.hand[g] > 0) return { hint: `不能拿回和付出同一种货物（${GOOD_LABEL[g]}）` };
  if (sum(hand) + hc > 7) return { hint: '交换后手牌会超过 7 张：少付骆驼、多付货物' };
  if (hc > herd) return { hint: '骆驼不够' };
  const atoms = [];
  for (let g = 0; g < 6; ++g) if (s.mkt[g] > 0) atoms.push(tradeTakeAction(g, s.mkt[g]));
  for (let x = 0; x < 6; ++x) if (s.hand[x] > 0) atoms.push(tradeGiveAction(x, s.hand[x]));
  if (hc > 0) atoms.push(tradeGiveAction(CAMEL, hc));
  if (!legal.has(atoms[0])) return { hint: '这个交换不合法' };
  const takeTxt = [0,1,2,3,4,5].filter(g => s.mkt[g]).map(g => GOOD_ICON[g] + '×' + s.mkt[g]).join(' ');
  const giveTxt = [0,1,2,3,4,5].filter(g => s.hand[g]).map(g => GOOD_ICON[g] + '×' + s.hand[g])
    .concat(hc ? ['🐫×' + hc] : []).join(' ');
  return { atoms, label: `交换：换入 ${takeTxt}，付出 ${giveTxt}` };
}

// Submit a whole move. Trade atoms before the last go straight to the
// session; the last one goes through the app so the AI reply and the
// animation run as for any move. A failure rolls the partial trade back.
async function submitMove(ctx, atoms) {
  const sid = ctx.state && ctx.state.sessionId;
  if (!sid || !atoms.length) return;
  try {
    for (const a of atoms.slice(0, -1)) {
      if (!ctx.state.gameState.legal_actions.includes(a)) throw new Error(`illegal atom ${a}`);
      ctx.state.gameState = await apiPost(`${API_BASE}/${sid}/action`, { action_id: a });
    }
  } catch (e) {
    await rollbackTrade(ctx);
    ctx.setInfoStatus(`交换提交失败：${e.message}`);
    return;
  }
  ctx.submitAction(atoms[atoms.length - 1]);
}

// Undo the chosen trade atoms (client-side step-back; the rules have no
// cancel action — a trade, once started, always completes).
async function rollbackTrade(ctx) {
  const sid = ctx.state && ctx.state.sessionId;
  if (!sid) return;
  for (let i = 0; i < 12; i++) {
    const st = (ctx.state.gameState && ctx.state.gameState.state) || {};
    if (sum(st.trade_take || []) === 0) break;
    const data = await apiPost(`${API_BASE}/${sid}/step-back`, {});
    if (!data || !data.state) break;
    ctx.state.gameState = data;
  }
  ctx.rerender();
}

function renderBoard(container, gs, ctx) {
  const st = gs.state || {};
  // Replay frames carry no human_player; the app-level seat is the only
  // perspective that exists in both live play and replay.
  const my = ctx.state.humanPlayer;
  if (my !== 0 && my !== 1) throw new Error(`jaipur: invalid humanPlayer ${my}`);
  const opp = 1 - my;
  const mc = st.market_count || [0,0,0,0,0,0,0];  // [g0..g5, camels]
  const myHand = st['hand_count_'+my] || [0,0,0,0,0,0];
  const oppHand = st['hand_count_'+opp] || [0,0,0,0,0,0];
  const oppHandTotal = st['hand_size_'+opp] != null ? st['hand_size_'+opp] : sum(oppHand);
  const myHerd = st['camel_count_'+my] || 0;
  const tradeTake = st.trade_take || [0,0,0,0,0,0];
  const tradeGive = st.trade_give || [0,0,0,0,0,0,0];
  const midTrade = sum(tradeTake) > 0;
  const legal = new Set(gs.legal_actions || []);
  const deckRemain = st.deck_remaining != null ? st.deck_remaining : 0;
  const turn = gs.current_player;
  const selecting = ctx.canPlay && turn === my && !midTrade && !gs.is_terminal;
  const s = selecting ? selectionFor(gs, ctx) : freshSelection();

  let h = '<div class="jaipur-root">';

  // --- Opponent (face-down hand) ---
  h += `<div class="jaipur-player opponent${turn===opp?' active-turn':''}" data-player="${opp}">`;
  h += `<div class="jp-header"><span class="jp-name">对手${turn===opp?' ▶ 思考中':''}</span>`;
  h += `<span class="jp-stat">🪙 ${st['rupees_'+opp]||0}</span>`;
  h += `<span class="jp-stat camel">🐫 ${st['camel_count_'+opp]||0}</span>`;
  const obc = st['bonus_count_'+opp]||[0,0,0];
  h += `<span class="jp-stat bonus">🎁${obc[0]}·${obc[1]}·${obc[2]}</span></div>`;
  h += `<div class="jaipur-hand">`;
  if (ctx.revealHidden) {
    for (let g=0; g<6; ++g)
      for (let i=0; i<oppHand[g]; ++i)
        h += `<div class="jp-card ${GOOD_CSS[g]} revealed" data-g="${g}">${GOOD_ICON[g]}</div>`;
    for (let i=oppHandTotal; i<7; i++) h += `<div class="jp-card empty-slot"></div>`;
  } else {
    for (let i=0;i<7;i++) h += i<oppHandTotal ? `<div class="jp-card back" data-hand-idx="${i}" data-player="${opp}">?</div>` : `<div class="jp-card empty-slot"></div>`;
  }
  h += `</div></div>`;

  // --- Market: one card per slot, goods grouped by type, camels last ---
  h += `<div class="jaipur-market">`;
  h += `<div class="jp-deck" title="牌堆剩余 ${deckRemain} 张"><div class="jp-deck-icon">🂠</div><div class="jp-deck-num">${deckRemain}</div><div class="jp-deck-label">牌堆</div></div>`;
  let shown = 0;
  for (let g=0; g<=CAMEL; ++g) {
    for (let i=0; i<mc[g]; ++i) {
      const picked = selecting ? i < s.mkt[g] : (g < CAMEL && i < tradeTake[g]);
      const cls = `jp-market ${g===CAMEL?'good-camel':GOOD_CSS[g]}${picked?' selected':''}${selecting?' clickable':''}`;
      const attr = selecting ? `data-sel="mkt" data-x="${g}" data-i="${i}"` : '';
      h += `<div class="${cls}" ${attr} data-g="${g===CAMEL?-1:g}"><div class="icon">${g===CAMEL?'🐫':GOOD_ICON[g]}</div>${picked?'<div class="take-mark">✓</div>':''}</div>`;
      shown++;
    }
  }
  for (let i=shown; i<5; ++i) h += `<div class="jp-market empty"></div>`;
  h += `</div>`;

  // --- Token piles ---
  h += `<div class="jaipur-tokens">`;
  for (let g=0; g<6; ++g)
    h += `<div class="jp-token" data-token="${g}"><span>${GOOD_ICON[g]}</span><div class="val">${st['current_price_'+g]||0}</div><div class="rem">×${st['price_remaining_'+g]||0}</div></div>`;
  h += `<div class="jp-token bonus"><span>🎁3+</span><div class="rem">×${st.bonus_remaining_3||0}</div></div>`;
  h += `<div class="jp-token bonus"><span>🎁4+</span><div class="rem">×${st.bonus_remaining_4||0}</div></div>`;
  h += `<div class="jp-token bonus"><span>🎁5+</span><div class="rem">×${st.bonus_remaining_5||0}</div></div>`;
  h += `</div>`;

  // --- My area: hand + herd ---
  h += `<div class="jaipur-player self${turn===my?' active-turn':''}" data-player="${my}">`;
  h += `<div class="jp-header"><span class="jp-name">你${turn===my?' ▶ 你的回合':''}</span>`;
  h += `<span class="jp-stat">🪙 ${st['rupees_'+my]||0}</span>`;
  h += `<span class="jp-stat camel">🐫 ${myHerd}</span>`;
  const mbc = st['bonus_count_'+my]||[0,0,0];
  const mbt = st['bonus_total_'+my]||0;
  h += `<span class="jp-stat bonus" title="你的奖励标记总值">🎁${mbc[0]}·${mbc[1]}·${mbc[2]} (=${mbt})</span></div>`;
  h += `<div class="jaipur-hand">`;
  for (let g=0; g<6; ++g) {
    for (let i=0; i<myHand[g]; i++) {
      const picked = selecting ? i < s.hand[g] : i < tradeGive[g];
      const cls = `jp-card ${GOOD_CSS[g]}${picked?' selected':''}${selecting?' clickable':''}`;
      const attr = selecting ? `data-sel="hand" data-x="${g}" data-i="${i}"` : '';
      h += `<div class="${cls}" ${attr} data-g="${g}">${GOOD_ICON[g]}</div>`;
    }
  }
  for (let i=sum(myHand); i<7; i++) h += `<div class="jp-card empty-slot"></div>`;
  h += `</div>`;
  if (myHerd > 0) {
    h += `<div class="jaipur-herd">`;
    for (let i=0; i<myHerd; i++) {
      const picked = selecting ? i < s.herd : i < tradeGive[CAMEL];
      const attr = selecting ? `data-sel="herd" data-i="${i}"` : '';
      h += `<div class="jp-camel${picked?' selected':''}${selecting?' clickable':''}" ${attr}>🐫</div>`;
    }
    h += `</div>`;
  }
  h += `</div>`;

  // --- Action bar ---
  if (ctx.canPlay && turn === my) {
    h += `<div class="jaipur-actions">`;
    if (midTrade) {
      // A trade left half-done (e.g. a hint was applied atom by atom):
      // offer its legal continuations, or roll it back.
      h += `<span class="jaipur-hint">交换进行中</span>`;
      for (const a of [...legal].sort((x, y) => x - y)) {
        const info = a >= A_TRADE_GIVE ? `付出 ${goodName(Math.floor((a - A_TRADE_GIVE) / 5))} ×${(a - A_TRADE_GIVE) % 5 + 1}`
                                       : `换入 ${goodName(Math.floor((a - A_TRADE_TAKE) / 5))} ×${(a - A_TRADE_TAKE) % 5 + 1}`;
        h += `<button class="act" data-atom="${a}">${info}</button>`;
      }
      h += `<button class="act danger" data-rollback="1">撤回交换</button>`;
    } else {
      const plan = planMove(s, st, my, legal);
      if (plan.atoms) h += `<button class="act primary" data-move="${plan.atoms.join(',')}">${plan.label}</button>`;
      else h += `<span class="jaipur-hint">${plan.hint}</span>`;
      if (sum(s.mkt) + sum(s.hand) + s.herd > 0) h += `<button class="act" data-clear="1">清除选择</button>`;
    }
    h += `</div>`;
  }

  if (gs.is_terminal) {
    const w = gs.winner;
    const score = (p) => (st['rupees_'+p]||0) + (st['bonus_total_'+p]||0);
    h += `<div class="jaipur-result">`;
    h += w===my ? `🎉 你赢了！` : (w<0 ? `🤝 平局` : `😞 对手赢了`);
    h += `　(你 ${score(my)} : ${score(opp)} 对手)</div>`;
  }
  h += `</div>`;
  container.innerHTML = h;
  bindEvents(container, ctx, s, mc, myHand, myHerd);
}

function bindEvents(container, ctx, s, mc, myHand, myHerd) {
  container.querySelectorAll('[data-sel]').forEach(el => {
    el.onclick = () => {
      const kind = el.dataset.sel, i = parseInt(el.dataset.i);
      if (kind === 'mkt') {
        const x = parseInt(el.dataset.x);
        // Camels are taken all at once, so they are selected all at once.
        if (x === CAMEL) s.mkt[CAMEL] = s.mkt[CAMEL] ? 0 : mc[CAMEL];
        else s.mkt[x] = i < s.mkt[x] ? s.mkt[x] - 1 : Math.min(mc[x], s.mkt[x] + 1);
      } else if (kind === 'hand') {
        const x = parseInt(el.dataset.x);
        s.hand[x] = i < s.hand[x] ? s.hand[x] - 1 : Math.min(myHand[x], s.hand[x] + 1);
      } else {
        s.herd = i < s.herd ? s.herd - 1 : Math.min(myHerd, s.herd + 1);
      }
      ctx.rerender();
    };
  });
  container.querySelectorAll('[data-move]').forEach(el => {
    el.onclick = () => submitMove(ctx, el.dataset.move.split(',').map(Number));
  });
  container.querySelectorAll('[data-atom]').forEach(el => {
    el.onclick = () => ctx.submitAction(parseInt(el.dataset.atom));
  });
  container.querySelectorAll('[data-clear]').forEach(el => {
    el.onclick = () => { sel = freshSelection(); ctx.rerender(); };
  });
  container.querySelectorAll('[data-rollback]').forEach(el => {
    el.onclick = () => rollbackTrade(ctx);
  });
}

// Animation: fly cards between market/hand/token piles.
function describeTransition(prevState, newState, ai, actionId) {
  if (!ai || !ai.type) return [];
  if (!prevState || !prevState.state || !newState || !newState.state) return [];
  const prev = prevState.state, next = newState.state;
  const actor = prevState.current_player;
  const steps = [];
  const FLY = 750;            // slower, easier to follow
  const REVEAL_PAUSE = 800;   // hold so the player can read revealed opp cards
  const CARD_W = 64, CARD_H = 80;
  const mkCard = (cid) => {
    const el = document.createElement('div');
    el.className = 'jp-fly ' + (cid===0 ? 'good-camel' : GOOD_CSS[cid-1]);
    el.textContent = cardIcon(cid);
    return el;
  };
  const handSel = `[data-player="${actor}"] .jaipur-hand`;
  // Face-down iff the actor is the opponent and hidden info is not revealed
  // (live play); replay with reveal shows the opponent's hand face-up.
  const handFaceDown = !!document.querySelector(`${handSel} .jp-card.back`);
  const faceUpCards = (g) => Array.from(document.querySelectorAll(`${handSel} [data-g="${g}"]:not(.flew)`));
  const emptyMarketSlot = (srcEl) => { srcEl.style.opacity = '0.3'; };
  const mktSel = (g) => `.jaipur-market [data-g="${g}"]`;

  // Flip the opponent's face-down cards to show the goods it sells / gives
  // BEFORE they fly away. Returns the flipped elements as fly sources.
  const revealOppCards = (goodList) => {
    const sels = [];
    const backs = Array.from(document.querySelectorAll(
      `[data-player="${actor}"] .jaipur-hand .jp-card.back`));
    let bi = 0;
    for (const g of goodList) {
      if (bi >= backs.length) break;
      const el = backs[bi++];
      el.classList.remove('back');
      el.classList.add(GOOD_CSS[g], 'revealed');
      el.textContent = GOOD_ICON[g];
      sels.push(el);
    }
    return sels;
  };

  switch (ai.type) {
    case 'take': {
      const g = ai.good_type, cid = g + 1;
      steps.push({ type: 'fly', from: mktSel(g), to: handSel,
        createElement: () => mkCard(cid), duration: FLY, width: CARD_W, height: CARD_H,
        onStart: emptyMarketSlot });
      break;
    }
    case 'take_camels': {
      const camelEls = Array.from(document.querySelectorAll(`.jaipur-market [data-g="-1"]`));
      const flights = [];
      for (const src of camelEls)
        flights.push({ from: src,
          to: `[data-player="${actor}"] .jp-stat.camel`,
          createElement: () => mkCard(0), duration: FLY, width: CARD_W, height: CARD_H,
          onStart: emptyMarketSlot });
      steps.push({ type: 'flyGroup', flights });
      break;
    }
    case 'sell': {
      const g = ai.good_type, cid = g + 1, nSold = ai.count || 1;
      const pileSel = `[data-token="${g}"]`;
      let fromSrcs;
      if (handFaceDown) {
        fromSrcs = revealOppCards(Array(nSold).fill(g));
        if (fromSrcs.length) steps.push({ type: 'pause', duration: REVEAL_PAUSE });
      } else {
        fromSrcs = faceUpCards(g);
      }
      const flights = [];
      for (let i=0; i<nSold; ++i) {
        const src = (i < fromSrcs.length) ? fromSrcs[i] : handSel;
        flights.push({ from: src, to: pileSel,
          createElement: () => mkCard(cid), duration: FLY, width: CARD_W, height: CARD_H,
          onStart: (el) => { if (el && el.style) el.style.opacity = '0.3'; } });
      }
      steps.push({ type: 'flyGroup', flights });
      const before = prev['rupees_'+actor]||0, after = next['rupees_'+actor]||0;
      if (after > before) steps.push({ type: 'popup', target: `[data-player="${actor}"] .jp-header`,
        content: `+${after-before}🪙`, duration: 1100 });
      break;
    }
    default: break;
  }

  // Trade swap. A whole trade arrives as one transition (the pipeline
  // collapses the AI's atoms; a human trade is animated on its last atom,
  // and earlier atoms do not move cards), so detect it from the market
  // delta: goods left the market and goods / camels entered it.
  if (ai.type === 'trade_give' && !newState.is_terminal) {
    const pm = prev.market_count || [0,0,0,0,0,0,0];
    const nm = next.market_count || [0,0,0,0,0,0,0];
    const takenGoods = [], givenGoods = [];
    for (let g=0; g<6; ++g) {
      const d = nm[g] - pm[g];
      for (let i=0; i<-d; ++i) takenGoods.push(g);
      for (let i=0; i<d; ++i) givenGoods.push(g);
    }
    const givenCamels = Math.max(0, (nm[6]||0) - (pm[6]||0));
    let giveSrcs = [];
    if (handFaceDown && givenGoods.length) {
      giveSrcs = revealOppCards(givenGoods);
      steps.push({ type: 'pause', duration: REVEAL_PAUSE });
    } else {
      for (const g of givenGoods) {
        const c = faceUpCards(g)[0];
        if (c) { c.classList.add('flew'); giveSrcs.push(c); }
      }
    }
    const flights = [];
    for (const g of takenGoods)
      flights.push({ from: mktSel(g), to: handSel,
        createElement: () => mkCard(g+1), duration: FLY, width: CARD_W, height: CARD_H,
        onStart: emptyMarketSlot });
    givenGoods.forEach((g, i) => {
      flights.push({ from: giveSrcs[i] || handSel, to: `.jaipur-market`,
        createElement: () => mkCard(g+1), duration: FLY, width: CARD_W, height: CARD_H,
        onStart: (el) => { if (el && el.style) el.style.opacity = '0.3'; } });
    });
    for (let i=0; i<givenCamels; ++i)
      flights.push({ from: `[data-player="${actor}"] .jp-stat.camel`, to: `.jaipur-market`,
        createElement: () => mkCard(0), duration: FLY, width: CARD_W, height: CARD_H });
    if (flights.length) steps.push({ type: 'flyGroup', flights });
  }
  return steps;
}

function formatOpponentMove(ai, id, actor) {
  if (!ai || !ai.type) return `动作 ${id}`;
  switch (ai.type) {
    case 'sell': return `出售 ${goodName(ai.good_type)} ×${ai.count}`;
    case 'take': return `拿取 ${goodName(ai.good_type)}`;
    case 'take_camels': return '拿走全部骆驼';
    case 'trade_take': return `交换：换入 ${goodName(ai.good_type)} ×${ai.count}`;
    case 'trade_give': return `交换：付出 ${goodName(ai.good_type)} ×${ai.count}`;
    default: return `动作 ${id}`;
  }
}
function formatSuggestedMove(ai, id, actor) { return formatOpponentMove(ai, id, actor); }

createApp({
  gameId: 'jaipur',
  players: { min: 2, max: 2 },
  renderBoard,
  describeTransition,
  formatOpponentMove,
  formatSuggestedMove,
  defaultDifficulty: 'expert',
  defaultSeat: 'random',
  showWinrateDefault: false,
  disableForce: true,
  gameIntro: '和 BGA 一样：点选市场的牌、手牌或自己的骆驼，再按出现的按钮——拿取 / 拿走全部骆驼 / 出售 ×k（可以只卖一部分）/ 交换（至少 2 张，不能拿回同种货物）。钻石/黄金/白银至少卖 2 张。',
});
