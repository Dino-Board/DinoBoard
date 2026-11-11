import { createApp } from '/static/general/app.js';
import { apiPost, API_BASE } from '/static/general/api.js';

// Count-based Jaipur frontend. Market & hands are count vectors; the
// opponent's hand is face-down (only its size is public). Actions are
// type-based (sell g / take g / take_camels / exchange sub-actions by type).
const GOOD_ICON = ['💎','💰','🥈','🌶','🧵','🧶'];
const GOOD_LABEL = ['钻石','黄金','白银','香料','布匹','皮革'];
const GOOD_CSS = ['good-diamond','good-gold','good-silver','good-spice','good-cloth','good-leather'];

// Action ids (must match jaipur_state.h)
const A_SELL=0, A_TAKE=6, A_TAKE_CAMELS=12, A_ENTER=13;
const A_EXCH_MKT=14, A_EXCH_MKT_DONE=20, A_EXCH_CAMEL=21, A_EXCH_HAND=29, A_EXCH_CANCEL=35;
const BONUS_LABEL = ['三连','四连','五连'];

function cardIcon(c){return c===0?'🐫':GOOD_ICON[c-1];}
function cardLabel(c){return c===0?'骆驼':GOOD_LABEL[c-1];}

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
  const oppHandTotal = st['hand_size_'+opp] != null ? st['hand_size_'+opp]
                                                    : oppHand.reduce((a,b)=>a+b,0);
  const ep = st.exch_phase || 0;
  const tc = st.exch_take_count || [0,0,0,0,0,0];
  const gv = st.exch_give_count || [0,0,0,0,0,0];
  const ccam = st.exch_camel_count || 0, minCam = st.exch_min_camels || 0, maxCam = st.exch_max_camels || 0;
  const total = st.exch_total || 0;
  const legal = new Set(gs.legal_actions || []);
  const has = (id) => legal.has(id);
  const deckRemain = st.deck_remaining != null ? st.deck_remaining : 0;

  let h = '<div class="jaipur-root">';

  // --- Opponent (face-down hand) ---
  const turn = gs.current_player;
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

  // --- Market: render each card individually (always 5 cards = fixed slots,
  // no layout shift). Goods of the same type sit next to each other; camels
  // last. Clicking a goods card takes/selects it; clicking a camel takes all
  // camels. Empty placeholders pad to 5 if the deck is exhausted. ---
  const myHandTotalNow = myHand.reduce((a,b)=>a+b,0);
  const handFull = myHandTotalNow >= 7;
  h += `<div class="jaipur-market">`;
  h += `<div class="jp-deck" title="牌堆剩余 ${deckRemain} 张"><div class="jp-deck-icon">🂠</div><div class="jp-deck-num">${deckRemain}</div><div class="jp-deck-label">牌堆</div></div>`;
  let marketCardsShown = 0;
  for (let g=0; g<6; ++g) {
    for (let i=0; i<mc[g]; ++i) {
      const takeAct = A_TAKE + g, exchAct = A_EXCH_MKT + g;
      // idle: take (legal iff hand not full); exchange market phase: select
      const idleOk = (ep===0 && has(takeAct));
      const exchOk = (ep===1 && has(exchAct));
      const clickable = idleOk || exchOk;
      // mark the first tc[g] cards of this type as "selected" in exchange
      const sel = (ep===1 && i < tc[g]);
      const act = idleOk ? takeAct : (exchOk ? exchAct : -1);
      const cls = `jp-market ${GOOD_CSS[g]}${sel?' selected':''}${clickable?' clickable':''}`;
      h += `<div class="${cls}" ${act>=0?`data-act="${act}"`:''} data-g="${g}"><div class="icon">${GOOD_ICON[g]}</div>${sel?'<div class="take-mark">✓</div>':''}</div>`;
      marketCardsShown++;
    }
  }
  for (let i=0; i<mc[6]; ++i) {
    const clickable = (ep===0 && has(A_TAKE_CAMELS));
    const cls = `jp-market good-camel${clickable?' clickable':''}`;
    h += `<div class="${cls}" ${clickable?`data-act="${A_TAKE_CAMELS}"`:''} data-g="-1" ${ep===0?'title="点击取走所有骆驼"':''}><div class="icon">🐫</div>${i===0&&ep===0?'<div class="hint">取全部</div>':''}</div>`;
    marketCardsShown++;
  }
  for (let i=marketCardsShown; i<5; ++i) h += `<div class="jp-market empty"></div>`;
  h += `</div>`;

  // --- Token piles ---
  h += `<div class="jaipur-tokens">`;
  for (let g=0; g<6; ++g)
    h += `<div class="jp-token" data-token="${g}"><span>${GOOD_ICON[g]}</span><div class="val">${st['current_price_'+g]||0}</div><div class="rem">×${st['price_remaining_'+g]||0}</div></div>`;
  h += `<div class="jp-token bonus"><span>🎁3+</span><div class="rem">×${st.bonus_remaining_3||0}</div></div>`;
  h += `<div class="jp-token bonus"><span>🎁4+</span><div class="rem">×${st.bonus_remaining_4||0}</div></div>`;
  h += `<div class="jp-token bonus"><span>🎁5+</span><div class="rem">×${st.bonus_remaining_5||0}</div></div>`;
  h += `</div>`;

  // --- Exchange phase indicator ---
  if (ep > 0) {
    const phases = ['','①选市场牌','②选骆驼数','③选手牌'];
    const takeTot = tc.reduce((a,b)=>a+b,0);
    const giveTot = gv.reduce((a,b)=>a+b,0);
    h += `<div class="jaipur-ex-phase">交换进行中：${phases[ep]}　市场${takeTot}｜🐫${ccam}｜手牌${giveTot} / 共${total}</div>`;
  }

  // --- My area (count form hand) ---
  h += `<div class="jaipur-player self${turn===my?' active-turn':''}" data-player="${my}">`;
  h += `<div class="jp-header"><span class="jp-name">你${turn===my?' ▶ 你的回合':''}</span>`;
  h += `<span class="jp-stat">🪙 ${st['rupees_'+my]||0}</span>`;
  h += `<span class="jp-stat camel">🐫 ${st['camel_count_'+my]||0}</span>`;
  const mbc = st['bonus_count_'+my]||[0,0,0];
  const mbt = st['bonus_total_'+my]||0;
  h += `<span class="jp-stat bonus" title="你的连击 token 总值">🎁${mbc[0]}·${mbc[1]}·${mbc[2]} (=${mbt})</span></div>`;
  h += `<div class="jaipur-hand">`;
  for (let g=0; g<6; ++g) {
    const n = myHand[g];
    const giving = (ep===3 && gv[g]>0);
    const handAct = A_EXCH_HAND + g;
    const clickable = (ep===3 && has(handAct));
    for (let i=0;i<n;i++) {
      const cls = `jp-card ${GOOD_CSS[g]}${giving?' giving':''}${clickable?' clickable':''}`;
      const attr = clickable ? `data-act="${handAct}"` : '';
      h += `<div class="${cls}" ${attr} data-g="${g}">${GOOD_ICON[g]}</div>`;
    }
  }
  // empty placeholders to keep layout
  const myTot = myHand.reduce((a,b)=>a+b,0);
  for (let i=myTot;i<7;i++) h += `<div class="jp-card empty-slot"></div>`;
  h += `</div></div>`;

  // --- Actions ---
  if (ctx.canPlay) {
    h += `<div class="jaipur-actions">`;
    if (ep === 0) {
      // Sell buttons (sell ALL of a type — needs an explicit button).
      // Taking goods/camels is done by CLICKING the market card directly
      // (no per-type take buttons).
      const thresh = [2,2,2,1,1,1];
      for (let g=0; g<6; ++g)
        if (myHand[g] >= thresh[g] && has(A_SELL+g))
          h += `<button class="act sell ${GOOD_CSS[g]}" data-act="${A_SELL+g}">卖 ${GOOD_ICON[g]} ×${myHand[g]}</button>`;
      if (has(A_ENTER)) h += `<button class="act" data-act="${A_ENTER}">交换</button>`;
    } else if (ep === 1) {
      h += `<button class="act primary" id="btn-mkt-done" data-act="${A_EXCH_MKT_DONE}" ${has(A_EXCH_MKT_DONE)?'':'disabled'}>确认市场选择</button>`;
      h += `<button class="act danger" data-cancel-exch="1">取消</button>`;
    } else if (ep === 2) {
      h += `<span class="camel-display">🐫 ${ccam} <span class="range">[${minCam}..${maxCam}]</span></span>`;
      for (let k=minCam; k<=maxCam; ++k)
        h += `<button class="act${k===ccam?' primary':''}" data-act="${A_EXCH_CAMEL+k}">${k}</button>`;
      h += `<button class="act danger" data-cancel-exch="1">取消</button>`;
    } else if (ep === 3) {
      h += `<button class="act danger" data-cancel-exch="1">取消交换</button>`;
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
  bindEvents(ctx);
}

function bindEvents(ctx) {
  document.querySelectorAll('[data-act]').forEach(el => {
    const a = parseInt(el.dataset.act);
    if (!isNaN(a) && a >= 0) {
      el.onclick = () => ctx.submitAction(a);
    }
  });
  // "Cancel exchange" is a CLIENT-SIDE undo, not a game action. Loop
  // step-back until exch_phase returns to idle (rolls back the player's
  // own sub-actions one by one). The rules layer has no cancel action —
  // adding one would create an infinite enter↔cancel MCTS loop and
  // selfplay would never terminate. See docs/plans/jaipur_ai_encoding.md
  // and rules.h's note next to kActExchHandBase.
  document.querySelectorAll('[data-cancel-exch]').forEach(el => {
    el.onclick = async () => {
      const sid = ctx.state && ctx.state.sessionId;
      if (!sid) return;
      // Step back repeatedly; stop when phase is idle (or after a hard cap
      // to avoid runaway in case of a server-side hiccup).
      for (let i = 0; i < 10; i++) {
        const data = await apiPost(API_BASE + '/' + sid + '/step-back', {});
        if (!data || !data.state) break;
        if ((data.state.exch_phase || 0) === 0) {
          // Refresh client state via the ctx render path.
          ctx.state.gameState = data;
          ctx.rerender();
          return;
        }
        ctx.state.gameState = data;
      }
      ctx.rerender();
    };
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

  // Reveal the opponent's face-down hand cards as a list of good types, so the
  // player sees what is being sold/given BEFORE it flies away. Flips the first
  // matching face-down cards in the opponent's hand to show each good icon.
  // Returns the selectors of the revealed cards (in order) for the fly source.
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
      const g = ai.good_type, cid = g + 1;
      const pileSel = `[data-token="${g}"]`;
      const nSold = (prev[`hand_count_${actor}`] || [])[g] || 1;
      let fromSrcs;
      if (handFaceDown) {
        // Opponent hand is face-down: flip the sold cards first, pause, fly.
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

  // Exchange swap. The pipeline collapses the AI's whole turn (multiple
  // exchange sub-actions) into ONE describeTransition call: prevState is the
  // pre-turn state (phase idle) and ai.type is only the LAST sub-action — so
  // we can't key on exch_phase. Instead detect a swap directly from the
  // market composition delta: an exchange both REMOVES goods from the market
  // (taken into hand) and ADDS goods/camels (given from hand). For a plain
  // take/sell the loop above already handled it (and won't be a 2-way swap).
  const pm = prev.market_count || [0,0,0,0,0,0,0];
  const nm = next.market_count || [0,0,0,0,0,0,0];
  {
    const takenGoods = [], givenGoods = [];
    for (let g=0; g<6; ++g) {
      const d = nm[g] - pm[g];
      for (let i=0; i<-d; ++i) takenGoods.push(g);   // left market → into hand
      for (let i=0; i<d; ++i) givenGoods.push(g);    // entered market → from hand
    }
    const givenCamels = Math.max(0, (nm[6]||0) - (pm[6]||0));
    // It's an exchange only if goods were BOTH taken and given (goods or camels).
    const isSwap = takenGoods.length > 0 && (givenGoods.length > 0 || givenCamels > 0)
                   && ai.type !== 'take' && ai.type !== 'sell' && ai.type !== 'take_camels';
    if (isSwap && !newState.is_terminal) {

    // Reveal opponent's surrendered goods first (pause), then swap.
    let giveSrcs = [];
    if (handFaceDown && givenGoods.length) {
      giveSrcs = revealOppCards(givenGoods);
      steps.push({ type: 'pause', duration: REVEAL_PAUSE });
    } else if (givenGoods.length) {
      // face-up hand: locate a card per given good type
      for (const g of givenGoods) {
        const c = faceUpCards(g)[0];
        if (c) { c.classList.add('flew'); giveSrcs.push(c); }
      }
    }
    const flights = [];
    // market → hand (taken goods)
    for (const g of takenGoods)
      flights.push({ from: mktSel(g), to: handSel,
        createElement: () => mkCard(g+1), duration: FLY, width: CARD_W, height: CARD_H,
        onStart: emptyMarketSlot });
    // hand → market (given goods), simultaneously
    givenGoods.forEach((g, i) => {
      const src = giveSrcs[i] || handSel;
      flights.push({ from: src, to: `.jaipur-market`,
        createElement: () => mkCard(g+1), duration: FLY, width: CARD_W, height: CARD_H,
        onStart: (el) => { if (el && el.style) el.style.opacity = '0.3'; } });
    });
    // camels → market
    for (let i=0; i<givenCamels; ++i)
      flights.push({ from: `[data-player="${actor}"] .jp-stat.camel`, to: `.jaipur-market`,
        createElement: () => mkCard(0), duration: FLY, width: CARD_W, height: CARD_H });
    if (flights.length) steps.push({ type: 'flyGroup', flights });
    }  // end if (isSwap)
  }  // end market-delta block
  return steps;
}

function formatOpponentMove(ai, id, actor) {
  if (!ai || !ai.type) return `动作 ${id}`;
  const GN = GOOD_ICON.map((icon, g) => icon + GOOD_LABEL[g]);
  switch (ai.type) {
    case 'sell': return `出售 ${GN[ai.good_type]}`;
    case 'take': return `拿取 ${GN[ai.good_type]}`;
    case 'take_camels': return '取走所有骆驼';
    case 'enter_exchange': return '进入交换';
    case 'exch_mkt_done': return '确认市场选择';
    case 'exch_camel': return `给 ${ai.camels} 骆驼`;
    case 'exch_cancel': return '取消交换';
    case 'exch_take': return `选市场 ${GN[ai.good_type]}`;
    case 'exch_give': return `给手牌 ${GN[ai.good_type]}`;
    default: return '交换中…';
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
  gameIntro: '收集商品与骆驼，择机卖出！钻石/黄金/白银需 ≥2 张才能卖；交换分步：选市场牌 → 选骆驼数 → 选手牌。',
});
