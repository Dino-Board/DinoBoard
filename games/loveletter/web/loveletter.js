import { createApp } from '/static/general/app.js';
import { t, register } from '/static/general/i18n.js';

register({
  zh: {
    'll.title': '情书',
    'll.intro': '点你的手牌或抽到的牌出牌；卫兵/王子/国王等需要先选目标对手再确认',
    'll.card_1': '侍卫',
    'll.card_2': '牧师',
    'll.card_3': '男爵',
    'll.card_4': '侍女',
    'll.card_5': '王子',
    'll.card_6': '国王',
    'll.card_7': '伯爵夫人',
    'll.card_8': '公主',
    'll.effect_1': '猜对手手牌',
    'll.effect_2': '偷看对手手牌',
    'll.effect_3': '比较手牌大小',
    'll.effect_4': '保护自己一轮',
    'll.effect_5': '迫使弃牌重摸',
    'll.effect_6': '交换手牌',
    'll.effect_7': '另一张为王子或国王时必须弃置',
    'll.effect_8': '弃出即淘汰',
    'll.deck_remaining': '牌堆剩余: {n}',
    'll.face_up_removed': ' | 公开移除: {list}',
    'll.face_up_entry': '{name}({id})',
    'll.player_n': '玩家{n}',
    'll.eliminated': '淘汰',
    'll.protected_status': '保护中',
    'll.in_turn': '行动中',
    'll.you_eliminated': '你已被淘汰',
    'll.guess_modal_title': '猜测玩家{n} 的手牌',
    'll.no_legal_guess': '没有可猜的牌',
    'll.change_target': '换一个目标',
    'll.action_unknown': '动作 #{id}',
    'll.action_start': '开局',
    'll.card_invalid': '{card}（无效）',
    'll.guard_text': '{card} → 玩家{target} 猜 {guess}',
    'll.priest_text': '{card} → 偷看玩家{target}',
    'll.baron_text': '{card} → 比较玩家{target}',
    'll.handmaid_text': '{card}（保护）',
    'll.prince_text': '{card} → 玩家{target} 弃牌',
    'll.king_text': '{card} → 与玩家{target} 交换',
    'll.countess_text': '{card}（弃出）',
    'll.princess_text': '{card}（弃出即淘汰）',
    'll.target_hint_prince': '请点击目标玩家（可点击你自己）',
    'll.target_hint_self_invalid': '所有对手被保护，点击自己弃出无效',
    'll.target_hint_default': '请点击目标玩家',
    'll.guess_hint_target': '猜测玩家{n} 的手牌',
    'll.priest_modal_title': '牧师 · 偷看手牌',
    'll.baron_modal_title': '男爵 · 比较手牌',
    'll.peek_summary': '只有你看到了这张牌。',
    'll.player_hand_label': '玩家{n}的手牌',
    'll.player_hand_label_sp': '玩家{n} 的手牌',
    'll.baron_tie': '平局，双方都不淘汰',
    'll.baron_human_actor_won': '你的牌更大，玩家{target} 被淘汰',
    'll.baron_human_actor_lost': '玩家{target} 的牌更大，你被淘汰',
    'll.baron_human_target_lost': '玩家{actor} 的牌更大，你被淘汰',
    'll.baron_human_target_won': '你的牌更大，玩家{actor} 被淘汰',
    'll.showdown_prefix': '终局底牌 ',
    'll.showdown_entry': '玩家{n}：{card}',
    'll.player_symbol': '玩家{n}',
  },
  en: {
    'll.title': 'Love Letter',
    'll.intro': 'Click your hand or drawn card to play; Guard/Prince/King etc. need a target opponent first.',
    'll.card_1': 'Guard',
    'll.card_2': 'Priest',
    'll.card_3': 'Baron',
    'll.card_4': 'Handmaid',
    'll.card_5': 'Prince',
    'll.card_6': 'King',
    'll.card_7': 'Countess',
    'll.card_8': 'Princess',
    'll.effect_1': "Guess opponent's hand",
    'll.effect_2': "Peek at opponent's hand",
    'll.effect_3': 'Compare hands',
    'll.effect_4': 'Protect self one round',
    'll.effect_5': 'Force discard and redraw',
    'll.effect_6': 'Swap hands',
    'll.effect_7': 'Must discard if other card is Prince or King',
    'll.effect_8': 'Eliminated if discarded',
    'll.deck_remaining': 'Deck: {n}',
    'll.face_up_removed': ' | Removed face-up: {list}',
    'll.face_up_entry': '{name}({id})',
    'll.player_n': 'Player {n}',
    'll.eliminated': 'Out',
    'll.protected_status': 'Protected',
    'll.in_turn': 'Acting',
    'll.you_eliminated': 'You are eliminated',
    'll.guess_modal_title': "Guess player {n}'s hand",
    'll.no_legal_guess': 'No legal guesses',
    'll.change_target': 'Choose another target',
    'll.action_unknown': 'Action #{id}',
    'll.action_start': 'Game start',
    'll.card_invalid': '{card} (invalid)',
    'll.guard_text': '{card} → player {target} guess {guess}',
    'll.priest_text': '{card} → peek player {target}',
    'll.baron_text': '{card} → compare with player {target}',
    'll.handmaid_text': '{card} (protected)',
    'll.prince_text': '{card} → player {target} discards',
    'll.king_text': '{card} → swap with player {target}',
    'll.countess_text': '{card} (discarded)',
    'll.princess_text': '{card} (discarded — eliminated)',
    'll.target_hint_prince': 'Click a target player (you may click yourself)',
    'll.target_hint_self_invalid': 'All opponents protected — click yourself to discard with no effect',
    'll.target_hint_default': 'Click a target player',
    'll.guess_hint_target': "Guess player {n}'s hand",
    'll.priest_modal_title': 'Priest · Peek',
    'll.baron_modal_title': 'Baron · Compare',
    'll.peek_summary': 'Only you saw this card.',
    'll.player_hand_label': "Player {n}'s hand",
    'll.player_hand_label_sp': "Player {n}'s hand",
    'll.baron_tie': 'Tie — neither is eliminated',
    'll.baron_human_actor_won': 'Your card is higher; player {target} is eliminated',
    'll.baron_human_actor_lost': "Player {target}'s card is higher; you are eliminated",
    'll.baron_human_target_lost': "Player {actor}'s card is higher; you are eliminated",
    'll.baron_human_target_won': 'Your card is higher; player {actor} is eliminated',
    'll.showdown_prefix': 'Final hands ',
    'll.showdown_entry': 'Player {n}: {card}',
    'll.player_symbol': 'P{n}',
  },
});

const CARD_VALUES = ['', '1', '2', '3', '4', '5', '6', '7', '8'];

function cardLabel(c) {
  if (c <= 0 || c > 8) return '';
  return t('ll.card_' + c);
}

function cardEffect(c) {
  if (c <= 0 || c > 8) return '';
  return t('ll.effect_' + c);
}

function playerLabel(n) {
  return t('ll.player_n', { n });
}

const GUARD_OFF = 0, PRIEST_OFF = 28, BARON_OFF = 32;
const HANDMAID_ACT = 36, PRINCE_OFF = 37, KING_OFF = 41;
const COUNTESS_ACT = 45, PRINCESS_ACT = 46;

let pendingCard = 0;
let pendingTarget = -1;
let currentCtx = null;

// State-side LL stores discards as a multiset (discard_count[player][card])
// for engine reasons (permutation invariance in the network features), so the
// state dict's `p.discards` comes back sorted by card value. Players read
// "what was played in what order" as table state, though, so we maintain a
// frontend-only ordered log per player. Round resets and replay jumps may
// temporarily desync; we recover by rebuilding from truth whenever the
// multiset doesn't match. This is a UI memory only — no engine impact.
const discardOrderByPlayer = new Map(); // player_idx -> [card, card, ...]

function multisetEqual(a, b) {
  if (a.length !== b.length) return false;
  const counts = new Array(9).fill(0);
  for (const c of a) counts[c]++;
  for (const c of b) counts[c]--;
  for (const v of counts) if (v !== 0) return false;
  return true;
}

// Returns the per-player discard list reordered to play-order. Two paths:
//   (a) Replay precompute attached `state.__llOrderByPlayer` on every frame
//       (see precomputeReplayDiscardOrder) — used unconditionally if present.
//   (b) Live play: walk the cached log; if the cache's multiset matches
//       truth use it, otherwise fall back to truth (sorted) and reset the
//       cache so the next describeTransition anchors from there.
function orderedDiscardsFor(playerIdx, truthDiscards, state) {
  const truth = truthDiscards || [];
  if (state && state.__llOrderByPlayer && Array.isArray(state.__llOrderByPlayer[playerIdx])) {
    return state.__llOrderByPlayer[playerIdx].slice();
  }
  const cached = discardOrderByPlayer.get(playerIdx);
  if (cached && multisetEqual(cached, truth)) {
    return cached.slice();
  }
  discardOrderByPlayer.set(playerIdx, truth.slice());
  return truth.slice();
}

function recordDiscard(playerIdx, cardValue) {
  if (cardValue <= 0) return;
  const cur = discardOrderByPlayer.get(playerIdx) || [];
  cur.push(cardValue);
  discardOrderByPlayer.set(playerIdx, cur);
}

function resetDiscardOrder() {
  discardOrderByPlayer.clear();
}

// Walk the replay frames forward and stamp each frame's state with
// `__llOrderByPlayer: number[][]` (per-player ordered discard list at that
// point in time). Same diff logic as describeTransition: for each ply,
// figure out per-player added cards and append in actor-first order, with
// the actor's played card landing before any forced (Prince) discards.
// Reset any player's ordered list when their multiset shrinks (round
// boundary — discards reset to empty in state).
function precomputeReplayDiscardOrder(frames) {
  if (!frames || !frames.length) return;
  // Idempotent: onReplayFrames fires every render. Once stamped, skip.
  const last = frames[frames.length - 1];
  if (last && last.state && last.state.__llOrderByPlayer) return;
  const orders = new Map(); // player_idx -> [card,...]
  let prevPlayers = null;

  function multisetCounts(arr) {
    const c = new Array(9).fill(0);
    for (const v of arr) c[v]++;
    return c;
  }

  for (const frame of frames) {
    const st = frame && frame.state;
    if (!st || !Array.isArray(st.players)) continue;
    const players = st.players;

    // Per-player diff vs prev frame.
    for (let p = 0; p < players.length; p++) {
      const cur = (players[p].discards || []).slice();
      const before = prevPlayers && prevPlayers[p] ? (prevPlayers[p].discards || []) : [];
      const beforeC = multisetCounts(before);
      const curC = multisetCounts(cur);
      // Round reset detection: if cur multiset is NOT a superset of before,
      // the round restarted (or this is a backward step). Adopt truth and
      // re-anchor the ordered list to it (best-effort: as multiset).
      let isSuperset = true;
      for (let c = 1; c <= 8; c++) {
        if (curC[c] < beforeC[c]) { isSuperset = false; break; }
      }
      if (!isSuperset) {
        orders.set(p, cur.slice().sort((a, b) => a - b));
        continue;
      }
      // Append the diff cards. We don't know intra-frame order between
      // players precisely, but actor-first + played-card-first is right
      // for the common case (Prince forced discard).
      const addCounts = new Array(9).fill(0);
      for (let c = 1; c <= 8; c++) addCounts[c] = curC[c] - beforeC[c];
      const ordered = orders.get(p) ? orders.get(p).slice() : [];
      // For the actor: their action card lands first.
      const actor = parseActorIdx(frame.actor);
      const playedCard = frame.action_info && frame.action_info.card;
      if (p === actor && playedCard && addCounts[playedCard] > 0) {
        ordered.push(playedCard);
        addCounts[playedCard]--;
      }
      for (let c = 1; c <= 8; c++) {
        while (addCounts[c] > 0) { ordered.push(c); addCounts[c]--; }
      }
      orders.set(p, ordered);
    }

    // Stamp this frame's state with a copy of the per-player order.
    const stamp = [];
    for (let p = 0; p < players.length; p++) {
      stamp[p] = (orders.get(p) || []).slice();
    }
    st.__llOrderByPlayer = stamp;
    prevPlayers = players;
  }
}

function parseActorIdx(actor) {
  if (typeof actor !== 'string') return -1;
  if (!actor.startsWith('player_')) return -1;
  const n = parseInt(actor.slice('player_'.length), 10);
  return Number.isFinite(n) ? n : -1;
}

function getLegalSet(gs) {
  if (!gs) return new Set();
  return new Set(gs.legal_actions || []);
}

function cardOfAction(aid) {
  if (aid >= GUARD_OFF && aid < PRIEST_OFF) return 1;
  if (aid >= PRIEST_OFF && aid < BARON_OFF) return 2;
  if (aid >= BARON_OFF && aid < HANDMAID_ACT) return 3;
  if (aid === HANDMAID_ACT) return 4;
  if (aid >= PRINCE_OFF && aid < KING_OFF) return 5;
  if (aid >= KING_OFF && aid < COUNTESS_ACT) return 6;
  if (aid === COUNTESS_ACT) return 7;
  if (aid === PRINCESS_ACT) return 8;
  return 0;
}

function canPlayCard(card, legalSet) {
  for (const aid of legalSet) {
    if (cardOfAction(aid) === card) return true;
  }
  return false;
}

function needsTarget(card) {
  return card === 1 || card === 2 || card === 3 || card === 5 || card === 6;
}

function needsGuess(card) {
  return card === 1;
}

function getTargetsForCard(card, legalSet) {
  const targets = new Set();
  for (const aid of legalSet) {
    if (cardOfAction(aid) !== card) continue;
    let tg = -1;
    if (card === 1) tg = Math.floor((aid - GUARD_OFF) / 7);
    else if (card === 2) tg = aid - PRIEST_OFF;
    else if (card === 3) tg = aid - BARON_OFF;
    else if (card === 5) tg = aid - PRINCE_OFF;
    else if (card === 6) tg = aid - KING_OFF;
    if (tg >= 0) targets.add(tg);
  }
  return targets;
}

function resolveAction(card, target, guess) {
  switch (card) {
    case 1: return GUARD_OFF + target * 7 + (guess - 2);
    case 2: return PRIEST_OFF + target;
    case 3: return BARON_OFF + target;
    case 4: return HANDMAID_ACT;
    case 5: return PRINCE_OFF + target;
    case 6: return KING_OFF + target;
    case 7: return COUNTESS_ACT;
    case 8: return PRINCESS_ACT;
    default: return -1;
  }
}

function submitNoTargetCard(card, legalSet) {
  const aid = resolveAction(card, -1, -1);
  if (legalSet.has(aid)) {
    resetPending();
    currentCtx.submitAction(aid);
  }
}

function resetPending() {
  pendingCard = 0;
  pendingTarget = -1;
}

function renderBoard(container, gs, ctx) {
  currentCtx = ctx;
  container.innerHTML = '';
  if (!gs || !gs.state) return;

  const st = gs.state;
  const legalSet = getLegalSet(gs);
  const playing = ctx.canPlay;
  const humanPlayer = ctx.state.humanPlayer;
  const numPlayers = st.num_players;
  const currentPlayer = st.current_player;
  const isTerminal = !!gs.is_terminal;

  const board = document.createElement('div');
  board.className = 'll-board';

  const deckInfo = document.createElement('div');
  deckInfo.className = 'll-deck-info';
  deckInfo.setAttribute('data-deck', '');
  deckInfo.textContent = t('ll.deck_remaining', { n: st.deck_size });
  if (st.face_up_removed && st.face_up_removed.length > 0) {
    const removed = st.face_up_removed
      .map(c => t('ll.face_up_entry', { name: cardLabel(c), id: c }))
      .join(', ');
    deckInfo.textContent += t('ll.face_up_removed', { list: removed });
  }
  board.appendChild(deckInfo);

  const opponents = document.createElement('div');
  opponents.className = 'll-opponents';
  for (let pi = 0; pi < numPlayers; pi++) {
    if (pi === humanPlayer) continue;
    const p = st.players[pi];
    const opp = document.createElement('div');
    opp.className = 'll-opponent';
    opp.setAttribute('data-opponent', String(pi));
    if (!p.alive) opp.classList.add('eliminated');
    if (p.protected) opp.classList.add('protected');
    if (pi === currentPlayer) opp.classList.add('current-turn');

    const canTarget = playing && pendingCard > 0 && needsTarget(pendingCard);
    const validTargets = pendingCard > 0 ? getTargetsForCard(pendingCard, legalSet) : new Set();
    const isValidTarget = canTarget && validTargets.has(pi);

    if (isValidTarget) {
      opp.classList.add('selectable');
      opp.addEventListener('click', () => {
        if (needsGuess(pendingCard)) {
          // Guard: target chosen, now show the guess modal (stage 2).
          // Don't submit yet.
          pendingTarget = pi;
          ctx.rerender();
          return;
        }
        const aid = resolveAction(pendingCard, pi, -1);
        if (legalSet.has(aid)) {
          resetPending();
          ctx.submitAction(aid);
        }
      });
    }

    if (pendingCard > 0 && pendingTarget === pi) {
      opp.classList.add('selected-target');
    }

    const nameEl = document.createElement('div');
    nameEl.className = 'll-opp-name';
    nameEl.textContent = playerLabel(pi);
    opp.appendChild(nameEl);

    const statusEl = document.createElement('div');
    statusEl.className = 'll-opp-status';
    if (!p.alive) {
      statusEl.classList.add('status-eliminated');
      statusEl.textContent = t('ll.eliminated');
    } else if (p.protected) {
      statusEl.classList.add('status-protected');
      statusEl.textContent = t('ll.protected_status');
    } else if (pi === currentPlayer) {
      statusEl.classList.add('status-turn');
      statusEl.textContent = t('ll.in_turn');
    } else {
      statusEl.textContent = ' ';
    }
    opp.appendChild(statusEl);

    // Replay-only "reveal hidden info" mode: surface the opponent's
    // current hand card. The replay's state dict carries truth (rebuilt
    // from action_history through a single GameSession that holds GT),
    // so p.hand is correct here. Live play never reaches this branch
    // because ctx.revealHidden is gated on state.replayMode.
    if (p.alive && !isTerminal) {
      // The hand row is always rendered for alive opponents so the
      // opponent box height stays anchored. In replay reveal-mode we
      // surface the actual cards; in live play we render dashed
      // placeholders sized by what's publicly known: 1 card normally,
      // 2 cards for the current acting player mid-turn (post-draw).
      const oppHand = document.createElement('div');
      oppHand.className = 'll-opp-revealed-hand';
      if (ctx.revealHidden) {
        const cards = [];
        if (p.hand > 0) cards.push(p.hand);
        if (pi === currentPlayer && st.drawn_card > 0) cards.push(st.drawn_card);
        for (const c of cards) {
          oppHand.appendChild(buildRevealedCard(c, pi));
        }
      } else {
        const slotCount = (pi === currentPlayer) ? 2 : 1;
        for (let s = 0; s < slotCount; s++) {
          const ph = document.createElement('div');
          ph.className = 'll-discard-card ll-hand-placeholder';
          oppHand.appendChild(ph);
        }
      }
      opp.appendChild(oppHand);
    }

    // End-of-round showdown: append the alive player's last hand card to
    // their visible discard pile so the table is fully revealed. Engine
    // leaves the hand intact for tie-break sum calculations, so we tack
    // it on the JS side without mutating game state.
    const renderedDiscards = orderedDiscardsFor(pi, p.discards || [], st);
    if (isTerminal && p.alive && p.hand > 0) renderedDiscards.push(p.hand);
    const disc = buildDiscardPile(renderedDiscards, pi);
    opp.appendChild(disc);

    opponents.appendChild(opp);
  }
  board.appendChild(opponents);

  // Guard two-stage interaction (OB-006):
  //   stage 1: player clicks a hand Guard → pendingCard=1, pendingTarget=-1.
  //            opponent areas become selectable (same affordance as Priest/Baron/King).
  //   stage 2: player clicks an opponent → pendingTarget=pi, then this modal
  //            shows up centered with one button per legal guess card.
  // Avoids the previous "target × guess" cartesian button grid which was
  // unreadable in 3p/4p. (Self-fallback when all opps protected is handled
  // earlier in the hand-card click handler — never reaches this UI.)
  if (playing && pendingCard === 1 && pendingTarget >= 0 && pendingTarget !== humanPlayer) {
    const overlay = document.createElement('div');
    overlay.className = 'll-guess-modal-overlay';
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) {
        // Click on backdrop cancels target selection but keeps the card pending.
        pendingTarget = -1;
        ctx.rerender();
      }
    });

    const modal = document.createElement('div');
    modal.className = 'll-guess-modal';

    const title = document.createElement('div');
    title.className = 'll-guess-modal-title';
    title.textContent = t('ll.guess_modal_title', { n: pendingTarget });
    modal.appendChild(title);

    const grid = document.createElement('div');
    grid.className = 'll-guess-grid';
    let buttonsAdded = 0;
    for (let g = 2; g <= 8; g++) {
      const aid = GUARD_OFF + pendingTarget * 7 + (g - 2);
      if (!legalSet.has(aid)) continue;
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'll-guess-btn ll-guess-btn-card card-' + g;
      btn.innerHTML =
        '<div class="ll-guess-btn-value">' + g + '</div>' +
        '<div class="ll-guess-btn-name">' + cardLabel(g) + '</div>';
      btn.addEventListener('click', () => {
        // Close the modal BEFORE submitting: submitAction kicks off
        // describeTransition which pops a bubble over the actor, and an
        // open guess-modal overlay would sit on top and hide it.
        resetPending();
        ctx.rerender();
        ctx.submitAction(aid);
      });
      grid.appendChild(btn);
      buttonsAdded++;
    }
    modal.appendChild(grid);

    if (buttonsAdded === 0) {
      // Defensive: target was selectable but no legal guess survived.
      // Shouldn't happen — clear pending and let player retry.
      const empty = document.createElement('div');
      empty.className = 'll-guess-empty';
      empty.textContent = t('ll.no_legal_guess');
      modal.appendChild(empty);
    }

    const actions = document.createElement('div');
    actions.className = 'll-guess-modal-actions';
    const cancel = document.createElement('button');
    cancel.type = 'button';
    cancel.className = 'll-guess-cancel';
    cancel.textContent = t('ll.change_target');
    cancel.addEventListener('click', () => {
      pendingTarget = -1;
      ctx.rerender();
    });
    actions.appendChild(cancel);
    modal.appendChild(actions);

    overlay.appendChild(modal);
    board.appendChild(overlay);
  }

  container.appendChild(board);
}

function buildDiscardPile(discards, playerIdx) {
  // Card-chip style pile: each discarded card shows up as a small card
  // rather than a text chip, so plays visually land as physical cards.
  // An empty trailing anchor (.incoming-slot) gives describeTransition
  // a stable target for fly-to-discard animations before the data
  // re-render adds the new card chip.
  const wrap = document.createElement('div');
  wrap.className = 'll-discard-pile';
  wrap.setAttribute('data-discard-pile', String(playerIdx));
  for (let i = 0; i < discards.length; i++) {
    const c = discards[i];
    const card = document.createElement('div');
    card.className = 'll-discard-card card-' + c;
    card.setAttribute('data-discard-slot', playerIdx + '-' + i);
    const v = document.createElement('div');
    v.className = 'll-discard-card-value';
    v.textContent = CARD_VALUES[c];
    card.appendChild(v);
    const n = document.createElement('div');
    n.className = 'll-discard-card-name';
    n.textContent = cardLabel(c);
    card.appendChild(n);
    wrap.appendChild(card);
  }
  const incoming = document.createElement('div');
  incoming.className = 'll-discard-card incoming-slot';
  incoming.setAttribute('data-discard-incoming', String(playerIdx));
  wrap.appendChild(incoming);
  return wrap;
}

function renderPlayerArea(container, gs, ctx) {
  container.innerHTML = '';
  if (!gs || !gs.state) return;

  const st = gs.state;
  const legalSet = getLegalSet(gs);
  const playing = ctx.canPlay;
  const humanPlayer = ctx.state.humanPlayer;
  const currentPlayer = st.current_player;
  const pd = st.players[humanPlayer];

  const area = document.createElement('div');
  area.className = 'll-player-area';
  area.setAttribute('data-player', String(humanPlayer));

  // Self-target highlight: Prince is the ONLY card that legitimately
  // targets self (forces self to discard and redraw). Priest/Baron/King
  // emit a self-fallback action (target=me) only when every opponent is
  // Handmaid-protected, and the engine treats it as a no-op discard —
  // peeking own hand, comparing self vs self, swapping with self all
  // yield zero info. So we don't let the player click their own area for
  // those cards; instead the hand card auto-resolves the fallback like
  // Guard's "弃出无效" path. Prince keeps the self-click affordance.
  const validTargets = pendingCard > 0 ? getTargetsForCard(pendingCard, legalSet) : new Set();
  const selfIsValidTarget = (
    playing && pendingCard === 5 /* Prince */
    && validTargets.has(humanPlayer)
  );
  if (selfIsValidTarget) {
    area.classList.add('selectable-self');
    area.addEventListener('click', () => {
      const aid = resolveAction(pendingCard, humanPlayer, -1);
      if (legalSet.has(aid)) {
        resetPending();
        ctx.submitAction(aid);
      }
    });
  }

  const isTerminal = !!gs.is_terminal;
  // Showdown: if the round ended and human is still alive, append their
  // last hand card to the discard pile so the player sees their own
  // table revealed alongside the opponents'.
  const myDiscards = orderedDiscardsFor(humanPlayer, pd.discards || [], st);
  if (isTerminal && pd.alive && pd.hand > 0) myDiscards.push(pd.hand);
  const disc = buildDiscardPile(myDiscards, humanPlayer);
  disc.classList.add('mine');
  area.appendChild(disc);

  if (!pd.alive) {
    const dead = document.createElement('div');
    dead.className = 'll-dead-msg';
    dead.textContent = t('ll.you_eliminated');
    area.appendChild(dead);
    // Reserve the hand row so the player area's height stays anchored —
    // .ll-hand has min-height: 180px which we'd otherwise lose at terminal
    // / eliminated states, causing the whole layout to jump.
    const handPlaceholder = document.createElement('div');
    handPlaceholder.className = 'll-hand';
    area.appendChild(handPlaceholder);
    container.appendChild(area);
    return;
  }

  // At terminal: the human's last card has been moved to the discard pile
  // visually. Don't also render it as a hand card (that would duplicate it).
  if (isTerminal) {
    const handPlaceholder = document.createElement('div');
    handPlaceholder.className = 'll-hand';
    area.appendChild(handPlaceholder);
    container.appendChild(area);
    return;
  }

  const hand = document.createElement('div');
  hand.className = 'll-hand';

  const handCard = pd.hand;
  const drawnCard = (currentPlayer === humanPlayer) ? st.drawn_card : 0;
  const cards = drawnCard > 0 ? [handCard, drawnCard] : [handCard];

  for (const c of cards) {
    if (c <= 0) continue;
    const cardEl = createCardElement(c, playing, legalSet);
    hand.appendChild(cardEl);
  }

  area.appendChild(hand);

  if (pendingCard > 0) {
    const hint = document.createElement('div');
    hint.className = 'll-pending-hint';
    if (needsGuess(pendingCard)) {
      hint.textContent = pendingTarget >= 0
        ? t('ll.guess_hint_target', { n: pendingTarget })
        : t('ll.target_hint_default');
    } else if (needsTarget(pendingCard)) {
      hint.textContent = selfIsValidTarget && pendingCard === 5
        ? t('ll.target_hint_prince')
        : selfIsValidTarget
          ? t('ll.target_hint_self_invalid')
          : t('ll.target_hint_default');
    }
    area.appendChild(hint);
  }

  container.appendChild(area);
}

// Non-interactive card chip used to surface an opponent's hand card in
// replay-mode reveal-hidden. Same visual language as the discard pile
// chips (.ll-discard-card) so it fits in the opponent's fixed slot
// without competing visually with the player's own hand.
function buildRevealedCard(cardValue, ownerIdx) {
  const el = document.createElement('div');
  el.className = 'll-discard-card ll-revealed-hand-card card-' + cardValue;
  el.setAttribute('data-revealed-hand', String(ownerIdx));

  const v = document.createElement('div');
  v.className = 'll-discard-card-value';
  v.textContent = CARD_VALUES[cardValue];
  el.appendChild(v);

  const n = document.createElement('div');
  n.className = 'll-discard-card-name';
  n.textContent = cardLabel(cardValue);
  el.appendChild(n);
  return el;
}

function createCardElement(cardValue, playing, legalSet) {
  const el = document.createElement('button');
  el.type = 'button';
  el.className = 'll-card card-' + cardValue;
  el.setAttribute('data-hand-card', String(cardValue));
  const playable = playing && canPlayCard(cardValue, legalSet);

  const val = document.createElement('div');
  val.className = 'll-card-value';
  val.textContent = CARD_VALUES[cardValue];
  el.appendChild(val);

  const name = document.createElement('div');
  name.className = 'll-card-name';
  name.textContent = cardLabel(cardValue);
  el.appendChild(name);

  const effect = document.createElement('div');
  effect.className = 'll-card-effect';
  effect.textContent = cardEffect(cardValue);
  el.appendChild(effect);

  if (pendingCard === cardValue) {
    el.classList.add('selected');
  }

  if (playable) {
    el.classList.add('playable');
    el.addEventListener('click', () => {
      if (pendingCard === cardValue) {
        resetPending();
        currentCtx.rerender();
        return;
      }
      pendingCard = cardValue;
      pendingTarget = -1;

      if (!needsTarget(cardValue)) {
        submitNoTargetCard(cardValue, legalSet);
      } else if (cardValue !== 5 /* not Prince */) {
        // Guard/Priest/Baron/King: if the only legal action is the
        // self-fallback (every opponent Handmaid-protected), auto-submit
        // as a no-op discard. Forcing the player through a target/guess
        // panel that has no real choices is meaningless and confusing.
        // Prince keeps the self-click affordance because targeting self
        // is a legitimate Prince play, not a forced no-op.
        const tgts = getTargetsForCard(cardValue, legalSet);
        const humanPlayer = currentCtx.state.humanPlayer;
        const hasOppTarget = [...tgts].some(tg => tg !== humanPlayer);
        if (!hasOppTarget && tgts.has(humanPlayer)) {
          // Guard's self-fallback uses guess=2 (lowest legal guess);
          // Priest/Baron/King ignore guess (-1 sentinel passes through
          // resolveAction without offset). Both produce the engine's
          // single self-target action id when every opp is protected.
          const guess = cardValue === 1 ? 2 : -1;
          const aid = resolveAction(cardValue, humanPlayer, guess);
          if (legalSet.has(aid)) {
            resetPending();
            currentCtx.submitAction(aid);
            return;
          }
        }
        currentCtx.rerender();
      } else {
        currentCtx.rerender();
      }
    });
  } else {
    el.disabled = true;
  }

  return el;
}

function formatMove(info, actionId, actor) {
  if (!info || !info.type) {
    if (actionId === null || actionId === undefined) return t('ll.action_start');
    return t('ll.action_unknown', { id: actionId });
  }
  // C++ emits English card_name/guess_name — translate at the JS layer
  // via cardLabel so no English leaks into bubbles or the info panel.
  const cardName = cardLabel(info.card || 0);
  const guessName = (typeof info.guess === 'number')
    ? cardLabel(info.guess)
    : (info.guess || '');
  // Self-target on a non-self-targeting card is the engine's no-op fallback
  // (every opponent Handmaid-protected). The bubble should say "无效"
  // instead of pretending the player meant to target themselves with a
  // particular guess/effect — that's confusing.
  const selfFallback =
    typeof actor === 'number' && actor >= 0 &&
    typeof info.target === 'number' && info.target === actor &&
    (info.type === 'guard' || info.type === 'priest' ||
     info.type === 'baron' || info.type === 'king');
  if (selfFallback) {
    return t('ll.card_invalid', { card: cardName });
  }
  switch (info.type) {
    case 'guard':
      return t('ll.guard_text', { card: cardName, target: info.target, guess: guessName });
    case 'priest':
      return t('ll.priest_text', { card: cardName, target: info.target });
    case 'baron':
      return t('ll.baron_text', { card: cardName, target: info.target });
    case 'handmaid':
      return t('ll.handmaid_text', { card: cardName });
    case 'prince':
      return t('ll.prince_text', { card: cardName, target: info.target });
    case 'king':
      return t('ll.king_text', { card: cardName, target: info.target });
    case 'countess':
      return t('ll.countess_text', { card: cardName });
    case 'princess':
      return t('ll.princess_text', { card: cardName });
    default:
      return t('ll.action_unknown', { id: actionId });
  }
}

function actorSelector(actor, humanPlayer) {
  return actor === humanPlayer
    ? '[data-player="' + actor + '"]'
    : '[data-opponent="' + actor + '"]';
}

function describeTransition(prevState, newState, actionInfo, actionId) {
  if (!actionInfo || !prevState || !prevState.state) return null;

  const steps = [];
  const actor = prevState.current_player;
  const humanPlayer = (currentCtx && currentCtx.state) ? currentCtx.state.humanPlayer : 0;
  const cardValue = actionInfo.card || 0;
  const target = actionInfo.target;

  // Append to the ordered-discard cache so play-order survives across the
  // post-animation re-render. We diff prev → new per player, so Prince's
  // target-also-discards effect is captured (otherwise the target's added
  // card would appear at multiset-sorted position rather than play-order).
  // The actor goes first (their played card lands before the target's
  // forced discard). Reconciliation in orderedDiscardsFor still recovers
  // if our log diverges (round reset / replay jump / undo).
  if (newState && newState.state && prevState.state) {
    const prevPlayers = prevState.state.players || [];
    const newPlayers = newState.state.players || [];
    const seatOrder = [];
    seatOrder.push(actor);
    for (let p = 0; p < newPlayers.length; p++) if (p !== actor) seatOrder.push(p);
    for (const p of seatOrder) {
      const before = (prevPlayers[p] && prevPlayers[p].discards) || [];
      const after = (newPlayers[p] && newPlayers[p].discards) || [];
      const counts = new Array(9).fill(0);
      for (const c of after) counts[c]++;
      for (const c of before) counts[c]--;
      // Anchor ordering: for the actor, the played card is the visible
      // play and lands first; remaining diff cards (e.g. self-Prince
      // post-redraw revealed Princess) come after. For non-actors only
      // forced discards appear in the diff.
      if (p === actor && cardValue > 0 && counts[cardValue] > 0) {
        recordDiscard(p, cardValue);
        counts[cardValue]--;
      }
      for (let c = 1; c <= 8; c++) {
        while (counts[c] > 0) {
          recordDiscard(p, c);
          counts[c]--;
        }
      }
    }
  }

  // Step 1: the card physically moves to the actor's discard pile, with
  // the action-bubble popping above them. This resolves BEFORE any target
  // highlight or reveal modal, so the player always sees "card A lands
  // in the play area" before the game adjudicates its effect. Without
  // this split the fly, the highlight, and the reveal all race each
  // other and the source card appears to vanish mid-air.
  const playGroup = { type: 'group', children: [] };
  playGroup.children.push({
    type: 'popup',
    target: actorSelector(actor, humanPlayer),
    content: formatMove(actionInfo, actionId, actor),
    className: 'action-bubble',
    width: 220,
    height: 36,
    duration: 1500,
  });
  // Shared onComplete: when the card lands in the incoming-slot anchor,
  // materialize a real discard-card DOM node INTO that anchor so the
  // discard pile visibly grows by one. Without this the incoming anchor
  // stays empty between the fly finishing and the final re-render, so
  // any subsequent step (target highlight, baron reveal modal) shows a
  // discard pile missing the card that just flew in — §3 "中间状态维护"
  // violation. We mutate DOM only; framework will re-render to real
  // state after all steps finish, which replaces our temporary node.
  const materializeInDiscard = (actor, cardValue) => () => {
    const incoming = document.querySelector('[data-discard-incoming="' + actor + '"]');
    if (!incoming) return;
    const card = document.createElement('div');
    card.className = 'll-discard-card card-' + cardValue;
    const v = document.createElement('div');
    v.className = 'll-discard-card-value';
    v.textContent = CARD_VALUES[cardValue];
    const n = document.createElement('div');
    n.className = 'll-discard-card-name';
    n.textContent = cardLabel(cardValue);
    card.appendChild(v);
    card.appendChild(n);
    // Insert before the incoming anchor so the pile grows leftward of it,
    // matching buildDiscardPile's output ordering.
    incoming.parentNode.insertBefore(card, incoming);
  };

  if (cardValue > 0 && actor === humanPlayer) {
    playGroup.children.push({
      type: 'fly',
      from: '[data-hand-card="' + cardValue + '"]',
      to: '[data-discard-incoming="' + actor + '"]',
      createElement() {
        const el = document.createElement('div');
        el.className = 'anim-flying-card ll-card-fly card-' + cardValue;
        el.textContent = CARD_VALUES[cardValue];
        return el;
      },
      duration: 450,
      hideFrom: true,
      onComplete: materializeInDiscard(actor, cardValue),
    });
  } else if (cardValue > 0 && actor !== humanPlayer) {
    playGroup.children.push({
      type: 'fly',
      from: '[data-opponent="' + actor + '"]',
      to: '[data-discard-incoming="' + actor + '"]',
      createElement() {
        const el = document.createElement('div');
        el.className = 'anim-flying-card ll-card-fly card-' + cardValue;
        el.textContent = CARD_VALUES[cardValue];
        return el;
      },
      width: 68,
      height: 92,
      duration: 450,
      onStart() {
        const cards = document.querySelectorAll('[data-revealed-hand="' + actor + '"]');
        for (const c of cards) {
          if (c.classList.contains('card-' + cardValue)) {
            c.style.visibility = 'hidden';
            break;
          }
        }
      },
      onComplete: materializeInDiscard(actor, cardValue),
    });
  }
  if (playGroup.children.length) steps.push(playGroup);

  // Step 2: target highlight runs after the play animation finishes, so
  // the player reads "card X is now in the pile, and it's targeting
  // player Y" as a sequence rather than a blur.
  if (target != null && target >= 0 && target !== actor) {
    const highlightSel = target === humanPlayer
      ? '[data-player="' + target + '"]'
      : '[data-opponent="' + target + '"]';
    steps.push({
      type: 'highlight',
      target: highlightSel,
      className: 'anim-highlight',
      duration: 600,
    });
  }

  // Information reveals: Priest peek and Baron compare expose private
  // cards to the perspective player. A head-bubble popup isn't enough
  // — the player must actually absorb the info before the next AI turn
  // starts, or the revealed card flashes by and is lost. Use a blocking
  // reveal-modal step that requires acknowledgement. See §3.7 of the
  // design doc for when to use this.
  const reveal = buildRevealStep(prevState, newState, actionInfo, actor, humanPlayer);
  if (reveal) steps.push(reveal);

  // End-of-round showdown: when this action ended the round (deck out,
  // or sole-survivor by elimination), reveal every still-alive player's
  // last hand card by flying it from their seat to their own discard pile
  // and materializing it there. The hand pile thus becomes the public
  // "table" and the player can see what everyone was holding at compare
  // time. The accompanying info-panel extension prints the same info.
  if (newState && newState.is_terminal && !prevState.is_terminal) {
    const showdown = buildShowdownStep(newState, actor, humanPlayer);
    if (showdown) steps.push(showdown);
  }

  return steps.length ? steps : null;
}

function buildShowdownStep(newState, actor, humanPlayer) {
  const players = (newState.state && newState.state.players) || [];
  const flights = [];
  for (let pi = 0; pi < players.length; pi++) {
    const p = players[pi];
    if (!p.alive) continue;
    const card = p.hand;
    if (!card || card <= 0) continue;
    const fromSel = pi === humanPlayer
      ? '[data-hand-card="' + card + '"]'
      : '[data-opponent="' + pi + '"]';
    flights.push({
      type: 'fly',
      from: fromSel,
      to: '[data-discard-incoming="' + pi + '"]',
      createElement() {
        const el = document.createElement('div');
        el.className = 'anim-flying-card ll-card-fly card-' + card;
        el.textContent = CARD_VALUES[card];
        return el;
      },
      width: 68,
      height: 92,
      duration: 500,
      onComplete: (() => {
        const incoming = document.querySelector('[data-discard-incoming="' + pi + '"]');
        if (!incoming) return;
        const cardEl = document.createElement('div');
        cardEl.className = 'll-discard-card card-' + card;
        const v = document.createElement('div');
        v.className = 'll-discard-card-value';
        v.textContent = CARD_VALUES[card];
        const n = document.createElement('div');
        n.className = 'll-discard-card-name';
        n.textContent = cardLabel(card);
        cardEl.appendChild(v);
        cardEl.appendChild(n);
        incoming.parentNode.insertBefore(cardEl, incoming);
      }),
    });
  }
  if (!flights.length) return null;
  return { type: 'group', children: flights };
}

function buildRevealStep(prevState, newState, actionInfo, actor, humanPlayer) {
  const type = actionInfo && actionInfo.type;
  if (type !== 'priest' && type !== 'baron') return null;

  const target = actionInfo.target;
  if (target == null || target < 0) return null;
  // Self-target is the engine's no-op fallback (every opponent protected).
  // No information changes hands — skip the reveal modal entirely so the
  // player isn't shown a useless "look at your own card" or "compare X vs
  // X" popup.
  if (target === actor) return null;

  // Only interrupt the player when they're actually learning something.
  // - Priest: only the actor learns. Skip unless human is the actor.
  // - Baron: both compared hands become known to actor and target.
  //   Skip if neither is the human.
  const humanIsActor = actor === humanPlayer;
  const humanIsTarget = target === humanPlayer;
  if (type === 'priest' && !humanIsActor) return null;
  if (type === 'baron' && !humanIsActor && !humanIsTarget) return null;

  // The cards this move showed us were hidden before it (an opponent's
  // hand is not in our pre-move state), so they are read from the state
  // AFTER the move — see comparedCard.
  const targetPrevHand = comparedCard(prevState, newState, target, 0);

  if (type === 'priest') {
    if (targetPrevHand <= 0) return null;
    return {
      type: 'reveal',
      title: t('ll.priest_modal_title'),
      body: revealBody([
        { label: t('ll.player_hand_label', { n: target }), card: targetPrevHand },
      ], t('ll.peek_summary')),
    };
  }

  // Baron: compare actor's kept card (after playing Baron, they retain
  // their OTHER card) with target's hand. We read what each player
  // actually held at compare time:
  //   - The actor played Baron. Their compare card is whichever of
  //     {hand, drawn_card} wasn't the Baron (value 3). In prevState
  //     the current_player's drawn_card is visible; their hand is the
  //     permanent one.
  //   - The target's hand is simply prevPlayers[target].hand.
  let actorKept = baronActorKeptCard(prevState, actor);
  if (actorKept <= 0) actorKept = comparedCard(prevState, newState, actor, 3);
  if (actorKept <= 0 || targetPrevHand <= 0) return null;

  const actorWon = actorKept > targetPrevHand;
  const tie = actorKept === targetPrevHand;
  const humanSide = humanIsActor ? 'actor' : 'target';
  // "outcomeText" from the human's perspective: won / lost / tied.
  const outcomeText = tie
    ? t('ll.baron_tie')
    : (humanSide === 'actor'
        ? (actorWon
            ? t('ll.baron_human_actor_won', { target })
            : t('ll.baron_human_actor_lost', { target }))
        : (actorWon
            ? t('ll.baron_human_target_lost', { actor })
            : t('ll.baron_human_target_won', { actor })));

  return {
    type: 'reveal',
    title: t('ll.baron_modal_title'),
    body: revealBody([
      { label: t('ll.player_hand_label_sp', { n: actor }), card: actorKept },
      { label: t('ll.player_hand_label_sp', { n: target }), card: targetPrevHand },
    ], outcomeText),
  };
}

// The hand card `seat` held when a Priest / Baron showed it to us.
//   - known before the move (our own card, or a full-information replay);
//   - else still in hand after the move: the peek / the Baron comparison
//     made it visible to us;
//   - else `seat` lost the comparison and was eliminated: its hand is the
//     card that newly appeared in its discard pile (`played` is the card
//     the seat itself just played, which lands there too).
function comparedCard(prevState, newState, seat, played) {
  const prevP = (prevState.state.players || [])[seat];
  const newP = newState && newState.state ? (newState.state.players || [])[seat] : null;
  if (prevP && prevP.hand > 0) return prevP.hand;
  if (!newP) return 0;
  if (newP.alive && newP.hand > 0) return newP.hand;
  const before = (prevP && prevP.discard_count) || [];
  const after = newP.discard_count || [];
  for (let c = 1; c <= 8; c++) {
    let added = (after[c] || 0) - (before[c] || 0);
    if (c === played) added -= 1;
    if (added > 0) return c;
  }
  return 0;
}

function baronActorKeptCard(prevState, actor) {
  const st = prevState.state;
  const p = st.players && st.players[actor];
  if (!p) return 0;
  const hand = p.hand || 0;
  // drawn_card in prevState.state is only populated for current_player.
  // In the pre-action snapshot, current_player == actor, so this is the
  // two-card hand the actor just picked from. Whichever isn't Baron (3)
  // is the card they kept to compare.
  const drawn = st.drawn_card || 0;
  if (hand === 3 && drawn > 0) return drawn;
  if (drawn === 3 && hand > 0) return hand;
  // Edge: neither equals Baron — shouldn't happen legally, but fall
  // back to whichever is the higher value so the reveal still shows
  // something sensible.
  return Math.max(hand, drawn);
}

function revealBody(cards, summaryText) {
  const wrap = document.createElement('div');
  wrap.style.display = 'flex';
  wrap.style.flexDirection = 'column';
  wrap.style.alignItems = 'center';
  wrap.style.gap = '12px';

  const row = document.createElement('div');
  row.style.display = 'flex';
  row.style.gap = '14px';
  row.style.justifyContent = 'center';
  for (const c of cards) {
    const item = document.createElement('div');
    item.style.display = 'flex';
    item.style.flexDirection = 'column';
    item.style.alignItems = 'center';
    item.style.gap = '6px';
    const label = document.createElement('div');
    label.style.fontSize = '12px';
    label.style.color = '#475569';
    label.textContent = c.label;
    item.appendChild(label);
    const card = document.createElement('div');
    card.className = 'll-reveal-card card-' + c.card;
    const val = document.createElement('div');
    val.className = 'll-reveal-card-value';
    val.textContent = CARD_VALUES[c.card];
    const name = document.createElement('div');
    name.className = 'll-reveal-card-name';
    name.textContent = cardLabel(c.card);
    card.appendChild(val);
    card.appendChild(name);
    item.appendChild(card);
    row.appendChild(item);
  }
  wrap.appendChild(row);

  if (summaryText) {
    const summary = document.createElement('div');
    summary.style.fontSize = '13px';
    summary.style.color = '#334155';
    summary.textContent = summaryText;
    wrap.appendChild(summary);
  }
  return wrap;
}

// Showdown extension: at game end, list every alive player's last hand
// card in the info panel so the showdown is also readable as text. Empty
// during normal play.
const showdownExtension = {
  render(el, gameState) {
    if (!gameState || !gameState.state || !gameState.is_terminal) {
      el.style.display = 'none';
      return;
    }
    const players = gameState.state.players || [];
    const parts = [];
    for (let pi = 0; pi < players.length; pi++) {
      const p = players[pi];
      if (!p.alive || !p.hand || p.hand <= 0) continue;
      parts.push(t('ll.showdown_entry', { n: pi, card: cardLabel(p.hand) }));
    }
    if (!parts.length) {
      el.style.display = 'none';
      return;
    }
    el.style.display = '';
    el.textContent = t('ll.showdown_prefix') + parts.join('  ');
  }
};

createApp({
  gameId: 'loveletter',
  gameTitle: t('ll.title'),
  gameIntro: t('ll.intro'),
  players: { min: 2, max: 4 },
  renderBoard,
  renderPlayerArea,
  describeTransition,
  formatOpponentMove: formatMove,
  formatSuggestedMove: formatMove,
  getPlayerSymbol: (p) => t('ll.player_symbol', { n: p }),
  extensions: [showdownExtension],
  difficulties: ['heuristic', 'casual', 'expert'],
  defaultDifficulty: 'expert',
  defaultSeat: 'random',
  // Hidden-info game: 替对手落子 makes no sense — you don't see opp's
  // hand. AI 胜率 also leaks: it's computed from MCTS that knows the
  // human's true hand, so it swings sharply when the human draws a
  // strong card and lets the user infer it.
  disableForce: true,
  showWinrateDefault: false,
  onActionSubmitted: () => { resetPending(); },
  onGameStart: () => { resetPending(); resetDiscardOrder(); },
  onUndo: () => { resetPending(); resetDiscardOrder(); },
  onReplayFrames: (frames) => { precomputeReplayDiscardOrder(frames); },
});
