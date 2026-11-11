import { createApp } from '/static/general/app.js';
import { t, register } from '/static/general/i18n.js';

register({
  zh: {
    'coup.title': '政变',
    'coup.intro': '点动作面板宣告操作；阻挡/质疑会自动进入对应阶段',
    'coup.char_duke': '公爵',
    'coup.char_assassin': '刺客',
    'coup.char_captain': '队长',
    'coup.char_ambassador': '大使',
    'coup.char_contessa': '伯爵夫人',
    'coup.stage_0': '声明行动',
    'coup.stage_1': '质疑行动',
    'coup.stage_2': '解决质疑',
    'coup.stage_3': '失去影响力',
    'coup.stage_4': '反制',
    'coup.stage_5': '质疑反制',
    'coup.stage_6': '解决反制质疑',
    'coup.stage_7': '失去影响力(反制)',
    'coup.stage_8': '交换还牌1',
    'coup.stage_9': '交换还牌2',
    'coup.stage_10': '失去影响力(行动)',
    'coup.player_n': '玩家{n}',
    'coup.player_n_eliminated': '玩家{n} (淘汰)',
    'coup.coins': '💰 {n}',
    'coup.reveal_panel_prove': '亮牌证明',
    'coup.reveal_panel_lose': '选择失去影响力',
    'coup.idle_wait_opp': '等待对手行动...',
    'coup.idle_wait_player': '等待玩家{n}行动...',
    'coup.choose_action': '选择行动',
    'coup.act_income_label': '收入',
    'coup.act_income_desc': '+1💰',
    'coup.act_foreign_aid_label': '外援',
    'coup.act_foreign_aid_desc': '+2💰',
    'coup.act_tax_label': '征税',
    'coup.act_tax_desc': '+3💰 (公爵)',
    'coup.act_exchange_label': '交换',
    'coup.act_exchange_desc': '换牌 (大使)',
    'coup.act_coup_label': '政变',
    'coup.act_coup_desc': '-7💰 (不可阻挡)',
    'coup.act_assassinate_label': '暗杀',
    'coup.act_assassinate_desc': '-3💰 (刺客)',
    'coup.act_steal_label': '偷窃',
    'coup.act_steal_desc': '偷2💰 (队长)',
    'coup.click_target_player': '请点击目标玩家',
    'coup.challenge_q': '是否质疑?',
    'coup.challenge_yes': '质疑!',
    'coup.allow': '允许',
    'coup.counter_q': '是否反制?',
    'coup.block_duke': '反制 (公爵)',
    'coup.block_contessa': '反制 (伯爵夫人)',
    'coup.block_ambassador': '反制 (大使)',
    'coup.block_captain': '反制 (队长)',
    'coup.allow_no_block': '不反制',
    'coup.exchange_return_1': '点击手牌还回第1张',
    'coup.exchange_return_2': '点击手牌还回第2张',
    'coup.exchange_hint_1': '选择还回第1张牌',
    'coup.exchange_hint_2': '选择还回第2张牌',
    'coup.you_eliminated': '你已被淘汰',
    'coup.action_unknown': '动作 #{id}',
    'coup.action_start': '开局',
    'coup.move_income': '收入 (+1💰)',
    'coup.move_foreign_aid': '外援 (+2💰)',
    'coup.move_coup': '政变 → 玩家{target}',
    'coup.move_tax': '征税 [公爵] (+3💰)',
    'coup.move_assassinate': '暗杀 → 玩家{target} [刺客]',
    'coup.move_steal': '偷窃 → 玩家{target} [队长]',
    'coup.move_exchange': '交换 [大使]',
    'coup.move_challenge': '质疑!',
    'coup.move_allow': '允许',
    'coup.move_block': '反制 [{role}]',
    'coup.move_allow_no_block': '不反制',
    'coup.move_reveal': '亮牌 第{n}张',
    'coup.move_lose_influence': '失去影响力 第{n}张',
    'coup.move_return_card': '还 {role}',
    'coup.move_return_card_hidden': '还回一张',
    'coup.challenge_modal_title': '质疑 · 对方亮牌',
    'coup.challenge_caption': '玩家{n} 亮出 {role}',
    'coup.challenge_failed_suffix': '，质疑失败（你将失去影响力）',
    'coup.challenge_succeeded_suffix': '，质疑成功（对手将失去影响力）',
    'coup.stage_pill_default': '阶段：--',
    'coup.stage_pill_label': '阶段：{name}',
    'coup.deck_pill': '牌堆：{n}',
    'coup.stage_dash': '--',
    'coup.banner_idle': '等待开始',
    'coup.banner_declared': '玩家{actor} 宣告 {action}',
    'coup.banner_declared_target': '玩家{actor} 宣告 {action} → 玩家{target}',
    'coup.banner_challenge_pending': '玩家{actor} 宣告 {action}，等待是否质疑',
    'coup.banner_challenge_pending_target': '玩家{actor} 宣告 {action} → 玩家{target}，等待是否质疑',
    'coup.banner_block_pending': '玩家{actor} 宣告 {action}，等待是否反制',
    'coup.banner_block_pending_target': '玩家{actor} 宣告 {action} → 玩家{target}，等待是否反制',
    'coup.banner_blocked': '玩家{blocker} 用 {role} 反制 玩家{actor} 的 {action}',
    'coup.banner_block_challenged': '玩家{challenger} 质疑 玩家{blocker} 的反制（{role}）',
    'coup.banner_action_challenged': '玩家{challenger} 质疑 玩家{actor} 的 {action}',
    'coup.banner_resolve_action': '解决质疑：玩家{actor} 需亮牌证明 {action}',
    'coup.banner_resolve_block': '解决反制质疑：玩家{blocker} 需亮牌证明 {role}',
    'coup.banner_lose_inf': '玩家{actor} 失去影响力',
    'coup.banner_lose_inf_action': '玩家{actor} 失去影响力（被政变/暗杀）',
    'coup.banner_exchange': '玩家{actor} 正在交换牌（大使）',
    'coup.act_short_income': '收入',
    'coup.act_short_foreign_aid': '外援',
    'coup.act_short_coup': '政变',
    'coup.act_short_tax': '征税(公爵)',
    'coup.act_short_assassinate': '暗杀(刺客)',
    'coup.act_short_steal': '偷窃(队长)',
    'coup.act_short_exchange': '交换(大使)',
  },
  en: {
    'coup.title': 'Coup',
    'coup.intro': 'Click an action button to declare; blocks/challenges enter their stages automatically.',
    'coup.char_duke': 'Duke',
    'coup.char_assassin': 'Assassin',
    'coup.char_captain': 'Captain',
    'coup.char_ambassador': 'Ambassador',
    'coup.char_contessa': 'Contessa',
    'coup.stage_0': 'Declare action',
    'coup.stage_1': 'Challenge action',
    'coup.stage_2': 'Resolve challenge',
    'coup.stage_3': 'Lose influence',
    'coup.stage_4': 'Block',
    'coup.stage_5': 'Challenge block',
    'coup.stage_6': 'Resolve block challenge',
    'coup.stage_7': 'Lose influence (block)',
    'coup.stage_8': 'Exchange return 1',
    'coup.stage_9': 'Exchange return 2',
    'coup.stage_10': 'Lose influence (action)',
    'coup.player_n': 'Player {n}',
    'coup.player_n_eliminated': 'Player {n} (out)',
    'coup.coins': '💰 {n}',
    'coup.reveal_panel_prove': 'Reveal to prove',
    'coup.reveal_panel_lose': 'Choose card to lose',
    'coup.idle_wait_opp': 'Waiting for opponent...',
    'coup.idle_wait_player': 'Waiting for player {n}...',
    'coup.choose_action': 'Choose an action',
    'coup.act_income_label': 'Income',
    'coup.act_income_desc': '+1💰',
    'coup.act_foreign_aid_label': 'Foreign Aid',
    'coup.act_foreign_aid_desc': '+2💰',
    'coup.act_tax_label': 'Tax',
    'coup.act_tax_desc': '+3💰 (Duke)',
    'coup.act_exchange_label': 'Exchange',
    'coup.act_exchange_desc': 'swap cards (Ambassador)',
    'coup.act_coup_label': 'Coup',
    'coup.act_coup_desc': '-7💰 (unblockable)',
    'coup.act_assassinate_label': 'Assassinate',
    'coup.act_assassinate_desc': '-3💰 (Assassin)',
    'coup.act_steal_label': 'Steal',
    'coup.act_steal_desc': 'take 2💰 (Captain)',
    'coup.click_target_player': 'Click a target player',
    'coup.challenge_q': 'Challenge?',
    'coup.challenge_yes': 'Challenge!',
    'coup.allow': 'Allow',
    'coup.counter_q': 'Block?',
    'coup.block_duke': 'Block (Duke)',
    'coup.block_contessa': 'Block (Contessa)',
    'coup.block_ambassador': 'Block (Ambassador)',
    'coup.block_captain': 'Block (Captain)',
    'coup.allow_no_block': 'Do not block',
    'coup.exchange_return_1': 'Click a card to return as #1',
    'coup.exchange_return_2': 'Click a card to return as #2',
    'coup.exchange_hint_1': 'Choose card to return (#1)',
    'coup.exchange_hint_2': 'Choose card to return (#2)',
    'coup.you_eliminated': 'You are eliminated',
    'coup.action_unknown': 'Action #{id}',
    'coup.action_start': 'Game start',
    'coup.move_income': 'Income (+1💰)',
    'coup.move_foreign_aid': 'Foreign Aid (+2💰)',
    'coup.move_coup': 'Coup → player {target}',
    'coup.move_tax': 'Tax [Duke] (+3💰)',
    'coup.move_assassinate': 'Assassinate → player {target} [Assassin]',
    'coup.move_steal': 'Steal → player {target} [Captain]',
    'coup.move_exchange': 'Exchange [Ambassador]',
    'coup.move_challenge': 'Challenge!',
    'coup.move_allow': 'Allow',
    'coup.move_block': 'Block [{role}]',
    'coup.move_allow_no_block': 'Do not block',
    'coup.move_reveal': 'Reveal card #{n}',
    'coup.move_lose_influence': 'Lose influence #{n}',
    'coup.move_return_card': 'Return {role}',
    'coup.move_return_card_hidden': 'Return a card',
    'coup.challenge_modal_title': 'Challenge · Reveal',
    'coup.challenge_caption': 'Player {n} reveals {role}',
    'coup.challenge_failed_suffix': ' — challenge failed (you will lose influence)',
    'coup.challenge_succeeded_suffix': ' — challenge succeeded (opponent will lose influence)',
    'coup.stage_pill_default': 'Stage: --',
    'coup.stage_pill_label': 'Stage: {name}',
    'coup.deck_pill': 'Deck: {n}',
    'coup.stage_dash': '--',
    'coup.banner_idle': 'Waiting to start',
    'coup.banner_declared': 'P{actor} declares {action}',
    'coup.banner_declared_target': 'P{actor} declares {action} → P{target}',
    'coup.banner_challenge_pending': 'P{actor} declares {action} — challenge?',
    'coup.banner_challenge_pending_target': 'P{actor} declares {action} → P{target} — challenge?',
    'coup.banner_block_pending': 'P{actor} declares {action} — block?',
    'coup.banner_block_pending_target': 'P{actor} declares {action} → P{target} — block?',
    'coup.banner_blocked': 'P{blocker} blocks P{actor}\'s {action} with {role}',
    'coup.banner_block_challenged': 'P{challenger} challenges P{blocker}\'s block ({role})',
    'coup.banner_action_challenged': 'P{challenger} challenges P{actor}\'s {action}',
    'coup.banner_resolve_action': 'Resolve challenge: P{actor} must reveal to prove {action}',
    'coup.banner_resolve_block': 'Resolve block challenge: P{blocker} must reveal {role}',
    'coup.banner_lose_inf': 'P{actor} must lose influence',
    'coup.banner_lose_inf_action': 'P{actor} must lose influence (coup/assassinate)',
    'coup.banner_exchange': 'P{actor} is exchanging cards (Ambassador)',
    'coup.act_short_income': 'Income',
    'coup.act_short_foreign_aid': 'Foreign Aid',
    'coup.act_short_coup': 'Coup',
    'coup.act_short_tax': 'Tax(Duke)',
    'coup.act_short_assassinate': 'Assassinate(Assassin)',
    'coup.act_short_steal': 'Steal(Captain)',
    'coup.act_short_exchange': 'Exchange(Ambassador)',
  },
});

const CHAR_KEYS = ['coup.char_duke', 'coup.char_assassin', 'coup.char_captain', 'coup.char_ambassador', 'coup.char_contessa'];
const CHAR_COLORS = ['#c8a', '#e55', '#58c', '#5a8', '#ea8'];

function charLabel(c) {
  if (c < 0 || c >= CHAR_KEYS.length) return '';
  return t(CHAR_KEYS[c]);
}

function stageName(s) {
  if (s == null || s < 0 || s > 10) return t('coup.stage_dash');
  return t('coup.stage_' + s);
}

// Short label for the action currently being declared. Used inside the
// top-of-board status banner so the player can see "P2 declared Tax" at
// a glance during multi-player rounds.
function declaredActionShort(declaredAction) {
  if (declaredAction == null || declaredAction < 0) return null;
  const a = declaredAction;
  // kIncomeAction=0, kForeignAidAction=1, kCoupOffset=2, kTaxAction=6,
  // kAssassinateOffset=7, kStealOffset=11, kExchangeAction=15.
  if (a === 0) return t('coup.act_short_income');
  if (a === 1) return t('coup.act_short_foreign_aid');
  if (a >= 2 && a <= 5) return t('coup.act_short_coup');
  if (a === 6) return t('coup.act_short_tax');
  if (a >= 7 && a <= 10) return t('coup.act_short_assassinate');
  if (a >= 11 && a <= 14) return t('coup.act_short_steal');
  if (a === 15) return t('coup.act_short_exchange');
  return null;
}

// Build the status banner shown at the top of the board. In multi-
// player rounds, the human can lose track of who declared what before
// it loops back to them — this banner re-states the action, target,
// and any pending block/challenge.
function renderStatusBanner(st) {
  const banner = document.createElement('div');
  banner.className = 'coup-status-banner';

  const stage = st.stage;
  const actor = st.active_player;
  const target = st.action_target;
  const action = declaredActionShort(st.declared_action);
  const blocker = st.blocker;
  const challenger = st.challenger;
  const blockRole = st.block_character != null && st.block_character >= 0
    ? charLabel(st.block_character) : null;
  const claimedRole = st.claimed_character != null && st.claimed_character >= 0
    ? charLabel(st.claimed_character) : null;

  const hasTarget = target != null && target >= 0;
  let text = '';

  if (stage === 0 || action == null) {
    // Pre-declaration or unknown: keep banner present but minimal so
    // the layout doesn't shift when stage advances.
    text = t('coup.banner_idle');
  } else if (stage === 1) {
    // Action declared, awaiting challenge.
    text = hasTarget
      ? t('coup.banner_challenge_pending_target', { actor, action, target })
      : t('coup.banner_challenge_pending', { actor, action });
  } else if (stage === 2) {
    // Action was challenged: actor must reveal.
    text = challenger != null && challenger >= 0
      ? t('coup.banner_action_challenged', { challenger, actor, action })
      : t('coup.banner_resolve_action', { actor, action });
  } else if (stage === 3) {
    text = t('coup.banner_lose_inf', { actor });
  } else if (stage === 4) {
    // Action passed challenge phase, awaiting block.
    text = hasTarget
      ? t('coup.banner_block_pending_target', { actor, action, target })
      : t('coup.banner_block_pending', { actor, action });
  } else if (stage === 5) {
    // Block declared, awaiting block-challenge.
    text = t('coup.banner_blocked', {
      blocker: blocker != null ? blocker : '?',
      actor, action,
      role: blockRole != null ? blockRole : '?',
    });
  } else if (stage === 6) {
    text = t('coup.banner_block_challenged', {
      challenger: challenger != null ? challenger : '?',
      blocker: blocker != null ? blocker : '?',
      role: blockRole != null ? blockRole : '?',
    });
  } else if (stage === 7) {
    text = t('coup.banner_lose_inf', { actor: blocker != null ? blocker : actor });
  } else if (stage === 8 || stage === 9) {
    text = t('coup.banner_exchange', { actor });
  } else if (stage === 10) {
    text = t('coup.banner_lose_inf_action', { actor });
  } else {
    text = t('coup.banner_idle');
  }

  banner.textContent = text;
  return banner;
}

function playerLabel(n) {
  return t('coup.player_n', { n });
}

const INCOME = 0, FOREIGN_AID = 1;
const COUP_OFF = 2, COUP_COUNT = 4;
const TAX = 6;
const ASSASSINATE_OFF = 7, ASSASSINATE_COUNT = 4;
const STEAL_OFF = 11, STEAL_COUNT = 4;
const EXCHANGE = 15;
const CHALLENGE = 16, ALLOW = 17;
const BLOCK_DUKE = 18, BLOCK_CONTESSA = 19, BLOCK_AMBASSADOR = 20, BLOCK_CAPTAIN = 21;
const ALLOW_NO_BLOCK = 22;
const REVEAL_0 = 23, REVEAL_1 = 24;
const LOSE_0 = 25, LOSE_1 = 26;
const RETURN_DUKE = 27;

let pendingActionType = null;
let currentCtx = null;
let autoCardChoiceKey = null;
let autoCardChoiceTimer = null;

function getLegalSet(gs) {
  if (!gs) return new Set();
  return new Set(gs.legal_actions || []);
}

function resetPending() {
  pendingActionType = null;
  autoCardChoiceKey = null;
  if (autoCardChoiceTimer !== null) {
    clearTimeout(autoCardChoiceTimer);
    autoCardChoiceTimer = null;
  }
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
  const stage = st.stage;

  const board = document.createElement('div');
  board.className = 'coup-board';

  // Status banner — anchored at the top so the player can see "who is
  // doing what / who is blocking" without scrolling. Especially useful
  // in 3p/4p where many ply pass before it's the human's turn again.
  board.appendChild(renderStatusBanner(st));

  // Stage / deck info used to live in an in-board info bar whose width
  // shifted between stages — moved to the sidebar-info extension (see
  // `extensions` in createApp) so the game area stays visually stable.
  const playersArea = document.createElement('div');
  playersArea.className = 'coup-players';

  for (let pi = 0; pi < numPlayers; pi++) {
    if (pi === humanPlayer) continue;
    const p = st.players[pi];
    const el = document.createElement('div');
    el.className = 'coup-player-card';
    el.setAttribute('data-player', String(pi));
    if (!p.alive) el.classList.add('eliminated');
    if (pi === st.current_player) el.classList.add('current-turn');
    if (pi === st.active_player) el.classList.add('active');

    const canTarget = playing && pendingActionType !== null;
    let isValidTarget = false;
    if (canTarget) {
      let actionId = -1;
      if (pendingActionType === 'coup') actionId = COUP_OFF + pi;
      else if (pendingActionType === 'assassinate') actionId = ASSASSINATE_OFF + pi;
      else if (pendingActionType === 'steal') actionId = STEAL_OFF + pi;
      if (actionId >= 0 && legalSet.has(actionId)) isValidTarget = true;
    }

    if (isValidTarget) {
      el.classList.add('selectable');
      el.addEventListener('click', () => {
        let aid;
        if (pendingActionType === 'coup') aid = COUP_OFF + pi;
        else if (pendingActionType === 'assassinate') aid = ASSASSINATE_OFF + pi;
        else if (pendingActionType === 'steal') aid = STEAL_OFF + pi;
        resetPending();
        ctx.submitAction(aid);
      });
    }

    const nameRow = document.createElement('div');
    nameRow.className = 'coup-pname';
    nameRow.textContent = p.alive
      ? playerLabel(pi)
      : t('coup.player_n_eliminated', { n: pi });
    el.appendChild(nameRow);

    const coinsEl = document.createElement('div');
    coinsEl.className = 'coup-coins';
    coinsEl.setAttribute('data-coins', String(pi));
    coinsEl.textContent = t('coup.coins', { n: p.coins });
    el.appendChild(coinsEl);

    const infArea = document.createElement('div');
    infArea.className = 'coup-influences';
    for (let sl = 0; sl < 2; sl++) {
      const inf = p.influences[sl];
      const card = document.createElement('div');
      card.className = 'coup-inf-card';
      card.setAttribute('data-influence', pi + '-' + sl);
      if (inf.revealed) {
        card.classList.add('revealed');
        card.textContent = charLabel(inf.character);
        card.style.borderColor = CHAR_COLORS[inf.character];
      } else if (ctx.revealHidden && inf.character != null && inf.character >= 0) {
        // Replay-only reveal: show the opponent's actual influence card.
        // ctx.revealHidden is gated on state.replayMode so live play can't
        // reach this branch.
        card.classList.add('hidden');
        card.classList.add('revealed-hidden');
        card.textContent = charLabel(inf.character);
        card.style.borderColor = CHAR_COLORS[inf.character];
      } else {
        card.classList.add('hidden');
        card.textContent = '?';
      }
      infArea.appendChild(card);
    }
    el.appendChild(infArea);

    playersArea.appendChild(el);
  }
  board.appendChild(playersArea);

  // Always render the action-panel slot — even when it's the AI's turn or
  // the human has nothing to decide — so the section below the players
  // area keeps a stable position and height. Each stage swaps in its own
  // button row; illegal actions are disabled rather than removed.
  let panel = null;
  if (playing && stage === 0) {
    panel = renderDeclareActions(st, legalSet, humanPlayer, ctx);
  } else if (playing && (stage === 1 || stage === 5)) {
    panel = renderChallengePanel(legalSet, ctx);
  } else if (playing && stage === 4) {
    panel = renderCounterPanel(st, legalSet, ctx);
  } else if (playing && (stage === 2 || stage === 6)) {
    panel = renderRevealPanel(st, legalSet, humanPlayer, ctx, t('coup.reveal_panel_prove'));
  } else if (playing && (stage === 3 || stage === 7 || stage === 10)) {
    panel = renderRevealPanel(st, legalSet, humanPlayer, ctx, t('coup.reveal_panel_lose'));
  } else if (playing && (stage === 8 || stage === 9)) {
    panel = renderExchangePanel(st, legalSet, ctx);
  } else {
    panel = renderIdlePanel(st, humanPlayer);
  }
  board.appendChild(panel);

  container.appendChild(board);
}

function renderIdlePanel(st, humanPlayer) {
  // Placeholder panel shown when it's not the human's turn or when the
  // current stage doesn't require human input. Keeps the action-bar slot
  // occupying the same footprint as any real panel.
  const panel = document.createElement('div');
  panel.className = 'coup-action-panel idle';

  const title = document.createElement('div');
  title.className = 'coup-panel-title';
  const actor = (st.active_player != null) ? st.active_player : st.current_player;
  title.textContent = actor === humanPlayer
    ? t('coup.idle_wait_opp')
    : t('coup.idle_wait_player', { n: actor });
  panel.appendChild(title);

  // Empty button row matching the real panels' layout — gives the panel
  // the same vertical footprint so switching into a real stage doesn't
  // jolt the layout.
  const btns = document.createElement('div');
  btns.className = 'coup-action-btns';
  const spacer = document.createElement('div');
  spacer.className = 'coup-action-btn-spacer';
  spacer.innerHTML = '&nbsp;';
  btns.appendChild(spacer);
  panel.appendChild(btns);

  return panel;
}

function renderDeclareActions(st, legalSet, humanPlayer, ctx) {
  const panel = document.createElement('div');
  panel.className = 'coup-action-panel';

  const title = document.createElement('div');
  title.className = 'coup-panel-title';
  title.textContent = t('coup.choose_action');
  panel.appendChild(title);

  const btns = document.createElement('div');
  btns.className = 'coup-action-btns';

  // Render ALL 7 declare-stage buttons in a fixed order. Illegal actions
  // (coins<7 → coup disabled; coins≥10 → everything except coup disabled)
  // are still rendered in their slot, only greyed out. This keeps the
  // action bar shape stable across turns so the layout doesn't jump.
  const actions = [
    { id: INCOME, labelKey: 'coup.act_income_label', descKey: 'coup.act_income_desc', type: 'safe' },
    { id: FOREIGN_AID, labelKey: 'coup.act_foreign_aid_label', descKey: 'coup.act_foreign_aid_desc', type: 'safe' },
    { id: TAX, labelKey: 'coup.act_tax_label', descKey: 'coup.act_tax_desc', type: 'bluff' },
    { id: EXCHANGE, labelKey: 'coup.act_exchange_label', descKey: 'coup.act_exchange_desc', type: 'bluff' },
  ];

  for (const a of actions) {
    const legal = legalSet.has(a.id);
    const btn = document.createElement('button');
    btn.className = 'coup-action-btn ' + a.type;
    btn.disabled = !legal;
    btn.innerHTML = '<span class="coup-btn-label">' + t(a.labelKey) + '</span>'
      + '<span class="coup-btn-desc">' + t(a.descKey) + '</span>';
    if (legal) {
      btn.addEventListener('click', () => {
        resetPending();
        ctx.submitAction(a.id);
      });
    }
    btns.appendChild(btn);
  }

  const targetActions = [
    { type: 'coup', labelKey: 'coup.act_coup_label', descKey: 'coup.act_coup_desc', off: COUP_OFF, count: COUP_COUNT, cls: 'coup-act' },
    { type: 'assassinate', labelKey: 'coup.act_assassinate_label', descKey: 'coup.act_assassinate_desc', off: ASSASSINATE_OFF, count: ASSASSINATE_COUNT, cls: 'bluff' },
    { type: 'steal', labelKey: 'coup.act_steal_label', descKey: 'coup.act_steal_desc', off: STEAL_OFF, count: STEAL_COUNT, cls: 'bluff' },
  ];

  for (const ta of targetActions) {
    let hasAny = false;
    for (let ti = 0; ti < 4; ti++) {
      if (legalSet.has(ta.off + ti)) { hasAny = true; break; }
    }
    const btn = document.createElement('button');
    btn.className = 'coup-action-btn ' + ta.cls;
    btn.disabled = !hasAny;
    if (pendingActionType === ta.type) btn.classList.add('selected');
    btn.innerHTML = '<span class="coup-btn-label">' + t(ta.labelKey) + '</span>'
      + '<span class="coup-btn-desc">' + t(ta.descKey) + '</span>';
    if (hasAny) {
      btn.addEventListener('click', () => {
        if (pendingActionType === ta.type) {
          resetPending();
          ctx.rerender();
        } else {
          pendingActionType = ta.type;
          ctx.rerender();
        }
      });
    }
    btns.appendChild(btn);
  }

  panel.appendChild(btns);

  // Reserved hint slot so the panel height doesn't pop when pendingActionType
  // appears / disappears.
  const hint = document.createElement('div');
  hint.className = 'coup-pending-hint';
  hint.textContent = pendingActionType ? t('coup.click_target_player') : ' ';
  if (!pendingActionType) hint.style.visibility = 'hidden';
  panel.appendChild(hint);

  return panel;
}

function renderChallengePanel(legalSet, ctx) {
  const panel = document.createElement('div');
  panel.className = 'coup-action-panel';

  const title = document.createElement('div');
  title.className = 'coup-panel-title';
  title.textContent = t('coup.challenge_q');
  panel.appendChild(title);

  const btns = document.createElement('div');
  btns.className = 'coup-action-btns';

  if (legalSet.has(CHALLENGE)) {
    const btn = document.createElement('button');
    btn.className = 'coup-action-btn challenge';
    btn.textContent = t('coup.challenge_yes');
    btn.addEventListener('click', () => ctx.submitAction(CHALLENGE));
    btns.appendChild(btn);
  }
  if (legalSet.has(ALLOW)) {
    const btn = document.createElement('button');
    btn.className = 'coup-action-btn allow';
    btn.textContent = t('coup.allow');
    btn.addEventListener('click', () => ctx.submitAction(ALLOW));
    btns.appendChild(btn);
  }

  panel.appendChild(btns);
  return panel;
}

function renderCounterPanel(st, legalSet, ctx) {
  const panel = document.createElement('div');
  panel.className = 'coup-action-panel';

  const title = document.createElement('div');
  title.className = 'coup-panel-title';
  title.textContent = t('coup.counter_q');
  panel.appendChild(title);

  const btns = document.createElement('div');
  btns.className = 'coup-action-btns';

  const blocks = [
    { id: BLOCK_DUKE, labelKey: 'coup.block_duke', cls: 'block' },
    { id: BLOCK_CONTESSA, labelKey: 'coup.block_contessa', cls: 'block' },
    { id: BLOCK_AMBASSADOR, labelKey: 'coup.block_ambassador', cls: 'block' },
    { id: BLOCK_CAPTAIN, labelKey: 'coup.block_captain', cls: 'block' },
  ];

  for (const b of blocks) {
    if (!legalSet.has(b.id)) continue;
    const btn = document.createElement('button');
    btn.className = 'coup-action-btn ' + b.cls;
    btn.textContent = t(b.labelKey);
    btn.addEventListener('click', () => ctx.submitAction(b.id));
    btns.appendChild(btn);
  }

  if (legalSet.has(ALLOW_NO_BLOCK)) {
    const btn = document.createElement('button');
    btn.className = 'coup-action-btn allow';
    btn.textContent = t('coup.allow_no_block');
    btn.addEventListener('click', () => ctx.submitAction(ALLOW_NO_BLOCK));
    btns.appendChild(btn);
  }

  panel.appendChild(btns);
  return panel;
}

function renderRevealPanel(st, legalSet, humanPlayer, ctx, titleText) {
  const panel = document.createElement('div');
  panel.className = 'coup-action-panel';

  const title = document.createElement('div');
  title.className = 'coup-panel-title';
  title.textContent = titleText;
  panel.appendChild(title);

  const btns = document.createElement('div');
  btns.className = 'coup-action-btns';

  const isReveal = legalSet.has(REVEAL_0) || legalSet.has(REVEAL_1);
  const p = st.players[humanPlayer];
  const legalCardActions = [];

  for (let sl = 0; sl < 2; sl++) {
    const revealId = isReveal ? (sl === 0 ? REVEAL_0 : REVEAL_1) : (sl === 0 ? LOSE_0 : LOSE_1);
    if (!legalSet.has(revealId)) continue;
    legalCardActions.push(revealId);

    const inf = p.influences[sl];
    const btn = document.createElement('button');
    btn.className = 'coup-action-btn card-choice';
    btn.style.borderColor = CHAR_COLORS[inf.character];
    btn.textContent = charLabel(inf.character);
    btn.addEventListener('click', () => ctx.submitAction(revealId));
    btns.appendChild(btn);
  }

  panel.appendChild(btns);
  scheduleAutoSingleCardChoice(st, humanPlayer, isReveal, legalCardActions, ctx);
  return panel;
}

function scheduleAutoSingleCardChoice(st, humanPlayer, isReveal, legalCardActions, ctx) {
  if (legalCardActions.length !== 1) return;
  const actionId = legalCardActions[0];
  const key = [
    st.stage,
    st.current_player,
    st.active_player,
    humanPlayer,
    isReveal ? 'reveal' : 'lose',
    actionId,
  ].join(':');
  if (autoCardChoiceKey === key) return;

  autoCardChoiceKey = key;
  if (autoCardChoiceTimer !== null) clearTimeout(autoCardChoiceTimer);
  autoCardChoiceTimer = setTimeout(() => {
    autoCardChoiceTimer = null;
    ctx.submitAction(actionId);
  }, 180);
}

function renderExchangePanel(st, legalSet, ctx) {
  // The return-card input lives in the player-area hand (every drawn/held
  // card is clickable) — see renderPlayerArea. This panel only shows the
  // prompt and keeps the action-bar footprint stable.
  const panel = document.createElement('div');
  panel.className = 'coup-action-panel';

  const title = document.createElement('div');
  title.className = 'coup-panel-title';
  title.textContent = st.stage === 8
    ? t('coup.exchange_return_1')
    : t('coup.exchange_return_2');
  panel.appendChild(title);

  const btns = document.createElement('div');
  btns.className = 'coup-action-btns';
  const spacer = document.createElement('div');
  spacer.className = 'coup-action-btn-spacer';
  spacer.innerHTML = '&nbsp;';
  btns.appendChild(spacer);
  panel.appendChild(btns);
  return panel;
}

function renderPlayerArea(container, gs, ctx) {
  container.innerHTML = '';
  if (!gs || !gs.state) return;

  const st = gs.state;
  const humanPlayer = ctx.state.humanPlayer;
  const p = st.players[humanPlayer];

  const area = document.createElement('div');
  area.className = 'coup-my-area';
  area.setAttribute('data-player', String(humanPlayer));

  if (!p.alive) {
    const dead = document.createElement('div');
    dead.className = 'coup-dead-msg';
    dead.textContent = t('coup.you_eliminated');
    area.appendChild(dead);
    container.appendChild(area);
    return;
  }

  const coinsEl = document.createElement('div');
  coinsEl.className = 'coup-my-coins';
  coinsEl.textContent = t('coup.coins', { n: p.coins });
  area.appendChild(coinsEl);

  const hand = document.createElement('div');
  hand.className = 'coup-my-hand';

  // Exchange stages (8, 9): the active player is holding up to 2 extra
  // drawn cards alongside their 2 hand cards. Show them all in the hand
  // row so the player physically sees "4 cards, pick one to return"
  // matching the tabletop feel, instead of a separate return-button row.
  const drawn = st.exchange_drawn || [];
  const inExchange = (st.stage === 8 || st.stage === 9)
    && st.active_player === humanPlayer
    && ctx.canPlay;
  const legalSet = getLegalSet(gs);

  // Hand cards first (in-hand slots 0, 1). During exchange stage 9, a hand
  // slot may hold character=-1 if the first returned card came out of that
  // slot — skip those so we don't render a blank tile.
  for (let sl = 0; sl < 2; sl++) {
    const inf = p.influences[sl];
    if (!inf.revealed && inf.character < 0) continue;
    const card = buildMyCard(inf.character, inf.revealed, sl, 'hand');
    if (inExchange && !inf.revealed) {
      wireReturnClick(card, inf.character, legalSet, ctx);
    }
    hand.appendChild(card);
  }

  // Drawn cards after, tagged as draw-0 / draw-1 so animation selectors
  // can target them individually. Only render when WE are the one mid-
  // exchange — when an opponent is exchanging their drawn slots are
  // viz=0 to us, but the wire may still ship stale leftover bytes there
  // (DEC-003: viz=0 slots hold semantically-undefined values). Without
  // this gate those leftovers would render as extra cards in our hand.
  if (inExchange) {
    for (let di = 0; di < 2; di++) {
      const c = drawn[di];
      if (c == null || c < 0) continue;
      const card = buildMyCard(c, false, di, 'draw');
      card.classList.add('drawn');
      wireReturnClick(card, c, legalSet, ctx);
      hand.appendChild(card);
    }
  }

  area.appendChild(hand);

  if (inExchange) {
    const hint = document.createElement('div');
    hint.className = 'coup-exchange-hint';
    hint.textContent = st.stage === 8
      ? t('coup.exchange_hint_1')
      : t('coup.exchange_hint_2');
    area.appendChild(hint);
  }

  container.appendChild(area);
}

// Build one card DOM node for the human's hand/draw row. `kind` is
// 'hand' (slot index 0/1 of the real influences) or 'draw' (slot index
// 0/1 of exchange_drawn). Sets data-mycard=<kind>-<slot> so animations
// (return_card fly) can find the right node.
function buildMyCard(character, revealed, slot, kind) {
  const card = document.createElement('div');
  card.className = 'coup-my-card';
  card.setAttribute('data-mycard', kind + '-' + slot);
  if (revealed) {
    card.classList.add('revealed');
    card.textContent = charLabel(character);
    card.style.borderColor = '#555';
    return card;
  }
  card.style.borderColor = CHAR_COLORS[character];
  card.style.background = 'linear-gradient(135deg, ' + CHAR_COLORS[character] + '22, #1e1e3a)';
  const name = document.createElement('div');
  name.className = 'coup-mycard-name';
  name.textContent = charLabel(character);
  card.appendChild(name);
  return card;
}

function wireReturnClick(card, character, legalSet, ctx) {
  const returnId = RETURN_DUKE + character;
  if (!legalSet.has(returnId)) return;
  card.classList.add('returnable');
  card.addEventListener('click', () => ctx.submitAction(returnId));
}

// Map C++-side English character names (from coup_register.cpp's
// `claimed` / `character_name` fields) to localized labels for display.
const CHAR_NAME_TO_INDEX = {
  'Duke': 0,
  'Assassin': 1,
  'Captain': 2,
  'Ambassador': 3,
  'Contessa': 4,
};
function charNameLocalized(name) {
  const idx = CHAR_NAME_TO_INDEX[name];
  if (idx == null) return name || '';
  return charLabel(idx);
}

function formatMove(info, actionId, actor) {
  if (!info || !info.type) {
    if (actionId === null || actionId === undefined) return t('coup.action_start');
    return t('coup.action_unknown', { id: actionId });
  }
  switch (info.type) {
    case 'income': return t('coup.move_income');
    case 'foreign_aid': return t('coup.move_foreign_aid');
    case 'coup': return t('coup.move_coup', { target: info.target });
    case 'tax': return t('coup.move_tax');
    case 'assassinate': return t('coup.move_assassinate', { target: info.target });
    case 'steal': return t('coup.move_steal', { target: info.target });
    case 'exchange': return t('coup.move_exchange');
    case 'challenge': return t('coup.move_challenge');
    case 'allow': return t('coup.move_allow');
    case 'block': return t('coup.move_block', { role: charNameLocalized(info.claimed) });
    case 'allow_no_block': return t('coup.move_allow_no_block');
    case 'reveal': return t('coup.move_reveal', { n: info.slot + 1 });
    case 'lose_influence': return t('coup.move_lose_influence', { n: info.slot + 1 });
    case 'return_card': {
      // Coup rule: the card a player returns to the deck during Exchange
      // is private to that player. The wire ships character_name because
      // the return action_id encodes the role (kReturnDuke..kReturnContessa)
      // — but we must not display it to anyone except the actor.
      const humanPlayer = currentCtx && currentCtx.state ? currentCtx.state.humanPlayer : -1;
      if (actor != null && actor !== humanPlayer) {
        return t('coup.move_return_card_hidden');
      }
      return t('coup.move_return_card', { role: charNameLocalized(info.character_name) });
    }
    default: return t('coup.action_unknown', { id: actionId });
  }
}

function actorSelector(actor) {
  // Opponent cards (renderBoard) and the human's area (renderPlayerArea)
  // both carry data-player="<idx>", so one selector works for any actor.
  return '[data-player="' + actor + '"]';
}

function describeTransition(prevState, newState, actionInfo, actionId) {
  if (!actionInfo || !prevState || !prevState.state) return null;

  const type = actionInfo.type;
  const actor = prevState.current_player;
  const target = actionInfo.target;

  // Every action gets a head-bubble popup so info-only actions (challenge,
  // allow, block declarations, reveal, return_card) still read visually
  // — without it AI turns flash past too fast to follow. See §3.6 of
  // docs/guide/WEB_DESIGN_PRINCIPLES.md.
  const group = { type: 'group', children: [] };
  group.children.push({
    type: 'popup',
    target: actorSelector(actor),
    content: formatMove(actionInfo, actionId, actor),
    className: 'action-bubble',
    width: 220,
    height: 36,
    duration: 1500,
  });

  if (type === 'coup' || type === 'assassinate') {
    if (target != null && target >= 0) {
      group.children.push({
        type: 'highlight',
        target: '[data-player="' + target + '"]',
        className: 'anim-highlight',
        duration: 600,
      });
    }
  } else if (type === 'steal') {
    if (target != null && target >= 0) {
      group.children.push({
        type: 'fly',
        from: '[data-coins="' + target + '"]',
        to: '[data-coins="' + actor + '"]',
        createElement() {
          const el = document.createElement('div');
          el.className = 'anim-flying-token';
          el.style.background = '#eab308';
          el.style.color = '#111';
          el.textContent = '💰';
          return el;
        },
        duration: 400,
      });
    }
  } else if (type === 'reveal' || type === 'lose_influence') {
    const slot = actionInfo.slot;
    if (slot != null) {
      group.children.push({
        type: 'highlight',
        target: '[data-influence="' + actor + '-' + slot + '"]',
        className: 'anim-highlight',
        duration: 600,
      });
    }
  } else if (type === 'return_card') {
    // Ambassador returning a drawn card back to the deck. The card
    // "travels to the middle and disappears". Target is the opponent's
    // hand area for AI actors, or the human's own hand for the human;
    // the card flies from the drawn slot toward the top-of-board deck
    // indicator and fades out. The actual card identity is only visible
    // to the actor (they chose it), so opponent returns just show a
    // generic card back flying.
    const humanPlayer = currentCtx && currentCtx.state ? currentCtx.state.humanPlayer : -1;
    const isHuman = actor === humanPlayer;
    const charId = (actionInfo.character != null && actionInfo.character >= 0) ? actionInfo.character : -1;
    // Prefer the specific slot DOM node the human actually clicked (the
    // one that went from holding `charId` to empty across the transition).
    // A return can come from either a hand slot OR a drawn slot, so diff
    // both against prev state to find the origin. Fall back to the hand
    // row if we can't identify it. Opponents fly from their hand area.
    let fromSel;
    if (isHuman && charId >= 0) {
      const prevSt = prevState.state;
      const newSt = newState.state;
      const prevP = prevSt.players[actor];
      const newP = newSt.players[actor];
      let located = null;
      // Hand slots: was unrevealed + charId, now character==-1 (placeholder)
      // or revealed (shouldn't happen for return but guard anyway).
      for (let sl = 0; sl < 2; sl++) {
        const was = prevP && prevP.influences && prevP.influences[sl];
        const now = newP && newP.influences && newP.influences[sl];
        if (!was || !now) continue;
        if (!was.revealed && was.character === charId && now.character < 0) {
          located = '[data-mycard="hand-' + sl + '"]';
          break;
        }
      }
      // Drawn slots: was drawn[i]==charId, now drawn[i]==-1.
      if (!located) {
        const prevDr = prevSt.exchange_drawn || [];
        const newDr = newSt.exchange_drawn || [];
        for (let di = 0; di < 2; di++) {
          if (prevDr[di] === charId && (newDr[di] == null || newDr[di] < 0)) {
            located = '[data-mycard="draw-' + di + '"]';
            break;
          }
        }
      }
      fromSel = located
        ? (document.querySelector(located) || document.querySelector('.coup-my-hand'))
        : document.querySelector('.coup-my-hand');
    } else {
      fromSel = actorSelector(actor);
    }
    // Destination: the center of the board container — "the card goes
    // back to the middle of the table and disappears".
    const toSel = '.coup-board';
    group.children.push({
      type: 'fly',
      from: fromSel,
      to: toSel,
      width: 84,
      height: 118,
      duration: 500,
      hideFrom: isHuman,
      createElement() {
        const el = document.createElement('div');
        el.className = 'anim-flying-card coup-card-fly';
        if (isHuman && charId >= 0) {
          el.style.borderColor = CHAR_COLORS[charId];
          el.style.background = 'linear-gradient(135deg, ' + CHAR_COLORS[charId] + '55, #1e1e3a)';
          el.innerHTML = '<div style="font-size:13px; font-weight:700;">' + charLabel(charId) + '</div>';
        } else {
          el.style.background = '#2a2a4a';
          el.style.borderColor = '#555';
          el.textContent = '?';
        }
        return el;
      },
    });
  }

  const steps = [group];

  // Challenge-reveal: if the human challenged someone and the revealer is
  // not the human, surface what was shown with a blocking modal so the
  // player doesn't miss it between AI turns. See §3.7.
  //
  // Additionally, when the challenge FAILED (revealed role matches the
  // claim — prevSt.revealed stays false across the transition because
  // the rules reshuffle the shown card into the deck and draw a fresh
  // one), animate a reshuffle: the revealed character card flies to the
  // deck and a face-down replacement flies back to the revealer's slot.
  // Fires for both human- and AI-revealer paths so the player always
  // sees the draw.
  if (type === 'reveal') {
    const humanPlayer = currentCtx && currentCtx.state ? currentCtx.state.humanPlayer : -1;
    const challenger = prevState.state.challenger;
    const revealer = actor;
    const slot = actionInfo.slot;
    const prevSt = prevState.state;
    const newSt = newState.state;
    if (slot != null) {
      const pPrev = prevSt.players[revealer];
      const infPrev = pPrev && pPrev.influences && pPrev.influences[slot];
      // The shown card is public from this move on, but it was hidden
      // before it (an opponent's face-down card is not in our state), so
      // it comes from the move's public events; the pre-move state only
      // knows it for our own cards and in full-information replays.
      let revealedChar = revealedRoleFromEvents(newState, revealer, slot);
      if (revealedChar == null && infPrev && infPrev.character != null) {
        revealedChar = infPrev.character;
      }
      if (revealedChar == null) revealedChar = -1;
      const wasRevealed = infPrev ? infPrev.revealed : false;
      const nowRevealed = newSt.players[revealer].influences[slot].revealed;
      const challengeFailed = !wasRevealed && !nowRevealed; // reshuffle case
      // Modal first (blocks on user confirm), THEN the reshuffle animation —
      // otherwise the player would see the card fly away to the deck before
      // the modal even appears, defeating the "make sure they saw it" intent.
      if (challenger === humanPlayer && revealer !== humanPlayer &&
          revealedChar >= 0 && revealedChar < 5) {
        steps.push(buildChallengeRevealStep(revealer, revealedChar, prevSt));
      }
      if (challengeFailed && revealedChar >= 0 && revealedChar < 5) {
        steps.push(buildReshuffleStep(revealer, slot, revealedChar));
      }
    }
  }

  return steps;
}

// Character a challenged player showed, from the public events of the
// move (`events` on the new frame): a card turned face up for good
// (card_revealed), or a truthful claim / block, where the shown card is
// the claimed role and goes back into the deck.
function revealedRoleFromEvents(newState, revealer, slot) {
  const events = (newState && newState.events) || [];
  for (const ev of events) {
    const p = ev && ev.payload;
    if (!p) continue;
    if (ev.kind === 'card_revealed' && p.player === revealer && p.slot === slot) return p.role;
    if (ev.kind === 'claim_resolved_truthful' && p.claimer === revealer) return p.role;
    if (ev.kind === 'block_resolved_truthful' && p.blocker === revealer) return p.role;
  }
  return null;
}

// Reshuffle animation after a failed challenge: shown card flies to the
// center (deck), then a face-down replacement flies back to the slot.
// Sequential group via two `fly` steps in an outer group with `mode:
// "sequence"`. The revealer's slot DOM uses data-influence="<p>-<slot>".
function buildReshuffleStep(revealer, slot, revealedChar) {
  const slotSel = '[data-influence="' + revealer + '-' + slot + '"]';
  const deckSel = '.coup-board';
  return {
    type: 'group',
    mode: 'sequence',
    children: [
      {
        type: 'fly',
        from: slotSel,
        to: deckSel,
        width: 84,
        height: 118,
        duration: 500,
        hideFrom: true,
        createElement() {
          const el = document.createElement('div');
          el.className = 'anim-flying-card coup-card-fly';
          el.style.borderColor = CHAR_COLORS[revealedChar];
          el.style.background = 'linear-gradient(135deg, ' + CHAR_COLORS[revealedChar] + '55, #1e1e3a)';
          el.innerHTML = '<div style="font-size:13px; font-weight:700;">'
            + charLabel(revealedChar) + '</div>';
          return el;
        },
      },
      {
        type: 'fly',
        from: deckSel,
        to: slotSel,
        width: 64,
        height: 44,
        duration: 500,
        hideTo: true,
        createElement() {
          const el = document.createElement('div');
          el.className = 'anim-flying-card coup-card-fly';
          el.style.background = '#2a2a4a';
          el.style.borderColor = '#555';
          el.style.color = '#888';
          el.textContent = '?';
          return el;
        },
      },
    ],
  };
}

function buildChallengeRevealStep(revealer, revealedChar, prevSt) {
  // Decide success/failure by comparing the revealed role to whatever
  // the revealer had just claimed. In kResolveChallengeAction (stage 2)
  // the claim is declared_action's role; in kResolveChallengeCounter
  // (stage 6) the claim is the declared block's role. If they match,
  // the challenge FAILED and the challenger will lose influence.
  const claimedRole = roleClaimedByRevealer(prevSt);
  const matched = claimedRole != null && claimedRole === revealedChar;
  const body = document.createElement('div');
  body.style.cssText = 'display:flex; flex-direction:column; align-items:center; gap:10px;';

  // Render the revealed card with the same visual language as the human's
  // own hand cards (buildMyCard) so the chip is immediately recognizable
  // as a character. Wrap it to add a drop shadow matching the old chip.
  const chipWrap = document.createElement('div');
  chipWrap.style.cssText = 'filter: drop-shadow(0 6px 18px rgba(0,0,0,0.35));';
  const chip = buildMyCard(revealedChar, false, 0, 'reveal');
  chipWrap.appendChild(chip);
  body.appendChild(chipWrap);

  const caption = document.createElement('div');
  caption.style.cssText = 'font-size:14px; color:#e2e8f0; text-align:center;';
  let captionText = t('coup.challenge_caption', { n: revealer, role: charLabel(revealedChar) });
  if (claimedRole != null) {
    captionText += matched
      ? t('coup.challenge_failed_suffix')
      : t('coup.challenge_succeeded_suffix');
  }
  caption.textContent = captionText;
  body.appendChild(caption);

  return {
    type: 'reveal',
    title: t('coup.challenge_modal_title'),
    body,
  };
}

function roleClaimedByRevealer(prevSt) {
  const stage = prevSt.stage;
  // Stage 2 (kResolveChallengeAction): revealer is the actor, claim comes from declared_action.
  if (stage === 2) {
    const a = prevSt.declared_action;
    if (a === 6) return 0; // tax → Duke
    if (a >= 7 && a <= 10) return 1; // assassinate → Assassin
    if (a >= 11 && a <= 14) return 2; // steal → Captain
    if (a === 15) return 3; // exchange → Ambassador
  }
  // Stage 6 (kResolveChallengeCounter): revealer is the blocker, claim is the block role.
  if (stage === 6) {
    const a = prevSt.declared_action;
    // Block-declaration action id isn't stored separately; infer from context:
    // if the base action was foreign_aid (1), block is Duke;
    // if assassinate, block is Contessa;
    // if steal, block is Captain or Ambassador — we can't tell here without
    // storing the block choice, so leave claimed=null in that case.
    if (a === 1) return 0; // foreign_aid block → Duke
    if (a >= 7 && a <= 10) return 4; // assassinate block → Contessa
    // Steal block (Ambassador OR Captain): ambiguous from state alone; skip.
  }
  return null;
}

// Stage + deck info pill shown as the 5th info-panel row. Lives in the
// sidebar rather than the game area so the game-area layout doesn't
// jump when the stage description changes width. See
// WEB_DESIGN_PRINCIPLES §1.5 for the design rule.
const stageExtension = {
  render(el, gameState) {
    const st = gameState && gameState.state ? gameState.state : null;
    if (!st) {
      el.textContent = t('coup.stage_pill_default');
      return;
    }
    const stage = st.stage != null ? st.stage : 0;
    const deck = st.deck_size != null ? st.deck_size : t('coup.stage_dash');
    el.innerHTML = '<span class="coup-stage">' + t('coup.stage_pill_label', { name: stageName(stage) }) + '</span>'
      + '<span class="coup-deck">' + t('coup.deck_pill', { n: deck }) + '</span>';
  },
};

createApp({
  gameId: 'coup',
  gameTitle: t('coup.title'),
  gameIntro: t('coup.intro'),
  players: { min: 2, max: 4 },
  renderBoard,
  renderPlayerArea,
  describeTransition,
  formatOpponentMove: formatMove,
  formatSuggestedMove: formatMove,
  getPlayerSymbol: (p) => playerLabel(p),
  difficulties: ['heuristic', 'casual', 'expert'],
  defaultDifficulty: 'expert',
  defaultSeat: 'random',
  extensions: [stageExtension],
  // Hidden-info game (opponent's influence cards are face-down): can't
  // play for them, and AI 胜率 reads MCTS root values seeded with the
  // human's true cards, so its swings would leak the player's roles.
  disableForce: true,
  showWinrateDefault: false,
  onActionSubmitted: () => { resetPending(); },
  onGameStart: () => { resetPending(); },
  onUndo: () => { resetPending(); },
});
