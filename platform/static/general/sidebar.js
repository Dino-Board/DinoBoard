import { apiGet } from './api.js';
import { t, getLang, setLang } from './i18n.js';

const SHOW_REPLAY_KEY = 'dinoboard.showReplayPanelAlways';

function loadShowReplayAlways() {
  try {
    const v = localStorage.getItem(SHOW_REPLAY_KEY);
    // Default: true (don't hide). Only respect an explicit "0" opt-out.
    return v === null ? true : v !== '0';
  } catch (e) { return true; }
}

function saveShowReplayAlways(flag) {
  try { localStorage.setItem(SHOW_REPLAY_KEY, flag ? '1' : '0'); } catch (e) {}
}

function showWinrateKey(gameId) {
  return 'dinoboard.showWinrate.' + (gameId || 'default');
}

function loadShowWinrate(gameId, defaultFlag) {
  try {
    const v = localStorage.getItem(showWinrateKey(gameId));
    if (v === null) return defaultFlag;
    return v !== '0';
  } catch (e) { return defaultFlag; }
}

function saveShowWinrate(gameId, flag) {
  try { localStorage.setItem(showWinrateKey(gameId), flag ? '1' : '0'); } catch (e) {}
}

const REVEAL_HIDDEN_REPLAY_KEY = 'dinoboard.revealHiddenInReplay';

function loadRevealHiddenInReplay() {
  // Default ON: opening a replay is an explicit request to review the
  // game with full information; hiding cards there serves no one. Live
  // play is unaffected (the toggle is gated on state.replayMode in app.js).
  // Users can opt out via the sidebar checkbox.
  try {
    const v = localStorage.getItem(REVEAL_HIDDEN_REPLAY_KEY);
    return v === null ? true : v !== '0';
  } catch (e) { return true; }
}

function saveRevealHiddenInReplay(flag) {
  try { localStorage.setItem(REVEAL_HIDDEN_REPLAY_KEY, flag ? '1' : '0'); } catch (e) {}
}

export function createSidebar(sidebarEl, config, callbacks) {
  const difficulties = config.difficulties || ['heuristic', 'casual', 'expert'];
  const defaultDiff = config.defaultDifficulty || 'expert';
  const defaultSeat = config.defaultSeat || '0';  // '0', 'random', 'last'
  const diffLabels = {
    heuristic: t('sidebar.diff_heuristic'),
    casual: t('sidebar.diff_casual'),
    expert: t('sidebar.diff_expert'),
  };
  const players = config.players || { min: 2, max: 2 };
  const defaultNumPlayers = config.defaultNumPlayers || players.min;
  const showPlayerCount = players.max > 2;
  // Some games (e.g. simultaneous-action games like 6 nimmt!) have no
  // meaningful "seat order" — the human is always treated as one of the
  // simultaneous actors and the seat picker is misleading. When the
  // game's createApp config sets hideSeatPicker: true, the seat section
  // is hidden and the human is fixed to seat 0.
  const hideSeatPicker = !!config.hideSeatPicker;

  let playerCountHtml = '';
  if (showPlayerCount) {
    const btns = [];
    for (let n = players.min; n <= players.max; n++) {
      const active = n === defaultNumPlayers ? ' active' : '';
      btns.push(`<button class="side-btn${active}" data-players="${n}" type="button">${t('sidebar.players_n', { n })}</button>`);
    }
    playerCountHtml = `
      <label>${t('sidebar.label_players')}</label>
      <div class="player-count-grid">${btns.join('')}</div>`;
  }

  sidebarEl.innerHTML = `
    <h1>
      <a href="/" style="color: inherit; text-decoration: none;" title="${t('sidebar.title_link_title')}">DinoBoard</a>
      <button class="lang-toggle" id="lang-toggle" type="button" title="${t('sidebar.lang_btn_title')}">${t('sidebar.lang_btn')}</button>
    </h1>
    <div class="game-switcher-wrap">
      <label for="game-selector">${t('sidebar.switch_game_label')}</label>
      <select id="game-selector"></select>
    </div>
    <div class="card">
      <h2>${t('sidebar.start_card_title')}</h2>
      ${playerCountHtml}
      ${hideSeatPicker ? '' : `<label>${t('sidebar.label_seat')}</label>
      <div id="seat-section"></div>`}
      <label>${t('sidebar.label_difficulty')}</label>
      <div class="difficulty-grid">
        ${difficulties.map(d =>
          `<button class="side-btn${d === defaultDiff ? ' active' : ''}" data-diff="${d}" type="button">${diffLabels[d] || d}</button>`
        ).join('')}
      </div>
      <button id="btn-start">${t('sidebar.btn_start')}</button>
      <div id="start-msg" class="muted"></div>
    </div>
    <div class="card">
      <h2>${t('sidebar.advanced_card_title')}</h2>
      <button id="btn-undo">${t('sidebar.btn_undo')}</button>
      <div id="force-section">
        <button id="btn-force">${t('sidebar.btn_force')}</button>
      </div>
      <button id="btn-hint">${t('sidebar.btn_hint')}</button>
      <button id="btn-load-replay" data-mobile-hide>${t('sidebar.btn_load_replay')}</button>
      <input type="file" id="replay-file-input" accept=".json" style="display:none" data-mobile-hide>
      <label class="side-toggle" data-mobile-hide>
        <input type="checkbox" id="toggle-show-replay">
        <span>${t('sidebar.toggle_show_replay')}</span>
      </label>
      <label class="side-toggle">
        <input type="checkbox" id="toggle-show-winrate">
        <span>${t('sidebar.toggle_show_winrate')}</span>
      </label>
      <label class="side-toggle">
        <input type="checkbox" id="toggle-reveal-hidden-replay">
        <span>${t('sidebar.toggle_reveal_hidden_replay')}</span>
      </label>
      <div id="ops-msg" class="muted"></div>
    </div>
  `;

  // Wire language toggle (page reload swaps everything to the other lang).
  const langBtn = sidebarEl.querySelector('#lang-toggle');
  if (langBtn) {
    langBtn.addEventListener('click', () => setLang(getLang() === 'zh' ? 'en' : 'zh'));
  }

  let sideMode = '0';
  let difficulty = defaultDiff;
  let numPlayers = defaultNumPlayers;

  function rebuildSeatButtons() {
    // For simultaneous-action games, fix the human to the last seat so
    // they act after every AI has already committed — gives a natural
    // "wait for opponents, then I play, all reveal" rhythm.
    sideMode = hideSeatPicker ? 'last' : defaultSeat;
    const section = sidebarEl.querySelector('#seat-section');
    if (!section) return;  // hideSeatPicker games skip this section entirely
    const btns = [];
    for (let i = 0; i < numPlayers; i++) {
      const active = (defaultSeat === String(i) || (defaultSeat === '0' && i === 0)) ? ' active' : '';
      btns.push(`<button class="side-btn${active}" data-seat="${i}" type="button">${t('sidebar.seat_player', { n: i })}</button>`);
    }
    const randomActive = defaultSeat === 'random' ? ' active' : '';
    btns.push(`<button class="side-btn${randomActive}" data-seat="random" type="button">${t('sidebar.seat_random')}</button>`);
    section.innerHTML = `<div class="side-grid">${btns.join('')}</div>`;

    const seatBtns = section.querySelectorAll('[data-seat]');
    seatBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        seatBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        sideMode = btn.dataset.seat;
      });
    });
  }

  if (showPlayerCount) {
    const playerBtns = sidebarEl.querySelectorAll('[data-players]');
    playerBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        playerBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        numPlayers = parseInt(btn.dataset.players);
        rebuildSeatButtons();
      });
    });
  }

  rebuildSeatButtons();

  const diffBtns = sidebarEl.querySelectorAll('[data-diff]');
  diffBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      diffBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      difficulty = btn.dataset.diff;
    });
  });

  sidebarEl.querySelector('#btn-start').addEventListener('click', () => {
    callbacks.onStart(sideMode, difficulty, numPlayers);
  });
  sidebarEl.querySelector('#btn-undo').addEventListener('click', () => callbacks.onUndo());
  // #btn-force may be removed entirely on games that disable the feature
  // (rebuildForceButtons clears the section when config.disableForce).
  // Bind it here only for the default 2p layout — multiplayer rebuilds
  // re-bind from inside rebuildForceButtons.
  if (!config.disableForce) {
    sidebarEl.querySelector('#btn-force').addEventListener('click', () => callbacks.onForce());
  } else {
    const forceSection = sidebarEl.querySelector('#force-section');
    if (forceSection) forceSection.innerHTML = '';
  }
  sidebarEl.querySelector('#btn-hint').addEventListener('click', () => callbacks.onHint());

  function rebuildForceButtons(aiPlayers) {
    const section = sidebarEl.querySelector('#force-section');
    // Per-game opt-out: hidden-info games like Love Letter / Coup can't
    // expose "play for opponent" — the human doesn't see the opponent's
    // hand so any move they pick would either reveal hidden info to the
    // engine on submission or be a guess.
    if (config.disableForce) {
      section.innerHTML = '';
      return;
    }
    if (!aiPlayers || aiPlayers.length <= 1) {
      section.innerHTML = `<button id="btn-force">${t('sidebar.btn_force')}</button>`;
    } else {
      const btns = aiPlayers.map(p => {
        const label = config.getPlayerSymbol ? config.getPlayerSymbol(p) : t('app.player_n', { n: p });
        return `<button class="btn-force-player" data-force-player="${p}">${t('sidebar.btn_force_player', { label })}</button>`;
      });
      section.innerHTML = btns.join('');
    }
    section.querySelectorAll('#btn-force').forEach(btn => {
      btn.addEventListener('click', () => callbacks.onForce());
    });
    section.querySelectorAll('.btn-force-player').forEach(btn => {
      btn.addEventListener('click', () => callbacks.onForce(parseInt(btn.dataset.forcePlayer)));
    });
  }

  const fileInput = sidebarEl.querySelector('#replay-file-input');
  sidebarEl.querySelector('#btn-load-replay').addEventListener('click', () => {
    fileInput.click();
  });
  fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (!file) return;
    fileInput.value = '';
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const data = JSON.parse(reader.result);
        if ((!data.frames || !data.frames.length) && !data.action_history) {
          callbacks.onLoadReplay(null, t('sidebar.replay_no_frames'));
          return;
        }
        callbacks.onLoadReplay(data);
      } catch (e) {
        callbacks.onLoadReplay(null, t('sidebar.replay_parse_error', { msg: e.message }));
      }
    };
    reader.readAsText(file);
  });

  loadGameSwitcher(sidebarEl.querySelector('#game-selector'), config.gameId);

  const showReplayToggle = sidebarEl.querySelector('#toggle-show-replay');
  // Handler is registered after sidebar creation via onShowReplayToggle()
  // — the app needs access to `replay` (created after the sidebar) when
  // the checkbox changes. Store it in a mutable slot instead of reading
  // it from callbacks at fire time.
  let showReplayHandler = null;
  if (showReplayToggle) {
    showReplayToggle.checked = loadShowReplayAlways();
    showReplayToggle.addEventListener('change', () => {
      saveShowReplayAlways(showReplayToggle.checked);
      if (showReplayHandler) showReplayHandler(showReplayToggle.checked);
    });
  }

  // Win-rate display toggle. Hidden-info games (loveletter, coup) default
  // to OFF because the displayed value reads root_values[humanPlayer]
  // straight from the AI's MCTS root — which was searched from truth
  // including the human's own dealt hand, so sharp swings would let the
  // user infer cards. Per-game default comes from config.showWinrateDefault.
  const showWinrateToggle = sidebarEl.querySelector('#toggle-show-winrate');
  let showWinrateHandler = null;
  const winrateDefault = config.showWinrateDefault !== false;
  if (showWinrateToggle) {
    showWinrateToggle.checked = loadShowWinrate(config.gameId, winrateDefault);
    showWinrateToggle.addEventListener('change', () => {
      saveShowWinrate(config.gameId, showWinrateToggle.checked);
      if (showWinrateHandler) showWinrateHandler(showWinrateToggle.checked);
    });
  }

  // Reveal-hidden-info-in-replay toggle. Only takes effect during replay
  // playback; live play is unaffected (the human still can't see the
  // opponent's hand, drawn deck reserves, etc.). Default OFF — opt in
  // for replay study / debug. Persisted globally (not per game), since
  // the user is opting into "show me everything when reviewing".
  const revealHiddenToggle = sidebarEl.querySelector('#toggle-reveal-hidden-replay');
  let revealHiddenHandler = null;
  if (revealHiddenToggle) {
    revealHiddenToggle.checked = loadRevealHiddenInReplay();
    revealHiddenToggle.addEventListener('change', () => {
      saveRevealHiddenInReplay(revealHiddenToggle.checked);
      if (revealHiddenHandler) revealHiddenHandler(revealHiddenToggle.checked);
    });
  }

  // Enable / disable the human-controlled sidebar buttons. The start
  // button has a slightly looser gate than undo/force/hint (it's still
  // clickable before any game starts and at the terminal screen — places
  // where "begin a game" is the natural next step), so the app passes
  // both flags. Locks everywhere undo/force/hint lock plus the AI move
  // in flight — never tear down the session mid-search.
  function setHumanCanAct(canAct, canStart) {
    const btnStart = sidebarEl.querySelector('#btn-start');
    const btnUndo = sidebarEl.querySelector('#btn-undo');
    const btnHint = sidebarEl.querySelector('#btn-hint');
    if (btnStart) btnStart.disabled = !canStart;
    if (btnUndo) btnUndo.disabled = !canAct;
    if (btnHint) btnHint.disabled = !canAct;
    sidebarEl
      .querySelectorAll('#force-section button')
      .forEach(b => { b.disabled = !canAct; });
  }

  // Start: no game yet — advanced ops disabled, start button enabled.
  setHumanCanAct(false, true);

  return {
    setStartMsg(text) { sidebarEl.querySelector('#start-msg').textContent = text; },
    setOpsMsg(text) { sidebarEl.querySelector('#ops-msg').textContent = text; },
    getSideMode() { return sideMode; },
    getDifficulty() { return difficulty; },
    getNumPlayers() { return numPlayers; },
    getShowReplayAlways() {
      return showReplayToggle ? showReplayToggle.checked : loadShowReplayAlways();
    },
    onShowReplayToggle(fn) { showReplayHandler = fn; },
    getShowWinrate() {
      return showWinrateToggle
        ? showWinrateToggle.checked
        : loadShowWinrate(config.gameId, winrateDefault);
    },
    onShowWinrateToggle(fn) { showWinrateHandler = fn; },
    getRevealHiddenInReplay() {
      return revealHiddenToggle ? revealHiddenToggle.checked : loadRevealHiddenInReplay();
    },
    onRevealHiddenInReplayToggle(fn) { revealHiddenHandler = fn; },
    rebuildForceButtons,
    setHumanCanAct,
  };
}

async function loadGameSwitcher(sel, currentGameId) {
  try {
    const data = await apiGet('/api/games/available');
    const lang = getLang();
    for (const g of data.games) {
      if (!g.has_web) continue;
      const opt = document.createElement('option');
      opt.value = g.game_id;
      opt.textContent = (lang === 'en' && g.display_name_en) ? g.display_name_en : g.display_name;
      if (g.game_id === currentGameId) opt.selected = true;
      sel.appendChild(opt);
    }
    sel.addEventListener('change', () => {
      window.location.href = '/games/' + sel.value + '/';
    });
  } catch (e) { /* ignore */ }
}
