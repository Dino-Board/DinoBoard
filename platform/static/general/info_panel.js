import { t } from './i18n.js';

export function createInfoPanel(infoCol, config) {
  const panel = document.createElement('div');
  panel.className = 'info-panel';
  // 4 pills: turn, opponent-move, AI-winrate, AI-suggest. Each pill has a
  // fixed prefix label so the semantics are clear even when the value
  // part is empty/--. The opponent-action pill shows only what the
  // opponent last played (human's own moves don't appear here);
  // transient app messages go to sidebar.setOpsMsg.
  panel.innerHTML = `
    <div class="info-pill" id="info-turn">${t('info.turn_default')}</div>
    <div class="info-pill" id="info-opp">${t('info.opp_default')}</div>
    <div class="info-pill" id="info-winrate">${t('info.winrate_default')}</div>
    <div class="info-pill" id="info-suggest">${t('info.suggest_default')}</div>
  `;
  infoCol.appendChild(panel);

  const extensionContainer = document.createElement('div');
  extensionContainer.className = 'info-extensions';
  infoCol.appendChild(extensionContainer);

  const els = {
    turn: panel.querySelector('#info-turn'),
    opp: panel.querySelector('#info-opp'),
    winrate: panel.querySelector('#info-winrate'),
    suggest: panel.querySelector('#info-suggest'),
  };

  function formatWinrate(wr) {
    if (wr === null || wr === undefined) return t('info.dash');
    return (Math.max(0, Math.min(1, wr)) * 100).toFixed(1) + '%';
  }

  return {
    setTurn(text) { els.turn.textContent = text; },
    // No-op kept so game code calling ctx.setInfoStatus() doesn't crash
    // — the dedicated status pill was removed per UI redesign. Games
    // that need a transient prompt should route to sidebar.setOpsMsg.
    setStatus(_text) {},
    // Shows the opponent's latest action. `null`/empty resets to "--".
    setMessage(text) {
      els.opp.textContent = t('info.opp_prefix') + (text && text.length ? text : t('info.dash'));
    },
    // wr is human's predicted win rate in [0, 1] or null. proven, when truthy,
    // means a tail solver supplied this number (0 / 1 / 0.5 are exact, not
    // estimates) — the AI only adopts the tail-solve action on ProvenWin, so
    // in the live path proven=true always implies the human is on the
    // losing side (wr ≈ 0). Draw outcomes pass proven='draw'.
    setWinrate(wr, proven) {
      const base = t('info.winrate_prefix') + formatWinrate(wr);
      if (proven === 'draw') {
        els.winrate.textContent = base + t('info.tail_solved_draw_suffix');
      } else if (proven) {
        els.winrate.textContent = base + t('info.tail_solved_suffix');
      } else {
        els.winrate.textContent = base;
      }
    },
    setSuggest(text) {
      els.suggest.textContent = t('info.suggest_prefix') + (text || t('info.dash'));
    },
    setVisible(visible) {
      panel.style.display = visible ? '' : 'none';
    },
    updateExtensions(gameState, extensions) {
      extensionContainer.innerHTML = '';
      if (!extensions || !extensions.length) return;
      for (const ext of extensions) {
        const el = document.createElement('div');
        el.className = 'info-pill info-extension';
        ext.render(el, gameState);
        extensionContainer.appendChild(el);
      }
    },
    reset() {
      els.turn.textContent = t('info.turn_default');
      els.opp.textContent = t('info.opp_default');
      els.winrate.textContent = t('info.winrate_default');
      els.suggest.textContent = t('info.suggest_default');
    },
  };
}
