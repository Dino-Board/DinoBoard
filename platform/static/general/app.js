import { apiPost, apiGet, API_BASE } from './api.js';
import { buildLayout } from './layout.js';
import { createSidebar } from './sidebar.js';
import { createInfoPanel } from './info_panel.js';
import { createReplayController } from './replay.js';
import { createPipelinePoller } from './pipeline.js';
import { createModal } from './modal.js';
import { playTransition } from './animate.js';
import { t } from './i18n.js';
import './i18n_strings.js';

export function createApp(config) {
  const state = {
    sessionId: null,
    gameState: null,
    humanPlayer: 0,
    aiPlayers: [1],
    busy: false,
    forceMode: false,
    hintPending: false,
    lastAiWinrate: null,
    lastAiWinrateProven: null,  // 'win' | 'loss' | 'draw' | null
    replayMode: false,
    difficulty: null,
  };

  const refs = buildLayout();
  const poller = createPipelinePoller();

  const sidebar = createSidebar(refs.sidebar, config, {
    onStart: startGame,
    onUndo: handleUndo,
    onForce: handleForce,
    onHint: handleHint,
    onLoadReplay: handleLoadReplay,
  });

  const infoPanel = createInfoPanel(refs.infoCol, config);
  const replay = createReplayController(refs.infoCol, config);
  const modal = createModal();

  // Wire the replay-panel-visibility toggle now that `replay` exists.
  // (Registering this inside the sidebar callbacks object above would put
  // the closure in a TDZ relative to `replay`, which Safari flags at any
  // synchronous eval of that closure.)
  sidebar.onShowReplayToggle((flag) => { replay.setAlwaysVisible(flag); });
  replay.setAlwaysVisible(sidebar.getShowReplayAlways());

  // Replay autoplay locks the start button — refresh it when autoplay
  // starts / pauses so the lock follows replay state without a full
  // render cycle.
  replay.onPlayStateChange(() => { updateSidebarButtons(); });

  // Win-rate visibility — re-render so the pill flips immediately when
  // the user toggles the checkbox mid-game (no need to wait for next move).
  sidebar.onShowWinrateToggle(() => { render(); });

  // Reveal-hidden-info-in-replay — re-render the current replay frame so
  // opponent hands / face-down reserves flip on/off without scrubbing.
  // No effect during live play (revealHidden in ctx is gated on replayMode).
  sidebar.onRevealHiddenInReplayToggle(() => {
    if (state.replayMode) replay.rerender();
  });

  modal.onReplay(() => enterReplay());
  modal.onRestart(() => {
    startGame(sidebar.getSideMode(), sidebar.getDifficulty(), sidebar.getNumPlayers());
  });
  replay.onExit(() => exitReplay());
  replay.onRenderFrame(async (frame, allFrames, prevFrame) => {
    if (config.onReplayFrames) config.onReplayFrames(allFrames);
    // Show the position BEFORE the analyzed move (frame[i] stores the
    // state AFTER action i), so the user can compare their own choice at
    // that vantage point with the AI suggestion in the analysis card.
    // The side panel still describes `frame` itself.
    //
    // Special case: the synthetic terminal-display frame appended by the
    // replay controller is itself a copy of the real last frame — it
    // displays the post-final-move position (the actual terminal layout
    // that no other replay step exposes). Skip the "show previous" shift
    // so it renders the post-state, not the pre-state of itself.
    const idx = allFrames.indexOf(frame);
    const displayFrame = frame.__terminal_display
        ? frame
        : (idx > 0 ? allFrames[idx - 1] : frame);
    const prevIdx = prevFrame ? allFrames.indexOf(prevFrame) : -1;
    const prevDisplay = prevFrame && prevFrame.__terminal_display
        ? prevFrame
        : (prevIdx > 0 ? allFrames[prevIdx - 1]
            : (prevIdx === 0 ? prevFrame : null));
    // Animate when the displayed positions are exactly one ply apart
    // (sequential playback / next button). The action that connects them
    // is whatever produced `displayFrame` — i.e. the move being analyzed
    // on the SOURCE step. So pressing "next" while at step k animates
    // action k playing out, then lands on step k+1's pre-action position.
    const prevPly = prevDisplay && prevDisplay.ply_index;
    const curPly = displayFrame && displayFrame.ply_index;
    const adjacent = prevDisplay && typeof prevPly === 'number'
        && typeof curPly === 'number' && curPly - prevPly === 1;
    if (adjacent && displayFrame.action_info && config.describeTransition) {
      await animateTransition(prevDisplay, displayFrame, displayFrame.action_info, displayFrame.action_id, { autoplay: replay.isPlaying() });
    }
    if (config.renderBoard) {
      config.renderBoard(refs.boardCol, displayFrame, ctx);
    }
    if (config.renderPlayerArea) {
      config.renderPlayerArea(refs.playerArea, displayFrame, ctx);
    }
    updateReplayInfo(frame);
  });

  const ctx = {
    get state() { return state; },
    get canPlay() {
      if (!state.gameState || state.gameState.is_terminal) return false;
      if (state.busy || poller.isPolling()) return false;
      return !state.aiPlayers.includes(state.gameState.current_player) || state.forceMode;
    },
    // Replay-only flag: when true, game frontends should render
    // opponent-private slots (LL hand, Splendor face-down reserved cards,
    // etc.) using the truth values present in the replay's state dict.
    // The replay's state dict already carries truth (the replay rebuilds
    // from action_history through a single GameSession that holds GT),
    // and live play is structurally unaffected because this is gated on
    // state.replayMode AND the user opt-in toggle.
    get revealHidden() {
      return state.replayMode && sidebar.getRevealHiddenInReplay();
    },
    submitAction(actionId) { onAction(actionId); },
    rerender() { render(); },
    // Game-provided transient status line that appears in the info panel
    // (position 2). Pass null/empty to hide. Used for "已选择 XX" prompts
    // etc. — the info panel reserves a hide-when-empty slot so calling
    // this doesn't shift other UI elements.
    setInfoStatus(text) { infoPanel.setStatus(text); },
  };

  // Advanced ops (undo / force / hint) require the human's turn during
  // a live game — replay, terminal, no-game, busy, AI's turn all lock.
  // `forceMode` counts as human's turn (we're making a move as the opp).
  function canHumanInteract() {
    if (state.replayMode) return false;
    if (!state.gameState || state.gameState.is_terminal) return false;
    if (state.busy || poller.isPolling()) return false;
    return !state.aiPlayers.includes(state.gameState.current_player) || state.forceMode;
  }

  // Start button: lockable everywhere `canHumanInteract` locks EXCEPT
  // idle (no game), terminal, and paused replay — all natural moments
  // to open a new game. Mid-AI-search and replay autoplay stay locked
  // (autoplay state lives inside replay.js, surfaced via isPlaying()).
  function canStartGame() {
    if (state.busy || poller.isPolling()) return false;
    if (state.replayMode) return !replay.isPlaying();
    if (!state.gameState) return true;
    if (state.gameState.is_terminal) return true;
    return !state.aiPlayers.includes(state.gameState.current_player) || state.forceMode;
  }

  function updateSidebarButtons() {
    sidebar.setHumanCanAct(canHumanInteract(), canStartGame());
  }

  function render() {
    if (state.replayMode) {
      updateSidebarButtons();
      return;
    }

    if (config.renderBoard) {
      config.renderBoard(refs.boardCol, state.gameState, ctx);
    }
    if (config.renderPlayerArea) {
      config.renderPlayerArea(refs.playerArea, state.gameState, ctx);
    }

    updateInfoPanel();
    updateSidebarButtons();

    if (config.extensions) {
      infoPanel.updateExtensions(state.gameState, config.extensions);
    }
  }

  async function animateTransition(prevState, newState, actionInfo, actionId, options) {
    if (!config.describeTransition) return;
    try {
      const steps = config.describeTransition(prevState, newState, actionInfo, actionId);
      if (steps && steps.length) await playTransition(steps, options);
    } catch (e) {
      console.warn('[app] animation failed, falling back to instant render:', e);
    }
  }

  // Display label for a seat index. We're a generic platform — the seat
  // could be human or AI. The human's own seat shows "你"/"You" since
  // that's the natural self-reference in the current single-user-vs-AI
  // deployment.
  function playerLabel(idx) {
    if (idx === state.humanPlayer) return t('app.player_self');
    return t('app.player_n', { n: idx });
  }

  function terminalPlayerLabel(idx) {
    if (idx === state.humanPlayer) return t('app.player_self');
    return config.getPlayerSymbol ? config.getPlayerSymbol(idx) : t('app.player_n', { n: idx });
  }

  function playerListLabel(players) {
    return players.map(terminalPlayerLabel).join(t('app.player_list_sep'));
  }

  function terminalResult(gs) {
    const victoryMode = gs.victory_mode || 'winner_takes';
    const winners = Array.isArray(gs.winners) ? gs.winners.slice() : [];
    const losers = Array.isArray(gs.losers) ? gs.losers.slice() : [];

    if (victoryMode === 'loser_pays') {
      if (losers.length === 0) return { kind: 'draw' };
      return { kind: 'losers', labels: playerListLabel(losers) };
    }

    if (winners.length === 0 && gs.winner >= 0) winners.push(gs.winner);
    if (winners.length === 0) return { kind: 'draw' };
    return { kind: 'winners', labels: playerListLabel(winners) };
  }

  function updateInfoPanel() {
    if (!state.gameState) {
      infoPanel.reset();
      return;
    }

    const gs = state.gameState;

    if (gs.is_terminal) {
      infoPanel.setTurn(t('app.terminal_turn'));
      const result = terminalResult(gs);
      let resultText = t('app.terminal_draw_msg');
      if (result.kind === 'winners') {
        resultText = t('app.terminal_winners_msg', { labels: result.labels });
      } else if (result.kind === 'losers') {
        resultText = t('app.terminal_losers_msg', { labels: result.labels });
      }
      sidebar.setOpsMsg(resultText);
      infoPanel.setWinrate(
        sidebar.getShowWinrate() ? state.lastAiWinrate : null,
        sidebar.getShowWinrate() ? state.lastAiWinrateProven : null);
      showGameOverModal();
      return;
    }

    if (poller.isPolling()) {
      infoPanel.setTurn(t('info.turn_prefix') + playerLabel(gs.current_player) + t('app.thinking_suffix'));
    } else if (state.forceMode) {
      infoPanel.setTurn(t('app.force_mode_label'));
    } else {
      infoPanel.setTurn(t('info.turn_prefix') + playerLabel(gs.current_player));
    }

    if (gs.last_action_info && config.formatOpponentMove) {
      infoPanel.setMessage(config.formatOpponentMove(gs.last_action_info, gs.last_action_id, gs.last_action_actor));
    }

    infoPanel.setWinrate(
      sidebar.getShowWinrate() ? state.lastAiWinrate : null,
      sidebar.getShowWinrate() ? state.lastAiWinrateProven : null);
  }

  function updateReplayInfo(frame) {
    if (frame.__terminal_display) {
      infoPanel.setTurn(t('app.replay_turn_terminal'));
      infoPanel.setMessage(t('app.replay_terminal_msg'));
      infoPanel.setWinrate(null);
      infoPanel.setSuggest(null);
      return;
    }
    const actor = resolveActorName(frame.actor);
    infoPanel.setTurn(t('app.replay_turn', { actor }));

    const actorIdx = (typeof frame.actor === 'string' && frame.actor.startsWith('player_'))
      ? parseInt(frame.actor.slice('player_'.length), 10) : null;
    let moveText = frame.actor === 'start' ? t('app.replay_move_start') : (
      config.formatOpponentMove
        ? config.formatOpponentMove(frame.action_info, frame.action_id, actorIdx)
        : (frame.action_id !== null && frame.action_id !== undefined
            ? t('app.replay_action_unknown', { id: frame.action_id })
            : '')
    );
    if (frame.is_terminal) moveText += ' ' + t('app.terminal_suffix_dot');
    infoPanel.setMessage(moveText);

    const a = frame.analysis;
    if (a) {
      infoPanel.setWinrate(a.best_win_rate);
      if (frame.action_id !== a.best_action && a.best_action_info) {
        const bestText = config.formatSuggestedMove
          ? config.formatSuggestedMove(a.best_action_info, a.best_action, actorIdx)
          : t('app.replay_action_unknown', { id: a.best_action });
        infoPanel.setSuggest(bestText);
      } else {
        infoPanel.setSuggest(t('app.replay_optimal'));
      }
    } else {
      infoPanel.setWinrate(null);
      infoPanel.setSuggest(null);
    }
  }

  function showGameOverModal() {
    const gs = state.gameState;
    const showReplay = state.difficulty === 'expert';
    if (config.customGameOverModal) {
      const result = config.customGameOverModal(gs, state.humanPlayer);
      if (result) {
        modal.show(result.title, result.text, showReplay);
        return;
      }
    }
    const result = terminalResult(gs);
    if (result.kind === 'draw') {
      modal.show(t('app.modal_draw_title'), t('app.modal_draw_text'), showReplay);
    } else if (result.kind === 'winners') {
      modal.show(t('app.modal_winners_title', { labels: result.labels }), t('app.modal_win_text'), showReplay);
    } else {
      modal.show(t('app.modal_losers_title', { labels: result.labels }), t('app.modal_loss_text'), showReplay);
    }
  }

  async function startGame(sideMode, difficulty, numPlayers) {
    exitReplay();
    modal.hide();

    numPlayers = numPlayers || 2;
    let humanPlayer;
    if (sideMode === 'random') {
      humanPlayer = Math.floor(Math.random() * numPlayers);
    } else if (sideMode === 'last') {
      // Used by simultaneous-action games (hideSeatPicker) — human takes
      // the last seat so every AI commits first.
      humanPlayer = numPlayers - 1;
    } else {
      humanPlayer = parseInt(sideMode) || 0;
    }

    const aiPlayers = [];
    for (let i = 0; i < numPlayers; i++) {
      if (i !== humanPlayer) aiPlayers.push(i);
    }

    state.humanPlayer = humanPlayer;
    state.aiPlayers = aiPlayers;
    state.difficulty = difficulty;
    state.forceMode = false;
    state.lastAiWinrate = null;
    state.lastAiWinrateProven = null;
    state.busy = false;
    poller.cancel();

    // Reset the info panel up-front so a previous game's "对手动作 / 提示
    // / 胜率" pills don't bleed into the new game. updateInfoPanel won't
    // re-clear them later because gameState is non-null after the create
    // call, so we have to do it here. The gameIntro then becomes the
    // initial guidance until the opponent's first move arrives.
    infoPanel.reset();

    try {
      const data = await apiPost(API_BASE, {
        game_id: config.gameId,
        seed: Math.floor(Math.random() * 1000000),
        human_player: humanPlayer,
        num_players: numPlayers,
        difficulty: difficulty,
      });

      state.sessionId = data.session_id;
      state.gameState = data;
      state.humanPlayer = data.human_player;
      state.aiPlayers = data.ai_players;
      if (config.onGameStart) config.onGameStart();
      sidebar.rebuildForceButtons(state.aiPlayers);

      const diffLabels = {
        heuristic: t('sidebar.diff_heuristic'),
        casual: t('sidebar.diff_casual'),
        expert: t('sidebar.diff_expert'),
      };
      const seatLabel = config.getPlayerSymbol ? config.getPlayerSymbol(humanPlayer) : t('app.player_n', { n: humanPlayer });
      sidebar.setStartMsg(t('app.start_msg', {
        n: numPlayers,
        seat: seatLabel,
        diff: diffLabels[difficulty] || difficulty,
      }));
      // Brief operation hint shown until the opponent makes the first
      // move — gives a fresh restart some context instead of a blank pill.
      // Cleared on the first opp action / undo / force.
      sidebar.setOpsMsg(config.gameIntro || '');

      render();

      if (state.aiPlayers.includes(state.gameState.current_player)) {
        await apiPost(API_BASE + '/' + state.sessionId + '/ai-action', {});
        await pollPipeline();
      }
    } catch (e) {
      sidebar.setStartMsg(e.message);
    }
  }

  async function onAction(actionId) {
    if (state.busy || state.replayMode || poller.isPolling()) return;
    if (!state.gameState || state.gameState.is_terminal) return;

    const cp = state.gameState.current_player;
    const canAct = !state.aiPlayers.includes(cp) || state.forceMode;
    if (!canAct) return;
    if (!state.gameState.legal_actions.includes(actionId)) return;

    poller.cancel();
    state.hintPending = false;
    infoPanel.setSuggest(null);
    state.busy = true;
    updateSidebarButtons();
    if (config.onActionSubmitted) config.onActionSubmitted();
    try {
      const prevState = state.gameState;
      const data = await apiPost(API_BASE + '/' + state.sessionId + '/action', { action_id: actionId });

      await animateTransition(prevState, data, data.action_info, actionId);

      state.gameState = data;
      // Human just moved — clear any stashed AI last-action so the
      // message pill doesn't keep showing the previous AI move while
      // it's our turn / AI is thinking.
      state.gameState.last_action_info = null;
      state.gameState.last_action_id = null;
      state.gameState.last_action_actor = null;

      if (state.forceMode) {
        state.forceMode = false;
        state.busy = false;
        render();
        sidebar.setOpsMsg(t('app.force_done'));
        return;
      }

      state.busy = false;
      render();

      if (state.gameState.is_terminal && state.difficulty === 'expert') {
        await pollPipeline();
      } else if (!state.gameState.is_terminal && state.aiPlayers.includes(state.gameState.current_player)) {
        await pollPipeline();
      }
    } catch (e) {
      sidebar.setOpsMsg(t('app.error_prefix', { msg: e.message }));
      state.busy = false;
      updateSidebarButtons();
    }
  }

  function pollOnce() {
    const thinkingLabel = () => {
      const cp = state.gameState ? state.gameState.current_player : -1;
      return t('info.turn_prefix') + playerLabel(cp) + t('app.thinking_suffix');
    };
    return new Promise(resolve => {
      infoPanel.setTurn(thinkingLabel());
      updateSidebarButtons();
      poller.poll(state.sessionId, state.humanPlayer, {
        onThinking() {
          infoPanel.setTurn(thinkingLabel());
        },
        onAnalysis(analysis) {
          // Drop-score warnings are derived from the same root_values
          // that drive the win-rate pill — both leak hidden info on
          // games where the opponent's hand changes the AI's value
          // estimate sharply. Gate them behind the same toggle.
          if (analysis && state.difficulty === 'expert' && sidebar.getShowWinrate()) {
            const drop = analysis.drop_score;
            if (drop !== undefined && drop !== null && drop >= 5) {
              const label = drop >= 10 ? t('app.blunder_label') : t('app.mistake_label');
              sidebar.setOpsMsg(t('app.mistake_msg', { label, drop: drop.toFixed(1) }));
            }
          }
        },
        onDone(data, humanWinrate, pipeStatus, provenForHuman) {
          resolve({
            data,
            humanWinrate,
            humanWinrateProven: provenForHuman || null,
            aiAction: pipeStatus ? pipeStatus.ai_action : null,
            aiActionInfo: pipeStatus ? pipeStatus.ai_action_info : null,
            aiEvents: pipeStatus ? pipeStatus.ai_events : null,
          });
        },
        onTimeout(data) {
          state.gameState = data;
          sidebar.setOpsMsg(t('app.opp_thinking_timeout'));
          render();
          resolve(null);
        },
        onError(e) {
          sidebar.setOpsMsg(t('app.error_prefix', { msg: e.message }));
          apiGet(API_BASE + '/' + state.sessionId).then(data => {
            state.gameState = data;
            render();
          }).catch(() => {});
          resolve(null);
        },
      });
    });
  }

  async function pollPipeline() {
    while (true) {
      const result = await pollOnce();
      if (!result) break;

      const prevState = state.gameState;
      // Actor of the just-finished AI move = whoever was current before the
      // move. Some games (Love Letter) need actor in formatMove to detect
      // self-target no-op fallbacks.
      const aiActor = prevState ? prevState.current_player : null;
      // Update the opponent-action pill BEFORE animation starts so the
      // player can read what the opponent did while watching the move
      // animate (instead of discovering it only after the animation
      // finishes and the state re-renders).
      if (result.aiActionInfo && config.formatOpponentMove) {
        infoPanel.setMessage(config.formatOpponentMove(result.aiActionInfo, result.aiAction, aiActor));
      }
      // Public events of the AI move (same shape as replay frames'
      // `events` and the /action response): games read them from the new
      // frame when the action id itself is a hidden stand-in.
      result.data.events = result.aiEvents || [];
      await animateTransition(prevState, result.data, result.aiActionInfo, result.aiAction);

      state.gameState = result.data;
      // Attach AI's last action so updateInfoPanel can render it via
      // formatOpponentMove. The backend's /session endpoint doesn't
      // include last_action_info itself; we stitch it in client-side
      // from the pipeline's ai_action_info so the info panel's message
      // pill updates each AI move.
      state.gameState.last_action_info = result.aiActionInfo;
      state.gameState.last_action_id = result.aiAction;
      state.gameState.last_action_actor = aiActor;
      // Surface the AI's just-completed search value to the human as a
      // winrate pill, in BOTH casual and expert difficulty. Casual skips
      // the analysis pipeline (drop-score, smart-hint) but the AI itself
      // still ran MCTS to pick its move, so root_values[humanPlayer] is
      // sitting right there — zero extra search cost. Expert keeps the
      // same behavior; the analysis path doesn't feed this slot.
      state.lastAiWinrate = result.humanWinrate;
      state.lastAiWinrateProven = result.humanWinrateProven || null;
      // Clear the start-of-game intro once the opponent has moved —
      // info panel "对手动作" pill now carries the live message and
      // the ops-msg slot is free for transient prompts (失误, etc).
      sidebar.setOpsMsg('');
      render();

      if (state.gameState.is_terminal) break;
      if (!state.aiPlayers.includes(state.gameState.current_player)) break;

      const aiPause = config.aiStepPause || 200;
      await new Promise(r => setTimeout(r, aiPause));
      await apiPost(API_BASE + '/' + state.sessionId + '/ai-action', {}).catch(() => {});
    }
  }

  async function handleUndo() {
    if (!state.sessionId || state.busy || state.replayMode) return;
    if (state.gameState.is_terminal) {
      sidebar.setOpsMsg(t('app.cant_undo_terminal'));
      return;
    }

    poller.cancel();
    state.hintPending = false;
    infoPanel.setSuggest(null);
    state.busy = true;
    updateSidebarButtons();
    sidebar.setOpsMsg('');
    try {
      const humanTag = 'player_' + state.humanPlayer;
      let attempts = 0;
      while (attempts < 20) {
        const data = await apiPost(API_BASE + '/' + state.sessionId + '/step-back', {});
        state.gameState = data;
        attempts++;
        // An opponent holding hidden cards has options the server does not
        // send (legal_actions_hidden): an empty list there is not a dead end.
        if (!data.legal_actions.length && !data.legal_actions_hidden) break;
        if (state.aiPlayers.includes(data.current_player)) continue;
        if (data.last_actor === humanTag) continue;
        break;
      }
      state.forceMode = false;
      state.lastAiWinrate = null;
      state.lastAiWinrateProven = null;
      if (config.onUndo) config.onUndo();
      state.busy = false;
      render();
      sidebar.setOpsMsg(t('app.undone'));
    } catch (e) {
      sidebar.setOpsMsg(e.message);
      state.busy = false;
      updateSidebarButtons();
    }
  }

  async function handleForce(targetPlayer) {
    if (!state.sessionId || state.busy || state.replayMode) return;
    if (state.gameState.is_terminal) {
      sidebar.setOpsMsg(t('app.cant_force_terminal'));
      return;
    }

    poller.cancel();
    state.hintPending = false;
    infoPanel.setSuggest(null);
    state.busy = true;
    updateSidebarButtons();
    sidebar.setOpsMsg('');
    try {
      let attempts = 0;
      while (attempts < 20) {
        const data = await apiPost(API_BASE + '/' + state.sessionId + '/step-back', {});
        state.gameState = data;
        attempts++;
        if (!data.legal_actions.length && !data.legal_actions_hidden) break;
        const isTarget = targetPlayer !== undefined
          ? data.current_player === targetPlayer
          : state.aiPlayers.includes(data.current_player);
        if (!isTarget) continue;
        const actorTag = 'player_' + data.current_player;
        if (data.last_actor === actorTag) continue;
        break;
      }
      const cp = state.gameState.current_player;
      const reached = targetPlayer !== undefined ? cp === targetPlayer : state.aiPlayers.includes(cp);
      if (reached && state.gameState.legal_actions_hidden) {
        // That opponent holds cards we cannot see: its options are not
        // ours to pick. Hand the turn back to the AI.
        state.busy = false;
        render();
        sidebar.setOpsMsg(t('app.force_hidden'));
        await apiPost(API_BASE + '/' + state.sessionId + '/ai-action', {});
        await pollPipeline();
      } else if (reached) {
        state.forceMode = true;
        state.busy = false;
        render();
        const name = config.getPlayerSymbol ? config.getPlayerSymbol(cp) : t('app.player_n', { n: cp });
        // Brief flow hint — user already clicked the button, so just
        // tell them what's expected next. (Re-undo cancels.)
        sidebar.setOpsMsg(t('app.force_hint', { label: name }));
      } else {
        state.busy = false;
        sidebar.setOpsMsg(t('app.force_unreachable'));
      }
    } catch (e) {
      sidebar.setOpsMsg(e.message);
      state.busy = false;
      updateSidebarButtons();
    }
  }

  async function handleHint() {
    if (!state.sessionId || state.busy || state.replayMode) return;
    if (state.gameState.is_terminal) {
      sidebar.setOpsMsg(t('app.cant_force_terminal'));
      return;
    }
    if (state.hintPending || poller.isPolling()) return;

    state.hintPending = true;
    infoPanel.setSuggest(t('app.hint_pending'));
    try {
      const data = await apiPost(API_BASE + '/' + state.sessionId + '/ai-hint', {});
      if (!state.hintPending) return;
      // Hint is for the player whose turn it currently is — usually the human.
      const hintActor = state.gameState ? state.gameState.current_player : null;
      const text = config.formatSuggestedMove
        ? config.formatSuggestedMove(data.action_info, data.action, hintActor)
        : t('replay.action_unknown', { id: data.action });
      infoPanel.setSuggest(text);
    } catch (e) {
      if (!state.hintPending) return;
      infoPanel.setSuggest('--');
      sidebar.setOpsMsg(e.message);
    }
    state.hintPending = false;
  }

  async function enterReplay() {
    if (!state.sessionId) return;
    modal.hide();
    try {
      state.replayMode = true;
      updateSidebarButtons();
      replayMeta = null;
      replay.setResolveActor(resolveActorName);
      const data = await replay.enter(state.sessionId);
      replayMeta = extractPlayersMeta(data);
    } catch (e) {
      sidebar.setOpsMsg(t('app.replay_load_failed', { msg: e.message }));
      state.replayMode = false;
      updateSidebarButtons();
    }
  }

  function exitReplay() {
    if (!state.replayMode) return;
    replay.exit();
    state.replayMode = false;
    render();
  }

  let replayMeta = null;

  function resolveActorName(actor) {
    if (actor === 'start') return t('replay.action_start');
    if (replayMeta) {
      const info = replayMeta[actor];
      if (info) return info.name || actor;
    }
    return actor;
  }

  function extractPlayersMeta(data) {
    if (data.players) return data.players;
    if (data.player_0 && typeof data.player_0 === 'string') {
      return {
        player_0: { name: data.player_0, type: 'unknown' },
        player_1: { name: data.player_1, type: 'unknown' },
      };
    }
    return null;
  }

  function buildReplayHeader(meta) {
    if (!meta) return t('app.replay_default_header');
    const names = [];
    for (let i = 0; ; i++) {
      const key = 'player_' + i;
      if (!meta[key]) break;
      names.push(meta[key].name || key);
    }
    return names.length > 0 ? names.join(' vs ') : t('app.replay_default_header');
  }

  async function handleLoadReplay(data, error) {
    if (error) {
      sidebar.setOpsMsg(error);
      return;
    }

    replayMeta = extractPlayersMeta(data);
    replay.setResolveActor(resolveActorName);

    const header = buildReplayHeader(replayMeta);
    sidebar.setOpsMsg(header);

    if (data.frames) {
      modal.hide();
      state.replayMode = true;
      updateSidebarButtons();
      replay.enterWithFrames(data.frames);
    } else if (data.action_history) {
      sidebar.setOpsMsg(header + t('app.replay_building_suffix'));
      try {
        const resp = await fetch('/api/replay/build', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            game_id: data.game_id,
            seed: data.seed,
            action_history: data.action_history,
          }),
        });
        if (!resp.ok) throw new Error(t('app.replay_build_failed_status', { status: resp.status }));
        const built = await resp.json();
        modal.hide();
        state.replayMode = true;
        updateSidebarButtons();
        replay.enterWithFrames(built.frames);
        sidebar.setOpsMsg(header);
      } catch (e) {
        sidebar.setOpsMsg(t('app.replay_build_failed', { msg: e.message }));
      }
    } else {
      sidebar.setOpsMsg(t('app.replay_format_error'));
    }
  }

  render();
}
