import { apiGet, API_BASE } from './api.js';

export function createPipelinePoller() {
  let polling = false;
  let cancelled = false;

  async function poll(sessionId, humanPlayer, callbacks) {
    polling = true;
    cancelled = false;
    const started = Date.now();
    let analysisDelivered = false;

    try {
      while (Date.now() - started < 45000) {
        if (cancelled) { polling = false; return; }

        const st = await apiGet(API_BASE + '/' + sessionId + '/pipeline');
        const phase = st.phase || '';

        if (!analysisDelivered && st.analysis && callbacks.onAnalysis) {
          callbacks.onAnalysis(st.analysis);
          analysisDelivered = true;
        }

        if (phase === 'analyzing' || phase === 'ai_thinking' || phase === 'queued') {
          if (callbacks.onThinking) callbacks.onThinking();
        }

        if (phase === 'error') {
          polling = false;
          if (callbacks.onError) callbacks.onError(new Error(st.error || 'pipeline error'));
          return;
        }

        if (phase === 'done' || phase === 'idle') {
          if (cancelled) { polling = false; return; }
          if (!analysisDelivered && st.analysis && callbacks.onAnalysis) {
            callbacks.onAnalysis(st.analysis);
          }
          const data = await apiGet(API_BASE + '/' + sessionId);
          if (cancelled) { polling = false; return; }
          // Prefer root_values[humanPlayer] — the per-seat value MCTS
          // tracks from the value head's N-dim output. Falls back to the
          // scalar best_value (AI-POV Q at root) for 2-player zero-sum
          // legacy nets: human's value = -ai_value so human's winrate is
          // (1 - best_value)/2.
          let humanWinrate = null;
          // 'win' | 'loss' | 'draw' | null. AI only adopts tail-solve on
          // ProvenWin (see net_mcts.cpp adoption gate), so in practice the
          // live path will only ever see 'loss' (human's POV) or 'draw'.
          // 'win' covers the symmetry in case the gate ever loosens.
          let provenForHuman = null;
          if (st.ai_stats) {
            // tail_solve_outcome encoded as TailSolveOutcome enum int:
            // 0=Unknown, 1=ProvenWin, 2=ProvenLoss, 3=ProvenDraw — perspective
            // is the AI player who just moved, so ProvenWin → human loses.
            if (st.ai_stats.tail_solved) {
              const oc = st.ai_stats.tail_solve_outcome;
              if (oc === 1) {
                humanWinrate = 0;
                provenForHuman = 'loss';
              } else if (oc === 2) {
                humanWinrate = 1;
                provenForHuman = 'win';
              } else if (oc === 3) {
                humanWinrate = 0.5;
                provenForHuman = 'draw';
              }
            }
            if (humanWinrate === null) {
              const rv = st.ai_stats.root_values;
              if (Array.isArray(rv) && humanPlayer >= 0 && humanPlayer < rv.length) {
                humanWinrate = (rv[humanPlayer] + 1) / 2;
              } else if (typeof st.ai_stats.best_value === 'number') {
                humanWinrate = (1 - st.ai_stats.best_value) / 2;
              }
              if (humanWinrate !== null) {
                humanWinrate = Math.max(0, Math.min(1, humanWinrate));
              }
            }
          }
          polling = false;
          if (callbacks.onDone) callbacks.onDone(data, humanWinrate, st, provenForHuman);
          return;
        }

        await new Promise(r => setTimeout(r, 300));
      }

      polling = false;
      const data = await apiGet(API_BASE + '/' + sessionId);
      if (callbacks.onTimeout) callbacks.onTimeout(data);
    } catch (e) {
      polling = false;
      if (callbacks.onError) callbacks.onError(e);
    }
  }

  return {
    poll,
    cancel() { cancelled = true; polling = false; },
    isPolling() { return polling; },
  };
}
