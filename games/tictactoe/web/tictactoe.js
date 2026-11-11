import { createApp } from '/static/general/app.js';
import { t, register } from '/static/general/i18n.js';

register({
  zh: {
    'ttt.title': '井字棋',
    'ttt.intro': '在 3x3 棋盘上先连成一条线即可获胜',
    'ttt.move_rc': '第{row}行第{col}列',
    'ttt.action_unknown': '动作 {id}',
    'ttt.action_start': '开局',
  },
  en: {
    'ttt.title': 'Tic-Tac-Toe',
    'ttt.intro': "First to make a line of 3 on the 3×3 board wins",
    'ttt.move_rc': 'row {row}, col {col}',
    'ttt.action_unknown': 'Action {id}',
    'ttt.action_start': 'Game start',
  },
});

const WIN_LINES = [
  [0,1,2],[3,4,5],[6,7,8],
  [0,3,6],[1,4,7],[2,5,8],
  [0,4,8],[2,4,6],
];

function getWinningLine(board, winner) {
  if (winner < 0) return null;
  for (const line of WIN_LINES) {
    if (line.every(i => board[i] === winner)) return line;
  }
  return null;
}

function renderBoard(container, gameState, ctx) {
  container.innerHTML = '';

  const boardEl = document.createElement('div');
  boardEl.className = 'ttt-board';

  if (!gameState) {
    for (let i = 0; i < 9; i++) {
      const btn = document.createElement('button');
      btn.className = 'ttt-cell';
      btn.disabled = true;
      boardEl.appendChild(btn);
    }
    container.appendChild(boardEl);
    return;
  }

  const board = gameState.state.board;
  const winLine = getWinningLine(board, gameState.winner);
  const gs = gameState;

  const canAct = ctx.canPlay;
  const legalSet = canAct ? new Set(gs.legal_actions) : null;

  for (let i = 0; i < 9; i++) {
    const btn = document.createElement('button');
    btn.className = 'ttt-cell';
    btn.setAttribute('data-cell', String(i));
    const val = board[i];
    if (val === 0) { btn.classList.add('x'); btn.textContent = 'X'; }
    else if (val === 1) { btn.classList.add('o'); btn.textContent = 'O'; }
    if (winLine && winLine.includes(i)) btn.classList.add('win-cell');
    if (canAct && legalSet.has(i)) {
      btn.addEventListener('click', () => ctx.submitAction(i));
    } else {
      btn.disabled = true;
    }
    boardEl.appendChild(btn);
  }

  container.appendChild(boardEl);
}

function formatMove(actionInfo, actionId) {
  if (actionInfo && actionInfo.row !== undefined) {
    return t('ttt.move_rc', { row: actionInfo.row + 1, col: actionInfo.col + 1 });
  }
  if (actionId !== null && actionId !== undefined) return t('ttt.action_unknown', { id: actionId });
  return t('ttt.action_start');
}

function describeTransition(prevState, newState, actionInfo, actionId) {
  if (actionId == null || !prevState) return null;
  return [{
    type: 'highlight',
    target: '[data-cell="' + actionId + '"]',
    className: 'anim-highlight',
    duration: 400,
  }];
}

createApp({
  gameId: 'tictactoe',
  gameTitle: t('ttt.title'),
  gameIntro: t('ttt.intro'),
  players: { min: 2, max: 2 },
  renderBoard,
  describeTransition,
  formatOpponentMove: formatMove,
  formatSuggestedMove: formatMove,
  getPlayerSymbol: (humanPlayer) => humanPlayer === 0 ? 'X' : 'O',
  difficulties: ['heuristic', 'casual', 'expert'],
  defaultDifficulty: 'expert',
  defaultSeat: 'random',
});
