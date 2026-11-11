const tileGrid = document.querySelector('#tileGrid');
const statusEl = document.querySelector('#status');
const selectionEl = document.querySelector('#selection');
const saveBtn = document.querySelector('#saveBtn');
const reloadBtn = document.querySelector('#reloadBtn');
const showLabels = document.querySelector('#showLabels');
const showGrid = document.querySelector('#showGrid');

let tileCodes = [];
let farmGroups = [];
let anchors = [];
let selected = null;
let drag = null;
let dirty = false;

function setStatus(message, kind = '') {
  statusEl.textContent = message;
  statusEl.className = `status ${kind}`.trim();
}

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value));
}

function clampCoord(value) {
  return Math.max(0, Math.min(100, value));
}

function roundCoord(value) {
  return Math.round(value * 10) / 10;
}

function pointFor(anchorRef) {
  const tile = anchors[anchorRef.tile];
  if (anchorRef.kind === 'feature') {
    return tile.featureGroups[anchorRef.index].point;
  }
  if (anchorRef.kind === 'farm') {
    return tile.farmGroups[anchorRef.index];
  }
  if (anchorRef.kind === 'center') {
    return tile.center;
  }
  throw new Error(`unknown anchor kind ${anchorRef.kind}`);
}

function labelFor(anchorRef) {
  if (anchorRef.kind === 'feature') {
    const sides = anchors[anchorRef.tile].featureGroups[anchorRef.index].sides.join('');
    return `F${sides}`;
  }
  if (anchorRef.kind === 'farm') return `G${anchorRef.index}`;
  if (anchorRef.kind === 'center') return 'C';
  throw new Error(`unknown anchor kind ${anchorRef.kind}`);
}

function titleFor(anchorRef) {
  const code = tileCodes[anchorRef.tile];
  if (anchorRef.kind === 'feature') {
    return `${anchorRef.tile}: ${code} 道路/城市 ${labelFor(anchorRef)}`;
  }
  if (anchorRef.kind === 'farm') {
    return `${anchorRef.tile}: ${code} 草地 ${labelFor(anchorRef)}`;
  }
  if (anchorRef.kind === 'center') {
    return `${anchorRef.tile}: ${code} 修道院`;
  }
  throw new Error(`unknown anchor kind ${anchorRef.kind}`);
}

function setPoint(anchorRef, x, y) {
  const point = pointFor(anchorRef);
  point[0] = roundCoord(clampCoord(x));
  point[1] = roundCoord(clampCoord(y));
  dirty = true;
  updateAnchorElement(anchorRef);
  updateSelection();
}

function updateAnchorElement(anchorRef) {
  const selector = `.anchor[data-tile="${anchorRef.tile}"][data-kind="${anchorRef.kind}"][data-index="${anchorRef.index}"]`;
  const el = tileGrid.querySelector(selector);
  if (!el) {
    throw new Error(`missing anchor element ${selector}`);
  }
  const point = pointFor(anchorRef);
  el.style.left = `${point[0]}%`;
  el.style.top = `${point[1]}%`;
}

function updateSelection() {
  tileGrid.querySelectorAll('.anchor.selected').forEach((el) => {
    el.classList.remove('selected');
  });
  if (!selected) {
    selectionEl.className = 'selection empty';
    selectionEl.textContent = '还没选中点';
    return;
  }
  const selector = `.anchor[data-tile="${selected.tile}"][data-kind="${selected.kind}"][data-index="${selected.index}"]`;
  const el = tileGrid.querySelector(selector);
  if (!el) {
    throw new Error(`missing selected anchor element ${selector}`);
  }
  el.classList.add('selected');
  const point = pointFor(selected);
  selectionEl.className = 'selection';
  selectionEl.innerHTML = `
    <div><strong>${titleFor(selected)}</strong></div>
    <div class="coord-row"><span class="coord-label">x</span><span>${point[0]}%</span></div>
    <div class="coord-row"><span class="coord-label">y</span><span>${point[1]}%</span></div>
    <div class="coord-row"><span class="coord-label">点</span><span>${labelFor(selected)}</span></div>
  `;
}

function makeAnchorElement(anchorRef) {
  const point = pointFor(anchorRef);
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = `anchor ${anchorRef.kind}`;
  btn.dataset.tile = String(anchorRef.tile);
  btn.dataset.kind = anchorRef.kind;
  btn.dataset.index = String(anchorRef.index);
  btn.style.left = `${point[0]}%`;
  btn.style.top = `${point[1]}%`;
  btn.title = titleFor(anchorRef);
  btn.textContent = labelFor(anchorRef);
  btn.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    selected = anchorRef;
    updateSelection();
    btn.setPointerCapture(event.pointerId);
    drag = {
      pointerId: event.pointerId,
      anchorRef,
      board: btn.closest('.tile-board'),
    };
    moveDrag(event);
  });
  return btn;
}

function moveDrag(event) {
  if (!drag) return;
  const rect = drag.board.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * 100;
  const y = ((event.clientY - rect.top) / rect.height) * 100;
  setPoint(drag.anchorRef, x, y);
}

function render() {
  tileGrid.innerHTML = '';
  const gridOn = showGrid.checked;
  for (let tile = 0; tile < anchors.length; tile++) {
    const card = document.createElement('article');
    card.className = 'tile-card';

    const title = document.createElement('div');
    title.className = 'tile-title';
    title.innerHTML = `<span>${tile}: ${tileCodes[tile]}</span><small>${featureCount(tile)}F / ${anchors[tile].farmGroups.length}G</small>`;
    card.appendChild(title);

    const board = document.createElement('div');
    board.className = `tile-board ${gridOn ? 'grid-on' : ''}`;

    const img = document.createElement('img');
    img.className = 'tile-img';
    img.src = `/assets/Base_Game_C2_Tile_${tileCodes[tile]}.png`;
    img.alt = tileCodes[tile];
    board.appendChild(img);

    for (let index = 0; index < anchors[tile].featureGroups.length; index++) {
      board.appendChild(makeAnchorElement({ tile, kind: 'feature', index }));
    }
    for (let index = 0; index < anchors[tile].farmGroups.length; index++) {
      board.appendChild(makeAnchorElement({ tile, kind: 'farm', index }));
    }
    if (Object.hasOwn(anchors[tile], 'center')) {
      board.appendChild(makeAnchorElement({ tile, kind: 'center', index: 0 }));
    }

    card.appendChild(board);
    tileGrid.appendChild(card);
  }
  updateSelection();
}

function featureCount(tile) {
  return anchors[tile].featureGroups.length + (Object.hasOwn(anchors[tile], 'center') ? 1 : 0);
}

async function loadAnchors() {
  setStatus('正在读取点位...');
  const response = await fetch('/api/anchors', { cache: 'no-store' });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text);
  }
  const data = await response.json();
  tileCodes = data.tileCodes;
  farmGroups = data.farmGroups;
  anchors = cloneJson(data.anchors);
  selected = null;
  dirty = false;
  render();
  setStatus('已读取，可以拖动编辑。', 'ok');
}

async function saveAnchors() {
  setStatus('正在保存...');
  const response = await fetch('/api/anchors', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ anchors }),
  });
  const body = await response.json();
  if (!response.ok || !body.ok) {
    throw new Error(body.error || '保存失败');
  }
  dirty = false;
  setStatus('已保存到 carcassonne.js。', 'ok');
}

function moveSelectedByKeyboard(event) {
  if (!selected) return;
  const delta = event.shiftKey ? 5 : 1;
  let dx = 0;
  let dy = 0;
  if (event.key === 'ArrowLeft') dx = -delta;
  else if (event.key === 'ArrowRight') dx = delta;
  else if (event.key === 'ArrowUp') dy = -delta;
  else if (event.key === 'ArrowDown') dy = delta;
  else return;

  event.preventDefault();
  const point = pointFor(selected);
  setPoint(selected, point[0] + dx, point[1] + dy);
}

window.addEventListener('pointermove', moveDrag);
window.addEventListener('pointerup', (event) => {
  if (drag && drag.pointerId === event.pointerId) {
    drag = null;
  }
});
window.addEventListener('keydown', moveSelectedByKeyboard);
window.addEventListener('beforeunload', (event) => {
  if (!dirty) return;
  event.preventDefault();
  event.returnValue = '';
});

showLabels.addEventListener('change', () => {
  document.body.classList.toggle('hide-labels', !showLabels.checked);
});
showGrid.addEventListener('change', () => {
  tileGrid.querySelectorAll('.tile-board').forEach((board) => {
    board.classList.toggle('grid-on', showGrid.checked);
  });
});
reloadBtn.addEventListener('click', () => {
  loadAnchors().catch((error) => setStatus(error.message, 'error'));
});
saveBtn.addEventListener('click', () => {
  saveAnchors().catch((error) => setStatus(error.message, 'error'));
});

loadAnchors().catch((error) => setStatus(error.message, 'error'));
