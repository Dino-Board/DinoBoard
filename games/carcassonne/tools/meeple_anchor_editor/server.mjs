import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const __filename = fileURLToPath(import.meta.url);
const TOOL_DIR = path.dirname(__filename);
const CARCASSONNE_DIR = path.resolve(TOOL_DIR, '../..');
const WEB_DIR = path.join(CARCASSONNE_DIR, 'web');
const ASSET_DIR = path.join(WEB_DIR, 'assets');
const GAME_JS = path.join(WEB_DIR, 'carcassonne.js');
const PORT = Number(process.env.PORT || 8010);

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.png': 'image/png',
};

function extractConstSource(source, name, endMarker) {
  const startMarker = `const ${name} = `;
  const start = source.indexOf(startMarker);
  if (start < 0) {
    throw new Error(`missing ${name} in carcassonne.js`);
  }
  const valueStart = start + startMarker.length;
  const end = source.indexOf(endMarker, valueStart);
  if (end < 0) {
    throw new Error(`missing end marker for ${name}`);
  }
  return source.slice(valueStart, end).trim().replace(/;$/, '');
}

function evaluateData(source) {
  const farmSource = extractConstSource(
    source,
    'FARM_GROUPS',
    ';\n\nconst TILE_MEEPLE_ANCHORS',
  );
  const anchorSource = extractConstSource(
    source,
    'TILE_MEEPLE_ANCHORS',
    ';\n\nlet selectedRotation',
  );
  const context = {};
  vm.createContext(context);
  vm.runInContext(
    `result = { FARM_GROUPS: ${farmSource}, TILE_MEEPLE_ANCHORS: ${anchorSource} };`,
    context,
  );
  return context.result;
}

function validateAnchors(anchors, farmGroups) {
  if (!Array.isArray(anchors) || anchors.length !== 24) {
    throw new Error('anchors must contain 24 tile entries');
  }
  if (!Array.isArray(farmGroups) || farmGroups.length !== 24) {
    throw new Error('farmGroups must contain 24 tile entries');
  }

  for (let tile = 0; tile < anchors.length; tile++) {
    const entry = anchors[tile];
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
      throw new Error(`tile ${tile} anchor entry must be an object`);
    }
    if (!Array.isArray(entry.featureGroups)) {
      throw new Error(`tile ${tile} featureGroups must be an array`);
    }
    if (!Array.isArray(entry.farmGroups)) {
      throw new Error(`tile ${tile} farmGroups must be an array`);
    }
    if (entry.farmGroups.length !== farmGroups[tile].length) {
      throw new Error(`tile ${tile} farmGroups length mismatch`);
    }

    for (let i = 0; i < entry.featureGroups.length; i++) {
      const group = entry.featureGroups[i];
      if (!group || typeof group !== 'object' || Array.isArray(group)) {
        throw new Error(`tile ${tile} feature group ${i} must be an object`);
      }
      if (!Array.isArray(group.sides) || !group.sides.length) {
        throw new Error(`tile ${tile} feature group ${i} needs sides`);
      }
      for (const side of group.sides) {
        if (!Number.isInteger(side) || side < 0 || side > 3) {
          throw new Error(`tile ${tile} feature group ${i} has invalid side`);
        }
      }
      validatePoint(group.point, `tile ${tile} feature group ${i}`);
    }

    for (let i = 0; i < entry.farmGroups.length; i++) {
      validatePoint(entry.farmGroups[i], `tile ${tile} farm group ${i}`);
    }

    if (Object.hasOwn(entry, 'center')) {
      validatePoint(entry.center, `tile ${tile} center`);
    }

    const allowedKeys = new Set(['featureGroups', 'farmGroups', 'center']);
    for (const key of Object.keys(entry)) {
      if (!allowedKeys.has(key)) {
        throw new Error(`tile ${tile} has unsupported key ${key}`);
      }
    }
  }
}

function validatePoint(point, label) {
  if (!Array.isArray(point) || point.length !== 2) {
    throw new Error(`${label} point must be [x, y]`);
  }
  for (const value of point) {
    if (typeof value !== 'number' || !Number.isFinite(value) ||
        value < 0 || value > 100) {
      throw new Error(`${label} point must stay within 0..100`);
    }
  }
}

function formatNumber(value) {
  const rounded = Math.round(value * 10) / 10;
  if (Object.is(rounded, -0)) return '0';
  return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
}

function formatPoint(point) {
  return `[${formatNumber(point[0])}, ${formatNumber(point[1])}]`;
}

function formatAnchors(anchors) {
  const lines = ['const TILE_MEEPLE_ANCHORS = ['];
  for (const entry of anchors) {
    lines.push('  {');
    if (entry.featureGroups.length === 0) {
      lines.push('    featureGroups: [],');
    } else if (entry.featureGroups.length === 1) {
      const group = entry.featureGroups[0];
      lines.push(
        `    featureGroups: [{ sides: [${group.sides.join(', ')}], point: ${formatPoint(group.point)} }],`,
      );
    } else {
      lines.push('    featureGroups: [');
      for (const group of entry.featureGroups) {
        lines.push(
          `      { sides: [${group.sides.join(', ')}], point: ${formatPoint(group.point)} },`,
        );
      }
      lines.push('    ],');
    }
    lines.push(`    farmGroups: [${entry.farmGroups.map(formatPoint).join(', ')}],`);
    if (Object.hasOwn(entry, 'center')) {
      lines.push(`    center: ${formatPoint(entry.center)},`);
    }
    lines.push('  },');
  }
  lines.push('];');
  return lines.join('\n');
}

async function readPayload(request) {
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > 1024 * 1024) {
      throw new Error('request body too large');
    }
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString('utf8');
}

async function sendJson(response, status, data) {
  response.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store',
  });
  response.end(JSON.stringify(data));
}

async function serveFile(response, filePath) {
  const ext = path.extname(filePath);
  const contentType = MIME_TYPES[ext];
  if (!contentType) {
    response.writeHead(415);
    response.end('Unsupported file type');
    return;
  }
  const data = await fs.readFile(filePath);
  response.writeHead(200, {
    'content-type': contentType,
    'cache-control': 'no-store',
  });
  response.end(data);
}

function resolveAssetPath(urlPath) {
  const rel = decodeURIComponent(urlPath.slice('/assets/'.length));
  const filePath = path.resolve(ASSET_DIR, rel);
  if (!filePath.startsWith(ASSET_DIR + path.sep)) {
    throw new Error('asset path escapes asset directory');
  }
  return filePath;
}

async function handleRequest(request, response) {
  const url = new URL(request.url || '/', `http://${request.headers.host || 'localhost'}`);
  try {
    if (request.method === 'GET' && (url.pathname === '/' || url.pathname === '/index.html')) {
      await serveFile(response, path.join(TOOL_DIR, 'index.html'));
      return;
    }
    if (request.method === 'GET' && url.pathname === '/editor.js') {
      await serveFile(response, path.join(TOOL_DIR, 'editor.js'));
      return;
    }
    if (request.method === 'GET' && url.pathname === '/styles.css') {
      await serveFile(response, path.join(TOOL_DIR, 'styles.css'));
      return;
    }
    if (request.method === 'GET' && url.pathname.startsWith('/assets/')) {
      await serveFile(response, resolveAssetPath(url.pathname));
      return;
    }
    if (request.method === 'GET' && url.pathname === '/api/anchors') {
      const source = await fs.readFile(GAME_JS, 'utf8');
      const data = evaluateData(source);
      await sendJson(response, 200, {
        tileCodes: 'ABCDEFGHIJKLMNOPQRSTUVWX'.split(''),
        farmGroups: data.FARM_GROUPS,
        anchors: data.TILE_MEEPLE_ANCHORS,
      });
      return;
    }
    if (request.method === 'POST' && url.pathname === '/api/anchors') {
      const body = await readPayload(request);
      const payload = JSON.parse(body);
      const source = await fs.readFile(GAME_JS, 'utf8');
      const data = evaluateData(source);
      validateAnchors(payload.anchors, data.FARM_GROUPS);

      const startMarker = 'const TILE_MEEPLE_ANCHORS = [';
      const start = source.indexOf(startMarker);
      if (start < 0) {
        throw new Error('missing TILE_MEEPLE_ANCHORS start');
      }
      const end = source.indexOf(';\n\nlet selectedRotation', start);
      if (end < 0) {
        throw new Error('missing TILE_MEEPLE_ANCHORS end');
      }
      const nextSource =
        source.slice(0, start) +
        formatAnchors(payload.anchors) +
        source.slice(end + 1);
      await fs.writeFile(GAME_JS, nextSource, 'utf8');
      await sendJson(response, 200, { ok: true });
      return;
    }

    response.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
    response.end('Not found');
  } catch (error) {
    await sendJson(response, 500, {
      error: error instanceof Error ? error.message : String(error),
    });
  }
}

const server = http.createServer((request, response) => {
  handleRequest(request, response);
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`Carcassonne meeple anchor editor: http://127.0.0.1:${PORT}/`);
});
