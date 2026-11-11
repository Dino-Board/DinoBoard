import { t } from './i18n.js';

const API_BASE = '/api/games';

// Retry network-level failures (connection reset, middlebox timeout,
// mobile network blip). HTTP error responses are NOT retried — those
// are business logic errors that won't resolve by trying again.
async function fetchWithRetry(url, init, { retries = 3, backoffMs = 500 } = {}) {
  let lastErr;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await fetch(url, init);
    } catch (err) {
      lastErr = err;
      if (attempt === retries) break;
      await new Promise(r => setTimeout(r, backoffMs * (attempt + 1)));
    }
  }
  const msg = lastErr && lastErr.message ? lastErr.message : 'network error';
  throw new Error(t('api.network_failed', { msg }));
}

export async function apiGet(url) {
  const resp = await fetchWithRetry(url);
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.detail || resp.statusText);
  }
  return resp.json();
}

export async function apiPost(url, body) {
  const resp = await fetchWithRetry(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.detail || resp.statusText);
  }
  return resp.json();
}

export { API_BASE };
