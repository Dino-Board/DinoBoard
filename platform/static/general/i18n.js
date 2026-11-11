// Tiny i18n runtime. No build step, no framework — just a registry +
// localStorage-backed current-language flag. Switching language is a
// page reload (setLang); all UI is rendered fresh from t() at next paint.
//
// Usage:
//   import { t, getLang, setLang, register, tList } from '/static/general/i18n.js';
//   register({ zh: { 'foo.bar': '中文' }, en: { 'foo.bar': 'English' } });
//   t('foo.bar')                        // -> '中文' or 'English'
//   t('home.players', { min: 2, max: 4 })  // {param} interpolation
//   tList('azul.colors')                // returns array; for color/card-name lists
//
// Lookup order on a key:
//   1. current language
//   2. 'en' (fallback so partially-translated UI doesn't show blanks)
//   3. the key itself (visible "missing translation" rather than empty)

const LANG_KEY = 'dinoboard.lang';

function detectInitial() {
  try {
    const stored = localStorage.getItem(LANG_KEY);
    if (stored === 'zh' || stored === 'en') return stored;
  } catch (e) { /* ignore */ }
  const nav = (typeof navigator !== 'undefined' && navigator.language) || '';
  return nav.toLowerCase().startsWith('zh') ? 'zh' : 'en';
}

const STATE = {
  lang: detectInitial(),
  table: { zh: Object.create(null), en: Object.create(null) },
};

try {
  localStorage.setItem(LANG_KEY, STATE.lang);
} catch (e) { /* ignore */ }

if (typeof document !== 'undefined' && document.documentElement) {
  document.documentElement.lang = STATE.lang === 'zh' ? 'zh-CN' : 'en';
}

export function getLang() {
  return STATE.lang;
}

export function setLang(lang) {
  if (lang !== 'zh' && lang !== 'en') return;
  try { localStorage.setItem(LANG_KEY, lang); } catch (e) { /* ignore */ }
  if (typeof location !== 'undefined') location.reload();
}

export function register(dict) {
  if (!dict) return;
  for (const lng of Object.keys(dict)) {
    if (!STATE.table[lng]) STATE.table[lng] = Object.create(null);
    Object.assign(STATE.table[lng], dict[lng]);
  }
}

function applyParams(template, params) {
  if (!params || typeof template !== 'string') return template;
  return template.replace(/\{(\w+)\}/g, (m, k) => (k in params ? String(params[k]) : m));
}

export function t(key, params) {
  const cur = STATE.table[STATE.lang];
  let val = cur ? cur[key] : undefined;
  if (val === undefined && STATE.lang !== 'en') {
    val = STATE.table.en ? STATE.table.en[key] : undefined;
  }
  if (val === undefined) return key;
  return applyParams(val, params);
}

export function tList(key) {
  const cur = STATE.table[STATE.lang];
  let val = cur ? cur[key] : undefined;
  if (val === undefined && STATE.lang !== 'en') {
    val = STATE.table.en ? STATE.table.en[key] : undefined;
  }
  if (Array.isArray(val)) return val;
  return [];
}
