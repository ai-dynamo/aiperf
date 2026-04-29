import { signal } from '@preact/signals';

// Current route signal — path only (no query string)
export const route = signal(parseHash().path);
// Current query string signal — object map (decoded keys/values, never null)
export const query = signal(parseHash().query);

function parseHash() {
  const hash = window.location.hash;
  const raw = hash.startsWith('#') ? hash.slice(1) : hash;
  const [path, queryStr] = (raw || '/').split('?', 2);
  return { path: path || '/', query: parseQueryString(queryStr) };
}

function parseQueryString(queryStr) {
  const out = {};
  if (!queryStr) return out;
  for (const pair of queryStr.split('&')) {
    if (!pair) continue;
    const eq = pair.indexOf('=');
    if (eq === -1) {
      out[decodeURIComponent(pair)] = '';
    } else {
      out[decodeURIComponent(pair.slice(0, eq))] = decodeURIComponent(pair.slice(eq + 1));
    }
  }
  return out;
}

function encodeQueryString(q) {
  const parts = [];
  for (const [k, v] of Object.entries(q)) {
    if (v === undefined || v === null || v === '') continue;
    parts.push(`${encodeURIComponent(k)}=${encodeURIComponent(v)}`);
  }
  return parts.length ? '?' + parts.join('&') : '';
}

function syncFromHash() {
  const cur = parseHash();
  if (route.value !== cur.path) route.value = cur.path;
  // Always replace query — shallow compare keys/values to avoid extra rerenders
  if (!shallowEq(query.value, cur.query)) query.value = cur.query;
}

function shallowEq(a, b) {
  const ak = Object.keys(a);
  const bk = Object.keys(b);
  if (ak.length !== bk.length) return false;
  for (const k of ak) {
    if (a[k] !== b[k]) return false;
  }
  return true;
}

window.addEventListener('hashchange', syncFromHash);
window.addEventListener('load', syncFromHash);

/**
 * Navigate to a path. Preserves nothing — overwrites the hash entirely.
 * Pass query string directly in `path` (e.g. '/jobs?ns=foo') if needed.
 *
 * @param {string} path - e.g. '/jobs', '/jobs?ns=default', '/jobs/default/my-job'
 */
export function navigate(path) {
  const target = path.startsWith('/') ? path : `/${path}`;
  if (window.location.hash !== `#${target}`) {
    window.location.hash = target;
  }
}

/**
 * Build a hash URL for the given path + query map. Useful for `<a href>`.
 * Empty/null/undefined values are dropped.
 *
 * @param {string} path - e.g. '/jobs'
 * @param {object} q    - e.g. { ns: 'default', phase: 'running' }
 * @returns {string}    - e.g. '#/jobs?ns=default&phase=running'
 */
export function hashUrl(path, q) {
  return `#${path}${q ? encodeQueryString(q) : ''}`;
}

/**
 * Merge updates into the current query string at the current path. Keys whose
 * value is `undefined` / `null` / `''` are removed. Other keys retain prior values.
 *
 * @param {object} updates
 */
export function setQuery(updates) {
  const cur = parseHash();
  const merged = { ...cur.query };
  for (const [k, v] of Object.entries(updates)) {
    if (v === undefined || v === null || v === '') {
      delete merged[k];
    } else {
      merged[k] = String(v);
    }
  }
  if (shallowEq(merged, cur.query)) return;
  const next = `#${cur.path}${encodeQueryString(merged)}`;
  if (window.location.hash !== next) {
    window.location.hash = next;
  }
}

/**
 * Match a route pattern against a current path.
 * Pattern params use :paramName syntax.
 * Returns null if no match, otherwise returns an object with extracted params.
 *
 * @param {string} pattern - e.g. '/jobs/:ns/:name'
 * @param {string} path - e.g. '/jobs/default/my-job'
 * @returns {object|null}
 */
export function matchRoute(pattern, path) {
  const patternParts = pattern.split('/').filter(Boolean);
  const pathParts = path.split('/').filter(Boolean);

  if (patternParts.length !== pathParts.length) return null;

  const params = {};
  for (let i = 0; i < patternParts.length; i++) {
    const pp = patternParts[i];
    const vp = pathParts[i];
    if (pp.startsWith(':')) {
      params[pp.slice(1)] = decodeURIComponent(vp);
    } else if (pp !== vp) {
      return null;
    }
  }
  return params;
}

/**
 * Build a URL for the given route pattern and params.
 * @param {string} pattern - e.g. '/jobs/:ns/:name'
 * @param {object} params - e.g. { ns: 'default', name: 'my-job' }
 * @returns {string}
 */
export function buildRoute(pattern, params = {}) {
  return pattern.replace(/:([^/]+)/g, (_, key) => encodeURIComponent(params[key] ?? ''));
}
