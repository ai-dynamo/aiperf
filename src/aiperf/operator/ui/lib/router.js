import { signal, effect } from '@preact/signals';

// Current route signal - hash path without the leading '#' and without the query string.
// The query string is exposed separately via `search` / `queryParams()` so existing
// equality checks (e.g. `currentRoute === '/compare'`) keep working when callers add
// deep-link params like `#/compare?cluster=foo`.
export const route = signal(getPath());
export const search = signal(getSearch());

function splitHash() {
  const hash = window.location.hash;
  const raw = hash.startsWith('#') ? hash.slice(1) : hash;
  const qIdx = raw.indexOf('?');
  if (qIdx === -1) return { path: raw || '/', search: '' };
  return { path: raw.slice(0, qIdx) || '/', search: raw.slice(qIdx + 1) };
}

function getPath() { return splitHash().path; }
function getSearch() { return splitHash().search; }

function syncFromHash() {
  const { path, search: s } = splitHash();
  route.value = path;
  search.value = s;
}

// Listen for hash changes
window.addEventListener('hashchange', syncFromHash);

// Also capture initial load
window.addEventListener('load', syncFromHash);

/**
 * Navigate to a path. Updates the hash, which triggers the hashchange listener.
 * @param {string} path - Path like '/jobs' or '/jobs/default/my-job'
 */
export function navigate(path) {
  window.location.hash = path.startsWith('/') ? path : `/${path}`;
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

/**
 * Parse the query string portion of the current hash route into an object.
 * Reads from the `search` signal so callers can react to hash changes.
 *
 * @returns {Object<string, string>}
 */
export function queryParams() {
  const s = search.value;
  if (!s) return {};
  const out = {};
  for (const pair of s.split('&')) {
    if (!pair) continue;
    const eq = pair.indexOf('=');
    const k = eq === -1 ? pair : pair.slice(0, eq);
    const v = eq === -1 ? '' : pair.slice(eq + 1);
    try { out[decodeURIComponent(k)] = decodeURIComponent(v.replace(/\+/g, ' ')); }
    catch { out[k] = v; }
  }
  return out;
}
