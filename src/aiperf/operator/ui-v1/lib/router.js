import { signal, effect } from '@preact/signals';

// Current route signal - hash without the leading '#'
export const route = signal(getHash());

function getHash() {
  const hash = window.location.hash;
  // Strip leading '#', normalize empty to '/'
  const path = hash.startsWith('#') ? hash.slice(1) : hash;
  return path || '/';
}

// Listen for hash changes
window.addEventListener('hashchange', () => {
  route.value = getHash();
});

// Also capture initial load
window.addEventListener('load', () => {
  route.value = getHash();
});

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
