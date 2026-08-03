// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Minimal hash router built on a @preact/signals signal, modeled on the
// operator UI's ``lib/router.js`` but trimmed to the routes this dashboard
// needs (no query-string layer). Path only; navigation writes ``window.location.hash``.

import { signal } from '@preact/signals';

/** Current route path signal — e.g. ``/``, ``/runs/abc``, ``/compare``. */
export const route = signal(parseHash());

function parseHash() {
  const hash = window.location.hash;
  const raw = hash.startsWith('#') ? hash.slice(1) : hash;
  const path = (raw || '/').split('?', 1)[0];
  return path || '/';
}

function syncFromHash() {
  const cur = parseHash();
  if (route.value !== cur) route.value = cur;
}

window.addEventListener('hashchange', syncFromHash);
window.addEventListener('load', syncFromHash);

function safeDecode(value) {
  try {
    return decodeURIComponent(value);
  } catch (error) {
    if (error instanceof URIError) return value;
    throw error;
  }
}

/**
 * Navigate to a path, overwriting the hash entirely.
 * @param {string} path - e.g. ``/runs/abc``
 */
export function navigate(path) {
  const target = path.startsWith('/') ? path : `/${path}`;
  if (window.location.hash !== `#${target}`) {
    window.location.hash = target;
  }
}

/** Build a hash href for an anchor. */
export function hashUrl(path) {
  return `#${path.startsWith('/') ? path : '/' + path}`;
}

/**
 * Match a ``:param`` route pattern against the current path.
 * @param {string} pattern - e.g. ``/runs/:id``
 * @param {string} path - e.g. ``/runs/abc``
 * @returns {object|null} extracted params, or null when it does not match.
 */
export function matchRoute(pattern, path) {
  const pp = pattern.split('/').filter(Boolean);
  const vp = path.split('/').filter(Boolean);
  if (pp.length !== vp.length) return null;
  const params = {};
  for (let i = 0; i < pp.length; i++) {
    if (pp[i].startsWith(':')) {
      params[pp[i].slice(1)] = safeDecode(vp[i]);
    } else if (pp[i] !== vp[i]) {
      return null;
    }
  }
  return params;
}
