// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Theme switching: auto / light / dark, persisted to localStorage.
 *
 * "auto" follows (prefers-color-scheme: light) and updates live.
 * Sets document.documentElement.dataset.theme to the resolved value.
 * Dispatches a "themechange" CustomEvent on window after every apply
 * so consumers (e.g. the top-bar toggle glyph) can re-render.
 */

const STORAGE_KEY = 'aiperfTheme';
const VALID = ['auto', 'light', 'dark'];

let mediaQuery = null;
let mediaListener = null;
let initialized = false;

function isBrowser() {
  return typeof window !== 'undefined' && typeof document !== 'undefined';
}

export function getTheme() {
  if (!isBrowser()) return 'auto';
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    return VALID.includes(stored) ? stored : 'auto';
  } catch (_) {
    return 'auto';
  }
}

function resolveTheme(pref) {
  if (pref === 'light' || pref === 'dark') return pref;
  if (!isBrowser()) return 'dark';
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

export function getResolvedTheme() {
  return resolveTheme(getTheme());
}

function applyResolved() {
  if (!isBrowser()) return;
  const resolved = resolveTheme(getTheme());
  document.documentElement.dataset.theme = resolved;
  window.dispatchEvent(new CustomEvent('themechange', { detail: { resolved, pref: getTheme() } }));
}

export function setTheme(t) {
  if (!isBrowser()) return;
  const pref = VALID.includes(t) ? t : 'auto';
  try {
    window.localStorage.setItem(STORAGE_KEY, pref);
  } catch (_) {
    // localStorage unavailable (private mode, quota) — fall through to apply
  }
  applyResolved();
}

export function cycleTheme() {
  const order = ['auto', 'light', 'dark'];
  const current = getTheme();
  const next = order[(order.indexOf(current) + 1) % order.length];
  setTheme(next);
}

export function initTheme() {
  if (!isBrowser() || initialized) return;
  initialized = true;
  applyResolved();
  mediaQuery = window.matchMedia('(prefers-color-scheme: light)');
  mediaListener = () => {
    if (getTheme() === 'auto') applyResolved();
  };
  if (mediaQuery.addEventListener) {
    mediaQuery.addEventListener('change', mediaListener);
  } else if (mediaQuery.addListener) {
    mediaQuery.addListener(mediaListener);
  }
}
