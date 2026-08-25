// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Pins document.documentElement.dataset.theme to the one theme this dashboard has.
 *
 * There is no theme switching. style.css contains no [data-theme] selector, and
 * lib/theme.js — the palette every Chart.js consumer imports — is a hardcoded
 * dark palette that reads no CSS custom properties and observes no events. A
 * second theme would therefore restyle the chrome while leaving every chart
 * dark, so the attribute is fixed rather than resolved.
 *
 * Earlier builds resolved an 'auto' / 'light' / 'dark' preference from
 * localStorage and from (prefers-color-scheme: light). That put
 * data-theme="light" on <html> for every light-OS visitor and leaked a partial
 * light palette into the dark UI. Nothing here reads a preference any more.
 */

const THEME = 'dark';

let initialized = false;

function isBrowser() {
  return typeof window !== 'undefined' && typeof document !== 'undefined';
}

export function getTheme() {
  return THEME;
}

export function initTheme() {
  if (!isBrowser() || initialized) return;
  initialized = true;
  document.documentElement.dataset.theme = THEME;
}
