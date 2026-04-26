// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Hand-tuned dark-theme defaults for Chart.js — operator UI palette
 * aligned with src/aiperf/api/static-v2.
 *
 * Chart.js is loaded as UMD via <script> in index.html and exposed as
 * `window.Chart`. This module installs grid / tooltip / legend / animation
 * defaults onto Chart.defaults the first time it's invoked, then becomes a
 * no-op (the `_initialized` flag is module-scoped, so re-imports across
 * pages do not re-apply).
 *
 * Per-chart options still win: the caller's `options` object is merged in
 * by Chart.js on top of these defaults.
 */

let _initialized = false;

/** Canonical dataset palette — green leads (NV brand), then blue, cyan,
 *  pale-green, amber, red, pink, sky. Saturated for legibility against the
 *  --bg substrate. */
export const PALETTE = [
  '#76b900',  // accent / NV green
  '#3b82f6',  // blue
  '#26c6da',  // cyan
  '#9fe870',  // pale green
  '#ffc107',  // amber
  '#ef5350',  // red
  '#ab47bc',  // pink
  '#a0d8ff',  // sky
];

const MONO_FAMILY  = "'IBM Plex Mono', ui-monospace, monospace";
const NUMER_FAMILY = "'JetBrains Mono', monospace";

/**
 * Apply shared dark-theme defaults (idempotent) and return the passed-in
 * options object unchanged. Call once before `new Chart(ctx, options)`.
 *
 * @param {object} options - Chart.js options object; returned unmodified.
 * @returns {object} The same options object the caller passed in.
 */
export function applyChartTheme(options = {}) {
  if (!_initialized && typeof window !== 'undefined' && window.Chart) {
    const C = window.Chart;
    C.defaults.font.family = MONO_FAMILY;
    C.defaults.font.size = 10;
    C.defaults.color       = '#a7a7a7';                 // --sub
    C.defaults.borderColor = '#313131';                 // --border
    C.defaults.scale.grid.color       = 'rgba(49, 49, 49, 0.5)';
    C.defaults.scale.grid.tickColor   = 'transparent';
    C.defaults.scale.grid.borderColor = 'transparent';
    C.defaults.plugins.tooltip.backgroundColor = 'rgba(22, 22, 22, 0.96)';  // --bg-card @ 0.96
    C.defaults.plugins.tooltip.titleColor = '#eeeeee';                       // --text
    C.defaults.plugins.tooltip.bodyColor  = '#a7a7a7';                       // --sub
    C.defaults.plugins.tooltip.padding = 12;
    C.defaults.plugins.tooltip.cornerRadius = 6;                             // was 0
    C.defaults.plugins.tooltip.displayColors = false;
    C.defaults.plugins.tooltip.boxPadding = 6;
    C.defaults.plugins.tooltip.borderColor = '#313131';                      // --border
    C.defaults.plugins.tooltip.borderWidth = 1;
    C.defaults.plugins.tooltip.titleFont = { family: MONO_FAMILY,  size: 11, weight: '700' };
    C.defaults.plugins.tooltip.bodyFont  = { family: NUMER_FAMILY, size: 11 };
    C.defaults.plugins.legend.labels.usePointStyle = true;
    C.defaults.plugins.legend.labels.boxWidth = 8;
    C.defaults.plugins.legend.labels.padding = 12;
    C.defaults.plugins.legend.labels.font  = { family: MONO_FAMILY, size: 10, weight: '600' };
    C.defaults.elements.line.tension = 0.3;
    _initialized = true;
  }
  return options;
}
