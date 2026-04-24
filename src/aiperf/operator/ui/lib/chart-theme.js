/**
 * Hand-tuned dark-theme defaults for Chart.js — MCC palette.
 *
 * Chart.js is loaded as UMD via <script> in index.html and exposed as
 * ``window.Chart``. This module installs our grid / tooltip / legend /
 * animation defaults onto ``Chart.defaults`` the first time it is invoked,
 * then becomes a no-op on subsequent calls (the ``_initialized`` flag is
 * module-scoped, so re-imports across pages do not re-apply).
 *
 * Per-chart options still win: the callers' ``options`` object is merged
 * in by Chart.js on top of these defaults, so any explicit
 * ``scales.y.grid.color`` / ``plugins.tooltip.backgroundColor`` / etc.
 * at the callsite continues to override.
 */

let _initialized = false;

/** Canonical dataset palette — MCC: phosphor-amber first, cyan second, then
 *  green/red/purple for outlier / tertiary series. Fully saturated so lines
 *  punch against the near-black substrate. */
export const PALETTE = [
  '#76b900', '#7eeaff', '#9fe870', '#ff5964',
  '#c4a5ff', '#f4eede', '#8ce200', '#a0d8ff',
];

const MONO_FAMILY = "'IBM Plex Mono', 'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace";

/**
 * Apply the shared MCC dark-theme defaults (idempotent) and return the
 * passed-in options object unchanged. Call once before ``new Chart(ctx, options)``.
 *
 * @param {object} options - Chart.js options object; returned unmodified.
 * @returns {object} The same options object the caller passed in.
 */
export function applyChartTheme(options = {}) {
  if (!_initialized && typeof window !== 'undefined' && window.Chart) {
    const C = window.Chart;
    C.defaults.font.family = MONO_FAMILY;
    C.defaults.font.size = 10;
    C.defaults.color = 'rgba(244, 238, 222, 0.38)';
    C.defaults.borderColor = 'rgba(244, 238, 222, 0.06)';
    C.defaults.scale.grid.color = 'rgba(244, 238, 222, 0.06)';
    C.defaults.scale.grid.tickColor = 'transparent';
    C.defaults.scale.grid.borderColor = 'transparent';
    C.defaults.plugins.tooltip.backgroundColor = 'rgba(7, 7, 10, 0.96)';
    C.defaults.plugins.tooltip.titleColor = 'rgba(244, 238, 222, 0.96)';
    C.defaults.plugins.tooltip.bodyColor = 'rgba(244, 238, 222, 0.72)';
    C.defaults.plugins.tooltip.padding = 12;
    C.defaults.plugins.tooltip.cornerRadius = 0;
    C.defaults.plugins.tooltip.displayColors = false;
    C.defaults.plugins.tooltip.boxPadding = 6;
    C.defaults.plugins.tooltip.borderColor = 'rgba(118, 185, 0, 0.45)';
    C.defaults.plugins.tooltip.borderWidth = 1;
    C.defaults.plugins.tooltip.titleFont = { family: MONO_FAMILY, size: 11, weight: '700' };
    C.defaults.plugins.tooltip.bodyFont = { family: MONO_FAMILY, size: 11 };
    C.defaults.plugins.legend.labels.usePointStyle = true;
    C.defaults.plugins.legend.labels.boxWidth = 8;
    C.defaults.plugins.legend.labels.padding = 12;
    C.defaults.plugins.legend.labels.font = { family: MONO_FAMILY, size: 10, weight: '600' };
    C.defaults.elements.line.tension = 0.3;
    C.defaults.elements.line.borderWidth = 2;
    C.defaults.elements.point.radius = 3;
    C.defaults.elements.point.hoverRadius = 6;
    C.defaults.animation.duration = 500;
    C.defaults.animation.easing = 'easeOutCubic';
    _initialized = true;
  }
  return options;
}
