/**
 * Hand-tuned dark-theme defaults for Chart.js.
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

/** Canonical dataset palette — CONSOLE: amber, cyan, green, red, purple, paper. */
export const PALETTE = [
  '#ff9f1c', '#3ad8e3', '#7ccf5e', '#ff5c5c',
  '#b59aff', '#f4f0e1', '#ffb547', '#60a5fa',
];

/**
 * Apply the shared CONSOLE dark-theme defaults (idempotent) and return the
 * passed-in options object unchanged. Call once before ``new Chart(ctx, options)``.
 *
 * @param {object} options - Chart.js options object; returned unmodified.
 * @returns {object} The same options object the caller passed in.
 */
export function applyChartTheme(options = {}) {
  if (!_initialized && typeof window !== 'undefined' && window.Chart) {
    const C = window.Chart;
    C.defaults.font.family = "'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace";
    C.defaults.font.size = 10;
    C.defaults.color = 'rgba(244, 240, 225, 0.36)';
    C.defaults.borderColor = 'rgba(244, 240, 225, 0.06)';
    C.defaults.scale.grid.color = 'rgba(244, 240, 225, 0.06)';
    C.defaults.scale.grid.tickColor = 'transparent';
    C.defaults.scale.grid.borderColor = 'transparent';
    C.defaults.plugins.tooltip.backgroundColor = 'rgba(14, 16, 20, 0.98)';
    C.defaults.plugins.tooltip.titleColor = 'rgba(244, 240, 225, 0.94)';
    C.defaults.plugins.tooltip.bodyColor = 'rgba(244, 240, 225, 0.68)';
    C.defaults.plugins.tooltip.padding = 12;
    C.defaults.plugins.tooltip.cornerRadius = 0;
    C.defaults.plugins.tooltip.displayColors = false;
    C.defaults.plugins.tooltip.boxPadding = 6;
    C.defaults.plugins.tooltip.borderColor = 'rgba(255, 159, 28, 0.32)';
    C.defaults.plugins.tooltip.borderWidth = 1;
    C.defaults.plugins.legend.labels.usePointStyle = true;
    C.defaults.plugins.legend.labels.boxWidth = 8;
    C.defaults.plugins.legend.labels.padding = 12;
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
