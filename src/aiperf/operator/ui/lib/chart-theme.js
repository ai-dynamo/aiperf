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

/** Canonical dataset palette. Use ``PALETTE[index % PALETTE.length]`` for nth-dataset color. */
export const PALETTE = [
  '#a3e635', '#60a5fa', '#f472b6', '#fbbf24',
  '#34d399', '#c084fc', '#fb7185', '#22d3ee',
];

/**
 * Apply the shared dark-theme defaults (idempotent) and return the passed-in
 * options object unchanged. Call once before ``new Chart(ctx, options)``.
 *
 * @param {object} options - Chart.js options object; returned unmodified.
 * @returns {object} The same options object the caller passed in.
 */
export function applyChartTheme(options = {}) {
  if (!_initialized && typeof window !== 'undefined' && window.Chart) {
    const C = window.Chart;
    C.defaults.font.family = '"Geist", -apple-system, BlinkMacSystemFont, sans-serif';
    C.defaults.font.size = 12;
    C.defaults.color = 'rgba(255,255,255,0.68)';
    C.defaults.borderColor = 'rgba(255,255,255,0.06)';
    C.defaults.scale.grid.color = 'rgba(255,255,255,0.05)';
    C.defaults.scale.grid.tickColor = 'transparent';
    C.defaults.scale.grid.borderColor = 'transparent';
    C.defaults.plugins.tooltip.backgroundColor = 'rgba(16,16,18,0.95)';
    C.defaults.plugins.tooltip.titleColor = 'rgba(255,255,255,0.94)';
    C.defaults.plugins.tooltip.bodyColor = 'rgba(255,255,255,0.68)';
    C.defaults.plugins.tooltip.padding = 12;
    C.defaults.plugins.tooltip.cornerRadius = 8;
    C.defaults.plugins.tooltip.displayColors = true;
    C.defaults.plugins.tooltip.boxPadding = 6;
    C.defaults.plugins.tooltip.borderColor = 'rgba(255,255,255,0.08)';
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
