import { html } from 'htm/preact';

const slugifyLabel = (s) => String(s ?? '').toLowerCase().trim().replace(/\s+/g, '-');

// ─────────────────────────────────────────────────────────────────────────
// Material-style outlined SVG icon registry. Looked up by string key in
// the ``icon`` prop so callers can pass a stable identifier instead of
// importing each SVG. Same visual idiom as the cluster banner (24×24,
// 1.8 stroke, rounded joins).
// ─────────────────────────────────────────────────────────────────────────

function MetricIcon({ name }) {
  const stroke = 'fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"';
  switch (name) {
    case 'speed':
      // Half-circle gauge with needle — throughput / rate metrics.
      return html`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M3 16 a9 9 0 0 1 18 0" />
        <line x1="12" y1="16" x2="16.5" y2="9" />
        <circle cx="12" cy="16" r="1.4" fill="currentColor" stroke="none" />
      </svg>`;
    case 'clock':
      // Clock face — TTFT and time-to-* metrics.
      return html`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <circle cx="12" cy="12" r="9" />
        <line x1="12" y1="7" x2="12" y2="12" />
        <line x1="12" y1="12" x2="15.5" y2="14" />
      </svg>`;
    case 'timer':
      // Stopwatch — request latency and ITL.
      return html`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <circle cx="12" cy="13" r="8" />
        <line x1="9" y1="2.5" x2="15" y2="2.5" />
        <line x1="12" y1="2.5" x2="12" y2="6" />
        <line x1="12" y1="13" x2="16" y2="9" />
        <line x1="18.5" y1="6" x2="20.5" y2="8" />
      </svg>`;
    case 'tokens':
      // Stacked horizontal bars — output token counts / per-user rates.
      return html`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <line x1="4" y1="6" x2="16" y2="6" />
        <line x1="4" y1="12" x2="20" y2="12" />
        <line x1="4" y1="18" x2="11" y2="18" />
      </svg>`;
    case 'requests':
      // Stack of cards / receipts — total request count.
      return html`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <rect x="4" y="7" width="14" height="14" rx="1.5" />
        <path d="M7 4 h13 a1.5 1.5 0 0 1 1.5 1.5 v13" />
        <line x1="7" y1="11" x2="15" y2="11" />
        <line x1="7" y1="15" x2="15" y2="15" />
      </svg>`;
    case 'errors':
      // Warning triangle — error rate / failed runs.
      return html`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M10.3 3.5 L1.5 19 a2 2 0 0 0 1.7 3 h17.6 a2 2 0 0 0 1.7 -3 L13.7 3.5 a2 2 0 0 0 -3.4 0 z" />
        <line x1="12" y1="10" x2="12" y2="14" />
        <circle cx="12" cy="17.2" r="0.6" fill="currentColor" stroke="none" />
      </svg>`;
    case 'goodput':
      // Concentric target — SLO / goodput compliance.
      return html`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <circle cx="12" cy="12" r="8.5" />
        <circle cx="12" cy="12" r="5" />
        <circle cx="12" cy="12" r="1.6" fill="currentColor" stroke="none" />
      </svg>`;
    case 'trophy':
      // Trophy — best run / leaderboard hero values.
      return html`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M8 21 h8" />
        <path d="M12 17 v4" />
        <path d="M7 4 h10 v3 a5 5 0 0 1 -10 0 z" />
        <path d="M17 5 h3 v2 a3 3 0 0 1 -3 3" />
        <path d="M7 5 h-3 v2 a3 3 0 0 0 3 3" />
        <path d="M9 13 a4 4 0 0 0 6 0" />
      </svg>`;
    case 'trending-up':
      // Up-and-to-the-right line — token throughput, growth metrics.
      return html`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <polyline points="3,17 9,11 13,15 21,7" />
        <polyline points="14,7 21,7 21,14" />
      </svg>`;
    case 'check':
      // Check-circle — completed / success counts.
      return html`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <circle cx="12" cy="12" r="9" />
        <polyline points="8,12.5 11,15.5 16,9.5" />
      </svg>`;
    default:
      return null;
  }
}

/**
 * Metric card — small KPI tile with optional Material-style icon, tone-driven
 * coloring, and a progress bar slot. Backward-compatible: callers without
 * ``icon``/``tone``/``progress`` get the previous bare-card rendering.
 *
 * @param {object} props
 * @param {string} props.label - Short metric name (e.g. ``"TTFT P99"``).
 * @param {string|number} [props.value] - Primary value.
 * @param {string} [props.unit] - Unit suffix (e.g. ``"ms"``).
 * @param {string} [props.color] - Legacy: explicit value color override.
 * @param {string} [props.sub] - Optional sub-line below the value.
 * @param {string} [props.title] - Tooltip on hover.
 * @param {string} [props.icon] - Icon registry key: ``speed``, ``clock``,
 *   ``timer``, ``tokens``, ``requests``, ``errors``, ``goodput``, ``trophy``,
 *   ``trending-up``, ``check``.
 * @param {('ok'|'warn'|'bad'|'accent'|'neutral'|'gold'|'live')} [props.tone]
 *   - Tints the icon chip and (when ``color`` is unset) the value number.
 * @param {number} [props.progress] - 0–100; renders a thin progress bar.
 * @param {string} [props.progressTone] - Override tone for the bar.
 */
export function KpiCard({
  label,
  value,
  unit,
  color,
  sub,
  title,
  icon,
  tone,
  progress,
  progressTone,
}) {
  const valueStyle = color ? `color: ${color}` : '';
  const valueClass = 'metric-val' + (!color && tone ? ` metric-val--${tone}` : '');
  const cardClass = 'metric-card' + (icon || tone ? ' metric-card--rich' : '');
  const iconToneClass = tone ? ` metric-icon--${tone}` : ' metric-icon--neutral';
  const barTone = progressTone ?? tone ?? 'neutral';

  // Legacy layout when caller doesn't opt in to the rich treatment.
  if (!icon) {
    return html`
      <div class=${cardClass} data-testid=${'kpi-' + slugifyLabel(label)} title=${title}>
        <span class="metric-label">${label}</span>
        <div class="metric-val-row">
          <span class=${valueClass} style=${valueStyle}>
            ${value ?? '—'}
          </span>
          ${unit && html`<span class="metric-unit">${unit}</span>`}
        </div>
        ${sub && html`<div class="metric-sub">${sub}</div>`}
        ${progress != null && html`
          <div class="metric-bar" aria-hidden="true">
            <div class=${'metric-bar__fill metric-bar__fill--' + barTone}
                 style=${'width: ' + Math.min(100, Math.max(0, Number(progress))) + '%'}></div>
          </div>
        `}
      </div>
    `;
  }

  return html`
    <div class=${cardClass} data-testid=${'kpi-' + slugifyLabel(label)} title=${title}>
      <div class="metric-card__row">
        <div class=${'metric-icon' + iconToneClass}><${MetricIcon} name=${icon} /></div>
        <div class="metric-card__body">
          <span class="metric-label">${label}</span>
          <div class="metric-val-row">
            <span class=${valueClass} style=${valueStyle}>
              ${value ?? '—'}
            </span>
            ${unit && html`<span class="metric-unit">${unit}</span>`}
          </div>
          ${sub && html`<div class="metric-sub">${sub}</div>`}
        </div>
      </div>
      ${progress != null && html`
        <div class="metric-bar" aria-hidden="true">
          <div class=${'metric-bar__fill metric-bar__fill--' + barTone}
               style=${'width: ' + Math.min(100, Math.max(0, Number(progress))) + '%'}></div>
        </div>
      `}
    </div>
  `;
}
