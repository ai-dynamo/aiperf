import { html } from 'htm/preact';

const slugifyLabel = (s) => String(s ?? '').toLowerCase().trim().replace(/\s+/g, '-');

/**
 * Metric card — simple card, brand-colored value for key metrics.
 * @param {{ label: string, value: string|number, unit?: string, color?: string, sub?: string, title?: string }} props
 */
export function KpiCard({ label, value, unit, color, sub, title }) {
  const valueStyle = color ? `color: ${color}` : '';

  return html`
    <div class="metric-card" data-testid=${'kpi-' + slugifyLabel(label)} title=${title}>
      <span class="metric-label">${label}</span>
      <div class="metric-val-row">
        <span class="metric-val" style=${valueStyle}>
          ${value ?? '\u2014'}
        </span>
        ${unit && html`<span class="metric-unit">${unit}</span>`}
      </div>
      ${sub && html`<div class="metric-sub">${sub}</div>`}
    </div>
  `;
}
