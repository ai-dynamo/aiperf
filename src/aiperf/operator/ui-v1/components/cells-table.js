import { html } from 'htm/preact';
import { palette } from '../lib/theme.js';

/**
 * Per-cell metric table.
 *
 * Props:
 *   dimensions: [{ name, values }]
 *   cells:      [CellEntry]
 *   metric:     string
 *   stat:       string
 *   onCellClick: (cell) => void
 */
export function CellsTable({ dimensions, cells, metric, stat, onCellClick }) {
  if (!cells || cells.length === 0) {
    return html`<div data-testid="sweep-cells-table" class="text-dim">
      No cells completed yet.
    </div>`;
  }

  const dimNames = (dimensions || []).map(d => d.name);

  return html`
    <div data-testid="sweep-cells-table">
      <table class="data-table">
        <thead>
          <tr>
            <th>idx</th>
            <th>label</th>
            ${dimNames.map(n => html`<th key=${n}>${n}</th>`)}
            <th>trials ✓</th>
            <th>trials ✗</th>
            <th>${metric} (${stat})</th>
          </tr>
        </thead>
        <tbody>
          ${cells.map(c => html`
            <tr key=${c.variation_index}
                onclick=${() => onCellClick && onCellClick(c)}
                style="cursor: ${onCellClick ? 'pointer' : 'default'}">
              <td>${c.variation_index}</td>
              <td>${c.variation_label}</td>
              ${dimNames.map(n => html`<td key=${n}>${c.values?.[n] ?? '—'}</td>`)}
              <td>${c.trials_completed}</td>
              <td style="color: ${c.trials_failed > 0 ? palette.red : 'inherit'}">
                ${c.trials_failed}
              </td>
              <td>${formatStat(c.metrics?.[metric]?.[stat])}</td>
            </tr>
          `)}
        </tbody>
      </table>
    </div>
  `;
}

function formatStat(v) {
  if (v == null) return '—';
  if (Math.abs(v) >= 100) return v.toFixed(1);
  return v.toFixed(3);
}
