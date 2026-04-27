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
    return html`<div data-testid="sweep-cells-table" class="text-dim" style="padding:var(--space-3) 0">
      No cells completed yet.
    </div>`;
  }

  const dimNames = (dimensions || []).map(d => d.name);

  return html`
    <div data-testid="sweep-cells-table" class="job-table-wrapper">
      <table class="job-table">
        <thead>
          <tr>
            <th class="job-table-th">idx</th>
            <th class="job-table-th">label</th>
            ${dimNames.map(n => html`<th key=${n} class="job-table-th">${n}</th>`)}
            <th class="job-table-th" style="text-align:right">trials ✓</th>
            <th class="job-table-th" style="text-align:right">trials ✗</th>
            <th class="job-table-th" style="text-align:right">${metric} (${stat})</th>
          </tr>
        </thead>
        <tbody>
          ${cells.map(c => html`
            <tr key=${c.variation_index}
                class="job-table-row"
                onclick=${() => onCellClick && onCellClick(c)}
                style=${onCellClick ? 'cursor: pointer' : ''}
                data-testid=${'sweep-cell-row-' + c.variation_index}>
              <td class="job-table-td text-dim">${c.variation_index}</td>
              <td class="job-table-td job-table-name">${c.variation_label || '—'}</td>
              ${dimNames.map(n => html`<td key=${n} class="job-table-td">${c.values?.[n] ?? '—'}</td>`)}
              <td class="job-table-td" style="text-align:right">${c.trials_completed}</td>
              <td class="job-table-td"
                  style=${`text-align:right;color:${c.trials_failed > 0 ? palette.red : 'inherit'}`}>
                ${c.trials_failed}
              </td>
              <td class="job-table-td" style="text-align:right">
                ${formatStat(c.metrics?.[metric]?.[stat])}
              </td>
            </tr>
          `)}
        </tbody>
      </table>
    </div>
  `;
}

function formatStat(v) {
  if (v == null) return html`<span class="text-dim">—</span>`;
  if (Math.abs(v) >= 100) return v.toFixed(1);
  return v.toFixed(3);
}
