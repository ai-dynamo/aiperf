import { html } from 'htm/preact';
import { palette } from '../lib/theme.js';
import { fmtNumber } from '../lib/format.js';

/**
 * Per-variation aggregate table.
 *
 * Rendering only — fetching, grouping, and stat computation live in the
 * parent (``sweep-detail`` page) so the chart and table share one fetch.
 *
 * Props:
 *   variations:   [{ variation_index, label, n_trials, n_total, perMetric: {key.stat: {mean, std, cv}} }]
 *   headlineMetrics: [{ key, stat, label, unit }]
 */

function fmtMean(value, unit) {
  if (value == null) return '—';
  if (unit === 'req/s' || unit === 'tok/s' || unit === 'tok/s/u') return fmtNumber(value, 0);
  if (unit === 'ms') return fmtNumber(value, value < 1 ? 4 : 2);
  return fmtNumber(value, 3);
}

export function VariationsTable({ variations, headlineMetrics }) {
  if (!variations || variations.length === 0) {
    return html`<div class="text-dim" style="padding:var(--space-3) 0" data-testid="variations-table-empty">
      No variation data available yet.
    </div>`;
  }

  return html`
    <div data-testid="sweep-variations-table" class="job-table-wrapper">
      <table class="job-table">
        <thead>
          <tr>
            <th class="job-table-th">variation</th>
            <th class="job-table-th" style="text-align:right">trials</th>
            ${headlineMetrics.map(m => html`
              <th key=${m.key + '.' + m.stat} class="job-table-th" style="text-align:right">
                ${m.label}<br/>
                <span class="text-dim" style="font-size:var(--font-size-xs);font-weight:normal">${m.unit}</span>
              </th>
            `)}
          </tr>
        </thead>
        <tbody>
          ${variations.map(v => html`
            <tr key=${v.variation_index} data-testid=${'variation-row-' + v.variation_index}>
              <td class="job-table-td"><code style="font-size:var(--font-size-xs)">${v.label || `v${v.variation_index}`}</code></td>
              <td class="job-table-td" style="text-align:right">${v.n_trials}/${v.n_total}</td>
              ${headlineMetrics.map(m => {
                const r = v.perMetric?.[m.key + '.' + m.stat];
                if (!r || r.mean == null) {
                  return html`<td key=${m.key + '.' + m.stat} class="job-table-td" style="text-align:right;color:${palette.overlay0}">—</td>`;
                }
                const cvPct = r.cv != null ? `${(r.cv * 100).toFixed(2)}%` : '—';
                return html`
                  <td key=${m.key + '.' + m.stat} class="job-table-td" style="text-align:right;font-variant-numeric:tabular-nums">
                    ${fmtMean(r.mean, m.unit)}
                    <br/>
                    <span class="text-dim" style="font-size:var(--font-size-xs)">cv ${cvPct}</span>
                  </td>
                `;
              })}
            </tr>
          `)}
        </tbody>
      </table>
    </div>
  `;
}
