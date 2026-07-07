// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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

  // Pre-pass: build a Set of metric column keys that have at least one
  // non-null mean across visible variations. Columns missing from the set
  // are dropped from header AND body so sweeps that exercise only a subset
  // of metrics don't render a sea of em-dashes. Caller's headlineMetrics
  // array is not mutated.
  const populatedKeys = new Set();
  for (const v of variations) {
    const pm = v.perMetric;
    if (!pm) continue;
    for (const m of headlineMetrics) {
      const k = m.key + '.' + m.stat;
      if (populatedKeys.has(k)) continue;
      const r = pm[k];
      if (r && r.mean != null) populatedKeys.add(k);
    }
  }
  const visibleMetrics = headlineMetrics.filter(m => populatedKeys.has(m.key + '.' + m.stat));

  // Wide sweeps (many headline metrics) blow past viewport width and force the
  // wrapper into horizontal scroll. Pin the variation-label column to the left
  // so the user keeps their bearings while panning. The thead corner sits at
  // z:3 (above its row at z:2 and the body sticky cells at z:1) so it stays
  // on top during simultaneous vertical+horizontal scroll.
  const stickyTh = 'position:sticky;left:0;z-index:3;background:var(--ctp-base)';
  const stickyTd = 'position:sticky;left:0;z-index:1;background:var(--ctp-base)';

  return html`
    <div data-testid="sweep-variations-table" class="job-table-wrapper" style="max-height:520px;overflow:auto">
      <table class="job-table">
        <thead style="position:sticky;top:0;z-index:2;background:var(--ctp-base)">
          <tr>
            <th class="job-table-th" style=${stickyTh}>variation</th>
            <th class="job-table-th" style="text-align:right">trials</th>
            ${visibleMetrics.map(m => html`
              <th key=${m.key + '.' + m.stat} class="job-table-th" style="text-align:right" title=${`${m.label} (${m.unit}) — mean across trials, with coefficient of variation below`}>
                ${m.label}<br/>
                <span class="text-dim" style="font-size:var(--font-size-xs);font-weight:normal">${m.unit}</span>
              </th>
            `)}
          </tr>
        </thead>
        <tbody>
          ${variations.map(v => html`
            <tr key=${v.variation_index} data-testid=${'variation-row-' + v.variation_index}>
              <td class="job-table-td" style=${stickyTd}><code style="font-size:var(--font-size-xs)">${v.label || `v${v.variation_index}`}</code></td>
              <td class="job-table-td" style="text-align:right">${v.n_trials}/${v.n_total}</td>
              ${visibleMetrics.map(m => {
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
