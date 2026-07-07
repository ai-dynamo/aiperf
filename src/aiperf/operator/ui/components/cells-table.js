// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
    <div data-testid="sweep-cells-table" class="job-table-wrapper" style="max-height:520px;overflow:auto">
      <table class="job-table">
        <thead style="position:sticky;top:0;z-index:1;background:var(--ctp-base)">
          <tr>
            <th class="job-table-th" style="text-align:right">idx</th>
            <th class="job-table-th">label</th>
            ${dimNames.map(n => html`<th key=${n} class="job-table-th" style="text-align:right">${n}</th>`)}
            <th class="job-table-th" style="text-align:right" title="Trials that completed successfully for this cell">trials ✓</th>
            <th class="job-table-th" style="text-align:right" title="Trials that failed for this cell">trials ✗</th>
            <th class="job-table-th" style="text-align:right" title=${`Mean ${metric} across trials (${stat})`}>${metric} (${stat})</th>
          </tr>
        </thead>
        <tbody>
          ${cells.map(c => {
            const variationIndex = c.variation_index ?? c.variationIndex;
            const variationLabel = c.variation_label ?? c.variationLabel;
            const trialsCompleted = c.trials_completed ?? c.trialsCompleted ?? 0;
            const trialsFailed = c.trials_failed ?? c.trialsFailed ?? 0;
            return html`
              <tr key=${variationIndex}
                  class="job-table-row"
                  role="row"
                  tabindex=${onCellClick ? '0' : undefined}
                  onKeyDown=${(e) => { if (onCellClick && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); onCellClick(c); } }}
                  onclick=${() => onCellClick && onCellClick(c)}
                  style=${onCellClick ? 'cursor: pointer' : ''}
                  data-testid=${'sweep-cell-row-' + variationIndex}>
                <td class="job-table-td text-dim" style="text-align:right;font-variant-numeric:tabular-nums">${variationIndex}</td>
                <td class="job-table-td job-table-name">${variationLabel || '—'}</td>
                ${dimNames.map(n => html`<td key=${n} class="job-table-td" style="text-align:right;font-variant-numeric:tabular-nums">${c.values?.[n] ?? '—'}</td>`)}
                <td class="job-table-td" style="text-align:right;font-variant-numeric:tabular-nums">${trialsCompleted}</td>
                <td class="job-table-td"
                    style=${`text-align:right;font-variant-numeric:tabular-nums;color:${trialsFailed > 0 ? palette.red : 'inherit'}`}>
                  ${trialsFailed}
                </td>
                <td class="job-table-td" style="text-align:right;font-variant-numeric:tabular-nums">
                  ${formatStat(c.metrics?.[metric]?.[stat])}
                </td>
              </tr>
            `;
          })}
        </tbody>
      </table>
    </div>
  `;
}

function finiteNumber(value) {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value !== 'string' || value.trim() === '') return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatStat(v) {
  const value = finiteNumber(v);
  if (value == null) return html`<span class="text-dim">—</span>`;
  if (Math.abs(value) >= 100) return value.toFixed(1);
  return value.toFixed(3);
}
