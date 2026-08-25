// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { palette } from '../lib/theme.js';

/**
 * Display name and unit for the metric keys a cell entry can carry.
 *
 * The only producer of cells today is `_cells_from_live_children`
 * (src/aiperf/operator/routers/sweeps.py:229-268), which emits exactly
 * `request_throughput` from `job.throughput_rps` and `request_latency_p99`
 * from `job.latency_p99_ms` (:214-217) -- hence req/s and ms. The throughput
 * tags are listed too because their units are unambiguous wherever they come
 * from. Anything else falls through to a humanized key with no unit claim,
 * because inventing a unit is worse than omitting one.
 */
const METRIC_DISPLAY = {
  request_throughput: { label: 'Request throughput', unit: 'req/s' },
  request_latency_p99: { label: 'Request latency p99', unit: 'ms' },
  output_token_throughput: { label: 'Output token throughput', unit: 'tok/s' },
  total_token_throughput: { label: 'Total token throughput', unit: 'tok/s' },
};

/**
 * Column / axis heading naming the quantity, its statistic, and its unit.
 *
 * The raw key (`request_throughput`) is an internal metric tag, and a bare
 * number with no unit cannot be compared against anything else on the page.
 *
 * Not exported: `tests/unit/ui/test_operator_tables_adversarial.py` loads this
 * module through `eval`, which rejects any surviving `export` statement.
 */
function metricHeading(metric, stat) {
  const known = METRIC_DISPLAY[metric];
  const label = known?.label ?? String(metric ?? '').replace(/_/g, ' ');
  const withStat = stat ? `${label} ${stat}` : label;
  return known?.unit ? `${withStat} · ${known.unit}` : withStat;
}

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
            <th class="job-table-th" style="text-align:right" title=${`Mean ${metric} (${stat}) across this cell's completed trials`}>${metricHeading(metric, stat)}</th>
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
                  ${formatStat(c.metrics?.[metric]?.[stat], METRIC_DISPLAY[metric]?.unit)}
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

/**
 * Decimal count that keeps ~4 significant figures for a magnitude.
 *
 * Identical band to `readoutDecimals` in sweep-live-trial-board.js:53-59 and
 * `tooltipDecimals` in variations-pareto.js:63-69 -- duplicated rather than
 * shared for the reason those files already state: the unit-test harnesses
 * load each component with its imports stripped and a fixed stub prelude, so a
 * component may only import symbols those preludes provide.
 *
 * The previous rule here was `Math.abs(v) >= 100 ? toFixed(1) : toFixed(3)`,
 * which is not a precision policy at all -- it hands out 5 significant figures
 * at 12.3456 ("12.346") and 7 at 12345.678 ("12345.7"). Both over-claim: the
 * number in this column is the arithmetic mean of each completed trial's
 * `throughput_rps` (routers/sweeps.py:203-211 `_avg`), taken over the handful
 * of trials counted in the adjacent "trials" column, and the same per-run
 * quantity is displayed one table up at one decimal (job-table.js:217 ->
 * format.js:70-73). Averaging cannot add resolution the inputs never had, so
 * the aggregate must not be shown finer than its inputs.
 *
 * Four significant figures still resolves ~1 part in 1000-10000, one to two
 * orders finer than the run-to-run variance of any throughput benchmark, so
 * genuinely different cells never collapse to the same string.
 */
function readoutDecimals(value) {
  const abs = Math.abs(value);
  if (!isFinite(abs) || abs >= 1000) return 0;
  if (abs >= 100) return 1;
  if (abs >= 10) return 2;
  return 3;
}

function formatStat(v, unit) {
  const value = finiteNumber(v);
  if (value == null) return html`<span class="text-dim">—</span>`;
  if (unit === 'ms' || unit === 'req/s') {
    return value.toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  }
  return value.toFixed(readoutDecimals(value));
}
