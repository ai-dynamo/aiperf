// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { useMemo } from 'preact/hooks';
import { palette } from '../lib/theme.js';
import { ChartWrapper } from './chart-wrapper.js';
import { CHART_TYPOGRAPHY } from '../lib/typography.js';

/**
 * Per-cell metric chart for a sweep.
 *
 * Props:
 *   dimensions: [{ name, values }]   from /sweeps/:ns/:name/cells
 *   cells:      [CellEntry]
 *   metric:     string               e.g. 'request_throughput'
 *   stat:       string               e.g. 'avg' | 'p99'
 *
 * 1D dimension: line chart, x = dim values, y = chosen metric stat.
 * 2D dimension: small-multiples — one chart series per second-dim value.
 * 3+ D:        renders a single chart over the FIRST dimension and a
 *              note instructing to use the table view.
 */
const SERIES_COLORS = [
  palette.blue,
  palette.peach,
  palette.green,
  palette.mauve,
  palette.teal,
  palette.lavender,
  palette.red,
  palette.sapphire,
];

/**
 * Display name and unit for the metric keys a cell entry can carry.
 *
 * Kept in step with the identical table in cells-table.js rather than shared,
 * because the unit test harnesses load each component with its imports
 * stripped, so a component may only import symbols those harnesses stub.
 * Producer: `_cells_from_live_children`
 * (src/aiperf/operator/routers/sweeps.py:229-268) emits `request_throughput`
 * from `job.throughput_rps` and `request_latency_p99` from
 * `job.latency_p99_ms` (:214-217). Unknown keys get no unit claim.
 */
const METRIC_DISPLAY = {
  request_throughput: { label: 'Request throughput', unit: 'req/s' },
  request_latency_p99: { label: 'Request latency p99', unit: 'ms' },
  output_token_throughput: { label: 'Output token throughput', unit: 'tok/s' },
  total_token_throughput: { label: 'Total token throughput', unit: 'tok/s' },
};

function metricUnit(metric) {
  return METRIC_DISPLAY[metric]?.unit ?? null;
}

/**
 * Decimal count that keeps ~4 significant figures for a magnitude.
 *
 * Same band, and same duplication rationale, as `readoutDecimals` in
 * cells-table.js -- the tooltip and the table cell show the identical number,
 * so they must round it identically or the two views of one cell disagree.
 *
 * The previous rule was `Math.abs(v) >= 100 ? toFixed(1) : toFixed(3)`, which
 * yields 5 significant figures at 12.3456 on a value that is the mean of a
 * handful of per-trial `throughput_rps` samples (routers/sweeps.py:203-211).
 * A tooltip is the only place a reader gets a mark's numeric coordinate, so it
 * must keep enough resolution to order marks the way their positions do -- 4
 * significant figures is far finer than benchmark run-to-run variance and
 * cannot merge distinguishable cells.
 */
function readoutDecimals(value) {
  const abs = Math.abs(value);
  if (!isFinite(abs) || abs >= 1000) return 0;
  if (abs >= 100) return 1;
  if (abs >= 10) return 2;
  return 3;
}

function formatMetricValue(value, unit) {
  if (unit === 'ms' || unit === 'req/s') {
    return value.toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  }
  return value.toFixed(readoutDecimals(value));
}

/** Axis / series heading naming the quantity, its statistic, and its unit. */
function metricHeading(metric, stat) {
  const known = METRIC_DISPLAY[metric];
  const label = known?.label ?? String(metric ?? '').replace(/_/g, ' ');
  const withStat = stat ? `${label} ${stat}` : label;
  return known?.unit ? `${withStat} · ${known.unit}` : withStat;
}

function cellVariationIndex(cell, fallback) {
  return cell?.variation_index ?? cell?.variationIndex ?? fallback;
}

function cellVariationLabel(cell, fallback) {
  return cell?.variation_label ?? cell?.variationLabel ?? `v${cellVariationIndex(cell, fallback)}`;
}

export function CellsChart({ dimensions, cells, metric, stat }) {
  const { data, options, hasData, dimensionCount } = useMemo(() => {
    if (!cells || cells.length === 0) {
      return { data: null, options: null, hasData: false, dimensionCount: 0 };
    }

    const sourceDimensions = Array.isArray(dimensions) ? dimensions : [];
    const isDimensionless = sourceDimensions.length === 0;
    const effectiveDimensions = isDimensionless
      ? [{ name: 'variation', values: cells.map((cell, idx) => cellVariationLabel(cell, idx)) }]
      : sourceDimensions;
    const primaryDim = effectiveDimensions[0];
    const xValues = Array.isArray(primaryDim?.values) ? primaryDim.values : [];
    const datasets = [];

    if (effectiveDimensions.length <= 1) {
      const ys = isDimensionless
        ? cells.map(cell => cell?.metrics?.[metric]?.[stat] ?? null)
        : xValues.map(v => {
            const cell = cells.find(c => (c.values?.[primaryDim.name] === v));
            return cell?.metrics?.[metric]?.[stat] ?? null;
          });
      const c = SERIES_COLORS[0];
      datasets.push({
        label: metricHeading(metric, stat),
        data: ys,
        borderColor: c,
        backgroundColor: c + '22',
        pointBackgroundColor: c,
        pointBorderColor: c,
        pointRadius: 4,
        tension: 0.1,
        spanGaps: true,
      });
    } else {
      const secondDim = effectiveDimensions[1];
      const secondValues = Array.isArray(secondDim?.values) ? secondDim.values : [];
      secondValues.forEach((sv, idx) => {
        const ys = xValues.map(xv => {
          const cell = cells.find(cc =>
            cc.values?.[primaryDim.name] === xv &&
            cc.values?.[secondDim.name] === sv
          );
          return cell?.metrics?.[metric]?.[stat] ?? null;
        });
        const c = SERIES_COLORS[idx % SERIES_COLORS.length];
        datasets.push({
          label: `${secondDim.name}=${sv}`,
          data: ys,
          borderColor: c,
          backgroundColor: c + '22',
          pointBackgroundColor: c,
          pointBorderColor: c,
          pointRadius: 3,
          tension: 0.1,
          spanGaps: true,
        });
      });
    }

    const chartData = {
      labels: xValues.map(String),
      datasets,
    };
    // Long dim values (e.g. model paths, prompt-template names) overlap on
    // the x-axis at default rotation. Compute a worst-case label length so
    // we only rotate / truncate when needed and keep short numeric sweeps
    // (concurrency=1,2,4,...) horizontal.
    const maxLabelLen = chartData.labels.reduce((m, l) => Math.max(m, l.length), 0);
    const xTickRotation = maxLabelLen > 8 ? 35 : 0;
    const xTickCallback = maxLabelLen > 24
      ? function (value) {
          const lbl = this.getLabelForValue(value);
          return lbl != null && lbl.length > 24 ? lbl.slice(0, 22) + '…' : lbl;
        }
      : undefined;
    const chartOptions = {
      // Chart.js formats numeric axis ticks through
      // `Ticks.formatters.numeric`, which reads `chart.options.locale` and
      // passes it straight to `new Intl.NumberFormat(locale, ...)`
      // (chart.umd.min.js v4: `numeric(t,e,i){...const s=this.chart.options.locale`
      // and `function ne(t,e,i){...new Intl.NumberFormat(t,e)}`). Chart.js
      // needs the browser locale explicitly so its ticks agree with the table.
      locale: typeof navigator === 'undefined' ? undefined : navigator.language,
      plugins: {
        legend: {
          display: datasets.length > 1,
          labels: { color: palette.text, font: { size: CHART_TYPOGRAPHY.AXIS_LABEL } },
        },
        tooltip: {
          backgroundColor: palette.mantle,
          titleColor: palette.text,
          bodyColor: palette.text,
          borderColor: palette.surface0,
          borderWidth: 1,
          callbacks: {
            title: (items) => {
              if (!items || items.length === 0) return '';
              return `${primaryDim.name} = ${items[0].label}`;
            },
            label: (ctx) => {
              const v = ctx.parsed?.y;
              if (v == null) return `${ctx.dataset.label}: (no data)`;
              const n = formatMetricValue(v, metricUnit(metric));
              // With a known metric the unit is the useful suffix; the stat is
              // already on the y-axis title. For an unknown metric there is no
              // unit to state, so keep the stat rather than emit a bare number.
              const suffix = metricUnit(metric) ?? `(${stat})`;
              return `${ctx.dataset.label}: ${n} ${suffix}`;
            },
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: primaryDim.name, color: palette.overlay1, font: { size: CHART_TYPOGRAPHY.AXIS_LABEL } },
          grid: { color: palette.surface0 },
          ticks: {
            color: palette.overlay1,
            font: { size: CHART_TYPOGRAPHY.AXIS_TICK },
            autoSkip: true,
            maxRotation: xTickRotation,
            minRotation: xTickRotation,
            ...(xTickCallback ? { callback: xTickCallback } : {}),
          },
        },
        y: {
          title: { display: true, text: metricHeading(metric, stat), color: palette.overlay1, font: { size: CHART_TYPOGRAPHY.AXIS_LABEL } },
          grid: { color: palette.surface0 },
          ticks: { color: palette.overlay1, font: { size: CHART_TYPOGRAPHY.AXIS_TICK } },
        },
      },
    };
    return { data: chartData, options: chartOptions, hasData: true, dimensionCount: effectiveDimensions.length };
  }, [dimensions, cells, metric, stat]);

  if (!cells || cells.length === 0 || !hasData) {
    return html`<div data-testid="sweep-cells-chart" class="text-dim" style="padding:var(--space-3) 0">
      No cells completed yet.
    </div>`;
  }

  return html`
    <div data-testid="sweep-cells-chart">
      <${ChartWrapper} type="line" data=${data} options=${options} height=${360} />
      ${dimensionCount >= 3 && html`
        <p class="text-dim" style="margin-top:var(--space-2);font-size:var(--font-size-sm)">
          ${dimensionCount}-D sweep — chart shows the first dimension only.
          Use the table view to inspect higher-dim cells.
        </p>
      `}
    </div>
  `;
}
