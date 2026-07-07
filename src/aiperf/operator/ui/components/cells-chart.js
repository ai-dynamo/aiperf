// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { useMemo } from 'preact/hooks';
import { palette } from '../lib/theme.js';
import { ChartWrapper } from './chart-wrapper.js';

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
        label: `${metric} (${stat})`,
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
      plugins: {
        legend: {
          display: datasets.length > 1,
          labels: { color: palette.text, font: { size: 11 } },
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
              const n = Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(3);
              return `${ctx.dataset.label}: ${n} (${stat})`;
            },
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: primaryDim.name, color: palette.overlay1, font: { size: 11 } },
          grid: { color: palette.surface0 },
          ticks: {
            color: palette.overlay1,
            font: { size: 10 },
            autoSkip: true,
            maxRotation: xTickRotation,
            minRotation: xTickRotation,
            ...(xTickCallback ? { callback: xTickCallback } : {}),
          },
        },
        y: {
          title: { display: true, text: `${metric} (${stat})`, color: palette.overlay1, font: { size: 11 } },
          grid: { color: palette.surface0 },
          ticks: { color: palette.overlay1, font: { size: 10 } },
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
