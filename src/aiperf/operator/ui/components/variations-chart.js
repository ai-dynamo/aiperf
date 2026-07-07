// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { useMemo } from 'preact/hooks';
import { palette } from '../lib/theme.js';
import { ChartWrapper } from './chart-wrapper.js';

/**
 * Variation curve chart with error bars.
 *
 * Plots one point per variation: x = variation label (or the
 * single-variable value when the sweep has exactly one swept dimension),
 * y = mean of the chosen metric across trials, error bar = ±1 std.
 *
 * Props:
 *   variations: [{ variation_index, label, mean, std, cv, n }]
 *   metricLabel: string  (e.g. "Output tok/s")
 *   unit:        string  (e.g. "tok/s")
 */

function shortLabel(label) {
  if (!label) return '';
  // ``phases.profiling.concurrency=10`` → ``concurrency=10`` for compactness.
  const eq = label.indexOf('=');
  if (eq < 0) return label;
  const dot = label.lastIndexOf('.', eq);
  return dot >= 0 ? label.slice(dot + 1) : label;
}

function finiteOrNull(value) {
  return typeof value === 'number' && isFinite(value) ? value : null;
}

export function VariationsChart({ variations, metricLabel, unit }) {
  const chart = useMemo(() => {
    if (!variations || variations.length === 0) return null;
    const labels = variations.map(v => shortLabel(v.label) || `v${v.variation_index}`);
    const means = variations.map(v => finiteOrNull(v.mean));
    // If every variation is missing this metric (e.g. selected metric not yet
    // computed across any trial), Chart.js still renders an empty axis box —
    // signal "no data" up so the page can show its empty state instead.
    if (means.every(m => m == null)) return null;
    const stds = variations.map(v => finiteOrNull(v.std) ?? 0);
    const errorPlus = means.map((m, i) => (m == null ? null : m + stds[i]));
    const errorMinus = means.map((m, i) => (m == null ? null : m - stds[i]));

    const data = {
      labels,
      datasets: [
        // ±std band drawn as two filled lines stacked on the same axis.
        {
          label: 'mean + std',
          data: errorPlus,
          borderColor: 'transparent',
          backgroundColor: palette.blue + '22',
          pointRadius: 0,
          fill: '+1',
          tension: 0.2,
          order: 2,
        },
        {
          label: 'mean - std',
          data: errorMinus,
          borderColor: 'transparent',
          backgroundColor: palette.blue + '22',
          pointRadius: 0,
          fill: false,
          tension: 0.2,
          order: 3,
        },
        {
          label: metricLabel,
          data: means,
          borderColor: palette.blue,
          backgroundColor: palette.blue,
          pointBackgroundColor: palette.blue,
          pointBorderColor: palette.blue,
          pointRadius: 5,
          pointHoverRadius: 7,
          tension: 0.2,
          order: 1,
        },
      ],
    };

    const options = {
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: palette.mantle,
          titleColor: palette.text,
          bodyColor: palette.text,
          borderColor: palette.surface0,
          borderWidth: 1,
          filter: (item) => item.dataset.label === metricLabel,
          callbacks: {
            label: (ctx) => {
              const v = variations[ctx.dataIndex];
              const cv = typeof v?.cv === 'number' && isFinite(v.cv) ? ` (cv ${(v.cv * 100).toFixed(2)}%)` : '';
              return `  mean ${ctx.parsed.y?.toFixed(2)} ${unit}${cv}, n=${v?.n ?? 0}`;
            },
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: 'variation', color: palette.overlay1, font: { size: 11 } },
          grid: { color: palette.surface0 },
          ticks: { color: palette.overlay1, font: { size: 10 } },
        },
        y: {
          title: { display: true, text: `${metricLabel} (${unit})`, color: palette.overlay1, font: { size: 11 } },
          grid: { color: palette.surface0 },
          ticks: { color: palette.overlay1, font: { size: 10 } },
        },
      },
    };
    return { data, options };
  }, [variations, metricLabel, unit]);

  if (!chart) {
    return html`<div class="text-dim" style="padding:var(--space-3) 0" data-testid="variations-chart-empty">
      No ${metricLabel || 'variation'} data available for any variation yet.
    </div>`;
  }
  return html`
    <div data-testid="sweep-variations-chart">
      <${ChartWrapper} type="line" data=${chart.data} options=${chart.options} height=${280} />
    </div>
  `;
}
