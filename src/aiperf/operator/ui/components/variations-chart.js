// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { useMemo } from 'preact/hooks';
import { palette } from '../lib/theme.js';
import { fmtFixed, fmtNumber } from '../lib/format.js';
import { ChartWrapper } from './chart-wrapper.js';
import { CHART_TYPOGRAPHY } from '../lib/typography.js';

/**
 * Variation curve chart with error bars.
 *
 * Plots one point per variation: x = the swept parameter values for that
 * variation (`valuesLabel`) when the caller supplies them, otherwise the
 * variation label; y = mean of the chosen metric across that variation's
 * trials, shaded band = ±1 std across those same trials.
 *
 * The band is drawn only for variations with n >= 2 and a finite `std`. A
 * single-trial variation has no spread estimate at all, so it gets a plotted
 * mean and a gap in the band rather than a zero-width band that would read as
 * a measurement of perfect reproducibility.
 *
 * Props:
 *   variations: [{ variation_index, label, valuesLabel, mean, std, cv, n }]
 *     `std` is null when spread was not measured (n < 2); do not coerce it
 *     to 0 on the way in.
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

/**
 * Tick label for one variation: the parameter values it ran, if known.
 *
 * Adaptive planners name variations `search_iter_NNNN`
 * (src/aiperf/orchestrator/search_planner/optuna_planner.py:226). That string
 * is a cell identity for artifact paths; `shortLabel` cannot improve it
 * because there is no `=` to split on. An x axis of `search_iter_0004 ...
 * search_iter_0011` tells the reader nothing about what varied along it, which
 * defeats the point of a curve. `buildSweepVariations` attaches `valuesLabel`
 * (e.g. `"concurrency=17"`) from the CR's `status.runs[].values`; prefer it,
 * then the raw label, then the bare index.
 */
function tickLabel(variation) {
  return variation?.valuesLabel
    || shortLabel(variation?.label)
    || `v${variation?.variation_index}`;
}

function finiteOrNull(value) {
  return typeof value === 'number' && isFinite(value) ? value : null;
}

/**
 * Decimal count that keeps ~4 significant figures for a magnitude.
 *
 * Deliberately local rather than shared with variations-pareto.js: the unit
 * test harnesses load each component with its imports stripped and a fixed
 * stub prelude, so a component may only import symbols those preludes already
 * provide (`fmtNumber` is one, a new shared helper would not be).
 */
function readoutDecimals(value) {
  const abs = Math.abs(value);
  if (!isFinite(abs) || abs >= 1000) return 0;
  if (abs >= 100) return 1;
  if (abs >= 10) return 2;
  return 3;
}

export function VariationsChart({ variations, metricLabel, unit }) {
  const chart = useMemo(() => {
    if (!variations || variations.length === 0) return null;
    const labels = variations.map(tickLabel);
    const means = variations.map(v => finiteOrNull(v.mean));
    // If every variation is missing this metric (e.g. selected metric not yet
    // computed across any trial), Chart.js still renders an empty axis box —
    // signal "no data" up so the page can show its empty state instead.
    if (means.every(m => m == null)) return null;
    // A band is drawn only where spread was actually estimated. `meanStd`
    // reports `std: null` for a single trial (pages/sweep-detail-helpers.js:66)
    // -- unmeasured, not zero -- and coercing that to 0 drew a zero-width band
    // through the point, which reads as a confident measurement of perfect
    // reproducibility rather than the absence of one. Null here leaves a gap in
    // the band instead, saying the same thing the tooltip below already says.
    //
    // Gated on `n` as well as on `std` being finite, because a caller that
    // still coerces `std ?? 0` upstream would otherwise reintroduce exactly the
    // zero-width band this removes. The tooltip uses the identical `n >= 2`
    // test, so the shading and the readout can never disagree.
    const bandStds = variations.map(v =>
      (v?.n ?? 0) >= 2 ? finiteOrNull(v.std) : null
    );
    const withBand = (m, i) => (m == null || bandStds[i] == null ? null : m);
    const errorPlus = means.map((m, i) => (withBand(m, i) == null ? null : m + bandStds[i]));
    const errorMinus = means.map((m, i) => (withBand(m, i) == null ? null : m - bandStds[i]));

    // tension 0 on every dataset: the x axis is a category axis of discrete,
    // separately-executed operating points, so nothing was measured between
    // two ticks. Chart.js cubic smoothing draws a curve through them that can
    // overshoot past both neighbours, inventing local maxima -- an apparent
    // throughput peak "between" concurrency 8 and 16 that no run produced.
    // Straight segments claim only ordering, which is all the data supports.
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
          tension: 0,
          order: 2,
        },
        {
          label: 'mean - std',
          data: errorMinus,
          borderColor: 'transparent',
          backgroundColor: palette.blue + '22',
          pointRadius: 0,
          fill: false,
          tension: 0,
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
          tension: 0,
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
              const mean = ctx.parsed.y;
              const fixedPrecision = unit === 'ms' || unit === 'req/s';
              const decimals = fixedPrecision ? 2 : readoutDecimals(mean);
              const trials = v?.n ?? 0;
              const std = finiteOrNull(v?.std);
              // With one trial `meanStd` reports `std: null`
              // (pages/sweep-detail-helpers.js:66) and no band is drawn. A
              // "±0" readout would claim perfect reproducibility from a
              // measurement that never estimated spread at all, so say what
              // was actually run instead.
              const spread = trials >= 2 && std != null
                ? ` ±${fixedPrecision ? fmtFixed(std, decimals) : fmtNumber(std, decimals)}`
                : '';
              const cv = trials >= 2 && typeof v?.cv === 'number' && isFinite(v.cv)
                ? ` (cv ${(v.cv * 100).toFixed(1)}%)`
                : '';
              const sample = trials >= 2 ? `mean of ${trials} trials` : '1 trial, spread unknown';
              const formattedMean = fixedPrecision ? fmtFixed(mean, decimals) : fmtNumber(mean, decimals);
              return `  ${formattedMean}${spread} ${unit}${cv} — ${sample}`;
            },
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: 'variation', color: palette.overlay1, font: { size: CHART_TYPOGRAPHY.AXIS_LABEL } },
          grid: { color: palette.surface0 },
          ticks: { color: palette.overlay1, font: { size: CHART_TYPOGRAPHY.AXIS_TICK } },
        },
        y: {
          title: { display: true, text: `${metricLabel} (${unit})`, color: palette.overlay1, font: { size: CHART_TYPOGRAPHY.AXIS_LABEL } },
          grid: { color: palette.surface0 },
          ticks: {
            color: palette.overlay1,
            font: { size: CHART_TYPOGRAPHY.AXIS_TICK },
            callback: value => unit === 'ms' || unit === 'req/s' ? fmtFixed(Number(value), 2) : value,
          },
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
