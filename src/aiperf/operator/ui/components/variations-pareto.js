// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { useMemo } from 'preact/hooks';
import { palette } from '../lib/theme.js';
import { fmtMilliseconds, fmtNumber, fmtReqPerSecond } from '../lib/format.js';
import { ChartWrapper } from './chart-wrapper.js';
import { CHART_TYPOGRAPHY } from '../lib/typography.js';

/**
 * Pareto-frontier scatter for a sweep — mirrors the legacy ui's
 * ``analysis.js`` pattern: sort points ascending in x, walk left-to-right
 * tracking the best-so-far on y, push on improvement. Output is a
 * naturally-sorted frontier rendered as a dashed line; dominated points
 * stay rendered as muted scatter dots.
 *
 * Props:
 *   variations: [{ variation_index, label, valuesLabel,
 *                  perMetric: { "<key>.<stat>": { mean, std, cv, n } } }]
 *   xMetric:    { key, stat, label, unit }
 *   yMetric:    { key, stat, label, unit }
 *   yIsSmallerBetter: bool   (if y is a latency metric — frontier rule flips)
 */

function shortLabel(label) {
  if (!label) return '';
  const eq = label.indexOf('=');
  if (eq < 0) return label;
  const dot = label.lastIndexOf('.', eq);
  return dot >= 0 ? label.slice(dot + 1) : label;
}

/**
 * Identify a plotted point by the parameter values it ran, not by its cell id.
 *
 * `buildSweepVariations` attaches `valuesLabel` (e.g. `"concurrency=17"`) from
 * the CR's `status.runs[].values`. Adaptive planners name variations
 * `search_iter_NNNN` (src/aiperf/orchestrator/search_planner/optuna_planner.py:226),
 * which is an artifact-path identity: `shortLabel` cannot improve it because
 * there is no `=` to split on, so the tooltip and the frontier trail read
 * `search_iter_0004 -> search_iter_0008` and tell the reader nothing about
 * which operating point won. Prefer the swept values so each mark is
 * self-describing; fall back to the label, then the bare index.
 */
function pointLabel(variation) {
  return variation?.valuesLabel
    || shortLabel(variation?.label)
    || `v${variation?.variation_index}`;
}

function isFiniteNumber(value) {
  return typeof value === 'number' && isFinite(value);
}

/**
 * Decimal count that keeps ~4 significant figures for a magnitude.
 *
 * The tooltip is the only place a reader gets the numeric coordinates of a
 * mark, so it has to preserve enough resolution to order the points the same
 * way their positions do. `fmtNumber` widens below 1 on its own, so the
 * smallest band here just hands off to it.
 */
function tooltipDecimals(value) {
  const abs = Math.abs(value);
  if (!isFinite(abs) || abs >= 1000) return 0;
  if (abs >= 100) return 1;
  if (abs >= 10) return 2;
  return 3;
}

function fmtAxisValue(value, unit) {
  if (unit === 'ms') return fmtMilliseconds(value);
  if (unit === 'req/s') return fmtReqPerSecond(value);
  return fmtNumber(value, tooltipDecimals(value));
}

function bestPointForX(points, yIsSmallerBetter) {
  const byX = new Map();
  for (const point of points) {
    const current = byX.get(point.x);
    if (!current || (yIsSmallerBetter ? point.y < current.y : point.y > current.y)) {
      byX.set(point.x, point);
    }
  }
  return [...byX.values()].sort((a, b) => a.x - b.x);
}

const MUTED = palette.overlay1;

export function VariationsPareto({ variations, xMetric, yMetric, yIsSmallerBetter }) {
  const chart = useMemo(() => {
    if (!variations || variations.length === 0) return null;
    const points = variations
      .map(v => {
        const xr = v.perMetric?.[xMetric.key + '.' + xMetric.stat];
        const yr = v.perMetric?.[yMetric.key + '.' + yMetric.stat];
        if (!isFiniteNumber(xr?.mean) || !isFiniteNumber(yr?.mean)) return null;
        return {
          x: xr.mean,
          y: yr.mean,
          jobName: pointLabel(v),
          cluster: 'sweep',
        };
      })
      .filter(Boolean);
    if (points.length === 0) return null;

    // Monotone scan: sort by x asc, walk forward, push each point that
    // strictly improves bestY. Equal-y ties do not add another line step,
    // while every variation still remains visible in the scatter dataset.
    const candidates = bestPointForX(points, yIsSmallerBetter);
    const frontier = [];
    let bestY = yIsSmallerBetter ? Infinity : -Infinity;
    for (const p of candidates) {
      const better = yIsSmallerBetter ? p.y < bestY : p.y > bestY;
      if (better) {
        bestY = p.y;
        frontier.push({ x: p.x, y: p.y, jobName: p.jobName });
      }
    }

    const isSingleton = points.length < 2;
    const color = isSingleton ? MUTED : palette.blue;

    const datasets = [
      {
        label: 'Variation (mean across its trials)',
        data: points.map(p => ({ x: p.x, y: p.y, jobName: p.jobName, cluster: p.cluster })),
        backgroundColor: color,
        borderColor: color,
        borderWidth: 1.4,
        pointRadius: 7,
        pointHoverRadius: 11,
        showLine: false,
        order: 1,
      },
    ];
    if (frontier.length >= 2) {
      datasets.push({
        label: 'Pareto frontier',
        data: frontier,
        borderColor: color,
        backgroundColor: color,
        borderWidth: 1.6,
        borderDash: [4, 4],
        showLine: true,
        pointRadius: 0,
        pointHoverRadius: 0,
        fill: false,
        order: 2,
      });
    }

    // When every variation produces the same (x,y) — common in tiny sweeps or
    // when the chosen metric isn't differentiated by the swept dimension —
    // Chart.js auto-scales to a zero-width range and renders dots that hug
    // the axis lines, looking blank. Detect the degenerate case and force a
    // small ±5% pad around the singleton coordinate so the points read clearly.
    const xs = points.map(p => p.x);
    const ys = points.map(p => p.y);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const xCollapsed = xMax === xMin;
    const yCollapsed = yMax === yMin;
    const xPad = xCollapsed ? Math.max(Math.abs(xMin) * 0.05, 1) : undefined;
    const yPad = yCollapsed ? Math.max(Math.abs(yMin) * 0.05, 1) : undefined;

    // No legend. It could only ever carry one usable entry, and every
    // Chart.js legend swatch here renders as the same blue rect, so a
    // two-entry legend would distinguish the scatter from the dashed
    // frontier by text alone -- two identical swatches with different
    // labels. The caption below the chart states both marks in words
    // instead, which is direct labelling rather than a lookup key.
    const options = {
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: palette.mantle,
          titleColor: palette.text,
          bodyColor: palette.text,
          borderColor: palette.surface0,
          borderWidth: 1,
          callbacks: {
            title: ctx => ctx[0]?.raw?.jobName ?? '',
            label: ctx => [
              `${xMetric.label}: ${fmtAxisValue(ctx.raw.x, xMetric.unit)} ${xMetric.unit}`,
              `${yMetric.label}: ${fmtAxisValue(ctx.raw.y, yMetric.unit)} ${yMetric.unit}`,
            ],
          },
        },
      },
      scales: {
        x: {
          type: 'linear',
          title: { display: true, text: `${xMetric.label} (${xMetric.unit})`, color: palette.overlay1, font: { size: CHART_TYPOGRAPHY.AXIS_LABEL } },
          grid: { color: palette.surface0 },
          ticks: { color: palette.overlay1, font: { size: CHART_TYPOGRAPHY.AXIS_TICK }, callback: value => fmtAxisValue(Number(value), xMetric.unit) },
          ...(xCollapsed ? { min: xMin - xPad, max: xMax + xPad } : {}),
        },
        y: {
          type: 'linear',
          title: { display: true, text: `${yMetric.label} (${yMetric.unit})`, color: palette.overlay1, font: { size: CHART_TYPOGRAPHY.AXIS_LABEL } },
          grid: { color: palette.surface0 },
          ticks: { color: palette.overlay1, font: { size: CHART_TYPOGRAPHY.AXIS_TICK }, callback: value => fmtAxisValue(Number(value), yMetric.unit) },
          ...(yCollapsed ? { min: yMin - yPad, max: yMax + yPad } : {}),
        },
      },
    };

    return { datasets, options, frontier };
  }, [variations, xMetric, yMetric, yIsSmallerBetter]);

  if (!chart) {
    return html`<div class="text-dim" style="padding:var(--space-3) 0" data-testid="variations-pareto-empty">
      Awaiting data — need at least one variation with both metrics.
    </div>`;
  }
  // Caption replaces the removed legend. It has to say two things the marks
  // cannot say for themselves: that one dot is a whole variation (a mean over
  // its trials, not a single request) and what the dashed line selects. The
  // frontier direction is stated explicitly because "best" reverses between
  // the throughput-vs-latency axes and a throughput-vs-throughput pairing.
  const frontierDirection = yIsSmallerBetter ? 'lowest' : 'highest';
  return html`
    <div data-testid="sweep-variations-pareto">
      <${ChartWrapper} type="scatter" data=${{ datasets: chart.datasets }} options=${chart.options} height=${360} />
      <div class="text-dim" style="margin-top:var(--space-2);font-size:var(--font-size-xs)" data-testid="variations-pareto-caption">
        Each dot is one variation, plotted at the mean across its trials.
        ${chart.frontier.length >= 2 && html`
          <span> Dashed line is the Pareto frontier — the ${frontierDirection} ${yMetric.label}
          reached at each ${xMetric.label}: ${chart.frontier.map(p => p.jobName).join(' → ')}</span>
        `}
      </div>
    </div>
  `;
}
