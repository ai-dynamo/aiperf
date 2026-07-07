// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { useMemo } from 'preact/hooks';
import { palette } from '../lib/theme.js';
import { fmtNumber, fmtInt } from '../lib/format.js';
import { ChartWrapper } from './chart-wrapper.js';

/**
 * Pareto-frontier scatter for a sweep — mirrors the legacy ui's
 * ``analysis.js`` pattern: sort points ascending in x, walk left-to-right
 * tracking the best-so-far on y, push on improvement. Output is a
 * naturally-sorted frontier rendered as a dashed line; dominated points
 * stay rendered as muted scatter dots.
 *
 * Props:
 *   variations: [{ variation_index, label,
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

function isFiniteNumber(value) {
  return typeof value === 'number' && isFinite(value);
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
          jobName: shortLabel(v.label) || `v${v.variation_index}`,
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
        label: 'sweep',
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
        label: 'sweep · frontier',
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
        legend: false,
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

    const options = {
      plugins: {
        legend: {
          position: 'top',
          align: 'end',
          labels: {
            color: palette.text,
            usePointStyle: true,
            pointStyle: 'rect',
            boxWidth: 10,
            padding: 16,
            font: { size: 11 },
            filter: (item, data) => {
              const ds = data.datasets[item.datasetIndex];
              return ds && ds.legend !== false;
            },
          },
        },
        tooltip: {
          backgroundColor: palette.mantle,
          titleColor: palette.text,
          bodyColor: palette.text,
          borderColor: palette.surface0,
          borderWidth: 1,
          callbacks: {
            title: ctx => ctx[0]?.raw?.jobName ?? '',
            label: ctx => [
              `${xMetric.label}: ${fmtNumber(ctx.raw.x, 0)} ${xMetric.unit}`,
              `${yMetric.label}: ${fmtInt(ctx.raw.y)} ${yMetric.unit}`,
            ],
          },
        },
      },
      scales: {
        x: {
          type: 'linear',
          title: { display: true, text: `${xMetric.label} (${xMetric.unit})`, color: palette.overlay1, font: { size: 11 } },
          grid: { color: palette.surface0 },
          ticks: { color: palette.overlay1, font: { size: 10 } },
          ...(xCollapsed ? { min: xMin - xPad, max: xMax + xPad } : {}),
        },
        y: {
          type: 'linear',
          title: { display: true, text: `${yMetric.label} (${yMetric.unit})`, color: palette.overlay1, font: { size: 11 } },
          grid: { color: palette.surface0 },
          ticks: { color: palette.overlay1, font: { size: 10 } },
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
  return html`
    <div data-testid="sweep-variations-pareto">
      <${ChartWrapper} type="scatter" data=${{ datasets: chart.datasets }} options=${chart.options} height=${360} />
      ${chart.frontier.length >= 2 && html`
        <div class="text-dim" style="margin-top:var(--space-2);font-size:var(--font-size-xs)">
          frontier: ${chart.frontier.map(p => p.jobName).join(' → ')}
        </div>
      `}
    </div>
  `;
}
