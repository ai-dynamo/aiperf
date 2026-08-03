// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Pareto scatter for cross-run comparison — one point per run. X is throughput
// (larger better), Y is a latency percentile (smaller better). Non-dominated
// points are joined by a dashed frontier line. Frontier logic mirrors the
// operator UI's ``variations-pareto.js`` monotone scan.

import { html } from 'htm/preact';
import { useMemo } from 'preact/hooks';
import { ChartWrapper, CHART_PALETTE, CHART_THEME } from './chart-wrapper.js';
import { fmtNumber } from '../lib/format.js';

/**
 * Non-dominated subset. ``xLargerBetter`` / ``yLargerBetter`` orient which
 * direction wins on each axis. O(n^2) — n is the number of selected runs.
 */
function paretoFrontier(points, xLargerBetter, yLargerBetter) {
  if (!points || points.length < 2) return points ? points.slice() : [];
  const dominates = (a, b) => {
    const xBeat = xLargerBetter ? a.x >= b.x : a.x <= b.x;
    const yBeat = yLargerBetter ? a.y >= b.y : a.y <= b.y;
    const strict = a.x !== b.x || a.y !== b.y;
    return xBeat && yBeat && strict;
  };
  return points
    .filter((p) => !points.some((q) => q !== p && dominates(q, p)))
    .slice()
    .sort((a, b) => a.x - b.x);
}

/**
 * @param {object} props
 * @param {Array<{x:number,y:number,label:string}>} props.points
 * @param {{label:string,unit:string,largerBetter:boolean}} props.xAxis
 * @param {{label:string,unit:string,largerBetter:boolean}} props.yAxis
 */
export function ParetoChart({ points, xAxis, yAxis, height = 360 }) {
  const chart = useMemo(() => {
    const clean = (points ?? []).filter(
      (p) => typeof p.x === 'number' && isFinite(p.x) && typeof p.y === 'number' && isFinite(p.y),
    );
    if (clean.length === 0) return null;

    const frontier = paretoFrontier(clean, xAxis.largerBetter, yAxis.largerBetter);

    const datasets = [
      {
        label: 'runs',
        data: clean.map((p, i) => ({ ...p, _c: CHART_PALETTE[i % CHART_PALETTE.length] })),
        pointBackgroundColor: clean.map((_, i) => CHART_PALETTE[i % CHART_PALETTE.length]),
        pointBorderColor: clean.map((_, i) => CHART_PALETTE[i % CHART_PALETTE.length]),
        pointRadius: 7,
        pointHoverRadius: 11,
        showLine: false,
        order: 1,
      },
    ];
    if (frontier.length >= 2) {
      datasets.push({
        label: 'frontier',
        data: frontier.map((p) => ({ x: p.x, y: p.y, label: p.label })),
        borderColor: '#76b900',
        backgroundColor: '#76b900',
        borderWidth: 1.6,
        borderDash: [5, 4],
        showLine: true,
        pointRadius: 0,
        pointHoverRadius: 0,
        fill: false,
        order: 2,
      });
    }

    const options = {
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: CHART_THEME.tooltipBg,
          titleColor: '#ececec',
          bodyColor: '#c0c0c8',
          borderColor: CHART_THEME.grid,
          borderWidth: 1,
          callbacks: {
            title: (items) => items[0]?.raw?.label ?? '',
            label: (ctx) => {
              const p = ctx.raw;
              return [
                `${xAxis.label}: ${fmtNumber(p.x, 2)}${xAxis.unit ? ' ' + xAxis.unit : ''}`,
                `${yAxis.label}: ${fmtNumber(p.y, 2)}${yAxis.unit ? ' ' + yAxis.unit : ''}`,
              ];
            },
          },
        },
      },
      scales: {
        x: {
          type: 'linear',
          title: {
            display: true,
            text: xAxis.unit ? `${xAxis.label} (${xAxis.unit})` : xAxis.label,
            color: CHART_THEME.axisLabel,
            font: { size: 11 },
          },
          grid: { color: CHART_THEME.grid },
          ticks: { color: CHART_THEME.tick, font: { size: 10 } },
        },
        y: {
          type: 'linear',
          title: {
            display: true,
            text: yAxis.unit ? `${yAxis.label} (${yAxis.unit})` : yAxis.label,
            color: CHART_THEME.axisLabel,
            font: { size: 11 },
          },
          grid: { color: CHART_THEME.grid },
          ticks: { color: CHART_THEME.tick, font: { size: 11 } },
        },
      },
    };

    return { datasets, options, frontier };
  }, [points, xAxis, yAxis]);

  if (!chart) {
    return html`<div class="empty">Need at least one run with both ${xAxis.label} and ${yAxis.label}.</div>`;
  }

  return html`
    <${ChartWrapper} type="scatter" data=${{ datasets: chart.datasets }} options=${chart.options} height=${height} />
    ${chart.frontier.length >= 2 &&
    html`<div class="dim caption">frontier: ${chart.frontier.map((p) => p.label).join(' → ')}</div>`}
  `;
}
