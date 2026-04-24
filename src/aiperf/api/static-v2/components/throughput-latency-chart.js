// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Live throughput-vs-latency chart.
 *
 * Two stacked line charts sharing an X axis (wall clock since the first
 * sample). Top panel: request_throughput current. Bottom panel: p99
 * latencies (request_latency, time_to_first_token, inter_token_latency).
 *
 * This is AIPerf's Pareto-lite view during a live run: customers can see
 * whether latency is climbing as throughput rises (a sign they're past
 * the GPU-efficiency sweet spot) without waiting for the benchmark to
 * finish.
 */

import { html } from 'htm/preact';
import { useEffect, useRef } from 'preact/hooks';
import { timeseries } from '../lib/state.js';
import { pluck } from '../lib/timeseries.js';

/** Convert our {t, v} point list into Chart.js {x, y} format. */
function toXY(points, t0) {
  return points.map(p => ({ x: (p.t - t0) / 1000, y: p.v }));
}

/** Earliest sample timestamp across the metrics we chart, or now(). */
function pickT0(ts) {
  let t0 = Infinity;
  for (const tag of [
    'request_throughput', 'output_token_throughput',
    'request_latency', 'time_to_first_token', 'inter_token_latency',
  ]) {
    const series = ts[tag];
    if (series && series.length > 0 && series[0].t < t0) t0 = series[0].t;
  }
  return isFinite(t0) ? t0 : Date.now();
}

function axisLabel(unit) {
  return unit ? `(${unit})` : '';
}

const palette = {
  throughput: '#76b900',      // NVIDIA green
  tokenTput:  '#26c6da',      // cyan
  reqLatency: '#3b82f6',      // blue
  ttft:       '#ab47bc',      // purple
  itl:        '#ffc107',      // amber
  grid:       'rgba(49, 49, 49, 0.45)',
  tick:       '#757575',
  axisLabel:  '#a7a7a7',
};

function buildChartConfig() {
  return {
    type: 'line',
    data: { datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,  // realtime — animated tweens just distract
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          align: 'end',
          labels: {
            color: palette.axisLabel,
            boxWidth: 10,
            boxHeight: 2,
            font: { size: 10, family: "'JetBrains Mono', monospace" },
          },
        },
        tooltip: {
          backgroundColor: 'rgba(12, 12, 12, 0.95)',
          borderColor: palette.grid,
          borderWidth: 1,
          titleColor: '#eeeeee',
          bodyColor: '#a7a7a7',
          titleFont: { family: "'JetBrains Mono', monospace", size: 11 },
          bodyFont:  { family: "'JetBrains Mono', monospace", size: 11 },
          padding: 8,
          callbacks: {
            title: (items) => `t=${items[0].parsed.x.toFixed(1)}s`,
          },
        },
      },
      scales: {
        x: {
          type: 'linear',
          grid:  { color: palette.grid, drawTicks: false },
          ticks: {
            color: palette.tick,
            font: { size: 10 },
            callback: (v) => `${v}s`,
          },
          title: {
            display: true,
            text: 'elapsed',
            color: palette.axisLabel,
            font: { size: 10 },
          },
        },
        yThroughput: {
          type: 'linear',
          position: 'left',
          beginAtZero: true,
          grid:  { color: palette.grid, drawTicks: false },
          ticks: { color: palette.tick, font: { size: 10 } },
          title: {
            display: true,
            text: 'throughput',
            color: palette.axisLabel,
            font: { size: 10 },
          },
        },
        yLatency: {
          type: 'linear',
          position: 'right',
          beginAtZero: true,
          grid:  { display: false },
          ticks: { color: palette.tick, font: { size: 10 } },
          title: {
            display: true,
            text: 'latency (ms)',
            color: palette.axisLabel,
            font: { size: 10 },
          },
        },
      },
    },
  };
}

/** Try multiple stat keys in order; return points for the first one that
 *  yields data. AIPerf throughput metrics populate ``avg`` but not
 *  ``current``, so the chart has to fall back to match what the KPI
 *  sparklines already do for the same tiles. */
function pluckFirst(series, stats, t0) {
  if (!series) return [];
  for (const stat of stats) {
    const pts = toXY(pluck(series, stat), t0);
    if (pts.length > 0) return pts;
  }
  return [];
}

/** Build the four datasets from the current rolling timeseries. */
function buildDatasets(ts) {
  if (!ts || typeof ts !== 'object') return { datasets: [], hasData: false };
  const t0 = pickT0(ts);
  let hasData = false;

  // Throughputs: AIPerf only fills ``avg`` — fall back from current to avg.
  const reqTputPts = pluckFirst(ts.request_throughput, ['current', 'avg'], t0);
  const tokTputPts = pluckFirst(ts.output_token_throughput, ['current', 'avg'], t0);
  // Latencies: p99 is the headline stat, but ``avg`` often lands earlier.
  const reqLatPts  = pluckFirst(ts.request_latency, ['p99', 'avg'], t0);
  const ttftPts    = pluckFirst(ts.time_to_first_token, ['p99', 'avg'], t0);
  const itlPts     = pluckFirst(ts.inter_token_latency, ['p99', 'avg'], t0);

  if (reqTputPts.length || tokTputPts.length || reqLatPts.length
      || ttftPts.length || itlPts.length) hasData = true;

  const common = {
    tension: 0.25,
    borderWidth: 1.8,
    pointRadius: 0,
    pointHitRadius: 8,
    fill: false,
  };

  const datasets = [];
  if (reqTputPts.length) datasets.push({
    label: 'req/s',                   data: reqTputPts,
    borderColor: palette.throughput,  yAxisID: 'yThroughput', ...common,
  });
  if (tokTputPts.length) datasets.push({
    label: 'tok/s',                   data: tokTputPts,
    borderColor: palette.tokenTput,   yAxisID: 'yThroughput', ...common,
    borderDash: [3, 3],
  });
  if (reqLatPts.length) datasets.push({
    label: 'req latency p99 (ms)',    data: reqLatPts,
    borderColor: palette.reqLatency,  yAxisID: 'yLatency',    ...common,
  });
  if (ttftPts.length) datasets.push({
    label: 'TTFT p99 (ms)',           data: ttftPts,
    borderColor: palette.ttft,        yAxisID: 'yLatency',    ...common,
  });
  if (itlPts.length) datasets.push({
    label: 'ITL p99 (ms)',            data: itlPts,
    borderColor: palette.itl,         yAxisID: 'yLatency',    ...common,
  });
  return { datasets, hasData };
}

export function ThroughputLatencyChart() {
  const ts = timeseries.value;
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  const { datasets, hasData } = buildDatasets(ts);

  // Chart lifecycle. Must run AFTER the canvas element actually exists in
  // the DOM — since we conditionally render null before data arrives, keep
  // the creation effect keyed on ``hasData`` so it fires the first time
  // the canvas mounts.
  useEffect(() => {
    if (!hasData) return undefined;
    if (!canvasRef.current) return undefined;
    if (!globalThis.Chart) {
      console.warn('ThroughputLatencyChart: Chart.js not loaded');
      return undefined;
    }
    if (chartRef.current) return undefined;  // already created
    chartRef.current = new globalThis.Chart(canvasRef.current, buildChartConfig());
    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
        chartRef.current = null;
      }
    };
  }, [hasData]);

  // Update datasets whenever they change.
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    chart.data.datasets = datasets;
    chart.update('none');
  }, [JSON.stringify(datasets.map(d => ({ l: d.label, n: d.data.length, last: d.data.at(-1)?.y })))]);

  if (!hasData) return null;

  return html`
    <div class="card">
      <div class="card-title">Throughput vs Latency</div>
      <div class="chart-box">
        <canvas ref=${canvasRef} aria-label="throughput and latency over time"></canvas>
      </div>
    </div>
  `;
}
