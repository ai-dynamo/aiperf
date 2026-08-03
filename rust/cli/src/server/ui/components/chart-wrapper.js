// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Chart.js lifecycle wrapper for Preact — reused verbatim from the operator
// UI (``components/chart-wrapper.js``). Chart.js is loaded as UMD via <script>
// in index.html and accessed as ``window.Chart``. Only calls ``chart.update()``
// when a fast data/options fingerprint actually changes, so it is safe to
// render inside polling pages.

import { html } from 'htm/preact';
import { useRef, useEffect } from 'preact/hooks';

function dataFingerprint(data) {
  if (!data?.datasets) return '';
  return data.datasets
    .map((ds) =>
      (ds.label ?? '') +
      ':' +
      (ds.data ?? [])
        .map((pt) => {
          if (pt == null) return '';
          return typeof pt === 'object' ? `${pt.x},${pt.y}` : pt;
        })
        .join(';'),
    )
    .join('|');
}

function optionsFingerprint(value, seen = new WeakSet()) {
  if (typeof value === 'function') return `function:${value.toString()}`;
  if (value == null || typeof value !== 'object') return String(value);
  if (seen.has(value)) return '[Circular]';
  seen.add(value);
  if (Array.isArray(value)) return `[${value.map((item) => optionsFingerprint(item, seen)).join(',')}]`;
  return `{${Object.keys(value)
    .sort()
    .map((key) => `${key}:${optionsFingerprint(value[key], seen)}`)
    .join(',')}}`;
}

/**
 * @param {{ type: string, data: object, options?: object, height?: number }} props
 */
export function ChartWrapper({ type, data, options = {}, height = 300 }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);
  const prevFingerprintRef = useRef('');
  const prevOptionsRef = useRef('');

  const hasData =
    !!data?.datasets &&
    data.datasets.length > 0 &&
    data.datasets.some((ds) => (ds.data?.length ?? 0) > 0);

  useEffect(() => {
    if (!canvasRef.current) return;
    if (!hasData) return;
    if (!window.Chart) {
      console.warn('ChartWrapper: window.Chart not available - Chart.js not loaded');
      return;
    }

    chartRef.current = new window.Chart(canvasRef.current, {
      type,
      data,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        ...options,
      },
    });
    prevFingerprintRef.current = dataFingerprint(data);
    prevOptionsRef.current = optionsFingerprint(options);

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
        chartRef.current = null;
      }
    };
  }, [type, hasData]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!chartRef.current) return;
    const fp = dataFingerprint(data);
    if (fp === prevFingerprintRef.current) return;
    prevFingerprintRef.current = fp;
    chartRef.current.data = data;
    chartRef.current.update();
  }, [data]);

  useEffect(() => {
    if (!chartRef.current) return;
    const optStr = optionsFingerprint(options);
    if (optStr === prevOptionsRef.current) return;
    prevOptionsRef.current = optStr;
    chartRef.current.options = {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      ...options,
    };
    chartRef.current.update();
  }, [options]);

  return html`
    <div class="chart-container" style=${'height: ' + height + 'px'}>
      ${hasData
        ? html`<canvas ref=${canvasRef} />`
        : html`<div class="chart-empty">No data to display</div>`}
    </div>
  `;
}

// Shared categorical palette for datasets across the dashboard.
export const CHART_PALETTE = [
  '#76b900', // NVIDIA green
  '#3b82f6', // blue
  '#ab47bc', // purple
  '#26c6da', // cyan
  '#ffc107', // amber
  '#fb923c', // peach
  '#ef5350', // red
  '#10b981', // emerald
];

// Shared Chart.js axis/legend/tooltip theming so every chart reads as one
// system. Callers spread this and add scales/plugins as needed.
export const CHART_THEME = {
  grid: 'rgba(120, 120, 130, 0.14)',
  tick: '#9a9aa2',
  axisLabel: '#c0c0c8',
  tooltipBg: 'rgba(14, 14, 16, 0.96)',
};
