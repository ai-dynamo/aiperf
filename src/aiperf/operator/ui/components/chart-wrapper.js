// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { useRef, useEffect } from 'preact/hooks';

/**
 * Compute a fast fingerprint of chart data to detect actual changes.
 * Extracts only the numeric values that matter for rendering.
 *
 * Note ``typeof null === 'object'``: scatter datasets with sparse data
 * may interleave nulls between point objects, so the object branch must
 * also guard against null before reading ``pt.x`` / ``pt.y``.
 */
function dataFingerprint(data) {
  if (!data?.datasets) return '';
  return data.datasets.map(ds =>
    (ds.label ?? '') + ':' + (ds.data ?? []).map(pt => {
      if (pt == null) return '';
      return typeof pt === 'object' ? `${pt.x},${pt.y}` : pt;
    }).join(';')
  ).join('|');
}

function optionsFingerprint(value, seen = new WeakSet()) {
  if (typeof value === 'function') return `function:${value.toString()}`;
  if (value == null || typeof value !== 'object') return String(value);
  if (seen.has(value)) return '[Circular]';
  seen.add(value);
  if (Array.isArray(value)) return `[${value.map(item => optionsFingerprint(item, seen)).join(',')}]`;
  return `{${Object.keys(value).sort().map(key => `${key}:${optionsFingerprint(value[key], seen)}`).join(',')}}`;
}

/**
 * Chart.js lifecycle wrapper for Preact.
 * Chart.js is loaded as UMD via <script> in index.html, accessed as window.Chart.
 *
 * Only calls chart.update() when the data fingerprint actually changes,
 * preventing unnecessary redraws during polling cycles with no new data.
 *
 * @param {{ type: string, data: object, options?: object, height?: number }} props
 */
export function ChartWrapper({ type, data, options = {}, height = 300 }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);
  const prevFingerprintRef = useRef('');
  const prevOptionsRef = useRef('');

  // Treat null/undefined data and zero-dataset data as "no data" — Chart.js
  // throws or renders a blank canvas otherwise, which looks like a load bug.
  const hasData = !!data?.datasets && data.datasets.length > 0
    && data.datasets.some(ds => (ds.data?.length ?? 0) > 0);

  // Create chart on mount and whenever `type` changes (Chart.js cannot mutate
  // chart type in place — it must be destroyed and recreated).
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
  }, [type, hasData]); // eslint-disable-line react-hooks/exhaustive-deps - data/options handled by their own effects

  // Update data only when fingerprint changes
  useEffect(() => {
    if (!chartRef.current) return;
    const fp = dataFingerprint(data);
    if (fp === prevFingerprintRef.current) return;
    prevFingerprintRef.current = fp;
    chartRef.current.data = data;
    chartRef.current.update();
  }, [data]);

  // Update options when callbacks or plain option values change.
  useEffect(() => {
    if (!chartRef.current) return;
    const optStr = optionsFingerprint(options);
    if (optStr === prevOptionsRef.current) return;
    prevOptionsRef.current = optStr;
    chartRef.current.options = { responsive: true, maintainAspectRatio: false, animation: { duration: 300 }, ...options };
    chartRef.current.update();
  }, [options]);

  return html`
    <div class="chart-container" style=${'height: ' + height + 'px'}>
      ${hasData
        ? html`<canvas ref=${canvasRef} />`
        : html`<div
            class="chart-empty"
            style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--dim);font-size: var(--font-size-xs)"
          >No data to display</div>`
      }
    </div>
  `;
}
