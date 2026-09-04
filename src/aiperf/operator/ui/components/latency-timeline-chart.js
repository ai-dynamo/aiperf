// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Latency timeline chart — copied verbatim from
 * ``operator/ui/views/run.js::LatencyTimelineChart`` and its supporting
 * helpers (``recordLatencyMs``, ``strideSample``, the
 * ``LATENCY_CHART_MAX_POINTS`` constant). End-to-end latency in ms vs
 * request index, sourced from ``profile_export.jsonl``.
 *
 * Stride-samples above 10k points; renders directly with ``window.Chart``
 * (vendored UMD) — bypasses ``ChartWrapper`` to keep the legacy code
 * unchanged. On any render failure, falls back to a plain text badge so
 * a single broken run never blanks the whole view.
 */

import { html } from 'htm/preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import { fmtInt, fmtMilliseconds } from '../lib/format.js';
import { CHART_TYPOGRAPHY } from '../lib/typography.js';

const LATENCY_CHART_MAX_POINTS = 10000;

/** Pull end-to-end latency in ms out of a single ``profile_export.jsonl``
 *  record. The per-request stream stores metric values as
 *  ``metrics.request_latency = {value, unit}`` where ``unit`` is usually
 *  ``"ns"`` (raw) but some transports emit ``"s"``. Returns null when the
 *  metric is missing or the record is an error. */
function recordLatencyMs(rec) {
  if (!rec || rec.error != null) return null;
  const m = rec.metrics?.request_latency;
  if (!m || m.value == null) return null;
  const v = Number(m.value);
  if (!isFinite(v) || v < 0) return null;
  const unit = m.unit ?? 'ns';
  if (unit === 'ns') return v / 1e6;
  if (unit === 'us') return v / 1e3;
  if (unit === 'ms') return v;
  if (unit === 's')  return v * 1e3;
  return v;  // unknown unit — show raw value so it at least plots
}

/** Stride-sample a latency array down to ``max`` points by keeping every Nth
 *  entry. Stride sampling (vs random) preserves monotonic request-index
 *  order, which is what the x-axis encodes. */
function strideSample(values, max) {
  if (values.length <= max) return values;
  const stride = Math.ceil(values.length / max);
  const out = [];
  for (let i = 0; i < values.length; i += stride) out.push(values[i]);
  return out;
}

export function LatencyTimelineChart({ records, loading, skipped }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);
  const [state, setState] = useState({ kind: 'loading' });

  useEffect(() => {
    let cancel = false;
    if (loading) { setState({ kind: 'loading' }); return () => { cancel = true; }; }
    if (skipped) { setState({ kind: 'skip', msg: skipped }); return () => { cancel = true; }; }
    const ms = (records ?? []).map(recordLatencyMs).filter(v => v != null);
    setState(ms.length === 0
      ? { kind: 'skip', msg: 'no latency data' }
      : { kind: 'ok', sampled: strideSample(ms, LATENCY_CHART_MAX_POINTS), total: ms.length });
    return () => { cancel = true; };
  }, [records, loading, skipped]);

  useEffect(() => {
    if (state.kind !== 'ok') return;
    if (!canvasRef.current || !window.Chart) return;

    const points = state.sampled.map((y, x) => ({ x, y }));
    try {
      chartRef.current = new window.Chart(canvasRef.current, {
        type: 'line',
        data: {
          datasets: [{
            label: 'end-to-end latency (ms)',
            data: points,
            borderColor: 'rgba(126, 234, 255, 0.9)',
            backgroundColor: 'rgba(126, 234, 255, 0.15)',
            borderWidth: 1,
            pointRadius: 0,
            fill: false,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          parsing: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: (ctx) => `Latency: ${fmtMilliseconds(ctx.parsed.y)} ms`,
              },
            },
          },
          scales: {
            x: {
              type: 'linear',
              title: { display: true, text: 'request index' },
              ticks: { font: { family: "'JetBrains Mono', monospace", size: CHART_TYPOGRAPHY.AXIS_TICK } },
            },
            y: {
              title: { display: true, text: 'latency (ms)' },
              beginAtZero: true,
              ticks: {
                font: { family: "'JetBrains Mono', monospace", size: CHART_TYPOGRAPHY.AXIS_TICK },
                callback: value => fmtMilliseconds(Number(value)),
              },
            },
          },
        },
      });
    } catch (err) {
      setState({ kind: 'err', msg: `chart render failed: ${err.message}` });
    }

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
        chartRef.current = null;
      }
    };
  }, [state]);

  const header = (meta) => html`
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px">
      <div style="font-size:var(--font-size-xs); text-transform:uppercase; letter-spacing:0.08em; color:var(--dim); font-weight:600">Latency Timeline</div>
      <div style="font-size:var(--font-size-xs); color:var(--muted); font-family:var(--font-mono)">${meta}</div>
    </div>
  `;

  if (state.kind === 'loading') {
    return html`<section class="run-latency-chart" data-testid="run-latency-chart">${header('loading…')}</section>`;
  }
  if (state.kind === 'skip') {
    return html`
      <section class="run-latency-chart" data-testid="run-latency-chart">
        ${header(state.msg)}
      </section>
    `;
  }
  if (state.kind === 'err') {
    return html`
      <section class="run-latency-chart" data-testid="run-latency-chart">
        ${header('chart unavailable')}
        <div class="empty">${state.msg}</div>
      </section>
    `;
  }

  const { sampled, total } = state;
  const metaText = total > sampled.length
    ? `${fmtInt(sampled.length)} / ${fmtInt(total)} requests · sampled`
    : `${fmtInt(total)} requests`;
  return html`
    <section class="run-latency-chart" data-testid="run-latency-chart">
      ${header(metaText)}
      <div class="chart-box" style="height: 300px;">
        <canvas ref=${canvasRef}></canvas>
      </div>
    </section>
  `;
}
