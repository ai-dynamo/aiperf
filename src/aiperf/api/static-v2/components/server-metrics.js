// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Per-endpoint server metrics summary with context tooltips + saturation
 * color coding on the few metrics that have meaningful "healthy vs hot"
 * bands (KV cache utilization and request queue depth).
 *
 * The bands are not customer SLOs — they are operational guardrails
 * documented by the serving-stack vendors (vLLM, TGI). A tooltip calls
 * that out so the color scheme isn't mistaken for an SLO claim.
 */

import { html } from 'htm/preact';
import { serverMetrics } from '../lib/state.js';
import { fmtNumber } from '../lib/format.js';

/** Known operational guardrails. ``kind`` categorizes the color band only;
 *  it is not an assertion about the user's SLO. */
function saturationForKvCache(ratio) {
  if (ratio == null || typeof ratio !== 'number') return null;
  if (ratio < 0.70) return { kind: 'good', note: 'headroom — plenty of KV cache left' };
  if (ratio < 0.90) return { kind: 'warn', note: 'approaching saturation — queuing likely past 90%' };
  return { kind: 'bad',  note: 'saturated — expect TTFT spikes and request queueing' };
}

function saturationForQueueDepth(depth) {
  if (depth == null || typeof depth !== 'number') return null;
  if (depth < 10) return { kind: 'good', note: 'shallow queue — requests move straight through' };
  if (depth < 50) return { kind: 'warn', note: 'queue building — latency tail will start growing' };
  return { kind: 'bad',  note: 'deep queue — new requests will wait behind this backlog' };
}

/** Return {kind, note} or null. */
function saturationFor(metric) {
  const name = (metric?.name ?? '').toLowerCase();
  const v = metric?.value;
  if (name.includes('kv_cache') && name.includes('util')) return saturationForKvCache(v);
  if (name === 'queue_depth' || name.endsWith('_queue_depth')) return saturationForQueueDepth(v);
  return null;
}

function formatValue(value, unit) {
  if (value == null) return '---';
  if (typeof value !== 'number') return String(value);
  const body = Math.abs(value) >= 1000
    ? Math.round(value).toLocaleString('en-US')
    : fmtNumber(value, 2);
  return unit ? `${body} ${unit}` : body;
}

export function ServerMetrics() {
  const summaries = serverMetrics.value;

  return html`
    <div class="card">
      <div class="card-title">Server Metrics</div>
      ${summaries.length === 0
        ? html`<div class="empty">No server-side metrics reported yet.</div>`
        : summaries.map((s) => html`
          <div key=${s.endpoint ?? ''} style="margin-bottom: var(--space-4)">
            <div style="font-family: var(--font-mono); font-size: 11px; color: var(--sub); margin-bottom: var(--space-2)">
              ${s.endpoint ?? '---'}
            </div>
            <table class="server-metrics">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th style="text-align: right">Value</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                ${(s.metrics ?? []).map((m, i) => {
                  const sat = saturationFor(m);
                  return html`
                    <tr key=${m.name ?? i}
                        class=${sat ? 'server-metrics-row server-metrics-row--' + sat.kind : 'server-metrics-row'}
                        title=${sat ? sat.note : ''}>
                      <td>${m.name ?? '---'}</td>
                      <td style="text-align: right">${formatValue(m.value, m.unit)}</td>
                      <td style="text-align: right; width: 80px">
                        ${sat
                          ? html`<span class=${'server-chip server-chip--' + sat.kind}>${
                              sat.kind === 'good' ? 'headroom'
                              : sat.kind === 'warn' ? 'building'
                              : 'saturated'
                            }</span>`
                          : ''}
                      </td>
                    </tr>
                  `;
                })}
              </tbody>
            </table>
          </div>
        `)
      }
    </div>
  `;
}
