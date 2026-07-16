// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { fmtInt, fmtNumber, fmtThroughput } from '../../lib/format.js';
import { KpiCard } from '../kpi-card.js';
import { curateServerMetrics, normalizeServerMetrics } from './helpers.js';

function formatDuration(seconds) {
  if (seconds == null) return '—';
  if (seconds < 60) return `${fmtNumber(seconds, 0)}s`;
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.round(seconds - minutes * 60);
  return remainingSeconds > 0 ? `${minutes}m ${remainingSeconds}s` : `${minutes}m`;
}

function formatKpiValue(kpi) {
  if (kpi.value == null) return '—';
  if (kpi.unit === 'req/s' || kpi.unit === 'tok/s') return fmtThroughput(kpi.value);
  if (kpi.unit === '%') return fmtNumber(kpi.value, 1);
  if (kpi.unit === 'ms') return fmtNumber(kpi.value, kpi.value >= 100 ? 0 : 1);
  return fmtInt(kpi.value);
}

function formatDetailValue(value, unit) {
  if (value == null) return '—';
  if (unit === 'rate') return fmtThroughput(value);
  if (unit === 'percent') return `${fmtNumber(value, 1)}%`;
  if (unit === 'ms') return `${fmtNumber(value, value >= 100 ? 0 : 1)} ms`;
  return fmtInt(value);
}

function SummaryStrip({ summary }) {
  const endpointText = summary.endpointsSuccessful == null || summary.endpointsConfigured == null
    ? '—'
    : `${summary.endpointsSuccessful}/${summary.endpointsConfigured}`;
  return html`
    <div class="server-metrics-summary-strip">
      <div class="live-metric">
        <span class="live-metric-label">Endpoints</span>
        <span class="live-metric-value">${endpointText}</span>
        <span class="live-metric-unit">successful/configured</span>
      </div>
      <div class="live-metric">
        <span class="live-metric-label">Backends</span>
        <span class="live-metric-value">${summary.backends.length > 0 ? summary.backends.join(', ') : '—'}</span>
      </div>
      <div class="live-metric">
        <span class="live-metric-label">Scrape window</span>
        <span class="live-metric-value">${formatDuration(summary.durationSeconds)}</span>
      </div>
    </div>
  `;
}

function DetailsTable({ rows, sources }) {
  return html`
    <details class="server-metrics-details">
      <summary>Per-endpoint details and source metrics</summary>
      ${rows.length > 0 && html`
        <div class="server-metrics-table-wrap">
          <table class="per-endpoint-table">
            <thead>
              <tr>
                <th>Endpoint</th>
                <th>Backend</th>
                <th>Req/s</th>
                <th>Gen tok/s</th>
                <th>KV/cache</th>
                <th>Waiting</th>
                <th>p99 latency</th>
              </tr>
            </thead>
            <tbody>
              ${rows.map((row, i) => html`
                <tr key=${i}>
                  <td title=${row.endpoint} class="server-metrics-endpoint-cell">${row.endpoint || '—'}</td>
                  <td>${row.backend}</td>
                  <td>${formatDetailValue(row.reqRate, 'rate')}</td>
                  <td>${formatDetailValue(row.genRate, 'rate')}</td>
                  <td>${formatDetailValue(row.kvPressure, 'percent')}</td>
                  <td>${formatDetailValue(row.waiting, 'count')}</td>
                  <td>${formatDetailValue(row.latencyP99Ms, 'ms')}</td>
                </tr>
              `)}
            </tbody>
          </table>
        </div>
      `}
      ${sources.length > 0 && html`
        <div class="server-metrics-sources">
          <span class="text-dim">Sources:</span>
          ${sources.map(source => html`<code key=${source}>${source}</code>`)}
        </div>
      `}
    </details>
  `;
}

/**
 * Compact curated server-metrics section for completed job artifacts.
 *
 * @param {object} props
 * @param {object|null} props.serverMetrics - Parsed server_metrics_export.json
 *   or null if not yet loaded. The parent decides whether to render at all.
 */
export function ServerMetricsSection({ serverMetrics, source = 'final', sparklines = null }) {
  if (!serverMetrics) return null;
  const curated = curateServerMetrics(normalizeServerMetrics(serverMetrics), sparklines);
  const sourceLabel = source === 'live' ? 'LIVE' : 'FINAL';

  if (!curated) {
    return html`
      <div style="margin-top: var(--space-4)">
        <div class="card" data-testid="job-detail-server-metrics-empty">
          <div class="card-title">Server Metrics <span class="metric-source-chip">${sourceLabel}</span></div>
          <div class="text-dim" style="font-size: var(--font-size-sm); padding: var(--space-2) 0">
            No server metrics collected for this run. The endpoint did not expose compatible metrics, or the scrape interval did not capture any points.
          </div>
        </div>
      </div>
    `;
  }

  return html`
    <div class="server-metrics-section" style="margin-top: var(--space-4)">
      <div class="card" data-testid="job-detail-server-metrics-curated">
        <div class="card-title">Server Metrics <span class="metric-source-chip">${sourceLabel}</span></div>
        <${SummaryStrip} summary=${curated.summary} />
        ${curated.kpis.length > 0 && html`
          <div class="kpi-row server-metrics-kpis">
            ${curated.kpis.map(kpi => html`
              <${KpiCard}
                key=${kpi.id}
                label=${kpi.label}
                icon=${kpi.icon}
                tone=${kpi.tone}
                value=${formatKpiValue(kpi)}
                unit=${kpi.unit}
                sub=${kpi.sub}
                progress=${kpi.progress}
                title=${kpi.source ? `Source: ${kpi.source} (${kpi.stat})` : ''}
                sparkline=${kpi.points && kpi.points.length > 1 ? { points: kpi.points } : null}
              />
            `)}
          </div>
        `}
        <${DetailsTable} rows=${curated.detailRows} sources=${curated.sources} />
      </div>
    </div>
  `;
}
