// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Shared full-stat metrics table for benchmark, GPU, and server metrics. */

import { html } from 'htm/preact';
import { fmtInt, fmtNumber } from '../lib/format.js';

const STAT_COLUMNS = ['avg', 'min', 'max', 'p99', 'p90', 'p50'];

function isRecord(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function finiteStat(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function formatStat(value) {
  const finiteValue = finiteStat(value);
  if (finiteValue == null) return '---';
  return Math.abs(finiteValue) >= 1000 ? fmtInt(finiteValue) : fmtNumber(finiteValue, 2);
}

function rowStats(source) {
  return Object.fromEntries(
    STAT_COLUMNS.map((key) => [
      key,
      key === 'avg' ? finiteStat(source.avg) ?? finiteStat(source.current) : finiteStat(source[key]),
    ]),
  );
}

export function rowsFromMetrics(metrics) {
  if (!Array.isArray(metrics)) return [];
  return metrics.flatMap((metric, index) => {
    if (!isRecord(metric)) return [];
    const metricKey = metric.tag || metric.name || `metric-${index}`;
    const metricLabel = metric.header || metric.tag || metric.name || '---';
    return [{
      key: metricKey,
      metric: metricLabel,
      unit: metric.unit ?? '',
      ...rowStats(metric),
    }];
  });
}

export function rowsFromServerMetrics(summaries) {
  if (!Array.isArray(summaries)) return [];
  return summaries.flatMap((summary) => {
    if (!isRecord(summary) || !summary.endpoint || !Array.isArray(summary.metrics)) return [];
    const endpoint = summary.endpoint;
    return summary.metrics.flatMap((metric) => {
      if (!isRecord(metric) || !metric.name) return [];
      return [{
        key: `${endpoint}::${metric.name}`,
        metric: `${endpoint} · ${metric.name}`,
        unit: metric.unit ?? '',
        ...rowStats(metric),
      }];
    });
  });
}

export function FullMetricsTable({ title, rows, emptyText = 'No metrics reported yet.' }) {
  const safeRows = Array.isArray(rows) ? rows.filter(isRecord) : [];
  if (safeRows.length === 0) return null;

  return html`
    <div class="card full-metrics-card">
      <div class="card-title">
        <span>${title}</span>
        <span class="card-count">${safeRows.length}</span>
      </div>
      <div class="full-metrics-scroll">
        <table class="full-metrics-table" aria-label=${title ?? emptyText}>
          <thead>
            <tr>
              <th>Metric</th>
              ${STAT_COLUMNS.map((column) => html`<th style="text-align: right">${column}</th>`)}
            </tr>
          </thead>
          <tbody>
            ${safeRows.map((row, index) => html`
              <tr key=${row.key ?? index}>
                <td>
                  <span class="full-metric-name">${row.metric ?? row.key ?? '---'}</span>
                  ${row.unit ? html`<span class="full-metric-unit">${row.unit}</span>` : ''}
                </td>
                ${STAT_COLUMNS.map((column) => html`
                  <td style="text-align: right">${formatStat(row[column])}</td>
                `)}
              </tr>
            `)}
          </tbody>
        </table>
      </div>
    </div>
  `;
}
