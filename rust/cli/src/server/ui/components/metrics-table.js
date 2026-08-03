// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Sortable full-metrics table for the run-detail page. Consumes the projected
// ``metrics`` map from ``/api/runs/:id/summary`` — each entry has
// ``{ avg, min, max, count, sum, percentiles:{p1..p99}, unit }``. Any column
// can be absent (renders an em-dash); clicking a header sorts by it.

import { html } from 'htm/preact';
import { useState, useMemo } from 'preact/hooks';
import { fmtMetric, fmtInt, prettyTag } from '../lib/format.js';

// column key -> (row) => numeric value | null, plus its header label.
const COLUMNS = [
  { key: 'metric', label: 'Metric', numeric: false },
  { key: 'avg', label: 'avg', numeric: true },
  { key: 'min', label: 'min', numeric: true },
  { key: 'max', label: 'max', numeric: true },
  { key: 'p50', label: 'p50', numeric: true },
  { key: 'p90', label: 'p90', numeric: true },
  { key: 'p99', label: 'p99', numeric: true },
  { key: 'count', label: 'count', numeric: true },
  { key: 'unit', label: 'unit', numeric: false },
];

function cellValue(row, key) {
  if (key === 'metric') return row.tag;
  if (key === 'unit') return row.unit ?? '';
  if (key === 'count') return row.count;
  if (['p50', 'p90', 'p99'].includes(key)) return row.percentiles?.[key] ?? null;
  return row[key] ?? null;
}

function toRows(metrics) {
  if (!metrics || typeof metrics !== 'object') return [];
  return Object.entries(metrics).map(([tag, m]) => ({ tag, ...m }));
}

export function MetricsTable({ metrics }) {
  const [sortKey, setSortKey] = useState('metric');
  const [sortDir, setSortDir] = useState('asc');

  const rows = useMemo(() => {
    const all = toRows(metrics);
    const dir = sortDir === 'asc' ? 1 : -1;
    return all.sort((a, b) => {
      const av = cellValue(a, sortKey);
      const bv = cellValue(b, sortKey);
      if (typeof av === 'string' || typeof bv === 'string') {
        return String(av ?? '').localeCompare(String(bv ?? '')) * dir;
      }
      // Nulls sort last regardless of direction.
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      return (av - bv) * dir;
    });
  }, [metrics, sortKey, sortDir]);

  function onSort(key) {
    if (key === sortKey) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir(key === 'metric' || key === 'unit' ? 'asc' : 'desc');
    }
  }

  if (rows.length === 0) {
    return html`<div class="empty">No metrics reported.</div>`;
  }

  const arrow = (key) => (key === sortKey ? (sortDir === 'asc' ? ' ▲' : ' ▼') : '');

  return html`
    <div class="table-scroll">
      <table class="data-table">
        <thead>
          <tr>
            ${COLUMNS.map(
              (c) => html`
                <th
                  key=${c.key}
                  class=${'sortable' + (c.numeric ? ' num' : '')}
                  onClick=${() => onSort(c.key)}
                >
                  ${c.label}${arrow(c.key)}
                </th>
              `,
            )}
          </tr>
        </thead>
        <tbody>
          ${rows.map(
            (row) => html`
              <tr key=${row.tag}>
                <td class="mono" title=${row.tag}>${prettyTag(row.tag)}</td>
                <td class="num">${fmtMetric(row.avg)}</td>
                <td class="num">${fmtMetric(row.min)}</td>
                <td class="num">${fmtMetric(row.max)}</td>
                <td class="num">${fmtMetric(row.percentiles?.p50)}</td>
                <td class="num">${fmtMetric(row.percentiles?.p90)}</td>
                <td class="num">${fmtMetric(row.percentiles?.p99)}</td>
                <td class="num">${fmtInt(row.count)}</td>
                <td class="dim">${row.unit ?? ''}</td>
              </tr>
            `,
          )}
        </tbody>
      </table>
    </div>
  `;
}
