// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Eighteen-tile KPI rail for the job-detail page.
 *
 * Layout: 6 cols × 3 rows on ≥900px, 3 cols × 6 rows on laptop (CSS handles).
 * Rows: throughput family · latency family · workload+system.
 *
 * Inputs match the existing job-detail page surface:
 *   - summary  : tag-keyed metric snapshot from REST + WS overlay
 *   - slos     : user-declared SLO thresholds (cfg.slos)
 *   - timeseries: tag-keyed array of {t, values} samples (from WS feed)
 *   - mode     : 'live' | 'completed' | 'archived'
 *   - stale    : optional bool — propagates a "stale" badge to all tiles
 *
 * Missing series → tile shows '—' and an empty sparkline placeholder.
 */

import { html } from 'htm/preact';
import { fmtNumber, fmtInt, fmtPercent } from '../lib/format.js';
import { KpiTile } from './kpi-tile.js';

// 18 tiles, ordered by row.  ``summaryKey`` accepts dotted paths (resolved by
// readPath below).  ``seriesKey`` is the timeseries map key (top-level).
// ``toneRule`` decides good/warn/bad against ``slos`` or local rules.
export const TILE_CONFIG = [
  // ---- Throughput row ----
  { id: 'tok_s',       label: 'tok/s',      unit: 'tok/s',  summaryKey: 'output_token_throughput.avg', seriesKey: 'output_token_throughput', sloKey: 'output_token_throughput', toneRule: 'higher_is_better', fmt: 'thousands' },
  { id: 'req_s',       label: 'req/s',      unit: 'req/s',  summaryKey: 'request_throughput.avg',       seriesKey: 'request_throughput',       sloKey: 'request_throughput',       toneRule: 'higher_is_better', fmt: 'number2' },
  { id: 'concurrency', label: 'conc',       unit: '',       summaryKey: 'concurrency.avg',              seriesKey: 'concurrency',              sloKey: null,                       toneRule: 'neutral',          fmt: 'thousands' },
  { id: 'err_pct',     label: 'err %',      unit: '%',      summaryKey: 'error_rate.avg',               seriesKey: 'error_rate',               sloKey: 'error_rate',               toneRule: 'lower_is_better',  fmt: 'percent2' },
  { id: 'goodput',     label: 'good req/s', unit: 'req/s',  summaryKey: 'goodput.avg',                  seriesKey: 'goodput',                  sloKey: null,                       toneRule: 'higher_is_better', fmt: 'number2' },
  { id: 'in_flight',   label: 'in-flight',  unit: '',       summaryKey: 'in_flight_requests.avg',       seriesKey: 'in_flight_requests',       sloKey: null,                       toneRule: 'neutral',          fmt: 'thousands' },

  // ---- Latency row ----
  { id: 'ttft_p50',    label: 'ttft p50',   unit: 'ms',     summaryKey: 'time_to_first_token.p50',      seriesKey: 'time_to_first_token',      seriesStat: 'p50', sloKey: 'time_to_first_token', toneRule: 'lower_is_better', fmt: 'number0' },
  { id: 'ttft_p99',    label: 'ttft p99',   unit: 'ms',     summaryKey: 'time_to_first_token.p99',      seriesKey: 'time_to_first_token',      seriesStat: 'p99', sloKey: 'time_to_first_token', toneRule: 'lower_is_better', fmt: 'number0' },
  { id: 'itl_p50',     label: 'itl p50',    unit: 'ms/tok', summaryKey: 'inter_token_latency.p50',      seriesKey: 'inter_token_latency',      seriesStat: 'p50', sloKey: 'inter_token_latency', toneRule: 'lower_is_better', fmt: 'number0' },
  { id: 'itl_p99',     label: 'itl p99',    unit: 'ms/tok', summaryKey: 'inter_token_latency.p99',      seriesKey: 'inter_token_latency',      seriesStat: 'p99', sloKey: 'inter_token_latency', toneRule: 'lower_is_better', fmt: 'number0' },
  { id: 'e2e_p50',     label: 'e2e p50',    unit: 'ms',     summaryKey: 'request_latency.p50',          seriesKey: 'request_latency',          seriesStat: 'p50', sloKey: 'request_latency',     toneRule: 'lower_is_better', fmt: 'number0' },
  { id: 'e2e_p99',     label: 'e2e p99',    unit: 'ms',     summaryKey: 'request_latency.p99',          seriesKey: 'request_latency',          seriesStat: 'p99', sloKey: 'request_latency',     toneRule: 'lower_is_better', fmt: 'number0' },

  // ---- Workload + system row ----
  { id: 'isl_avg',     label: 'isl avg',    unit: 'tok',    summaryKey: 'input_sequence_length.avg',    seriesKey: null,                       sloKey: null,                       toneRule: 'neutral',          fmt: 'number0' },
  { id: 'osl_avg',     label: 'osl avg',    unit: 'tok',    summaryKey: 'output_sequence_length.avg',   seriesKey: null,                       sloKey: null,                       toneRule: 'neutral',          fmt: 'number0' },
  { id: 'pods',        label: 'pods',       unit: '',       summaryKey: 'pods_ready.avg',               seriesKey: 'pods_ready',               sloKey: null,                       toneRule: 'pod_health',       fmt: 'pod_ratio' },
  { id: 'gpu_util',    label: 'gpu util',   unit: '%',      summaryKey: 'server_metrics.gpu_util.avg',  seriesKey: 'gpu_util',                 sloKey: null,                       toneRule: 'neutral',          fmt: 'number0' },
  { id: 'kv_cache',    label: 'kv cache',   unit: '%',      summaryKey: 'server_metrics.kv_cache.avg',  seriesKey: 'kv_cache',                 sloKey: null,                       toneRule: 'neutral',          fmt: 'number0' },
  { id: 'records',     label: 'records',    unit: '',       summaryKey: 'records_processed.avg',        seriesKey: 'records_processed',        sloKey: null,                       toneRule: 'records_progress', fmt: 'records_ratio' },
];

const LARGER_IS_BETTER_SET = new Set(['output_token_throughput', 'request_throughput', 'goodput', 'pods_ready']);

// Read a dotted path from a possibly-nested object. Returns null on miss.
function readPath(obj, path) {
  if (!obj || !path) return null;
  let cur = obj;
  for (const seg of path.split('.')) {
    if (cur == null) return null;
    cur = cur[seg];
  }
  return cur ?? null;
}

function pluck(seriesArr, statKey) {
  if (!Array.isArray(seriesArr) || seriesArr.length === 0) return [];
  const out = [];
  for (const s of seriesArr) {
    const v = statKey ? s?.values?.[statKey] : s?.values?.avg ?? s?.value;
    if (typeof v === 'number' && isFinite(v)) out.push({ t: s.t, v });
  }
  return out;
}

function formatValue(value, fmt, summaryEntry) {
  if (value == null || (typeof value === 'number' && !isFinite(value))) return '—';
  switch (fmt) {
    case 'thousands':
      return Math.abs(value) >= 1000 ? `${(value / 1000).toFixed(1)}k` : fmtNumber(value, 1);
    case 'number0': return fmtInt(Math.round(value));
    case 'number2': return fmtNumber(value, 2);
    case 'percent2': return fmtPercent(value, 2);
    case 'pod_ratio': {
      const total = readPath(summaryEntry?.['_full_summary'], 'pods_total.avg');
      return total != null ? `${fmtInt(value)}/${fmtInt(total)}` : fmtInt(value);
    }
    case 'records_ratio': {
      const total = readPath(summaryEntry?.['_full_summary'], 'records_total.avg');
      return total != null ? `${fmtInt(value)}/${fmtInt(total)}` : fmtInt(value);
    }
    default: return fmtNumber(value);
  }
}

function computeDelta(series) {
  if (!series || series.length < 2) return { delta: null, direction: null };
  const last = series[series.length - 1].v;
  const earlierIdx = Math.max(0, series.length - 30);
  const earlier = series[earlierIdx].v;
  if (earlier === 0 || !isFinite(earlier)) return { delta: null, direction: null };
  const pct = ((last - earlier) / Math.abs(earlier)) * 100;
  if (Math.abs(pct) < 0.5) return { delta: '▬ flat', direction: 'flat' };
  const sign = pct >= 0 ? '▲' : '▼';
  return { delta: `${sign} ${Math.abs(pct).toFixed(1)}%`, direction: pct >= 0 ? 'up' : 'down' };
}

function computeTone(rule, value, sloThreshold, sloKey, fullSummary) {
  if (rule === 'neutral') return 'neutral';
  if (rule === 'pod_health') {
    const total = readPath(fullSummary, 'pods_total.avg');
    if (total == null || value == null) return 'neutral';
    const ratio = value / total;
    if (ratio >= 0.99) return 'good';
    if (ratio >= 0.9) return 'warn';
    return 'bad';
  }
  if (rule === 'records_progress') {
    return 'neutral';
  }
  if (sloThreshold == null || value == null) return 'neutral';
  const largerBetter = LARGER_IS_BETTER_SET.has(sloKey) || rule === 'higher_is_better';
  const ok = largerBetter ? value >= sloThreshold : value <= sloThreshold;
  return ok ? 'good' : 'bad';
}

export function KpiRail({ summary, slos, timeseries, mode = 'live', stale = false }) {
  const sum = summary ?? {};
  const ts = timeseries ?? {};
  const sloDict = slos ?? {};

  return html`
    <div class="kpi-rail-grid" data-testid="kpi-rail">
      ${TILE_CONFIG.map((cfg) => {
        const value = readPath(sum, cfg.summaryKey);
        const series = cfg.seriesKey ? pluck(ts[cfg.seriesKey], cfg.seriesStat ?? 'avg') : [];
        const { delta, direction } = computeDelta(series);
        const sloThreshold = cfg.sloKey ? sloDict[cfg.sloKey] ?? null : null;
        const tone = computeTone(cfg.toneRule, value, sloThreshold, cfg.sloKey, sum);
        const summaryEntry = { _full_summary: sum };
        const formatted = formatValue(value, cfg.fmt, summaryEntry);
        return html`
          <${KpiTile}
            tileId=${cfg.id}
            label=${cfg.label}
            value=${formatted}
            unit=${cfg.unit}
            delta=${delta}
            deltaWindow=${delta ? '30s' : null}
            deltaDirection=${direction}
            sparkSeries=${series}
            tone=${tone}
            stale=${stale}
            meta=${mode === 'completed' ? 'final' : (mode === 'archived' ? 'archived' : 'live')}
            key=${cfg.id} />
        `;
      })}
    </div>
  `;
}
