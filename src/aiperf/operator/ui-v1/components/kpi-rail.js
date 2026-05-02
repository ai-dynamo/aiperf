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
import { KpiCard } from './kpi-card.js';

// Hero tile id → KpiCard icon registry key. Keeps the hero/secondary mapping
// the only place we hardcode hero membership; everything else stays generic.
const HERO_ICONS = {
  tok_s: 'trending-up',
  goodput: 'goodput',
  ttft_p99: 'clock',
  e2e_tok_s: 'tokens',
};

// Priority-ordered hero candidate list. selectHero walks these in order and
// keeps the first up-to-3 whose tile resolved to a non-null value. The list
// is longer than 3 so non-streaming runs (no TTFT) still get a 3-up rail.
const HERO_CANDIDATES = ['tok_s', 'goodput', 'ttft_p99', 'e2e_tok_s', 'req_s'];

// Tone → KpiCard tone. KpiRail uses 'good'/'warn'/'bad'/'neutral'; KpiCard
// accepts 'ok'/'warn'/'bad'/'accent'/'neutral'/'gold'/'live'.
function railToneToCardTone(tone) {
  if (tone === 'good') return 'ok';
  if (tone === 'warn') return 'warn';
  if (tone === 'bad') return 'bad';
  return 'neutral';
}

// Hero trend badge: compute % delta between sparkline last and ~10 points
// back. Returns null when fewer than 10 points (run too short / no series).
// ``goodDirection`` says whether "up" is the desirable direction for this
// metric (throughput up = good; latency up = bad).
function computeTrend(series, goodDirection) {
  if (!Array.isArray(series) || series.length < 10) return null;
  const last = series[series.length - 1].v;
  const prev = series[series.length - 10].v;
  if (!isFinite(last) || !isFinite(prev) || prev === 0) return null;
  const pct = ((last - prev) / Math.abs(prev)) * 100;
  if (!isFinite(pct)) return null;
  const direction = pct >= 0 ? 'up' : 'down';
  const good = (direction === 'up') === goodDirection;
  return { delta: pct, direction, good };
}

// 18 tiles, ordered by row.  ``summaryKey`` accepts dotted paths (resolved by
// readPath below).  ``seriesKey`` is the timeseries map key (top-level).
// ``toneRule`` decides good/warn/bad against ``slos`` or local rules.
export const TILE_CONFIG = [
  // ---- Throughput row ----
  { id: 'tok_s',       label: 'tok/s',      unit: 'tok/s',  summaryKey: 'output_token_throughput.avg', seriesKey: 'output_token_throughput', sloKey: 'output_token_throughput', toneRule: 'higher_is_better', fmt: 'thousands' },
  { id: 'e2e_tok_s',   label: 'e2e tok/s',  unit: 'tok/s',  summaryKey: 'e2e_output_token_throughput.avg', seriesKey: 'e2e_output_token_throughput', sloKey: null, toneRule: 'higher_is_better', fmt: 'thousands' },
  { id: 'req_s',       label: 'req/s',      unit: 'req/s',  summaryKey: 'request_throughput.avg',       seriesKey: 'request_throughput',       sloKey: 'request_throughput',       toneRule: 'higher_is_better', fmt: 'number2' },
  { id: 'concurrency', label: 'conc',       unit: '',       summaryKey: null,                           seriesKey: null,                       resolverKey: 'concurrency_from_phases', sloKey: null,                       toneRule: 'neutral',          fmt: 'thousands' },
  { id: 'err_pct',     label: 'err %',      unit: '%',      summaryKey: 'error_rate.avg',               seriesKey: 'error_rate',               sloKey: 'error_rate',               toneRule: 'lower_is_better',  fmt: 'percent2' },
  { id: 'goodput',     label: 'good req/s', unit: 'req/s',  summaryKey: 'goodput.avg',                  seriesKey: 'goodput',                  sloKey: null,                       toneRule: 'higher_is_better', fmt: 'number2' },
  { id: 'in_flight',   label: 'in-flight',  unit: '',       summaryKey: null,                           seriesKey: null,                       resolverKey: 'in_flight_from_summary',  sloKey: null,                       toneRule: 'neutral',          fmt: 'thousands' },

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
  { id: 'pods',        label: 'pods',       unit: '',       summaryKey: null,                           seriesKey: null,                       resolverKey: 'pods_ratio',          sloKey: null,                       toneRule: 'pod_health',       fmt: 'pod_ratio' },
  { id: 'gpu_util',    label: 'gpu util',   unit: '%',      summaryKey: null,                           seriesKey: null,                       resolverKey: 'gpu_util_server',     sloKey: null,                       toneRule: 'neutral',          fmt: 'number0' },
  { id: 'kv_cache',    label: 'kv cache',   unit: '%',      summaryKey: null,                           seriesKey: null,                       resolverKey: 'kv_cache_server',     sloKey: null,                       toneRule: 'neutral',          fmt: 'number0' },
  { id: 'records',     label: 'records',    unit: '',       summaryKey: null,                           seriesKey: null,                       resolverKey: 'records_progress',    sloKey: null,                       toneRule: 'records_progress', fmt: 'records_ratio' },
];

const LARGER_IS_BETTER_SET = new Set(['output_token_throughput', 'e2e_output_token_throughput', 'request_throughput', 'goodput', 'pods_ready']);

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

// Resolvers for tiles whose data lives outside the flat ``summary`` dict (phases,
// pod CR list, server-metrics summary/timeseries). Each returns
// `{ value, series, _full? }` — _full carries extra context (ready/total,
// processed/total) consumed by the formatter.
const RESOLVERS = {
  concurrency_from_phases: ({ phases }) => {
    const list = Array.isArray(phases) ? phases : [];
    const active = list.find((p) => p?.status === 'active');
    if (active && active.targetConcurrency != null) return { value: active.targetConcurrency, series: [] };
    const last = list[list.length - 1];
    if (last && last.targetConcurrency != null) return { value: last.targetConcurrency, series: [] };
    return { value: null, series: [] };
  },
  in_flight_from_summary: ({ summary }) => {
    const v = readPath(summary, 'request_count.current') ?? readPath(summary, 'in_flight_requests.avg');
    return { value: v, series: [] };
  },
  pods_ratio: ({ pods }) => {
    if (!Array.isArray(pods)) return { value: null, series: [], _full: { ready: null, total: null } };
    const ready = pods.filter((p) => {
      const ph = (p?.phase ?? p?.status?.phase ?? '').toLowerCase();
      return ph === 'running';
    }).length;
    return { value: ready, series: [], _full: { ready, total: pods.length } };
  },
  gpu_util_server: ({ serverSummary, serverTimeseries }) => {
    const v = readPath(serverSummary, 'gpu_util.avg') ?? readPath(serverSummary, 'gpu_util_avg');
    const series = pluck(serverTimeseries?.['gpu_util'], 'avg');
    return { value: v, series };
  },
  kv_cache_server: ({ serverSummary, serverTimeseries }) => {
    const v = readPath(serverSummary, 'kv_cache.avg') ?? readPath(serverSummary, 'kv_cache_avg');
    const series = pluck(serverTimeseries?.['kv_cache'], 'avg');
    return { value: v, series };
  },
  records_progress: ({ phases }) => {
    const list = Array.isArray(phases) ? phases : [];
    let processed = 0;
    let total = 0;
    for (const p of list) {
      processed += p?.recordsSuccess ?? 0;
      total += p?.requestsTotal ?? 0;
    }
    return { value: processed > 0 ? processed : null, series: [], _full: { processed, total } };
  },
};

function formatValue(value, fmt, summaryEntry) {
  if (value == null || (typeof value === 'number' && !isFinite(value))) return '—';
  switch (fmt) {
    case 'thousands':
      return Math.abs(value) >= 1000 ? `${(value / 1000).toFixed(1)}k` : fmtNumber(value, 1);
    case 'number0': return fmtInt(Math.round(value));
    case 'number2': return fmtNumber(value, 2);
    case 'percent2': return fmtPercent(value, 2);
    case 'pod_ratio': {
      const total = summaryEntry?.total ?? readPath(summaryEntry?._full_summary, 'pods_total.avg');
      return total != null ? `${fmtInt(value)}/${fmtInt(total)}` : fmtInt(value);
    }
    case 'records_ratio': {
      const proc = summaryEntry?.processed ?? value;
      const total = summaryEntry?.total ?? readPath(summaryEntry?._full_summary, 'records_total.avg');
      return total != null ? `${fmtInt(proc)}/${fmtInt(total)}` : fmtInt(proc);
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

function computePodHealthTone(ready, total) {
  if (total == null || ready == null || total === 0) return 'neutral';
  const ratio = ready / total;
  if (ratio >= 0.99) return 'good';
  if (ratio >= 0.9) return 'warn';
  return 'bad';
}

// Resolve a tile config to its concrete render-ready snapshot. Used both for
// the secondary KpiTile rail and for the hero KpiCard rail (the latter only
// reads the value/series/tone fields and re-renders via KpiCard).
function resolveTile(cfg, ctx) {
  const { summary: sum, timeseries: ts, slos: sloDict } = ctx;
  let value;
  let series;
  let fullCtx = null;
  if (cfg.resolverKey) {
    const r = RESOLVERS[cfg.resolverKey](ctx);
    value = r.value;
    series = r.series ?? [];
    fullCtx = r._full ?? null;
  } else {
    value = readPath(sum, cfg.summaryKey);
    series = cfg.seriesKey ? pluck(ts[cfg.seriesKey], cfg.seriesStat ?? 'avg') : [];
  }
  const { delta, direction } = computeDelta(series);
  const sloThreshold = cfg.sloKey ? sloDict[cfg.sloKey] ?? null : null;
  const tone = cfg.toneRule === 'pod_health'
    ? computePodHealthTone(fullCtx?.ready, fullCtx?.total)
    : computeTone(cfg.toneRule, value, sloThreshold, cfg.sloKey, sum);
  const summaryEntry = { _full_summary: sum, ...(fullCtx ?? {}) };
  const formatted = formatValue(value, cfg.fmt, summaryEntry);
  return { cfg, value, series, delta, direction, tone, formatted, sloThreshold };
}

// Pick up to 3 hero tiles from the resolved set. Walks HERO_CANDIDATES in
// priority order and keeps the first 3 whose tile resolved to a non-null
// value. Always returns at least one hero (the first non-null candidate, or
// — as a last resort — the first config in the rail) so the hero strip never
// renders empty.
//
// Returns ``{ hero: ResolvedTile[], secondary: ResolvedTile[] }`` where the
// secondary list preserves TILE_CONFIG order minus the promoted hero ids.
export function selectHero(resolved) {
  const byId = new Map(resolved.map((r) => [r.cfg.id, r]));
  const heroIds = [];
  for (const id of HERO_CANDIDATES) {
    if (heroIds.length >= 3) break;
    const r = byId.get(id);
    if (r && r.value != null) heroIds.push(id);
  }
  if (heroIds.length === 0 && resolved.length > 0) {
    // Degenerate run: nothing resolved. Still render the throughput slot so
    // the hero strip isn't empty; the tile will show '—'.
    heroIds.push(resolved[0].cfg.id);
  }
  const heroSet = new Set(heroIds);
  const hero = heroIds.map((id) => byId.get(id)).filter(Boolean);
  const secondary = resolved.filter((r) => !heroSet.has(r.cfg.id));
  return { hero, secondary };
}

// Hero rendering: KpiCard with size="hero", an icon, sparkline, progress
// (when an SLO threshold is in scope), and an optional trend badge.
function renderHero(resolvedTile, { stale, mode }) {
  const { cfg, value, series, formatted, tone, sloThreshold } = resolvedTile;
  const cardTone = railToneToCardTone(tone);
  const icon = HERO_ICONS[cfg.id] ?? 'speed';
  // Progress bar: only when an SLO is in scope. The bar tracks value/threshold
  // either way — for higher-is-better the bar fills toward the goal; for
  // lower-is-better it fills toward the ceiling (headroom-burnt indicator).
  let progress;
  if (sloThreshold != null && sloThreshold > 0 && typeof value === 'number' && isFinite(value)) {
    const ratio = (value / sloThreshold) * 100;
    if (isFinite(ratio)) progress = Math.min(100, Math.max(0, ratio));
  }
  // Trend direction-good: throughput / goodput up = good; latency up = bad.
  const trendGoodDirection = cfg.toneRule === 'higher_is_better';
  const trend = (cfg.toneRule === 'higher_is_better' || cfg.toneRule === 'lower_is_better')
    ? computeTrend(series, trendGoodDirection)
    : null;
  const sub = stale
    ? 'stale'
    : (mode === 'completed' ? 'final' : (mode === 'archived' ? 'archived' : 'live'));
  return html`
    <${KpiCard}
      key=${cfg.id}
      label=${cfg.label}
      value=${formatted}
      unit=${cfg.unit}
      icon=${icon}
      tone=${cardTone}
      size="hero"
      sparkline=${{ points: series ?? [] }}
      progress=${progress}
      trend=${trend}
      sub=${sub}
      title=${cfg.label} />
  `;
}

// Secondary rendering: existing dense KpiTile (unchanged shape).
function renderSecondary(resolvedTile, { stale, mode }) {
  const { cfg, formatted, series, delta, direction, tone } = resolvedTile;
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
}

export function KpiRail({ summary, slos, timeseries, mode = 'live', stale = false,
                         pods, phases, serverSummary, serverTimeseries }) {
  const sum = summary ?? {};
  const ts = timeseries ?? {};
  const sloDict = slos ?? {};
  const ctx = { summary: sum, timeseries: ts, slos: sloDict, pods, phases, serverSummary, serverTimeseries };

  const resolved = TILE_CONFIG.map((cfg) => resolveTile(cfg, ctx));
  const { hero, secondary } = selectHero(resolved);
  const renderOpts = { stale, mode };

  return html`
    <div data-testid="kpi-rail">
      <div class="kpi-rail__hero" data-testid="kpi-rail-hero">
        ${hero.map((r) => renderHero(r, renderOpts))}
      </div>
      <div class="kpi-rail__secondary kpi-rail-grid" data-testid="kpi-rail-secondary">
        ${secondary.map((r) => renderSecondary(r, renderOpts))}
      </div>
    </div>
  `;
}
