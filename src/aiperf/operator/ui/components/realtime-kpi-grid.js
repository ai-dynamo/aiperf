// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Realtime KPI grid for the per-job detail page.
 *
 * Ported from ``api/static-v2/components/realtime-metrics.js`` and adapted
 * to the operator's REST snapshot shape:
 *
 *   - Input ``summary`` is an object keyed by metric tag:
 *       ``{ [tag]: { avg, p1, p10, p50, p90, p95, p99, std, min, max, ... } }``
 *     (no streaming ``current`` — the operator polls).
 *   - Input ``slos`` is the user's declared ``cfg.slos`` dict from the CR
 *     (``spec.benchmark.slos`` / ``spec.slos``), keyed by metric tag.
 *   - Input ``timeseries`` is optional; when present, a per-tag array of
 *     ``{t, values}`` samples drives sparklines. Empty by default until a
 *     WS feed is wired up — sparkline placeholders render unobtrusively.
 *
 * SLO policy: only chip metrics where the user declared a threshold.
 * Direction: latency-style ``<=`` by default; the throughput-side tags in
 * ``LARGER_IS_BETTER_SLO_TAGS`` flip to ``>=`` (mirrors ``MetricFlags``
 * server-side and matches the existing SLACompliance card on this page).
 */

import { html } from 'htm/preact';
import { fmtNumber, fmtInt, fmtPercent } from '../lib/format.js';
import { Sparkline } from './sparkline.js';

/** Tile specs, ordered by research-grounded priority.
 *
 *  Throughput primary is ``current`` (the controller's per-frame instantaneous
 *  rate, populated by the WS feed) with ``avg`` as a fallback in ``pickStat``
 *  for the REST-only state. Latency tiles headline ``p99`` (tail SLO).
 *  ITL stays on ``avg`` for streaming smoothness with ``p99`` in the sub-line.
 */
const TILES = [
  {
    tag: 'output_token_throughput',
    label: 'Output Tokens/s',
    primary: 'current',
    secondary: 'avg',
    secondaryLabel: 'avg',
    sloTag: 'output_token_throughput',
    fallbackTags: ['e2e_output_token_throughput'],
  },
  {
    tag: 'request_throughput',
    label: 'Requests/s',
    primary: 'current',
    secondary: 'avg',
    secondaryLabel: 'avg',
    sloTag: 'request_throughput',
  },
  {
    tag: 'time_to_first_token',
    label: 'TTFT',
    primary: 'p99',
    secondary: 'avg',
    secondaryLabel: 'avg',
    sloTag: 'time_to_first_token',
  },
  {
    tag: 'request_latency',
    label: 'Request Latency',
    primary: 'p99',
    secondary: 'avg',
    secondaryLabel: 'avg',
    sloTag: 'request_latency',
  },
  {
    tag: 'inter_token_latency',
    label: 'ITL',
    primary: 'avg',
    secondary: 'p99',
    secondaryLabel: 'p99',
    sloTag: 'inter_token_latency',
  },
];

// Tags whose SLO comparison is ``>=`` (larger is better). Mirrors
// ``MetricFlags.LARGER_IS_BETTER`` server-side; same set as the
// SLACompliance card on this page.
const LARGER_IS_BETTER_SLO_TAGS = new Set([
  'output_token_throughput',
  'output_token_throughput_per_user',
  'request_throughput',
  'total_token_throughput',
  'e2e_output_token_throughput',
  'prefill_throughput_per_user',
]);

const UNIT_BY_TAG = {
  request_latency: 'ms',
  time_to_first_token: 'ms',
  time_to_second_token: 'ms',
  inter_token_latency: 'ms',
  output_token_throughput: 'tok/s',
  output_token_throughput_per_user: 'tok/s',
  request_throughput: 'req/s',
  total_token_throughput: 'tok/s',
  e2e_output_token_throughput: 'tok/s',
  prefill_throughput_per_user: 'tok/s',
};

function pickMetric(summary, spec) {
  const primary = summary?.[spec.tag];
  if (primary != null) return primary;
  for (const tag of spec.fallbackTags ?? []) {
    const fallback = summary?.[tag];
    if (fallback != null) return fallback;
  }
  return null;
}

function pickStat(metric, key) {
  if (!metric) return { value: null, stat: null };
  const v = metric[key];
  if (v != null) return { value: v, stat: key };
  if (metric.avg != null) return { value: metric.avg, stat: 'avg' };
  return { value: null, stat: null };
}

function formatStat(value, unit) {
  if (value == null) return { body: '---', unit: '' };
  if (typeof value !== 'number') return { body: String(value), unit: unit ?? '' };
  if (!isFinite(value)) return { body: '---', unit: '' };
  const body = Math.abs(value) >= 1000 ? fmtInt(Math.round(value)) : fmtNumber(value, 2);
  return { body, unit: unit ?? '' };
}

/** Binary SLO check against the user's declared threshold. Returns ``null``
 *  if the user hasn't declared an SLO for this tag — no fabricated defaults.
 */
function sloStatus(value, threshold, tag) {
  if (value == null || !isFinite(value)) return null;
  if (threshold == null) return null;
  const largerBetter = LARGER_IS_BETTER_SLO_TAGS.has(tag);
  const ok = largerBetter ? value >= threshold : value <= threshold;
  return {
    kind: ok ? 'good' : 'bad',
    label: (largerBetter ? '≥ ' : '≤ ') + threshold,
  };
}

function pluck(series, statKey) {
  if (!series || series.length === 0) return [];
  const out = [];
  for (const s of series) {
    const v = s?.values?.[statKey];
    if (typeof v === 'number' && isFinite(v)) out.push({ t: s.t, v });
  }
  return out;
}

function KpiTile({ spec, metric, slos, series }) {
  const { value: primaryVal, stat: primaryStat } = pickStat(metric, spec.primary);
  const { value: secondaryVal } = pickStat(metric, spec.secondary);
  const unit = metric?.unit ?? UNIT_BY_TAG[spec.tag] ?? '';

  const sloThreshold = (slos && spec.sloTag) ? slos[spec.sloTag] ?? null : null;
  const slo = sloStatus(primaryVal, sloThreshold, spec.sloTag);

  const primary = formatStat(primaryVal, unit);
  const secondaryDisplay = secondaryVal != null && typeof secondaryVal === 'number' && isFinite(secondaryVal)
    ? (Math.abs(secondaryVal) >= 1000
        ? fmtInt(Math.round(secondaryVal))
        : fmtNumber(secondaryVal, 2))
    : '---';

  // Sparkline pulls from the *actual* stat the headline used so the line
  // matches the big number when REST data falls back from `current` → `avg`.
  const sparkPoints = pluck(series, primaryStat ?? spec.primary);
  // Surface the stat actually displayed (may differ from spec.primary when
  // current fell back to avg on REST-only views) so users don't read "current"
  // next to a value that's really an average.
  const primaryLabel = primaryStat ?? spec.primary;
  const sparkStroke = slo?.kind === 'bad' ? 'var(--red)'
    : slo?.kind === 'good' ? 'var(--accent)'
    : 'var(--sub)';
  const sparkFill = slo?.kind === 'bad' ? 'rgba(239,83,80,0.15)'
    : slo?.kind === 'good' ? 'var(--accent-dim)'
    : 'rgba(167,167,167,0.10)';

  return html`
    <div class=${'kpi-tile' + (slo ? ' kpi-tile--slo-' + slo.kind : '')} key=${spec.tag}>
      <div class="kpi-tile-head">
        <div class="kpi-tile-label">
          <span>${spec.label}</span>
          <span class="kpi-tile-primary-stat">${primaryLabel}</span>
        </div>
        ${slo && html`
          <span class=${'kpi-chip kpi-chip--' + slo.kind}
                title="Your SLO from cfg.slos">
            ${slo.kind === 'good' ? '✓' : '✗'}
            <span class="kpi-chip-thresh">${slo.label}</span>
          </span>
        `}
      </div>
      <div class="kpi-big">
        <span class="kpi-big-val">${primary.body}</span>
        ${primary.unit && html`<span class="kpi-big-unit">${primary.unit}</span>`}
      </div>
      <${Sparkline} points=${sparkPoints} stroke=${sparkStroke} fill=${sparkFill}
                    width=${140} height=${26} />
      <div class="kpi-tile-sub">
        <span>${spec.secondaryLabel}</span>
        <span class="kpi-tile-sub-val">${secondaryDisplay}</span>
      </div>
    </div>
  `;
}

/** Composite goodput / success-rate tile. Goodput when SLOs are declared,
 *  Success Rate (1 - error_rate) otherwise. */
function ReliabilityTile({ summary, slos, timeseries }) {
  const ts = timeseries ?? {};
  const hasSlo = slos && Object.keys(slos).length > 0;

  if (hasSlo) {
    const gp = summary['goodput'] ?? summary['good_request_count'] ?? null;
    const { value: primary } = pickStat(gp, 'avg');
    const goodCount = summary['good_request_count'];
    const reqCount = summary['request_count'];
    const goodVal = goodCount?.avg;
    const reqVal = reqCount?.avg;
    const hasFiniteCounts = typeof goodVal === 'number' && isFinite(goodVal)
      && typeof reqVal === 'number' && isFinite(reqVal);
    const failedCount = hasFiniteCounts
      ? Math.max(0, Math.round(reqVal - goodVal))
      : null;
    const pct = (hasFiniteCounts && reqVal > 0)
      ? (goodVal / reqVal) * 100
      : null;
    const kind = pct == null ? null : (pct >= 100 ? 'good' : 'warn');
    const primaryDisplay = formatStat(primary, gp?.unit ?? 'req/s');
    const sloList = Object.keys(slos).join(', ');
    // Sparkline series: prefer goodput timeseries directly; fall back to
    // good_request_count (the underlying counter — useful while goodput is
    // still NoMetricValue early in the run before benchmark_duration is set).
    const goodputSeries = pluck(ts['goodput'], 'avg');
    const sparkPoints = goodputSeries.length > 0
      ? goodputSeries
      : pluck(ts['good_request_count'], 'avg');
    // kind is null | 'good' | 'warn'; warn means at least one request missed
    // an SLO. Match KpiTile's failure coloring (red) and the success-rate
    // branch below — anything other than 'good' that has data is a miss.
    const sparkStroke = kind === 'good' ? 'var(--accent)'
      : kind === 'warn' ? 'var(--red)'
      : 'var(--sub)';
    const sparkFill = kind === 'good' ? 'var(--accent-dim)'
      : kind === 'warn' ? 'rgba(239,83,80,0.15)'
      : 'rgba(167,167,167,0.10)';
    return html`
      <div class=${'kpi-tile' + (kind ? ' kpi-tile--slo-' + kind : '')} key="goodput">
        <div class="kpi-tile-head">
          <div class="kpi-tile-label">
            <span>Goodput</span>
            <span class="kpi-tile-primary-stat" title=${'Requests meeting all of: ' + sloList}>SLO pass</span>
          </div>
          ${kind && html`
            <span class=${'kpi-chip kpi-chip--' + kind}
                  title=${'Requests that missed at least one SLO (' + sloList + ')'}>
              ${failedCount != null
                ? (kind === 'good'
                   ? html`✓ <span class="kpi-chip-thresh">0 failed</span>`
                   : html`✗ <span class="kpi-chip-thresh">${fmtInt(failedCount)} failed</span>`)
                : (kind === 'good' ? '✓ 100%' : fmtPercent(pct, 1))}
            </span>
          `}
        </div>
        <div class="kpi-big">
          <span class="kpi-big-val">${primaryDisplay.body}</span>
          ${primaryDisplay.unit && html`<span class="kpi-big-unit">${primaryDisplay.unit}</span>`}
        </div>
        <${Sparkline} points=${sparkPoints} stroke=${sparkStroke} fill=${sparkFill}
                      width=${140} height=${26} />
        <div class="kpi-tile-sub">
          ${pct != null
            ? html`<span>${fmtPercent(pct, 1)}</span>
                   <span class="kpi-tile-sub-val">of ${fmtInt(reqVal)}</span>`
            : html`<span>of ${fmtInt(reqVal)} completed</span>`}
        </div>
      </div>
    `;
  }

  // No SLOs declared → Success Rate from error_rate / error_request_count.
  const errorRate = summary['error_rate'];
  const errorCount = summary['error_request_count'];
  const reqCount = summary['request_count'];
  const rate = (typeof errorRate === 'number' ? errorRate * 100 : null)
    ?? (errorRate?.avg != null ? errorRate.avg * 100 : null)
    ?? (reqCount?.avg != null && errorCount?.avg != null && reqCount.avg > 0
        ? (errorCount.avg / reqCount.avg) * 100
        : null);
  if (rate == null || !isFinite(rate)) return null;
  const success = Math.max(0, 100 - rate);
  const rawErrVal = (typeof errorCount === 'number') ? errorCount : (errorCount?.avg ?? 0);
  const errVal = (typeof rawErrVal === 'number' && isFinite(rawErrVal)) ? rawErrVal : 0;
  const kind = errVal === 0 ? 'good' : 'warn';
  // Sparkline: error_rate trend (lower is better) — fall back to
  // error_request_count if error_rate isn't streamed.
  const errorRateSeries = pluck(ts['error_rate'], 'avg');
  const sparkPoints = errorRateSeries.length > 0
    ? errorRateSeries
    : pluck(ts['error_request_count'], 'avg');
  const sparkStroke = kind === 'good' ? 'var(--accent)' : 'var(--red)';
  const sparkFill = kind === 'good' ? 'var(--accent-dim)' : 'rgba(239,83,80,0.15)';

  return html`
    <div class=${'kpi-tile kpi-tile--slo-' + kind} key="success-rate">
      <div class="kpi-tile-head">
        <div class="kpi-tile-label">
          <span>Success Rate</span>
          <span class="kpi-tile-primary-stat">reliability</span>
        </div>
        <span class=${'kpi-chip kpi-chip--' + kind}>
          ${kind === 'good' ? '✓' : '✗'}
          <span class="kpi-chip-thresh">${kind === 'good' ? '0 errors' : fmtInt(errVal) + ' errors'}</span>
        </span>
      </div>
      <div class="kpi-big">
        <span class="kpi-big-val">${fmtPercent(success, 2)}</span>
      </div>
      <${Sparkline} points=${sparkPoints} stroke=${sparkStroke} fill=${sparkFill}
                    width=${140} height=${26} />
      <div class="kpi-tile-sub">
        <span>errors</span>
        <span class="kpi-tile-sub-val">${fmtInt(errVal)}</span>
      </div>
    </div>
  `;
}

/**
 * @param {object} props
 * @param {Object<string, object>} props.summary - tag-keyed metric snapshot
 * @param {Object<string, number>} [props.slos] - user-declared SLO thresholds
 * @param {Object<string, Array>} [props.timeseries] - optional per-tag series
 *   for sparklines. Empty by default; populated once a WS feed lands.
 */
export function RealtimeKpiGrid({ summary, slos, timeseries }) {
  const ts = timeseries ?? {};
  const sloDict = slos ?? null;

  const hasHero = TILES.some(t => pickMetric(summary, t) != null);
  const hasReliability =
    summary?.['goodput'] != null
    || summary?.['good_request_count'] != null
    || summary?.['error_rate'] != null
    || (summary?.['request_count'] != null && summary?.['error_request_count'] != null);
  if (!hasHero && !hasReliability) return null;

  return html`
    <div class="kpi-grid">
      ${TILES.map((spec) => html`
        <${KpiTile} spec=${spec} metric=${pickMetric(summary, spec)} slos=${sloDict}
                    series=${ts[spec.tag] ?? []} key=${spec.tag} />
      `)}
      <${ReliabilityTile} summary=${summary ?? {}} slos=${sloDict} timeseries=${ts} />
    </div>
  `;
}
