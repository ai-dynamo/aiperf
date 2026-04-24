// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Customer-facing realtime KPI grid.
 *
 * Metric selection + presentation are driven by what LLM serving customers
 * actually care about, as documented by:
 *   - NVIDIA NIM Benchmarking Metrics (TTFT, ITL, E2E latency, TPS, RPS)
 *   - AIPerf's own customer docs (Pareto curve: TPS/GPU vs TPS/user, Goodput)
 *   - BentoML LLM Inference Handbook (TTFT as UX headline, Goodput as SLO health)
 *   - vLLM production guide (p99 for tail SLOs, ITL for streaming smoothness)
 *
 * SLO policy: the dashboard only passes judgment against thresholds the USER
 * declared via ``cfg.slos`` (same dict AIPerf's goodput feature consumes —
 * e.g. ``--goodput "time_to_first_token:100 inter_token_latency:10"``).
 * There are no hard-coded "industry defaults" here: if the user hasn't told
 * us what good looks like for a given metric, we show the value without a
 * status chip. This keeps the chip honest — green means *you* declared this
 * SLO and the run met it, red means *you* declared it and the run missed it.
 *
 * Stat-selection rules:
 *   - Throughput metrics → ``current`` (or ``avg`` fallback). p99 of a rate
 *     is noise; the hero number is the sustained value.
 *   - Latency metrics   → ``p99``. This is the tail guarantee customers
 *     promise end-users; ``avg`` goes in the sub-line for context.
 *   - ITL               → ``avg`` is the headline (streaming smoothness);
 *     p99 sits in the sub-line for spike detection.
 */

import { html } from 'htm/preact';
import { realtimeMetrics, config, timeseries } from '../lib/state.js';
import { fmtNumber, fmtInt, fmtPercent } from '../lib/format.js';
import { pluck } from '../lib/timeseries.js';
import { Sparkline } from './sparkline.js';

/** Tile specs, ordered by research-grounded priority.
 *
 *  ``sloTag`` is the ``cfg.slos`` key to read when the user declared a
 *  threshold for this metric. ``sloCompare``:
 *    - ``'lt'`` → value must be <= threshold (latency metrics).
 *    - ``'gt'`` → value must be >= threshold (rare; for reserved-floor use).
 */
const TILES = [
  {
    tag: 'output_token_throughput',
    label: 'Output Tokens/s',
    primary: 'current',
    secondary: 'avg',
    secondaryLabel: 'avg',
    sloTag: null,          // throughput isn't an SLO metric by AIPerf convention
  },
  {
    tag: 'request_throughput',
    label: 'Requests/s',
    primary: 'current',
    secondary: 'avg',
    secondaryLabel: 'avg',
    sloTag: null,
  },
  {
    tag: 'time_to_first_token',
    label: 'TTFT',
    primary: 'p99',
    secondary: 'avg',
    secondaryLabel: 'avg',
    sloTag: 'time_to_first_token',
    sloCompare: 'lt',
  },
  {
    tag: 'request_latency',
    label: 'Request Latency',
    primary: 'p99',
    secondary: 'avg',
    secondaryLabel: 'avg',
    sloTag: 'request_latency',
    sloCompare: 'lt',
  },
  {
    tag: 'inter_token_latency',
    label: 'ITL',
    primary: 'avg',
    secondary: 'p99',
    secondaryLabel: 'p99',
    sloTag: 'inter_token_latency',
    sloCompare: 'lt',
  },
];

function byTag(metrics) {
  const m = {};
  for (const r of metrics) if (r?.tag) m[r.tag] = r;
  return m;
}

/** Choose the best stat to report. Falls back to avg when `current` is null. */
function pickStat(metric, key) {
  if (!metric) return null;
  const v = metric[key];
  if (v != null) return v;
  if (key === 'current' && metric.avg != null) return metric.avg;
  return null;
}

/** Format one numeric stat with unit; returns {body, unit} or {body: '---'}. */
function formatStat(value, unit) {
  if (value == null) return { body: '---', unit: '' };
  if (typeof value !== 'number') return { body: String(value), unit: unit ?? '' };
  const body = Math.abs(value) >= 1000 ? fmtInt(Math.round(value)) : fmtNumber(value, 2);
  return { body, unit: unit ?? '' };
}

/** Binary SLO check against the user's declared ``cfg.slos`` threshold.
 *
 *  Returns ``null`` if the user hasn't declared an SLO for this metric —
 *  in which case no chip is rendered. No fabricated defaults.
 */
function sloStatus(value, threshold, compare = 'lt') {
  if (value == null || !isFinite(value)) return null;
  if (threshold == null) return null;
  const ok = compare === 'lt' ? value <= threshold : value >= threshold;
  return {
    kind: ok ? 'good' : 'bad',
    label: (compare === 'lt' ? '≤ ' : '≥ ') + threshold,
  };
}

/** Pull the user's configured SLO for `metricTag` from ``cfg.slos``, if any. */
function userSlo(cfg, metricTag) {
  const slos = cfg?.slos;
  if (!slos || typeof slos !== 'object') return null;
  return slos[metricTag] ?? null;
}

/** Render a single tile. */
function KpiTile({ spec, metric, cfg, series }) {
  const primaryVal = pickStat(metric, spec.primary);
  const secondaryVal = pickStat(metric, spec.secondary);
  const unit = metric?.unit ?? '';

  const slo = spec.sloTag
    ? sloStatus(primaryVal, userSlo(cfg, spec.sloTag), spec.sloCompare)
    : null;

  const primary = formatStat(primaryVal, unit);
  const secondaryDisplay = secondaryVal != null && typeof secondaryVal === 'number'
    ? (Math.abs(secondaryVal) >= 1000
        ? fmtInt(Math.round(secondaryVal))
        : fmtNumber(secondaryVal, 2))
    : '---';

  // Sparkline: prefer the same stat the tile headlines so the line
  // matches the big number. Throughput metrics from AIPerf only populate
  // ``avg`` (no ``current`` field), so fall back through the same chain
  // pickStat uses for the headline number.
  const sparkStats = spec.primary === 'current' ? ['current', 'avg'] : [spec.primary];
  let sparkPoints = [];
  for (const stat of sparkStats) {
    sparkPoints = pluck(series ?? [], stat);
    if (sparkPoints.length >= 2) break;
  }
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
          <span class="kpi-tile-primary-stat">${spec.primary}</span>
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

/** Composite goodput / success-rate tile.
 *
 *  Goodput is AIPerf's canonical "how much of my traffic met every SLO I
 *  declared". We headline the *violation count* (not the pass rate) so a
 *  glance sees the size of the problem — chasing "99.6% passes" hides
 *  that 56 real requests missed the user's SLA. Chip is green iff every
 *  completed request passed, warn otherwise.
 *
 *  When no SLOs are configured we fall back to Success Rate (errors /
 *  total), which is an objective reliability reading — green iff 0 errors.
 */
function ReliabilityTile({ byT, cfg }) {
  const hasSlo = cfg?.slos && Object.keys(cfg.slos).length > 0;

  if (hasSlo) {
    const gp = byT['goodput'] ?? byT['good_request_count'] ?? null;
    const primary = pickStat(gp, 'current') ?? pickStat(gp, 'avg');
    const reqCount = byT['request_count'];
    const goodCount = byT['good_request_count'];
    // Counters (request_count, good_request_count) and derived metrics (goodput)
    // populate ``avg`` with the single scalar value — ``current`` is undefined.
    // Fall back so the failed-count and pass-rate stay live during the run.
    const goodVal = goodCount?.current ?? goodCount?.avg;
    const reqVal = reqCount?.current ?? reqCount?.avg;
    const failedCount = (goodVal != null && reqVal != null)
      ? Math.max(0, Math.round(reqVal - goodVal))
      : null;
    const pct = (goodVal != null && reqVal != null && reqVal > 0)
      ? (goodVal / reqVal) * 100
      : null;
    const kind = pct == null ? null : (pct >= 100 ? 'good' : 'warn');
    const primaryDisplay = formatStat(primary, gp?.unit ?? 'req/s');
    const sloList = Object.keys(cfg.slos).join(', ');
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
        <div class="kpi-tile-sub">
          ${pct != null
            ? html`<span>${fmtPercent(pct, 1)}</span>
                   <span class="kpi-tile-sub-val">of ${fmtInt(reqVal) ?? '---'}</span>`
            : html`<span>of ${fmtInt(reqVal) ?? '---'} completed</span>`}
        </div>
      </div>
    `;
  }

  // No SLOs configured → Success Rate derived from error_request_count.
  const errorRate = byT['error_request_rate'] ?? null;
  const errorCount = byT['error_request_count'] ?? null;
  const reqCount = byT['request_count'] ?? null;

  const rate = (errorRate?.current ?? errorRate?.avg)
    ?? (reqCount?.current != null && errorCount?.current != null && reqCount.current > 0
        ? (errorCount.current / reqCount.current) * 100
        : null);
  if (rate == null) return null;
  const success = Math.max(0, 100 - rate);
  const kind = (errorCount?.current ?? 0) === 0 ? 'good' : 'warn';

  return html`
    <div class=${'kpi-tile kpi-tile--slo-' + kind} key="success-rate">
      <div class="kpi-tile-head">
        <div class="kpi-tile-label">
          <span>Success Rate</span>
          <span class="kpi-tile-primary-stat">reliability</span>
        </div>
        <span class=${'kpi-chip kpi-chip--' + kind}>
          ${kind === 'good' ? '✓' : '✗'}
          <span class="kpi-chip-thresh">${kind === 'good' ? '0 errors' : fmtInt(errorCount?.current ?? 0) + ' errors'}</span>
        </span>
      </div>
      <div class="kpi-big">
        <span class="kpi-big-val">${fmtPercent(success, 2)}</span>
      </div>
      <div class="kpi-tile-sub">
        <span>errors</span>
        <span class="kpi-tile-sub-val">${fmtInt(errorCount?.current ?? 0)}</span>
      </div>
    </div>
  `;
}

export function RealtimeMetricsCard() {
  const metrics = realtimeMetrics.value;
  const cfg = config.value;
  const ts = timeseries.value;
  const byT = byTag(metrics);

  // Hide the whole card until something actionable arrives.
  const hasHero = TILES.some(t => byT[t.tag] != null);
  const hasReliability =
    byT['goodput'] != null
    || byT['good_request_count'] != null
    || (byT['request_count'] != null && byT['error_request_count'] != null);
  if (!hasHero && !hasReliability) return null;

  return html`
    <div class="card">
      <div class="card-title">Realtime Metrics</div>
      <div class="kpi-grid">
        ${TILES.map((spec) => html`
          <${KpiTile} spec=${spec} metric=${byT[spec.tag]} cfg=${cfg}
                      series=${ts[spec.tag] ?? []} key=${spec.tag} />
        `)}
        <${ReliabilityTile} byT=${byT} cfg=${cfg} />
      </div>
    </div>
  `;
}
