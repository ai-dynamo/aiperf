// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Hero strip — focal point for Job Detail (and the Dashboard live run).
 *
 * Ported from static-v2/components/hero-strip.js, but accepts props so it
 * can be rendered outside the v2 signal-driven single-run dashboard.
 *
 * Props:
 *   - info    : AIPerfJobInfo flat dict (``phase``, ``startTime``, ``completionTime``).
 *   - status  : raw CR status (``summary``, ``phases``, ``conditions``, ``error``).
 *   - config  : /api/v1/config response (``{source, spec}``); may be ``null``.
 *
 * Answers three questions at a glance:
 *   1. Is my run healthy right now?  (SLO-compliance traffic light)
 *   2. How much longer?               (elapsed + ETA)
 *   3. What's it doing?               (active-phase big progress bar)
 */

import { html } from 'htm/preact';
import { fmtInt, fmtDuration, fmtPercent } from '../lib/format.js';

/** Pull the SLO dict out of the config response, checking both shapes.
 *
 *  The config endpoint wraps CR-spec data (``{source, spec}``); summary
 *  fallback stuffs the original ``input_config`` under ``spec.benchmark``,
 *  but index-sourced specs may already be the benchmark dict itself.
 */
function extractSlos(config) {
  return config?.spec?.benchmark?.slos ?? config?.spec?.slos ?? null;
}

/** Convert the flat operator ``status.summary`` into the v2-style metric list
 *  the classifyHealth / display code expects.
 */
function summaryToMetrics(summary) {
  const metrics = [];
  if (!summary) return metrics;
  if (summary.throughput_rps != null) {
    metrics.push({
      tag: 'request_throughput',
      current: summary.throughput_rps,
      avg: summary.throughput_rps,
    });
  }
  if (summary.ttft_avg_ms != null || summary.ttft_p99_ms != null) {
    metrics.push({
      tag: 'time_to_first_token',
      avg: summary.ttft_avg_ms,
      p99: summary.ttft_p99_ms,
    });
  }
  if (summary.latency_avg_ms != null || summary.latency_p99_ms != null) {
    metrics.push({
      tag: 'request_latency',
      avg: summary.latency_avg_ms,
      p99: summary.latency_p99_ms,
    });
  }
  if (summary.itl_avg_ms != null || summary.itl_p99_ms != null) {
    metrics.push({
      tag: 'inter_token_latency',
      avg: summary.itl_avg_ms,
      p99: summary.itl_p99_ms,
    });
  }
  return metrics;
}

/** Classify overall run health (SLO p99 miss -> errors -> idle -> ok).
 *
 *  Mirrors the v2 ranking: SLO violation is the strongest signal (error),
 *  then request errors warn, then fall back to ok / idle.
 */
function classifyHealth(slos, metrics, recs) {
  const byT = {};
  for (const m of metrics) if (m?.tag) byT[m.tag] = m;

  if (
    metrics.length === 0 &&
    (recs.successRecords ?? 0) === 0 &&
    (recs.errorRecords ?? 0) === 0 &&
    !recs.complete
  ) {
    return { status: 'idle', category: null, reasons: [] };
  }

  let status = 'ok';
  let category = null;
  const reasons = [];

  if (slos && typeof slos === 'object') {
    for (const [key, thr] of Object.entries(slos)) {
      const metric = byT[key];
      if (!metric) continue;
      const probe = metric.p99 ?? metric.current ?? metric.avg;
      if (probe != null && probe > thr) {
        status = 'error';
        category = 'slo';
        reasons.push(`${key} p99 ${probe.toFixed(0)} > ${thr}`);
      }
    }
  }

  const errorCount = recs.errorRecords ?? 0;
  if (errorCount > 0) {
    if (status !== 'error') {
      status = 'warn';
      category = category ?? 'errors';
    }
    reasons.push(`${fmtInt(errorCount)} request errors`);
  }

  return { status, category, reasons };
}

/** Pick the active phase to hero.
 *
 *  Prefers phases explicitly marked ``active`` (and not ``complete``).
 *  When no phase is marked active — common for live CRs whose status
 *  snapshot hasn't flipped the flag yet — fall back to the incomplete
 *  phase with the most completed requests, and finally to the first
 *  incomplete phase (pre-start state).
 */
function pickActivePhase(phaseMap) {
  const arr = Object.entries(phaseMap ?? {}).map(([name, p]) => ({ ...p, name }));
  const done = p => p.complete === true;
  const completedOf = p =>
    p.final_requests_completed ??
    p.requestsCompleted ??
    p.requests_completed ??
    p.completed ??
    0;

  const explicit = arr.filter(p => p.active && !done(p));
  if (explicit.length > 0) {
    explicit.sort((a, b) => completedOf(b) - completedOf(a));
    return explicit[0];
  }

  const inProgress = arr.filter(p => !done(p) && completedOf(p) > 0);
  if (inProgress.length > 0) {
    inProgress.sort((a, b) => completedOf(b) - completedOf(a));
    return inProgress[0];
  }

  const pending = arr.filter(p => !done(p));
  return pending[0] ?? null;
}

/** Live ETA (seconds) from current completion rate. Null when phase lacks
 *  the backend-supplied ``start_ns``.
 */
function estimateEtaSec(phase) {
  if (!phase) return null;
  const total =
    phase.total_expected_requests ??
    phase.expected_requests ??
    phase.requestsTotal ??
    phase.requests_total ??
    null;
  const completed =
    phase.final_requests_completed ??
    phase.requestsCompleted ??
    phase.requests_completed ??
    phase.completed ??
    0;
  const startNs = phase.start_ns;
  if (!total || !startNs || completed <= 0) return null;
  const elapsedSec = (Date.now() - Number(startNs) / 1e6) / 1000;
  if (elapsedSec <= 0) return null;
  const rate = completed / elapsedSec;
  if (rate <= 0) return null;
  return Math.max(0, total - completed) / rate;
}

/** Overall run elapsed time in seconds. Uses completionTime for finished
 *  jobs so archived runs still show a sensible duration.
 */
function elapsedSec(info) {
  const start = info?.startTime;
  if (!start) return null;
  const startMs = new Date(start).getTime();
  if (!isFinite(startMs)) return null;
  const end = info?.completionTime
    ? new Date(info.completionTime).getTime()
    : Date.now();
  if (!isFinite(end)) return Math.max(0, (Date.now() - startMs) / 1000);
  return Math.max(0, (end - startMs) / 1000);
}

function activePct(phase) {
  if (!phase) return null;
  const total =
    phase.total_expected_requests ??
    phase.expected_requests ??
    phase.requestsTotal ??
    phase.requests_total ??
    null;
  const completed =
    phase.final_requests_completed ??
    phase.requestsCompleted ??
    phase.requests_completed ??
    phase.completed ??
    0;
  if (total && total > 0) return Math.min(100, (completed / total) * 100);
  if (phase.requestsProgressPercent != null) {
    return Math.min(100, Math.max(0, Number(phase.requestsProgressPercent)));
  }
  return null;
}

export function HeroStrip({ info, status, config, onClick }) {
  const summary = status?.liveSummary ?? status?.summary ?? null;
  const phaseMap = status?.phases ?? {};
  const metrics = summaryToMetrics(summary);
  const phase = info?.phase ?? '';
  const terminalPhase =
    phase === 'Succeeded' ||
    phase === 'Failed' ||
    phase === 'Cancelled' ||
    phase === 'Archived' ||
    phase === 'Completed' ||
    phase === 'Error';
  const livePhase = phase === 'Running' || phase === 'Pending';
  const recs = {
    successRecords: summary?.total_requests ?? 0,
    errorRecords: 0,
    complete: terminalPhase,
  };
  const slosRaw = extractSlos(config);
  const slosDeclared = !!(slosRaw && typeof slosRaw === 'object' && Object.keys(slosRaw).length > 0);
  const health = classifyHealth(slosRaw, metrics, recs);

  const active = pickActivePhase(phaseMap);
  const pct = activePct(active);
  const eta = estimateEtaSec(active);
  const elapsed = elapsedSec(info);

  // Derive the headline + subtitle + visual variant from the job's lifecycle
  // position. ``visualStatus`` drives the CSS accent (green/amber/red/grey);
  // it can differ from ``health.status`` when we want to show neutral copy
  // for no-SLO runs even though classifyHealth returned ``ok``.
  let healthLabel;
  let healthSub;
  let visualStatus = health.status;
  let headlineIcon = null;

  if (terminalPhase) {
    if (phase === 'Failed' || phase === 'Error') {
      healthLabel = 'Failed';
      healthSub = info?.error || health.reasons[0] || 'run reported failure';
      visualStatus = 'error';
      headlineIcon = 'ph-x-circle';
    } else if (phase === 'Cancelled') {
      healthLabel = 'Cancelled';
      healthSub = 'run stopped before completion';
      visualStatus = 'neutral';
      headlineIcon = 'ph-x-circle';
    } else if (!slosDeclared) {
      healthLabel = 'Completed';
      healthSub = 'no SLOs declared — no pass/fail judgment';
      visualStatus = 'neutral';
      headlineIcon = 'ph-check-circle';
    } else if (health.status === 'ok') {
      healthLabel = 'Passed SLOs';
      healthSub = 'all declared SLOs met';
      visualStatus = 'ok';
      headlineIcon = 'ph-check-circle';
    } else {
      healthLabel = 'Missed SLOs';
      healthSub = health.reasons.slice(0, 2).join(' · ') || 'one or more SLOs violated';
      visualStatus = health.status === 'warn' ? 'warn' : 'error';
      headlineIcon = 'ph-x-circle';
    }
  } else if (livePhase) {
    if (!slosDeclared) {
      healthLabel = 'Running';
      healthSub = 'no SLOs declared — no live judgment';
      visualStatus = 'neutral';
      headlineIcon = 'ph-clock';
    } else if (health.status === 'idle') {
      healthLabel = 'Waiting for data';
      healthSub = health.reasons.slice(0, 2).join(' · ') || 'no metrics reported yet';
      headlineIcon = 'ph-clock';
    } else if (health.status === 'ok') {
      healthLabel = 'On target';
      healthSub = 'all declared SLOs passing';
      headlineIcon = 'ph-check-circle';
    } else if (health.category === 'slo') {
      healthLabel = 'SLO violated';
      healthSub = health.reasons.slice(0, 2).join(' · ');
      headlineIcon = 'ph-warning';
    } else if (health.category === 'goodput') {
      healthLabel = 'SLO slipping';
      healthSub = health.reasons.slice(0, 2).join(' · ');
      headlineIcon = 'ph-warning';
    } else if (health.category === 'errors') {
      healthLabel = 'Errors reported';
      healthSub = health.reasons.slice(0, 2).join(' · ');
      headlineIcon = 'ph-warning';
    } else {
      healthLabel = 'Attention needed';
      healthSub = health.reasons.slice(0, 2).join(' · ');
      headlineIcon = 'ph-warning';
    }
  } else {
    healthLabel = phase || 'Unknown';
    healthSub = '';
    visualStatus = 'neutral';
  }

  const completedCount =
    active?.final_requests_completed ??
    active?.requestsCompleted ??
    active?.requests_completed ??
    active?.completed ??
    0;
  const totalCount =
    active?.total_expected_requests ??
    active?.expected_requests ??
    active?.requestsTotal ??
    active?.requests_total ??
    null;

  const clickable = typeof onClick === 'function';
  const style = clickable ? 'cursor: pointer' : '';

  return html`
    <div
      class=${'hero hero--' + visualStatus}
      style=${style}
      onclick=${clickable ? onClick : null}
      data-testid="hero-strip"
    >
      <div class="hero-health">
        <div class=${'hero-health-dot hero-health-dot--' + visualStatus + (visualStatus === 'ok' && livePhase ? ' live' : '')}></div>
        <div class="hero-health-text">
          <div class="hero-health-label">
            ${headlineIcon && html`<i class=${'ph ' + headlineIcon} aria-hidden="true" style="margin-right: var(--space-2)"></i>`}
            ${healthLabel}
          </div>
          <div class="hero-health-reasons">${healthSub}</div>
        </div>
      </div>

      <div class="hero-clock">
        <div class="hero-clock-line">
          <span class="hero-clock-label">elapsed</span>
          <span class="hero-clock-val">${elapsed != null ? fmtDuration(elapsed) : '---'}</span>
        </div>
        <div class="hero-clock-line">
          <span class="hero-clock-label">eta</span>
          <span class=${'hero-clock-val' + (eta != null ? '' : ' hero-clock-val--dim')}>
            ${eta != null ? fmtDuration(eta) : '—'}
          </span>
        </div>
      </div>

      <div class="hero-phase">
        ${terminalPhase
          ? html`
              <div class="hero-phase-head">
                <span class="hero-phase-name">${phase}</span>
                <span class="hero-phase-pct">${fmtPercent(100)}</span>
              </div>
              <div class="hero-phase-track">
                <div class="hero-phase-fill hero-phase-fill--done" style="width: 100%"></div>
              </div>
              <div class="hero-phase-sub">
                ${elapsed != null ? fmtDuration(elapsed) + ' · ' : ''}${fmtInt(recs.successRecords)} records
              </div>
            `
          : active
          ? html`
              <div class="hero-phase-head">
                <span class="hero-phase-name">${active.name}</span>
                <span class="hero-phase-pct">${pct != null ? fmtPercent(pct, 1) : '—'}</span>
              </div>
              <div class="hero-phase-track">
                <div class="hero-phase-fill" style=${'width: ' + (pct ?? 0) + '%'}></div>
              </div>
              <div class="hero-phase-sub">
                ${fmtInt(completedCount)}${totalCount ? ' / ' + fmtInt(totalCount) : ''} completed
              </div>
            `
          : html`
              <div class="hero-phase-head">
                <span class="hero-phase-name hero-phase-name--idle">no active phase</span>
              </div>
              <div class="hero-phase-track"></div>
              <div class="hero-phase-sub">waiting for first phase to start</div>
            `}
      </div>
    </div>
  `;
}
