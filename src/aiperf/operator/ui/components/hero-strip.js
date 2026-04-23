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

/** Pick the active phase (running, not complete) with the most completed
 *  requests as the focal point for the big progress bar.
 */
function pickActivePhase(phaseMap) {
  const entries = Object.entries(phaseMap ?? {});
  const running = entries.filter(([, p]) => p.active && !p.complete);
  if (running.length === 0) return null;
  running.sort(([, a], [, b]) => {
    const ac = a.requestsCompleted ?? a.requests_completed ?? a.completed ?? 0;
    const bc = b.requestsCompleted ?? b.requests_completed ?? b.completed ?? 0;
    return bc - ac;
  });
  const [name, data] = running[0];
  return { name, ...data };
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
  const complete =
    phase === 'Succeeded' ||
    phase === 'Failed' ||
    phase === 'Completed' ||
    phase === 'Error';
  const recs = {
    successRecords: summary?.total_requests ?? 0,
    errorRecords: 0,
    complete,
  };
  const slos = extractSlos(config);
  const health = classifyHealth(slos, metrics, recs);

  const active = pickActivePhase(phaseMap);
  const pct = activePct(active);
  const eta = estimateEtaSec(active);
  const elapsed = elapsedSec(info);

  const healthLabel = (() => {
    if (health.status === 'idle') return 'Waiting for data';
    if (health.status === 'ok') return 'On target';
    switch (health.category) {
      case 'slo':
        return 'SLO violated';
      case 'goodput':
        return 'SLO slipping';
      case 'errors':
        return 'Errors reported';
      default:
        return 'Attention needed';
    }
  })();

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
      class=${'hero hero--' + health.status}
      style=${style}
      onclick=${clickable ? onClick : null}
      data-testid="hero-strip"
    >
      <div class="hero-health">
        <div class=${'hero-health-dot hero-health-dot--' + health.status}></div>
        <div class="hero-health-text">
          <div class="hero-health-label">${healthLabel}</div>
          <div class="hero-health-reasons">
            ${health.reasons.length > 0
              ? health.reasons.slice(0, 2).join(' · ')
              : health.status === 'ok'
              ? 'all declared SLOs passing'
              : 'no judgment — no SLOs declared'}
          </div>
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
        ${active
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
          : complete
          ? html`
              <div class="hero-phase-head">
                <span class="hero-phase-name">benchmark complete</span>
                <span class="hero-phase-pct">${fmtPercent(100)}</span>
              </div>
              <div class="hero-phase-track">
                <div class="hero-phase-fill hero-phase-fill--done" style="width: 100%"></div>
              </div>
              <div class="hero-phase-sub">
                ${fmtInt(recs.successRecords)} records processed
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
