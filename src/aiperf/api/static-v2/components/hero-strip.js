// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Hero strip — the focal point of the live dashboard.
 *
 * Answers three questions at a glance:
 *   1. Is my run healthy right now?  (SLO-compliance traffic light)
 *   2. How much longer?               (elapsed + ETA)
 *   3. What's it doing?               (active-phase big progress bar)
 *
 * Everything shown here is derived from existing signals — no new
 * message types required.
 */

import { html } from 'htm/preact';
import {
  connection, config, phases, records, realtimeMetrics, runStartedAt,
} from '../lib/state.js';
import { fmtInt, fmtDuration, fmtPercent } from '../lib/format.js';

/** Look up a metric by tag in the realtimeMetrics list. */
function byTag(metrics, tag) {
  for (const m of metrics) if (m?.tag === tag) return m;
  return null;
}

/** Classify overall run health.
 *
 *  Three signals, ranked:
 *   1. SLO p99 violation (value > user threshold)   → 'error' + 'slo'
 *   2. Goodput < 100% (some requests missed SLO)    → bump to warn + 'goodput'
 *   3. Any request errors                            → bump to warn + 'errors'
 *
 *  Errors alone do not trip 'error' state — they're a different reliability
 *  dimension from the latency SLO. Only an actual SLO p99 miss is 'error'.
 *  Labels are driven by which category fired so the headline matches the
 *  reason text underneath.
 */
function classifyHealth(cfg, metrics, recs) {
  const slos = cfg?.slos ?? null;
  const byT = {};
  for (const m of metrics) if (m?.tag) byT[m.tag] = m;

  if (metrics.length === 0 && recs.successRecords === 0 && recs.errorRecords === 0) {
    return { status: 'idle', category: null, reasons: [] };
  }

  let status = 'ok';
  let category = null;
  const reasons = [];

  // SLO p99 violations — the strongest signal.
  if (slos) {
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

  // Goodput violations → warn (unless already error).
  // Counter/derived metrics populate ``avg`` (the single scalar value), not
  // ``current`` — fall back so the hero goes amber during live runs too.
  const goodCount = byT['good_request_count']?.current ?? byT['good_request_count']?.avg;
  const reqCount = byT['request_count']?.current ?? byT['request_count']?.avg;
  if (goodCount != null && reqCount != null && goodCount < reqCount) {
    const failed = Math.round(reqCount - goodCount);
    if (status !== 'error') { status = 'warn'; category = category ?? 'goodput'; }
    reasons.push(`${fmtInt(failed)} requests missed SLO`);
  }

  // Request errors → warn (unless already error).
  const errorCount = byT['error_request_count']?.current ?? byT['error_request_count']?.avg ?? recs.errorRecords ?? 0;
  if (errorCount > 0) {
    if (status !== 'error') { status = 'warn'; category = category ?? 'errors'; }
    // Errors reason after goodput so the SLO-centric reason comes first.
    reasons.push(`${fmtInt(errorCount)} request errors`);
  }

  return { status, category, reasons };
}

/** Big active-phase bar. If multiple phases are running simultaneously (rare)
 *  we pick the one with the most completed requests as the focal point. */
function pickActivePhase(phaseMap) {
  const running = Object.values(phaseMap).filter(p => p.active && !p.complete);
  if (running.length === 0) return null;
  running.sort((a, b) => {
    const ac = a.requests_completed ?? a.completed ?? 0;
    const bc = b.requests_completed ?? b.completed ?? 0;
    return bc - ac;
  });
  return running[0];
}

/** Live ETA based on current completion rate. Returns seconds or null.
 *  Covers both count-based phases (total_expected_requests) and
 *  duration-based phases (expected_duration_sec) by picking whichever
 *  the backend actually reports. */
function estimateEtaSec(phase) {
  if (!phase) return null;
  if (phase.complete) return 0;
  const startNs = phase.start_ns;
  if (!startNs) return null;
  const elapsedSec = Math.max(0, (Date.now() - Number(startNs) / 1e6) / 1000);

  // Duration-controlled phase (rate mode) — remaining = expected - elapsed.
  const expectedDur = phase.expected_duration_sec ?? phase.expected_duration;
  if (expectedDur != null && expectedDur > 0) {
    return Math.max(0, expectedDur - elapsedSec);
  }

  // Count-controlled phase — extrapolate from rate.
  const total = phase.total_expected_requests ?? phase.expected_requests;
  const completed = phase.final_requests_completed
    ?? phase.requests_completed ?? phase.completed ?? 0;
  if (!total || completed <= 0 || elapsedSec <= 0) return null;
  const rate = completed / elapsedSec;
  if (rate <= 0) return null;
  return Math.max(0, total - completed) / rate;
}

/** Elapsed seconds for the active run, derived from whichever phase started
 *  first. Covers the case where the dashboard joins mid-run and never sees
 *  ``credit_phase_start`` (which is when ``runStartedAt`` gets set).
 */
function overallElapsedSec(phaseMap) {
  const startedAt = runStartedAt.value;
  let minStartMs = startedAt ?? Infinity;
  for (const p of Object.values(phaseMap)) {
    if (typeof p?.start_ns === 'number' && p.start_ns > 0) {
      const ms = Number(p.start_ns) / 1e6;
      if (ms < minStartMs) minStartMs = ms;
    }
  }
  if (!isFinite(minStartMs)) return null;
  return Math.max(0, (Date.now() - minStartMs) / 1000);
}

export function HeroStrip() {
  // Touch signals so Preact re-renders when they change.
  const conn = connection.value;
  const cfg = config.value;
  const phaseMap = phases.value;
  const metrics = realtimeMetrics.value;
  const recs = records.value;

  // Show nothing before the WS has even connected.
  if (conn !== 'connected' && metrics.length === 0 && Object.keys(phaseMap).length === 0) {
    return null;
  }

  const health = classifyHealth(cfg, metrics, recs);
  const active = pickActivePhase(phaseMap);
  const activePct = (() => {
    if (!active) return null;
    const total = active.total_expected_requests ?? active.expected_requests;
    const completed = active.final_requests_completed
      ?? active.requests_completed ?? active.completed ?? 0;
    if (total && total > 0) return Math.min(100, (completed / total) * 100);

    // Duration-controlled phase — use elapsed / expected_duration_sec so
    // the hero bar shows meaningful progress during rate-based runs too.
    const expectedDur = active.expected_duration_sec ?? active.expected_duration;
    const startNs = active.start_ns;
    if (expectedDur && expectedDur > 0 && typeof startNs === 'number') {
      const elapsed = Math.max(0, (Date.now() - Number(startNs) / 1e6) / 1000);
      return Math.min(100, (elapsed / expectedDur) * 100);
    }
    return null;
  })();
  const eta = estimateEtaSec(active);
  const elapsed = overallElapsedSec(phaseMap);

  // Primary health label — driven by the category that actually fired so
  // the headline matches the reasons list underneath instead of saying
  // "SLO violated" when what actually happened is request errors.
  const healthLabel = (() => {
    if (health.status === 'idle') return 'Waiting for data';
    if (health.status === 'ok')   return 'On target';
    switch (health.category) {
      case 'slo':     return 'SLO violated';
      case 'goodput': return 'SLO slipping';
      case 'errors':  return 'Errors reported';
      default:        return 'Attention needed';
    }
  })();

  return html`
    <div class=${'hero hero--' + health.status}>
      <div class="hero-health">
        <div class=${'hero-health-dot hero-health-dot--' + health.status}></div>
        <div class="hero-health-text">
          <div class="hero-health-label">${healthLabel}</div>
          <div class="hero-health-reasons">
            ${health.reasons.length > 0
              ? health.reasons.slice(0, 2).join(' · ')
              : (health.status === 'ok' ? 'all declared SLOs passing' : 'no judgment — no SLOs declared')}
          </div>
        </div>
      </div>

      <div class="hero-clock">
        <div class="hero-clock-line">
          <span class="hero-clock-label">elapsed</span>
          <span class="hero-clock-val">${elapsed != null ? fmtDuration(elapsed) : '--'}</span>
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
              <span class="hero-phase-pct">${activePct != null ? fmtPercent(activePct, 1) : '—'}</span>
            </div>
            <div class="hero-phase-track">
              <div class="hero-phase-fill" style=${'width: ' + (activePct ?? 0) + '%'}></div>
            </div>
            <div class="hero-phase-sub">
              ${fmtInt(active.requests_completed ?? active.completed ?? 0)}
              ${active.total_expected_requests ? ' / ' + fmtInt(active.total_expected_requests) : ''}
              completed
            </div>
          `
          : recs.complete
          ? html`
            <div class="hero-phase-head">
              <span class="hero-phase-name">benchmark complete</span>
              <span class="hero-phase-pct">${fmtPercent(100)}</span>
            </div>
            <div class="hero-phase-track"><div class="hero-phase-fill hero-phase-fill--done" style="width: 100%"></div></div>
            <div class="hero-phase-sub">${fmtInt(recs.successRecords + recs.errorRecords)} records processed</div>
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
