// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * One card per phase (keyed by the backend's real phase name).
 * Shows: progress bar, badge (pending/running/grace/complete), and
 * quick stats (sent / completed / errors).
 */

import { html } from 'htm/preact';
import { phases } from '../lib/state.js';
import { fmtInt, fmtPercent, fmtDuration, fmtNumber } from '../lib/format.js';

function computeProgress(p) {
  const total = p.total_expected_requests ?? p.expected_requests ?? null;
  const completed = p.final_requests_completed ?? p.requests_completed ?? p.completed ?? 0;
  if (total && total > 0) {
    return { pct: Math.min(100, (completed / total) * 100), completed, total };
  }
  // Duration-controlled phase: no request total, so express progress as
  // elapsed / expected_duration so the bar and the Progress stat still move.
  const expectedDur = p.expected_duration_sec ?? p.expected_duration;
  const startNs = p.start_ns;
  if (expectedDur && expectedDur > 0 && typeof startNs === 'number') {
    const elapsed = Math.max(0, (Date.now() - Number(startNs) / 1e6) / 1000);
    const pct = Math.min(100, (elapsed / expectedDur) * 100);
    return { pct, completed, total: null };
  }
  return { pct: null, completed, total: null };
}

/** Live elapsed / ETA for this phase, derived from start_ns + completion rate.
 *  Covers both count-controlled (total_expected_requests) and
 *  duration-controlled (expected_duration_sec) phases. */
function computeTiming(p) {
  const startNs = p.start_ns;
  if (!startNs) return { elapsedSec: null, rate: null, etaSec: null };
  const elapsedSec = Math.max(0, (Date.now() - Number(startNs) / 1e6) / 1000);
  const completed = p.final_requests_completed ?? p.requests_completed ?? p.completed ?? 0;
  const rate = elapsedSec > 0 ? completed / elapsedSec : null;

  if (p.complete) return { elapsedSec, rate, etaSec: 0 };

  // Duration-controlled phase (rate mode): ETA = expected - elapsed.
  const expectedDur = p.expected_duration_sec ?? p.expected_duration;
  if (expectedDur && expectedDur > 0) {
    return { elapsedSec, rate, etaSec: Math.max(0, expectedDur - elapsedSec) };
  }

  // Count-controlled phase: extrapolate from current rate.
  const total = p.total_expected_requests ?? p.expected_requests;
  const etaSec = (rate && rate > 0 && total && completed < total)
    ? (total - completed) / rate
    : null;
  return { elapsedSec, rate, etaSec };
}

function badgeClass(p) {
  if (p.complete) return 'phase-badge--complete';
  if (p.grace) return 'phase-badge--grace';
  if (p.active) return 'phase-badge--running';
  return 'phase-badge--pending';
}

function badgeText(p) {
  if (p.complete) return 'Complete';
  if (p.grace) return 'Grace';
  if (p.active) return 'Running';
  return 'Pending';
}

function cardClass(p) {
  const classes = ['phase-card'];
  if (p.complete) classes.push('complete');
  else if (p.grace) classes.push('grace');
  return classes.join(' ');
}

export function PhaseCards() {
  const all = phases.value;
  const names = Object.keys(all);

  if (names.length === 0) {
    return html`
      <div class="card">
        <div class="card-title">Phases</div>
        <div class="empty">Waiting for benchmark to start...</div>
      </div>
    `;
  }

  return html`
    <div>
      <div class="card-title" style="padding-left: 4px; margin-bottom: 8px">Phases</div>
      <div class="phases-grid">
        ${names.map((name) => {
          const p = all[name];
          const { pct, completed, total } = computeProgress(p);
          const { elapsedSec, rate, etaSec } = computeTiming(p);
          return html`
            <div class=${cardClass(p)} key=${name}>
              <div class="phase-header">
                <span class="phase-name">${name}</span>
                <span class=${'phase-badge ' + badgeClass(p)}>${badgeText(p)}</span>
              </div>
              <div class="phase-track">
                <div class="phase-fill" style=${'width: ' + (pct != null ? pct + '%' : '0%')}></div>
              </div>
              <div class="phase-stats">
                <div class="phase-stat">
                  <span class="phase-stat-label">Progress</span>
                  <span class="phase-stat-val">${pct != null ? fmtPercent(pct) : '---'}</span>
                </div>
                <div class="phase-stat">
                  <span class="phase-stat-label">Completed</span>
                  <span class="phase-stat-val">${fmtInt(completed)}${total ? ` / ${fmtInt(total)}` : ''}</span>
                </div>
                <div class="phase-stat">
                  <span class="phase-stat-label">Errors</span>
                  <span class="phase-stat-val">${fmtInt(p.request_errors ?? p.errors ?? 0)}</span>
                </div>
                <div class="phase-stat">
                  <span class="phase-stat-label">Rate</span>
                  <span class="phase-stat-val">${rate != null ? fmtNumber(rate, 1) + ' req/s' : '---'}</span>
                </div>
                <div class="phase-stat">
                  <span class="phase-stat-label">Elapsed</span>
                  <span class="phase-stat-val">${elapsedSec != null ? fmtDuration(elapsedSec) : '---'}</span>
                </div>
                <div class="phase-stat">
                  <span class="phase-stat-label">ETA</span>
                  <span class="phase-stat-val">${p.complete ? '—' : (etaSec != null ? fmtDuration(etaSec) : '---')}</span>
                </div>
              </div>
            </div>
          `;
        })}
      </div>
    </div>
  `;
}
