// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Per-phase cards (progress bar + badge + quick stats).
 *
 * Ported from static-v2/components/phase-cards.js. Accepts the phases dict
 * straight from the operator's ``status.phases`` — key = phase name, value =
 * per-phase stats (camelCase or snake_case both accepted).
 *
 * Fields like ``start_ns`` / ``active`` / ``complete`` / ``grace`` come from
 * the operator's live-status writer; when absent (archived runs) the timing
 * stats degrade gracefully to ``---``.
 */

import { html } from 'htm/preact';
import { fmtInt, fmtPercent, fmtDuration, fmtNumber } from '../lib/format.js';

function computeProgress(p) {
  const total =
    p.total_expected_requests ??
    p.expected_requests ??
    p.requestsTotal ??
    p.requests_total ??
    null;
  const completed =
    p.final_requests_completed ??
    p.requestsCompleted ??
    p.requests_completed ??
    p.completed ??
    0;
  if (total && total > 0) {
    return { pct: Math.min(100, (completed / total) * 100), completed, total };
  }
  const pctFromField = p.requestsProgressPercent;
  if (pctFromField != null) {
    return {
      pct: Math.min(100, Math.max(0, Number(pctFromField))),
      completed,
      total: null,
    };
  }
  return { pct: null, completed, total: null };
}

function computeTiming(p) {
  const startNs = p.start_ns;
  if (!startNs) return { elapsedSec: null, rate: null, etaSec: null };
  const elapsedSec = Math.max(0, (Date.now() - Number(startNs) / 1e6) / 1000);
  const completed =
    p.final_requests_completed ??
    p.requestsCompleted ??
    p.requests_completed ??
    p.completed ??
    0;
  const total =
    p.total_expected_requests ??
    p.expected_requests ??
    p.requestsTotal ??
    p.requests_total ??
    null;
  const rate = elapsedSec > 0 ? completed / elapsedSec : null;
  const etaSec =
    rate && rate > 0 && total && completed < total
      ? (total - completed) / rate
      : p.complete
      ? 0
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

export function PhaseCards({ phases }) {
  const all = phases ?? {};
  const names = Object.keys(all);

  if (names.length === 0) {
    return html`
      <div class="card">
        <div class="card-title">Phases</div>
        <div class="text-dim">Waiting for benchmark to start...</div>
      </div>
    `;
  }

  return html`
    <div data-testid="phase-cards">
      <div class="card-title" style="padding-left: 4px; margin-bottom: var(--space-2)">Phases</div>
      <div class="phases-grid">
        ${names.map((name) => {
          const p = all[name];
          const { pct, completed, total } = computeProgress(p);
          const { elapsedSec: elapsed, rate, etaSec } = computeTiming(p);
          const errorCount = p.request_errors ?? p.requestErrors ?? p.errors ?? 0;
          return html`
            <div class=${cardClass(p)} key=${name} data-testid=${'phase-card-' + name}>
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
                  <span class="phase-stat-val">
                    ${fmtInt(completed)}${total ? ` / ${fmtInt(total)}` : ''}
                  </span>
                </div>
                <div class="phase-stat">
                  <span class="phase-stat-label">Errors</span>
                  <span class="phase-stat-val">${fmtInt(errorCount)}</span>
                </div>
                <div class="phase-stat">
                  <span class="phase-stat-label">Rate</span>
                  <span class="phase-stat-val">${rate != null ? fmtNumber(rate, 1) + ' req/s' : '---'}</span>
                </div>
                <div class="phase-stat">
                  <span class="phase-stat-label">Elapsed</span>
                  <span class="phase-stat-val">${elapsed != null ? fmtDuration(elapsed) : '---'}</span>
                </div>
                <div class="phase-stat">
                  <span class="phase-stat-label">ETA</span>
                  <span class="phase-stat-val">${p.complete ? '—' : etaSec != null ? fmtDuration(etaSec) : '---'}</span>
                </div>
              </div>
            </div>
          `;
        })}
      </div>
    </div>
  `;
}
