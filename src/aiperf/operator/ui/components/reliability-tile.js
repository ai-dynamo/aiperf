// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Reliability KPI tile — operator-UI port of the ``ReliabilityTile`` inside
 * ``static-v2/components/realtime-metrics.js``.
 *
 * Prop-driven: ``<ReliabilityTile summary=${status.summary} config=${config} />``.
 *
 * Behavior:
 *   - When SLOs are declared (``config.spec.benchmark.slos`` or
 *     ``config.spec.slos`` is non-empty): headline the *violation count*
 *     (``N failed``). N = ``total_requests - goodput_count`` if
 *     ``goodput_count`` is present on the summary; otherwise falls back to
 *     ``error_count`` / ``total_requests * error_rate``. Chip is green iff
 *     ``N === 0``, warn otherwise.
 *   - When no SLOs are declared: headline the success rate
 *     (``100 - error_rate``%). Chip is green iff ``error_count === 0``.
 *
 * Returns ``null`` when the summary carries nothing actionable (no request
 * count + no error rate + no goodput) so the host page can keep the KPI row
 * tight on archived/idle jobs.
 */

import { html } from 'htm/preact';
import { fmtNumber, fmtInt, fmtPercent } from '../lib/format.js';

/** Slug-consistent testid so e2e can locate the tile. */
const TESTID = 'kpi-reliability';

function userSlos(config) {
  return config?.spec?.benchmark?.slos ?? config?.spec?.slos ?? null;
}

function hasUserSlos(config) {
  const slos = userSlos(config);
  return slos && typeof slos === 'object' && Object.keys(slos).length > 0;
}

/** Pull a numeric summary stat, tolerating missing fields. */
function num(value) {
  if (value == null) return null;
  const n = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(n) ? n : null;
}

export function ReliabilityTile({ summary, config }) {
  if (!summary || typeof summary !== 'object') return null;

  const totalRequests = num(summary.total_requests);
  const errorRate = num(summary.error_rate);
  const goodputCount = num(summary.goodput_count);
  // ``error_count`` isn't always on the summary — derive from rate if needed.
  const errorCount = num(summary.error_count)
    ?? (totalRequests != null && errorRate != null
        ? Math.round((errorRate / 100) * totalRequests)
        : null);

  if (hasUserSlos(config)) {
    // SLO-aware "N failed" headline. Prefer goodput_count when present; else
    // fall back to error_count so the tile stays informative.
    let failedCount = null;
    if (totalRequests != null && goodputCount != null) {
      failedCount = Math.max(0, Math.round(totalRequests - goodputCount));
    } else if (errorCount != null) {
      failedCount = Math.max(0, errorCount);
    }
    if (failedCount == null) return null;

    const kind = failedCount === 0 ? 'good' : 'warn';
    const passPct = (totalRequests != null && totalRequests > 0)
      ? Math.max(0, 100 - (failedCount / totalRequests) * 100)
      : null;
    const sloList = Object.keys(userSlos(config) ?? {}).join(', ');
    const chipTitle = sloList
      ? 'Requests that missed at least one SLO (' + sloList + ')'
      : 'Requests that missed at least one SLO';

    return html`
      <div
        class="metric-card"
        data-testid=${TESTID}
      >
        <div class="metric-label-row">
          <span class="metric-label">Reliability</span>
          <span
            class=${'kpi-chip kpi-chip--' + kind}
            title=${chipTitle}
          >
            ${kind === 'good'
              ? html`<span>✓</span><span class="kpi-chip-thresh">0 failed</span>`
              : html`<span>✗</span><span class="kpi-chip-thresh">${fmtInt(failedCount)} failed</span>`}
          </span>
        </div>
        <div class="metric-val-row">
          <span class="metric-val" style=${'color: var(--' + (kind === 'good' ? 'green' : 'amber') + ')'}>
            ${fmtInt(failedCount)}
          </span>
          <span class="metric-unit">failed</span>
        </div>
        <div class="metric-sub">
          ${passPct != null
            ? html`<span>${fmtPercent(passPct, 1)} passed</span>`
            : html`<span>of ${totalRequests != null ? fmtInt(totalRequests) : '---'} requests</span>`}
        </div>
      </div>
    `;
  }

  // No SLOs → fall back to Success Rate derived from error_rate / error_count.
  if (errorRate == null && errorCount == null) return null;

  const rate = errorRate ?? 0;
  const success = Math.max(0, 100 - rate);
  const kind = (errorCount ?? 0) === 0 ? 'good' : 'warn';

  return html`
    <div
      class="metric-card"
      data-testid=${TESTID}
    >
      <div class="metric-label-row">
        <span class="metric-label">Reliability</span>
        <span class=${'kpi-chip kpi-chip--' + kind}>
          ${kind === 'good'
            ? html`<span>✓</span><span class="kpi-chip-thresh">0 errors</span>`
            : html`<span>✗</span><span class="kpi-chip-thresh">${fmtInt(errorCount ?? 0)} errors</span>`}
        </span>
      </div>
      <div class="metric-val-row">
        <span class="metric-val" style=${'color: var(--' + (kind === 'good' ? 'green' : 'amber') + ')'}>
          ${fmtNumber(success, 2)}
        </span>
        <span class="metric-unit">%</span>
      </div>
      <div class="metric-sub">
        <span>errors ${fmtInt(errorCount ?? 0)}</span>
      </div>
    </div>
  `;
}
