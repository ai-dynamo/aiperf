// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { fmtInt, fmtNumber, fmtReqPerSecond } from '../lib/format.js';

/**
 * Records-processing panel — visualises the records-manager pipeline that
 * trails the request loop. Each `PhaseProgress` entry on the CR carries
 * `recordsSuccess` / `recordsError` / `recordsPerSecond` /
 * `recordsProgressPercent` / `recordsEtaSeconds` plus the corresponding
 * `requestsCompleted` figure, so we can show how far behind the records
 * pipeline is at a glance — the number the api-server dashboard's status
 * bar exposes as "Records: N".
 *
 * Backed by ``aiperf.kubernetes.crd_models.PhaseProgress`` (K8sCamelModel — JSON
 * keys are camelCase).
 *
 * @param {{ phases: Record<string, any> }} props
 */
export function RecordProcessing({ phases }) {
  const phaseEntries = Object.entries(phases ?? {});
  if (phaseEntries.length === 0) return null;

  // Aggregate across all phases for the headline numbers.
  let totalSuccess = 0;
  let totalError = 0;
  let totalRequestsCompleted = 0;
  let activeRps = 0;
  let activeEta = null;

  for (const [, p] of phaseEntries) {
    // Phase value can be null/undefined when the operator emits a
    // partially-populated entry (e.g. a phase declared but never started).
    // Skip rather than crash on `p.recordsSuccess`.
    if (p == null || typeof p !== 'object') continue;
    totalSuccess += p.recordsSuccess ?? 0;
    totalError += p.recordsError ?? 0;
    totalRequestsCompleted += p.requestsCompleted ?? 0;
    const isRequestsComplete = p.isRequestsComplete ?? false;
    const isRecordsComplete = p.isRecordsComplete ?? false;
    const isActive = !isRequestsComplete || !isRecordsComplete;
    if (isActive) {
      activeRps += p.recordsPerSecond ?? p.records_per_second ?? 0;
      const eta = p.recordsEtaSeconds ?? p.records_eta_seconds ?? null;
      if (eta != null && (activeEta == null || eta > activeEta)) {
        activeEta = eta;
      }
    }
  }

  const totalRecords = totalSuccess + totalError;
  const errorPct = totalRecords > 0 ? (totalError / totalRecords) * 100 : 0;
  // "Records lag" = requests already completed but not yet flowing through
  // the records-manager pipeline. Negative shouldn't happen in practice
  // (records can't outrun requests) — clamp at 0.
  const recordsLag = Math.max(0, totalRequestsCompleted - totalRecords);

  return html`
    <div class="card record-processing" data-testid="record-processing">
      <div class="card-title" style="display:flex;align-items:center;gap:8px">
        <span>Record Processing</span>
        ${activeRps > 0 ? html`
          <span class="record-live-pulse" title="Records currently flowing through the records-manager"></span>
        ` : null}
      </div>

      <div class="record-summary">
        <div class="record-summary-cell">
          <div class="record-summary-label">Total</div>
          <div class="record-summary-value">${fmtInt(totalRecords)}</div>
        </div>
        <div class="record-summary-cell">
          <div class="record-summary-label">Success</div>
          <div class="record-summary-value record-summary-value--ok">${fmtInt(totalSuccess)}</div>
        </div>
        <div class="record-summary-cell">
          <div class="record-summary-label">Errors</div>
          <div class=${'record-summary-value' + (totalError > 0 ? ' record-summary-value--bad' : '')}>${fmtInt(totalError)}</div>
        </div>
        <div class="record-summary-cell" title="Errors as % of total records">
          <div class="record-summary-label">Err %</div>
          <div class=${'record-summary-value' + (errorPct >= 1 ? ' record-summary-value--bad' : errorPct > 0 ? ' record-summary-value--warn' : '')}>
            ${totalRecords > 0 ? fmtNumber(errorPct, errorPct < 1 ? 2 : 1) + '%' : '---'}
          </div>
        </div>
        <div class="record-summary-cell" title="Records currently being processed per second (sum across active phases)">
          <div class="record-summary-label">rec/s</div>
          <div class="record-summary-value">${activeRps > 0 ? fmtReqPerSecond(activeRps) : '---'}</div>
        </div>
        <div class="record-summary-cell" title="Requests completed but not yet processed by the records-manager">
          <div class="record-summary-label">Lag</div>
          <div class=${'record-summary-value' + (recordsLag > 100 ? ' record-summary-value--warn' : '')}>${fmtInt(recordsLag)}</div>
        </div>
        ${activeEta != null ? html`
          <div class="record-summary-cell" title="Estimated seconds until all records are processed (longest active phase)">
            <div class="record-summary-label">ETA</div>
            <div class="record-summary-value">${formatEta(activeEta)}</div>
          </div>
        ` : null}
      </div>

      <table class="record-phase-table">
        <thead>
          <tr>
            <th>Phase</th>
            <th>Records</th>
            <th>Errors</th>
            <th>rec/s</th>
            <th>Progress</th>
          </tr>
        </thead>
        <tbody>
          ${phaseEntries.map(([name, p]) => {
            if (p == null || typeof p !== 'object') return null;
            const rs = p.recordsSuccess ?? 0;
            const re = p.recordsError ?? 0;
            const rps = p.recordsPerSecond ?? 0;
            const recPct = p.recordsProgressPercent ?? 0;
            const phaseRecords = rs + re;
            const isRequestsComplete = p.isRequestsComplete ?? false;
            const isRecordsComplete = p.isRecordsComplete ?? false;
            const wasCancelled = p.wasCancelled ?? false;
            const timedOut = p.timeoutTriggered ?? false;
            const isActive = !isRequestsComplete || !isRecordsComplete;
            const rowState = wasCancelled ? 'cancelled'
              : timedOut ? 'timeout'
              : (isRequestsComplete && isRecordsComplete) ? 'done'
              : isActive ? 'active'
              : 'pending';

            return html`
              <tr key=${name} data-state=${rowState}>
                <td>
                  <span class="record-phase-name">${name}</span>
                  ${rowState === 'cancelled' ? html`<span class="record-phase-tag record-phase-tag--bad">cancelled</span>` : null}
                  ${rowState === 'timeout' ? html`<span class="record-phase-tag record-phase-tag--warn">timeout</span>` : null}
                </td>
                <td class="num">${fmtInt(phaseRecords)}</td>
                <td class=${'num' + (re > 0 ? ' record-cell--bad' : '')}>${fmtInt(re)}</td>
                <td class="num">${rps > 0 ? fmtReqPerSecond(rps) : '---'}</td>
                <td>
                  <div class="record-progress-cell">
                    <div class="record-progress-track">
                      <div class=${'record-progress-fill record-progress-fill--' + rowState}
                           style=${'width:' + Math.min(100, recPct).toFixed(1) + '%'} />
                    </div>
                    <span class="record-progress-pct">${fmtNumber(recPct, 1)}%</span>
                  </div>
                </td>
              </tr>
            `;
          })}
        </tbody>
      </table>
    </div>
  `;
}

function formatEta(seconds) {
  if (seconds == null || !isFinite(seconds)) return '---';
  if (seconds < 60) return Math.round(seconds) + 's';
  if (seconds < 3600) {
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return s > 0 ? `${m}m${s}s` : `${m}m`;
  }
  const h = Math.floor(seconds / 3600);
  const m = Math.round((seconds % 3600) / 60);
  return m > 0 ? `${h}h${m}m` : `${h}h`;
}
