// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { Strip } from './strip.js';
import { fmtInt } from '../lib/format.js';

/**
 * Records-processed progress strip. Replaces the records-progress portion
 * of the old RecordProcessing component.
 */
export function RecordsStrip({ processed, total, ratePerSec, etaSeconds }) {
  if (processed == null || total == null || total <= 0) {
    return html`
      <${Strip} label="records" testId="strip-records"
                meta=${processed != null ? `${fmtInt(processed)} processed` : '—'}>
        <div class="seg" style="left:0;width:0%;background:var(--accent)"></div>
      <//>
    `;
  }
  const pct = Math.min(1, Math.max(0, processed / total)) * 100;
  const rate = ratePerSec != null ? `${fmtInt(ratePerSec)}/s` : null;
  const eta = etaSeconds != null && isFinite(etaSeconds) && etaSeconds > 0
    ? `ETA ${formatEta(etaSeconds)}`
    : null;
  const meta = [`${fmtInt(processed)} / ${fmtInt(total)}`, rate, eta].filter(Boolean).join(' · ');
  return html`
    <${Strip} label="records" testId="strip-records" meta=${meta}>
      <div class="seg" style=${`left:0;width:${pct.toFixed(2)}%;background:rgba(118,185,0,0.7)`}></div>
    <//>
  `;
}

function formatEta(seconds) {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}:${String(Math.round(seconds % 60)).padStart(2, '0')}`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}:${String(m).padStart(2, '0')}:00`;
}
