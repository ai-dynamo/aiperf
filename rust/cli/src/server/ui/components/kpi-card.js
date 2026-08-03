// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Compact KPI tile with an optional Material-style outlined icon — distilled
// from the operator UI's ``kpi-card.js`` icon registry. Big tabular-number
// value, unit suffix, and an optional sub-line. Renders an em-dash for a
// missing value so hero rows stay aligned when a run lacks a metric.

import { html } from 'htm/preact';

const STROKE =
  'fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"';

function MetricIcon({ name }) {
  switch (name) {
    case 'speed':
      return html`<svg viewBox="0 0 24 24" ${STROKE} aria-hidden="true"><path d="M3 16 a9 9 0 0 1 18 0" /><line x1="12" y1="16" x2="16.5" y2="9" /><circle cx="12" cy="16" r="1.4" fill="currentColor" stroke="none" /></svg>`;
    case 'clock':
      return html`<svg viewBox="0 0 24 24" ${STROKE} aria-hidden="true"><circle cx="12" cy="12" r="9" /><line x1="12" y1="7" x2="12" y2="12" /><line x1="12" y1="12" x2="15.5" y2="14" /></svg>`;
    case 'timer':
      return html`<svg viewBox="0 0 24 24" ${STROKE} aria-hidden="true"><circle cx="12" cy="13" r="8" /><line x1="9" y1="2.5" x2="15" y2="2.5" /><line x1="12" y1="2.5" x2="12" y2="6" /><line x1="12" y1="13" x2="16" y2="9" /></svg>`;
    case 'tokens':
      return html`<svg viewBox="0 0 24 24" ${STROKE} aria-hidden="true"><line x1="4" y1="6" x2="16" y2="6" /><line x1="4" y1="12" x2="20" y2="12" /><line x1="4" y1="18" x2="11" y2="18" /></svg>`;
    case 'requests':
      return html`<svg viewBox="0 0 24 24" ${STROKE} aria-hidden="true"><rect x="4" y="7" width="14" height="14" rx="1.5" /><path d="M7 4 h13 a1.5 1.5 0 0 1 1.5 1.5 v13" /><line x1="7" y1="11" x2="15" y2="11" /><line x1="7" y1="15" x2="15" y2="15" /></svg>`;
    case 'trending-up':
      return html`<svg viewBox="0 0 24 24" ${STROKE} aria-hidden="true"><polyline points="3,17 9,11 13,15 21,7" /><polyline points="14,7 21,7 21,14" /></svg>`;
    case 'trophy':
      return html`<svg viewBox="0 0 24 24" ${STROKE} aria-hidden="true"><path d="M8 21 h8" /><path d="M12 17 v4" /><path d="M7 4 h10 v3 a5 5 0 0 1 -10 0 z" /><path d="M17 5 h3 v2 a3 3 0 0 1 -3 3" /><path d="M7 5 h-3 v2 a3 3 0 0 0 3 3" /><path d="M9 13 a4 4 0 0 0 6 0" /></svg>`;
    case 'check':
      return html`<svg viewBox="0 0 24 24" ${STROKE} aria-hidden="true"><circle cx="12" cy="12" r="9" /><polyline points="8,12.5 11,15.5 16,9.5" /></svg>`;
    default:
      return null;
  }
}

/**
 * @param {object} props
 * @param {string} props.label
 * @param {string|number} [props.value] - already-formatted display value.
 * @param {string} [props.unit]
 * @param {string} [props.sub]
 * @param {string} [props.icon] - registry key (speed/clock/timer/tokens/…).
 * @param {string} [props.title] - hover tooltip.
 */
export function KpiCard({ label, value, unit, sub, icon, title }) {
  const shown = value == null || value === '' ? '—' : value;
  return html`
    <div class="kpi-card" title=${title ?? label}>
      ${icon && html`<div class="kpi-icon"><${MetricIcon} name=${icon} /></div>`}
      <div class="kpi-body">
        <div class="kpi-label">${label}</div>
        <div class="kpi-value-row">
          <span class="kpi-value">${shown}</span>
          ${unit && html`<span class="kpi-unit">${unit}</span>`}
        </div>
        ${sub && html`<div class="kpi-sub">${sub}</div>`}
      </div>
    </div>
  `;
}
