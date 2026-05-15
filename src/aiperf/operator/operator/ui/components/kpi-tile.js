// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * One streaming KPI tile. Pure presentational; the consumer (KpiRail)
 * computes value / delta / tone / sparkSeries from the live data signals.
 *
 * Sparkline rendering reuses the existing components/sparkline.js, sized to
 * 14px tall to fit the dense 6×3 grid on a laptop.
 */

import { html } from 'htm/preact';
import { Sparkline } from './sparkline.js';

export function KpiTile({
  label,
  value,           // preformatted string ('8.42k', '142', '—')
  unit,            // 'tok/s', '%', 'ms', etc.
  delta,           // string or null — '▲ 3.1%' / '▼ 8%' / '▬'
  deltaWindow,     // '30s' / '5m' / null
  deltaDirection,  // 'up' | 'down' | 'flat' | null — colors the delta
  sparkSeries,     // Array<{t, v}> — passes through to Sparkline; empty array OK
  tone,            // 'neutral' | 'good' | 'warn' | 'bad'
  stale,           // bool — tile shows 'stale Ns' meta
  meta,            // string — small top-right corner badge ('live' / 'final')
  tileId,          // for data-tile-id (test hook)
}) {
  const toneClass = tone && tone !== 'neutral' ? ` kpi-tile--${tone}` : '';
  const deltaClass = deltaDirection ? ` kpi-tile-delta--${deltaDirection}` : '';
  return html`
    <div class=${'kpi-tile' + toneClass} data-tile-id=${tileId}>
      <div class="kpi-tile-label">${label}</div>
      <div class="kpi-tile-val">
        <span class="kpi-tile-num">${value}</span>
        ${unit && html`<span class="kpi-tile-unit">${unit}</span>`}
      </div>
      ${delta != null && html`
        <div class=${'kpi-tile-delta' + deltaClass}>
          ${delta}${deltaWindow ? html`<span class="kpi-tile-window"> · ${deltaWindow}</span>` : null}
        </div>
      `}
      <div class="kpi-tile-spark">
        <${Sparkline} points=${sparkSeries ?? []} width=${140} height=${14}
                      stroke=${tone === 'bad' ? 'var(--red)' : tone === 'warn' ? 'var(--warn)' : 'var(--accent)'}
                      fill=${tone === 'bad' ? 'rgba(239,83,80,0.12)' : 'var(--accent-dim)'} />
      </div>
      ${(stale || meta) && html`
        <span class="kpi-tile-meta">${stale ? 'stale' : meta}</span>
      `}
    </div>
  `;
}
