// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { palette } from '../lib/theme.js';
import { fmtNumber } from '../lib/format.js';

function formatMetricValue(value, unit) {
  const decimals = unit === 'ms' ? 1 : 0;
  return fmtNumber(value, decimals);
}

function formatCv(cv) {
  return cv == null ? '---' : `${fmtNumber(cv * 100, 1)}%`;
}

export function SweepWinnerSummary({ winner, metric }) {
  const metricLabel = metric?.label ?? winner?.metricKey ?? 'selected metric';
  const unit = metric?.unit ?? '';

  if (!winner) {
    return html`
      <section
        class="card"
        data-testid="sweep-winner-summary"
        style="margin-bottom: var(--space-4)"
      >
        <div class="card-title" style="margin:0 0 var(--space-1) 0">Winner summary</div>
        <div class="text-dim" style="font-size:var(--font-size-sm)">
          No completed variation has a finite ${metricLabel} value yet.
        </div>
      </section>
    `;
  }

  const label = winner.label || `v${winner.variation_index}`;
  const direction = winner.higherIsBetter ? 'higher is better' : 'lower is better';

  return html`
    <section
      class="card"
      data-testid="sweep-winner-summary"
      style=${
        `margin-bottom:var(--space-4);` +
        `border-color:${palette.peach}55;` +
        `background:linear-gradient(135deg, ${palette.peach}12, ${palette.bgCard} 44%);`
      }
    >
      <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:var(--space-3);flex-wrap:wrap">
        <div>
          <div class="card-title" style="margin:0">Winner summary</div>
          <div class="text-dim" style="font-size:var(--font-size-xs);margin-top:2px">
            ${metricLabel} · ${direction}
          </div>
        </div>
        <span
          style=${
            `border:1px solid ${palette.peach}66;` +
            `background:${palette.peach}18;` +
            `color:${palette.peach};` +
            `border-radius:999px;padding:4px 10px;` +
            `font-size:var(--font-size-xs);font-weight:700;`
          }
        >${direction}</span>
      </div>
      <div style="display:flex;align-items:end;justify-content:space-between;gap:var(--space-4);flex-wrap:wrap;margin-top:var(--space-3)">
        <div>
          <div style=${`font-size:var(--font-size-xs);color:${palette.muted};text-transform:uppercase;letter-spacing:0.08em`}>Variation</div>
          <div style="font-size:var(--font-size-lg);font-weight:800;margin-top:2px">${label}</div>
          <div class="text-dim" style="font-size:var(--font-size-xs);margin-top:2px">variation ${winner.variation_index}</div>
        </div>
        <div style="text-align:right">
          <div style=${`font-size:var(--font-size-xs);color:${palette.muted};text-transform:uppercase;letter-spacing:0.08em`}>${metricLabel}</div>
          <div style="font-size:28px;font-weight:850;line-height:1.1;font-variant-numeric:tabular-nums">
            ${formatMetricValue(winner.mean, unit)}${unit ? html`<span style="font-size:var(--font-size-sm);font-weight:700;margin-left:6px;color:${palette.subtext0}">${unit}</span>` : null}
          </div>
          <div class="text-dim" style="font-size:var(--font-size-xs);margin-top:4px">
            CV ${formatCv(winner.cv)} · n ${winner.n ?? '---'}
          </div>
        </div>
      </div>
    </section>
  `;
}
