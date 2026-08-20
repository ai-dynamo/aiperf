// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { fmtInt } from '../lib/format.js';

/**
 * Phase progress bar row.
 * @param {{ phases: Array<{name: string, completed: number, total: number}> }} props
 */
export function PhaseBar({ phases }) {
  if (!phases || phases.length === 0) {
    return html`<div class="phase-bar phase-bar--empty">No phases</div>`;
  }

  // High-cardinality sweeps (12+ phases) are wrapped by the page in
  // overflow-x:auto, but the items themselves use no min-width:0 so they
  // would refuse to shrink and force horizontal scroll on narrow viewports.
  // Allow each item to shrink to a sensible floor and cap the row width
  // so the inner content can wrap rather than push past the viewport.
  return html`
    <div class="phase-bar" style="min-width:0;max-width:100%">
      ${phases.map((phase) => {
        const pct =
          phase.total > 0 ? Math.round((phase.completed / phase.total) * 100) : 0;
        const done = phase.completed >= phase.total && phase.total > 0;
        // Math.round carries 99.5%+ up to 100, so 999 of 1000 rendered a full
        // bar, a "100%" tooltip and aria-valuenow=100 while the phase was
        // still labelled "in progress" -- three readouts contradicting each
        // other, and a screen reader announcing a phase as complete before it
        // was. Reserve 100 for phases that actually finished; an unfinished
        // phase tops out at 99. When done, clamp any over-count back to 100.
        const boundedPct = done ? Math.min(100, pct) : Math.min(99, pct);
        const active = !done && phase.completed > 0;
        const statusClass = done
          ? 'phase-bar-item--done'
          : active
          ? 'phase-bar-item--active'
          : 'phase-bar-item--pending';
        const statusLabel = done ? 'done' : active ? 'in progress' : 'pending';
        const tooltip =
          `${phase.name}: ${statusLabel} — ${fmtInt(phase.completed)} of ${fmtInt(phase.total)} (${boundedPct}%)`;

        return html`
          <div
            key=${phase.name}
            class=${'phase-bar-item ' + statusClass}
            title=${tooltip}
            aria-label=${tooltip}
            style="min-width:0;flex:1 1 0"
          >
            <div class="phase-bar-header" style="min-width:0;gap:4px">
              <span class="phase-bar-name" style="min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${phase.name}</span>
              <span class="phase-bar-fraction" style="flex-shrink:0">
                ${fmtInt(phase.completed)}/${fmtInt(phase.total)}
              </span>
            </div>
            <div class="phase-bar-track">
              <div
                class="phase-bar-fill"
                style=${'width: ' + boundedPct + '%'}
                role="progressbar"
                aria-valuenow=${boundedPct}
                aria-valuemin="0"
                aria-valuemax="100"
              />
            </div>
          </div>
        `;
      })}
    </div>
  `;
}
