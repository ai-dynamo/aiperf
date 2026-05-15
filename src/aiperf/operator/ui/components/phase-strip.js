// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { Strip } from './strip.js';

/**
 * Phase progress strip. Replaces the old PhaseBar component.
 *
 * `phases` is an ordered array of { name, status, progress } where
 * status ∈ {'pending', 'active', 'completed'} and progress ∈ [0, 1] (active only).
 *
 * Each phase is a horizontal segment proportional to a fixed weight (1 by default).
 * The active segment has a partial fill showing in-phase progress.
 */
const PHASE_COLORS = {
  pending:   'rgba(180, 180, 180, 0.25)',
  active:    'rgba(118, 185, 0, 0.85)',
  completed: 'rgba(118, 185, 0, 0.50)',
};

export function PhaseStrip({ phases, current, etaText }) {
  const list = Array.isArray(phases) ? phases : [];
  const total = list.length || 1;
  // Build the meta as a Preact fragment so the current phase renders as <strong>
  // without ever round-tripping through innerHTML / markdown literals.
  const metaParts = [];
  list.forEach((p, i) => {
    if (i > 0) metaParts.push(' · ');
    if (p.name === current) {
      metaParts.push(html`<strong style="color:#d8ff90">${p.name}</strong>`);
    } else {
      metaParts.push(p.name);
    }
  });
  if (etaText) metaParts.push(html` · ${etaText}`);
  return html`
    <${Strip} label="phase" testId="strip-phase" meta=${metaParts}>
      ${list.map((p, i) => {
        const left = (i / total) * 100;
        const width = 100 / total;
        const color = PHASE_COLORS[p.status] ?? PHASE_COLORS.pending;
        return html`<div class="seg" style=${`left:${left}%;width:${width}%;background:${color}`}></div>`;
      })}
    <//>
  `;
}
