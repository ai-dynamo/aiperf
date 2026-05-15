// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Right-rail context cards for the job-detail page. Each card is a
 * compact panel-style block in the 280px sticky aside; the cards are
 * intentionally split as separate exports so ``pages/job-detail.js``
 * can decide which ones to render based on run state (e.g. SweepInfoCard
 * is gated on ``info.sweepName``).
 *
 * The shared ``RailCard`` wrapper renders the title bar + body. Inner
 * content is plain children — the cards know nothing about preact state.
 */

import { html } from 'htm/preact';

/** Reusable rail card chrome. */
export function RailCard({ title, testId, children }) {
  return html`
    <div class="rail-card panel" data-testid=${testId}>
      <div class="panel-title">${title}</div>
      ${children}
    </div>
  `;
}

/** A single key/value row in a rail card body. */
export function RailKv({ k, v, tone }) {
  const cls = 'rail-kv__v' + (tone ? ` rail-kv__v--${tone}` : '');
  return html`
    <div class="rail-kv">
      <span class="rail-kv__k">${k}</span>
      <span class=${cls}>${v ?? '—'}</span>
    </div>
  `;
}

/** Action row used in the Actions card. */
export function RailAction({ icon, label, onClick, href, testId, danger, target }) {
  const cls = 'rail-action' + (danger ? ' rail-action--danger' : '');
  if (href) {
    return html`
      <a class=${cls} href=${href} data-testid=${testId} target=${target} onClick=${onClick}>
        <span class="rail-action__gly">${icon}</span>
        <span>${label}</span>
      </a>
    `;
  }
  return html`
    <button class=${cls} type="button" onClick=${onClick} data-testid=${testId}>
      <span class="rail-action__gly">${icon}</span>
      <span>${label}</span>
    </button>
  `;
}
