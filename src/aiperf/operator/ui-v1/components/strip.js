// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Canonical thin status strip. Three columns:
 *   [LABEL] [BAR (children)] [META]
 *
 * Single-row, ~28px tall. Used by PhaseStrip / RecordsStrip / PodsStrip.
 */

import { html } from 'htm/preact';

export function Strip({ label, meta, onBarClick, children, testId }) {
  return html`
    <div class="strip" data-testid=${testId}>
      <span class="strip-label">${label}</span>
      <div class=${'strip-bar' + (onBarClick ? ' strip-bar--clickable' : '')}
           onClick=${onBarClick}
           role=${onBarClick ? 'button' : undefined}
           tabindex=${onBarClick ? 0 : undefined}>
        ${children}
      </div>
      ${meta != null && html`<span class="strip-meta">${meta}</span>`}
    </div>
  `;
}
