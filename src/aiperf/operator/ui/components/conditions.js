// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { visibleConditionBadgeSummary } from './conditions-helpers.js';

/**
 * Defensive cap on rendered condition badges. K8s conditions for AIPerfJob
 * top out at ~10 in normal operation; a malformed status block has no upper
 * bound, so cap rendering to keep DOM bounded and prevent runaway layouts.
 */
const MAX_VISIBLE_CONDITIONS = 50;

/**
 * Row of condition status badges.
 * @param {{ conditions: Array<{type: string, status: string, reason?: string, message?: string}> }} props
 */
export function Conditions({ conditions }) {
  if (!conditions || conditions.length === 0) {
    return html`<div class="conditions conditions--empty">No conditions</div>`;
  }

  const { badges: visible, overflow } = visibleConditionBadgeSummary(conditions, MAX_VISIBLE_CONDITIONS);
  if (visible.length === 0) return null;

  return html`
    <div
      class="conditions"
      role="list"
      aria-label="Conditions"
      style="display:flex;flex-wrap:wrap;gap:var(--space-1,4px);align-items:center"
    >
      ${visible.map((cond) => {
        const title = cond.message
          ? `${cond.type}: ${cond.message}`
          : cond.type;

        return html`
          <span
            key=${cond.type}
            class=${'condition-badge ' + cond.className}
            title=${title}
            role="listitem"
            style="word-break:break-word;max-width:100%"
          >
            ${cond.label}
          </span>
        `;
      })}
      ${overflow > 0 && html`
        <span
          class="condition-badge condition-badge--unknown"
          role="listitem"
          title=${'+' + overflow + ' more conditions hidden (showing first ' + MAX_VISIBLE_CONDITIONS + ')'}
          style="word-break:break-word;max-width:100%"
        >
          +${overflow} more
        </span>
      `}
    </div>
  `;
}
