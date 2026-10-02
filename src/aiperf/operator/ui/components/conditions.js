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
 *
 * Three empty-ish outcomes, deliberately distinct because a reader cannot
 * tell them apart from a blank pane:
 *   - ``null``/``undefined`` conditions: the caller has no status block, so
 *     we do not know. Saying "No conditions" would assert a fact about the CR.
 *   - empty array: the CR reports no conditions.
 *   - non-empty but every badge suppressed: every condition is healthy, which
 *     is what ``visibleConditionBadgeSummary`` filters out by design. This
 *     used to render nothing at all while the diagnostics tab header counted
 *     "conditions 1" right above it.
 *
 * @param {{ conditions: Array<{type: string, status: string, reason?: string, message?: string}> | null }} props
 */
export function Conditions({ conditions }) {
  if (conditions == null) {
    return html`<div class="conditions conditions--empty" data-testid="conditions-unknown">Conditions not reported</div>`;
  }
  if (conditions.length === 0) {
    return html`<div class="conditions conditions--empty">No conditions</div>`;
  }

  const { badges: visible, overflow } = visibleConditionBadgeSummary(conditions, MAX_VISIBLE_CONDITIONS);
  if (visible.length === 0) {
    const n = conditions.length;
    return html`
      <div class="conditions conditions--empty" data-testid="conditions-all-ok">
        All ${n} condition${n === 1 ? '' : 's'} healthy
      </div>
    `;
  }

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
