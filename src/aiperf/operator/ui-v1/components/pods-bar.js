// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';

/**
 * Dot color for a pod based on phase and ready state.
 * @param {{ phase: string, ready: boolean }} pod
 * @returns {string} CSS class
 */
function podDotClass(pod) {
  const phase = (pod.phase ?? '').toLowerCase();
  if (phase === 'failed' || phase === 'error') return 'pod-dot--failed';
  if (pod.ready) return 'pod-dot--ready';
  if (phase === 'running') return 'pod-dot--not-ready';
  return 'pod-dot--pending';
}

/**
 * Defensive cap on per-pod rendering. Sweep jobs at very high concurrency can
 * spawn 200+ pods; the table caps visible rows and surfaces the overflow as a
 * trailing aggregate row. The summary still reflects all pods so the
 * ready/restarts counts stay correct.
 */
const MAX_VISIBLE_PODS = 100;

/**
 * Pods status table — colored status dot, name, phase, ready, restarts.
 * @param {{ pods: Array<{name: string, phase: string, ready: boolean, restarts: number}> }} props
 */
export function PodsBar({ pods }) {
  if (!pods || pods.length === 0) {
    return html`<div class="pods-bar pods-bar--empty">No pods</div>`;
  }

  const readyCount = pods.filter((p) => p.ready).length;
  const totalRestarts = pods.reduce((sum, p) => sum + (p.restarts ?? 0), 0);

  const overflowCount = Math.max(0, pods.length - MAX_VISIBLE_PODS);
  const visiblePods = overflowCount > 0 ? pods.slice(0, MAX_VISIBLE_PODS) : pods;

  return html`
    <div class="pods-bar">
      <table class="pods-table" data-testid="pods-table">
        <thead>
          <tr>
            <th class="pods-table-status" aria-label="Status"></th>
            <th>Pod</th>
            <th>Phase</th>
            <th class="pods-table-num">Ready</th>
            <th class="pods-table-num">Restarts</th>
          </tr>
        </thead>
        <tbody>
          ${visiblePods.map((pod) => {
            const phase = (pod.phase ?? 'unknown').toLowerCase();
            const restarts = pod.restarts ?? 0;
            return html`
              <tr key=${pod.name}>
                <td class="pods-table-status">
                  <span
                    class=${'pod-dot ' + podDotClass(pod)}
                    title=${(pod.phase ?? 'unknown') + (pod.ready ? ' · ready' : '')}
                  />
                </td>
                <td class="pods-table-name" title=${pod.name}>${pod.name}</td>
                <td class="pods-table-phase">${phase}</td>
                <td class="pods-table-num">${pod.ready ? 'yes' : 'no'}</td>
                <td class=${'pods-table-num' + (restarts > 0 ? ' pods-table-restarts' : '')}>
                  ${restarts}
                </td>
              </tr>
            `;
          })}
          ${overflowCount > 0 && html`
            <tr class="pods-table-overflow">
              <td class="pods-table-status"></td>
              <td colspan="4">+${overflowCount} more pods (showing first ${MAX_VISIBLE_PODS})</td>
            </tr>
          `}
        </tbody>
      </table>
      <div class="pods-bar-summary">
        <span class="pods-bar-ready">${readyCount}/${pods.length} ready</span>
        ${totalRestarts > 0 && html`
          <span class="pods-bar-restarts">${totalRestarts} restart${totalRestarts !== 1 ? 's' : ''}</span>
        `}
      </div>
    </div>
  `;
}
