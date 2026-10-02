// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Pods tab for DiagnosticsPanel — ports the full per-pod table from
 * ``components/pods-bar.js`` expanded mode (status dot, name, phase,
 * ready, restarts) and adds click-to-sort on every column. The slim /
 * collapsed dot strip is intentionally dropped: the tab itself is the
 * "expanded" state. Pods are passed in by the parent page; this
 * component does not fetch.
 */

import { html } from 'htm/preact';
import { useMemo, useState } from 'preact/hooks';

/**
 * Defensive cap on per-pod rendering. Sweep jobs at very high concurrency can
 * spawn 200+ pods; the table caps visible rows and surfaces the overflow as a
 * trailing aggregate row. The summary still reflects all pods so the
 * ready/restarts counts stay correct.
 */
const MAX_VISIBLE_PODS = 100;

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

function compareBy(col, a, b) {
  switch (col) {
    case 'name':
      return (a.name ?? '').localeCompare(b.name ?? '');
    case 'phase':
      return (a.phase ?? '').localeCompare(b.phase ?? '');
    case 'ready':
      return (a.ready === b.ready) ? 0 : (a.ready ? -1 : 1);
    case 'restarts':
      return (a.restarts ?? 0) - (b.restarts ?? 0);
    default:
      return 0;
  }
}

export function PodsTab({ pods }) {
  const [sortCol, setSortCol] = useState('name');
  const [sortDir, setSortDir] = useState('asc');

  const sorted = useMemo(() => {
    if (!pods || pods.length === 0) return [];
    const out = [...pods];
    out.sort((a, b) => {
      const cmp = compareBy(sortCol, a, b);
      return sortDir === 'asc' ? cmp : -cmp;
    });
    return out;
  }, [pods, sortCol, sortDir]);

  if (!pods || pods.length === 0) {
    return html`<div class="diag-tab-body pods-bar pods-bar--empty">No pods</div>`;
  }

  const readyCount = pods.filter((p) => p.ready).length;
  const totalRestarts = pods.reduce((sum, p) => sum + (p.restarts ?? 0), 0);

  const overflowCount = Math.max(0, sorted.length - MAX_VISIBLE_PODS);
  const visiblePods = overflowCount > 0 ? sorted.slice(0, MAX_VISIBLE_PODS) : sorted;

  const onHeaderClick = (col) => {
    if (sortCol === col) {
      setSortDir(d => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortCol(col);
      setSortDir('asc');
    }
  };

  const sortIndicator = (col) => {
    if (sortCol !== col) return '';
    return sortDir === 'asc' ? ' ▲' : ' ▼';
  };

  const headerCell = (col, label, extraClass) => html`
    <th
      class=${(extraClass ?? '') + ' pods-table-sortable' + (sortCol === col ? ' pods-table-sorted' : '')}
      role="button"
      tabindex="0"
      onkeydown=${(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onHeaderClick(col);
        }
      }}
      onclick=${() => onHeaderClick(col)}
      aria-sort=${sortCol === col ? (sortDir === 'asc' ? 'ascending' : 'descending') : 'none'}
      style="cursor:pointer; user-select:none"
    >${label}${sortIndicator(col)}</th>
  `;

  return html`
    <div class="diag-tab-body pods-bar">
      <table class="pods-table" data-testid="pods-table">
        <thead>
          <tr>
            <th class="pods-table-status" aria-label="Status"></th>
            ${headerCell('name', 'Pod')}
            ${headerCell('phase', 'Phase')}
            ${headerCell('ready', 'Ready', 'pods-table-num')}
            ${headerCell('restarts', 'Restarts', 'pods-table-num')}
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
      <div class="pods-bar-meta">
        <div class="pods-bar-summary">
          <span class="pods-bar-ready">${readyCount}/${pods.length} ready</span>
          ${totalRestarts > 0 && html`
            <span class="pods-bar-restarts">${totalRestarts} restart${totalRestarts !== 1 ? 's' : ''}</span>
          `}
        </div>
      </div>
    </div>
  `;
}
