// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Worker-group roster table. One row per WorkerGroupManager.
 *
 * When there is exactly one group, the row is expandable and reveals a
 * nested per-worker table (the dropdown). With multiple groups, only the
 * group-level row is shown to keep the dashboard scannable.
 */

import { html } from 'htm/preact';
import { useState } from 'preact/hooks';
import { workerGroups } from '../lib/state.js';
import { fmtInt, fmtBytes, fmtPercent } from '../lib/format.js';

const KNOWN_STATUSES = ['healthy', 'high_load', 'error', 'idle', 'stale'];

function shortId(id) {
  if (!id) return '';
  const parts = id.split('-');
  return parts.length <= 2 ? id : parts.slice(-2).join('-');
}

function safeRecord(value) {
  return value && typeof value === 'object' ? value : {};
}

function safeStatusClass(status) {
  const normalized = String(status ?? 'idle');
  return KNOWN_STATUSES.includes(normalized) ? normalized : 'idle';
}

function displayStatus(g) {
  const record = safeRecord(g);
  const s = String(record.status ?? 'idle').replace(/_/g, ' ');
  if (record.startupState && record.startupState !== 'ready') {
    return `${s} (${String(record.startupState).replace(/_/g, ' ')})`;
  }
  return s;
}

function childRows(children) {
  const childMap = safeRecord(children);
  const ids = Object.keys(childMap).sort();
  if (ids.length === 0) {
    return [html`<tr key="empty-children"><td colspan="7" class="empty">No worker children yet.</td></tr>`];
  }
  return ids.map((id) => {
    const w = safeRecord(childMap[id]);
    return html`
      <tr key=${'child-' + id} class="worker-child-row">
        <td><span class="worker-id" style="padding-left: 18px">↳ ${shortId(id)}</span></td>
        <td><span class=${'worker-status ' + safeStatusClass(w.status)}>${displayStatus(w)}</span></td>
        <td style="text-align: right">${fmtInt(w.inFlight ?? 0)}</td>
        <td style="text-align: right">${fmtInt(w.completed ?? 0)}</td>
        <td style="text-align: right">${fmtInt(w.failed ?? 0)}</td>
        <td style="text-align: right">${w.cpu != null ? fmtPercent(w.cpu) : '---'}</td>
        <td style="text-align: right">${fmtBytes(w.memory)}</td>
      </tr>
    `;
  });
}

export function WorkerTable() {
  const map = safeRecord(workerGroups.value);
  const groupIds = Object.keys(map).sort();
  const singleGroup = groupIds.length === 1;
  const [expanded, setExpanded] = useState(true);  // default open when single group

  const rows = groupIds.flatMap((gid) => {
    const g = safeRecord(map[gid]);
    const children = safeRecord(g.workers);
    const childCount = Object.keys(children).length;
    const declaredWorkers = Number(g.declaredWorkers) > 0 ? g.declaredWorkers : childCount;
    const groupRow = html`
      <tr key=${'group-' + gid} class="worker-group-row">
        <td><span class="worker-id">${shortId(gid)} <span class="text-dim">(${g.readyWorkers ?? 0}/${declaredWorkers} ready)</span></span></td>
        <td><span class=${'worker-status ' + safeStatusClass(g.status)}>${displayStatus(g)}</span></td>
        <td style="text-align: right">${fmtInt(g.inFlight ?? 0)}</td>
        <td style="text-align: right">${fmtInt(g.completed ?? 0)}</td>
        <td style="text-align: right">${fmtInt(g.failed ?? 0)}</td>
        <td style="text-align: right">${g.cpu != null ? fmtPercent(g.cpu) : '---'}</td>
        <td style="text-align: right">${fmtBytes(g.memory)}</td>
      </tr>
    `;
    if (singleGroup && expanded) {
      return [groupRow, ...childRows(children)];
    }
    return [groupRow];
  });

  return html`
    <div class="card">
      <div class="card-title">Worker Groups <span class="text-dim" style="margin-left: 6px; font-weight: 400">(${groupIds.length})</span></div>
      ${groupIds.length === 0
        ? html`<div class="empty">No worker-group reports yet.</div>`
        : html`
          <div style="overflow-x: auto">
            <table class="worker-table">
              <thead>
                <tr>
                  <th>${singleGroup ? html`<span class="group-toggle" onClick=${() => setExpanded(!expanded)}>${expanded ? '▾' : '▸'}</span> ` : ''}Group</th>
                  <th>Status</th>
                  <th style="text-align: right">In-flight</th>
                  <th style="text-align: right">Completed</th>
                  <th style="text-align: right">Failed</th>
                  <th style="text-align: right">CPU</th>
                  <th style="text-align: right">Memory</th>
                </tr>
              </thead>
              <tbody>
                ${rows}
              </tbody>
            </table>
          </div>
        `
      }
    </div>
  `;
}
