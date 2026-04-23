// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Worker roster table. Matches the columns from v1 but renders from the
 * live workers signal. Status CSS classes are whitelisted to prevent
 * arbitrary string values from being embedded as class tokens.
 */

import { html } from 'htm/preact';
import { workers } from '../lib/state.js';
import { fmtInt, fmtBytes, fmtPercent } from '../lib/format.js';

const KNOWN_STATUSES = ['healthy', 'high_load', 'error', 'idle', 'stale'];

function shortId(id) {
  if (!id) return '';
  const parts = id.split('-');
  return parts.length <= 2 ? id : parts.slice(-2).join('-');
}

function safeStatusClass(status) {
  return KNOWN_STATUSES.includes(status) ? status : 'idle';
}

function displayStatus(w) {
  const s = (w.status ?? 'idle').replace('_', ' ');
  if (w.startupState && w.startupState !== 'ready') {
    return `${s} (${String(w.startupState).replace(/_/g, ' ')})`;
  }
  return s;
}

export function WorkerTable() {
  const map = workers.value;
  const ids = Object.keys(map).sort();

  return html`
    <div class="card">
      <div class="card-title">Workers <span class="text-dim" style="margin-left: 6px; font-weight: 400">(${ids.length})</span></div>
      ${ids.length === 0
        ? html`<div class="empty">No worker reports yet.</div>`
        : html`
          <div style="overflow-x: auto">
            <table class="worker-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Status</th>
                  <th style="text-align: right">In-flight</th>
                  <th style="text-align: right">Completed</th>
                  <th style="text-align: right">Failed</th>
                  <th style="text-align: right">CPU</th>
                  <th style="text-align: right">Memory</th>
                </tr>
              </thead>
              <tbody>
                ${ids.map((id) => {
                  const w = map[id];
                  return html`
                    <tr key=${id}>
                      <td><span class="worker-id">${shortId(id)}</span></td>
                      <td><span class=${'worker-status ' + safeStatusClass(w.status)}>${displayStatus(w)}</span></td>
                      <td style="text-align: right">${fmtInt(w.inFlight ?? 0)}</td>
                      <td style="text-align: right">${fmtInt(w.completed ?? 0)}</td>
                      <td style="text-align: right">${fmtInt(w.failed ?? 0)}</td>
                      <td style="text-align: right">${w.cpu != null ? fmtPercent(w.cpu) : '---'}</td>
                      <td style="text-align: right">${fmtBytes(w.memory)}</td>
                    </tr>
                  `;
                })}
              </tbody>
            </table>
          </div>
        `
      }
    </div>
  `;
}
