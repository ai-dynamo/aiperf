// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Connection status bar (green dot = WS live, amber = connecting, red = down).
 * Shows phase count and record count at a glance.
 */

import { html } from 'htm/preact';
import { connection, phases, records } from '../lib/state.js';
import { fmtInt } from '../lib/format.js';

export function StatusBar() {
  const conn = connection.value;
  const phaseMap = phases.value;
  const rec = records.value;

  const phaseCount = Object.keys(phaseMap).length;
  const activePhase = Object.values(phaseMap).find(p => p.active);

  return html`
    <div class="status-bar" role="status">
      <div class="status-item">
        <span class=${'status-dot ' + conn}></span>
        <span>${conn === 'connected' ? 'Connected' : conn === 'connecting' ? 'Connecting...' : 'Disconnected'}</span>
      </div>
      <span class="status-sep">|</span>
      <div class="status-item">
        <span>Phases</span>
        <span class="status-val">${phaseCount}</span>
      </div>
      ${activePhase && html`
        <span class="status-sep">|</span>
        <div class="status-item">
          <span>Active</span>
          <span class="status-val">${activePhase.name}</span>
        </div>
      `}
      <span class="status-sep">|</span>
      <div class="status-item">
        <span>Records</span>
        <span class="status-val">${fmtInt(rec.successRecords + rec.errorRecords)}</span>
      </div>
      ${rec.complete && html`
        <span class="status-sep">|</span>
        <span class="text-accent">complete</span>
      `}
    </div>
  `;
}
