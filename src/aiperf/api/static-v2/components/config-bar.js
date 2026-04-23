// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Top row showing current benchmark config: models, endpoint, per-phase
 * controls. Matches the label set v1 renderConfig produced after the
 * v2 config-shape fix (models.items[*].name, phases[*].{type,...}).
 */

import { html } from 'htm/preact';
import { config } from '../lib/state.js';
import { fmtInt, fmtDuration } from '../lib/format.js';

function buildItems(cfg) {
  const items = [];
  const add = (label, value) => items.push({ label, value: String(value) });

  const modelNames = (cfg.models?.items || []).map(m => m?.name).filter(Boolean);
  if (modelNames.length) add('Model', modelNames.join(', '));

  const ep = cfg.endpoint || {};
  if (ep.type) add('Endpoint', ep.type + (ep.streaming ? ' (streaming)' : ''));
  if (ep.urls?.length) {
    add('URL', ep.urls.length === 1 ? ep.urls[0] : `${ep.urls.length} URLs`);
  }

  const phaseEntries = Object.entries(cfg.phases || {});
  const showPrefix = phaseEntries.length > 1;
  for (const [name, phase] of phaseEntries) {
    if (!phase) continue;
    const prefix = showPrefix ? `${name} ` : '';
    if (phase.type) add(`${prefix}Type`, phase.type);
    if (phase.concurrency != null) add(`${prefix}Concurrency`, phase.concurrency);
    if (phase.prefill_concurrency != null) add(`${prefix}Prefill`, phase.prefill_concurrency);
    if (phase.rate != null) add(`${prefix}Rate`, `${phase.rate} QPS`);
    if (phase.users != null) add(`${prefix}Users`, phase.users);
    if (phase.requests != null) add(`${prefix}Requests`, fmtInt(phase.requests));
    if (phase.duration != null) {
      const secs = typeof phase.duration === 'number' ? phase.duration : null;
      add(`${prefix}Duration`, secs != null ? fmtDuration(secs) : String(phase.duration));
    }
    if (phase.sessions != null) add(`${prefix}Sessions`, fmtInt(phase.sessions));
  }
  return items;
}

export function ConfigBar() {
  const cfg = config.value;
  if (!cfg) return null;

  const items = buildItems(cfg);
  if (!items.length) return null;

  return html`
    <div class="config-bar visible" id="config-bar">
      ${items.map((item, i) => html`
        <div class="config-item" key=${item.label}>
          <span class="config-label">${item.label}</span>
          <span class="config-value">${item.value}</span>
          ${i < items.length - 1 && html`<span class="config-sep"></span>`}
        </div>
      `)}
    </div>
  `;
}
