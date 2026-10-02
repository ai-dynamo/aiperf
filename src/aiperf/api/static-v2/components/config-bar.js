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

function sanitizeDisplayUrl(rawUrl) {
  try {
    const url = new URL(rawUrl);
    url.username = '';
    url.password = '';
    for (const key of Array.from(url.searchParams.keys())) {
      if (['api_key', 'key', 'token', 'access_token'].includes(key.toLowerCase())) {
        url.searchParams.set(key, '<redacted>');
      }
    }
    return url.toString();
  } catch (_) {
    return String(rawUrl).replace(/\/\/[^/@\s]+@/, '//');
  }
}

function buildItems(cfg) {
  const items = [];
  const add = (label, value) => items.push({ label, value: String(value) });

  const modelNames = (cfg.models?.items || []).map(m => m?.name).filter(Boolean);
  if (modelNames.length) add('Model', modelNames.join(', '));

  const ep = cfg.endpoint || {};
  if (ep.type) add('Endpoint', ep.type + (ep.streaming ? ' (streaming)' : ''));
  if (ep.urls?.length) {
    const urls = ep.urls.map(sanitizeDisplayUrl);
    add('URL', urls.length === 1 ? urls[0] : `${urls.length} URLs`);
  }

  // phases is a list of named phase configs (post-2026-04 list-with-name shape).
  const phaseList = cfg.phases || [];
  const showPrefix = phaseList.length > 1;
  for (const phase of phaseList) {
    if (!phase) continue;
    const prefix = showPrefix ? `${phase.name ?? ''} ` : '';
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
