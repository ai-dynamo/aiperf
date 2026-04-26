// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * NAMESPACE PICKER — the cross-namespace landing surface mounted at ``/``.
 *
 * One tile per namespace observed in ``jobs.value`` (no separate API
 * call — we group the existing job list by ``j.namespace``). Each tile
 * surfaces "is anything broken / live here?" at a glance: phase chips
 * for live counts, last-activity timestamp, left-edge state tint.
 */

import { html } from 'htm/preact';
import { useMemo, useState } from 'preact/hooks';
import { jobs } from '../lib/state.js';
import { navigate } from '../lib/router.js';
import { setLastNamespace } from '../lib/ns-prefs.js';

const FAILED_RECENT_WINDOW_MS = 24 * 60 * 60 * 1000;

function bucketForJob(j) {
  const p = (j.phase ?? '').toLowerCase();
  if (p === 'running' || p === 'initializing' || p === 'pending') return 'live';
  if (p === 'failed' || p === 'error')                              return 'fault';
  if (p === 'completed' || p === 'succeeded')                       return 'passed';
  return 'other';
}

function fmtAgo(ts) {
  if (!ts) return '—';
  const s = Math.max(0, Math.round((Date.now() - ts) / 1000));
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 48) return `${h}h ago`;
  const d = Math.round(h / 24);
  return `${d}d ago`;
}

function aggregate(nsJobs) {
  let running = 0, failedRecent = 0, completed = 0;
  const total = nsJobs.length;
  let lastActivity = 0;
  const now = Date.now();
  for (const j of nsJobs) {
    const b = bucketForJob(j);
    if (b === 'live') running += 1;
    if (b === 'passed') completed += 1;
    const ts = (j.lastUpdate ?? j.startTime) ? Date.parse(j.lastUpdate ?? j.startTime) : 0;
    if (ts > lastActivity) lastActivity = ts;
    if (b === 'fault' && (now - ts) <= FAILED_RECENT_WINDOW_MS) failedRecent += 1;
  }
  let tint = 'quiet';
  if (running > 0) tint = 'live';
  else if (failedRecent > 0) tint = 'fault';
  return { running, failedRecent, completed, total, lastActivity, tint };
}

function NamespaceTile({ name, agg, onPick }) {
  return html`
    <div
      class=${'np-tile np-tile--' + agg.tint}
      data-testid=${'np-tile-' + name}
      onclick=${() => onPick(name)}
    >
      <div class="np-tile-name">${name}</div>
      <div class="np-tile-summary">${agg.running} active · ${agg.total} total</div>
      <div class="np-tile-chips">
        ${agg.running > 0 && html`<span class="np-chip np-chip-running">Running ${agg.running}</span>`}
        ${agg.failedRecent > 0 && html`<span class="np-chip np-chip-failed">Failed ${agg.failedRecent}</span>`}
        ${agg.completed > 0 && html`<span class="np-chip np-chip-completed">Completed ${agg.completed}</span>`}
      </div>
      <div class="np-tile-time">${fmtAgo(agg.lastActivity)}</div>
    </div>
  `;
}

export function NamespacePicker() {
  const [query, setQuery] = useState('');
  const list = jobs.value ?? [];

  const tiles = useMemo(() => {
    const groups = new Map();
    for (const j of list) {
      const ns = j.namespace || 'default';
      if (!groups.has(ns)) groups.set(ns, []);
      groups.get(ns).push(j);
    }
    const out = [];
    for (const [name, nsJobs] of groups) {
      out.push({ name, agg: aggregate(nsJobs) });
    }
    out.sort((a, b) => b.agg.lastActivity - a.agg.lastActivity);
    return out;
  }, [list]);

  const filtered = query
    ? tiles.filter(t => t.name.toLowerCase().includes(query.toLowerCase()))
    : tiles;

  function pick(name) {
    setLastNamespace(name);
    navigate('/ns/' + encodeURIComponent(name));
  }

  return html`
    <div class="page-namespace-picker" data-testid="page-namespace-picker">
      <div class="np-header">
        <h1 class="np-title">Pick a namespace</h1>
        <input
          class="np-search"
          data-testid="np-search"
          placeholder="filter namespaces…"
          value=${query}
          oninput=${(e) => setQuery(e.target.value)}
        />
      </div>
      ${tiles.length === 0 && html`
        <div class="np-empty" data-testid="np-empty">
          <p>No AIPerfJob runs visible in any namespace yet.</p>
          <p>If you expected to see runs here, check the operator's RBAC against your kubeconfig context.</p>
        </div>
      `}
      <div class="np-grid">
        ${filtered.map(t => html`
          <${NamespaceTile} key=${t.name} name=${t.name} agg=${t.agg} onPick=${pick} />
        `)}
      </div>
    </div>
  `;
}
