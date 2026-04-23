// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * LEFT RAIL — the persistent fleet "rack" on the left of the Flight Deck.
 *
 * Shows every run currently tracked, partitioned into three stacks:
 *   · LIVE        — Running / Initializing / Pending
 *   · PASSED      — Completed / Succeeded
 *   · FAULT       — Failed / Error   (collapsed by default when empty)
 *
 * Each row carries a pilot-light indicator, callsign (ns / name), and a trio
 * of mini telemetry cells (R/S, P99, progress%). Clicking navigates the main
 * viewport to the run detail — the rail itself never unmounts, so orientation
 * is preserved.
 *
 * A filter search box at the top narrows across all three stacks. Badge counts
 * always show unfiltered totals so the user sees fleet-wide state.
 */

import { html } from 'htm/preact';
import { useState } from 'preact/hooks';
import { jobs } from '../lib/state.js';
import { navigate } from '../lib/router.js';
import { fmtNumber, fmtInt } from '../lib/format.js';

function phaseBucket(phase) {
  const p = (phase ?? '').toLowerCase();
  if (p === 'running' || p === 'initializing' || p === 'pending') return 'live';
  if (p === 'failed'  || p === 'error')                            return 'fault';
  if (p === 'completed' || p === 'succeeded')                      return 'passed';
  return 'other';
}

function matchQuery(job, q) {
  if (!q) return true;
  const hay = [job.name, job.namespace, job.model, job.endpoint]
    .filter(Boolean).join(' ').toLowerCase();
  return hay.includes(q.toLowerCase());
}

function Row({ job, active, onClick }) {
  const bucket = phaseBucket(job.phase);
  const rps = job.throughputRps;
  const p99 = job.latencyP99Ms;
  const pct = job.progressPercent;
  return html`
    <button
      class=${'rail-row rail-row--' + bucket + (active ? ' is-active' : '')}
      onclick=${onClick}
      title=${`${job.namespace}/${job.name}`}
      data-testid=${'rail-row-' + job.namespace + '-' + job.name}
    >
      <span class=${'rail-row-dot rail-row-dot--' + bucket}></span>
      <span class="rail-row-body">
        <span class="rail-row-name">${job.name}</span>
        <span class="rail-row-meta">
          <span class="rail-row-ns">${job.namespace}</span>
          ${job.model && html`<span class="rail-row-model">${job.model.split('/').pop()}</span>`}
        </span>
      </span>
      <span class="rail-row-stats">
        <span class="rail-row-stat">
          <span class="rail-row-stat-label">R/S</span>
          <span class="rail-row-stat-val">${rps != null ? fmtNumber(rps, 0) : '—'}</span>
        </span>
        <span class="rail-row-stat">
          <span class="rail-row-stat-label">P99</span>
          <span class="rail-row-stat-val">${p99 != null ? fmtInt(p99) : '—'}</span>
        </span>
        ${bucket === 'live' && pct != null && html`
          <span class="rail-row-stat">
            <span class="rail-row-stat-label">%</span>
            <span class="rail-row-stat-val is-amber">${Math.round(pct)}</span>
          </span>
        `}
      </span>
    </button>
  `;
}

function Group({ title, count, tone, children }) {
  return html`
    <div class=${'rail-group rail-group--' + tone}>
      <div class="rail-group-head">
        <span class=${'rail-group-light rail-group-light--' + tone}></span>
        <span class="rail-group-title">${title}</span>
        <span class="rail-group-count">${count}</span>
      </div>
      <div class="rail-group-body">
        ${children}
      </div>
    </div>
  `;
}

export function LeftRail({ viewKind, runParams, open, onToggle }) {
  const [q, setQ] = useState('');
  const list = jobs.value ?? [];

  const buckets = { live: [], passed: [], fault: [] };
  for (const j of list) {
    const b = phaseBucket(j.phase);
    if (buckets[b]) buckets[b].push(j);
  }
  // Sort: live by progress desc, passed by throughput desc, fault by age desc.
  buckets.live.sort((a, b) => (b.progressPercent ?? 0) - (a.progressPercent ?? 0));
  buckets.passed.sort((a, b) => (b.throughputRps ?? 0) - (a.throughputRps ?? 0));
  buckets.fault.sort((a, b) => new Date(b.created ?? 0) - new Date(a.created ?? 0));

  const activeKey = runParams ? `${runParams.ns}/${runParams.name}` : null;

  const filtLive   = buckets.live.filter(j => matchQuery(j, q));
  const filtPassed = buckets.passed.filter(j => matchQuery(j, q)).slice(0, 10);
  const filtFault  = buckets.fault.filter(j => matchQuery(j, q));

  const onPick = (j) => navigate('/run/' + encodeURIComponent(j.namespace ?? 'default') + '/' + encodeURIComponent(j.name ?? ''));

  return html`
    <aside class="rail-left" aria-label="Fleet rail" data-testid="left-rail" data-open=${open}>
      <div class="rail-left-head">
        <div class="rail-left-title">
          <span class="rail-left-caret">▸</span>
          FLEET
        </div>
        <button
          class="rail-left-collapse"
          onclick=${onToggle}
          title="Collapse rail (Ctrl+B)"
          aria-label="Collapse fleet rail"
        ><i class="ph ph-caret-double-left"></i></button>
      </div>

      <div class="rail-left-search">
        <i class="ph ph-magnifying-glass"></i>
        <input
          type="text"
          placeholder="filter runs…"
          value=${q}
          oninput=${e => setQ(e.target.value)}
          data-testid="rail-filter"
        />
        ${q && html`
          <button class="rail-left-clear" onclick=${() => setQ('')} aria-label="Clear filter">
            <i class="ph ph-x"></i>
          </button>
        `}
      </div>

      <div class="rail-left-body">
        ${filtLive.length > 0 && html`
          <${Group} title="LIVE" count=${buckets.live.length} tone="live">
            ${filtLive.map(j => html`
              <${Row}
                key=${j.namespace + '/' + j.name}
                job=${j}
                active=${activeKey === `${j.namespace}/${j.name}`}
                onClick=${() => onPick(j)}
              />
            `)}
          </${Group}>
        `}

        ${filtPassed.length > 0 && html`
          <${Group} title="PASSED" count=${buckets.passed.length} tone="passed">
            ${filtPassed.map(j => html`
              <${Row}
                key=${j.namespace + '/' + j.name}
                job=${j}
                active=${activeKey === `${j.namespace}/${j.name}`}
                onClick=${() => onPick(j)}
              />
            `)}
            ${buckets.passed.length > 10 && html`
              <button class="rail-see-all" onclick=${() => navigate('/fleet')}>
                <span>+ ${buckets.passed.length - 10} more completed</span>
                <i class="ph ph-arrow-right"></i>
              </button>
            `}
          </${Group}>
        `}

        ${filtFault.length > 0 && html`
          <${Group} title="FAULT" count=${buckets.fault.length} tone="fault">
            ${filtFault.map(j => html`
              <${Row}
                key=${j.namespace + '/' + j.name}
                job=${j}
                active=${activeKey === `${j.namespace}/${j.name}`}
                onClick=${() => onPick(j)}
              />
            `)}
          </${Group}>
        `}

        ${list.length === 0 && html`
          <div class="rail-left-empty">
            <div class="rail-left-empty-glyph">
              <span class="ph ph-crosshair"></span>
            </div>
            <div class="rail-left-empty-title">NO RUNS TRACKED</div>
            <div class="rail-left-empty-body">
              Launch one with <code>aiperf kube run</code>
            </div>
          </div>
        `}

        ${list.length > 0 && filtLive.length + filtPassed.length + filtFault.length === 0 && html`
          <div class="rail-left-empty">
            <div class="rail-left-empty-title">NO MATCH</div>
            <div class="rail-left-empty-body">filter "${q}" matched nothing</div>
          </div>
        `}
      </div>

      <div class="rail-left-foot">
        <button
          class=${'rail-foot-tab' + (viewKind === 'overview' ? ' is-active' : '')}
          onclick=${() => navigate('/overview')}
          data-testid="rail-foot-overview"
        >
          <i class="ph ph-crosshair"></i>
          OVERVIEW
        </button>
        <button
          class=${'rail-foot-tab' + (viewKind === 'fleet' ? ' is-active' : '')}
          onclick=${() => navigate('/fleet')}
          data-testid="rail-foot-fleet"
        >
          <i class="ph ph-list-numbers"></i>
          ALL
        </button>
      </div>
    </aside>
  `;
}
