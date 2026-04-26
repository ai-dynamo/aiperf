// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * NAMESPACE OVERVIEW — per-namespace dashboard mounted at ``/ns/:ns``.
 *
 * Same shape as the prior global Home but filtered to ``j.namespace === ns``.
 * Renders an empty-state with a single "Launch in <ns>" CTA when the
 * namespace has zero current and zero historical jobs.
 */

import { html } from 'htm/preact';
import { useMemo } from 'preact/hooks';
import { jobs, clusterInfo } from '../lib/state.js';
import { navigate } from '../lib/router.js';
import { fmtDuration, fmtInt, fmtNumber } from '../lib/format.js';

function phaseBucket(phase) {
  const p = (phase ?? '').toLowerCase();
  if (p === 'running' || p === 'initializing' || p === 'pending') return 'live';
  if (p === 'failed' || p === 'error')                              return 'fault';
  if (p === 'completed' || p === 'succeeded')                       return 'passed';
  return 'other';
}

function titleCase(s) {
  if (!s) return '—';
  const lower = String(s).toLowerCase();
  return lower.charAt(0).toUpperCase() + lower.slice(1);
}

function modelShort(model) {
  if (!model) return '';
  return String(model).split('/').pop();
}

function progressPct(j) {
  if (j.progressPct != null) return Math.max(0, Math.min(100, Number(j.progressPct)));
  if (j.requestsCompleted != null && j.requestsTotal) {
    return Math.max(0, Math.min(100, (j.requestsCompleted / j.requestsTotal) * 100));
  }
  return null;
}

function StatTile({ label, value, mod }) {
  return html`
    <div class=${'no-stat no-stat--' + mod}>
      <div class="no-stat-label">${label}</div>
      <div class="no-stat-val">${value}</div>
    </div>
  `;
}

function ActiveCard({ job, ns }) {
  const pct = progressPct(job);
  const elapsed = job.startTime ? (Date.now() - new Date(job.startTime).getTime()) / 1000 : null;
  const href = `/ns/${encodeURIComponent(ns)}/run/${encodeURIComponent(job.name)}`;
  return html`
    <div
      class="no-active-card"
      data-testid=${'no-active-' + ns + '-' + job.name}
      onclick=${() => navigate(href)}
    >
      <div class="no-active-card-head">
        <div>
          <div class="no-active-card-name">${job.name}</div>
          <div class="no-active-card-ns">${modelShort(job.model) || 'no model'}</div>
        </div>
        <span class="chip chip--info">${titleCase(job.phase) || 'Running'}</span>
      </div>
      ${pct != null && html`
        <div class="no-active-card-track">
          <div class="no-active-card-fill" style=${'width:' + pct + '%'}></div>
        </div>
      `}
      <div class="no-active-card-stats">
        <div><div class="no-active-card-stat-lab">Throughput</div><div class="no-active-card-stat-val">${job.throughputRps != null ? fmtNumber(job.throughputRps, 1) : '—'}</div></div>
        <div><div class="no-active-card-stat-lab">Latency p99</div><div class="no-active-card-stat-val">${job.latencyP99Ms != null ? fmtInt(job.latencyP99Ms) + ' ms' : '—'}</div></div>
        <div><div class="no-active-card-stat-lab">Elapsed</div><div class="no-active-card-stat-val">${elapsed != null ? fmtDuration(elapsed) : '—'}</div></div>
      </div>
    </div>
  `;
}

function RecentRow({ job, ns }) {
  const href = `/ns/${encodeURIComponent(ns)}/run/${encodeURIComponent(job.name)}`;
  return html`
    <tr
      class=${'no-row no-row--' + phaseBucket(job.phase)}
      data-testid=${'no-row-' + ns + '-' + job.name}
      onclick=${() => navigate(href)}
    >
      <td class="no-row-name">${job.name}</td>
      <td>${modelShort(job.model)}</td>
      <td>${titleCase(job.phase)}</td>
      <td>${job.throughputRps != null ? fmtNumber(job.throughputRps, 1) : '—'}</td>
      <td>${job.latencyP99Ms != null ? fmtInt(job.latencyP99Ms) + ' ms' : '—'}</td>
    </tr>
  `;
}

export function NamespaceOverview({ ns }) {
  const all = jobs.value ?? [];
  const list = useMemo(() => all.filter(j => (j.namespace || 'default') === ns), [all, ns]);

  const counts = useMemo(() => {
    const c = { live: 0, passed: 0, fault: 0, total: list.length };
    for (const j of list) {
      const b = phaseBucket(j.phase);
      c[b] = (c[b] || 0) + 1;
    }
    return c;
  }, [list]);

  if (list.length === 0) {
    return html`
      <div class="page-namespace-overview" data-testid="page-namespace-overview">
        <div class="no-empty" data-testid="no-empty">
          <h1 class="no-empty-title">No runs yet in <code>${ns}</code></h1>
          <p class="no-empty-sub">Launch your first benchmark in this namespace.</p>
          <button
            class="btn btn--primary no-empty-cta"
            data-testid="no-empty-launch-cta"
            onclick=${() => navigate('/ns/' + encodeURIComponent(ns) + '/launch')}
          >Launch in ${ns}</button>
        </div>
      </div>
    `;
  }

  const active = list.filter(j => phaseBucket(j.phase) === 'live');
  const recent = [...list]
    .sort((a, b) => Date.parse(b.lastUpdate ?? b.startTime ?? 0) - Date.parse(a.lastUpdate ?? a.startTime ?? 0))
    .slice(0, 25);

  const gpus = clusterInfo.value?.gpus ?? null;

  return html`
    <div class="page-namespace-overview" data-testid="page-namespace-overview">
      <div class="no-stats" data-testid="no-stats">
        <${StatTile} label="Running" value=${fmtInt(counts.live)} mod="live" />
        <${StatTile} label="Passed"  value=${fmtInt(counts.passed)} mod="passed" />
        <${StatTile} label="Failed"  value=${fmtInt(counts.fault)} mod="fault" />
        <${StatTile} label="Total"   value=${fmtInt(counts.total)} mod="total" />
        <${StatTile} label="GPUs"    value=${gpus != null ? fmtInt(gpus) : '—'} mod="gpus" />
      </div>
      ${active.length > 0 && html`
        <div class="no-active">
          ${active.map(j => html`<${ActiveCard} key=${j.name} job=${j} ns=${ns} />`)}
        </div>
      `}
      <table class="no-recent">
        <thead><tr><th>Name</th><th>Model</th><th>Phase</th><th>RPS</th><th>p99</th></tr></thead>
        <tbody>${recent.map(j => html`<${RecentRow} key=${j.name} job=${j} ns=${ns} />`)}</tbody>
      </table>
    </div>
  `;
}
