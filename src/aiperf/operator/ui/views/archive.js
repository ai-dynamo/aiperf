// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * ARCHIVE — namespace-scoped past-runs browser.
 *
 * Mounted at ``/ns/:ns/archive``. The route segment is the namespace
 * filter; there is no cross-namespace grouping or "All namespaces"
 * toggle. The view renders a flat list of every run whose
 * ``j.namespace === ns``, plus a phase-bucket strip and a text search
 * that narrows within the current namespace.
 */

import { html } from 'htm/preact';
import { useMemo, useState } from 'preact/hooks';
import { jobs } from '../lib/state.js';
import { navigate } from '../lib/router.js';

const BUCKETS = [
  { key: 'all',   label: 'All',    match: () => true },
  { key: 'live',  label: 'Live',   match: j => ['running', 'initializing', 'pending'].includes((j.phase ?? '').toLowerCase()) },
  { key: 'pass',  label: 'Passed', match: j => ['completed', 'succeeded'].includes((j.phase ?? '').toLowerCase()) },
  { key: 'fault', label: 'Failed', match: j => ['failed', 'error'].includes((j.phase ?? '').toLowerCase()) },
];

const SORTS = [
  { key: 'newest', label: 'Newest first' },
  { key: 'oldest', label: 'Oldest first' },
  { key: 'rps',    label: 'Throughput (desc)' },
  { key: 'p99',    label: 'P99 latency (desc)' },
  { key: 'dur',    label: 'Duration (desc)' },
];

function phaseBucket(phase) {
  const p = (phase ?? '').toLowerCase();
  if (p === 'running' || p === 'initializing' || p === 'pending') return 'live';
  if (p === 'failed' || p === 'error')                              return 'fault';
  if (p === 'completed' || p === 'succeeded')                       return 'passed';
  return 'other';
}

function statusTone(phase) {
  const b = phaseBucket(phase);
  if (b === 'fault')  return 'bad';
  if (b === 'live')   return 'info';
  if (b === 'passed') return 'good';
  return 'neutral';
}

function statusLabel(phase) {
  const b = phaseBucket(phase);
  if (b === 'fault')  return 'Failed';
  if (b === 'live')   return 'Running';
  if (b === 'passed') return 'Passed';
  return phase ? String(phase) : '—';
}

function relAge(ts) {
  if (!ts) return '—';
  const s = Math.floor((Date.now() - new Date(ts).getTime()) / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h`;
  return `${Math.floor(h / 86400)}d`;
}

function jobStartMs(j) {
  return j.startTime ? new Date(j.startTime).getTime() : 0;
}

function jobDurationSec(j) {
  if (j.startTime && j.completionTime) {
    return (new Date(j.completionTime) - new Date(j.startTime)) / 1000;
  }
  if (j.startTime) return (Date.now() - new Date(j.startTime).getTime()) / 1000;
  return 0;
}

function compareJobs(sort) {
  const num = v => (v == null ? -Infinity : Number(v));
  switch (sort) {
    case 'oldest': return (a, b) => jobStartMs(a) - jobStartMs(b);
    case 'rps':    return (a, b) => num(b.throughputRps) - num(a.throughputRps);
    case 'p99':    return (a, b) => num(b.latencyP99Ms) - num(a.latencyP99Ms);
    case 'dur':    return (a, b) => jobDurationSec(b) - jobDurationSec(a);
    case 'newest':
    default:       return (a, b) => jobStartMs(b) - jobStartMs(a);
  }
}

export function Archive({ ns }) {
  const [bucket, setBucket] = useState('all');
  const [q, setQ] = useState('');
  const [sort, setSort] = useState('newest');
  const list = (jobs.value ?? []).filter(j => j.namespace === ns);
  const cur = BUCKETS.find(b => b.key === bucket) ?? BUCKETS[0];

  const filtered = useMemo(() => {
    let r = list.filter(cur.match);
    if (q) {
      const needle = q.toLowerCase();
      r = r.filter(j => (j.name ?? '').toLowerCase().includes(needle)
                    || (j.model ?? '').toLowerCase().includes(needle));
    }
    return [...r].sort(compareJobs(sort));
  }, [list, bucket, q, sort]);

  const bucketCount = key => list.filter(BUCKETS.find(b => b.key === key).match).length;

  const shownCount = filtered.length;
  const hiddenCount = Math.max(0, list.length - shownCount);
  const liveShown   = filtered.filter(j => phaseBucket(j.phase) === 'live').length;
  const passedShown = filtered.filter(j => phaseBucket(j.phase) === 'passed').length;
  const faultShown  = filtered.filter(j => phaseBucket(j.phase) === 'fault').length;

  return html`
    <div class="v-archive" data-testid="page-archive">
      <section class="hm-summary" data-testid="arch-summary">
        <div class="hm-summary-item"><span>Shown</span><b>${shownCount}</b></div>
        <div class="hm-summary-item"><span>Hidden</span><b>${hiddenCount}</b></div>
        <div class="hm-summary-item"><span>Running</span><b>${liveShown}</b></div>
        <div class="hm-summary-item"><span>Passed</span><b>${passedShown}</b></div>
        <div class="hm-summary-item"><span>Failed</span><b>${faultShown}</b></div>
      </section>

      <div class="arch-tabs" role="tablist">
        ${BUCKETS.map(b => html`
          <button
            key=${b.key}
            class=${'arch-tab' + (bucket === b.key ? ' arch-tab--active' : '')}
            data-testid=${'tab-' + b.key}
            onclick=${() => setBucket(b.key)}
            role="tab"
            aria-selected=${bucket === b.key}
          >${b.label}<span class="arch-tab-count">${bucketCount(b.key)}</span></button>
        `)}
      </div>

      <div class="arch-toolbar">
        <input
          type="text"
          value=${q}
          oninput=${e => setQ(e.target.value)}
          placeholder="filter name / model…"
          data-testid="arch-search"
        />
        <select
          value=${sort}
          onchange=${e => setSort(e.target.value)}
          data-testid="arch-sort"
          aria-label="Sort runs"
        >
          ${SORTS.map(s => html`<option key=${s.key} value=${s.key}>${s.label}</option>`)}
        </select>
      </div>

      ${filtered.length === 0
        ? html`
          <div class="empty" data-testid="arch-empty">
            No matches — try changing the filter${q ? ' or clearing the search' : ''}.
          </div>`
        : filtered.map(j => {
          const tone = statusTone(j.phase);
          const label = statusLabel(j.phase);
          const modelShort = j.model ? String(j.model).split('/').pop() : '—';
          return html`
            <div
              key=${j.namespace + '/' + j.name}
              class="arch-row"
              data-testid=${'arch-row-' + j.namespace + '-' + j.name}
              onclick=${() => navigate(`/ns/${encodeURIComponent(j.namespace)}/run/${encodeURIComponent(j.name)}`)}
            >
              <div class="arch-row-name">${j.name}</div>
              <div class="arch-row-meta">${modelShort}</div>
              <div class="arch-row-meta">${relAge(j.completionTime ?? j.created)}</div>
              <div class="arch-row-status"><span class=${'chip chip--' + tone}>${label}</span></div>
            </div>
          `;
        })
      }
    </div>
  `;
}
