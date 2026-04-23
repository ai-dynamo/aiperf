// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * FLEET — all-runs table view.
 *
 * Dense, sortable, searchable list of every run the operator knows about
 * (live + passed + fault + archived). The left rail still shows the short
 * "live/passed/fault" cut; this view is the exhaustive grid.
 *
 * Filter tabs pivot by phase; a free-text search narrows by name / namespace.
 * Clicking a row swaps the main viewport to its RUN view without leaving the
 * Flight Deck shell.
 */

import { html } from 'htm/preact';
import { useMemo, useState } from 'preact/hooks';
import { jobs } from '../lib/state.js';
import { navigate } from '../lib/router.js';
import { fmtInt, fmtNumber } from '../lib/format.js';

const BUCKETS = [
  { key: 'all',   label: 'ALL',       match: () => true },
  { key: 'live',  label: 'LIVE',      match: j => ['running', 'initializing', 'pending'].includes((j.phase ?? '').toLowerCase()) },
  { key: 'pass',  label: 'PASSED',    match: j => ['completed', 'succeeded'].includes((j.phase ?? '').toLowerCase()) },
  { key: 'fault', label: 'FAULT',     match: j => ['failed', 'error'].includes((j.phase ?? '').toLowerCase()) },
  { key: 'arch',  label: 'ARCHIVED',  match: j => j.source === 'archived' },
];

const COLS = [
  { key: 'name',       label: 'NAME' },
  { key: 'namespace',  label: 'NS' },
  { key: 'phase',      label: 'PHASE' },
  { key: 'model',      label: 'MODEL' },
  { key: 'rps',        label: 'R/S' },
  { key: 'p99',        label: 'P99' },
  { key: 'progress',   label: 'PROGRESS' },
  { key: 'age',        label: 'AGE' },
];

function relAge(ts) {
  if (!ts) return '—';
  const s = Math.floor((Date.now() - new Date(ts).getTime()) / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h`;
  return `${Math.floor(h / 24)}d`;
}

function cell(job, key) {
  switch (key) {
    case 'name':      return job.name ?? '';
    case 'namespace': return job.namespace ?? '';
    case 'phase':     return job.phase ?? '';
    case 'model':     return job.model ?? '';
    case 'rps':       return job.throughputRps ?? -1;
    case 'p99':       return job.latencyP99Ms ?? -1;
    case 'progress':  return job.progressPercent ?? -1;
    case 'age':       return job.created ? new Date(job.created).getTime() : 0;
    default: return '';
  }
}

function phaseKind(phase) {
  const p = (phase ?? '').toLowerCase();
  if (['running', 'initializing', 'pending'].includes(p)) return 'live';
  if (['failed', 'error'].includes(p)) return 'fault';
  if (['completed', 'succeeded'].includes(p)) return 'passed';
  return 'other';
}

export function Fleet() {
  const [bucket, setBucket] = useState('all');
  const [q, setQ] = useState('');
  const [sortKey, setSortKey] = useState('age');
  const [sortDir, setSortDir] = useState(-1);

  const list = jobs.value ?? [];
  const cur = BUCKETS.find(b => b.key === bucket);

  const filtered = useMemo(() => {
    let r = list.filter(cur?.match ?? (() => true));
    if (q) {
      const needle = q.toLowerCase();
      r = r.filter(j => (j.name ?? '').toLowerCase().includes(needle)
                    || (j.namespace ?? '').toLowerCase().includes(needle)
                    || (j.model ?? '').toLowerCase().includes(needle));
    }
    r.sort((a, b) => {
      const av = cell(a, sortKey);
      const bv = cell(b, sortKey);
      if (av < bv) return -sortDir;
      if (av > bv) return sortDir;
      return 0;
    });
    return r;
  }, [list, bucket, q, sortKey, sortDir]);

  function toggleSort(key) {
    if (sortKey === key) setSortDir(d => -d);
    else { setSortKey(key); setSortDir(1); }
  }

  const bucketCount = key => list.filter(BUCKETS.find(b => b.key === key).match).length;

  return html`
    <div class="v-fleet" data-testid="page-jobs">
      <header class="v-head">
        <div class="v-head-title">
          <span class="v-head-caret">▸</span>
          <h1>FLEET GRID</h1>
        </div>
        <div class="v-head-meta">
          ${filtered.length} of ${list.length} RUNS
        </div>
      </header>

      <div class="v-fleet-controls">
        <div class="v-fleet-tabs" role="tablist">
          ${BUCKETS.map(b => html`
            <button
              key=${b.key}
              class=${'v-fleet-tab' + (bucket === b.key ? ' is-active' : '')}
              onclick=${() => setBucket(b.key)}
              role="tab"
              aria-selected=${bucket === b.key}
            >
              ${b.label}
              <span class="v-fleet-tab-count">${bucketCount(b.key)}</span>
            </button>
          `)}
        </div>
        <div class="v-fleet-search">
          <i class="ph ph-magnifying-glass"></i>
          <input
            type="text"
            value=${q}
            oninput=${e => setQ(e.target.value)}
            placeholder="filter name / namespace / model…"
            data-testid="fleet-search"
          />
          ${q && html`<button onclick=${() => setQ('')} aria-label="Clear"><i class="ph ph-x"></i></button>`}
        </div>
      </div>

      <div class="v-fleet-table" role="table">
        <div class="v-fleet-thead" role="row">
          ${COLS.map(c => html`
            <button
              key=${c.key}
              class=${'v-fleet-th' + (sortKey === c.key ? ' is-sorted' : '')}
              onclick=${() => toggleSort(c.key)}
              role="columnheader"
              aria-sort=${sortKey === c.key ? (sortDir > 0 ? 'ascending' : 'descending') : 'none'}
              data-testid=${'col-header-' + c.key}
            >
              ${c.label}
              ${sortKey === c.key && html`<span class="v-fleet-sort">${sortDir > 0 ? '↑' : '↓'}</span>`}
            </button>
          `)}
        </div>
        <div class="v-fleet-tbody" data-testid="job-table">
          ${filtered.length === 0
            ? html`<div class="v-fleet-empty">NO RUNS MATCH</div>`
            : filtered.map(j => {
                const kind = phaseKind(j.phase);
                return html`
                  <button
                    key=${j.namespace + '/' + j.name}
                    class=${'v-fleet-row v-fleet-row--' + kind}
                    onclick=${() => navigate('/run/' + encodeURIComponent(j.namespace ?? 'default') + '/' + encodeURIComponent(j.name ?? ''))}
                    role="row"
                    data-testid=${'job-row-' + (j.namespace ?? '') + '-' + (j.name ?? '')}
                  >
                    <span class="v-fleet-td v-fleet-td--name">
                      <span class=${'v-fleet-dot v-fleet-dot--' + kind}></span>
                      ${j.name}
                    </span>
                    <span class="v-fleet-td v-fleet-td--dim">${j.namespace}</span>
                    <span class="v-fleet-td">
                      <span class=${'v-fleet-phase v-fleet-phase--' + kind}>${(j.phase ?? '—').toUpperCase()}</span>
                    </span>
                    <span class="v-fleet-td v-fleet-td--dim v-fleet-td--ellipsis">${j.model ?? '—'}</span>
                    <span class="v-fleet-td v-fleet-td--num">${j.throughputRps != null ? fmtNumber(j.throughputRps, 0) : '—'}</span>
                    <span class="v-fleet-td v-fleet-td--num">${j.latencyP99Ms != null ? fmtInt(j.latencyP99Ms) : '—'}</span>
                    <span class="v-fleet-td v-fleet-td--bar">
                      ${j.progressPercent != null
                        ? html`
                            <span class="v-fleet-bar">
                              <span class="v-fleet-bar-fill" style=${'width: ' + Math.round(j.progressPercent) + '%'}></span>
                            </span>
                            <span class="v-fleet-bar-label">${Math.round(j.progressPercent)}%</span>
                          `
                        : html`<span class="v-fleet-td--dim">—</span>`}
                    </span>
                    <span class="v-fleet-td v-fleet-td--dim">${relAge(j.created)}</span>
                  </button>
                `;
              })
          }
        </div>
      </div>
    </div>
  `;
}
