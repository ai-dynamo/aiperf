// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * ARCHIVE — namespace-grouped history of every run the operator has tracked.
 *
 * Same organizing principle as Home (namespace-first, no drill-down) but:
 *   - Denser row layout
 *   - Includes a global phase-filter strip at the top (live / passed / fault)
 *   - Text search narrows across every namespace at once
 *   - A namespace whose runs are all filtered out disappears (no empty blocks)
 *
 * Home is for "what's happening right now per namespace"; Archive is for
 * "show me the full history sliced by namespace".
 */

import { html } from 'htm/preact';
import { useMemo, useState } from 'preact/hooks';
import { jobs } from '../lib/state.js';
import { navigate } from '../lib/router.js';
import { fmtDuration, fmtInt, fmtNumber } from '../lib/format.js';

const BUCKETS = [
  { key: 'all',   label: 'ALL',    match: () => true },
  { key: 'live',  label: 'LIVE',   match: j => ['running', 'initializing', 'pending'].includes((j.phase ?? '').toLowerCase()) },
  { key: 'pass',  label: 'PASSED', match: j => ['completed', 'succeeded'].includes((j.phase ?? '').toLowerCase()) },
  { key: 'fault', label: 'FAULT',  match: j => ['failed', 'error'].includes((j.phase ?? '').toLowerCase()) },
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

function groupByNamespace(list) {
  const map = new Map();
  for (const j of list) {
    const ns = j.namespace || 'default';
    if (!map.has(ns)) map.set(ns, []);
    map.get(ns).push(j);
  }
  return [...map.entries()].sort((a, b) => a[0].localeCompare(b[0]));
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
  // Stable comparators; unknown numeric fields sort last.
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

export function Archive() {
  const [bucket, setBucket] = useState('all');
  const [q, setQ] = useState('');
  const [sort, setSort] = useState('newest');
  const list = jobs.value ?? [];
  const cur = BUCKETS.find(b => b.key === bucket) ?? BUCKETS[0];

  const filtered = useMemo(() => {
    let r = list.filter(cur.match);
    if (q) {
      const needle = q.toLowerCase();
      r = r.filter(j => (j.name ?? '').toLowerCase().includes(needle)
                    || (j.namespace ?? '').toLowerCase().includes(needle)
                    || (j.model ?? '').toLowerCase().includes(needle));
    }
    return [...r].sort(compareJobs(sort));
  }, [list, bucket, q, sort]);

  const byNs = groupByNamespace(filtered);
  const bucketCount = key => list.filter(BUCKETS.find(b => b.key === key).match).length;

  const shownCount = filtered.length;
  const hiddenCount = Math.max(0, list.length - shownCount);
  const liveShown   = filtered.filter(j => phaseBucket(j.phase) === 'live').length;
  const passedShown = filtered.filter(j => phaseBucket(j.phase) === 'passed').length;
  const faultShown  = filtered.filter(j => phaseBucket(j.phase) === 'fault').length;

  return html`
    <div class="v-archive" data-testid="page-archive">
      <header class="v-head">
        <div class="v-head-title">
          <span class="v-head-caret">▸</span>
          <h1>ARCHIVE</h1>
        </div>
        <div class="v-head-meta">
          ${filtered.length} of ${list.length} RUNS · ${byNs.length} NAMESPACE${byNs.length === 1 ? '' : 'S'}
        </div>
      </header>

      <section class="hm-summary" data-testid="arch-summary">
        <div class="hm-cell hm-cell--dim">
          <span class="hm-cell-label">Shown</span>
          <span class="hm-cell-val">${shownCount}</span>
        </div>
        <div class=${'hm-cell ' + (hiddenCount > 0 ? 'hm-cell--accent' : 'hm-cell--dim')}>
          <span class="hm-cell-label">Hidden</span>
          <span class="hm-cell-val">${hiddenCount}</span>
        </div>
        <div class=${'hm-cell ' + (liveShown > 0 ? 'hm-cell--live' : 'hm-cell--dim')}>
          <span class="hm-cell-label">Live</span>
          <span class="hm-cell-val">${liveShown}</span>
        </div>
        <div class=${'hm-cell ' + (passedShown > 0 ? 'hm-cell--pass' : 'hm-cell--dim')}>
          <span class="hm-cell-label">Passed</span>
          <span class="hm-cell-val">${passedShown}</span>
        </div>
        <div class=${'hm-cell ' + (faultShown > 0 ? 'hm-cell--fault' : 'hm-cell--dim')}>
          <span class="hm-cell-label">Fault</span>
          <span class="hm-cell-val">${faultShown}</span>
        </div>
        <div class="hm-cell hm-cell--dim">
          <span class="hm-cell-label">NS</span>
          <span class="hm-cell-val">${byNs.length}</span>
        </div>
      </section>

      <div class="v-archive-controls">
        <div class="v-archive-tabs" role="tablist">
          ${BUCKETS.map(b => html`
            <button
              key=${b.key}
              class=${'v-archive-tab' + (bucket === b.key ? ' is-active' : '')}
              onclick=${() => setBucket(b.key)}
              role="tab"
              aria-selected=${bucket === b.key}
            >
              ${b.label}
              <span class="v-archive-tab-count">${bucketCount(b.key)}</span>
            </button>
          `)}
        </div>
        <div class="v-archive-search">
          <i class="ph ph-magnifying-glass"></i>
          <input
            type="text"
            value=${q}
            oninput=${e => setQ(e.target.value)}
            placeholder="filter name / namespace / model…"
            data-testid="archive-search"
          />
          ${q && html`<button onclick=${() => setQ('')} aria-label="Clear"><i class="ph ph-x"></i></button>`}
        </div>
        <select
          class="v-archive-tab"
          value=${sort}
          onchange=${e => setSort(e.target.value)}
          data-testid="archive-sort"
          aria-label="Sort runs"
          style="font-family: var(--f-mono); font-size: 11px; letter-spacing: 0.18em; text-transform: uppercase; padding: 0 var(--s-3);"
        >
          ${SORTS.map(s => html`<option key=${s.key} value=${s.key}>${s.label}</option>`)}
        </select>
      </div>

      ${byNs.length === 0
        ? html`
          <section class="arch-ns" data-testid="arch-empty">
            <div class="arch-ns-rows">
              <div
                class="arch-row"
                style="display: block; text-align: center; color: var(--paper-faint); font-family: var(--f-mono); font-size: 12px; letter-spacing: 0.12em; padding: var(--s-5) var(--s-4);"
              >
                NO MATCHES — TRY CHANGING THE BUCKET FILTER${q ? ' OR CLEARING THE SEARCH' : ''}
              </div>
            </div>
          </section>`
        : byNs.map(([ns, runs]) => {
          const live = runs.filter(j => phaseBucket(j.phase) === 'live').length;
          const passed = runs.filter(j => phaseBucket(j.phase) === 'passed').length;
          const fault = runs.filter(j => phaseBucket(j.phase) === 'fault').length;
          return html`
            <section key=${ns} class="arch-ns" data-testid=${'arch-ns-' + ns}>
              <header class="arch-ns-head">
                <div class="arch-ns-left">
                  <div class="arch-ns-eyebrow">NAMESPACE</div>
                  <h2 class="arch-ns-name">${ns}</h2>
                </div>
                <div class="arch-ns-counts">
                  ${live > 0   && html`<span class="ns-count ns-count--live">${live} LIVE</span>`}
                  ${passed > 0 && html`<span class="ns-count ns-count--pass">${passed} PASSED</span>`}
                  ${fault > 0  && html`<span class="ns-count ns-count--fail">${fault} FAULT</span>`}
                  <span class="arch-ns-total">${runs.length} total</span>
                </div>
              </header>

              <div class="arch-ns-rows">
                ${runs.map(j => {
                  const kind = phaseBucket(j.phase);
                  const modelShort = j.model ? String(j.model).split('/').pop() : null;
                  const conc = j.concurrency;
                  return html`
                    <button
                      key=${j.namespace + '/' + j.name}
                      class=${'arch-row arch-row--' + kind}
                      onclick=${() => navigate(`/run/${encodeURIComponent(j.namespace)}/${encodeURIComponent(j.name)}`)}
                      data-testid=${'arch-row-' + j.namespace + '-' + j.name}
                    >
                      <span class=${'arch-row-tag arch-row-tag--' + kind}>
                        <span class="arch-row-dot"></span>
                        ${(j.phase ?? '—').toUpperCase()}
                      </span>
                      <span class="arch-row-name">
                        ${j.name}
                        ${(modelShort || conc != null) && html`
                          <small>
                            ${modelShort ?? ''}${modelShort && conc != null ? ' · ' : ''}${conc != null ? `c=${conc}` : ''}
                          </small>
                        `}
                      </span>
                      <span class="arch-row-stat"><small>R/S</small>${j.throughputRps != null ? fmtNumber(j.throughputRps, 0) : '—'}</span>
                      <span class="arch-row-stat"><small>P99</small>${j.latencyP99Ms != null ? fmtInt(j.latencyP99Ms) : '—'}</span>
                      <span class="arch-row-dur">
                        ${j.startTime && j.completionTime
                          ? fmtDuration((new Date(j.completionTime) - new Date(j.startTime)) / 1000)
                          : j.startTime
                          ? fmtDuration((Date.now() - new Date(j.startTime).getTime()) / 1000)
                          : '—'}
                      </span>
                      <span class="arch-row-age">${relAge(j.completionTime ?? j.created)}</span>
                      <i class="ph ph-arrow-right"></i>
                    </button>
                  `;
                })}
              </div>
            </section>
          `;
        })
      }
    </div>
  `;
}
