// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * HOME — dense, functional list grouped by namespace.
 *
 * Previous iterations (magazine-style blocks, editorial serif titles)
 * over-weighted decoration and under-served density. This version is pure
 * dashboard: one compact summary strip at the top, one dense
 * list of runs below, subtle namespace lane dividers between groups. No
 * giant namespace typography, no corner brackets, no aggregate panels per
 * namespace — all the aggregate numbers that matter live in the single
 * summary strip. The user comes to Home to scan "what's running right now,
 * sorted so the live stuff is at the top", and click into whatever they
 * care about.
 *
 * Sort order within each namespace: LIVE first, then FAULT, then PASSED
 * (newest first within each bucket). Namespaces ordered by: has-live first,
 * then has-fault, then alphabetical — the top of the page is always the
 * most time-sensitive content.
 */

import { html } from 'htm/preact';
import { useEffect, useRef, useState } from 'preact/hooks';
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

function groupByNamespace(list) {
  const map = new Map();
  for (const j of list) {
    const ns = j.namespace || 'default';
    if (!map.has(ns)) map.set(ns, []);
    map.get(ns).push(j);
  }
  // Order namespaces by priority: live > fault > alphabetical
  const rank = ns => {
    const runs = map.get(ns);
    if (runs.some(j => phaseBucket(j.phase) === 'live')) return 0;
    if (runs.some(j => phaseBucket(j.phase) === 'fault')) return 1;
    return 2;
  };
  return [...map.entries()].sort((a, b) => {
    const ra = rank(a[0]), rb = rank(b[0]);
    if (ra !== rb) return ra - rb;
    return a[0].localeCompare(b[0]);
  });
}

function orderWithin(runs) {
  const bucketRank = { live: 0, fault: 1, passed: 2, other: 3 };
  return [...runs].sort((a, b) => {
    const ra = bucketRank[phaseBucket(a.phase)] ?? 9;
    const rb = bucketRank[phaseBucket(b.phase)] ?? 9;
    if (ra !== rb) return ra - rb;
    const ta = new Date(a.startTime ?? a.created ?? 0).getTime();
    const tb = new Date(b.startTime ?? b.created ?? 0).getTime();
    return tb - ta;
  });
}

/* ──────────────────────── summary strip ────────────────────── */

function SummaryStrip({ list, ci, nsCount }) {
  const live   = list.filter(j => phaseBucket(j.phase) === 'live');
  const passed = list.filter(j => phaseBucket(j.phase) === 'passed');
  const fault  = list.filter(j => phaseBucket(j.phase) === 'fault');
  const gpus = ci?.gpus ?? ci?.gpuCount ?? ci?.gpu_count ?? null;
  const gpuCap = ci?.gpuCapacity ?? ci?.gpu_capacity ?? null;
  return html`
    <section class="hm-summary" data-testid="hm-summary">
      <${Cell} label="Running" value=${live.length}   tone=${live.length > 0 ? 'live' : 'dim'} />
      <${Cell} label="Passed"  value=${passed.length} tone=${passed.length > 0 ? 'pass' : 'dim'} />
      <${Cell} label="Fault"   value=${fault.length}  tone=${fault.length > 0 ? 'fault' : 'dim'} />
      <${Cell} label="NS"      value=${nsCount}       tone="dim" />
      ${gpus != null && html`
        <${Cell} label="GPUs" value=${gpuCap ? `${gpus} / ${gpuCap}` : String(gpus)} tone="dim" />
      `}
    </section>
  `;
}

function Cell({ label, value, tone }) {
  return html`
    <div class=${'hm-cell hm-cell--' + (tone ?? 'dim')}>
      <span class="hm-cell-label">${label}</span>
      <span class="hm-cell-val">${value}</span>
    </div>
  `;
}

/* ──────────────────────────── row ─────────────────────────── */

function Row({ job }) {
  const bucket = phaseBucket(job.phase);
  const rps = job.throughputRps;
  const p99 = job.latencyP99Ms;
  const pct = job.progressPercent;
  const age = job.startTime ? (Date.now() - new Date(job.startTime).getTime()) / 1000 : null;
  const href = `/run/${encodeURIComponent(job.namespace)}/${encodeURIComponent(job.name)}`;
  return html`
    <button
      class=${'hm-row hm-row--' + bucket}
      onclick=${() => navigate(href)}
      data-testid=${'hm-row-' + job.namespace + '-' + job.name}
    >
      <span class=${'hm-row-dot hm-row-dot--' + bucket} aria-hidden="true"></span>

      <span class="hm-row-name">
        ${job.name}
        ${job.model && html`<small>${job.model.split('/').pop()}</small>`}
      </span>

      <span class=${'hm-row-phase hm-row-phase--' + bucket}>
        ${(job.phase ?? '—').toUpperCase()}
      </span>

      <span class="hm-row-rps">
        ${rps != null ? fmtNumber(rps, 0) : '—'}
      </span>
      <span class="hm-row-p99">
        ${p99 != null ? fmtInt(p99) : '—'}
      </span>

      ${bucket === 'live' && pct != null ? html`
        <span class="hm-row-prog">
          <span class="hm-row-prog-track">
            <span class="hm-row-prog-fill" style=${'width: ' + pct + '%'}></span>
          </span>
          <span class="hm-row-prog-val">${Math.round(pct)}%</span>
        </span>
      ` : html`
        <span class="hm-row-prog hm-row-prog--static">
          ${bucket === 'passed' ? 'done' : bucket === 'fault' ? 'fail' : '—'}
        </span>
      `}

      <span class="hm-row-age">${age != null ? fmtDuration(age) : '—'}</span>
      <i class="ph ph-caret-right hm-row-arrow" aria-hidden="true"></i>
    </button>
  `;
}

/* ─────────────────────────────── view ──────────────────────────── */

export function Home() {
  const list = jobs.value ?? [];
  const ci = clusterInfo.value;
  const byNs = groupByNamespace(list);
  const [firstTick, setFirstTick] = useState(true);
  const mountRef = useRef(Date.now());

  useEffect(() => {
    if (!firstTick) return;
    if (list.length > 0) { setFirstTick(false); return; }
    const elapsed = Date.now() - mountRef.current;
    const remaining = Math.max(0, 2000 - elapsed);
    const id = setTimeout(() => setFirstTick(false), remaining);
    return () => clearTimeout(id);
  }, [firstTick, list.length]);

  if (list.length === 0 && firstTick) {
    return html`
      <div class="v-home" data-testid="page-home">
        <section class="home-pitch home-pitch--scan" data-testid="home-scanning">
          <div class="home-pitch-tag">
            <span class="home-pitch-light home-pitch-light--pulse"></span>
            SCANNING…
          </div>
        </section>
      </div>
    `;
  }

  if (list.length === 0) {
    return html`
      <div class="v-home" data-testid="page-home">
        <section class="home-pitch">
          <div class="home-pitch-tag">
            <span class="home-pitch-light"></span>
            NO RUNS TRACKED
          </div>
          <h1 class="home-pitch-headline">Launch a benchmark.</h1>
          <p class="home-pitch-body">
            The operator hasn't seen any AIPerfJobs yet. Kick one off from a
            template or paste your own YAML — the new CR lands in whatever
            namespace you target, and it shows up here immediately.
          </p>
          <div class="home-pitch-actions">
            <button class="home-pitch-cta" onclick=${() => navigate('/launch')} data-testid="home-launch-cta">
              <i class="ph ph-plus"></i>
              Launch new run
              <kbd>⌘N</kbd>
            </button>
          </div>
        </section>
      </div>
    `;
  }

  return html`
    <div class="v-home v-home--list" data-testid="page-home">
      <${SummaryStrip} list=${list} ci=${ci} nsCount=${byNs.length} />

      <section class="hm-table" role="table" aria-label="All runs grouped by namespace">
        <header class="hm-thead" role="row">
          <span></span>
          <span class="hm-th">Run</span>
          <span class="hm-th">Phase</span>
          <span class="hm-th hm-th--num">r/s</span>
          <span class="hm-th hm-th--num">p99 ms</span>
          <span class="hm-th">Progress</span>
          <span class="hm-th hm-th--num">Age</span>
          <span></span>
        </header>

        ${byNs.map(([ns, runs]) => {
          const sorted = orderWithin(runs);
          const live = runs.filter(j => phaseBucket(j.phase) === 'live').length;
          const fault = runs.filter(j => phaseBucket(j.phase) === 'fault').length;
          const passed = runs.filter(j => phaseBucket(j.phase) === 'passed').length;
          const state = live > 0 ? 'live' : fault > 0 ? 'fault' : 'idle';
          return html`
            <div class=${'hm-ns-group hm-ns-group--' + state} key=${ns}>
              <div class="hm-ns-bar" role="rowheader">
                <span class=${'hm-ns-dot hm-ns-dot--' + state}></span>
                <span class="hm-ns-name">${ns}</span>
                <span class="hm-ns-meta">
                  ${live > 0   && html`<span class="hm-ns-chip hm-ns-chip--live">${live} live</span>`}
                  ${fault > 0  && html`<span class="hm-ns-chip hm-ns-chip--fault">${fault} fault</span>`}
                  ${passed > 0 && html`<span class="hm-ns-chip hm-ns-chip--pass">${passed} passed</span>`}
                </span>
                <span class="hm-ns-total">${runs.length}</span>
              </div>
              ${sorted.map(j => html`<${Row} key=${j.namespace + '/' + j.name} job=${j} />`)}
            </div>
          `;
        })}
      </section>
    </div>
  `;
}
