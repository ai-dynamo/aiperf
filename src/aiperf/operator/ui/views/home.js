// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * HOME — v2 reskin: hairline-card list grouped (sorted) so live runs surface
 * at the top.
 *
 * Pitch card (empty state) + scan card (initial poll) + summary bar +
 * one card per run. Status chips use the .chip family (info/good/bad/neutral)
 * with normal-case labels — no all-caps FAULT/FAILED shoutmarks.
 */

import { html } from 'htm/preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import { jobs, clusterInfo } from '../lib/state.js';
import { navigate } from '../lib/router.js';
import { fmtDuration } from '../lib/format.js';

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

function chipForPhase(phase) {
  const p = (phase ?? '').toLowerCase();
  if (p === 'running')      return { tone: 'info',    label: 'Running' };
  if (p === 'initializing') return { tone: 'info',    label: 'Initializing' };
  if (p === 'pending')      return { tone: 'info',    label: 'Pending' };
  if (p === 'failed')       return { tone: 'bad',     label: 'Failed' };
  if (p === 'error')        return { tone: 'bad',     label: 'Failed' };
  if (p === 'completed')    return { tone: 'good',    label: 'Completed' };
  if (p === 'succeeded')    return { tone: 'good',    label: 'Completed' };
  return { tone: 'neutral', label: titleCase(phase) };
}

function orderRuns(list) {
  const bucketRank = { live: 0, fault: 1, passed: 2, other: 3 };
  return [...list].sort((a, b) => {
    const ra = bucketRank[phaseBucket(a.phase)] ?? 9;
    const rb = bucketRank[phaseBucket(b.phase)] ?? 9;
    if (ra !== rb) return ra - rb;
    const ta = new Date(a.startTime ?? a.created ?? 0).getTime();
    const tb = new Date(b.startTime ?? b.created ?? 0).getTime();
    return tb - ta;
  });
}

function modelShort(model) {
  if (!model) return '';
  return String(model).split('/').pop();
}

function Row({ job }) {
  const { tone, label } = chipForPhase(job.phase);
  const age = job.startTime ? (Date.now() - new Date(job.startTime).getTime()) / 1000 : null;
  const href = `/run/${encodeURIComponent(job.namespace)}/${encodeURIComponent(job.name)}`;
  return html`
    <div
      class="hm-row"
      onclick=${() => navigate(href)}
      data-testid=${'hm-row-' + job.namespace + '-' + job.name}
    >
      <div>
        <div class="hm-row-name">${job.name}</div>
        <div class="hm-row-ns">${job.namespace}</div>
      </div>
      <div class="hm-row-meta">${modelShort(job.model) || '—'}</div>
      <div class="hm-row-meta">${age != null ? fmtDuration(age) : '—'}</div>
      <div class="hm-row-status">
        <span class=${'chip chip--' + tone}>${label}</span>
      </div>
    </div>
  `;
}

export function Home() {
  const list = jobs.value ?? [];
  const ci = clusterInfo.value;
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
        <section class="home-pitch" data-testid="home-scanning">
          <div class="home-pitch-title">Scanning…</div>
          <div class="home-pitch-sub">Looking for AIPerfJobs in the cluster.</div>
        </section>
      </div>
    `;
  }

  if (list.length === 0) {
    return html`
      <div class="v-home" data-testid="page-home">
        <section class="home-pitch">
          <div class="home-pitch-title">Launch a benchmark.</div>
          <div class="home-pitch-sub">
            The operator hasn't seen any AIPerfJobs yet. Kick one off from a
            template or paste your own YAML.
          </div>
          <button
            class="home-pitch-cta"
            onclick=${() => navigate('/launch')}
            data-testid="home-launch-cta"
          >
            Launch new run
          </button>
        </section>
      </div>
    `;
  }

  const live   = list.filter(j => phaseBucket(j.phase) === 'live').length;
  const passed = list.filter(j => phaseBucket(j.phase) === 'passed').length;
  const fault  = list.filter(j => phaseBucket(j.phase) === 'fault').length;
  const total  = list.length;
  const gpus = ci?.gpus ?? ci?.gpuCount ?? ci?.gpu_count ?? null;
  const gpuCap = ci?.gpuCapacity ?? ci?.gpu_capacity ?? null;
  const gpuVal = gpus != null ? (gpuCap ? `${gpus} / ${gpuCap}` : String(gpus)) : null;

  const sorted = orderRuns(list);

  return html`
    <div class="v-home" data-testid="page-home">
      <section class="hm-summary" data-testid="hm-summary">
        <span class="hm-summary-item"><b>${live}</b> running</span>
        <span class="hm-summary-item"><b>${passed}</b> completed</span>
        <span class="hm-summary-item"><b>${fault}</b> failed</span>
        <span class="hm-summary-item"><b>${total}</b> total</span>
        ${gpuVal != null && html`<span class="hm-summary-item"><b>${gpuVal}</b> GPUs</span>`}
      </section>

      <div class="hm-rows">
        ${sorted.map(j => html`<${Row} key=${j.namespace + '/' + j.name} job=${j} />`)}
      </div>
    </div>
  `;
}
