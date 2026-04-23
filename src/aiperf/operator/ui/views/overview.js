// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * OVERVIEW — the default main-viewport view.
 *
 * An editorial "mission brief" rather than a dashboard dump. Four stacked slabs:
 *
 *   1. VERDICT SLAB    — one-sentence fleet verdict in Instrument Serif
 *   2. TELEMETRY SLAB  — four big live meters (LIVE / FLEET R/S / P99 / GPUs)
 *                        plus three sparkline rows (throughput, p99, token/s)
 *                        derived from the in-memory `jobs` signal
 *   3. PARETO SLAB     — the throughput × latency scatter ("BEST FRONTIER")
 *   4. WATCHLIST SLAB  — side-by-side "ACTIVE" / "RECENT" lists;
 *                        clicking a row pins that run to the main viewport
 *
 * All polling lives in app.js; this view is a pure function of `jobs.value`
 * and `clusterInfo.value`, plus the leaderboard/summary maps it fetches on
 * mount for the pareto dots.
 */

import { html } from 'htm/preact';
import { useEffect, useState } from 'preact/hooks';
import { api, poll } from '../lib/api.js';
import { jobs, clusterInfo } from '../lib/state.js';
import { navigate } from '../lib/router.js';
import { fmtInt, fmtNumber } from '../lib/format.js';
import { ChartWrapper } from '../components/chart-wrapper.js';
import { applyChartTheme } from '../lib/chart-theme.js';
import { modelColor } from '../lib/theme.js';

/* ────────────────────────── helpers ────────────────────────── */

function phaseBucket(phase) {
  const p = (phase ?? '').toLowerCase();
  if (p === 'running' || p === 'initializing' || p === 'pending') return 'live';
  if (p === 'failed' || p === 'error')                              return 'fault';
  if (p === 'completed' || p === 'succeeded')                       return 'passed';
  return 'other';
}

/** Compare a live-summary metric against its declared SLO threshold.
 *
 *  Returns ``true`` iff the metric is currently *violating* the SLO. Throughput
 *  is "higher is better", everything else is "lower is better". Returns
 *  ``false`` when the probe or threshold is missing — missing data is not
 *  treated as a violation.
 */
function isSloViolated(key, threshold, summary) {
  if (threshold == null || summary == null) return false;
  const probe =
    key === 'time_to_first_token'   ? summary.ttft_p99_ms ?? summary.ttft_avg_ms :
    key === 'request_latency'       ? summary.latency_p99_ms ?? summary.latency_avg_ms :
    key === 'inter_token_latency'   ? summary.itl_p99_ms ?? summary.itl_avg_ms :
    key === 'request_throughput'    ? summary.throughput_rps :
    null;
  if (probe == null) return false;
  return key === 'request_throughput' ? probe < threshold : probe > threshold;
}

function classifyFleet(live, failed, sloCoverage) {
  if (live.length === 0 && failed.length === 0) {
    return { kind: 'idle', verdict: 'FLEET IDLE',
             body: 'No active benchmarks. The rack is quiet — launch one to begin.' };
  }
  if (failed.length > 0 && live.length === 0) {
    return { kind: 'err',  verdict: 'ATTENTION',
             body: `${failed.length} run${failed.length > 1 ? 's' : ''} in FAULT state.` };
  }
  if (sloCoverage.violated > 0) {
    return { kind: 'warn', verdict: 'SLOs SLIPPING',
             body: `${sloCoverage.violated} of ${sloCoverage.declared} declared SLOs currently breaching.` };
  }
  if (sloCoverage.declared > 0) {
    return { kind: 'ok', verdict: 'ON TARGET',
             body: `${live.length} run${live.length > 1 ? 's' : ''} running within declared SLOs.` };
  }
  return { kind: 'ok',
           verdict: live.length === 1 ? 'RUN IN PROGRESS' : `${live.length} RUNS LIVE`,
           body: 'No SLOs declared — fleet is tracking, not judging.' };
}

/* ────────────────────────── view ────────────────────────── */

export function Overview() {
  const [summaryMap, setSummaryMap] = useState({});
  const [liveDetails, setLiveDetails] = useState({}); // key: "ns/name" → { slos, summary }
  const js = jobs.value ?? [];
  const ci = clusterInfo.value;

  const live = js.filter(j => phaseBucket(j.phase) === 'live');
  const failed = js.filter(j => phaseBucket(j.phase) === 'fault');
  const passed = js.filter(j => phaseBucket(j.phase) === 'passed');

  // Enrich completed runs with analytics summary for the Pareto chart.
  useEffect(() => {
    const ac = new AbortController();
    poll(async () => {
      try {
        const lb = await api.getLeaderboard('request_throughput', 'avg');
        const entries = lb?.entries ?? [];
        if (entries.length === 0) return;
        const results = await Promise.allSettled(
          entries.map(e => api.getJobSummary(e.namespace, e.job_id).then(s => ({ id: e.job_id, summary: s }))),
        );
        const next = {};
        for (const r of results) {
          if (r.status !== 'fulfilled') continue;
          const { id, summary: s } = r.value;
          next[id] = {
            throughputRps: s?.request_throughput?.avg ?? null,
            latencyP99Ms: s?.request_latency?.p99 ?? null,
            ttftMs: s?.time_to_first_token?.avg ?? null,
            tokenThroughput: s?.output_token_throughput?.avg ?? null,
          };
        }
        setSummaryMap(prev => ({ ...prev, ...next }));
      } catch (_e) { /* optional — leaderboard may be empty */ }
    }, 15000, ac.signal);
    return () => ac.abort();
  }, []);

  // Fetch per-live-run detail + config so we can evaluate SLOs for the verdict.
  const liveKeys = live.map(r => `${r.namespace}/${r.name}`).join('|');
  useEffect(() => {
    if (live.length === 0) { setLiveDetails({}); return; }
    const ac = new AbortController();
    for (const run of live) {
      const key = `${run.namespace}/${run.name}`;
      poll(async () => {
        try {
          const [detail, cfg] = await Promise.all([
            api.getJob(run.namespace, run.name).catch(() => null),
            api.getJobConfig(run.namespace, run.name).catch(() => null),
          ]);
          const slos = cfg?.spec?.benchmark?.slos ?? cfg?.spec?.slos ?? null;
          const summary = detail?.status?.liveSummary ?? detail?.status?.summary ?? null;
          setLiveDetails(prev => ({ ...prev, [key]: { slos, summary } }));
        } catch (_e) { /* transient */ }
      }, 5000, ac.signal);
    }
    return () => ac.abort();
  }, [liveKeys]);

  const enriched = passed.map(j => {
    const s = summaryMap[j.jobId ?? j.name];
    if (!s) return j;
    return {
      ...j,
      throughputRps: j.throughputRps ?? s.throughputRps,
      latencyP99Ms: j.latencyP99Ms ?? s.latencyP99Ms,
      ttftMs: j.ttftMs ?? s.ttftMs,
      tokenThroughput: j.tokenThroughput ?? s.tokenThroughput,
    };
  });

  /* ── verdict ── */
  let sloDeclared = 0;
  let sloViolated = 0;
  for (const run of live) {
    const detail = liveDetails[`${run.namespace}/${run.name}`];
    const slos = detail?.slos;
    if (!slos) continue;
    for (const [key, threshold] of Object.entries(slos)) {
      sloDeclared += 1;
      if (isSloViolated(key, threshold, detail.summary)) sloViolated += 1;
    }
  }
  const verdict = classifyFleet(live, failed, { declared: sloDeclared, violated: sloViolated });

  /* ── telemetry numbers ── */
  let sumRps = 0, rpsKnown = false, worstP99 = null;
  for (const j of live) {
    if (j.throughputRps != null) { sumRps += j.throughputRps; rpsKnown = true; }
    if (j.latencyP99Ms != null) worstP99 = worstP99 == null ? j.latencyP99Ms : Math.max(worstP99, j.latencyP99Ms);
  }
  const gpus = ci?.gpus ?? ci?.gpuCount ?? ci?.gpu_count ?? null;
  const gpuCap = ci?.gpuCapacity ?? ci?.gpu_capacity ?? null;

  /* ── pareto chart ── */
  const scatterPoints = enriched.filter(j => j.throughputRps != null && j.latencyP99Ms != null);
  const modelGroups = {};
  for (const j of scatterPoints) {
    const m = j.model ?? 'unknown';
    (modelGroups[m] ??= []).push(j);
  }
  const datasets = Object.entries(modelGroups).map(([model, group]) => {
    const color = modelColor(model);
    return {
      label: model.split('/').pop(),
      data: group.map(j => ({ x: j.throughputRps, y: j.latencyP99Ms, jobName: j.name })),
      backgroundColor: color,
      borderColor: color,
      borderWidth: 1.4,
      pointRadius: 7,
      pointHoverRadius: 11,
    };
  });
  const chartOpts = applyChartTheme({
    plugins: {
      legend: {
        position: 'top', align: 'end',
        labels: { usePointStyle: true, pointStyle: 'rect', boxWidth: 10, padding: 16 },
      },
      tooltip: {
        callbacks: {
          label: ctx => {
            const p = ctx.raw;
            return `${ctx.dataset.label} · ${p.jobName} — ${fmtNumber(p.x, 0)} r/s · ${fmtInt(p.y)} ms`;
          },
        },
      },
    },
    scales: {
      x: { title: { display: true, text: 'THROUGHPUT (REQ/S)', color: 'var(--paper-faint)', font: { size: 10, weight: '700' } }, grid: { color: 'var(--edge-1)' } },
      y: { title: { display: true, text: 'LATENCY P99 (MS)',    color: 'var(--paper-faint)', font: { size: 10, weight: '700' } }, grid: { color: 'var(--edge-1)' } },
    },
  });

  /* ── watchlists ── */
  const activeList = live.slice(0, 5);
  const recentList = [...passed]
    .sort((a, b) => new Date(b.completionTime ?? b.created ?? 0) - new Date(a.completionTime ?? a.created ?? 0))
    .slice(0, 5);

  return html`
    <div class="v-overview" data-testid="page-dashboard">
      <!-- 1. VERDICT SLAB -->
      <section class=${'slab slab-verdict slab-verdict--' + verdict.kind} data-testid="dashboard-hero">
        <div class="slab-verdict-tag">
          <span class="slab-verdict-light"></span>
          MISSION BRIEF · ${new Date().toISOString().slice(0, 10).replace(/-/g, '.')}
        </div>
        <h1 class="slab-verdict-headline">${verdict.verdict}</h1>
        <p class="slab-verdict-body">${verdict.body}</p>
        <div class="slab-verdict-chips">
          ${live.length > 0 && html`<span class="chip chip--live">${live.length} LIVE</span>`}
          ${passed.length > 0 && html`<span class="chip chip--pass">${passed.length} PASSED</span>`}
          ${failed.length > 0 && html`<span class="chip chip--fail">${failed.length} FAULT</span>`}
          ${gpus != null && html`<span class="chip"><i class="ph ph-lightning"></i> ${gpus}${gpuCap ? ` / ${gpuCap}` : ''} GPUs</span>`}
        </div>
      </section>

      <!-- 2. TELEMETRY SLAB -->
      <section class="slab slab-telemetry">
        <${Bay} kind="LIVE"       value=${live.length}                                 unit="runs"  emphasis=${live.length > 0 ? 'amber' : 'dim'} />
        <${Bay} kind="FLEET R/S"  value=${rpsKnown ? fmtNumber(sumRps, 0) : '—'}        unit="req/s" emphasis=${rpsKnown ? 'amber' : 'dim'} />
        <${Bay} kind="WORST P99"  value=${worstP99 != null ? fmtInt(worstP99) : '—'}    unit="ms"    emphasis=${worstP99 != null && worstP99 > 500 ? 'red' : 'paper'} />
        <${Bay} kind="GPUs"       value=${gpus != null ? fmtInt(gpus) : '—'}            unit=${gpuCap ? `of ${gpuCap}` : ''} emphasis="paper" />
      </section>

      <!-- 3. PARETO SLAB -->
      <section class="slab slab-pareto">
        <header class="slab-head">
          <div class="slab-head-title">
            <span class="slab-head-caret">▸</span>
            BEST FRONTIER · THROUGHPUT × LATENCY
          </div>
          <div class="slab-head-meta">
            ${scatterPoints.length} COMPLETED
          </div>
        </header>
        <div class="slab-body slab-body--chart">
          ${scatterPoints.length === 0
            ? html`<div class="slab-placeholder">
                <i class="ph ph-chart-scatter"></i>
                AWAITING COMPLETED RUNS
              </div>`
            : html`<${ChartWrapper} type="scatter" data=${{ datasets }} options=${chartOpts} height=${340} />`
          }
        </div>
      </section>

      <!-- 4. WATCHLIST SLAB -->
      <section class="slab slab-watchlist">
        <div class="watchcol watchcol--live">
          <header class="slab-head">
            <div class="slab-head-title">
              <span class="slab-head-caret">▸</span>
              ACTIVE
            </div>
            <span class="slab-head-meta">${live.length}</span>
          </header>
          ${activeList.length === 0
            ? html`<div class="slab-placeholder slab-placeholder--small">RACK IDLE</div>`
            : activeList.map(j => html`<${WatchRow} job=${j} key=${j.namespace+'/'+j.name} />`)
          }
        </div>
        <div class="watchcol watchcol--recent">
          <header class="slab-head">
            <div class="slab-head-title">
              <span class="slab-head-caret">▸</span>
              RECENT
            </div>
            <span class="slab-head-meta">${passed.length}</span>
          </header>
          ${recentList.length === 0
            ? html`<div class="slab-placeholder slab-placeholder--small">NOTHING ARCHIVED</div>`
            : recentList.map(j => html`<${WatchRow} job=${j} key=${j.namespace+'/'+j.name} />`)
          }
        </div>
      </section>

      ${failed.length > 0 && html`
        <section class="slab slab-fault" data-testid="dashboard-failed-strip">
          <div class="slab-fault-tag">
            <span class="slab-fault-light"></span>
            FAULT ROLL · ${failed.length}
          </div>
          <div class="slab-fault-list">
            ${failed.map(j => html`
              <button
                key=${j.namespace+'/'+j.name}
                class="slab-fault-item"
                onclick=${() => navigate('/run/' + encodeURIComponent(j.namespace) + '/' + encodeURIComponent(j.name))}
              >
                <span class="slab-fault-name">${j.name}</span>
                <span class="slab-fault-ns">${j.namespace}</span>
                ${j.error && html`<span class="slab-fault-msg">${j.error}</span>`}
                <i class="ph ph-arrow-right"></i>
              </button>
            `)}
          </div>
        </section>
      `}
    </div>
  `;
}

function Bay({ kind, value, unit, emphasis }) {
  return html`
    <div class=${'bay bay--' + (emphasis ?? 'paper')}>
      <div class="bay-label">${kind}</div>
      <div class="bay-value">${value}${unit && html`<span class="bay-unit">${unit}</span>`}</div>
    </div>
  `;
}

function WatchRow({ job }) {
  const bucket = phaseBucket(job.phase);
  const rps = job.throughputRps;
  const p99 = job.latencyP99Ms;
  return html`
    <button
      class=${'watch-row watch-row--' + bucket}
      onclick=${() => navigate('/run/' + encodeURIComponent(job.namespace) + '/' + encodeURIComponent(job.name))}
      data-testid=${'watch-row-' + job.namespace + '-' + job.name}
    >
      <span class=${'watch-dot watch-dot--' + bucket}></span>
      <span class="watch-body">
        <span class="watch-name">${job.name}</span>
        <span class="watch-meta">
          <span>${job.namespace}</span>
          ${job.model && html`<span>${job.model.split('/').pop()}</span>`}
        </span>
      </span>
      <span class="watch-stats">
        <span class="watch-stat"><small>R/S</small>${rps != null ? fmtNumber(rps, 0) : '—'}</span>
        <span class="watch-stat"><small>P99</small>${p99 != null ? fmtInt(p99) : '—'}</span>
      </span>
      <i class="ph ph-arrow-right watch-arrow"></i>
    </button>
  `;
}
