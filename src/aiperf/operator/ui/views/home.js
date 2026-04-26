// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * HOME — mission-control dashboard.
 *
 * Layout (top to bottom):
 *   1. Stats hero — running / passed / failed / total / GPUs as big numerics
 *   2. Active runs strip — full cards with phase progress for live runs
 *   3. Scatter pod + Top performers — side-by-side grid
 *   4. Recent runs — compact rows with state-tinted left edge
 */

import { html } from 'htm/preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import { jobs, clusterInfo } from '../lib/state.js';
import { navigate } from '../lib/router.js';
import { fmtDuration, fmtInt, fmtNumber } from '../lib/format.js';
import { applyChartTheme, PALETTE } from '../lib/chart-theme.js';

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

function modelShort(model) {
  if (!model) return '';
  return String(model).split('/').pop();
}

function bestBy(jobList, field, dir = 'max') {
  let best = null;
  for (const j of jobList) {
    const b = phaseBucket(j.phase);
    if (b !== 'passed') continue;
    const v = j[field];
    if (v == null) continue;
    if (best === null) { best = j; continue; }
    if (dir === 'max' && v > best[field]) best = j;
    if (dir === 'min' && v < best[field]) best = j;
  }
  return best;
}

function progressPct(j) {
  if (j.progressPct != null) return Math.max(0, Math.min(100, Number(j.progressPct)));
  if (j.requestsCompleted != null && j.requestsTotal) {
    return Math.max(0, Math.min(100, (j.requestsCompleted / j.requestsTotal) * 100));
  }
  return null;
}

function StatTile({ label, value, sub, mod }) {
  return html`
    <div class=${'hm-stat hm-stat--' + mod}>
      <div class="hm-stat-label">${label}</div>
      <div class="hm-stat-val">${value}</div>
      ${sub && html`<div class="hm-stat-sub">${sub}</div>`}
    </div>
  `;
}

function ActiveCard({ job }) {
  const pct = progressPct(job);
  const elapsed = job.startTime ? (Date.now() - new Date(job.startTime).getTime()) / 1000 : null;
  const href = `/run/${encodeURIComponent(job.namespace)}/${encodeURIComponent(job.name)}`;
  return html`
    <div
      class="hm-active-card"
      onclick=${() => navigate(href)}
      data-testid=${'hm-active-' + job.namespace + '-' + job.name}
    >
      <div class="hm-active-card-head">
        <div>
          <div class="hm-active-card-name">${job.name}</div>
          <div class="hm-active-card-ns">${job.namespace} · ${modelShort(job.model) || 'no model'}</div>
        </div>
        <span class="chip chip--info">${titleCase(job.phase) || 'Running'}</span>
      </div>
      ${pct != null && html`
        <div class="hm-active-card-track">
          <div class="hm-active-card-fill" style=${'width:' + pct + '%'}></div>
        </div>
      `}
      <div class="hm-active-card-stats">
        <div>
          <div class="hm-active-card-stat-lab">Throughput</div>
          <div class="hm-active-card-stat-val">${job.throughputRps != null ? fmtNumber(job.throughputRps, 1) : '—'}</div>
        </div>
        <div>
          <div class="hm-active-card-stat-lab">Latency p99</div>
          <div class="hm-active-card-stat-val">${job.latencyP99Ms != null ? fmtInt(job.latencyP99Ms) + ' ms' : '—'}</div>
        </div>
        <div>
          <div class="hm-active-card-stat-lab">Elapsed</div>
          <div class="hm-active-card-stat-val">${elapsed != null ? fmtDuration(elapsed) : '—'}</div>
        </div>
      </div>
    </div>
  `;
}

function modelColorFor(model, models) {
  const i = models.indexOf(model);
  return PALETTE[i >= 0 ? i % PALETTE.length : 0];
}

const SCATTER_AXES = [
  { key: 'tps_p99',   x: 'throughputRps',   y: 'latencyP99Ms', xL: 'Throughput (req/s)', yL: 'Latency p99 (ms)' },
  { key: 'tps_ttft',  x: 'throughputRps',   y: 'ttftMs',       xL: 'Throughput (req/s)', yL: 'TTFT (ms)' },
  { key: 'tokps_p99', x: 'tokenThroughput', y: 'latencyP99Ms', xL: 'Token/s',            yL: 'Latency p99 (ms)' },
];

function ScatterPod({ completed }) {
  const [axisKey, setAxisKey] = useState('tps_p99');
  const canvasRef = useRef(null);
  const chartRef = useRef(null);
  const axes = SCATTER_AXES.find(a => a.key === axisKey);

  const points = completed.filter(j => j[axes.x] != null && j[axes.y] != null);
  const models = [...new Set(points.map(j => j.model || 'unknown'))];

  useEffect(() => {
    if (!canvasRef.current || !window.Chart) return;
    if (chartRef.current) { chartRef.current.destroy(); chartRef.current = null; }
    if (points.length === 0) return;

    applyChartTheme();
    const datasets = models.map((m, i) => ({
      label: m,
      data: points.filter(p => (p.model || 'unknown') === m).map(p => ({
        x: p[axes.x],
        y: p[axes.y],
        name: p.name,
      })),
      backgroundColor: modelColorFor(m, models) + 'cc',
      borderColor: modelColorFor(m, models),
      borderWidth: 1.5,
      pointRadius: 7,
      pointHoverRadius: 10,
    }));

    chartRef.current = new window.Chart(canvasRef.current, {
      type: 'scatter',
      data: { datasets },
      options: {
        maintainAspectRatio: false,
        plugins: {
          legend: { display: true, position: 'top', labels: { boxWidth: 8, padding: 10 } },
          tooltip: {
            callbacks: {
              label: ctx => `${ctx.dataset.label} · ${ctx.raw.name}: ${fmtNumber(ctx.raw.x, 1)} / ${fmtNumber(ctx.raw.y, 0)}`,
            },
          },
        },
        scales: {
          x: { title: { display: true, text: axes.xL }, ticks: { font: { family: "'JetBrains Mono', monospace", size: 10 } } },
          y: { title: { display: true, text: axes.yL }, ticks: { font: { family: "'JetBrains Mono', monospace", size: 10 } } },
        },
      },
    });

    return () => { if (chartRef.current) { chartRef.current.destroy(); chartRef.current = null; } };
  }, [axisKey, points.length]);

  return html`
    <div class="hm-scatter-pod" data-testid="hm-scatter">
      <div class="hm-scatter-head">
        <div class="hm-scatter-title">Throughput × Latency</div>
        <div class="hm-scatter-axes">
          ${SCATTER_AXES.map(a => html`
            <button
              class=${axisKey === a.key ? 'is-active' : ''}
              onclick=${() => setAxisKey(a.key)}
            >${a.key.replace('_', '×')}</button>
          `)}
        </div>
      </div>
      <div class="hm-scatter-body">
        ${points.length === 0
          ? html`<div class="hm-scatter-empty">No completed runs yet.</div>`
          : html`<canvas ref=${canvasRef}></canvas>`}
      </div>
    </div>
  `;
}

function PodiumCard({ label, value, unit, name, mod }) {
  if (value == null) {
    return html`
      <div class=${'hm-podium-card hm-podium-card--' + mod}>
        <div class="hm-podium-label">${label}</div>
        <div class="hm-podium-val">—</div>
        <div class="hm-podium-name">No completed runs</div>
      </div>
    `;
  }
  return html`
    <div class=${'hm-podium-card hm-podium-card--' + mod}>
      <div class="hm-podium-label">${label}</div>
      <div class="hm-podium-val">${value}<span class="hm-podium-unit"> ${unit}</span></div>
      <div class="hm-podium-name">${name}</div>
    </div>
  `;
}

function Row({ job }) {
  const { tone, label } = chipForPhase(job.phase);
  const age = job.startTime ? (Date.now() - new Date(job.startTime).getTime()) / 1000 : null;
  const href = `/run/${encodeURIComponent(job.namespace)}/${encodeURIComponent(job.name)}`;
  return html`
    <div
      class="hm-row"
      data-state=${phaseBucket(job.phase)}
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

  const live      = list.filter(j => phaseBucket(j.phase) === 'live');
  const completed = list.filter(j => phaseBucket(j.phase) === 'passed');
  const failed    = list.filter(j => phaseBucket(j.phase) === 'fault');
  const total     = list.length;

  const gpus = ci?.gpus ?? ci?.gpuCount ?? ci?.gpu_count ?? null;
  const gpuCap = ci?.gpuCapacity ?? ci?.gpu_capacity ?? null;
  const nodes = ci?.nodes ?? ci?.nodeCount ?? ci?.node_count ?? null;

  const bestThr  = bestBy(list, 'throughputRps');
  const bestTtft = bestBy(list, 'ttftMs', 'min');
  const bestTok  = bestBy(list, 'tokenThroughput');

  // Recent runs: completed + failed, newest first, top 8.
  const recent = [...completed, ...failed]
    .sort((a, b) => new Date(b.startTime ?? b.created ?? 0) - new Date(a.startTime ?? a.created ?? 0))
    .slice(0, 8);

  return html`
    <div class="v-home" data-testid="page-home">

      <section class="hm-stats" data-testid="hm-summary">
        <${StatTile} mod="running" label="Running" value=${fmtInt(live.length)} sub=${live.length === 1 ? 'live now' : 'live now'} />
        <${StatTile} mod="passed"  label="Completed" value=${fmtInt(completed.length)} sub="passed" />
        <${StatTile} mod="failed"  label="Failed"    value=${fmtInt(failed.length)}    sub=${failed.length > 0 ? 'needs attention' : 'all clear'} />
        <${StatTile} mod="total"   label="Total"     value=${fmtInt(total)}            sub="all-time" />
        <${StatTile} mod="gpus"    label="GPUs"      value=${gpus != null ? fmtInt(gpus) : '—'} sub=${gpuCap != null ? fmtInt(gpuCap) + ' allocatable' : (nodes != null ? fmtInt(nodes) + ' nodes' : 'cluster')} />
      </section>

      ${live.length > 0 && html`
        <section class="hm-active-section" data-testid="hm-active">
          <div class="hm-active-head">
            <div class="hm-active-title">Active runs · ${live.length}</div>
          </div>
          <div class="hm-active-grid">
            ${live.map(j => html`<${ActiveCard} key=${j.namespace + '/' + j.name} job=${j} />`)}
          </div>
        </section>
      `}

      <section class="hm-mission">
        <${ScatterPod} completed=${completed} />
        <div class="hm-podium">
          <div class="hm-section-title">Top performers</div>
          <div class="hm-podium-grid">
            <${PodiumCard}
              mod="thr"
              label="Best throughput"
              value=${bestThr ? fmtNumber(bestThr.throughputRps, 1) : null}
              unit="req/s"
              name=${bestThr ? bestThr.name : null}
            />
            <${PodiumCard}
              mod="ttft"
              label="Lowest TTFT"
              value=${bestTtft ? fmtInt(bestTtft.ttftMs) : null}
              unit="ms"
              name=${bestTtft ? bestTtft.name : null}
            />
            <${PodiumCard}
              mod="tok"
              label="Best token/s"
              value=${bestTok ? fmtInt(bestTok.tokenThroughput) : null}
              unit="tok/s"
              name=${bestTok ? bestTok.name : null}
            />
          </div>
        </div>
      </section>

      ${recent.length > 0 && html`
        <section class="hm-recent-section">
          <div class="hm-section-title">Recent runs · ${recent.length}</div>
          <div class="hm-rows">
            ${recent.map(j => html`<${Row} key=${j.namespace + '/' + j.name} job=${j} />`)}
          </div>
        </section>
      `}

    </div>
  `;
}
