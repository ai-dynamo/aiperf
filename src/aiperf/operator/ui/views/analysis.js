// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * ANALYSIS — combined "Pareto lab" view.
 *
 * Replaces the previous /leaderboard + /compare + /history pages with a single
 * workspace: a large pareto scatter on the left (pick axes + stat), a ranked
 * leaderboard on the right. Clicking a leaderboard row pins the run to the
 * main viewport (RUN view).
 */

import { html } from 'htm/preact';
import { useEffect, useState } from 'preact/hooks';
import { api } from '../lib/api.js';
import { navigate } from '../lib/router.js';
import { fmtInt, fmtNumber } from '../lib/format.js';
import { ChartWrapper } from '../components/chart-wrapper.js';
import { applyChartTheme } from '../lib/chart-theme.js';
import { modelColor } from '../lib/theme.js';

const METRICS = [
  { value: 'request_throughput',     label: 'Request Throughput' },
  { value: 'request_latency',        label: 'Request Latency' },
  { value: 'time_to_first_token',    label: 'Time to First Token' },
  { value: 'inter_token_latency',    label: 'Inter-Token Latency' },
  { value: 'output_token_throughput', label: 'Output Token Throughput' },
];
const STATS = [
  { value: 'avg', label: 'avg' },
  { value: 'p50', label: 'p50' },
  { value: 'p99', label: 'p99' },
  { value: 'min', label: 'min' },
  { value: 'max', label: 'max' },
];

const AXES = [
  { key: 'tps_p99',   x: 'request_throughput',      y: 'request_latency',   xl: 'Throughput (req/s)', yl: 'Latency P99 (ms)', xs: 'avg', ys: 'p99' },
  { key: 'tps_ttft',  x: 'request_throughput',      y: 'time_to_first_token', xl: 'Throughput (req/s)', yl: 'TTFT (ms)',        xs: 'avg', ys: 'avg' },
  { key: 'tok_p99',   x: 'output_token_throughput', y: 'request_latency',   xl: 'Token Throughput (tok/s)', yl: 'Latency P99 (ms)', xs: 'avg', ys: 'p99' },
];

export function Analysis() {
  const [axes, setAxes] = useState('tps_p99');
  const [board, setBoard] = useState({ metric: 'request_throughput', stat: 'avg' });
  const [lb, setLb] = useState(null);
  const [summaryMap, setSummaryMap] = useState({});
  const [err, setErr] = useState(null);
  const axis = AXES.find(a => a.key === axes);

  useEffect(() => {
    let cancel = false;
    api.getLeaderboard(board.metric, board.stat)
      .then(async r => {
        if (cancel) return;
        setLb(r); setErr(null);
        const entries = r?.entries ?? [];
        const results = await Promise.allSettled(
          entries.map(e => api.getJobSummary(e.namespace, e.job_id).then(s => ({ id: e.job_id, summary: s })))
        );
        if (cancel) return;
        const next = {};
        for (const rr of results) {
          if (rr.status !== 'fulfilled') continue;
          const { id, summary } = rr.value;
          next[id] = summary ?? {};
        }
        setSummaryMap(next);
      })
      .catch(e => { if (!cancel) setErr(e.message); });
    return () => { cancel = true; };
  }, [board.metric, board.stat]);

  const entries = lb?.entries ?? [];

  /* ── scatter ── */
  const modelGroups = {};
  for (const e of entries) {
    const s = summaryMap[e.job_id];
    if (!s) continue;
    const xv = s[axis.x]?.[axis.xs];
    const yv = s[axis.y]?.[axis.ys];
    if (xv == null || yv == null) continue;
    const m = e.model ?? 'unknown';
    (modelGroups[m] ??= []).push({ ...e, x: xv, y: yv });
  }
  const datasets = Object.entries(modelGroups).map(([model, pts]) => {
    const color = modelColor(model);
    return {
      label: model.split('/').pop(),
      data: pts.map(p => ({ x: p.x, y: p.y, jobName: p.name })),
      backgroundColor: color, borderColor: color, borderWidth: 1.4,
      pointRadius: 7, pointHoverRadius: 11,
    };
  });
  const chartOpts = applyChartTheme({
    plugins: {
      legend: { position: 'top', align: 'end', labels: { usePointStyle: true, pointStyle: 'rect', boxWidth: 10, padding: 16 } },
      tooltip: {
        callbacks: {
          label: ctx => `${ctx.dataset.label} · ${ctx.raw.jobName} — ${fmtNumber(ctx.raw.x, 0)} · ${fmtInt(ctx.raw.y)}`,
        },
      },
    },
    scales: {
      x: { title: { display: true, text: axis.xl.toUpperCase(), color: 'var(--paper-faint)', font: { size: 10, weight: '700' } }, grid: { color: 'var(--edge-1)' } },
      y: { title: { display: true, text: axis.yl.toUpperCase(), color: 'var(--paper-faint)', font: { size: 10, weight: '700' } }, grid: { color: 'var(--edge-1)' } },
    },
  });

  return html`
    <div class="v-analysis" data-testid="page-leaderboard">
      <header class="v-head">
        <div class="v-head-title">
          <span class="v-head-caret">▸</span>
          <h1>ANALYSIS LAB</h1>
        </div>
        <div class="v-head-meta">PARETO · LEADERBOARD</div>
      </header>

      ${err && html`<div class="v-analysis-err">FETCH FAILED — ${err}</div>`}

      <div class="v-analysis-grid">
        <section class="v-analysis-chart">
          <header class="slab-head slab-head--flush">
            <div class="slab-head-title">
              <span class="slab-head-caret">▸</span>
              PARETO · ${axis.xl} × ${axis.yl}
            </div>
            <div class="v-analysis-axes">
              ${AXES.map(a => html`
                <button key=${a.key} class=${axes === a.key ? 'is-active' : ''} onclick=${() => setAxes(a.key)}>${a.key.replace('_', '×').toUpperCase()}</button>
              `)}
            </div>
          </header>
          <div class="v-analysis-chart-body">
            ${datasets.length === 0
              ? html`<div class="slab-placeholder"><i class="ph ph-chart-scatter"></i>AWAITING DATA</div>`
              : html`<${ChartWrapper} type="scatter" data=${{ datasets }} options=${chartOpts} height=${520} />`
            }
          </div>
        </section>

        <aside class="v-analysis-board">
          <header class="slab-head slab-head--flush">
            <div class="slab-head-title">
              <span class="slab-head-caret">▸</span>
              LEADERBOARD
            </div>
          </header>
          <div class="v-analysis-board-controls">
            <select value=${board.metric} onchange=${e => setBoard(b => ({ ...b, metric: e.target.value }))}>
              ${METRICS.map(m => html`<option key=${m.value} value=${m.value}>${m.label}</option>`)}
            </select>
            <select value=${board.stat} onchange=${e => setBoard(b => ({ ...b, stat: e.target.value }))}>
              ${STATS.map(s => html`<option key=${s.value} value=${s.value}>${s.label}</option>`)}
            </select>
          </div>
          <ol class="v-analysis-board-list">
            ${entries.length === 0
              ? html`<li class="v-analysis-board-empty">NO ENTRIES</li>`
              : entries.map((e, i) => html`
                  <li key=${e.job_id}>
                    <button
                      class=${'v-analysis-board-row' + (i === 0 ? ' is-top' : '')}
                      onclick=${() => navigate('/run/' + encodeURIComponent(e.namespace) + '/' + encodeURIComponent(e.name))}
                    >
                      <span class="v-analysis-rank">${String(i + 1).padStart(2, '0')}</span>
                      <span class="v-analysis-name">
                        ${e.name}
                        <small>${e.model ? e.model.split('/').pop() : '—'}${e.concurrency ? ' · conc ' + e.concurrency : ''}</small>
                      </span>
                      <span class="v-analysis-val">
                        ${e.value != null ? fmtNumber(e.value, 2) : '—'}
                      </span>
                    </button>
                  </li>
                `)
            }
          </ol>
        </aside>
      </div>
    </div>
  `;
}
