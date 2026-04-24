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
import { useEffect, useRef, useState } from 'preact/hooks';
import { api } from '../lib/api.js';
import { navigate, queryParams } from '../lib/router.js';
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

const jobKey = (ns, id) => `${ns}/${id}`;

export function Analysis() {
  const [axes, setAxes] = useState('tps_p99');
  const [board, setBoard] = useState({ metric: 'request_throughput', stat: 'avg' });
  const [lb, setLb] = useState(null);
  const [summaryMap, setSummaryMap] = useState({});
  const [err, setErr] = useState(null);
  const [selected, setSelected] = useState(new Set());
  const [overlayMode, setOverlayMode] = useState(false);
  const [enabledClusters, setEnabledClusters] = useState(null);
  // Deep-link: read `?cluster=...` once and auto-enable only that cluster when data loads.
  // Tracked via a ref so user chip-bar toggles after the initial apply are preserved.
  const deepLinkCluster = queryParams().cluster ?? null;
  const deepLinkAppliedRef = useRef(false);
  const axis = AXES.find(a => a.key === axes);
  const yIsSmallerBetter = axis.y === 'request_latency' || axis.y === 'time_to_first_token' || axis.y === 'inter_token_latency';
  const clusterKeyOf = (ns, model) => `${ns ?? 'unknown'} · ${model ?? 'unknown'}`;
  const shortModel = (m) => (m ? m.split('/').pop() : 'unknown');

  const toggleSelected = (ns, id) => {
    setSelected(prev => {
      const next = new Set(prev);
      const k = jobKey(ns, id);
      if (next.has(k)) next.delete(k); else next.add(k);
      return next;
    });
  };
  const clearSelection = () => { setSelected(new Set()); setOverlayMode(false); };

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

  // Auto-select the deep-linked cluster exactly once, after leaderboard entries
  // and at least one summary have loaded. If the cluster key is missing from the
  // computed groups, fall through to the default (all enabled).
  useEffect(() => {
    if (deepLinkAppliedRef.current) return;
    if (!deepLinkCluster) return;
    if (entries.length === 0) return;
    if (Object.keys(summaryMap).length === 0) return;
    const keys = new Set();
    for (const e of entries) {
      const s = summaryMap[e.job_id];
      if (!s) continue;
      keys.add(clusterKeyOf(e.namespace, e.model));
    }
    if (keys.has(deepLinkCluster)) {
      setEnabledClusters(new Set([deepLinkCluster]));
    }
    deepLinkAppliedRef.current = true;
  }, [deepLinkCluster, entries, summaryMap]);

  /* ── overlay datasets (line-per-run across stat keys of axis.y) ── */
  const OVERLAY_STATS = ['min', 'avg', 'p50', 'p90', 'p99', 'max'];
  const overlayEntries = entries.filter(e => selected.has(jobKey(e.namespace, e.job_id)));
  const overlayDatasets = [];
  for (const e of overlayEntries) {
    const s = summaryMap[e.job_id];
    if (!s) continue;
    const metricObj = s[axis.y];
    if (!metricObj || typeof metricObj !== 'object') continue;
    const data = OVERLAY_STATS.map(k => {
      const v = metricObj[k];
      return typeof v === 'number' ? v : null;
    });
    const present = data.filter(v => v != null).length;
    if (present < 2) continue;
    const color = modelColor(e.model ?? 'unknown');
    overlayDatasets.push({
      label: e.name,
      data,
      borderColor: color,
      backgroundColor: color,
      borderWidth: 1.8,
      pointRadius: 4,
      pointHoverRadius: 8,
      tension: 0.25,
      spanGaps: true,
    });
  }
  const overlayRenderable = overlayDatasets.length >= 2;
  const overlayActive = overlayMode && overlayRenderable;

  /* ── scatter ── */
  const clusterGroups = {};
  for (const e of entries) {
    const s = summaryMap[e.job_id];
    if (!s) continue;
    const xv = s[axis.x]?.[axis.xs];
    const yv = s[axis.y]?.[axis.ys];
    if (xv == null || yv == null) continue;
    const ck = clusterKeyOf(e.namespace, e.model);
    (clusterGroups[ck] ??= { ns: e.namespace ?? 'unknown', model: e.model ?? 'unknown', points: [] })
      .points.push({ ...e, x: xv, y: yv });
  }
  const allClusterKeys = Object.keys(clusterGroups);
  const activeClusters = enabledClusters ?? new Set(allClusterKeys);
  const toggleCluster = (ck) => {
    setEnabledClusters(prev => {
      const base = prev ?? new Set(allClusterKeys);
      const next = new Set(base);
      if (next.has(ck)) next.delete(ck); else next.add(ck);
      return next;
    });
  };

  const MUTED = 'rgba(244, 238, 222, 0.35)';
  const datasets = [];
  for (const [ck, grp] of Object.entries(clusterGroups)) {
    if (!activeClusters.has(ck)) continue;
    const isSingleton = grp.points.length < 2;
    const color = isSingleton ? MUTED : modelColor(grp.model);
    datasets.push({
      label: ck,
      data: grp.points.map(p => ({ x: p.x, y: p.y, jobName: p.name, cluster: ck })),
      backgroundColor: color, borderColor: color, borderWidth: 1.4,
      pointRadius: 7, pointHoverRadius: 11,
      showLine: false,
      order: 1,
    });
    if (!isSingleton) {
      const sorted = [...grp.points].sort((a, b) => a.x - b.x);
      const frontier = [];
      let bestY = yIsSmallerBetter ? Infinity : -Infinity;
      for (const p of sorted) {
        const better = yIsSmallerBetter ? p.y <= bestY : p.y >= bestY;
        if (better) { bestY = p.y; frontier.push({ x: p.x, y: p.y }); }
      }
      if (frontier.length >= 2) {
        datasets.push({
          label: `${ck} · frontier`,
          data: frontier,
          borderColor: color, backgroundColor: color,
          borderWidth: 1.6,
          borderDash: [4, 4],
          showLine: true,
          pointRadius: 0,
          pointHoverRadius: 0,
          fill: false,
          order: 2,
          legend: false,
        });
      }
    }
  }
  const chartOpts = applyChartTheme({
    plugins: {
      legend: {
        position: 'top', align: 'end',
        labels: {
          usePointStyle: true, pointStyle: 'rect', boxWidth: 10, padding: 16,
          filter: (item, data) => {
            const ds = data.datasets[item.datasetIndex];
            return ds && ds.legend !== false;
          },
        },
      },
      tooltip: {
        callbacks: {
          label: ctx => `${ctx.raw.cluster ?? ctx.dataset.label} · ${ctx.raw.jobName ?? ''} — ${fmtNumber(ctx.raw.x, 0)} · ${fmtInt(ctx.raw.y)}`,
        },
      },
    },
    scales: {
      x: { title: { display: true, text: axis.xl.toUpperCase(), color: 'var(--paper-faint)', font: { size: 10, weight: '700' } }, grid: { color: 'var(--edge-1)' } },
      y: { title: { display: true, text: axis.yl.toUpperCase(), color: 'var(--paper-faint)', font: { size: 10, weight: '700' } }, grid: { color: 'var(--edge-1)' } },
    },
  });
  const overlayOpts = applyChartTheme({
    plugins: {
      legend: { position: 'top', align: 'end', labels: { usePointStyle: true, pointStyle: 'line', boxWidth: 18, padding: 16 } },
      tooltip: {
        callbacks: {
          label: ctx => `${ctx.dataset.label} — ${ctx.label}: ${fmtNumber(ctx.parsed.y, 2)}`,
        },
      },
    },
    scales: {
      x: { title: { display: true, text: 'STAT', color: 'var(--paper-faint)', font: { size: 10, weight: '700' } }, grid: { color: 'var(--edge-1)' } },
      y: { title: { display: true, text: axis.yl.toUpperCase(), color: 'var(--paper-faint)', font: { size: 10, weight: '700' } }, grid: { color: 'var(--edge-1)' } },
    },
  });

  return html`
    <div class="v-analysis" data-testid="page-leaderboard">
      <header class="v-head">
        <div class="v-head-title">
          <span class="v-head-caret">▸</span>
          <h1>COMPARE</h1>
        </div>
        <div class="v-head-meta">PARETO · LEADERBOARD · OVERLAY</div>
      </header>

      ${err && html`<div class="v-analysis-err">FETCH FAILED — ${err}</div>`}

      <div class="v-analysis-grid">
        <section class="v-analysis-chart">
          <header class="slab-head slab-head--flush">
            <div class="slab-head-title">
              <span class="slab-head-caret">▸</span>
              ${overlayActive ? `OVERLAY · ${axis.yl}` : `PARETO · ${axis.xl} × ${axis.yl}`}
            </div>
            <div class="v-analysis-axes">
              ${AXES.map(a => html`
                <button key=${a.key} class=${axes === a.key ? 'is-active' : ''} onclick=${() => setAxes(a.key)}>${a.key.replace('_', '×').toUpperCase()}</button>
              `)}
            </div>
          </header>
          ${allClusterKeys.length > 0 && html`
            <div class="v-analysis-axes" style=${{ padding: '8px 12px', borderBottom: '1px solid var(--edge-1)', gap: '6px', flexWrap: 'wrap' }}>
              ${allClusterKeys.map(ck => {
                const grp = clusterGroups[ck];
                const isSingleton = grp.points.length < 2;
                const on = activeClusters.has(ck);
                const dot = isSingleton ? MUTED : modelColor(grp.model);
                return html`
                  <button
                    key=${ck}
                    class=${on ? 'is-active' : ''}
                    onclick=${() => toggleCluster(ck)}
                    title=${ck}
                    style=${{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}
                  >
                    <span style=${{ display: 'inline-block', width: '8px', height: '8px', borderRadius: '50%', background: dot }}></span>
                    <span>${grp.ns} · ${shortModel(grp.model)}</span>
                    <span style=${{ opacity: 0.6 }}>· ${grp.points.length}${isSingleton ? ' · singleton' : ''}</span>
                  </button>
                `;
              })}
            </div>
          `}
          ${selected.size > 0 && html`
            <div class="v-analysis-axes" style=${{ padding: '8px 12px', borderBottom: '1px solid var(--edge-1)', gap: '8px' }}>
              <button
                class=${overlayActive ? 'is-active' : ''}
                disabled=${!overlayRenderable}
                onclick=${() => setOverlayMode(m => !m)}
                title=${overlayRenderable ? '' : 'Need at least 2 selected runs with renderable data'}
              >${overlayActive ? 'SHOW PARETO' : `OVERLAY ${selected.size} RUNS`}</button>
              <button onclick=${clearSelection}>CLEAR SELECTION</button>
              ${selected.size >= 2 && !overlayRenderable && html`
                <span class="cond cond--idle" style=${{ marginLeft: '8px' }}>INSUFFICIENT SERIES DATA</span>
              `}
            </div>
          `}
          <div class="v-analysis-chart-body">
            ${overlayActive
              ? html`<${ChartWrapper} type="line" data=${{ labels: OVERLAY_STATS, datasets: overlayDatasets }} options=${overlayOpts} height=${520} />`
              : datasets.length === 0
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
              : entries.map((e, i) => {
                  const isSel = selected.has(jobKey(e.namespace, e.job_id));
                  return html`
                  <li key=${e.job_id}>
                    <button
                      class=${'v-analysis-board-row' + (i === 0 ? ' is-top' : '') + (isSel ? ' is-active' : '')}
                      onclick=${() => navigate('/run/' + encodeURIComponent(e.namespace) + '/' + encodeURIComponent(e.name))}
                    >
                      <span
                        class=${'cond ' + (isSel ? 'cond--pass' : 'cond--idle')}
                        role="checkbox"
                        aria-checked=${isSel}
                        title=${isSel ? 'Selected for overlay' : 'Add to overlay selection'}
                        style=${{ cursor: 'pointer', marginRight: '6px', padding: '2px 6px', minWidth: '22px', textAlign: 'center' }}
                        onclick=${(ev) => { ev.stopPropagation(); ev.preventDefault(); toggleSelected(e.namespace, e.job_id); }}
                      >${isSel ? '✓' : '·'}</span>
                      <span class="v-analysis-rank">${String(i + 1).padStart(2, '0')}</span>
                      <span class="v-analysis-name">
                        ${e.name}
                        <small>${e.model ? e.model.split('/').pop() : '—'}${e.concurrency ? ' · conc ' + e.concurrency : ''}</small>
                        <small>cluster: ${e.namespace ?? 'unknown'} · ${shortModel(e.model)}</small>
                      </span>
                      <span class="v-analysis-val">
                        ${e.value != null ? fmtNumber(e.value, 2) : '—'}
                      </span>
                    </button>
                  </li>
                `;})
            }
          </ol>
        </aside>
      </div>
    </div>
  `;
}
