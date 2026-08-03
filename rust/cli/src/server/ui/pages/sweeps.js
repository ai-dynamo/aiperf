// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Sweeps page: group runs by ``sweep_id`` and, for a selected sweep, plot a
// variation curve of a chosen metric across the sweep's runs (x = trial index
// or label). Summaries are loaded lazily for the active sweep's runs only.

import { html } from 'htm/preact';
import { useState, useEffect, useMemo } from 'preact/hooks';
import { runs } from '../lib/state.js';
import { getSummary } from '../lib/state.js';
import { fmtMetric, prettyTag, isLowerBetter } from '../lib/format.js';
import { navigate } from '../lib/router.js';
import { ChartWrapper, CHART_THEME } from '../components/chart-wrapper.js';

const STATS = ['avg', 'min', 'max', 'p50', 'p90', 'p99', 'p95'];

function statValue(metrics, tag, stat) {
  const m = metrics?.[tag];
  if (!m) return null;
  if (stat.startsWith('p')) return m.percentiles?.[stat] ?? null;
  return m[stat] ?? null;
}

/** Group runs by sweep_id; returns [{ sweepId, runs:[...] }] sorted by id. */
function groupSweeps(allRuns) {
  const groups = new Map();
  for (const r of allRuns) {
    if (r.sweep_id == null || r.sweep_id === '') continue;
    if (!groups.has(r.sweep_id)) groups.set(r.sweep_id, []);
    groups.get(r.sweep_id).push(r);
  }
  return [...groups.entries()]
    .map(([sweepId, list]) => ({
      sweepId,
      runs: [...list].sort((a, b) => (a.trial ?? 0) - (b.trial ?? 0)),
    }))
    .sort((a, b) => String(a.sweepId).localeCompare(String(b.sweepId)));
}

export function Sweeps() {
  const allRuns = runs.value;
  const sweeps = useMemo(() => groupSweeps(allRuns), [allRuns]);

  const [activeId, setActiveId] = useState(null);
  const [summaries, setSummaries] = useState({});
  const [metric, setMetric] = useState('output_token_throughput');
  const [stat, setStat] = useState('avg');

  // Default to the first sweep once data lands.
  useEffect(() => {
    if (activeId == null && sweeps.length > 0) setActiveId(sweeps[0].sweepId);
  }, [sweeps]);

  const active = sweeps.find((s) => s.sweepId === activeId) ?? null;

  // Load summaries for the active sweep's runs.
  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    const ids = active.runs.map((r) => r.id);
    Promise.allSettled(ids.map((id) => getSummary(id))).then((results) => {
      if (cancelled) return;
      setSummaries((prev) => {
        const next = { ...prev };
        results.forEach((res, i) => {
          if (res.status === 'fulfilled') next[ids[i]] = res.value;
        });
        return next;
      });
    });
    return () => {
      cancelled = true;
    };
  }, [activeId]);

  const metricTags = useMemo(() => {
    const set = new Set();
    for (const r of active?.runs ?? []) {
      for (const tag of Object.keys(summaries[r.id]?.metrics ?? {})) set.add(tag);
    }
    return [...set].sort();
  }, [active, summaries]);

  useEffect(() => {
    if (metricTags.length > 0 && !metricTags.includes(metric)) {
      setMetric(metricTags.includes('output_token_throughput') ? 'output_token_throughput' : metricTags[0]);
    }
  }, [metricTags]);

  const points = useMemo(() => {
    if (!active) return [];
    return active.runs
      .map((r) => ({
        run: r,
        x: r.trial != null ? String(r.trial) : r.label ?? r.id,
        y: statValue(summaries[r.id]?.metrics, metric, stat),
      }))
      .filter((p) => p.y != null);
  }, [active, summaries, metric, stat]);

  const unit = active ? summaries[active.runs[0]?.id]?.metrics?.[metric]?.unit ?? '' : '';

  const chartData = {
    labels: points.map((p) => p.x),
    datasets: [
      {
        label: `${metric} (${stat})`,
        data: points.map((p) => p.y),
        borderColor: '#76b900',
        backgroundColor: 'rgba(118,185,0,0.15)',
        borderWidth: 2,
        pointRadius: 4,
        pointHoverRadius: 7,
        tension: 0.2,
        fill: true,
      },
    ],
  };

  const chartOptions = {
    plugins: { legend: { display: false } },
    scales: {
      x: {
        grid: { color: CHART_THEME.grid },
        ticks: { color: CHART_THEME.tick, font: { size: 10 } },
        title: { display: true, text: 'variation (trial)', color: CHART_THEME.axisLabel, font: { size: 10 } },
      },
      y: {
        grid: { color: CHART_THEME.grid },
        ticks: { color: CHART_THEME.tick, font: { size: 10 } },
        title: { display: true, text: unit || metric, color: CHART_THEME.axisLabel, font: { size: 10 } },
      },
    },
  };

  return html`
    <div class="page">
      <div class="page-head">
        <h1 class="page-title">Sweeps</h1>
      </div>

      ${sweeps.length === 0
        ? html`<div class="card empty">No sweeps found — runs are grouped here when they carry a <code>sweep_id</code>.</div>`
        : html`
            <div class="sweep-chips">
              ${sweeps.map(
                (s) => html`
                  <button
                    key=${s.sweepId}
                    class=${'chip' + (s.sweepId === activeId ? ' active' : '')}
                    onClick=${() => setActiveId(s.sweepId)}
                  >
                    ${s.sweepId} <span class="dim">· ${s.runs.length}</span>
                  </button>
                `,
              )}
            </div>

            <div class="toolbar">
              <label class="sort-select">
                <span class="dim">metric</span>
                <select value=${metric} onChange=${(e) => setMetric(e.target.value)}>
                  ${metricTags.length === 0
                    ? html`<option value=${metric}>${prettyTag(metric)}</option>`
                    : metricTags.map((t) => html`<option value=${t}>${prettyTag(t)}</option>`)}
                </select>
              </label>
              <label class="sort-select">
                <span class="dim">stat</span>
                <select value=${stat} onChange=${(e) => setStat(e.target.value)}>
                  ${STATS.map((s) => html`<option value=${s}>${s}</option>`)}
                </select>
              </label>
              <span class="dim caption">${isLowerBetter(metric) ? '↓ lower is better' : '↑ higher is better'}</span>
            </div>

            <div class="card">
              <div class="card-title">${active?.sweepId} · ${prettyTag(metric)} (${stat}) across trials</div>
              ${points.length === 0
                ? html`<div class="empty">No data for this metric yet.</div>`
                : html`<${ChartWrapper} type="line" data=${chartData} options=${chartOptions} height=${320} />`}
            </div>

            <div class="card">
              <div class="card-title">Trials <span class="card-count">${active?.runs.length ?? 0}</span></div>
              <div class="table-scroll">
                <table class="data-table">
                  <thead>
                    <tr>
                      <th class="num">trial</th>
                      <th>Run</th>
                      <th class="num">${prettyTag(metric)} (${stat})</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    ${(active?.runs ?? []).map(
                      (r) => html`
                        <tr key=${r.id} class="clickable" onClick=${() => navigate('/runs/' + encodeURIComponent(r.id))}>
                          <td class="num dim">${r.trial ?? '—'}</td>
                          <td class="run-label">${r.label ?? r.id}</td>
                          <td class="num">${fmtMetric(statValue(summaries[r.id]?.metrics, metric, stat))}</td>
                          <td><span class=${'badge badge-' + (r.success ? 'ok' : 'fail')}>${r.success ? 'ok' : 'fail'}</span></td>
                        </tr>
                      `,
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          `}
    </div>
  `;
}
