// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Leaderboard: pick any metric present across the loaded runs plus a stat
// (avg/min/max/p50/p90/p99), then rank every run. Renders a top-N horizontal
// bar plus a full ranked table. Ranking direction follows the latency-aware
// ``isLowerBetter`` heuristic. All run summaries are loaded once and memoized.

import { html } from 'htm/preact';
import { useState, useEffect, useMemo } from 'preact/hooks';
import { runs } from '../lib/state.js';
import { getSummary } from '../lib/state.js';
import { fmtMetric, prettyTag, isLowerBetter } from '../lib/format.js';
import { navigate } from '../lib/router.js';
import { ChartWrapper, CHART_PALETTE, CHART_THEME } from '../components/chart-wrapper.js';

const STATS = ['avg', 'min', 'max', 'p50', 'p90', 'p99', 'p95'];
const TOP_N = 15;

/** Extract a (tag, stat) value from a projected summary metrics map. */
function statValue(metrics, tag, stat) {
  const m = metrics?.[tag];
  if (!m) return null;
  if (stat.startsWith('p')) return m.percentiles?.[stat] ?? null;
  return m[stat] ?? null;
}

export function Leaderboard() {
  const allRuns = runs.value;
  const [summaries, setSummaries] = useState({}); // id -> summary
  const [loading, setLoading] = useState(true);
  const [metric, setMetric] = useState('output_token_throughput');
  const [stat, setStat] = useState('avg');

  useEffect(() => {
    let cancelled = false;
    const ids = allRuns.map((r) => r.id);
    if (ids.length === 0) {
      setLoading(false);
      return;
    }
    setLoading(true);
    Promise.allSettled(ids.map((id) => getSummary(id))).then((results) => {
      if (cancelled) return;
      const map = {};
      results.forEach((res, i) => {
        if (res.status === 'fulfilled') map[ids[i]] = res.value;
      });
      setSummaries(map);
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [allRuns.map((r) => r.id).join(',')]);

  // Union of metric tags across all loaded summaries.
  const metricTags = useMemo(() => {
    const set = new Set();
    for (const s of Object.values(summaries)) {
      for (const tag of Object.keys(s?.metrics ?? {})) set.add(tag);
    }
    return [...set].sort();
  }, [summaries]);

  // Keep the selected metric valid as data loads in.
  useEffect(() => {
    if (metricTags.length > 0 && !metricTags.includes(metric)) {
      setMetric(metricTags.includes('output_token_throughput') ? 'output_token_throughput' : metricTags[0]);
    }
  }, [metricTags]);

  const lower = isLowerBetter(metric);

  const ranked = useMemo(() => {
    const rows = allRuns
      .map((r) => ({ run: r, value: statValue(summaries[r.id]?.metrics, metric, stat), unit: summaries[r.id]?.metrics?.[metric]?.unit ?? '' }))
      .filter((row) => row.value != null);
    rows.sort((a, b) => (lower ? a.value - b.value : b.value - a.value));
    return rows;
  }, [allRuns, summaries, metric, stat, lower]);

  const unit = ranked[0]?.unit ?? '';
  const top = ranked.slice(0, TOP_N);

  const chartData = {
    labels: top.map((r) => r.run.label ?? r.run.id),
    datasets: [
      {
        label: `${metric} (${stat})`,
        data: top.map((r) => r.value),
        backgroundColor: top.map((_, i) => CHART_PALETTE[i % CHART_PALETTE.length] + 'cc'),
        borderColor: top.map((_, i) => CHART_PALETTE[i % CHART_PALETTE.length]),
        borderWidth: 1,
        maxBarThickness: 24,
      },
    ],
  };

  const chartOptions = {
    indexAxis: 'y',
    plugins: { legend: { display: false } },
    scales: {
      x: {
        grid: { color: CHART_THEME.grid },
        ticks: { color: CHART_THEME.tick, font: { size: 10 } },
        title: { display: true, text: unit || metric, color: CHART_THEME.axisLabel, font: { size: 10 } },
      },
      y: { grid: { display: false }, ticks: { color: CHART_THEME.tick, font: { size: 10 } } },
    },
  };

  return html`
    <div class="page">
      <div class="page-head">
        <h1 class="page-title">Leaderboard</h1>
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
        <span class="dim caption">${lower ? '↓ lower is better' : '↑ higher is better'}</span>
      </div>

      ${loading && ranked.length === 0
        ? html`<div class="card empty">Loading run summaries…</div>`
        : ranked.length === 0
        ? html`<div class="card empty">No runs report <code>${metric}</code> (${stat}). Try another metric or stat.</div>`
        : html`
            <div class="card">
              <div class="card-title">Top ${top.length} · ${prettyTag(metric)} (${stat})</div>
              <${ChartWrapper} type="bar" data=${chartData} options=${chartOptions} height=${Math.max(220, top.length * 28)} />
            </div>

            <div class="card">
              <div class="card-title">Ranked · ${ranked.length} run${ranked.length === 1 ? '' : 's'}</div>
              <div class="table-scroll">
                <table class="data-table">
                  <thead>
                    <tr>
                      <th class="num">#</th>
                      <th>Run</th>
                      <th>Source</th>
                      <th class="num">${prettyTag(metric)} (${stat})</th>
                      <th>Unit</th>
                    </tr>
                  </thead>
                  <tbody>
                    ${ranked.map((row, i) => {
                      const rank = i + 1;
                      const medal = rank === 1 ? 'gold' : rank === 2 ? 'silver' : rank === 3 ? 'bronze' : '';
                      return html`
                        <tr
                          key=${row.run.id}
                          class="clickable"
                          onClick=${() => navigate('/runs/' + encodeURIComponent(row.run.id))}
                        >
                          <td class=${'num rank ' + medal}>${rank}</td>
                          <td class="run-label">${row.run.label ?? row.run.id}</td>
                          <td>
                            <span class=${'badge badge-' + (row.run.source === 'session' ? 'session' : 'disk')}>${row.run.source ?? 'disk'}</span>
                          </td>
                          <td class="num strong">${fmtMetric(row.value)}</td>
                          <td class="dim">${row.unit}</td>
                        </tr>
                      `;
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          `}
    </div>
  `;
}
