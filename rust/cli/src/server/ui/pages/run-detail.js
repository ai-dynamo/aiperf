// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Rich single-run view. Pulls the projected ``/summary`` (headline + per-metric
// stats) for the KPI hero, charts, and table, and the raw ``native-v2.json``
// report for the run-metadata panel (transport / workload / endpoint profiles).
// Every widget degrades to an em-dash / empty state when a metric is absent.

import { html } from 'htm/preact';
import { useState, useEffect } from 'preact/hooks';
import { navigate } from '../lib/router.js';
import { api } from '../lib/api.js';
import { getSummary } from '../lib/state.js';
import { HEADLINE, fmtMetric, fmtInt, prettyTag } from '../lib/format.js';
import { KpiCard } from '../components/kpi-card.js';
import { MetricsTable } from '../components/metrics-table.js';
import { ChartWrapper, CHART_PALETTE, CHART_THEME } from '../components/chart-wrapper.js';

const PCTL_KEYS = ['p50', 'p90', 'p99'];
const LATENCY_TAGS = ['time_to_first_token', 'inter_token_latency', 'request_latency'];
const THROUGHPUT_TAGS = [
  'output_token_throughput',
  'request_throughput',
  'total_token_throughput',
  'output_token_throughput_per_user',
];

function baseChartOptions(xTitle, yTitle, indexAxis) {
  return {
    indexAxis: indexAxis ?? 'x',
    plugins: {
      legend: {
        display: true,
        labels: { color: CHART_THEME.axisLabel, boxWidth: 12, font: { size: 11 } },
      },
      tooltip: {
        backgroundColor: CHART_THEME.tooltipBg,
        titleColor: '#ececec',
        bodyColor: '#c0c0c8',
        borderColor: CHART_THEME.grid,
        borderWidth: 1,
      },
    },
    scales: {
      x: {
        grid: { color: CHART_THEME.grid },
        ticks: { color: CHART_THEME.tick, font: { size: 10 } },
        title: xTitle ? { display: true, text: xTitle, color: CHART_THEME.axisLabel, font: { size: 10 } } : undefined,
      },
      y: {
        beginAtZero: true,
        grid: { color: CHART_THEME.grid },
        ticks: { color: CHART_THEME.tick, font: { size: 10 } },
        title: yTitle ? { display: true, text: yTitle, color: CHART_THEME.axisLabel, font: { size: 10 } } : undefined,
      },
    },
  };
}

/** Grouped bars: category = percentile, one dataset per latency metric. */
function latencyPercentileData(metrics) {
  const present = LATENCY_TAGS.filter((tag) => metrics[tag]?.percentiles);
  if (present.length === 0) return null;
  const datasets = present.map((tag, i) => ({
    label: prettyTag(tag),
    data: PCTL_KEYS.map((p) => metrics[tag].percentiles?.[p] ?? null),
    backgroundColor: CHART_PALETTE[i % CHART_PALETTE.length] + 'cc',
    borderColor: CHART_PALETTE[i % CHART_PALETTE.length],
    borderWidth: 1,
  }));
  return { labels: PCTL_KEYS, datasets };
}

/** Horizontal bars of throughput averages (whichever tags are present). */
function throughputData(metrics) {
  const present = THROUGHPUT_TAGS.filter((tag) => metrics[tag]?.avg != null);
  if (present.length === 0) return null;
  return {
    labels: present.map(prettyTag),
    datasets: [
      {
        label: 'avg',
        data: present.map((tag) => metrics[tag].avg),
        backgroundColor: present.map((_, i) => CHART_PALETTE[i % CHART_PALETTE.length] + 'cc'),
        borderColor: present.map((_, i) => CHART_PALETTE[i % CHART_PALETTE.length]),
        borderWidth: 1,
        maxBarThickness: 30,
      },
    ],
  };
}

function MetaRow({ k, v }) {
  return html`<div class="meta-row"><span class="meta-k">${k}</span><span class="meta-v mono">${v}</span></div>`;
}

function RunMeta({ report, summaryRun }) {
  const run = report?.run ?? {};
  const profiles = Array.isArray(run.endpoint_profiles) ? run.endpoint_profiles : [];
  return html`
    <div class="card">
      <div class="card-title">Run Metadata</div>
      <div class="meta-grid">
        <${MetaRow} k="id" v=${summaryRun?.id ?? '—'} />
        <${MetaRow} k="schema" v=${report?.schema_version ?? '—'} />
        <${MetaRow} k="transport" v=${run.transport ?? '—'} />
        <${MetaRow} k="workload" v=${run.workload ?? '—'} />
        ${summaryRun?.sweep_id && html`<${MetaRow} k="sweep" v=${summaryRun.sweep_id} />`}
        ${summaryRun?.trial != null && html`<${MetaRow} k="trial" v=${String(summaryRun.trial)} />`}
      </div>
      ${profiles.length > 0 &&
      html`
        <div class="card-subtitle">Endpoint Profiles (${profiles.length})</div>
        <div class="table-scroll">
          <table class="data-table">
            <thead>
              <tr><th>#</th><th>profile</th><th>endpoint</th><th>model</th></tr>
            </thead>
            <tbody>
              ${profiles.map(
                (p, i) => html`
                  <tr key=${i}>
                    <td class="dim">${i}</td>
                    <td class="mono">${p.profile_id ?? p.name ?? '—'}</td>
                    <td class="mono">${p.endpoint_id ?? p.endpoint ?? p.type ?? p.url ?? '—'}</td>
                    <td class="mono">${p.model ?? p.model_name ?? run.model ?? '—'}</td>
                  </tr>
                `,
              )}
            </tbody>
          </table>
        </div>
      `}
    </div>
  `;
}

export function RunDetail({ id }) {
  const [summary, setSummary] = useState(null);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setSummary(null);
    setReport(null);
    // Summary is required; the raw report is best-effort (only powers the
    // metadata panel), so a report failure must not blank the whole page.
    getSummary(id)
      .then((s) => {
        if (!cancelled) setSummary(s);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    api
      .run(id)
      .then((r) => {
        if (!cancelled) setReport(r);
      })
      .catch(() => {
        /* metadata panel simply omits transport/workload */
      });
    return () => {
      cancelled = true;
    };
  }, [id]);

  const backLink = html`
    <button class="btn back-btn" onClick=${() => navigate('/')}>← Runs</button>
  `;

  if (loading) {
    return html`<div class="page">${backLink}<div class="empty">Loading run…</div></div>`;
  }
  if (error) {
    return html`<div class="page">${backLink}<div class="empty error-text">Failed to load run: ${error}</div></div>`;
  }
  if (!summary) {
    return html`<div class="page">${backLink}<div class="empty">No summary available for this run.</div></div>`;
  }

  const runInfo = summary.run ?? {};
  const headline = summary.headline ?? {};
  const metrics = summary.metrics ?? {};
  const latData = latencyPercentileData(metrics);
  const tputData = throughputData(metrics);

  return html`
    <div class="page">
      <div class="page-head detail-head">
        <div class="detail-title-block">
          ${backLink}
          <h1 class="page-title">${runInfo.label ?? id}</h1>
          <div class="badge-row">
            <span class=${'badge badge-' + (runInfo.source === 'session' ? 'session' : 'disk')}>
              ${runInfo.source ?? 'disk'}
            </span>
            <span class=${'badge badge-' + (runInfo.success ? 'ok' : 'fail')}>
              ${runInfo.success ? 'ok' : 'fail'}
            </span>
            ${runInfo.sweep_id && html`<span class="badge badge-neutral">sweep ${runInfo.sweep_id}</span>`}
          </div>
        </div>
        <div class="dim caption artifact-dir" title=${runInfo.artifact_dir ?? ''}>
          <code>${runInfo.artifact_dir ?? ''}</code>
        </div>
      </div>

      <div class="kpi-grid">
        ${HEADLINE.map(
          (h) => html`
            <${KpiCard}
              key=${h.tag}
              label=${h.label}
              value=${fmtMetric(headline[h.tag])}
              unit=${headline[h.tag] == null ? '' : h.unit}
              icon=${h.icon}
              sub=${metrics[h.tag]?.count != null ? fmtInt(metrics[h.tag].count) + ' samples' : null}
            />
          `,
        )}
      </div>

      <div class="chart-row">
        <div class="card chart-card">
          <div class="card-title">Latency Percentiles</div>
          ${latData
            ? html`<${ChartWrapper}
                type="bar"
                data=${latData}
                options=${baseChartOptions('percentile', 'ms')}
                height=${300}
              />`
            : html`<div class="empty">No latency percentile data.</div>`}
        </div>
        <div class="card chart-card">
          <div class="card-title">Throughput</div>
          ${tputData
            ? html`<${ChartWrapper}
                type="bar"
                data=${tputData}
                options=${baseChartOptions('', 'per second', 'y')}
                height=${300}
              />`
            : html`<div class="empty">No throughput data.</div>`}
        </div>
      </div>

      <${RunMeta} report=${report} summaryRun=${runInfo} />

      <div class="card">
        <div class="card-title">All Metrics <span class="card-count">${Object.keys(metrics).length}</span></div>
        <${MetricsTable} metrics=${metrics} />
      </div>
    </div>
  `;
}
