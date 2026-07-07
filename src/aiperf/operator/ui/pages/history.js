// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { useState, useEffect } from 'preact/hooks';
import { api } from '../lib/api.js';
import { palette } from '../lib/theme.js';
import { buildJobPath, navigate, query, setQuery } from '../lib/router.js';
import { MetricSelector } from '../components/metric-selector.js';
import { ChartWrapper } from '../components/chart-wrapper.js';
import { NsPill, ModelPill } from '../components/pills.js';
import { LoadingPanel } from '../components/spinner.js';
import { fmtNumber } from '../lib/format.js';

// Backend caps history responses at this many rows. When ``entries.length``
// equals the cap we surface a "may be truncated" hint so the user doesn't
// silently assume they're seeing the full record.
const HISTORY_TRUNCATION_HINT = 10000;

function formatDate(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleDateString([], {
    month: 'short',
    day: 'numeric',
    year: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function formatDateShort(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

function formatNum(v) {
  if (v == null) return '\u2014';
  return typeof v === 'number' ? fmtNumber(v, 3) : String(v);
}

// Tighten generic API errors into something actionable. The api lib throws
// `API <status>: <body>` on HTTP errors; unknown errors still surface as-is.
function describeLoadError(raw) {
  const s = String(raw ?? '');
  if (/API 404/.test(s)) return 'history endpoint not found \u2014 results-server may be older than this UI build';
  if (/API 401|API 403/.test(s)) return 'no permission to read history \u2014 check RBAC for the results-server';
  if (/API 503|API 502|API 504/.test(s)) return 'results-server unreachable \u2014 try `kubectl -n aiperf-operator get pods -l app=results-server`';
  if (/Failed to fetch|NetworkError|ECONNREFUSED/i.test(s)) return 'network error reaching results-server \u2014 port-forward may have dropped';
  return s;
}

export function History() {
  const [selected, setSelected] = useState({ metric: 'request_throughput', stat: 'avg' });
  const q = query.value;
  const ns = q.ns ?? '';
  const urlModel = q.model ?? '';
  const urlEndpoint = q.endpoint ?? '';
  const [model, setModel] = useState(urlModel);
  const [endpoint, setEndpoint] = useState(urlEndpoint);
  useEffect(() => { if (model !== urlModel) setModel(urlModel); /* eslint-disable-line */ }, [urlModel]);
  useEffect(() => { if (endpoint !== urlEndpoint) setEndpoint(urlEndpoint); /* eslint-disable-line */ }, [urlEndpoint]);
  useEffect(() => {
    const t = setTimeout(() => { if (model !== urlModel) setQuery({ model }); }, 200);
    return () => clearTimeout(t);
  }, [model, urlModel]);
  useEffect(() => {
    const t = setTimeout(() => { if (endpoint !== urlEndpoint) setQuery({ endpoint }); }, 200);
    return () => clearTimeout(t);
  }, [endpoint, urlEndpoint]);

  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    api
      .getHistory(selected.metric, selected.stat)
      .then((resp) => {
        if (!cancelled) setData(resp);
      })
      .catch((err) => {
        if (!cancelled) setError(describeLoadError(err?.message ?? err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selected.metric, selected.stat]);

  const entries = data?.entries ?? [];

  // Sort by start_time ascending so the line chart and table are stable
  // across poll refreshes (API may return entries in arbitrary order).
  const filtered = entries
    .filter((e) => {
      if (model && !(e.model ?? '').toLowerCase().includes(model.toLowerCase())) return false;
      if (endpoint && !(e.endpoint ?? '').toLowerCase().includes(endpoint.toLowerCase())) return false;
      if (ns && (e.namespace ?? '') !== ns) return false;
      return true;
    })
    .slice()
    .sort((a, b) => {
      const ta = a.start_time ? Date.parse(a.start_time) : 0;
      const tb = b.start_time ? Date.parse(b.start_time) : 0;
      if (ta !== tb) return ta - tb;
      // Tiebreak on job_id so equal timestamps don't reshuffle.
      return String(a.job_id ?? '').localeCompare(String(b.job_id ?? ''));
    });

  const unit = filtered[0]?.unit ?? '';

  // Chart.js renders a `line` chart with a single point as just a dot —
  // easy to miss at the default radius. Bump radius so it's visible, and
  // surface a hint below the chart so the user knows nothing is broken.
  const isSinglePoint = filtered.length === 1;

  const chartData = {
    labels: filtered.map((e) => formatDateShort(e.start_time)),
    datasets: [
      {
        label: `${selected.metric} (${selected.stat})${unit ? ' [' + unit + ']' : ''}`,
        data: filtered.map((e) => e.value ?? null),
        borderColor: palette.blue,
        backgroundColor: palette.blue + '22',
        fill: true,
        tension: 0.3,
        pointRadius: isSinglePoint ? 8 : 4,
        pointHoverRadius: isSinglePoint ? 10 : 6,
        pointBackgroundColor: palette.blue,
        borderWidth: 2,
      },
    ],
  };

  const chartOptions = {
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          title: (items) => {
            const idx = items[0]?.dataIndex;
            if (idx == null) return '';
            const e = filtered[idx];
            return e ? `${e.job_id ?? ''} — ${formatDate(e.start_time)}` : '';
          },
          label: (item) => {
            const e = filtered[item.dataIndex];
            const val = item.raw;
            return `${formatNum(val)}${unit ? ' ' + unit : ''}`;
          },
        },
      },
    },
    scales: {
      x: {
        ticks: { color: palette.overlay0, font: { size: 11 }, maxTicksLimit: 12 },
        grid: { color: palette.surface0 + '40' },
      },
      y: {
        ticks: { color: palette.overlay0, font: { size: 11 } },
        grid: { color: palette.surface0 + '40' },
        title: {
          display: true,
          text: unit || selected.metric,
          color: palette.overlay1,
          font: { size: 11 },
        },
      },
    },
  };

  return html`
    <div class="history-page" data-testid="page-history">
      <div class="section-header" style="margin-bottom: var(--space-4)">
        <span class="section-title">History</span>
      </div>

      <!-- Controls -->
      <div class="card" style="margin-bottom: var(--space-4); display: flex; align-items: center; gap: var(--space-6); flex-wrap: wrap">
        <${MetricSelector} value=${selected} onSelect=${setSelected} />
        <div style="display: flex; gap: var(--space-3); align-items: center; flex-wrap: wrap">
          ${ns && html`
            <span
              class="meta-pill meta-pill--clickable"
              style=${'background:' + palette.teal + '22;color:' + palette.teal + ';border-color:' + palette.teal + '55'}
              title=${'Namespace filter: ' + ns + ' (click to clear)'}
              onclick=${() => setQuery({ ns: undefined })}
              data-testid="ns-filter-chip"
            >
              <span class="meta-pill__prefix">ns</span>${ns}
              <span style="margin-left:4px;opacity:0.7">×</span>
            </span>
          `}
          <div style="display: flex; align-items: center; gap: var(--space-2)">
            <label class="metric-selector-label">Model</label>
            <div style="position: relative; display: inline-block">
              <input
                class="metric-selector-select"
                type="text"
                placeholder="Filter by model…"
                value=${model}
                oninput=${(e) => setModel(e.target.value)}
                style=${'min-width: 160px;' + (model ? ' padding-right: 22px;' : '')}
              />
              ${model && html`
                <button
                  type="button"
                  onclick=${() => setModel('')}
                  aria-label="Clear model filter"
                  title="Clear"
                  style="position: absolute; right: 4px; top: 50%; transform: translateY(-50%); width: 18px; height: 18px; padding: 0; border: 0; background: transparent; color: var(--overlay0); cursor: pointer; font-size: 14px; line-height: 1; border-radius: 50%"
                >×</button>
              `}
            </div>
          </div>
          <div style="display: flex; align-items: center; gap: var(--space-2)">
            <label class="metric-selector-label">Endpoint</label>
            <div style="position: relative; display: inline-block">
              <input
                class="metric-selector-select"
                type="text"
                placeholder="Filter by endpoint…"
                value=${endpoint}
                oninput=${(e) => setEndpoint(e.target.value)}
                style=${'min-width: 160px;' + (endpoint ? ' padding-right: 22px;' : '')}
              />
              ${endpoint && html`
                <button
                  type="button"
                  onclick=${() => setEndpoint('')}
                  aria-label="Clear endpoint filter"
                  title="Clear"
                  style="position: absolute; right: 4px; top: 50%; transform: translateY(-50%); width: 18px; height: 18px; padding: 0; border: 0; background: transparent; color: var(--overlay0); cursor: pointer; font-size: 14px; line-height: 1; border-radius: 50%"
                >×</button>
              `}
            </div>
          </div>
        </div>
      </div>

      ${error && html`
        <div class="card" style="border-color: var(--error); color: var(--error); margin-bottom: var(--space-4)">
          Failed to load history: ${error}
        </div>
      `}

      ${loading && html`
        <div class="card" style="margin-bottom: var(--space-4)">
          <${LoadingPanel} label="Loading history…" testid="history-loading" />
        </div>
      `}

      ${!loading && !error && filtered.length === 0 && html`
        <div class="card empty-state" style="margin-bottom: var(--space-4)">
          ${entries.length === 0
            ? html`<p class="text-dim">No completed benchmarks yet. As AIPerfJobs finish, their ${selected.metric} (${selected.stat}) will plot here over time.</p>`
            : html`<p class="text-dim">No data points match the current filters. ${entries.length} run${entries.length === 1 ? '' : 's'} are hidden — clear the model/endpoint${ns ? '/namespace' : ''} filter to see them.</p>`}
        </div>
      `}

      ${!loading && filtered.length > 0 && html`
        <!-- Line chart -->
        <div class="card" style="margin-bottom: var(--space-4)">
          <div class="card-title" style="display: flex; align-items: baseline; gap: var(--space-3); flex-wrap: wrap">
            <span>${selected.metric} (${selected.stat}) over time</span>
            ${(model || endpoint || ns) && html`
              <span class="text-dim" style="font-size: var(--font-size-xs); font-weight: normal" data-testid="history-chart-filtered">
                (filtered${ns ? ' · ns=' + ns : ''}${model ? ' · model=' + model : ''}${endpoint ? ' · endpoint=' + endpoint : ''})
              </span>
            `}
          </div>
          <${ChartWrapper} type="line" data=${chartData} options=${chartOptions} height=${300} />
          ${isSinglePoint && html`
            <div class="text-dim" style="margin-top: var(--space-2); font-size: var(--font-size-xs)">
              Only one data point — line chart will trend once a second matching run finishes.
            </div>
          `}
        </div>

        <!-- Data table -->
        <div class="card">
          <div class="card-title" style="display: flex; align-items: baseline; gap: var(--space-3); flex-wrap: wrap">
            <span>Data Points</span>
            <span class="text-dim" style="font-size: var(--font-size-xs); font-weight: normal" data-testid="history-count">
              ${filtered.length === entries.length
                ? `${filtered.length} run${filtered.length === 1 ? '' : 's'}`
                : `${filtered.length} of ${entries.length} runs`}
              ${entries.length >= HISTORY_TRUNCATION_HINT
                ? html`<span style=${'margin-left: var(--space-2); padding: 1px 6px; border: 1px solid ' + palette.peach + '55; border-radius: 6px; color: ' + palette.peach}
                    title="The history endpoint caps responses at ${HISTORY_TRUNCATION_HINT}. Older runs may be missing — narrow by model/endpoint to refine."
                  >may be truncated</span>`
                : ''}
            </span>
          </div>
          <div style="overflow-x: auto">
            <table style="width: 100%; border-collapse: collapse; font-size: var(--font-size-sm)">
              <thead>
                <tr style="color: var(--subtext0); border-bottom: 1px solid var(--surface1)">
                  <th style="text-align: left; padding: var(--space-2) var(--space-3)">Job</th>
                  <th style="text-align: left; padding: var(--space-2) var(--space-3)">Namespace</th>
                  <th style="text-align: right; padding: var(--space-2) var(--space-3)">Value</th>
                  <th style="text-align: left; padding: var(--space-2) var(--space-3)">Model</th>
                  <th style="text-align: left; padding: var(--space-2) var(--space-3)">Date</th>
                </tr>
              </thead>
              <tbody>
                ${filtered.map((entry) => html`
                  <tr key=${entry.job_id + entry.start_time} style="border-bottom: 1px solid var(--surface0)">
                    <td style="padding: var(--space-2) var(--space-3)">
                      <span
                        onclick=${() => navigate(buildJobPath(entry))}
                        style="color: var(--blue); cursor: pointer; font-family: var(--font-mono); font-size: var(--font-size-xs)"
                      >
                        ${entry.job_id ?? '—'}
                      </span>
                    </td>
                    <td style="padding: var(--space-2) var(--space-3)">
                      <${NsPill} ns=${entry.namespace} onClick=${nsClicked => setQuery({ ns: nsClicked })} testId=${'history-row-ns-' + (entry.namespace ?? '')} />
                    </td>
                    <td style="text-align: right; padding: var(--space-2) var(--space-3); font-weight: 600">
                      ${formatNum(entry.value)}${entry.unit ? html` <span style="color: var(--overlay0); font-weight: normal">${entry.unit}</span>` : ''}
                    </td>
                    <td style="padding: var(--space-2) var(--space-3); color: var(--subtext0)">
                      ${entry.model
                        ? html`<${ModelPill} model=${entry.model} onClick=${m => setModel(m)} testId=${'history-row-model-' + entry.model} />`
                        : html`<span class="text-dim">—</span>`}
                    </td>
                    <td style="padding: var(--space-2) var(--space-3); color: var(--overlay0)">
                      ${formatDate(entry.start_time)}
                    </td>
                  </tr>
                `)}
              </tbody>
            </table>
          </div>
        </div>
      `}
    </div>
  `;
}
