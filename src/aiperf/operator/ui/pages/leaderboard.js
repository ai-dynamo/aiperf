// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { useState, useEffect, useMemo } from 'preact/hooks';
import { api } from '../lib/api.js';
import { palette } from '../lib/theme.js';
import { buildJobPath, navigate } from '../lib/router.js';
import { MetricSelector } from '../components/metric-selector.js';
import { applyJobFilters, extractCrossFacets, FILTER_NONE } from './compare-filters.js';
import { ChartWrapper } from '../components/chart-wrapper.js';
import { LoadingPanel } from '../components/spinner.js';
import { fmtNumber } from '../lib/format.js';

// Metrics where smaller values are better (latency-like). Used for the
// "lower = better" cue; ranking direction itself is owned by the API.
const LOWER_IS_BETTER_PREFIXES = ['request_latency', 'inter_token_latency', 'time_to_first_token', 'time_to_second_token'];

function isLowerBetter(metric) {
  if (!metric) return false;
  return LOWER_IS_BETTER_PREFIXES.some((p) => metric.startsWith(p));
}

const CHART_COLORS = [
  palette.mauve,
  palette.blue,
  palette.green,
  palette.peach,
  palette.pink,
  palette.teal,
  palette.sapphire,
  palette.yellow,
  palette.flamingo,
  palette.lavender,
];

function formatDate(iso) {
  if (!iso) return '---';
  return new Date(iso).toLocaleDateString([], { month: 'short', day: 'numeric', year: '2-digit' });
}

function formatValue(value, unit) {
  if (value == null) return '---';
  const formatted = typeof value === 'number' ? fmtNumber(value, 2) : value;
  return unit ? `${formatted} ${unit}` : String(formatted);
}

// Tighten generic API errors into something actionable. The api lib throws
// `API <status>: <body>` on HTTP errors; unknown errors still surface as-is.
function describeLoadError(raw) {
  const s = String(raw ?? '');
  if (/API 404/.test(s)) return 'leaderboard endpoint not found — results-server may be older than this UI build';
  if (/API 401|API 403/.test(s)) return 'no permission to read leaderboard — check RBAC for the results-server';
  if (/API 503|API 502|API 504/.test(s)) return 'results-server unreachable — try `kubectl -n aiperf-operator get pods -l app=results-server`';
  if (/Failed to fetch|NetworkError|ECONNREFUSED/i.test(s)) return 'network error reaching results-server — port-forward may have dropped';
  return s;
}

export function Leaderboard() {
  const [selected, setSelected] = useState({ metric: 'request_throughput', stat: 'avg' });
  const [nsFilter, setNsFilter] = useState(new Set());
  const [modelFilter, setModelFilter] = useState(new Set());
  const [endpointFilter, setEndpointFilter] = useState(new Set());
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    api
      .getLeaderboard(selected.metric, selected.stat, 1000)
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
  const rankableEntries = useMemo(
    () => entries.filter((entry) => entry.value != null),
    [entries],
  );

  // Cross-filter chips: each dimension's facet map is recomputed from
  // entries filtered by the OTHER active dimensions, so clicking one
  // narrows the rest to only what still co-occurs.
  const facets = useMemo(
    () => extractCrossFacets(rankableEntries, { nsFilter, modelFilter, endpointFilter, search: '' }),
    [rankableEntries, nsFilter, modelFilter, endpointFilter],
  );
  const filtered = useMemo(
    () => applyJobFilters(rankableEntries, { nsFilter, modelFilter, endpointFilter, search: '' }),
    [rankableEntries, nsFilter, modelFilter, endpointFilter],
  );
  const anyFilterActive = nsFilter.size + modelFilter.size + endpointFilter.size > 0;
  const clearAll = () => {
    setNsFilter(new Set());
    setModelFilter(new Set());
    setEndpointFilter(new Set());
  };
  const toggleFacet = (setter, value) => {
    setter((prev) => {
      const next = new Set(prev);
      if (next.has(value)) next.delete(value);
      else next.add(value);
      return next;
    });
  };

  const FACET_VISIBLE = 8;

  const renderFacetRow = (label, dim, facetMap, filterSet, setFilterFn) => {
    const allEntries = Array.from(facetMap.entries()).sort((a, b) => b[1] - a[1]);
    if (allEntries.length === 0) return null;
    const visible = allEntries.slice(0, FACET_VISIBLE);
    const overflow = allEntries.length - visible.length;
    return html`
      <div style="margin-bottom: var(--space-2)" data-testid=${'leaderboard-facet-' + dim}>
        <div style="font-size: var(--font-size-xs); color: var(--overlay0); margin-bottom: var(--space-1)">${label}</div>
        <div style="display: flex; flex-wrap: wrap; gap: var(--space-1)">
          ${visible.map(([value, count]) => {
            const on = filterSet.has(value);
            const display = value === FILTER_NONE ? '(none)' : value;
            return html`
              <span
                key=${value}
                onclick=${() => toggleFacet(setFilterFn, value)}
                onkeydown=${(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    toggleFacet(setFilterFn, value);
                  }
                }}
                role="button"
                tabindex="0"
                aria-pressed=${on}
                title=${value === FILTER_NONE ? '(no value)' : value}
                style=${'display: inline-flex; align-items: center; gap: var(--space-1); padding: var(--space-1) var(--space-2); border-radius: 999px; font-size: var(--font-size-xs); cursor: pointer; border: 1px solid; max-width: 100%; min-width: 0;'
                  + (on
                    ? ' background: var(--mauve)22; color: var(--mauve); border-color: var(--mauve);'
                    : ' background: transparent; color: var(--subtext0); border-color: var(--surface1);')}
              >
                <span style="font-family: var(--font-mono); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0">${display}</span>
                <span style="opacity: 0.6; flex-shrink: 0">· ${count}</span>
              </span>
            `;
          })}
          ${overflow > 0 && html`
            <span
              style="display: inline-flex; align-items: center; padding: var(--space-1) var(--space-2); border-radius: 999px; font-size: var(--font-size-xs); background: var(--surface0); color: var(--subtext0); border: 1px solid var(--surface1)"
              title=${'+' + overflow + ' more values not shown'}
            >+${overflow} more</span>
          `}
        </div>
      </div>
    `;
  };

  const unit = filtered[0]?.unit ?? '';

  const top10 = filtered.slice(0, 10);
  const chartData = {
    labels: top10.map((e) => e.job_id ?? ''),
    datasets: [
      {
        label: selected.metric,
        data: top10.map((e) => e.value),
        backgroundColor: top10.map((_, i) => CHART_COLORS[i % CHART_COLORS.length] + 'cc'),
        borderColor: top10.map((_, i) => CHART_COLORS[i % CHART_COLORS.length]),
        borderWidth: 1,
        // Cap bar thickness so a single-bar chart doesn't stretch to fill
        // the full canvas height, which looks broken.
        maxBarThickness: 28,
      },
    ],
  };

  const chartOptions = {
    indexAxis: 'y',
    plugins: {
      legend: { display: false },
    },
    scales: {
      x: {
        ticks: { color: palette.overlay0, font: { size: 11 } },
        grid: { color: palette.surface0 + '40' },
        title: {
          display: true,
          text: unit || selected.metric,
          color: palette.overlay1,
          font: { size: 11 },
        },
      },
      y: {
        ticks: { color: palette.overlay0, font: { size: 11 } },
        grid: { color: palette.surface0 + '40' },
      },
    },
  };

  return html`
    <div class="leaderboard" data-testid="page-leaderboard">
      <div class="section-header" style="margin-bottom: var(--space-4)">
        <span class="section-title">Leaderboard</span>
      </div>

      <!-- Controls -->
      <div class="card" style="margin-bottom: var(--space-4); display: flex; flex-direction: column; gap: var(--space-3)">
        <div style="display: flex; align-items: center; gap: var(--space-3); flex-wrap: wrap; justify-content: space-between">
          <${MetricSelector} value=${selected} onSelect=${setSelected} />
          ${anyFilterActive && html`
            <button
              type="button"
              onclick=${clearAll}
              style="background: transparent; border: 1px solid var(--surface1); color: var(--subtext0); border-radius: var(--radius-sm); padding: var(--space-1) var(--space-2); font-size: var(--font-size-xs); cursor: pointer"
              title="Clear all chip filters"
            >Clear filters</button>
          `}
        </div>
        ${renderFacetRow('Namespace', 'ns',       facets.ns,       nsFilter,       setNsFilter)}
        ${renderFacetRow('Model',     'model',    facets.model,    modelFilter,    setModelFilter)}
        ${renderFacetRow('Endpoint',  'endpoint', facets.endpoint, endpointFilter, setEndpointFilter)}
      </div>

      ${error && html`
        <div class="card" style="border-color: var(--error); color: var(--error); margin-bottom: var(--space-4)">
          Failed to load leaderboard: ${error}
        </div>
      `}

      ${loading && html`
        <div class="card" style="margin-bottom: var(--space-4)">
          <${LoadingPanel} label="Loading leaderboard…" testid="leaderboard-loading" />
        </div>
      `}

      ${!loading && !error && filtered.length === 0 && html`
        <div class="card empty-state" style="margin-bottom: var(--space-4)">
          ${entries.length === 0
            ? html`<p class="text-dim">No completed benchmarks yet. Submit an AIPerfJob and once it finishes it will appear here, ranked by your selected metric.</p>`
            : html`<p class="text-dim">No results match the current filters. Click selected chips above (or "Clear filters") to widen back to all ${entries.length} run${entries.length === 1 ? '' : 's'}.</p>`}
        </div>
      `}

      ${!loading && filtered.length > 0 && html`
        <!-- Bar chart -->
        <div class="card" style="margin-bottom: var(--space-4)">
          <div class="card-title" style="display: flex; align-items: baseline; gap: var(--space-3); flex-wrap: wrap">
            <span>Top ${top10.length} -- ${selected.metric} (${selected.stat})</span>
            <span style="font-size: var(--font-size-xs); color: var(--overlay0); font-weight: normal">
              ${isLowerBetter(selected.metric) ? '↓ lower = better' : '↑ higher = better'}${unit ? ' • ' + unit : ''}
            </span>
          </div>
          <${ChartWrapper} type="bar" data=${chartData} options=${chartOptions} height=${Math.max(200, top10.length * 32)} />
        </div>

        <!-- Ranked table -->
        <div class="card">
          <div class="card-title" style="display: flex; align-items: baseline; gap: var(--space-3); flex-wrap: wrap">
            <span>All Results</span>
            <span class="text-dim" style="font-size: var(--font-size-xs); font-weight: normal" data-testid="leaderboard-count">
              ${filtered.length === entries.length
                ? `${filtered.length} run${filtered.length === 1 ? '' : 's'}`
                : `${filtered.length} of ${entries.length} runs`}
            </span>
          </div>
          <div style="overflow-x: auto">
            <table style="width: 100%; border-collapse: collapse; font-size: var(--font-size-sm)">
              <thead>
                <tr style="color: var(--subtext0); border-bottom: 1px solid var(--surface1)">
                  <th style="text-align: right; padding: var(--space-2) var(--space-3)">#</th>
                  <th style="text-align: left; padding: var(--space-2) var(--space-3)">Job</th>
                  <th style="text-align: left; padding: var(--space-2) var(--space-3)">Namespace</th>
                  <th style="text-align: right; padding: var(--space-2) var(--space-3)">Value</th>
                  <th style="text-align: left; padding: var(--space-2) var(--space-3)">Model</th>
                  <th style="text-align: left; padding: var(--space-2) var(--space-3)">Endpoint</th>
                  <th style="text-align: left; padding: var(--space-2) var(--space-3)">Date</th>
                </tr>
              </thead>
              <tbody>
                ${filtered.map((entry, idx) => {
                  const rank = idx + 1;
                  const isTop3 = rank <= 3;
                  const rowColor = rank === 1
                    ? palette.yellow
                    : rank === 2
                    ? palette.subtext1
                    : rank === 3
                    ? palette.peach
                    : null;
                  const canNavigate = !!entry.job_id;
                  const goToJob = () => {
                    if (canNavigate) {
                      navigate(buildJobPath(entry));
                    }
                  };
                  const baseBg = isTop3 ? rowColor + '0a' : 'transparent';
                  const hoverBg = 'var(--surface0)';

                  return html`
                    <tr
                      key=${entry.job_id}
                      role="row"
                      tabindex=${canNavigate ? '0' : undefined}
                      onkeydown=${(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); goToJob(); } }}
                      onclick=${goToJob}
                      onmouseenter=${(e) => { if (canNavigate) e.currentTarget.style.background = hoverBg; }}
                      onmouseleave=${(e) => { if (canNavigate) e.currentTarget.style.background = baseBg; }}
                      title=${canNavigate ? 'View job details' : ''}
                      style=${'border-bottom: 1px solid var(--surface0); background: ' + baseBg + ';' + (canNavigate ? ' cursor: pointer;' : '')}
                    >
                      <td style=${'padding: var(--space-2) var(--space-3); text-align: right; font-weight: 600;' + (isTop3 ? ' color: ' + rowColor : ' color: var(--overlay0)')}>
                        ${rank}
                      </td>
                      <td style="padding: var(--space-2) var(--space-3); font-family: var(--font-mono); font-size: var(--font-size-xs)">
                        ${entry.job_id ?? '---'}
                      </td>
                      <td style="padding: var(--space-2) var(--space-3); color: var(--subtext0)">
                        ${entry.namespace ?? '---'}
                      </td>
                      <td style=${'padding: var(--space-2) var(--space-3); text-align: right; font-weight: 600;' + (isTop3 ? ' color: ' + rowColor : '')}>
                        ${formatValue(entry.value, entry.unit)}
                      </td>
                      <td style="padding: var(--space-2) var(--space-3); color: var(--subtext0)">
                        ${entry.model ?? '---'}
                      </td>
                      <td
                        title=${entry.endpoint ?? ''}
                        style="padding: var(--space-2) var(--space-3); color: var(--subtext0); font-size: var(--font-size-xs); max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap"
                      >
                        ${entry.endpoint ?? '---'}
                      </td>
                      <td style="padding: var(--space-2) var(--space-3); color: var(--overlay0)">
                        ${formatDate(entry.start_time)}
                      </td>
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
