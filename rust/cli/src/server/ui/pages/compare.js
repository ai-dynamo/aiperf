// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Compare page: a run picker (synced with the shared compare tray) plus, for
// the 2+ selected runs, per-metric grouped bar charts and a throughput-vs-
// latency Pareto scatter. Summaries are loaded lazily and memoized in the
// shared cache, tolerating individual per-run failures.

import { html } from 'htm/preact';
import { useState, useEffect, useMemo } from 'preact/hooks';
import { runs, compareSel, toggleCompare, loadSummaries } from '../lib/state.js';
import { HEADLINE, fmtMetric, prettyTag, isLowerBetter } from '../lib/format.js';
import { ChartWrapper, CHART_PALETTE, CHART_THEME } from '../components/chart-wrapper.js';
import { ParetoChart } from '../components/pareto-chart.js';

// Metrics charted as grouped bars (headline four + request throughput).
const COMPARE_TAGS = [...HEADLINE.map((h) => h.tag), 'request_throughput'];

function shortLabel(run) {
  return run?.label ?? run?.id ?? '';
}

/** avg for a tag from a summary, falling back to the run's headline scalar. */
function metricValue(entry, tag) {
  const m = entry.summary?.metrics?.[tag];
  if (m?.avg != null) return m.avg;
  return entry.summary?.headline?.[tag] ?? null;
}

function barOptions(unit) {
  return {
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: CHART_THEME.tooltipBg,
        titleColor: '#ececec',
        bodyColor: '#c0c0c8',
        borderColor: CHART_THEME.grid,
        borderWidth: 1,
      },
    },
    scales: {
      x: { grid: { display: false }, ticks: { color: CHART_THEME.tick, font: { size: 10 }, maxRotation: 30 } },
      y: {
        beginAtZero: true,
        grid: { color: CHART_THEME.grid },
        ticks: { color: CHART_THEME.tick, font: { size: 10 } },
        title: unit ? { display: true, text: unit, color: CHART_THEME.axisLabel, font: { size: 10 } } : undefined,
      },
    },
  };
}

export function Compare() {
  const allRuns = runs.value;
  const sel = compareSel.value;

  const [entries, setEntries] = useState([]); // [{ id, run, summary }]
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState(null);

  // Resolve selected ids -> summaries whenever the selection changes.
  useEffect(() => {
    let cancelled = false;
    if (sel.length === 0) {
      setEntries([]);
      return;
    }
    setLoading(true);
    setLoadError(null);
    loadSummaries(sel)
      .then((results) => {
        if (cancelled) return;
        const runById = new Map(allRuns.map((r) => [r.id, r]));
        const ok = [];
        for (const res of results) {
          if (!res) continue;
          ok.push({ id: res.id, run: runById.get(res.id) ?? res.summary.run ?? { id: res.id }, summary: res.summary });
        }
        setEntries(ok);
        if (ok.length < sel.length) {
          setLoadError(`${sel.length - ok.length} run(s) failed to load and were skipped.`);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [sel.join(','), allRuns.length]);

  // Per-metric grouped bar data: labels = run names, one bar per run.
  const barCharts = useMemo(() => {
    if (entries.length < 1) return [];
    return COMPARE_TAGS.map((tag) => {
      const values = entries.map((e) => metricValue(e, tag));
      if (values.every((v) => v == null)) return null;
      const unit =
        HEADLINE.find((h) => h.tag === tag)?.unit ?? entries.find((e) => e.summary?.metrics?.[tag]?.unit)?.summary?.metrics?.[tag]?.unit ?? '';
      return {
        tag,
        unit,
        data: {
          labels: entries.map(shortLabel),
          datasets: [
            {
              label: prettyTag(tag),
              data: values,
              backgroundColor: entries.map((_, i) => CHART_PALETTE[i % CHART_PALETTE.length] + 'cc'),
              borderColor: entries.map((_, i) => CHART_PALETTE[i % CHART_PALETTE.length]),
              borderWidth: 1,
              maxBarThickness: 48,
            },
          ],
        },
      };
    }).filter(Boolean);
  }, [entries]);

  // Pareto: output token throughput (x, larger better) vs request latency p95
  // (y, smaller better). One point per run.
  const paretoPoints = useMemo(
    () =>
      entries
        .map((e) => {
          const x = e.summary?.metrics?.output_token_throughput?.avg ?? e.summary?.headline?.output_token_throughput;
          const y =
            e.summary?.metrics?.request_latency?.percentiles?.p95 ??
            e.summary?.metrics?.request_latency?.avg ??
            e.summary?.headline?.request_latency;
          return { x, y, label: shortLabel(e) };
        })
        .filter((p) => p.x != null && p.y != null),
    [entries],
  );

  return html`
    <div class="page">
      <div class="page-head">
        <h1 class="page-title">Compare Runs</h1>
      </div>

      <div class="compare-layout">
        <div class="card run-picker">
          <div class="card-title">Select Runs <span class="card-count">${sel.length}</span></div>
          ${allRuns.length === 0
            ? html`<div class="empty">No runs to compare.</div>`
            : html`
                <div class="picker-list">
                  ${allRuns.map((r) => {
                    const checked = sel.includes(r.id);
                    return html`
                      <label key=${r.id} class=${'picker-row' + (checked ? ' sel' : '')}>
                        <input type="checkbox" checked=${checked} onChange=${() => toggleCompare(r.id)} />
                        <span class="picker-label">
                          <span class="run-label">${r.label ?? r.id}</span>
                          <span class="dim caption">${r.source}${r.success ? '' : ' · failed'}</span>
                        </span>
                      </label>
                    `;
                  })}
                </div>
                ${sel.length > 0 &&
                html`<button class="btn" style="margin-top:10px" onClick=${() => (compareSel.value = [])}>Clear selection</button>`}
              `}
        </div>

        <div class="compare-results">
          ${loadError && html`<div class="warn-strip">${loadError}</div>`}
          ${sel.length < 2
            ? html`<div class="card empty">Select 2 or more runs from the list to compare them.</div>`
            : loading && entries.length === 0
            ? html`<div class="card empty">Loading summaries…</div>`
            : html`
                <div class="card">
                  <div class="card-title">Throughput vs Latency · Pareto</div>
                  <${ParetoChart}
                    points=${paretoPoints}
                    xAxis=${{ label: 'Output Throughput', unit: 'tok/s', largerBetter: true }}
                    yAxis=${{ label: 'Request Latency p95', unit: 'ms', largerBetter: false }}
                    height=${360}
                  />
                </div>

                <div class="card">
                  <div class="card-title">Metric Comparison</div>
                  <div class="table-scroll">
                    <table class="data-table">
                      <thead>
                        <tr>
                          <th>Metric</th>
                          ${entries.map((e, i) => html`<th key=${e.id} class="num" style=${'color:' + CHART_PALETTE[i % CHART_PALETTE.length]}>${shortLabel(e)}</th>`)}
                        </tr>
                      </thead>
                      <tbody>
                        ${COMPARE_TAGS.map((tag) => {
                          const vals = entries.map((e) => metricValue(e, tag));
                          const finite = vals.filter((v) => v != null);
                          const best = finite.length
                            ? isLowerBetter(tag)
                              ? Math.min(...finite)
                              : Math.max(...finite)
                            : null;
                          return html`
                            <tr key=${tag}>
                              <td class="mono">${prettyTag(tag)}</td>
                              ${vals.map(
                                (v, i) => html`
                                  <td class=${'num' + (v != null && v === best ? ' best' : '')} key=${i}>
                                    ${fmtMetric(v)}
                                  </td>
                                `,
                              )}
                            </tr>
                          `;
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div class="chart-grid">
                  ${barCharts.map(
                    (c) => html`
                      <div class="card chart-card" key=${c.tag}>
                        <div class="card-title">${prettyTag(c.tag)}${c.unit ? html` <span class="dim caption">(${c.unit})</span>` : ''}</div>
                        <${ChartWrapper} type="bar" data=${c.data} options=${barOptions(c.unit)} height=${240} />
                      </div>
                    `,
                  )}
                </div>
              `}
        </div>
      </div>
    </div>
  `;
}
