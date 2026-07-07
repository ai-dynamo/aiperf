// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { useState, useEffect, useRef } from 'preact/hooks';
import { api, DASHBOARD_MUTATIONS_DISABLED_MESSAGE, DASHBOARD_MUTATIONS_ENABLED, poll } from '../lib/api.js';
import { openJobWs } from '../lib/job-ws.js';
import { buildRunSelectorRows } from '../lib/run-selector.js';
import { phaseColor, colors, palette } from '../lib/theme.js';
import { deriveJobRunState } from './job-detail-state.js';
import { navigate, replaceRoute } from '../lib/router.js';
import { KpiCard } from '../components/kpi-card.js';
import { RealtimeKpiGrid } from '../components/realtime-kpi-grid.js';
import { ChartWrapper } from '../components/chart-wrapper.js';
import { PhaseBar } from '../components/phase-bar.js';
import { RecordProcessing } from '../components/record-processing.js';
import { Conditions } from '../components/conditions.js';
import { PodsBar } from '../components/pods-bar.js';
import { DiagnosticsPanel } from '../components/diagnostics-panel.js';
import { LatencyTimelineChart } from '../components/latency-timeline-chart.js';
import { EpochSelector } from '../components/epoch-selector.js';
import { NsPill, ModelPill } from '../components/pills.js';
import { RelativeTime } from '../components/time.js';
import { LoadingPanel, Spinner } from '../components/spinner.js';
import { jobs as jobsSignal, freshness, clearFreshnessSource } from '../lib/state.js';
import { FreshnessPill, StaleBanner } from '../components/freshness.js';
import { fmtNumber, fmtInt, fmtThroughput, fmtBytes } from '../lib/format.js';
import { ServerMetricsSection } from '../components/server-metrics/index.js';
import { RelaunchButton, redactConfigForYaml } from '../components/relaunch-button.js';
import { ArtifactsCard } from '../components/artifacts-card.js';

// Keep secret-like config keys grep-visible where display redaction is applied:
// api_key, apiKey, authorization, bearerToken, client_secret, password, secret, secretRef, token.
const MAX_CHART_POINTS = 60;

// Stable module-scope options for the streaming live-throughput chart.
// Defining these inside the component would create a new object literal
// every poll — even though ChartWrapper diffs by JSON fingerprint, the
// stringify is wasted work and re-applying options retriggers Chart.js
// layout. ``animation: false`` is critical: this is a real-time chart,
// and the default 300ms tween makes the whole panel look like it's
// refreshing on every sample. Latency-distribution / one-shot charts
// keep their animation; only this streaming one disables it.
const LIVE_THROUGHPUT_OPTIONS = {
  animation: false,
  plugins: { legend: { display: false } },
  scales: {
    x: {
      ticks: { color: palette.overlay0, maxTicksLimit: 6, font: { size: 10 } },
      grid: { color: palette.surface0 },
    },
    y: {
      ticks: { color: palette.overlay0, font: { size: 10 } },
      grid: { color: palette.surface0 },
      title: { display: true, text: 'tok/s', color: palette.overlay1, font: { size: 10 } },
    },
  },
};

function formatDuration(ms) {
  if (ms == null) return null;
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  const h = Math.floor(m / 60);
  if (h > 0) return `${h}h ${m % 60}m ${s % 60}s`;
  if (m > 0) return `${m}m ${s % 60}s`;
  return `${s}s`;
}

function extractSummary(data) {
  // data is the full API response: {job: {...}, status: {...}, pods: [...]}
  const status = data?.status ?? {};
  return status.liveSummary ?? status.summary ?? null;
}

function fmtNum(val, decimals = 1) {
  if (val == null) return '---';
  return fmtNumber(val, decimals);
}

// Metrics table: column set and group definitions.
//
// METRIC_COLUMNS is the full numeric column list rendered in every group's
// table header. Each row's `cols` whitelist gates which cells render data
// vs `---` so noisy aggregates (counts, totals) don't pretend to have
// percentiles. Auto-discovery rows in the "Other Metrics" tail group bypass
// the whitelist and show every column where the value is non-null.
const METRIC_COLUMNS = ['avg', 'std', 'p1', 'p10', 'p50', 'p90', 'p95', 'p99', 'min', 'max'];

const METRIC_COL_TITLES = {
  avg: 'Arithmetic mean across all requests',
  std: 'Standard deviation across observations',
  p1: '1st percentile — best-case (only 1% of requests faster/below)',
  p10: '10th percentile — 10% of requests at or below this value',
  p50: '50th percentile (median) — half of requests at or below this value',
  p90: '90th percentile — 90% of requests at or below this value',
  p95: '95th percentile — 95% of requests at or below this value',
  p99: '99th percentile — 99% of requests at or below this value (tail latency)',
  min: 'Minimum observed value',
  max: 'Maximum observed value',
};

const FULL_PERCENTILES = ['avg', 'std', 'p1', 'p10', 'p50', 'p90', 'p95', 'p99', 'min', 'max'];

const METRIC_GROUPS = [
  {
    label: 'Throughput',
    color: palette.blue,
    rows: [
      { key: 'request_throughput', label: 'Request Throughput', cols: ['avg'] },
      { key: 'output_token_throughput', label: 'Output Token Throughput', cols: ['avg'] },
      { key: 'total_token_throughput', label: 'Total Token Throughput', cols: ['avg'] },
      { key: 'goodput', label: 'Goodput', cols: ['avg'] },
      { key: 'output_token_throughput_per_user', label: 'Output Token Throughput per User', cols: FULL_PERCENTILES },
      { key: 'e2e_output_token_throughput', label: 'E2E Output Token Throughput', cols: ['avg'] },
      { key: 'prefill_throughput_per_user', label: 'Prefill Throughput per User', cols: FULL_PERCENTILES },
    ],
  },
  {
    label: 'Latency',
    color: palette.peach,
    rows: [
      { key: 'request_latency', label: 'Request Latency', cols: FULL_PERCENTILES },
      { key: 'time_to_first_token', label: 'Time to First Token', cols: FULL_PERCENTILES },
      { key: 'inter_token_latency', label: 'Inter-Token Latency', cols: FULL_PERCENTILES },
      { key: 'time_to_second_token', label: 'Time to Second Token', cols: FULL_PERCENTILES },
      { key: 'inter_chunk_latency', label: 'Inter-Chunk Latency', cols: FULL_PERCENTILES },
      { key: 'time_to_first_output_token', label: 'Time to First Output Token', cols: FULL_PERCENTILES },
      { key: 'image_latency', label: 'Image Latency', cols: FULL_PERCENTILES },
    ],
  },
  {
    label: 'Tokens',
    color: palette.mauve,
    rows: [
      { key: 'usage_prompt_tokens', label: 'Usage Prompt Tokens', cols: FULL_PERCENTILES },
      { key: 'usage_completion_tokens', label: 'Usage Completion Tokens', cols: FULL_PERCENTILES },
      { key: 'usage_total_tokens', label: 'Usage Total Tokens', cols: FULL_PERCENTILES },
      { key: 'reasoning_token_count', label: 'Reasoning Tokens', cols: FULL_PERCENTILES },
      { key: 'output_token_count', label: 'Output Tokens', cols: FULL_PERCENTILES },
    ],
  },
  {
    label: 'Sequence Lengths',
    color: palette.teal,
    rows: [
      { key: 'input_sequence_length', label: 'Input Sequence Length', cols: FULL_PERCENTILES },
      { key: 'output_sequence_length', label: 'Output Sequence Length', cols: FULL_PERCENTILES },
      { key: 'osl_mismatch_diff_pct', label: 'OSL Mismatch (diff %)', cols: FULL_PERCENTILES },
      { key: 'error_isl', label: 'Error ISL', cols: FULL_PERCENTILES },
    ],
  },
  {
    label: 'Counts & Totals',
    color: palette.amber,
    rows: [
      { key: 'request_count', label: 'Request Count', cols: ['avg'] },
      { key: 'good_request_count', label: 'Good Request Count', cols: ['avg'] },
      { key: 'error_request_count', label: 'Error Request Count', cols: ['avg'] },
      { key: 'total_output_tokens', label: 'Total Output Tokens', cols: ['avg'] },
      { key: 'total_isl', label: 'Total ISL', cols: ['avg'] },
      { key: 'total_osl', label: 'Total OSL', cols: ['avg'] },
      { key: 'total_error_isl', label: 'Total Error ISL', cols: ['avg'] },
      { key: 'total_usage_prompt_tokens', label: 'Total Usage Prompt Tokens', cols: ['avg'] },
      { key: 'total_usage_completion_tokens', label: 'Total Usage Completion Tokens', cols: ['avg'] },
      { key: 'total_usage_total_tokens', label: 'Total Usage Total Tokens', cols: ['avg'] },
      { key: 'total_reasoning_tokens', label: 'Total Reasoning Tokens', cols: ['avg'] },
      { key: 'benchmark_duration', label: 'Benchmark Duration', cols: ['avg'] },
    ],
  },
  {
    label: 'HTTP',
    color: palette.pink,
    rows: [
      { key: 'http_req_duration', label: 'HTTP Request Duration', cols: FULL_PERCENTILES },
      { key: 'http_req_total', label: 'HTTP Request Total', cols: FULL_PERCENTILES },
      { key: 'http_req_waiting', label: 'HTTP Waiting (TTFB)', cols: FULL_PERCENTILES },
      { key: 'http_req_connecting', label: 'HTTP Connecting', cols: FULL_PERCENTILES },
      { key: 'http_req_sending', label: 'HTTP Sending', cols: FULL_PERCENTILES },
      { key: 'http_req_receiving', label: 'HTTP Receiving', cols: FULL_PERCENTILES },
      { key: 'http_req_blocked', label: 'HTTP Blocked', cols: FULL_PERCENTILES },
      { key: 'http_req_dns_lookup', label: 'HTTP DNS Lookup', cols: FULL_PERCENTILES },
      { key: 'http_req_connection_overhead', label: 'HTTP Connection Overhead', cols: FULL_PERCENTILES },
      { key: 'http_req_data_sent', label: 'HTTP Data Sent', cols: FULL_PERCENTILES },
      { key: 'http_req_data_received', label: 'HTTP Data Received', cols: FULL_PERCENTILES },
      { key: 'http_req_chunks_sent', label: 'HTTP Chunks Sent', cols: FULL_PERCENTILES },
      { key: 'http_req_chunks_received', label: 'HTTP Chunks Received', cols: FULL_PERCENTILES },
      { key: 'http_req_connection_reused', label: 'HTTP Connection Reused', cols: FULL_PERCENTILES },
    ],
  },
  {
    label: 'Vision',
    color: palette.green,
    rows: [
      { key: 'num_images', label: 'Images per Request', cols: FULL_PERCENTILES },
      { key: 'image_throughput', label: 'Image Throughput', cols: FULL_PERCENTILES },
      { key: 'video_inference_time', label: 'Video Inference Time', cols: FULL_PERCENTILES },
      { key: 'video_peak_memory', label: 'Video Peak Memory', cols: FULL_PERCENTILES },
    ],
  },
];

// Tags carrying MetricFlags.INTERNAL or MetricFlags.EXPERIMENTAL in the
// metric registry. These are deliberately omitted from the curated groups
// and also filtered out of the auto-discovery tail so that internal
// scaffolding metrics (timestamps used to derive other metrics) and
// not-yet-stable experimental ones don't appear in the user-facing UI.
// Sourced from `MetricRegistry.all_classes()` filtered by
// `flags & (INTERNAL | EXPERIMENTAL)`.
const EXCLUDED_KEYS = new Set([
  'credit_drop_latency',
  'max_response_timestamp',
  'min_request_timestamp',
  'requested_osl',
  'stream_setup_latency',
  'stream_prefill_latency',
  'thinking_efficiency',
  'overall_thinking_efficiency',
]);

// Tags claimed by curated groups; the auto-discovery tail subtracts these
// from the full results key set so each metric appears at most once.
const CURATED_KEYS = new Set(
  METRIC_GROUPS.flatMap(g => g.rows.map(r => r.key)),
);

function isMetricStruct(v) {
  // A metric entry is an object carrying at least one stat field. Filters
  // out scalars (error_rate is a bare number) and meta-structs that don't
  // belong in a percentile table.
  if (v == null || typeof v !== 'object' || Array.isArray(v)) return false;
  return v.avg != null || v.p50 != null || v.sum != null || v.count != null;
}

function prettifyTag(tag) {
  return tag
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase());
}

function buildOtherMetricsRows(results) {
  const rows = [];
  for (const [key, value] of Object.entries(results ?? {})) {
    if (CURATED_KEYS.has(key)) continue;
    if (EXCLUDED_KEYS.has(key)) continue;
    if (!isMetricStruct(value)) continue;
    // Show every column where the metric actually has data; auto-discovery
    // doesn't know what's meaningful so it just surfaces what's there.
    const cols = METRIC_COLUMNS.filter(c => value[c] != null);
    if (cols.length === 0) continue;
    rows.push({ key, label: prettifyTag(key), cols });
  }
  rows.sort((a, b) => a.key.localeCompare(b.key));
  return rows;
}

function MetricsTable({ results }) {
  const [collapsed, setCollapsed] = useState({});

  function toggleGroup(label) {
    setCollapsed(prev => ({ ...prev, [label]: !prev[label] }));
  }

  const otherRows = buildOtherMetricsRows(results);
  const allGroups = otherRows.length > 0
    ? [...METRIC_GROUPS, { label: 'Other Metrics', color: palette.overlay1, rows: otherRows }]
    : METRIC_GROUPS;

  return html`
    <div class="card" style="margin-top: var(--space-4)">
      <div class="card-title">Full Metrics Breakdown</div>
      ${allGroups.map(group => {
        const visibleRows = group.rows.filter(row => results[row.key] != null);
        if (visibleRows.length === 0) return null;
        const isOpen = !collapsed[group.label];
        return html`
          <div key=${group.label} style="margin-bottom: var(--space-3)">
            <div
              onclick=${() => toggleGroup(group.label)}
              style=${'display: flex; align-items: center; gap: var(--space-2); padding: var(--space-2) var(--space-3); background: ' + group.color + '18; border-radius: var(--radius-sm); cursor: pointer; user-select: none; border-left: 3px solid ' + group.color}
            >
              <span style=${'color: ' + group.color + '; font-weight: 600; font-size: var(--font-size-sm)'}>${group.label}</span>
              <span class="text-dim" style="font-size: var(--font-size-xs); margin-left: auto">${isOpen ? '\u25B2' : '\u25BC'}</span>
            </div>
            ${isOpen && html`
              <div style="overflow-x: auto">
                <table style="width: 100%; border-collapse: collapse; font-size: var(--font-size-sm); margin-top: var(--space-1)">
                  <thead>
                    <tr>
                      <th style=${'text-align: left; padding: var(--space-2) var(--space-3); color: ' + palette.overlay1 + '; font-weight: 500; font-size: var(--font-size-xs); border-bottom: 1px solid ' + palette.surface0}>Metric</th>
                      <th style=${'text-align: right; padding: var(--space-2) var(--space-3); color: ' + palette.overlay1 + '; font-weight: 500; font-size: var(--font-size-xs); border-bottom: 1px solid ' + palette.surface0}>Unit</th>
                      ${METRIC_COLUMNS.map(col => html`
                        <th key=${col} title=${METRIC_COL_TITLES[col]} style=${'text-align: right; padding: var(--space-2) var(--space-3); color: ' + palette.overlay1 + '; font-weight: 500; font-size: var(--font-size-xs); border-bottom: 1px solid ' + palette.surface0 + '; cursor: help'}>${col}</th>
                      `)}
                    </tr>
                  </thead>
                  <tbody>
                    ${visibleRows.map((row, i) => {
                      const m = results[row.key];
                      if (!m) return null;
                      const bg = i % 2 === 0 ? palette.base : palette.mantle;
                      return html`
                        <tr key=${row.key} style=${'background: ' + bg}>
                          <td style=${'padding: var(--space-2) var(--space-3); color: ' + palette.text}>${row.label}</td>
                          <td style=${'padding: var(--space-2) var(--space-3); text-align: right; color: ' + palette.overlay0 + '; font-size: var(--font-size-xs)'}>${m.unit ?? ''}</td>
                          ${METRIC_COLUMNS.map(col => {
                            const val = m[col];
                            const shown = row.cols.includes(col);
                            return html`
                              <td key=${col} style=${'padding: var(--space-2) var(--space-3); text-align: right; color: ' + (shown && val != null ? palette.text : palette.overlay0)}>
                                ${shown && val != null ? fmtNum(val) : '---'}
                              </td>
                            `;
                          })}
                        </tr>
                      `;
                    })}
                  </tbody>
                </table>
              </div>
            `}
          </div>
        `;
      })}
    </div>
  `;
}

function LatencyPercentileChart({ results }) {
  const lat = results?.request_latency;
  if (!lat) return null;

  const percentiles = ['p1', 'p5', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99'];
  const labels = [];
  const values = [];
  for (const p of percentiles) {
    if (lat[p] != null) {
      labels.push(p);
      values.push(lat[p]);
    }
  }
  if (values.length === 0) return null;

  const chartData = {
    labels,
    datasets: [
      {
        label: 'Latency (ms)',
        data: values,
        backgroundColor: [
          palette.green + 'cc',
          palette.teal + 'cc',
          palette.sapphire + 'cc',
          palette.blue + 'cc',
          palette.lavender + 'cc',
          palette.mauve + 'cc',
          palette.peach + 'cc',
          palette.red + 'cc',
        ],
        borderColor: [
          palette.green,
          palette.teal,
          palette.sapphire,
          palette.blue,
          palette.lavender,
          palette.mauve,
          palette.peach,
          palette.red,
        ],
        borderWidth: 1,
        borderRadius: 3,
      },
    ],
  };

  const chartOptions = {
    indexAxis: 'y',
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: ctx => ` ${fmtNumber(ctx.parsed.x, 1)} ms`,
        },
      },
    },
    scales: {
      x: {
        ticks: { color: palette.overlay0, font: { size: 10 } },
        grid: { color: palette.surface0 },
        title: { display: true, text: 'Latency (ms)', color: palette.overlay1, font: { size: 10 } },
      },
      y: {
        ticks: { color: palette.overlay0, font: { size: 11 } },
        grid: { color: palette.surface0 },
      },
    },
  };

  return html`
    <div class="card" style="margin-top: var(--space-4)">
      <div class="card-title">Request Latency Percentiles</div>
      <${ChartWrapper} type="bar" data=${chartData} options=${chartOptions} height=${220} />
    </div>
  `;
}

// Feature 3: Concurrency vs Throughput chart
function ConcurrencyThroughputChart({ status }) {
  // Look for phase-level metrics that indicate different concurrency levels
  const phases = status?.phases ?? {};
  const phaseResults = status?.results?.phases ?? status?.results?.phase_results ?? null;

  // Try to extract concurrency/throughput pairs from phases
  const points = [];

  if (phaseResults && typeof phaseResults === 'object') {
    for (const [name, data] of Object.entries(phaseResults)) {
      const conc = data.concurrency ?? data.virtual_users ?? null;
      const tps = data.throughput_rps ?? data.request_throughput?.avg ?? null;
      if (conc != null && tps != null) {
        points.push({ concurrency: conc, throughput: tps, name });
      }
    }
  }

  // Also try phases dict with embedded metrics
  if (points.length === 0) {
    for (const [name, data] of Object.entries(phases)) {
      const conc = data.concurrency ?? data.virtualUsers ?? null;
      const tps = data.throughputRps ?? data.throughput_rps ?? null;
      if (conc != null && tps != null) {
        points.push({ concurrency: conc, throughput: tps, name });
      }
    }
  }

  if (points.length < 2) return null;

  // Sort by concurrency
  points.sort((a, b) => a.concurrency - b.concurrency);

  const chartData = {
    labels: points.map(p => String(p.concurrency)),
    datasets: [{
      label: 'Throughput (req/s)',
      data: points.map(p => p.throughput),
      borderColor: palette.blue,
      backgroundColor: palette.blue + '22',
      fill: true,
      tension: 0.3,
      pointRadius: 5,
      pointBackgroundColor: palette.blue,
      borderWidth: 2,
    }],
  };

  const chartOptions = {
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: ctx => ` ${fmtThroughput(ctx.parsed.y)} req/s at concurrency ${ctx.label}`,
        },
      },
    },
    scales: {
      x: {
        title: { display: true, text: 'Concurrency', color: palette.overlay1, font: { size: 11 } },
        ticks: { color: palette.overlay0, font: { size: 10 } },
        grid: { color: palette.surface0 + '60' },
      },
      y: {
        title: { display: true, text: 'Throughput (req/s)', color: palette.overlay1, font: { size: 11 } },
        ticks: { color: palette.overlay0, font: { size: 10 } },
        grid: { color: palette.surface0 + '60' },
      },
    },
  };

  return html`
    <div class="card" style="margin-top: var(--space-4)">
      <div class="card-title">Concurrency vs Throughput</div>
      <${ChartWrapper} type="line" data=${chartData} options=${chartOptions} height=${220} />
    </div>
  `;
}

// Feature 4: ISL Distribution Histogram
function ISLDistributionChart({ results }) {
  const isl = results?.input_sequence_length;
  if (!isl) return null;

  // Build a distribution visualization from available percentiles
  const percentiles = ['p1', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99'];
  const labels = [];
  const values = [];
  for (const p of percentiles) {
    if (isl[p] != null) {
      labels.push(p);
      values.push(isl[p]);
    }
  }

  if (values.length < 2) return null;

  const chartData = {
    labels,
    datasets: [{
      label: 'Input Sequence Length (tokens)',
      data: values,
      backgroundColor: palette.teal + '88',
      borderColor: palette.teal,
      borderWidth: 1,
      borderRadius: 3,
    }],
  };

  const chartOptions = {
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: ctx => ` ${fmtInt(ctx.parsed.y)} tokens`,
        },
      },
    },
    scales: {
      x: {
        ticks: { color: palette.overlay0, font: { size: 10 } },
        grid: { color: palette.surface0 + '60' },
        title: { display: true, text: 'Percentile', color: palette.overlay1, font: { size: 10 } },
      },
      y: {
        ticks: { color: palette.overlay0, font: { size: 10 } },
        grid: { color: palette.surface0 + '60' },
        title: { display: true, text: 'Tokens', color: palette.overlay1, font: { size: 10 } },
      },
    },
  };

  return html`
    <div class="card" style="margin-top: var(--space-4)">
      <div class="card-title">Input Sequence Length Distribution</div>
      <${ChartWrapper} type="bar" data=${chartData} options=${chartOptions} height=${200} />
    </div>
  `;
}

// Feature 5: Token Efficiency Card
function TokenEfficiencyCard({ results, info }) {
  const outputTps = results?.output_token_throughput?.avg ?? null;
  if (outputTps == null) return null;

  const gpuCount = info?.gpuCount ?? info?.gpu_count ?? info?.gpus ?? null;
  const efficiency = gpuCount != null && gpuCount > 0 ? outputTps / gpuCount : null;

  return html`
    <${KpiCard}
      label=${efficiency != null ? 'Token Efficiency (per GPU)' : 'Output Token Throughput'}
      value=${efficiency != null ? fmtNum(efficiency, 1) : fmtNum(outputTps, 0)}
      unit="tok/s"
      color=${palette.yellow}
    />
  `;
}

// Feature 6: SLA Compliance Indicator
//
// Thresholds come from the user-declared SLOs on the AIPerfJob CR
// (``spec.benchmark.slos`` per ``src/aiperf/config/_models_core.py`` —
// ``SLOsConfig = dict[str, float]`` keyed by metric tag, value in the
// metric's display unit). If no SLOs were declared, this card renders
// nothing rather than show invented thresholds.
//
// Direction (smaller-is-better for latency, larger-is-better for throughput)
// mirrors ``MetricFlags.LARGER_IS_BETTER`` on each metric class. We hard-code
// the small set of throughput-side tags realistic for ``--goodput`` because
// the registry isn't reachable from the browser; unknown tags default to the
// latency-style ``<=`` comparison.
const LARGER_IS_BETTER_SLO_TAGS = new Set([
  'output_token_throughput',
  'output_token_throughput_per_user',
  'request_throughput',
  'total_token_throughput',
  'e2e_output_token_throughput',
  'prefill_throughput_per_user',
]);

const SLO_PRETTY_LABEL = {
  request_latency: 'Request Latency',
  time_to_first_token: 'TTFT',
  time_to_second_token: 'TTST',
  inter_token_latency: 'ITL',
  output_token_throughput: 'Output Token Throughput',
  output_token_throughput_per_user: 'Per-User Output Throughput',
  request_throughput: 'Request Throughput',
  total_token_throughput: 'Total Token Throughput',
  e2e_output_token_throughput: 'E2E Output Throughput',
  prefill_throughput_per_user: 'Prefill Per-User Throughput',
};

const SLO_UNIT = {
  request_latency: 'ms',
  time_to_first_token: 'ms',
  time_to_second_token: 'ms',
  inter_token_latency: 'ms',
  output_token_throughput: 'tok/s',
  output_token_throughput_per_user: 'tok/s',
  request_throughput: 'req/s',
  total_token_throughput: 'tok/s',
  e2e_output_token_throughput: 'tok/s',
  prefill_throughput_per_user: 'tok/s',
};

function SLACompliance({ results, summary, config }) {
  const slos =
    config?.spec?.benchmark?.slos
    ?? config?.spec?.slos
    ?? null;
  if (!slos || typeof slos !== 'object') return null;

  const sloEntries = Object.entries(slos).filter(
    ([, threshold]) => threshold != null && isFinite(Number(threshold)),
  );
  if (sloEntries.length === 0) return null;

  const checks = [];

  for (const [tag, rawThreshold] of sloEntries) {
    const stats = results?.[tag] ?? summary?.[tag] ?? null;
    if (stats == null || typeof stats !== 'object') continue;

    const threshold = Number(rawThreshold);
    const largerIsBetter = LARGER_IS_BETTER_SLO_TAGS.has(tag);
    // Latency SLOs are reported as p99 (worst-tail compliance);
    // throughput SLOs as avg (the headline rate users typically target).
    const statName = largerIsBetter ? 'avg' : 'p99';
    const value = stats[statName] ?? stats.avg ?? null;
    if (value == null || !isFinite(Number(value))) continue;

    const numValue = Number(value);
    const pass = largerIsBetter
      ? numValue >= threshold
      : numValue <= threshold;
    const op = largerIsBetter ? '>=' : '<=';
    const unit = SLO_UNIT[tag] ?? '';
    const pretty = SLO_PRETTY_LABEL[tag] ?? tag;
    const digits = unit === 'req/s' || unit === 'tok/s' ? 1 : 0;

    checks.push({
      label: `${pretty} ${statName} ${op} ${fmtNumber(threshold, digits)}${unit ? ' ' + unit : ''}`,
      pass,
      value: `${fmtNumber(numValue, digits)}${unit ? ' ' + unit : ''}`,
    });
  }

  // Overall goodput pass-rate, when the run actually computed it.
  const goodCount = results?.good_request_count?.avg
    ?? summary?.good_request_count?.avg
    ?? null;
  const totalCount = results?.request_count?.avg
    ?? summary?.request_count?.avg
    ?? null;
  if (goodCount != null && totalCount != null && totalCount > 0) {
    const pct = (goodCount / totalCount) * 100;
    checks.unshift({
      label: 'Goodput (all SLOs per request)',
      pass: goodCount >= totalCount,
      value: `${fmtNumber(pct, 1)}% (${fmtInt(goodCount)}/${fmtInt(totalCount)})`,
    });
  }

  if (checks.length === 0) return null;

  return html`
    <div class="card" style="margin-top: var(--space-4)">
      <div class="card-title">SLA Compliance</div>
      <div style="display: flex; gap: var(--space-4); flex-wrap: wrap">
        ${checks.map(check => html`
          <div
            key=${check.label}
            style=${'display: flex; align-items: center; gap: var(--space-2); padding: var(--space-2) var(--space-3); border-radius: var(--radius-sm); background: ' + (check.pass ? palette.green + '12' : palette.red + '12') + '; border: 1px solid ' + (check.pass ? palette.green + '30' : palette.red + '30')}
          >
            <span style=${'font-size: var(--font-size-base); color: ' + (check.pass ? palette.green : palette.red)}>
              ${check.pass ? '\u2713' : '\u2717'}
            </span>
            <div style="display: flex; flex-direction: column">
              <span style=${'font-size: var(--font-size-xs); color: ' + palette.subtext0}>${check.label}</span>
              <span style=${'font-size: var(--font-size-sm); font-weight: 600; color: ' + (check.pass ? palette.green : palette.red)}>${check.value}</span>
            </div>
          </div>
        `)}
      </div>
    </div>
  `;
}

// Job Configuration Section
function JobConfigSection({ config, namespace, name }) {
  const [showSpec, setShowSpec] = useState(false);

  if (!config) return null;

  const spec = config.spec ?? {};
  const benchmark = spec.benchmark ?? spec;
  const redactedConfig = redactConfigForYaml(config);
  const redactedSpec = redactedConfig.spec ?? {};
  const configKind = config.kind ?? (spec.sweep ? 'AIPerfSweep' : 'AIPerfJob');

  // Extract key config items for the summary row
  const endpoint = benchmark.endpoint ?? {};
  const models = benchmark.models ?? {};
  const phases = benchmark.phases ?? {};
  const datasets = benchmark.datasets ?? {};
  const runtime = benchmark.runtime ?? {};

  const modelItems = models.items ?? models.modelNames ?? [];
  const modelName = Array.isArray(modelItems) && modelItems.length > 0
    ? (typeof modelItems[0] === 'object' ? modelItems[0].name : modelItems[0])
    : null;
  const urls = endpoint.urls ?? endpoint.url ?? [];
  const endpointUrl = Array.isArray(urls) ? urls[0] : urls;
  const streaming = endpoint.streaming ?? null;
  const endpointType = endpoint.type ?? null;

  // Phase summary
  const phaseNames = Array.isArray(phases) ? phases.map(p => p.name ?? 'unnamed') : Object.keys(phases);

  // Config key-value pairs for the summary grid
  const summaryItems = [];
  if (modelName) summaryItems.push({ label: 'Model', value: modelName });
  if (endpointUrl) summaryItems.push({ label: 'Endpoint', value: endpointUrl });
  if (endpointType) summaryItems.push({ label: 'API Type', value: endpointType });
  if (streaming != null) summaryItems.push({ label: 'Streaming', value: streaming ? 'Yes' : 'No' });
  if (phaseNames.length > 0) summaryItems.push({ label: 'Phases', value: phaseNames.join(', ') });

  // Extract concurrency/request info from phases
  const phaseList = Array.isArray(phases) ? phases : Object.values(phases);
  for (const p of phaseList) {
    const pName = p.name ?? '';
    if (p.concurrency != null) {
      summaryItems.push({ label: `${pName} Concurrency`, value: fmtInt(p.concurrency) });
    }
    const rc = p.request_count ?? p.requestCount ?? p.num_requests ?? null;
    if (rc != null) {
      summaryItems.push({ label: `${pName} Requests`, value: fmtInt(rc) });
    }
  }

  // Image
  const image = spec.image ?? null;
  if (image) summaryItems.push({ label: 'Image', value: image });

  // Workers
  const workers = spec.workers ?? spec.numWorkers ?? runtime.workers ?? null;
  if (workers != null) summaryItems.push({ label: 'Workers', value: fmtInt(workers) });

  const displayedManifest = {
    apiVersion: redactedConfig.apiVersion ?? 'aiperf.nvidia.com/v1alpha1',
    kind: configKind,
    metadata: { name: name ?? 'aiperfjob', namespace: namespace ?? 'default' },
    spec: redactedSpec,
  };
  const displayedYaml = serializeYaml(displayedManifest) + '\n';

  return html`
    <div class="card" style="margin-top: var(--space-4)">
      <div style=${'display: flex; align-items: center; justify-content: space-between'}>
        <div class="card-title" style="margin: 0">Job Configuration</div>
        <button type="button"
          onclick=${() => setShowSpec(true)}
          data-testid="job-config-view-spec"
          style=${'background: ' + palette.teal + '22; color: ' + palette.teal + '; border: 1px solid ' + palette.teal + '44; padding: var(--space-1) var(--space-3); border-radius: var(--radius-md); cursor: pointer; font-size: var(--font-size-xs)'}
        >View YAML · ${config.source ?? 'spec'}</button>
      </div>

      ${summaryItems.length > 0 && html`
        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: var(--space-3); margin-top: var(--space-3)">
          ${summaryItems.map(item => html`
            <div key=${item.label} style="display: flex; flex-direction: column; gap: var(--space-1)">
              <span style=${'font-size: var(--font-size-xs); color: ' + palette.overlay0 + '; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600'}>${item.label}</span>
              <span style=${'font-size: var(--font-size-sm); color: ' + palette.text + '; font-weight: 500; word-break: break-all'}>${item.value}</span>
            </div>
          `)}
        </div>
      `}

      ${showSpec && html`
        <${SpecViewerModal}
          filename=${(name ?? 'aiperfjob') + '.yaml'}
          content=${displayedYaml}
          onClose=${() => setShowSpec(false)}
        />
      `}
    </div>
  `;
}

// Feature 8: Run Metadata
function RunMetadata({ status, results, info }) {
  const startTime = info?.startTime ?? status?.startTime;
  const endTime = status?.completionTime ?? status?.endTime;
  let duration = null;
  if (startTime && endTime) {
    duration = formatDuration(new Date(endTime).getTime() - new Date(startTime).getTime());
  }

  const totalRequests = status?.results?.total_requests
    ?? status?.results?.totalRequests
    ?? status?.summary?.total_requests
    ?? null;

  const isl = results?.input_sequence_length;
  const osl = results?.output_sequence_length;
  const islMean = isl?.avg ?? null;
  const oslMean = osl?.avg ?? null;

  const streaming = info?.streaming ?? status?.config?.streaming ?? null;

  const items = [];
  if (duration) items.push({ label: 'Duration', value: duration });
  if (totalRequests != null) items.push({ label: 'Total Requests', value: fmtInt(totalRequests) });
  if (islMean != null) items.push({ label: 'Avg ISL', value: `${fmtInt(islMean)} tokens` });
  if (oslMean != null) items.push({ label: 'Avg OSL', value: `${fmtInt(oslMean)} tokens` });
  if (streaming != null) items.push({ label: 'Streaming', value: streaming ? 'Yes' : 'No' });

  if (items.length === 0) return null;

  return html`
    <div class="card" style="margin-top: var(--space-4)">
      <div class="card-title">Run Metadata</div>
      <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: var(--space-3)">
        ${items.map(item => html`
          <div key=${item.label} style="display: flex; flex-direction: column; gap: var(--space-1)">
            <span style=${'font-size: var(--font-size-xs); color: ' + palette.overlay0 + '; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600'}>${item.label}</span>
            <span style=${'font-size: var(--font-size-sm); color: ' + palette.text + '; font-weight: 500'}>${item.value}</span>
          </div>
        `)}
      </div>
    </div>
  `;
}

// --- Shared Modal Chrome ---

// Shared modal chrome styles
const BACKDROP_STYLE = [
  'position: fixed; inset: 0; z-index: 1000;',
  'background: ' + palette.base + 'cc;',
  'backdrop-filter: blur(4px);',
  'display: flex; align-items: center; justify-content: center;',
].join(' ');

const MODAL_BASE_STYLE = [
  'background: ' + palette.mantle + ';',
  'border: 1px solid ' + palette.surface0 + ';',
  'border-radius: var(--radius-md);',
  'max-height: 80vh;',
  'display: flex; flex-direction: column;',
  'overflow: hidden;',
].join(' ');

// Default modal sizing (used by the spec/YAML viewer).
const MODAL_STYLE = MODAL_BASE_STYLE + ' max-width: 80vw; width: 900px;';

function ModalChrome({ filename, onCopy, onDownload, onClose, copyLabel, children }) {
  return html`
    <div style=${BACKDROP_STYLE} onclick=${e => { if (e.target === e.currentTarget) onClose(); }}>
      <div style=${MODAL_STYLE}>
        <div style=${'display: flex; align-items: center; justify-content: space-between; padding: var(--space-3) var(--space-4); border-bottom: 1px solid ' + palette.surface0 + '; flex-shrink: 0'}>
          <span style=${'font-size: var(--font-size-sm); font-weight: 600; color: ' + palette.text + '; font-family: monospace'}>${filename}</span>
          <div style="display: flex; gap: var(--space-2); align-items: center">
            ${onCopy && html`
              <button type="button"
                onclick=${onCopy}
                style=${'background: ' + palette.teal + '22; color: ' + palette.teal + '; border: 1px solid ' + palette.teal + '44; padding: var(--space-1) var(--space-3); border-radius: var(--radius-md); cursor: pointer; font-size: var(--font-size-xs)'}
              >${copyLabel ?? 'Copy'}</button>
            `}
            <button type="button"
              onclick=${onDownload}
              style=${'background: ' + palette.blue + '22; color: ' + palette.blue + '; border: 1px solid ' + palette.blue + '44; padding: var(--space-1) var(--space-3); border-radius: var(--radius-md); cursor: pointer; font-size: var(--font-size-xs)'}
            >Download</button>
            <button type="button"
              onclick=${onClose}
              style=${'background: transparent; color: ' + palette.overlay1 + '; border: 1px solid ' + palette.surface1 + '; padding: var(--space-1) var(--space-2); border-radius: var(--radius-md); cursor: pointer; font-size: var(--font-size-sm); line-height: 1'}
            >\u00d7</button>
          </div>
        </div>
        <div style="overflow: auto; flex: 1; padding: var(--space-4)">
          ${children}
        </div>
      </div>
    </div>
  `;
}


// Minimal YAML emitter for AIPerfJob CR specs. Handles strings, numbers,
// bools, null, lists, objects. Quotes strings that contain YAML-significant
// characters; not a full emitter.
function serializeYaml(obj, indent = 0) {
  const pad = ' '.repeat(indent);
  if (obj === null || obj === undefined) return 'null';
  if (typeof obj === 'boolean') return obj ? 'true' : 'false';
  if (typeof obj === 'number') return String(obj);
  if (typeof obj === 'string') {
    if (obj === '') return "''";
    if (obj.includes('\n')) {
      const blockPad = ' '.repeat(indent + 2);
      return `|\n${obj.split('\n').map(line => blockPad + line).join('\n')}`;
    }
    if (/^[\w./@\-+]+$/.test(obj) && !/^(true|false|null|~)$/i.test(obj) && !/^-?\d+(\.\d+)?$/.test(obj)) {
      return obj;
    }
    return "'" + obj.replace(/'/g, "''") + "'";
  }
  if (Array.isArray(obj)) {
    if (obj.length === 0) return '[]';
    return obj.map(item => {
      if (item !== null && typeof item === 'object' && !Array.isArray(item)) {
        const body = serializeYaml(item, indent + 2);
        const lines = body.split('\n');
        const first = lines[0].trimStart();
        const rest = lines.slice(1).join('\n');
        return `${pad}- ${first}${rest ? '\n' + rest : ''}`;
      }
      return `${pad}- ${serializeYaml(item, indent + 2).trimStart()}`;
    }).join('\n');
  }
  if (typeof obj === 'object') {
    const keys = Object.keys(obj);
    if (keys.length === 0) return '{}';
    return keys.map(k => {
      const v = obj[k];
      if (v !== null && typeof v === 'object') {
        const isEmpty = Array.isArray(v) ? v.length === 0 : Object.keys(v).length === 0;
        if (isEmpty) return `${pad}${k}: ${Array.isArray(v) ? '[]' : '{}'}`;
        return `${pad}${k}:\n${serializeYaml(v, indent + 2)}`;
      }
      return `${pad}${k}: ${serializeYaml(v, indent + 2)}`;
    }).join('\n');
  }
  return String(obj);
}

function colorYamlScalar(s) {
  if (!s) return [];
  if (/^(true|false)$/.test(s)) return [{ text: s, color: palette.blue }];
  if (/^(null|~)$/.test(s)) return [{ text: s, color: palette.overlay0 }];
  if (/^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(s)) return [{ text: s, color: palette.peach }];
  if (s === '[]' || s === '{}') return [{ text: s, color: null }];
  // Strings (quoted or unquoted) — our emitter never emits flow sequences
  // containing scalars, so any leftover scalar is a string value.
  return [{ text: s, color: palette.green }];
}

function findYamlCommentStart(line) {
  // `#` only starts a comment when not inside a quoted string and preceded
  // by whitespace or start-of-line.
  let inSingle = false;
  let inDouble = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === "'" && !inDouble) inSingle = !inSingle;
    else if (c === '"' && !inSingle) inDouble = !inDouble;
    else if (c === '#' && !inSingle && !inDouble && (i === 0 || /\s/.test(line[i - 1]))) {
      return i;
    }
  }
  return -1;
}

function syntaxHighlightYaml(text) {
  // Line-oriented tokenizer. Returns the same {text, color} shape as
  // syntaxHighlight so the rendering loop is symmetric.
  const tokens = [];
  const lines = text.split('\n');
  for (let li = 0; li < lines.length; li++) {
    const line = lines[li];
    const commentIdx = findYamlCommentStart(line);
    const code = commentIdx >= 0 ? line.slice(0, commentIdx) : line;
    const comment = commentIdx >= 0 ? line.slice(commentIdx) : '';

    const m = code.match(/^(\s*)(- +)?(.*)$/);
    const indent = m[1];
    const dash = m[2] || '';
    const rest = m[3];
    if (indent) tokens.push({ text: indent, color: null });
    if (dash) tokens.push({ text: dash, color: null });

    const kv = rest.match(/^([^:\s][^:]*?)(:)(\s*)(.*)$/);
    if (kv) {
      tokens.push({ text: kv[1], color: palette.mauve });
      tokens.push({ text: kv[2], color: null });
      if (kv[3]) tokens.push({ text: kv[3], color: null });
      if (kv[4]) tokens.push(...colorYamlScalar(kv[4]));
    } else if (rest) {
      tokens.push(...colorYamlScalar(rest));
    }

    if (comment) tokens.push({ text: comment, color: palette.overlay0 });
    if (li < lines.length - 1) tokens.push({ text: '\n', color: null });
  }
  return tokens;
}

// Spec viewer modal: in-memory YAML content (no URL fetch). Reuses
// shared modal chrome but owns its own Escape listener so it's self-contained —
// JobConfigSection state is local to that component.
function SpecViewerModal({ filename, content, onClose }) {
  const [copyLabel, setCopyLabel] = useState('Copy');

  useEffect(() => {
    function onKeyDown(e) { if (e.key === 'Escape') onClose(); }
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [onClose]);

  function handleDownload() {
    const blob = new Blob([content], { type: 'application/yaml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  function handleCopy() {
    navigator.clipboard.writeText(content).then(() => {
      setCopyLabel('Copied!');
      setTimeout(() => setCopyLabel('Copy'), 2000);
    });
  }

  const tokens = syntaxHighlightYaml(content);
  const body = html`
    <pre style=${'margin: 0; font-family: monospace; font-size: var(--font-size-xs); line-height: 1.6; white-space: pre; color: ' + palette.text}>
      ${tokens.map((t, i) =>
        t.color
          ? html`<span key=${i} style=${'color: ' + t.color}>${t.text}</span>`
          : t.text
      )}
    </pre>
  `;

  return html`
    <${ModalChrome}
      filename=${filename}
      onCopy=${handleCopy}
      onDownload=${handleDownload}
      onClose=${onClose}
      copyLabel=${copyLabel}
    >
      ${body}
    </${ModalChrome}>
  `;
}


// --- Per-Record Analysis (Feature 3 from spec) ---

const PHASE_COLORS = [
  palette.blue,
  palette.teal,
  palette.peach,
  palette.mauve,
  palette.green,
  palette.sapphire,
  palette.lavender,
  palette.yellow,
  palette.red,
  palette.pink,
];

function extractJsonlMetric(record, key) {
  const v = record?.metrics?.[key];
  if (v == null) return null;
  return typeof v === 'object' ? (v.value ?? null) : v;
}

function PerRecordAnalysis({ records }) {
  const [tableExpanded, setTableExpanded] = useState(false);
  const [sortCol, setSortCol] = useState('#');
  const [sortAsc, setSortAsc] = useState(true);

  if (!records || records.length === 0) return null;

  // Extract per-record data
  const rows = records.map((rec, i) => {
    const isl = extractJsonlMetric(rec, 'input_sequence_length');
    const osl = extractJsonlMetric(rec, 'output_sequence_length');
    const ttft = extractJsonlMetric(rec, 'time_to_first_token');
    const latency = extractJsonlMetric(rec, 'request_latency');
    const itl = extractJsonlMetric(rec, 'inter_chunk_latency') ?? extractJsonlMetric(rec, 'inter_token_latency');
    const errorIsl = extractJsonlMetric(rec, 'error_isl');
    // ErrorDetails carries (message, code, type); pick the most concise label
    // (type > code > generic "error") so the column stays narrow and sortable.
    const errorObj = rec?.error ?? null;
    const errorLabel = errorObj
      ? (errorObj.type ?? (errorObj.code != null ? `HTTP ${errorObj.code}` : 'error'))
      : null;
    const phase = rec?.metadata?.phase ?? rec?.metadata?.credit_phase ?? null;
    return { index: i + 1, isl, osl, ttft, latency, itl, errorIsl, errorObj, errorLabel, phase };
  });

  // Collect unique phase values for coloring (only use if >1 distinct phase)
  const phaseSet = [...new Set(rows.map(r => r.phase).filter(p => p != null))].sort();
  const multiPhase = phaseSet.length > 1;
  const phaseColorMap = {};
  if (multiPhase) {
    phaseSet.forEach((p, i) => { phaseColorMap[p] = PHASE_COLORS[i % PHASE_COLORS.length]; });
  }

  // Scatter: latency vs request index
  const latencyScatterData = {
    datasets: multiPhase
      ? phaseSet.map(p => ({
          label: String(p),
          data: rows.filter(r => r.phase === p && r.latency != null).map(r => ({ x: r.index, y: r.latency })),
          backgroundColor: (phaseColorMap[p] ?? palette.blue) + 'bb',
          pointRadius: 3,
          pointHoverRadius: 5,
        }))
      : [{
          label: 'Latency',
          data: rows.filter(r => r.latency != null).map(r => ({ x: r.index, y: r.latency })),
          backgroundColor: palette.peach + 'bb',
          pointRadius: 3,
          pointHoverRadius: 5,
        }],
  };

  const latencyScatterOptions = {
    plugins: {
      legend: { display: multiPhase, labels: { color: palette.overlay1, font: { size: 10 } } },
      quadrantLabels: false,
      tooltip: {
        callbacks: {
          label: ctx => ` Request #${fmtInt(ctx.parsed.x)}: ${fmtNumber(ctx.parsed.y, 1)} ms`,
        },
      },
    },
    scales: {
      x: {
        title: { display: true, text: 'Request #', color: palette.overlay1, font: { size: 10 } },
        ticks: { color: palette.overlay0, font: { size: 10 } },
        grid: { color: palette.surface0 + '60' },
      },
      y: {
        title: { display: true, text: 'Latency (ms)', color: palette.overlay1, font: { size: 10 } },
        ticks: { color: palette.overlay0, font: { size: 10 } },
        grid: { color: palette.surface0 + '60' },
      },
    },
  };

  // Scatter: TTFT vs ISL
  const hasTtftIsl = rows.some(r => r.ttft != null && r.isl != null);
  const ttftIslScatterData = hasTtftIsl ? {
    datasets: multiPhase
      ? phaseSet.map(p => ({
          label: String(p),
          data: rows.filter(r => r.phase === p && r.ttft != null && r.isl != null).map(r => ({ x: r.isl, y: r.ttft })),
          backgroundColor: (phaseColorMap[p] ?? palette.teal) + 'bb',
          pointRadius: 3,
          pointHoverRadius: 5,
        }))
      : [{
          label: 'TTFT',
          data: rows.filter(r => r.ttft != null && r.isl != null).map(r => ({ x: r.isl, y: r.ttft })),
          backgroundColor: palette.teal + 'bb',
          pointRadius: 3,
          pointHoverRadius: 5,
        }],
  } : null;

  const ttftIslOptions = {
    plugins: {
      legend: { display: multiPhase, labels: { color: palette.overlay1, font: { size: 10 } } },
      quadrantLabels: false,
      tooltip: {
        callbacks: {
          label: ctx => ` ISL ${fmtInt(ctx.parsed.x)} tokens: TTFT ${fmtNumber(ctx.parsed.y, 1)} ms`,
        },
      },
    },
    scales: {
      x: {
        title: { display: true, text: 'Input Sequence Length (tokens)', color: palette.overlay1, font: { size: 10 } },
        ticks: { color: palette.overlay0, font: { size: 10 } },
        grid: { color: palette.surface0 + '60' },
      },
      y: {
        title: { display: true, text: 'TTFT (ms)', color: palette.overlay1, font: { size: 10 } },
        ticks: { color: palette.overlay0, font: { size: 10 } },
        grid: { color: palette.surface0 + '60' },
      },
    },
  };

  // Sortable table
  const hasItl = rows.some(r => r.itl != null);
  const hasErrors = rows.some(r => r.errorObj != null);
  const errorCount = rows.filter(r => r.errorObj != null).length;
  const COL_DEFS = [
    { key: '#', label: '#', get: r => r.index, fmt: v => fmtInt(v) },
    // ISL collapses input_sequence_length (success) and error_isl (failure)
    // into a single column — they're the same quantity, just produced by
    // different code paths depending on whether the request errored.
    { key: 'isl', label: 'ISL', get: r => r.isl ?? r.errorIsl, fmt: v => fmtInt(v) },
    { key: 'osl', label: 'OSL', get: r => r.osl, fmt: v => fmtInt(v) },
    { key: 'ttft', label: 'TTFT (ms)', get: r => r.ttft, fmt: v => fmtNumber(v, 1) },
    { key: 'latency', label: 'Latency (ms)', get: r => r.latency, fmt: v => fmtNumber(v, 1) },
    ...(hasItl ? [{ key: 'itl', label: 'ITL (ms)', get: r => r.itl, fmt: v => fmtNumber(v, 1) }] : []),
    ...(hasErrors ? [{ key: 'error', label: 'Error', get: r => r.errorLabel, fmt: v => v ?? '' }] : []),
  ];

  function handleSort(col) {
    if (sortCol === col) setSortAsc(a => !a);
    else { setSortCol(col); setSortAsc(true); }
  }

  const def = COL_DEFS.find(d => d.key === sortCol) ?? COL_DEFS[0];
  const sorted = [...rows].sort((a, b) => {
    const av = def.get(a);
    const bv = def.get(b);
    // Nulls always sort last regardless of direction.
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    const cmp = typeof av === 'string' ? av.localeCompare(bv) : (av - bv);
    return sortAsc ? cmp : -cmp;
  });
  // Hard cap on expanded view: rendering 100k <tr>s freezes the browser.
  // Users who need every row should download profile_export.jsonl directly.
  const EXPANDED_MAX = 1000;
  const truncated = tableExpanded && sorted.length > EXPANDED_MAX;
  const displayRows = tableExpanded ? sorted.slice(0, EXPANDED_MAX) : sorted.slice(0, 50);

  const thStyle = col => [
    'padding: var(--space-2) var(--space-3);',
    'text-align: right; font-weight: 600;',
    'font-size: var(--font-size-xs);',
    'color: ' + (sortCol === col ? palette.blue : palette.overlay1) + ';',
    'border-bottom: 1px solid ' + palette.surface0 + ';',
    'cursor: pointer; user-select: none; white-space: nowrap;',
    'background: ' + palette.surface0 + ';',
  ].join(' ');

  const th1Style = [
    'padding: var(--space-2) var(--space-3);',
    'text-align: left; font-weight: 600;',
    'font-size: var(--font-size-xs);',
    'color: ' + (sortCol === '#' ? palette.blue : palette.overlay1) + ';',
    'border-bottom: 1px solid ' + palette.surface0 + ';',
    'cursor: pointer; user-select: none;',
    'background: ' + palette.surface0 + ';',
  ].join(' ');

  return html`
    <div class="card" style="margin-top: var(--space-4)">
      <div class="card-title">Per-Record Analysis</div>
      <div style="font-size: var(--font-size-xs); color: ${palette.overlay0}; margin-bottom: var(--space-3)">
        ${fmtInt(records.length)} requests${hasErrors ? html` <span style=${'color: ' + colors.error}>(${fmtInt(errorCount)} ${errorCount === 1 ? 'error' : 'errors'})</span>` : ''}
      </div>

      <!-- Scatter: Latency vs Request # -->
      <div style="margin-bottom: var(--space-4)">
        <div style=${'font-size: var(--font-size-xs); font-weight: 600; color: ' + palette.overlay1 + '; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: var(--space-2)'}>Request Latency Over Time</div>
        <${ChartWrapper} type="scatter" data=${latencyScatterData} options=${latencyScatterOptions} height=${220} />
      </div>

      <!-- Scatter: TTFT vs ISL -->
      ${hasTtftIsl && html`
        <div style="margin-bottom: var(--space-4)">
          <div style=${'font-size: var(--font-size-xs); font-weight: 600; color: ' + palette.overlay1 + '; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: var(--space-2)'}>TTFT vs Input Sequence Length</div>
          <${ChartWrapper} type="scatter" data=${ttftIslScatterData} options=${ttftIslOptions} height=${220} />
        </div>
      `}

      <!-- Per-request table (collapsed by default) -->
      <div>
        <div
          onclick=${() => setTableExpanded(e => !e)}
          style=${'display: flex; align-items: center; gap: var(--space-2); padding: var(--space-2) var(--space-3); background: ' + palette.surface0 + '60; border-radius: var(--radius-sm); cursor: pointer; user-select: none; margin-bottom: var(--space-2)'}
        >
          <span style=${'font-size: var(--font-size-xs); font-weight: 600; color: ' + palette.overlay1 + '; text-transform: uppercase; letter-spacing: 0.06em'}>Per-Request Table</span>
          <span class="text-dim" style="font-size: var(--font-size-xs); margin-left: auto">${tableExpanded ? '\u25B2 Collapse' : '\u25BC Expand'}</span>
        </div>
        ${tableExpanded && html`
          <div style="overflow-x: auto">
            <table style="width: 100%; border-collapse: collapse; font-size: var(--font-size-xs); font-family: monospace">
              <thead>
                <tr>
                  ${COL_DEFS.map((col, i) => html`
                    <th
                      key=${col.key}
                      role="columnheader"
                      tabindex="0"
                      onkeydown=${(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleSort(col.key); } }}
                      onclick=${() => handleSort(col.key)}
                      style=${i === 0 ? th1Style : thStyle(col.key)}
                    >
                      ${col.label}${sortCol === col.key ? (sortAsc ? ' \u25B2' : ' \u25BC') : ''}
                    </th>
                  `)}
                </tr>
              </thead>
              <tbody>
                ${displayRows.map((row, ri) => {
                  const isErr = row.errorObj != null;
                  // Faint red tint on error rows so failures are visible at a
                  // glance even when sorted away from the top.
                  const rowBg = isErr
                    ? colors.error + '14'
                    : (ri % 2 === 0 ? palette.base : palette.mantle);
                  return html`
                    <tr key=${row.index} style=${'background: ' + rowBg}>
                      ${COL_DEFS.map((col, ci) => {
                        const isErrCol = col.key === 'error';
                        const cellColor = (isErrCol && isErr) ? colors.error : palette.text;
                        return html`
                          <td key=${col.key} style=${'padding: var(--space-1) var(--space-3); color: ' + cellColor + '; text-align: ' + (ci === 0 ? 'left' : 'right') + '; border-bottom: 1px solid ' + palette.surface0 + '40'}>
                            ${col.fmt(col.get(row))}
                          </td>
                        `;
                      })}
                    </tr>
                  `;
                })}
              </tbody>
            </table>
            ${truncated && html`
              <div style=${'margin-top: var(--space-2); padding: var(--space-2) var(--space-3); font-size: var(--font-size-xs); color: ' + palette.overlay1 + '; font-style: italic; text-align: center'}>
                Showing first ${fmtInt(EXPANDED_MAX)} of ${fmtInt(sorted.length)} rows. Download profile_export.jsonl for the full set.
              </div>
            `}
          </div>
        `}
      </div>
    </div>
  `;
}

// "Similar runs" chip — ports the legacy ui's `IdentityStrip` sibling
// counter (``src/aiperf/operator/ui/views/run.js::siblingCount``).
//
// Definition of "similar": same namespace AND same model, excluding the
// current run itself. Comparability is count-only — we never aggregate
// metrics across independent benchmarks (the legacy comment is verbatim
// here on purpose). Clicking the chip jumps to ``/compare?cluster=<ns>·<model>``
// where the compare page auto-selects every matching run.
//
// The ns·model URL shape (with the spaced middle-dot) matches the
// legacy ui exactly so deep-links shared between the two UIs resolve to
// the same set of jobs.
function formatRunSelectorTime(epochSeconds) {
  if (epochSeconds == null || epochSeconds === 0) return '—';
  return new Date(epochSeconds * 1000).toLocaleString([], {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

function RunSelectorCard({ namespace, name, epochs, current, hasLive, isRunning }) {
  const rows = buildRunSelectorRows({ namespace, name, epochs, current, hasLive, isRunning });
  if (rows.length === 0) return null;
  const epochCount = rows.filter(r => r.kind === 'epoch').length;
  return html`
    <div class="run-selector-card" data-testid="job-detail-run-selector">
      <div class="run-selector-bar">
        <span class="run-selector-bar-title">Runs</span>
        ${epochCount > 0 && html`<span class="run-selector-bar-count">${epochCount}</span>`}
        <div class="run-selector-pills" role="tablist">
          ${rows.map(row => {
            const isLive = row.kind === 'live';
            const dotClass = isLive
              ? (isRunning ? ' run-pill-dot--running' : ' run-pill-dot--idle')
              : '';
            const meta = isLive
              ? null
              : formatRunSelectorTime(row.mtimeEpoch);
            const fileSuffix = isLive
              ? null
              : (row.fileCount != null ? ` · ${fmtInt(row.fileCount)} files` : '');
            const title = isLive
              ? (isRunning ? 'Live — streaming current-run metrics' : 'Latest persisted run')
              : `Epoch ${row.label}${fileSuffix ?? ''}${row.isLatest ? ' (latest)' : ''}`;
            return html`
              <a
                key=${row.kind + ':' + (row.epoch || 'live')}
                href=${row.href}
                onclick=${e => { e.preventDefault(); navigate(row.href.slice(1)); }}
                class=${'run-pill'
                  + (isLive ? ' run-pill--live' : '')
                  + (row.selected ? ' run-pill--selected' : '')
                  + (row.isLatest && !row.selected ? ' run-pill--latest' : '')}
                data-testid=${isLive ? 'run-selector-live' : 'run-selector-epoch'}
                role="tab"
                aria-selected=${row.selected ? 'true' : 'false'}
                title=${title}
              >
                ${isLive
                  ? html`<span class=${'run-pill-dot' + dotClass}></span>`
                  : null}
                <span class="run-pill-label">${row.label}</span>
                ${meta && html`<span class="run-pill-meta">${meta}</span>`}
                ${row.isLatest && !isLive && html`<span class="run-pill-badge">latest</span>`}
              </a>
            `;
          })}
        </div>
      </div>
    </div>
  `;
}

function SimilarRunsLink({ namespace, model, currentName }) {
  if (!namespace || !model) return null;
  const all = jobsSignal.value ?? [];
  let n = 0;
  for (const r of all) {
    if (r.namespace === namespace && r.model === model && r.name !== currentName) n++;
  }
  if (n === 0) return null;

  const clusterKey = `${namespace} · ${model}`;
  const onClick = (e) => {
    e.preventDefault();
    navigate('/compare?cluster=' + encodeURIComponent(clusterKey));
  };

  return html`
    <a
      href=${'#/compare?cluster=' + encodeURIComponent(clusterKey)}
      onclick=${onClick}
      data-testid="job-detail-similar-runs"
      title=${`Compare against the other ${n} run${n === 1 ? '' : 's'} in ${clusterKey}`}
      style=${'display: inline-flex; align-items: center; gap: var(--space-1);'
        + ' padding: 2px var(--space-2);'
        + ' border-radius: 999px;'
        + ' font-size: var(--font-size-xs);'
        + ' font-weight: 600;'
        + ' background: ' + palette.accent + '14;'
        + ' color: ' + palette.accent + ';'
        + ' border: 1px solid ' + palette.accent + '33;'
        + ' text-decoration: none;'
        + ' cursor: pointer'}
    >
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <rect x="3" y="3" width="7" height="7" rx="1" />
        <rect x="14" y="3" width="7" height="7" rx="1" />
        <rect x="3" y="14" width="7" height="7" rx="1" />
        <rect x="14" y="14" width="7" height="7" rx="1" />
      </svg>
      <span>+${n} similar run${n === 1 ? '' : 's'}</span>
      <span aria-hidden="true" style="opacity: 0.7; font-size: 10px">→</span>
    </a>
  `;
}


export function JobDetail({ namespace, name, epoch }) {
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const [files, setFiles] = useState([]);
  // ``filesLoaded`` flips to true once the first /results listing fetch
  // resolves (success OR 404/error). Lets the Artifacts section
  // distinguish "still fetching" from "fetched, empty" so an always-on
  // card can show a real message instead of a permanent loader.
  const [filesLoaded, setFilesLoaded] = useState(false);
  const [polling, setPolling] = useState(true);
  const [serverMetrics, setServerMetrics] = useState(null);
  const [serverMetricsLoaded, setServerMetricsLoaded] = useState(false);
  const [serverMetricsError, setServerMetricsError] = useState(null);
  const [jsonlRecords, setJsonlRecords] = useState(null);
  const [jsonlLoaded, setJsonlLoaded] = useState(false);
  const [jsonlError, setJsonlError] = useState(null);
  // Progress for the JSONL parse so users see a count tick up instead of a
  // multi-second blank skeleton on 50k+ row exports.
  const [jsonlProgress, setJsonlProgress] = useState(null);
  const [jobConfig, setJobConfig] = useState(null);
  const [jobConfigLoaded, setJobConfigLoaded] = useState(false);
  const [jobConfigError, setJobConfigError] = useState(null);
  const [epochs, setEpochs] = useState([]);
  // Cancel-button state: 'idle' shows the button, 'confirm' shows an inline
  // confirm/abort pair, 'pending' disables both while the API call is in flight.
  // Replaces native confirm()/alert() which provided no in-flight feedback and
  // let users double-click to fire two cancels.
  const [cancelState, setCancelState] = useState('idle');
  const [cancelError, setCancelError] = useState(null);

  const resultsBase = epoch
    ? `/api/v1/results/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}/runs/${encodeURIComponent(epoch)}`
    : null;


  useEffect(() => {
    let cancelled = false;
    api.getJobEpochs(namespace, name)
      .then(d => { if (!cancelled) setEpochs(d.epochs ?? []); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [namespace, name]);

  function pickEpoch(next) {
    const latest = epochs.find(e => e.isLatest)?.epoch;
    const target = next ?? latest;
    if (target === undefined) navigate(`/jobs/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}`);
    else navigate(`/jobs/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}/runs/${encodeURIComponent(target)}`);
  }

  // Rolling throughput chart data - kept in a ref so we don't trigger re-renders for
  // each append; we rebuild the data object for ChartWrapper on each render.
  const throughputPoints = useRef({ labels: [], values: [] });
  const [chartData, setChartData] = useState(null);

  // Live realtime feed proxied through the operator into the controller pod's
  // ``/ws``. Empty until ``isRunning`` opens the socket below.
  const [liveData, setLiveData] = useState({
    summary: {}, timeseries: {}, serverSummary: null, serverTimeseries: {}, connected: false,
  });
  const jobFreshness = freshness.value['job-detail'] ?? null;

  // Open the per-job WebSocket whenever the run is active AND the URL points
  // at the currently-running epoch — either the no-epoch live URL, or
  // /runs/<currentRunEpoch> (which is what every dashboard/history link
  // produces via buildJobPath). Pinned views of *past* archived epochs of
  // a now-rerunning job skip the WS so live current-run stats don't bleed
  // into the archived render. The proxy refuses non-running CRs anyway,
  // but gating here saves a connect/4404/reconnect loop.
  const liveRunState = deriveJobRunState({
    phase: job?.job?.phase ?? job?.status?.phase,
    epoch,
    runEpoch: job?.status?.runEpoch,
  });
  const viewingCurrentRun = liveRunState.viewingCurrentRun;
  const wsActive = liveRunState.isRunning && viewingCurrentRun;
  useEffect(() => {
    if (!wsActive) {
      // Clear stale live state so a finished job doesn't keep painting old samples.
      setLiveData({ summary: {}, timeseries: {}, serverSummary: null, serverTimeseries: {}, connected: false });
      return;
    }
    const handle = openJobWs(namespace, name, (snap) => setLiveData(snap));
    return () => handle.close();
  }, [namespace, name, wsActive]);

  useEffect(() => {
    const ac = new AbortController();
    const pollAc = new AbortController();
    let stopped = false;
    // Reset chart points when job changes
    throughputPoints.current = { labels: [], values: [] };
    setChartData(null);
    setPolling(true);
    // Reset the artifact state so navigating between jobs doesn't briefly
    // show the previous job's file list under the new header.
    setFiles([]);
    setFilesLoaded(false);
    setServerMetrics(null);
    setServerMetricsLoaded(false);
    setServerMetricsError(null);
    setJsonlRecords(null);
    setJsonlLoaded(false);
    setJsonlError(null);
    setJsonlProgress(null);
    setJobConfig(null);
    setJobConfigLoaded(false);
    setJobConfigError(null);
    setJob(null);
    setError(null);
    clearFreshnessSource('job-detail');
    let firstLoadDone = false;

    poll(
      async ({ stopFreshness }) => {
        let data;
        try {
          data = await api.getJob(namespace, name, epoch);
          if (stopped) return;
        } catch (err) {
          if (stopped) return;
          if (!firstLoadDone) {
            // First-load failures replace the page because there is no job
            // content to preserve yet; later poll failures re-throw into the
            // freshness path so the stale banner/pill reflects retrying state.
            setError(err?.message ?? String(err));
          }
          throw err;
        }
        if (data == null) {
          // 204 / explicit-null body lands here. First-load failure needs a
          // page error; later polls must preserve rendered content and let
          // freshness show retry/stale state.
          const emptyError = new Error('Empty response from operator');
          if (!firstLoadDone) setError(emptyError.message);
          throw emptyError;
        }
        setJob(data);
        setError(null);
        firstLoadDone = true;

        const state = deriveJobRunState({
          phase: data?.job?.phase ?? data?.status?.phase,
          epoch,
          runEpoch: data?.status?.runEpoch,
        });
        if (state.pollingDone) {
          setPolling(false);
          stopFreshness('terminal');
          pollAc.abort();
        }

        // Append to throughput chart
        const summary = extractSummary(data);
        const tps =
          summary?.output_token_throughput?.avg ??
          data?.status?.liveMetrics?.metrics?.output_token_throughput?.avg ??
          null;

        if (tps != null) {
          const pts = throughputPoints.current;
          const label = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
          pts.labels.push(label);
          pts.values.push(tps);
          if (pts.labels.length > MAX_CHART_POINTS) {
            pts.labels.shift();
            pts.values.shift();
          }
          setChartData({
            labels: [...pts.labels],
            datasets: [
              {
                label: 'Output Token Throughput (tok/s)',
                data: [...pts.values],
                borderColor: palette.blue,
                backgroundColor: palette.blue + '22',
                fill: true,
                tension: 0.3,
                pointRadius: 0,
                borderWidth: 2,
              },
            ],
          });
        }
      },
      3000,
      pollAc.signal,
      { source: 'job-detail' },
    );

    // Fetch job config (original CR spec)
    api.getJobConfig(namespace, name, epoch)
      .then(d => {
        if (ac.signal.aborted) return;
        setJobConfig(d);
        setJobConfigLoaded(true);
        setJobConfigError(d ? null : 'Job configuration unavailable.');
      })
      .catch((err) => {
        if (ac.signal.aborted) return;
        setJobConfig(null);
        setJobConfigLoaded(true);
        setJobConfigError(err?.message ?? 'Job configuration unavailable.');
      });

    // Final artifacts are run-scoped. Do not hit the non-epoch results
    // endpoint; wait until the route is pinned to /runs/<epoch>.
    if (!resultsBase) {
      setFilesLoaded(true);
      setServerMetricsLoaded(true);
      setJsonlLoaded(true);
    } else {
      fetch(resultsBase, { signal: ac.signal })
        .then(r => (r.ok ? r.json() : null))
        .then(d => {
          if (ac.signal.aborted) return;
          if (!d) {
            setFilesLoaded(true);
            setServerMetricsLoaded(true);
            setJsonlLoaded(true);
            return;
          }
          const fileList = d?.files ?? [];
          setFiles(fileList);
          setFilesLoaded(true);
          if (fileList.some(f => f.name === 'server_metrics_export.json')) {
            fetch(`${resultsBase}/server_metrics_export.json`, { signal: ac.signal })
              .then(r => (r.ok ? r.json() : null))
              .then(sm => {
                if (ac.signal.aborted) return;
                setServerMetrics(sm);
                setServerMetricsLoaded(true);
                setServerMetricsError(sm ? null : 'Server metrics artifact could not be read.');
              })
              .catch(err => {
                if (ac.signal.aborted) return;
                setServerMetrics(null);
                setServerMetricsLoaded(true);
                setServerMetricsError(err?.message ?? 'Server metrics artifact could not be read.');
              });
          } else {
            setServerMetricsLoaded(true);
          }
          if (fileList.some(f => f.name === 'profile_export.jsonl')) {
            fetch(`${resultsBase}/profile_export.jsonl`, { signal: ac.signal })
              .then(r => r.ok ? r.text() : null)
              .then(async text => {
                if (!text || ac.signal.aborted) return;
                const lines = text.split('\n');
                // Chunk the parse so the main thread can paint between batches.
                // 50k records parse in well under a second total; 2k per yield
                // keeps each frame under ~16ms even on slow hardware.
                const CHUNK = 2000;
                const recs = [];
                setJsonlProgress({ done: 0, total: lines.length });
                for (let i = 0; i < lines.length; i += CHUNK) {
                  if (ac.signal.aborted) return;
                  const end = Math.min(i + CHUNK, lines.length);
                  for (let j = i; j < end; j++) {
                    const line = lines[j];
                    if (!line || line.trim() === '') continue;
                    try { recs.push(JSON.parse(line)); } catch { /* skip bad line */ }
                  }
                  setJsonlProgress({ done: end, total: lines.length });
                  // Yield to the event loop so the UI can repaint.
                  await new Promise(r => setTimeout(r, 0));
                }
                if (ac.signal.aborted) return;
                setJsonlProgress(null);
                setJsonlRecords(recs.length > 0 ? recs : null);
                setJsonlLoaded(true);
              })
              .catch(err => {
                if (ac.signal.aborted) return;
                setJsonlProgress(null);
                setJsonlLoaded(true);
                setJsonlError(err?.message ?? 'Per-request records could not be read.');
              });
          } else {
            setJsonlLoaded(true);
          }
        })
        .catch(() => {
          if (ac.signal.aborted) return;
          setFilesLoaded(true);
          setServerMetricsLoaded(true);
          setJsonlLoaded(true);
        });
    }

    return () => {
      stopped = true;
      pollAc.abort();
      ac.abort();
    };
  }, [namespace, name, epoch, resultsBase]);


  // job detail response: { job: {AIPerfJobInfo}, status: {raw CR status}, pods: [...] }
  // job.job has flat camelCase fields, job.status has raw CR status
  const info = job?.job ?? {};
  const status = job?.status ?? {};
  // Redirect target falls back through three sources so the URL gets pinned
  // to /runs/<epoch> for any state where one is knowable: pinned URL > CR
  // status.runEpoch (current/last run) > latest persisted epoch from the
  // index (covers archived jobs whose CR is gone or never had runEpoch set).
  const latestPersistedEpoch = epochs.find(e => e?.isLatest)?.epoch;
  const resolvedEpoch = epoch
    ?? (status.runEpoch != null ? String(status.runEpoch) : null)
    ?? (latestPersistedEpoch != null ? String(latestPersistedEpoch) : null);

  useEffect(() => {
    if (epoch !== undefined || resolvedEpoch == null) return;
    replaceRoute(`/jobs/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}/runs/${encodeURIComponent(resolvedEpoch)}`);
  }, [epoch, resolvedEpoch, namespace, name]);

  async function handleCancel() {
    setCancelError(null);
    setCancelState('pending');
    try {
      await api.cancelJob(namespace, name);
      // Stay in 'pending' until the next poll flips phase out of running.
    } catch (e) {
      setCancelError(e?.message ?? String(e));
      setCancelState('idle');
    }
  }

  if (!job && !error) {
    return html`
      <div class="card">
        <${LoadingPanel} label=${'Loading ' + namespace + '/' + name + '…'} testid="job-detail-loading" />
      </div>
    `;
  }

  if (error) {
    return html`
      <div class="card" style="border-color: ${colors.error}44; color: ${colors.error}" data-testid="job-detail-error">
        <div style="font-weight: 600; margin-bottom: var(--space-1)">Failed to load job</div>
        <div style="font-size: var(--font-size-sm); word-break: break-word; margin-bottom: var(--space-2)">${error}</div>
        <div style="font-size: var(--font-size-sm); color: var(--muted)">
          The operator may be unreachable, or this job may have been deleted. Try
          <a href="#/jobs" onclick=${e => { e.preventDefault(); navigate('/jobs'); }} style=${'color: ' + palette.blue + '; cursor: pointer'}>back to all jobs</a>
          or reload the page.
        </div>
      </div>
    `;
  }

  const phase = info.phase ?? status.phase ?? 'Unknown';
  const phaseClr = phaseColor(phase);
  const model = info.model ?? '---';
  const endpointUrl = info.endpoint ?? null;
  const startTime = info.startTime ?? status.startTime;
  const runState = deriveJobRunState({
    phase,
    epoch,
    runEpoch: status.runEpoch,
  });
  const isRunning = runState.isRunning;
  const isCompleted = runState.isCompleted;
  const isCancelled = runState.isCancelled;
  const isPartiallyFailed = runState.isPartiallyFailed;
  const isArchived = runState.isArchived;
  const isTerminal = runState.isTerminal;
  const isFinal = isCompleted || isArchived;
  const showLiveRunPanels = runState.showLiveRunPanels;
  const liveServerMetricsBase = viewingCurrentRun ? status.serverMetrics : null;
  const liveServerMetrics = (liveData.connected && liveData.serverSummary)
    ? liveData.serverSummary
    : liveServerMetricsBase;
  const displayedServerMetrics = serverMetrics || liveServerMetrics;
  const serverMetricsSource = serverMetrics ? 'final' : 'live';

  // status.summary (completed) and status.liveSummary (running) carry the same
  // curated nested ``{tag: {avg, p50, p99, ...}}`` projection of the AIPerf
  // metrics dict. status.results.metrics and status.liveMetrics.metrics are the
  // unfiltered superset; they fall in as fallbacks when summary is empty.
  //
  // When the per-job WS is connected, ``liveData.summary`` overlays the REST
  // snapshot per-tag — its ``current``/``avg``/``p99`` fields move at the
  // controller's emit rate (~1Hz) instead of the page poll cadence.
  const restSummary =
    status.results?.metrics ??
    status.liveMetrics?.metrics ??
    status.summary ??
    status.liveSummary ??
    {};
  const summary = (liveData.connected && Object.keys(liveData.summary).length > 0)
    ? { ...restSummary, ...liveData.summary }
    : restSummary;
  const throughput = summary.request_throughput?.avg ?? info.throughputRps ?? null;
  const ttftAvg = summary.time_to_first_token?.avg ?? null;
  const latP99 = summary.request_latency?.p99 ?? info.latencyP99Ms ?? null;

  // Convenience alias: results = summary so percentile-aware components work
  // unchanged whether the job is running (liveMetrics) or completed (results).
  const results = summary;
  const outputTokenThroughput = summary.output_token_throughput?.avg ?? null;

  const conditions = status.conditions ?? [];
  // User-declared SLO thresholds from the AIPerfJob CR (same dict the
  // SLACompliance card consumes). Drives chip + border color on the
  // dynamic KPI grid; absent SLOs leave tiles uncolored.
  const slos =
    jobConfig?.spec?.benchmark?.slos
    ?? jobConfig?.spec?.slos
    ?? null;
  // Convert phases dict {name: {requestsCompleted, requestsTotal, ...}} to array.
  // ``p`` may be null briefly during a phase transition, so ``?.`` the inner
  // reads. Operator emits camelCase per CRD convention; no snake fallback.
  const rawPhases = status.phases ?? {};
  const phasesArray = Object.entries(rawPhases).map(([phaseName, p]) => ({
    name: phaseName,
    completed: p?.requestsCompleted ?? 0,
    total: p?.requestsTotal ?? 0,
  }));
  const pods = job?.pods ?? [];
  const jobError = info.error ?? status.error ?? null;

  // Build latency histogram from completed results if available
  const latencyHistogram = (() => {
    const buckets = job?.status?.results?.latency_histogram ?? job?.status?.results?.histograms?.request_latency ?? null;
    if (!buckets || !Array.isArray(buckets) || buckets.length === 0) return null;
    // Bucket upper bound ``le`` is in seconds. Tick labels swap to "s" past 1s
    // so a 60-second tail doesn't render as "60000ms".
    const fmtBucket = (le) => {
      if (typeof le !== 'number') return String(le);
      if (le >= 1) return le.toFixed(le >= 10 ? 0 : 1) + 's';
      return (le * 1000).toFixed(0) + 'ms';
    };
    return {
      labels: buckets.map((b) => fmtBucket(b.le)),
      datasets: [
        {
          label: 'Requests',
          data: buckets.map((b) => b.count ?? b.value ?? 0),
          backgroundColor: palette.mauve + '88',
          borderColor: palette.mauve,
          borderWidth: 1,
        },
      ],
    };
  })();

  const throughputChartOptions = LIVE_THROUGHPUT_OPTIONS;

  const histogramOptions = {
    plugins: { legend: { display: false } },
    scales: {
      x: {
        ticks: { color: palette.overlay0, font: { size: 10 } },
        grid: { color: palette.surface0 },
        title: { display: true, text: 'Latency', color: palette.overlay1, font: { size: 10 } },
      },
      y: {
        ticks: { color: palette.overlay0, font: { size: 10 } },
        grid: { color: palette.surface0 },
        title: { display: true, text: 'Count', color: palette.overlay1, font: { size: 10 } },
      },
    },
  };


  // Warmup hint: running, but no live KPI numbers yet — typical for the first ~30s
  // while the workers ramp and TimingManager hasn't issued enough credits to populate
  // any percentile. Without this hint, all-`---` KPIs read as "broken" instead of "soon".
  const noKpisYet = throughput == null && ttftAvg == null && latP99 == null && outputTokenThroughput == null;
  const showWarmupHint = isRunning && noKpisYet;
  const currentSubPhase = info.currentPhase ?? status.currentPhase ?? null;


  return html`
    <div class="job-detail" data-testid="page-job-detail">
      <!-- Header -->
      <div class="card" style="margin-bottom: var(--space-4)">
        <div style="display: flex; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; gap: var(--space-3)">
          <div>
            <div style="display: flex; align-items: center; gap: var(--space-3); flex-wrap: wrap">
              <h2 style="margin: 0; font-size: var(--font-size-lg)">${name}</h2>
              <span class="phase-badge" style=${'background: ' + phaseClr + '22; color: ' + phaseClr + '; border-color: ' + phaseClr + '44'}>
                ${phase}
              </span>
              <${NsPill} ns=${namespace} onClick=${ns => navigate('/jobs?ns=' + encodeURIComponent(ns))} testId="job-detail-ns-pill" />
              ${model && html`<${ModelPill} model=${model} onClick=${m => navigate('/jobs?model=' + encodeURIComponent(m))} testId="job-detail-model-pill" />`}
              ${model && model !== '---' && html`<${SimilarRunsLink} namespace=${namespace} model=${model} currentName=${name} />`}
              ${startTime && html`<${RelativeTime} ts=${startTime} mode="elapsed" className="text-dim" />`}
              <!-- Live / Completed indicator -->
              ${polling
                ? html`
                  <span style="display: inline-flex; align-items: center; gap: var(--space-1); font-size: var(--font-size-xs); color: ${palette.green}">
                    <span style=${'display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: ' + palette.green + '; animation: pulse 1.5s ease-in-out infinite'} />
                    Live
                  </span>
                `
                : isCompleted
                  ? html`<span style=${'font-size: var(--font-size-xs); color: ' + palette.green + '; opacity: 0.7'}>Completed</span>`
                  : isArchived
                    ? html`<span style=${'font-size: var(--font-size-xs); color: ' + palette.subtext0 + '; opacity: 0.85'} title="Archived run loaded from persisted results.">Archived</span>`
                    : isCancelled
                      ? html`<span style=${'font-size: var(--font-size-xs); color: ' + palette.subtext0 + '; opacity: 0.85'} title="Run was cancelled before completion — KPIs reflect partial data.">Cancelled</span>`
                      : isPartiallyFailed
                        ? html`<span style=${'font-size: var(--font-size-xs); color: ' + colors.error + '; opacity: 0.85'} title="Run finished but some workers failed — KPIs reflect surviving data.">Partially failed</span>`
                        : null
              }
              ${jobFreshness && html`<${FreshnessPill} source=${jobFreshness} compact=${true} />`}
              <${EpochSelector} epochs=${epochs} current=${epoch} onPick=${pickEpoch} />
            </div>
            ${endpointUrl && html`
              <div class="text-dim" style="font-size: var(--font-size-sm); margin-top: var(--space-1); max-width: 100%; overflow: hidden">
                <span
                  title=${endpointUrl}
                  style=${'color: ' + palette.blue + '; display: inline-block; max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; vertical-align: bottom'}
                >${endpointUrl}</span>
              </div>
            `}
            ${info.sweepName && html`
              <p class="text-dim" data-testid="job-detail-sweep-link" style="margin: var(--space-1) 0 0 0; font-size: var(--font-size-sm)">
                Part of sweep
                <a href=${`#/sweeps/${encodeURIComponent(namespace)}/${encodeURIComponent(info.sweepName)}`}
                   onclick=${e => { e.preventDefault(); navigate(`/sweeps/${encodeURIComponent(namespace)}/${encodeURIComponent(info.sweepName)}`); }}>
                  ${info.sweepName}
                </a>
                ${info.variationLabel && html` — variation ${info.variationLabel}`}
              </p>
            `}
          </div>
          ${isRunning && DASHBOARD_MUTATIONS_ENABLED && html`
            <div style="display: flex; flex-direction: column; align-items: flex-end; gap: var(--space-1)">
              ${cancelState === 'idle' && html`
                <button type="button"
                  class="btn btn--danger"
                  onclick=${() => setCancelState('confirm')}
                  style=${'background: ' + colors.error + '22; color: ' + colors.error + '; border: 1px solid ' + colors.error + '44; padding: var(--space-2) var(--space-4); border-radius: var(--radius-md); cursor: pointer; font-size: var(--font-size-sm)'}
                  data-testid="job-detail-cancel"
                  title="Stop the running benchmark. The AIPerfJob CR is kept; controller pod is terminated."
                >
                  Cancel run
                </button>
              `}
              ${cancelState === 'confirm' && html`
                <div style=${'display: flex; align-items: center; gap: var(--space-2); padding: var(--space-2) var(--space-3); background: ' + colors.error + '11; border: 1px solid ' + colors.error + '44; border-radius: var(--radius-md)'}>
                  <span style=${'font-size: var(--font-size-sm); color: ' + colors.error}>
                    Stop benchmark for <strong>${name}</strong>? The CR is kept (use "kubectl delete" to remove it).
                  </span>
                  <button type="button"
                    onclick=${handleCancel}
                    style=${'background: ' + colors.error + '; color: white; border: none; padding: var(--space-1) var(--space-3); border-radius: var(--radius-sm); cursor: pointer; font-size: var(--font-size-sm)'}
                    data-testid="job-detail-cancel-confirm"
                  >
                    Yes, cancel
                  </button>
                  <button type="button"
                    onclick=${() => { setCancelState('idle'); setCancelError(null); }}
                    style=${'background: transparent; color: ' + palette.subtext0 + '; border: 1px solid ' + palette.overlay0 + '44; padding: var(--space-1) var(--space-3); border-radius: var(--radius-sm); cursor: pointer; font-size: var(--font-size-sm)'}
                  >
                    Keep running
                  </button>
                </div>
              `}
              ${cancelState === 'pending' && html`
                <button type="button"
                  disabled
                  style=${'background: ' + colors.error + '22; color: ' + colors.error + '; border: 1px solid ' + colors.error + '44; padding: var(--space-2) var(--space-4); border-radius: var(--radius-md); cursor: not-allowed; font-size: var(--font-size-sm); display: inline-flex; align-items: center; gap: var(--space-2); opacity: 0.7'}
                  data-testid="job-detail-cancel"
                >
                  <${Spinner} size=${12} />
                  Cancelling…
                </button>
              `}
              ${cancelError && html`
                <span style=${'font-size: var(--font-size-xs); color: ' + colors.error}>Cancel failed: ${cancelError}</span>
              `}
            </div>
          `}
          ${isRunning && !DASHBOARD_MUTATIONS_ENABLED && html`
            <div
              data-testid="job-detail-cancel-readonly"
              title=${DASHBOARD_MUTATIONS_DISABLED_MESSAGE}
              style=${`max-width: 360px; font-size: var(--font-size-xs); color: ${palette.yellow}; text-align: right;`}
            >Cancel is disabled in the dashboard. Use <code>aiperf kube</code> or <code>kubectl</code> from an authenticated terminal.</div>
          `}
          ${isTerminal && jobConfig?.spec && html`
            <div style="display: flex; flex-direction: column; align-items: flex-end; gap: var(--space-1)">
              <${RelaunchButton} namespace=${namespace} name=${name} config=${jobConfig} />
            </div>
          `}
        </div>
      </div>

      <${StaleBanner} source=${jobFreshness} label="Job detail" />

      <${RunSelectorCard}
        namespace=${namespace}
        name=${name}
        epochs=${epochs}
        current=${epoch}
        hasLive=${true}
        isRunning=${isRunning}
      />

      <!-- Conditions -->
      ${conditions.length > 0 && html`
        <div style="margin-bottom: var(--space-4)">
          <${Conditions} conditions=${conditions} />
        </div>
      `}

      <!-- Error banner -->
      ${jobError && html`
        <div class="card" style="border-color: ${colors.error}44; color: ${colors.error}; margin-bottom: var(--space-4)" title=${jobError}>
          <strong>Error:</strong> <span style="word-break: break-word; white-space: pre-wrap">${jobError}</span>
        </div>
      `}

      <!-- Warmup hint: running but no KPIs yet -->
      ${showWarmupHint && html`
        <div
          class="card"
          data-testid="job-detail-warmup-hint"
          aria-live="polite"
          style=${'margin-bottom: var(--space-4); border-color: ' + palette.amber + '44; background: ' + palette.amber + '0d; display: flex; align-items: center; gap: var(--space-2); font-size: var(--font-size-sm); color: ' + palette.subtext0}
        >
          <${Spinner} size=${14} />
          <span>
            ${currentSubPhase
              ? html`Warming up — current phase <strong>${currentSubPhase}</strong>. First metrics typically arrive within 30 seconds.`
              : html`Warming up — workers are spinning up. First metrics typically arrive within 30 seconds.`
            }
          </span>
        </div>
      `}

      <!-- KPI row -->
      <!-- "Live" / "Final" tag clarifies whether the KPI numbers are still moving (running)
           or are the run's final values (completed). -->
      <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: var(--space-2)">
        <div style=${'font-size: var(--font-size-xs); font-weight: 600; color: ' + palette.overlay1 + '; text-transform: uppercase; letter-spacing: 0.06em'}>Key Metrics</div>
        ${(isRunning || isFinal) && html`
          <span
            title=${isRunning
              ? 'Numbers below are updating live — they will change until the run completes.'
              : 'Numbers below are the final values for this run.'}
            style=${'font-size: var(--font-size-xs); font-weight: 600; padding: 2px var(--space-2); border-radius: var(--radius-sm); '
              + (isRunning
                ? 'background: ' + palette.green + '22; color: ' + palette.green + '; border: 1px solid ' + palette.green + '44'
                : 'background: ' + palette.overlay0 + '22; color: ' + palette.subtext0 + '; border: 1px solid ' + palette.overlay0 + '44')}
          >${isRunning ? 'LIVE' : 'FINAL'}</span>
        `}
      </div>
      <div style="margin-bottom: var(--space-6)" title=${isRunning ? 'Live values — still updating' : (isFinal ? 'Final values for this run' : '')}>
        <${RealtimeKpiGrid} summary=${summary} slos=${slos} timeseries=${liveData.timeseries} />
        ${isFinal && results && html`
          <div style="margin-top: var(--space-4)">
            <${TokenEfficiencyCard} results=${results} info=${info} />
          </div>
        `}
      </div>

      <!-- Two-column split -->
      <div class="detail-split">
        <!-- Left: Phase progress + pods -->
        <div>
          ${showLiveRunPanels && phasesArray.length > 0 && html`
            <div class="card" style="margin-bottom: var(--space-4)">
              <div class="card-title">Phases</div>
              <!-- Sweep children with warmup + 5+ stages overflow on narrow viewports;
                   horizontal scroll keeps the layout intact instead of wrapping items
                   into broken rows. -->
              <div style="overflow-x: auto; max-width: 100%">
                <${PhaseBar} phases=${phasesArray} />
              </div>
            </div>
          `}

          ${showLiveRunPanels && Object.keys(rawPhases).length > 0 && html`
            <div style="margin-bottom: var(--space-4)">
              <${RecordProcessing} phases=${rawPhases} />
            </div>
          `}

          ${viewingCurrentRun && pods.length > 0 && html`
            <div class="card" data-testid="job-detail-pods">
              <div class="card-title">Pods</div>
              <${PodsBar} pods=${pods} />
            </div>
          `}
        </div>

        <!-- Right: Charts -->
        <div>
          ${showLiveRunPanels && chartData && html`
            <div class="card" style="margin-bottom: var(--space-4)">
              <div class="card-title">Live Throughput</div>
              <${ChartWrapper} type="line" data=${chartData} options=${throughputChartOptions} height=${200} />
            </div>
          `}

          ${isFinal && latencyHistogram && html`
            <div class="card">
              <div class="card-title">Latency Distribution</div>
              <${ChartWrapper} type="bar" data=${latencyHistogram} options=${histogramOptions} height=${200} />
            </div>
          `}
        </div>
      </div>

      <!-- Events / Logs / Conditions / Pods (tabbed) -->
      ${showLiveRunPanels && html`
        <div style="margin-top: var(--space-4)">
          <${DiagnosticsPanel}
            ns=${namespace}
            name=${name}
            conditions=${conditions}
            pods=${pods}
            mode=${viewingCurrentRun ? (isRunning ? 'live' : 'completed') : 'archived'}
            archived=${!viewingCurrentRun}
            eventCount=${null}
            logSeverityCounts=${null}
            conditionWarnCount=${(conditions || []).filter(c => c.status !== 'True').length}
            podCrashCount=${(pods || []).filter(p => /crashloop/i.test(p.reason || '')).length} />
        </div>
      `}

      <!-- Feature 6: SLA Compliance (completed only, only when SLOs declared on the CR) -->
      ${isFinal && html`<${SLACompliance} results=${results} summary=${summary} config=${jobConfig} />`}

      <!-- Server Metrics -->
      ${displayedServerMetrics
        ? html`<${ServerMetricsSection}
                 serverMetrics=${displayedServerMetrics}
                 source=${serverMetricsSource}
                 sparklines=${viewingCurrentRun ? liveData.serverTimeseries : null} />`
        : (isTerminal && files.some(f => f.name === 'server_metrics_export.json') && !serverMetricsLoaded && html`
          <div class="card" style="margin-top: var(--space-4); display: flex; align-items: center; gap: var(--space-2); min-height: 120px">
            <${Spinner} size="sm" />
            <span class="text-dim" style="font-size: var(--font-size-sm)">Loading server metrics…</span>
          </div>
        `)
      }
      ${isTerminal && serverMetricsError && html`
        <div class="card" style=${'margin-top: var(--space-4); border-color: ' + colors.error + '44; color: ' + colors.error}>
          <div class="card-title">Server Metrics</div>
          <span style="font-size: var(--font-size-sm)">${serverMetricsError}</span>
        </div>
      `}

      <!-- Job Configuration (always shown if available) -->
      ${jobConfig
        ? html`<${JobConfigSection} config=${jobConfig} namespace=${namespace} name=${name} />`
        : jobConfigLoaded
          ? html`
            <div class="card" style="margin-top: var(--space-4); min-height: 120px">
              <div class="card-title">Job Configuration</div>
              <span class="text-dim" style="font-size: var(--font-size-sm)">${jobConfigError ?? 'Job configuration unavailable.'}</span>
            </div>
          `
          : html`
            <div class="card" style="margin-top: var(--space-4); display: flex; align-items: center; gap: var(--space-2); min-height: 160px">
              <${Spinner} size="sm" />
              <span class="text-dim" style="font-size: var(--font-size-sm)">Loading job configuration…</span>
            </div>
          `
      }

      <!-- Feature 8: Run Metadata (completed only) -->
      ${isFinal && html`<${RunMetadata} status=${status} results=${results} info=${info} />`}

      <!-- Per-Record Analysis from profile_export.jsonl -->
      ${isFinal && jsonlRecords
        ? html`<${PerRecordAnalysis} records=${jsonlRecords} />`
        : (isFinal && files.some(f => f.name === 'profile_export.jsonl') && !jsonlLoaded && html`
          <div class="card" style="margin-top: var(--space-4); display: flex; align-items: center; gap: var(--space-2); min-height: 160px">
            <${Spinner} size="sm" />
            <span class="text-dim" style="font-size: var(--font-size-sm)">
              ${jsonlProgress
                ? `Parsing per-request records — ${fmtInt(jsonlProgress.done)} of ${fmtInt(jsonlProgress.total)}…`
                : 'Loading per-request records…'}
            </span>
          </div>
        `)
      }
      ${isFinal && jsonlError && html`
        <div class="card" style=${'margin-top: var(--space-4); border-color: ' + colors.error + '44; color: ' + colors.error}>
          <div class="card-title">Per-Record Analysis</div>
          <span style="font-size: var(--font-size-sm)">${jsonlError}</span>
        </div>
      `}

      <!-- Feature 3: Concurrency vs Throughput (completed only) -->
      ${isFinal && html`<${ConcurrencyThroughputChart} status=${status} />`}

      <!-- Latency percentile chart (completed only) -->
      ${isFinal && results && html`<${LatencyPercentileChart} results=${results} />`}

      <!-- Latency Timeline (completed only; needs a pinned epoch — the
           non-epoch results endpoint refuses run-scoped artifacts) -->
      ${isFinal && epoch !== undefined && html`
        <div style="margin-top: var(--space-4)">
          <${LatencyTimelineChart} ns=${namespace} name=${name} epoch=${epoch} />
        </div>
      `}

      <!-- Feature 4: ISL Distribution (completed only) -->
      ${isFinal && results && html`<${ISLDistributionChart} results=${results} />`}

      <!-- Full metrics breakdown (completed only) -->
      ${isFinal && results && html`<${MetricsTable} results=${results} />`}

      <${ArtifactsCard}
        files=${files}
        filesLoaded=${filesLoaded}
        namespace=${namespace}
        name=${name}
        epoch=${epoch}
        resolvedEpoch=${resolvedEpoch}
        isCompleted=${isFinal}
        isRunning=${isRunning}
        api=${api}
        fmtBytes=${fmtBytes}
        title="Result Files"
        testIdPrefix="job-detail-artifacts"
        cardTestId="artifacts-card"
      />
    </div>
    <style>
      @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.75); }
      }
    </style>
  `;
}
