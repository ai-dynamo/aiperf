import { html } from 'htm/preact';
import { useState, useEffect, useRef } from 'preact/hooks';
import { api, poll } from '../lib/api.js';
import { openJobWs } from '../lib/job-ws.js';
import { phaseColor, colors, palette } from '../lib/theme.js';
import { navigate } from '../lib/router.js';
import { KpiCard } from '../components/kpi-card.js';
import { ChartWrapper } from '../components/chart-wrapper.js';
import { Panel } from '../components/panel.js';
import { KpiRail } from '../components/kpi-rail.js';
import { PhaseStrip } from '../components/phase-strip.js';
import { RecordsStrip } from '../components/records-strip.js';
import { PodsStrip } from '../components/pods-strip.js';
import { LiveChartsPanel } from '../components/live-charts-panel.js';
import { DiagnosticsDrawer } from '../components/diagnostics-drawer.js';
import { LatencyTimelineChart } from '../components/latency-timeline-chart.js';
import { RunPicker } from '../components/run-picker.js';
import { NsPill, ModelPill } from '../components/pills.js';
import { LoadingPanel, Spinner } from '../components/spinner.js';
import { jobs as jobsSignal } from '../lib/state.js';
import { fmtNumber, fmtInt, fmtThroughput, fmtBytes } from '../lib/format.js';
import { ServerMetricsSection } from '../components/server-metrics/index.js';
import { RelaunchButton } from '../components/relaunch-button.js';
import { IdentityBar } from '../components/identity-bar.js';
import { RailCard, RailKv, RailAction } from '../components/job-detail-rail.js';

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
      { key: 'input_sequence_length', label: 'Input Sequence Length', cols: FULL_PERCENTILES },
      { key: 'output_sequence_length', label: 'Output Sequence Length', cols: FULL_PERCENTILES },
      { key: 'osl_mismatch_diff_pct', label: 'OSL Mismatch (diff %)', cols: FULL_PERCENTILES },
      { key: 'osl_mismatch_count', label: 'OSL Mismatch Count', cols: ['avg'] },
      { key: 'error_isl', label: 'Error ISL', cols: FULL_PERCENTILES },
      { key: 'usage_prompt_tokens', label: 'Usage Prompt Tokens', cols: FULL_PERCENTILES },
      { key: 'usage_completion_tokens', label: 'Usage Completion Tokens', cols: FULL_PERCENTILES },
      { key: 'usage_total_tokens', label: 'Usage Total Tokens', cols: FULL_PERCENTILES },
      { key: 'usage_reasoning_tokens', label: 'Usage Reasoning Tokens', cols: FULL_PERCENTILES },
      { key: 'usage_prompt_tokens_diff_pct', label: 'Usage Prompt Diff %', cols: FULL_PERCENTILES },
      { key: 'usage_completion_tokens_diff_pct', label: 'Usage Completion Diff %', cols: FULL_PERCENTILES },
      { key: 'usage_reasoning_tokens_diff_pct', label: 'Usage Reasoning Diff %', cols: FULL_PERCENTILES },
      { key: 'usage_discrepancy_count', label: 'Usage Discrepancy Count', cols: ['avg'] },
      { key: 'reasoning_token_count', label: 'Reasoning Tokens', cols: FULL_PERCENTILES },
      { key: 'output_token_count', label: 'Output Tokens', cols: FULL_PERCENTILES },
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

  return html`
    <div class="card" style="margin-top: var(--space-4)">
      <div style=${'display: flex; align-items: center; justify-content: space-between'}>
        <div class="card-title" style="margin: 0">Job Configuration</div>
        <button
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
          content=${serializeYaml({
            apiVersion: 'aiperf.nvidia.com/v1alpha1',
            kind: 'AIPerfJob',
            metadata: { name: name ?? 'aiperfjob', namespace: namespace ?? 'default' },
            spec,
          }) + '\n'}
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

// --- File Viewer Modal ---

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

// Wider sizing for file viewers (profile_export_aiperf.json and friends —
// pretty-printed JSON nested several levels deep needs the horizontal room).
const MODAL_STYLE_WIDE = MODAL_BASE_STYLE + ' max-width: 95vw; width: 1400px;';

function ModalChrome({ filename, onCopy, onDownload, onClose, copyLabel, wide, children }) {
  const modalStyle = wide ? MODAL_STYLE_WIDE : MODAL_STYLE;
  return html`
    <div style=${BACKDROP_STYLE} onclick=${e => { if (e.target === e.currentTarget) onClose(); }}>
      <div style=${modalStyle}>
        <div style=${'display: flex; align-items: center; justify-content: space-between; padding: var(--space-3) var(--space-4); border-bottom: 1px solid ' + palette.surface0 + '; flex-shrink: 0'}>
          <span style=${'font-size: var(--font-size-sm); font-weight: 600; color: ' + palette.text + '; font-family: monospace'}>${filename}</span>
          <div style="display: flex; gap: var(--space-2); align-items: center">
            ${onCopy && html`
              <button
                onclick=${onCopy}
                style=${'background: ' + palette.teal + '22; color: ' + palette.teal + '; border: 1px solid ' + palette.teal + '44; padding: var(--space-1) var(--space-3); border-radius: var(--radius-md); cursor: pointer; font-size: var(--font-size-xs)'}
              >${copyLabel ?? 'Copy'}</button>
            `}
            <button
              onclick=${onDownload}
              style=${'background: ' + palette.blue + '22; color: ' + palette.blue + '; border: 1px solid ' + palette.blue + '44; padding: var(--space-1) var(--space-3); border-radius: var(--radius-md); cursor: pointer; font-size: var(--font-size-xs)'}
            >Download</button>
            <button
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

function syntaxHighlight(json) {
  // Split formatted JSON into tokens and wrap with color spans.
  // Returns array of {text, color} objects.
  const tokens = [];
  const re = /("(?:[^"\\]|\\.)*")\s*:|("(?:[^"\\]|\\.)*")|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|(\btrue\b|\bfalse\b)|(\bnull\b)|([\[\]{},])|(\s+)/g;
  let match;
  let lastIndex = 0;
  while ((match = re.exec(json)) !== null) {
    if (match.index > lastIndex) {
      tokens.push({ text: json.slice(lastIndex, match.index), color: null });
    }
    if (match[1] !== undefined) {
      // object key (includes trailing `:` from the pattern)
      tokens.push({ text: match[0], color: palette.mauve });
    } else if (match[2] !== undefined) {
      // string value
      tokens.push({ text: match[2], color: palette.green });
    } else if (match[3] !== undefined) {
      // number
      tokens.push({ text: match[3], color: palette.peach });
    } else if (match[4] !== undefined) {
      // boolean
      tokens.push({ text: match[4], color: palette.blue });
    } else if (match[5] !== undefined) {
      // null
      tokens.push({ text: match[5], color: palette.overlay0 });
    } else {
      // punctuation or whitespace - no color
      tokens.push({ text: match[0], color: null });
    }
    lastIndex = re.lastIndex;
  }
  if (lastIndex < json.length) {
    tokens.push({ text: json.slice(lastIndex), color: null });
  }
  return tokens;
}

function parseCSV(text) {
  // Simple CSV parser: handles quoted fields with embedded commas/newlines.
  const rows = [];
  const lines = text.split('\n');
  for (const line of lines) {
    if (line.trim() === '') continue;
    const cols = [];
    let cur = '';
    let inQuote = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') {
        if (inQuote && line[i + 1] === '"') { cur += '"'; i++; }
        else { inQuote = !inQuote; }
      } else if (ch === ',' && !inQuote) {
        cols.push(cur);
        cur = '';
      } else {
        cur += ch;
      }
    }
    cols.push(cur);
    rows.push(cols);
  }
  return rows;
}

function stripAnsi(text) {
  // Remove ANSI escape sequences (color codes, cursor control, etc.)
  return text.replace(/\x1b\[[0-9;]*[mGKHFJ]/g, '');
}

// Generic file viewer modal: dispatches to JSON/CSV/TXT renderers based on extension.
function FileViewerModal({ filename, url, onClose }) {
  const [rawContent, setRawContent] = useState(null);
  const [parsedJson, setParsedJson] = useState(null);
  const [copyLabel, setCopyLabel] = useState('Copy');
  const ext = filename.split('.').pop().toLowerCase();

  useEffect(() => {
    if (ext === 'json') {
      fetch(url)
        .then(r => r.json())
        .then(d => { setParsedJson(d); setRawContent(JSON.stringify(d, null, 2)); })
        .catch(() => { setRawContent(null); });
    } else {
      fetch(url)
        .then(r => r.text())
        .then(t => setRawContent(t))
        .catch(() => setRawContent(null));
    }
  }, [url, ext]);

  function handleDownload() {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
  }

  function handleCopy() {
    if (rawContent == null) return;
    navigator.clipboard.writeText(rawContent).then(() => {
      setCopyLabel('Copied!');
      setTimeout(() => setCopyLabel('Copy'), 2000);
    });
  }

  let body;
  if (rawContent == null) {
    body = html`<span style="display: inline-flex; align-items: center; gap: var(--space-2)"><${Spinner} size=${14} /><span class="text-dim">Loading file…</span></span>`;
  } else if (ext === 'json') {
    const tokens = syntaxHighlight(rawContent);
    body = html`
      <pre style=${'margin: 0; font-family: monospace; font-size: var(--font-size-xs); line-height: 1.6; white-space: pre; color: ' + palette.text}>
        ${tokens.map((t, i) =>
          t.color
            ? html`<span key=${i} style=${'color: ' + t.color}>${t.text}</span>`
            : t.text
        )}
      </pre>
    `;
  } else if (ext === 'csv') {
    const rows = parseCSV(rawContent);
    if (rows.length === 0) {
      body = html`<span class="text-dim">Empty file</span>`;
    } else {
      const header = rows[0];
      const dataRows = rows.slice(1);
      body = html`
        <div style="overflow-x: auto">
          <table style=${'border-collapse: collapse; font-size: var(--font-size-xs); font-family: monospace; min-width: 100%'}>
            <thead>
              <tr>
                ${header.map((col, i) => html`
                  <th key=${i} style=${'padding: var(--space-2) var(--space-3); text-align: left; font-weight: 700; color: ' + palette.text + '; background: ' + palette.surface0 + '; border-bottom: 2px solid ' + palette.surface1 + '; white-space: nowrap'}>${col}</th>
                `)}
              </tr>
            </thead>
            <tbody>
              ${dataRows.map((row, ri) => html`
                <tr key=${ri} style=${'background: ' + (ri % 2 === 0 ? palette.base : palette.mantle)}>
                  ${row.map((cell, ci) => html`
                    <td key=${ci} style=${'padding: var(--space-1) var(--space-3); color: ' + palette.text + '; border-bottom: 1px solid ' + palette.surface0 + '; white-space: nowrap'}>${cell}</td>
                  `)}
                </tr>
              `)}
            </tbody>
          </table>
        </div>
      `;
    }
  } else {
    // txt or ansi: strip ANSI codes and show as plain monospace text
    const plain = ext === 'ansi' ? stripAnsi(rawContent) : rawContent;
    body = html`
      <pre style=${'margin: 0; font-family: monospace; font-size: var(--font-size-xs); line-height: 1.6; white-space: pre; color: ' + palette.text + '; tab-size: 4'}>${plain}</pre>
    `;
  }

  return html`
    <${ModalChrome}
      filename=${filename}
      onCopy=${handleCopy}
      onDownload=${handleDownload}
      onClose=${onClose}
      copyLabel=${copyLabel}
      wide=${true}
    >
      ${body}
    </${ModalChrome}>
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
    if (/^[\w./:@\-+]+$/.test(obj) && !/^(true|false|null|~)$/i.test(obj) && !/^-?\d+(\.\d+)?$/.test(obj)) {
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

// Spec viewer modal: in-memory YAML content (no URL fetch). Mirrors
// FileViewerModal's chrome but owns its own Escape listener so it's
// self-contained — JobConfigSection state is local to that component.
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
  // Flips true when the live poll throws and has not yet recovered. Kept
  // distinct from ``error`` (which only renders the full-page block on
  // first-load failure) so an outage that hits an already-loaded page
  // can downgrade the green "Live" pulse to an amber "Stale" badge
  // without nuking the rest of the rendered state.
  const [liveStale, setLiveStale] = useState(false);
  const [serverMetrics, setServerMetrics] = useState(null);
  const [serverMetricsLoaded, setServerMetricsLoaded] = useState(false);
  const [serverMetricsError, setServerMetricsError] = useState(null);
  const [fileViewer, setFileViewer] = useState(null); // { filename, url }
  const [jsonlRecords, setJsonlRecords] = useState(null);
  const [jsonlLoaded, setJsonlLoaded] = useState(false);
  const [jsonlError, setJsonlError] = useState(null);
  // Progress for the JSONL parse so users see a count tick up instead of a
  // multi-second blank skeleton on 50k+ row exports.
  const [jsonlProgress, setJsonlProgress] = useState(null);
  const [jobConfig, setJobConfig] = useState(null);
  const [epochs, setEpochs] = useState([]);
  // Cancel-button state: 'idle' shows the button, 'confirm' shows an inline
  // confirm/abort pair, 'pending' disables both while the API call is in flight.
  // Replaces native confirm()/alert() which provided no in-flight feedback and
  // let users double-click to fire two cancels.
  const [cancelState, setCancelState] = useState('idle');
  const [cancelError, setCancelError] = useState(null);
  // Diagnostics drawer is opened from the rail's actions card or by a
  // ``?diag=...`` URL param (deep-link from log strips, condition rows,
  // etc.). When closed, the entire DiagnosticsDrawer (and the wrapped
  // DiagnosticsPanel) is unmounted to keep the live event-stream
  // websocket from running off-screen.
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(() => {
    if (typeof window === 'undefined') return false;
    return new URL(window.location.href).searchParams.get('diag') != null;
  });

  const PREVIEWABLE = new Set(['json', 'csv', 'txt', 'ansi']);

  const resultsBase = epoch
    ? `/api/v1/results/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}/runs/${encodeURIComponent(epoch)}`
    : null;

  // Close file viewer on Escape
  useEffect(() => {
    function onKeyDown(e) {
      if (e.key === 'Escape') setFileViewer(null);
    }
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, []);

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

  // Open the per-job WebSocket whenever the run is active AND the URL points
  // at the currently-running epoch — either the no-epoch live URL, or
  // /runs/<currentRunEpoch> (which is what every dashboard/history link
  // produces via buildJobPath). Pinned views of *past* archived epochs of
  // a now-rerunning job skip the WS so live current-run stats don't bleed
  // into the archived render. The proxy refuses non-running CRs anyway,
  // but gating here saves a connect/4404/reconnect loop.
  const livePhaseLower = (job?.job?.phase ?? job?.status?.phase ?? '').toLowerCase();
  const liveRunEpoch = job?.status?.runEpoch != null ? String(job.status.runEpoch) : null;
  const viewingCurrentRun = epoch === undefined
    || (liveRunEpoch != null && epoch === liveRunEpoch);
  const wsActive = livePhaseLower === 'running' && viewingCurrentRun;
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

    poll(
      async () => {
        let data;
        try {
          data = await api.getJob(namespace, name, epoch);
        } catch (e) {
          // Transport/5xx — keep prior ``job`` rendered, but flip the
          // header indicator to "Stale" so the user knows the live
          // numbers are frozen. Re-throw so poll()'s shared health gate
          // can raise the app-level banner once enough ticks fail.
          setLiveStale(true);
          throw e;
        }
        setJob(data);
        setError(null);
        setLiveStale(false);

        // Terminal-state detection. Two signals, ORed:
        //   1) Recognized terminal phase string. Includes ``archived``
        //      because ``job_union._archived_from_summary`` stamps that
        //      literal phase whenever the response is built from the PVC
        //      summary (no live CR — i.e. the job has finished and the CR
        //      has been removed, or a non-latest epoch was requested).
        //   2) Non-null ``completionTime`` on the job summary. The
        //      operator only writes this once the run is over, so it's a
        //      reliable backstop for any future phase string we haven't
        //      enumerated here yet.
        const phase = (data?.job?.phase ?? data?.status?.phase ?? '').toLowerCase();
        const terminalPhases = new Set([
          'completed', 'succeeded', 'failed', 'error',
          'cancelled', 'canceled', 'partiallyfailed', 'archived',
        ]);
        const completionTime = data?.job?.completionTime ?? data?.status?.completionTime ?? null;
        const done = terminalPhases.has(phase) || completionTime != null;
        if (done) setPolling(false);

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
      ac.signal,
    );

    // Fetch job config (original CR spec)
    fetch(`/api/v1/config/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}`, { signal: ac.signal })
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (d) setJobConfig(d); })
      .catch(() => {});

    // Final artifacts are run-scoped. Do not hit the non-epoch results
    // endpoint; wait until the route is pinned to /runs/<epoch>.
    if (!resultsBase) {
      setFilesLoaded(true);
      setServerMetricsLoaded(true);
      setJsonlLoaded(true);
    } else {
      fetch(resultsBase, { signal: ac.signal })
        .then(r => r.ok ? r.json() : null)
        .then(d => {
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
              .then(r => r.ok ? r.json() : null)
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
          // Per-request records (profile_export.jsonl) intentionally NOT
          // auto-fetched. At high concurrency these files reach hundreds of
          // MB compressed / multiple GB decompressed, which OOM-kills the
          // browser tab. Users who need the data download the file.
          setJsonlLoaded(true);
        })
        .catch(() => {
          setFilesLoaded(true);
          setServerMetricsLoaded(true);
          setJsonlLoaded(true);
        });
    }

    return () => ac.abort();
  }, [namespace, name, epoch, resultsBase]);

  function humanSize(bytes) {
    return fmtBytes(bytes);
  }

  function downloadFile(fileName) {
    const url = `${resultsBase}/${encodeURIComponent(fileName)}`;
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    a.click();
  }

  function openFile(fileName) {
    const url = `${resultsBase}/${encodeURIComponent(fileName)}`;
    const ext = fileName.split('.').pop().toLowerCase();
    if (PREVIEWABLE.has(ext)) {
      setFileViewer({ filename: fileName, url });
    } else {
      const a = document.createElement('a');
      a.href = url;
      a.download = fileName;
      a.click();
    }
  }

  function downloadAll() {
    // Browsers throttle synthetic download clicks from one user gesture;
    // space them out so all files actually start downloading.
    files.forEach((f, i) => {
      setTimeout(() => downloadFile(f.name), i * 300);
    });
  }

  function exportJson() {
    const exportFile = files.find(f => f.name === 'profile_export_aiperf.json');
    if (exportFile) downloadFile(exportFile.name);
  }

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

  const phase = info.phase ?? status.phase ?? 'Unknown';
  const phaseClr = phaseColor(phase);
  const model = info.model ?? '---';
  const endpointUrl = info.endpoint ?? null;
  const startTime = info.startTime ?? status.startTime;
  const isRunning = phase.toLowerCase() === 'running';
  // ``archived`` covers responses sourced purely from the PVC summary
  // (CR has been deleted, or a non-latest epoch was requested) — see
  // ``job_union._archived_from_summary``. We treat it as a successful
  // completion so the Final KPIs, SLA, server-metrics, and per-record
  // panels render the same way as a live ``Completed`` CR; the alternative
  // hides perfectly good results behind a phase string the page never
  // generated itself.
  const isCompleted = phase.toLowerCase() === 'completed'
    || phase.toLowerCase() === 'succeeded'
    || phase.toLowerCase() === 'archived';
  const phaseLower = phase.toLowerCase();
  const isCancelled = phaseLower === 'cancelled' || phaseLower === 'canceled';
  const isPartiallyFailed = phaseLower === 'partiallyfailed';
  // Terminal phases that still surface "Final" KPIs and stop the live polling loop —
  // includes cancelled/partial so the page doesn't get stuck pretending to poll forever.
  const isTerminal = isCompleted || isCancelled || isPartiallyFailed || phaseLower === 'failed' || phaseLower === 'error';
  // Strip-mode signal passed to KPI / phase / pods strips. ``archived``
  // wins over ``completed`` so kpi-rail tiles label values "archived"
  // instead of "final" when the response is summary-only — distinguishes
  // "live CR finished, results just landed" from "no CR exists, results
  // are persisted history".
  const stripMode = phaseLower === 'archived'
    ? 'archived'
    : isCompleted
      ? 'completed'
      : !viewingCurrentRun
        ? 'archived'
        : 'live';
  const liveServerMetricsBase = viewingCurrentRun ? status.serverMetrics : null;
  const liveServerMetrics = (liveData.connected && liveData.serverSummary)
    ? liveData.serverSummary
    : liveServerMetricsBase;
  const displayedServerMetrics = serverMetrics || liveServerMetrics;
  const serverMetricsSource = serverMetrics ? 'final' : 'live';

  useEffect(() => {
    if (epoch !== undefined || resolvedEpoch == null) return;
    navigate(`/jobs/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}/runs/${encodeURIComponent(resolvedEpoch)}`);
  }, [epoch, resolvedEpoch, namespace, name]);

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
    targetConcurrency: p?.targetConcurrency ?? p?.concurrency ?? null,
    recordsSuccess: p?.recordsSuccess ?? 0,
    requestsTotal: p?.requestsTotal ?? 0,
  }));
  // PhaseStrip needs {name, status, progress}. Derive from completed/total.
  const phaseStripData = phasesArray.map((ph) => {
    const done = ph.total > 0 && ph.completed >= ph.total;
    const active = !done && ph.completed > 0;
    const progress = ph.total > 0 ? Math.min(1, ph.completed / ph.total) : 0;
    return {
      name: ph.name,
      status: done ? 'completed' : active ? 'active' : 'pending',
      progress,
      targetConcurrency: ph.targetConcurrency,
      recordsSuccess: ph.recordsSuccess,
      requestsTotal: ph.requestsTotal,
    };
  });
  const currentPhaseName = (phaseStripData.find((p) => p.status === 'active') || {}).name
    ?? info.currentPhase
    ?? status.phase
    ?? null;
  const etaText = null;

  // Records-pipeline aggregates (formerly inside RecordProcessing). RecordsStrip
  // wants flat numbers; sum success+error across phases for processed/total,
  // sum recordsPerSecond across active phases for rate, take the longest
  // active recordsEtaSeconds for the headline ETA.
  let recordProcessed = 0;
  let recordTotal = 0;
  let recordRate = 0;
  let recordEta = null;
  for (const [, p] of Object.entries(rawPhases)) {
    if (p == null || typeof p !== 'object') continue;
    const rs = p.recordsSuccess ?? 0;
    const re = p.recordsError ?? 0;
    recordProcessed += rs + re;
    // Total target = requestsTotal (records pipeline trails requests 1:1).
    recordTotal += p.requestsTotal ?? 0;
    const sendingComplete = p.sendingComplete ?? false;
    const recPct = p.recordsProgressPercent ?? 0;
    const isActive = !sendingComplete || recPct < 100;
    if (isActive) {
      recordRate += p.recordsPerSecond ?? p.records_per_second ?? 0;
      const eta = p.recordsEtaSeconds ?? p.records_eta_seconds ?? null;
      if (eta != null && (recordEta == null || eta > recordEta)) {
        recordEta = eta;
      }
    }
  }
  if (recordRate === 0) recordRate = null;
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

  const hasExportFile = files.some(f => f.name === 'profile_export_aiperf.json');
  // Sum file sizes for the "Download .zip (N MB)" label. Files without a known
  // size (older API shapes) contribute 0; the label hides the size suffix
  // entirely when the total is 0 so we never show a misleading "0 B" badge.
  const totalArtifactBytes = files.reduce((s, f) => s + (Number(f.size_bytes) || 0), 0);

  // Warmup hint: running, but no live KPI numbers yet — typical for the first ~30s
  // while the workers ramp and TimingManager hasn't issued enough credits to populate
  // any percentile. Without this hint, all-`---` KPIs read as "broken" instead of "soon".
  const noKpisYet = throughput == null && ttftAvg == null && latP99 == null && outputTokenThroughput == null;
  const showWarmupHint = isRunning && noKpisYet;
  const currentSubPhase = info.currentPhase ?? status.currentPhase ?? null;

  function fileColor(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    if (ext === 'json' || ext === 'jsonl') return palette.mauve;
    if (ext === 'csv') return palette.teal;
    if (ext === 'txt' || ext === 'ansi') return palette.blue;
    return palette.overlay1;
  }

  // Per-extension chip that renders before each filename in the artifact list.
  // Label is short (the extension, uppercase); color tints the background +
  // border so users can scan the table by type. Unknown extensions fall back
  // to a neutral grey so we never silently drop the chip.
  function fileTypeChip(filename) {
    const ext = (filename.split('.').pop() || '').toLowerCase();
    const TYPES = {
      json:    { label: 'JSON',    color: palette.yellow },
      jsonl:   { label: 'JSONL',   color: palette.peach },
      csv:     { label: 'CSV',     color: palette.green },
      parquet: { label: 'PARQUET', color: palette.lavender },
      txt:     { label: 'TXT',     color: palette.blue },
      log:     { label: 'LOG',     color: palette.sapphire },
      ansi:    { label: 'ANSI',    color: palette.sky },
      yaml:    { label: 'YAML',    color: palette.teal },
      yml:     { label: 'YAML',    color: palette.teal },
      html:    { label: 'HTML',    color: palette.pink },
      htm:     { label: 'HTML',    color: palette.pink },
      zip:     { label: 'ZIP',     color: palette.overlay1 },
      gz:      { label: 'GZ',      color: palette.overlay1 },
      tar:     { label: 'TAR',     color: palette.overlay1 },
      png:     { label: 'PNG',     color: palette.mauve },
      jpg:     { label: 'JPG',     color: palette.mauve },
      jpeg:    { label: 'JPG',     color: palette.mauve },
      svg:     { label: 'SVG',     color: palette.mauve },
    };
    return TYPES[ext] ?? { label: (ext || 'FILE').toUpperCase().slice(0, 6), color: palette.overlay1 };
  }

  // Live / Stale / Completed status indicator rendered in the identity bar's
  // actions row. Pulled out so the IdentityBar prop tree stays readable.
  const liveIndicator = polling
    ? liveStale
      ? html`
        <span
          title="Live updates paused — operator API is not responding. Retrying in the background; numbers shown are from the last successful poll."
          data-testid="job-detail-live-stale"
          style=${'display: inline-flex; align-items: center; gap: var(--space-1); font-size: var(--font-size-xs); color: ' + palette.amber}
        >
          <span style=${'display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: ' + palette.amber} />
          Stale
        </span>
      `
      : html`
        <span
          data-testid="job-detail-live"
          style="display: inline-flex; align-items: center; gap: var(--space-1); font-size: var(--font-size-xs); color: ${palette.green}"
        >
          <span style=${'display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: ' + palette.green + '; animation: pulse 1.5s ease-in-out infinite'} />
          Live
        </span>
      `
    : isCompleted
      ? phaseLower === 'archived'
        ? html`<span style=${'font-size: var(--font-size-xs); color: ' + palette.subtext0 + '; opacity: 0.85'} title="Run finished and the live CR has been archived — values shown come from the persisted summary.">Archived</span>`
        : html`<span style=${'font-size: var(--font-size-xs); color: ' + palette.green + '; opacity: 0.7'}>Completed</span>`
      : isCancelled
        ? html`<span style=${'font-size: var(--font-size-xs); color: ' + palette.subtext0 + '; opacity: 0.85'} title="Run was cancelled before completion — KPIs reflect partial data.">Cancelled</span>`
        : isPartiallyFailed
          ? html`<span style=${'font-size: var(--font-size-xs); color: ' + colors.error + '; opacity: 0.85'} title="Run finished but some workers failed — KPIs reflect surviving data.">Partially failed</span>`
          : null;

  const cancelControls = isRunning ? html`
    ${cancelState === 'idle' && html`
      <button
        class="btn btn--danger"
        onclick=${() => setCancelState('confirm')}
        style=${'background: ' + colors.error + '22; color: ' + colors.error + '; border: 1px solid ' + colors.error + '44; padding: var(--space-1) var(--space-3); border-radius: var(--radius-md); cursor: pointer; font-size: var(--font-size-xs)'}
        data-testid="job-detail-cancel"
        title="Stop the running benchmark. The AIPerfJob CR is kept; controller pod is terminated."
      >
        Cancel
      </button>
    `}
    ${cancelState === 'confirm' && html`
      <span style=${'display: inline-flex; align-items: center; gap: var(--space-2); padding: var(--space-1) var(--space-2); background: ' + colors.error + '11; border: 1px solid ' + colors.error + '44; border-radius: var(--radius-md); font-size: var(--font-size-xs)'}>
        <span style=${'color: ' + colors.error}>Stop run?</span>
        <button
          onclick=${handleCancel}
          style=${'background: ' + colors.error + '; color: white; border: none; padding: 2px var(--space-2); border-radius: var(--radius-sm); cursor: pointer; font-size: var(--font-size-xs)'}
          data-testid="job-detail-cancel-confirm"
        >Yes</button>
        <button
          onclick=${() => { setCancelState('idle'); setCancelError(null); }}
          style=${'background: transparent; color: ' + palette.subtext0 + '; border: 1px solid ' + palette.overlay0 + '44; padding: 2px var(--space-2); border-radius: var(--radius-sm); cursor: pointer; font-size: var(--font-size-xs)'}
        >No</button>
      </span>
    `}
    ${cancelState === 'pending' && html`
      <button
        disabled
        style=${'background: ' + colors.error + '22; color: ' + colors.error + '; border: 1px solid ' + colors.error + '44; padding: var(--space-1) var(--space-3); border-radius: var(--radius-md); cursor: not-allowed; font-size: var(--font-size-xs); display: inline-flex; align-items: center; gap: var(--space-1); opacity: 0.7'}
        data-testid="job-detail-cancel"
      >
        <${Spinner} size=${10} />
        Cancelling…
      </button>
    `}
    ${cancelError && html`
      <span style=${'font-size: var(--font-size-xs); color: ' + colors.error}>Cancel failed: ${cancelError}</span>
    `}
  ` : null;

  const identityActions = html`
    <${NsPill} ns=${namespace} onClick=${ns => navigate('/jobs?ns=' + encodeURIComponent(ns))} testId="job-detail-ns-pill" />
    ${model && html`<${ModelPill} model=${model} onClick=${m => navigate('/jobs?model=' + encodeURIComponent(m))} testId="job-detail-model-pill" />`}
    ${model && model !== '---' && html`<${SimilarRunsLink} namespace=${namespace} model=${model} currentName=${name} />`}
    ${liveIndicator}
    <${RunPicker} namespace=${namespace} name=${name} epochs=${epochs} current=${epoch} onPick=${pickEpoch} />
    ${cancelControls}
    ${isTerminal && jobConfig?.spec && html`<${RelaunchButton} namespace=${namespace} name=${name} config=${jobConfig} />`}
  `;

  const sweepLineBeforeKv = info.sweepName ? html`
    <p class="text-dim" data-testid="job-detail-sweep-link" style="margin: var(--space-1) 0 0 0; font-size: var(--font-size-xs)">
      Part of sweep
      <a href=${`#/sweeps/${encodeURIComponent(namespace)}/${encodeURIComponent(info.sweepName)}`}
         onclick=${e => { e.preventDefault(); navigate(`/sweeps/${encodeURIComponent(namespace)}/${encodeURIComponent(info.sweepName)}`); }}>
        ${info.sweepName}
      </a>
      ${info.variationLabel && html` — variation ${info.variationLabel}`}
    </p>
  ` : null;

  // Pre-formatted elapsed text for the identity-bar KV strip. Mirrors the
  // existing RelativeTime "elapsed" mode but as a static string so it
  // sits cleanly inside the KV value slot.
  const elapsedText = startTime ? (() => {
    try {
      const ms = Date.now() - new Date(startTime).getTime();
      return formatDuration(ms);
    } catch { return null; }
  })() : null;

  const runLabel = epoch != null ? String(epoch) : (resolvedEpoch != null ? String(resolvedEpoch) : 'live');

  // Rail: Phase / Pods / Records / SLA / Config / Sweep / Actions cards.
  // Built once per render; cards self-gate on data presence.
  const railContent = html`
    ${phaseStripData.length > 0 && html`
      <${RailCard} title="Phase" testId="job-detail-rail-phase">
        <${PhaseStrip} phases=${phaseStripData} current=${currentPhaseName} etaText=${etaText} />
      <//>
    `}
    ${viewingCurrentRun
      ? html`
        <${RailCard} title="Pods" testId="job-detail-rail-pods">
          <div data-testid="job-detail-pods">
            <${PodsStrip} pods=${pods} onExpand=${() => {
              const url = new URL(window.location.href);
              url.searchParams.set('diag', 'pods');
              window.history.replaceState(null, '', url.toString());
              setDiagnosticsOpen(true);
            }} />
          </div>
        <//>
      `
      : html`
        <${RailCard} title="Pods" testId="job-detail-rail-pods">
          <div data-testid="job-detail-archived-note" class="text-dim" style="font-style: italic; font-size: var(--font-size-xs)">
            Pods and events are not retained for archived epochs.
          </div>
        <//>
      `
    }
    ${recordTotal > 0 && html`
      <${RailCard} title="Records" testId="job-detail-rail-records">
        <${RecordsStrip}
          processed=${recordProcessed}
          total=${recordTotal}
          ratePerSec=${recordRate}
          etaSeconds=${recordEta} />
      <//>
    `}
    ${isCompleted && html`
      <${RailCard} title="SLA Compliance" testId="job-detail-rail-sla">
        <${SLACompliance} results=${results} summary=${summary} config=${jobConfig} />
      <//>
    `}
    ${info.sweepName && html`
      <${RailCard} title="Sweep" testId="job-detail-rail-sweep">
        <${RailKv} k="name" v=${info.sweepName} />
        ${info.variationLabel && html`<${RailKv} k="variation" v=${info.variationLabel} />`}
        <a href=${`#/sweeps/${encodeURIComponent(namespace)}/${encodeURIComponent(info.sweepName)}`}
           onclick=${e => { e.preventDefault(); navigate(`/sweeps/${encodeURIComponent(namespace)}/${encodeURIComponent(info.sweepName)}`); }}
           class="rail-action">
          <span class="rail-action__gly">↗</span>
          <span>open sweep view</span>
        </a>
      <//>
    `}
    <${RailCard} title="Actions" testId="job-detail-rail-actions">
      <div class="rail-actions">
        ${epoch != null && html`
          <${RailAction}
            icon="⤓"
            label=${'Download artifacts' + (totalArtifactBytes > 0 ? ` (${fmtBytes(totalArtifactBytes)})` : '')}
            href=${api.resultBundleUrl(namespace, name, epoch)}
            testId="job-detail-rail-download" />
        `}
        ${model && model !== '---' && html`
          <${RailAction}
            icon="⊞"
            label="Compare to similar runs"
            onClick=${() => navigate('/compare?cluster=' + encodeURIComponent(`${namespace} · ${model}`))} />
        `}
        <${RailAction}
          icon="ⓘ"
          label="Open diagnostics"
          onClick=${() => setDiagnosticsOpen(true)}
          testId="job-detail-rail-open-diagnostics" />
      </div>
    <//>
  `;

  return html`
    <div class="job-detail" data-testid="page-job-detail">
      <${IdentityBar}
        name=${name}
        namespace=${namespace}
        phase=${phase}
        model=${model}
        runLabel=${runLabel}
        elapsed=${elapsedText}
        endpointUrl=${endpointUrl}
        info=${info}
        actions=${identityActions}
        beforeKv=${sweepLineBeforeKv} />

      <div class="job-detail__body">
        <main class="job-detail__main">

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

      <!-- KPI rail -->
      <!-- "Live" / "Final" tag clarifies whether the KPI numbers are still moving (running)
           or are the run's final values (completed). -->
      <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: var(--space-2)">
        <div style=${'font-size: var(--font-size-xs); font-weight: 600; color: ' + palette.overlay1 + '; text-transform: uppercase; letter-spacing: 0.06em'}>Key Metrics</div>
        ${(isRunning || isCompleted) && html`
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
      <div style="margin-bottom: var(--space-4)" title=${isRunning ? 'Live values — still updating' : (isCompleted ? 'Final values for this run' : '')}>
        <${KpiRail}
          summary=${summary}
          slos=${slos}
          timeseries=${liveData.timeseries}
          pods=${pods}
          phases=${phaseStripData}
          serverSummary=${liveServerMetrics}
          serverTimeseries=${liveData.serverTimeseries}
          mode=${stripMode}
          stale=${liveData.connected === false} />
        ${isCompleted && results && html`
          <div style="margin-top: var(--space-4)">
            <${TokenEfficiencyCard} results=${results} info=${info} />
          </div>
        `}
      </div>

      <!-- Canonical strips relocated to the right rail. -->

      <!-- Live charts (running only). The diagnostics panel moved into a
           slide-in drawer; previously co-rendered here in a 2-col grid. -->
      <div style="margin-bottom: var(--space-4)">
        <${LiveChartsPanel}
          mode=${stripMode}
          throughputChartData=${chartData}
          throughputChartOptions=${throughputChartOptions}
          histogramChartData=${latencyHistogram}
          histogramChartOptions=${histogramOptions}
          windowLabel=${isCompleted ? 'whole run' : 'last 60s · auto'} />
      </div>

      <!-- SLA Compliance has moved into the right rail; the in-main panel
           is intentionally dropped to avoid duplication. -->

      <!-- Server Metrics -->
      ${displayedServerMetrics
        ? html`
          <div style="margin-top: var(--space-4)">
            <${Panel} title="Server Metrics" collapsible defaultOpen=${isCompleted} testId="panel-server-metrics">
              <${ServerMetricsSection}
                serverMetrics=${displayedServerMetrics}
                source=${serverMetricsSource}
                sparklines=${viewingCurrentRun ? liveData.serverTimeseries : null} />
            <//>
          </div>
        `
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
        ? html`
          <div style="margin-top: var(--space-4)">
            <${Panel} title="Job Configuration" collapsible defaultOpen=${isCompleted} testId="panel-job-config">
              <${JobConfigSection} config=${jobConfig} namespace=${namespace} name=${name} />
            <//>
          </div>
        `
        : html`
          <div class="card" style="margin-top: var(--space-4); display: flex; align-items: center; gap: var(--space-2); min-height: 160px">
            <${Spinner} size="sm" />
            <span class="text-dim" style="font-size: var(--font-size-sm)">Loading job configuration…</span>
          </div>
        `
      }

      <!-- Run Metadata (completed only) -->
      ${isCompleted && html`
        <div style="margin-top: var(--space-4)">
          <${Panel} title="Run Metadata" collapsible defaultOpen=${isCompleted} testId="panel-run-metadata">
            <${RunMetadata} status=${status} results=${results} info=${info} />
          <//>
        </div>
      `}

      <!-- Per-Record Analysis: never auto-fetched (large runs OOM the browser).
           Show a static card with a download link when the file is present. -->
      ${isCompleted && (() => {
        const f = files.find(x => x.name === 'profile_export.jsonl');
        if (!f) return null;
        return html`
          <div style="margin-top: var(--space-4)">
            <${Panel} title="Per-Record Analysis" collapsible defaultOpen=${isCompleted} testId="panel-per-record">
              <div style="font-size: var(--font-size-sm); color: var(--text-dim); margin-bottom: var(--space-2)">
                ${humanSize(f.size_bytes)} compressed. Not loaded in-browser — download to analyze offline.
              </div>
              <button class="btn btn-secondary" onclick=${() => downloadFile('profile_export.jsonl')}>
                Download profile_export.jsonl
              </button>
            <//>
          </div>
        `;
      })()}

      <!-- Concurrency vs Throughput (completed only) -->
      ${isCompleted && html`
        <div style="margin-top: var(--space-4)">
          <${Panel} title="Concurrency vs Throughput" collapsible defaultOpen=${isCompleted} testId="panel-concurrency-throughput">
            <${ConcurrencyThroughputChart} status=${status} />
          <//>
        </div>
      `}

      <!-- Latency Percentiles (completed only) -->
      ${isCompleted && results && html`
        <div style="margin-top: var(--space-4)">
          <${Panel} title="Latency Percentiles" collapsible defaultOpen=${isCompleted} testId="panel-latency-percentiles">
            <${LatencyPercentileChart} results=${results} />
          <//>
        </div>
      `}

      <!-- Latency Timeline (completed only; needs a pinned epoch — the
           non-epoch results endpoint refuses run-scoped artifacts).
           Skipped when profile_export.jsonl is too big to parse safely
           (large runs would OOM the tab). -->
      ${isCompleted && epoch !== undefined && (() => {
        const f = files.find(x => x.name === 'profile_export.jsonl');
        const LATENCY_CHART_MAX_BYTES = 10 * 1024 * 1024;  // 10 MB compressed
        if (f && f.size_bytes > LATENCY_CHART_MAX_BYTES) {
          return html`
            <div style="margin-top: var(--space-4)">
              <${Panel} title="Latency Timeline" collapsible defaultOpen=${isCompleted} testId="panel-latency-timeline">
                <span class="text-dim" style="font-size: var(--font-size-sm)">
                  Skipped — profile_export.jsonl is ${humanSize(f.size_bytes)} compressed
                  (chart loads up to ${humanSize(LATENCY_CHART_MAX_BYTES)}).
                </span>
              <//>
            </div>
          `;
        }
        return html`
          <div style="margin-top: var(--space-4)">
            <${Panel} title="Latency Timeline" collapsible defaultOpen=${isCompleted} testId="panel-latency-timeline">
              <${LatencyTimelineChart} ns=${namespace} name=${name} epoch=${epoch} />
            <//>
          </div>
        `;
      })()}

      <!-- ISL Distribution (completed only) -->
      ${isCompleted && results && html`
        <div style="margin-top: var(--space-4)">
          <${Panel} title="ISL Distribution" collapsible defaultOpen=${isCompleted} testId="panel-isl-distribution">
            <${ISLDistributionChart} results=${results} />
          <//>
        </div>
      `}

      <!-- Full Metrics Breakdown (completed only) -->
      ${isCompleted && results && html`
        <div style="margin-top: var(--space-4)">
          <${Panel} title="Full Metrics Breakdown" collapsible defaultOpen=${isCompleted} testId="panel-metrics-table">
            <${MetricsTable} results=${results} />
          <//>
        </div>
      `}

      <!-- Artifacts — always rendered so the section's existence and
           location are predictable; falls back to a contextual empty/
           loading message instead of disappearing while the job is
           still producing files. -->
      <div class="card" style="margin-top: var(--space-4)" data-testid="artifacts-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--space-3); flex-wrap: wrap; gap: var(--space-2)">
          <div class="card-title" style="margin: 0">Result Files</div>
          ${files.length > 0 && html`
            <div style="display: flex; gap: var(--space-2); flex-wrap: wrap">
              ${hasExportFile && html`
                <button
                  onclick=${exportJson}
                  style=${'background: ' + palette.teal + '22; color: ' + palette.teal + '; border: 1px solid ' + palette.teal + '44; padding: var(--space-1) var(--space-3); border-radius: var(--radius-md); cursor: pointer; font-size: var(--font-size-sm)'}
                >
                  Export JSON
                </button>
              `}
              <a
                class="btn"
                href=${api.resultBundleUrl(namespace, name, epoch)}
                download
                data-testid="artifacts-bundle"
                style=${'background: ' + palette.green + '22; color: ' + palette.green + '; border: 1px solid ' + palette.green + '44; padding: var(--space-1) var(--space-3); border-radius: var(--radius-md); cursor: pointer; font-size: var(--font-size-sm); text-decoration: none'}
                title=${'Download all ' + files.length + ' file' + (files.length === 1 ? '' : 's') + ' as a single .zip'}
              >
                Download .zip${totalArtifactBytes > 0 ? ` (${fmtBytes(totalArtifactBytes)})` : ''}
              </a>
              <button
                onclick=${downloadAll}
                style=${'background: ' + palette.blue + '22; color: ' + palette.blue + '; border: 1px solid ' + palette.blue + '44; padding: var(--space-1) var(--space-3); border-radius: var(--radius-md); cursor: pointer; font-size: var(--font-size-sm)'}
                title="Trigger one download per file (browser saves them individually)"
              >
                Download All
              </button>
            </div>
          `}
        </div>

        ${!filesLoaded && html`
          <${LoadingPanel} label="Looking up result files…" inline=${true} testid="artifacts-loading" />
        `}

        ${filesLoaded && files.length === 0 && html`
          <div data-testid="artifacts-empty" style=${'padding: var(--space-5) var(--space-4); border-radius: var(--radius-lg); border: 1px dashed ' + palette.surface0 + '; color: ' + palette.subtext0 + '; font-size: var(--font-size-sm); display: flex; align-items: center; gap: var(--space-3)'}>
            <span style=${'flex-shrink: 0; color: ' + palette.overlay0}>
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M14 3 H7 a2 2 0 0 0 -2 2 v14 a2 2 0 0 0 2 2 h10 a2 2 0 0 0 2 -2 V8 z" />
                <polyline points="14,3 14,8 19,8" />
                <line x1="9" y1="13" x2="15" y2="13" />
                <line x1="9" y1="17" x2="13" y2="17" />
              </svg>
            </span>
            <div style="display: flex; flex-direction: column; gap: 2px">
              <div style=${'font-weight: 600; color: ' + palette.text}>
                ${resolvedEpoch == null
                  ? 'Waiting for a run epoch before showing result files.'
                  : isCompleted
                    ? 'No result files persisted for this run.'
                    : isRunning
                      ? 'No result files yet.'
                      : 'No result files available.'}
              </div>
              <div class="text-dim" style="font-size: var(--font-size-xs)">
                ${resolvedEpoch == null
                  ? 'This page now requires a pinned run epoch before it will fetch final artifacts, so the status and results cannot drift to different runs.'
                  : isCompleted
                    ? 'The job completed but no artifacts were uploaded — check the operator logs or the controller pod for this run.'
                    : isRunning
                      ? 'Files (profile_export_aiperf.json, profile_export.jsonl, server_metrics_export.json, ...) appear here once the run finishes and uploads them to the results PVC.'
                      : 'Artifacts will appear here after the run starts producing output.'}
              </div>
            </div>
          </div>
        `}

        ${filesLoaded && files.length > 0 && html`
          <div style="display: flex; flex-direction: column; gap: var(--space-1)">
            ${files.map(f => {
              const ext = f.name.split('.').pop().toLowerCase();
              const previewable = PREVIEWABLE.has(ext);
              const chip = fileTypeChip(f.name);
              const action = () => openFile(f.name);
              return html`
                <div
                  key=${f.name}
                  onclick=${action}
                  onkeydown=${e => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      action();
                    }
                  }}
                  role="button"
                  tabindex="0"
                  aria-label=${(previewable ? 'Preview ' : 'Download ') + f.name}
                  title=${previewable ? 'Click to preview' : 'Click to download'}
                  style=${'display: flex; justify-content: space-between; align-items: center; padding: var(--space-2) var(--space-3); background: ' + palette.base + '; border-radius: var(--radius-sm); cursor: pointer; transition: background 0.15s; border: 1px solid ' + palette.surface0 + '60; outline: none'}
                  onmouseenter=${e => { e.currentTarget.style.background = palette.surface0; }}
                  onmouseleave=${e => { e.currentTarget.style.background = palette.base; }}
                  onfocus=${e => { e.currentTarget.style.background = palette.surface0; e.currentTarget.style.borderColor = palette.blue + '88'; }}
                  onblur=${e => { e.currentTarget.style.background = palette.base; e.currentTarget.style.borderColor = palette.surface0 + '60'; }}
                >
                  <div style="display: flex; align-items: center; gap: var(--space-2); min-width: 0">
                    <span
                      class="file-type-chip"
                      style=${'background: ' + chip.color + '22; color: ' + chip.color + '; border: 1px solid ' + chip.color + '55'}
                      title=${'File type: ' + chip.label.toLowerCase()}
                    >${chip.label}</span>
                    <span style=${'font-size: var(--font-size-sm); color: ' + fileColor(f.name) + '; overflow: hidden; text-overflow: ellipsis; white-space: nowrap'}>${f.name}</span>
                  </div>
                  <div style="display: flex; align-items: center; gap: var(--space-2); flex-shrink: 0">
                    <span style=${'font-size: var(--font-size-xs); color: ' + palette.overlay0 + '; font-style: italic'}>${previewable ? 'preview' : 'download'}</span>
                    <span class="text-dim" style="font-size: var(--font-size-xs)">${humanSize(f.size_bytes)}</span>
                  </div>
                </div>
              `;
            })}
          </div>
        `}
      </div>
        </main>
        <aside class="job-detail__rail" data-testid="job-detail-rail" aria-label="Run context">
          ${railContent}
        </aside>
      </div>
      <${DiagnosticsDrawer}
        open=${diagnosticsOpen}
        onClose=${() => setDiagnosticsOpen(false)}
        ns=${namespace}
        name=${name}
        conditions=${conditions}
        pods=${pods}
        mode=${stripMode}
        archived=${!viewingCurrentRun}
        eventCount=${null}
        logSeverityCounts=${null}
        conditionWarnCount=${(conditions || []).filter(c => c.status !== 'True').length}
        podCrashCount=${(pods || []).filter(p => /crashloop/i.test(p.reason || '')).length} />
    </div>
    ${fileViewer && html`
      <${FileViewerModal}
        filename=${fileViewer.filename}
        url=${fileViewer.url}
        onClose=${() => setFileViewer(null)}
      />
    `}
    <style>
      @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.75); }
      }
    </style>
  `;
}
