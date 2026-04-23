import { html } from 'htm/preact';
import { useState, useEffect, useRef } from 'preact/hooks';
import { api, poll } from '../lib/api.js';
import { phaseColor, colors, palette } from '../lib/theme.js';
import { navigate } from '../lib/router.js';
import { KpiCard } from '../components/kpi-card.js';
import { ChartWrapper } from '../components/chart-wrapper.js';
import { Conditions } from '../components/conditions.js';
import { PodsBar } from '../components/pods-bar.js';
import { HeroStrip } from '../components/hero-strip.js';
import { PhaseCards } from '../components/phase-cards.js';
import { fmtNumber, fmtInt, fmtThroughput, fmtBytes } from '../lib/format.js';

const MAX_CHART_POINTS = 60;

function formatElapsed(startTime) {
  if (!startTime) return null;
  const ms = Date.now() - new Date(startTime).getTime();
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  const h = Math.floor(m / 60);
  if (h > 0) return `${h}h ${m % 60}m`;
  if (m > 0) return `${m}m ${s % 60}s`;
  return `${s}s`;
}

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

function fmtMs(val) {
  if (val == null) return null;
  return fmtNumber(val, 0);
}

function fmtRps(val) {
  if (val == null) return null;
  return fmtThroughput(val);
}

function fmtNum(val, decimals = 1) {
  if (val == null) return '---';
  return fmtNumber(val, decimals);
}

// Metrics table: group definitions
const METRIC_GROUPS = [
  {
    label: 'Throughput',
    color: palette.blue,
    rows: [
      { key: 'request_throughput', label: 'Request Throughput', cols: ['avg'], colLabels: ['avg'] },
      { key: 'output_token_throughput', label: 'Output Token Throughput', cols: ['avg'], colLabels: ['avg'] },
      { key: 'total_token_throughput', label: 'Total Token Throughput', cols: ['avg'], colLabels: ['avg'] },
    ],
  },
  {
    label: 'Latency',
    color: palette.peach,
    rows: [
      { key: 'request_latency', label: 'Request Latency', cols: ['avg', 'p50', 'p90', 'p95', 'p99', 'min', 'max'], colLabels: ['avg', 'p50', 'p90', 'p95', 'p99', 'min', 'max'] },
      { key: 'time_to_first_token', label: 'Time to First Token', cols: ['avg', 'p50', 'p90', 'p95', 'p99'], colLabels: ['avg', 'p50', 'p90', 'p95', 'p99'] },
      { key: 'inter_token_latency', label: 'Inter-Token Latency', cols: ['avg', 'p50', 'p90', 'p95', 'p99'], colLabels: ['avg', 'p50', 'p90', 'p95', 'p99'] },
      { key: 'time_to_second_token', label: 'Time to Second Token', cols: ['avg', 'p50', 'p95', 'p99'], colLabels: ['avg', 'p50', 'p95', 'p99'] },
    ],
  },
  {
    label: 'Sequence Lengths',
    color: palette.teal,
    rows: [
      { key: 'input_sequence_length', label: 'Input Sequence Length', cols: ['avg', 'p50', 'p99'], colLabels: ['avg', 'p50', 'p99'] },
      { key: 'output_sequence_length', label: 'Output Sequence Length', cols: ['avg', 'p50', 'p99'], colLabels: ['avg', 'p50', 'p99'] },
    ],
  },
  {
    label: 'HTTP',
    color: palette.mauve,
    rows: [
      { key: 'request_duration', label: 'Request Duration', cols: ['avg', 'p50', 'p99'], colLabels: ['avg', 'p50', 'p99'] },
      { key: 'connection_overhead', label: 'Connection Overhead', cols: ['avg'], colLabels: ['avg'] },
      { key: 'dns_lookup', label: 'DNS Lookup', cols: ['avg'], colLabels: ['avg'] },
    ],
  },
];

function MetricsTable({ results }) {
  const [collapsed, setCollapsed] = useState({});

  function toggleGroup(label) {
    setCollapsed(prev => ({ ...prev, [label]: !prev[label] }));
  }

  return html`
    <div class="card" style="margin-top: var(--space-4)">
      <div class="card-title">Full Metrics Breakdown</div>
      ${METRIC_GROUPS.map(group => {
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
                      ${['avg', 'p50', 'p90', 'p95', 'p99', 'min', 'max'].map(col => html`
                        <th key=${col} style=${'text-align: right; padding: var(--space-2) var(--space-3); color: ' + palette.overlay1 + '; font-weight: 500; font-size: var(--font-size-xs); border-bottom: 1px solid ' + palette.surface0}>${col}</th>
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
                          ${['avg', 'p50', 'p90', 'p95', 'p99', 'min', 'max'].map(col => {
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
function SLACompliance({ results, summary }) {
  const ttftP99 = results?.time_to_first_token?.p99 ?? null;
  const itlP99 = results?.inter_token_latency?.p99 ?? null;
  const errorRate = summary?.error_rate ?? null;

  // Only show if we have at least one metric
  if (ttftP99 == null && itlP99 == null && errorRate == null) return null;

  const checks = [];

  if (ttftP99 != null) {
    checks.push({
      label: 'TTFT p99 < 500ms',
      pass: ttftP99 < 500,
      value: `${fmtNumber(ttftP99, 0)} ms`,
    });
  }

  if (itlP99 != null) {
    checks.push({
      label: 'ITL p99 < 100ms',
      pass: itlP99 < 100,
      value: `${fmtNumber(itlP99, 0)} ms`,
    });
  }

  if (errorRate != null) {
    checks.push({
      label: 'Error rate < 1%',
      pass: errorRate < 1,
      value: `${fmtNumber(errorRate, 2)}%`,
    });
  }

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
function JobConfigSection({ config }) {
  const [expanded, setExpanded] = useState(false);

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
      <div
        onclick=${() => setExpanded(e => !e)}
        style=${'display: flex; align-items: center; justify-content: space-between; cursor: pointer; user-select: none'}
      >
        <div class="card-title" style="margin: 0">Job Configuration</div>
        <span class="text-dim" style="font-size: var(--font-size-xs)">${expanded ? '\u25B2 Collapse' : '\u25BC Expand'}</span>
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

      ${expanded && html`
        <div style="margin-top: var(--space-4)">
          <div style=${'font-size: var(--font-size-xs); font-weight: 600; color: ' + palette.overlay1 + '; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: var(--space-2)'}>Full Spec (${config.source ?? 'unknown'})</div>
          <pre style=${'margin: 0; font-family: monospace; font-size: var(--font-size-xs); line-height: 1.6; white-space: pre; color: ' + palette.text + '; background: ' + palette.base + '; padding: var(--space-3); border-radius: var(--radius-sm); max-height: 400px; overflow: auto'}>
            ${JSON.stringify(spec, null, 2)}
          </pre>
        </div>
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

const MODAL_STYLE = [
  'background: ' + palette.mantle + ';',
  'border: 1px solid ' + palette.surface0 + ';',
  'border-radius: var(--radius-md);',
  'max-width: 80vw; max-height: 80vh;',
  'width: 900px;',
  'display: flex; flex-direction: column;',
  'overflow: hidden;',
].join(' ');

function ModalChrome({ filename, onCopy, onDownload, onClose, copyLabel, children }) {
  return html`
    <div style=${BACKDROP_STYLE} onclick=${e => { if (e.target === e.currentTarget) onClose(); }}>
      <div style=${MODAL_STYLE}>
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
    body = html`<span class="text-dim">Loading...</span>`;
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
    >
      ${body}
    </${ModalChrome}>
  `;
}

// --- Server Metrics Section ---

function getSeriesValue(metric) {
  // Server metrics store stats under series[0].stats.avg (or series[0].value for simple counters)
  const series = metric?.series;
  if (!Array.isArray(series) || series.length === 0) return null;
  const s = series[0];
  return s.stats?.avg ?? s.value ?? s.avg ?? null;
}

function buildHistogramChartData(metric, color) {
  const series = metric?.series;
  if (!Array.isArray(series) || series.length === 0) return null;
  const raw = series[0].buckets;
  if (!raw) return null;

  // Buckets can be a dict {"0.001": count, ...} or array [{le, count}]
  let labels, counts;
  if (Array.isArray(raw)) {
    if (raw.length === 0) return null;
    labels = raw.map(b => {
      // JSON cannot carry Infinity; backends typically send +Inf as null
      // or a large sentinel. Normalize them all.
      if (b.le == null || b.le === Infinity || b.le === '+Inf' || b.le === 'Inf') return '+Inf';
      if (typeof b.le === 'number' && b.le >= 1e10) return '+Inf';
      return String(b.le);
    });
    counts = raw.map(b => b.count ?? 0);
  } else if (typeof raw === 'object') {
    const entries = Object.entries(raw);
    if (entries.length === 0) return null;
    labels = entries.map(([k]) => k);
    counts = entries.map(([, v]) => v);
  } else {
    return null;
  }

  // Convert cumulative histogram to deltas (Prometheus histograms are cumulative)
  const isCumulative = counts.length > 1 && counts.every((v, i) => i === 0 || v >= counts[i - 1]);
  if (isCumulative) {
    for (let i = counts.length - 1; i > 0; i--) {
      counts[i] = counts[i] - counts[i - 1];
    }
  }

  // Filter out zero-count buckets at the tails for cleaner charts
  let start = 0;
  while (start < counts.length && counts[start] === 0) start++;
  let end = counts.length - 1;
  while (end > start && counts[end] === 0) end--;
  labels = labels.slice(start, end + 1);
  counts = counts.slice(start, end + 1);
  return {
    labels,
    datasets: [{
      label: 'Count',
      data: counts,
      backgroundColor: color + '88',
      borderColor: color,
      borderWidth: 1,
      borderRadius: 2,
    }],
  };
}

function ServerMetricsSection({ serverMetrics }) {
  if (!serverMetrics) return null;
  const metrics = serverMetrics.metrics ?? {};

  const kvCacheRaw = getSeriesValue(metrics['vllm:kv_cache_usage_perc']);
  const reqRunning = getSeriesValue(metrics['vllm:num_requests_running']);
  const reqWaiting = getSeriesValue(metrics['vllm:num_requests_waiting']);
  const preemptions = getSeriesValue(metrics['vllm:num_preemptions']);

  const prefixHits = getSeriesValue(metrics['vllm:prefix_cache_hits']);
  const prefixQueries = getSeriesValue(metrics['vllm:prefix_cache_queries']);
  const prefixHitRate = prefixHits != null && prefixQueries != null && prefixQueries > 0
    ? prefixHits / prefixQueries
    : null;

  const kpiCards = [];
  if (kvCacheRaw != null) {
    kpiCards.push(html`
      <${KpiCard}
        key="kv"
        label="KV Cache Usage"
        value=${fmtNumber(kvCacheRaw * 100, 1)}
        unit="%"
        color=${palette.peach}
      />
    `);
  }
  if (reqRunning != null) {
    kpiCards.push(html`
      <${KpiCard}
        key="running"
        label="Requests Running"
        value=${fmtInt(reqRunning)}
        color=${palette.blue}
      />
    `);
  }
  if (reqWaiting != null) {
    kpiCards.push(html`
      <${KpiCard}
        key="waiting"
        label="Requests Waiting"
        value=${fmtInt(reqWaiting)}
        color=${palette.yellow}
      />
    `);
  }
  if (preemptions != null) {
    kpiCards.push(html`
      <${KpiCard}
        key="preempt"
        label="Preemptions"
        value=${fmtInt(preemptions)}
        color=${palette.maroon}
      />
    `);
  }
  if (prefixHitRate != null) {
    kpiCards.push(html`
      <${KpiCard}
        key="prefix"
        label="Prefix Cache Hit Rate"
        value=${fmtNumber(prefixHitRate * 100, 1)}
        unit="%"
        color=${palette.teal}
      />
    `);
  }

  const histogramDefs = [
    { key: 'vllm:time_to_first_token_seconds', label: 'Server-Side TTFT', color: palette.teal, unit: 's' },
    { key: 'vllm:e2e_request_latency_seconds', label: 'Server-Side E2E Latency', color: palette.mauve, unit: 's' },
    { key: 'vllm:request_queue_time_seconds', label: 'Queue Time', color: palette.sapphire, unit: 's' },
  ];

  const histogramChartOptions = (unit) => ({
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: ctx => ` ${ctx.parsed.y} requests` } },
    },
    scales: {
      x: {
        ticks: { color: palette.overlay0, font: { size: 10 }, maxRotation: 45 },
        grid: { color: palette.surface0 },
        title: { display: true, text: `Bucket (${unit})`, color: palette.overlay1, font: { size: 10 } },
      },
      y: {
        ticks: { color: palette.overlay0, font: { size: 10 } },
        grid: { color: palette.surface0 },
        title: { display: true, text: 'Count', color: palette.overlay1, font: { size: 10 } },
      },
    },
  });

  const histogramCharts = histogramDefs
    .map(def => {
      const data = buildHistogramChartData(metrics[def.key], def.color);
      if (!data) return null;
      return html`
        <div key=${def.key} class="card" style="margin-top: var(--space-4)">
          <div class="card-title">${def.label}</div>
          <${ChartWrapper} type="bar" data=${data} options=${histogramChartOptions(def.unit)} height=${200} />
        </div>
      `;
    })
    .filter(Boolean);

  if (kpiCards.length === 0 && histogramCharts.length === 0) return null;

  return html`
    <div style="margin-top: var(--space-4)">
      <div class="card">
        <div class="card-title">Server Metrics</div>
        ${kpiCards.length > 0 && html`
          <div class="kpi-row" style="margin-top: var(--space-3)">
            ${kpiCards}
          </div>
        `}
      </div>
      ${histogramCharts}
    </div>
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
    const phase = rec?.metadata?.phase ?? rec?.metadata?.credit_phase ?? null;
    return { index: i + 1, isl, osl, ttft, latency, itl, phase };
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
  const COL_DEFS = [
    { key: '#', label: '#', get: r => r.index, fmt: v => fmtInt(v) },
    { key: 'isl', label: 'ISL', get: r => r.isl, fmt: v => fmtInt(v) },
    { key: 'osl', label: 'OSL', get: r => r.osl, fmt: v => fmtInt(v) },
    { key: 'ttft', label: 'TTFT (ms)', get: r => r.ttft, fmt: v => fmtNumber(v, 1) },
    { key: 'latency', label: 'Latency (ms)', get: r => r.latency, fmt: v => fmtNumber(v, 1) },
    ...(hasItl ? [{ key: 'itl', label: 'ITL (ms)', get: r => r.itl, fmt: v => fmtNumber(v, 1) }] : []),
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
    return sortAsc ? av - bv : bv - av;
  });
  const displayRows = tableExpanded ? sorted : sorted.slice(0, 50);

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
      <div style="font-size: var(--font-size-xs); color: ${palette.overlay0}; margin-bottom: var(--space-3)">${records.length} requests</div>

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
                ${displayRows.map((row, ri) => html`
                  <tr key=${row.index} style=${'background: ' + (ri % 2 === 0 ? palette.base : palette.mantle)}>
                    ${COL_DEFS.map((col, ci) => html`
                      <td key=${col.key} style=${'padding: var(--space-1) var(--space-3); color: ' + palette.text + '; text-align: ' + (ci === 0 ? 'left' : 'right') + '; border-bottom: 1px solid ' + palette.surface0 + '40'}>
                        ${col.fmt(col.get(row))}
                      </td>
                    `)}
                  </tr>
                `)}
              </tbody>
            </table>
          </div>
        `}
      </div>
    </div>
  `;
}

export function JobDetail({ namespace, name }) {
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const [files, setFiles] = useState([]);
  const [polling, setPolling] = useState(true);
  const [serverMetrics, setServerMetrics] = useState(null);
  const [fileViewer, setFileViewer] = useState(null); // { filename, url }
  const [jsonlRecords, setJsonlRecords] = useState(null);
  const [jobConfig, setJobConfig] = useState(null);

  const PREVIEWABLE = new Set(['json', 'csv', 'txt', 'ansi']);

  // Close file viewer on Escape
  useEffect(() => {
    function onKeyDown(e) {
      if (e.key === 'Escape') setFileViewer(null);
    }
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, []);

  // Rolling throughput chart data - kept in a ref so we don't trigger re-renders for
  // each append; we rebuild the data object for ChartWrapper on each render.
  const throughputPoints = useRef({ labels: [], values: [] });
  const [chartData, setChartData] = useState(null);

  // Per-KPI rolling window of {t, v} samples for inline sparklines. Keyed by
  // the same short tag (``throughput``, ``token_throughput``, ``ttft``, ...)
  // we pass into each KpiCard's ``points`` prop below. Capped at 60 samples
  // per key — matches the throughput chart buffer.
  const [kpiSeries, setKpiSeries] = useState({});

  useEffect(() => {
    const ac = new AbortController();
    // Reset chart points when job changes
    throughputPoints.current = { labels: [], values: [] };
    setChartData(null);
    setKpiSeries({});
    setPolling(true);

    poll(
      async () => {
        const data = await api.getJob(namespace, name);
        setJob(data);
        setError(null);

        const phase = (data?.job?.phase ?? data?.status?.phase ?? '').toLowerCase();
        const done = phase === 'completed' || phase === 'succeeded' || phase === 'failed' || phase === 'error';
        if (done) setPolling(false);

        // Append to throughput chart
        const summary = extractSummary(data);
        const tps =
          summary?.throughput_rps ??
          data?.status?.metrics?.request_throughput?.avg ??
          null;

        // Accumulate rolling samples per KPI tile (60 max). Keys mirror the
        // short tag we hand to each ``<KpiCard points=...>`` below — mutate
        // locally, then setState once so React re-renders all tiles in sync.
        const liveOrStaticSummary =
          data?.status?.liveSummary ?? data?.status?.summary ?? summary ?? {};
        const kpiSample = {
          throughput: tps,
          token_throughput: liveOrStaticSummary?.output_token_throughput_tps ?? null,
          ttft:
            liveOrStaticSummary?.ttft_p99_ms ??
            liveOrStaticSummary?.ttft_avg_ms ??
            null,
          latency: liveOrStaticSummary?.latency_p99_ms ?? null,
          errors: liveOrStaticSummary?.error_rate ?? null,
        };
        const now = Date.now();
        setKpiSeries((prev) => {
          const next = { ...prev };
          for (const [k, v] of Object.entries(kpiSample)) {
            if (v == null || !isFinite(v)) continue;
            const buf = next[k] ? [...next[k]] : [];
            buf.push({ t: now, v });
            if (buf.length > MAX_CHART_POINTS) buf.shift();
            next[k] = buf;
          }
          return next;
        });

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
                label: 'Throughput (req/s)',
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

    // Fetch available result files directly
    fetch(`/api/v1/results/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}`, { signal: ac.signal })
      .then(r => r.ok ? r.json() : null)
      .then(d => {
        if (!d) return;
        const fileList = d?.files ?? [];
        setFiles(fileList);
        if (fileList.some(f => f.name === 'server_metrics_export.json')) {
          fetch(`/api/v1/results/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}/server_metrics_export.json`, { signal: ac.signal })
            .then(r => r.ok ? r.json() : null)
            .then(sm => { if (sm) setServerMetrics(sm); })
            .catch(() => {});
        }
        if (fileList.some(f => f.name === 'profile_export.jsonl')) {
          fetch(`/api/v1/results/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}/profile_export.jsonl`, { signal: ac.signal })
            .then(r => r.ok ? r.text() : null)
            .then(text => {
              if (!text) return;
              const recs = text
                .split('\n')
                .filter(line => line.trim() !== '')
                .map(line => { try { return JSON.parse(line); } catch { return null; } })
                .filter(Boolean);
              if (recs.length > 0) setJsonlRecords(recs);
            })
            .catch(() => {});
        }
      })
      .catch(() => {});

    return () => ac.abort();
  }, [namespace, name]);

  function humanSize(bytes) {
    return fmtBytes(bytes);
  }

  function downloadFile(fileName) {
    const url = `/api/v1/results/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}/${encodeURIComponent(fileName)}`;
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    a.click();
  }

  function openFile(fileName) {
    const url = `/api/v1/results/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}/${encodeURIComponent(fileName)}`;
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
    if (!confirm(`Cancel job ${name}?`)) return;
    try {
      await api.cancelJob(namespace, name);
    } catch (e) {
      alert('Cancel failed: ' + e.message);
    }
  }

  if (!job && !error) {
    return html`
      <div class="card" style="text-align: center; padding: var(--space-8)">
        <span class="text-dim">Loading ${namespace}/${name}...</span>
      </div>
    `;
  }

  if (error) {
    return html`
      <div class="card" style="border-color: ${colors.error}44; color: ${colors.error}">
        <strong>Error:</strong> ${error}
      </div>
    `;
  }

  // job detail response: { job: {AIPerfJobInfo}, status: {raw CR status}, pods: [...] }
  // job.job has flat camelCase fields, job.status has raw CR status
  const info = job?.job ?? {};
  const status = job?.status ?? {};

  const phase = info.phase ?? status.phase ?? 'Unknown';
  const phaseClr = phaseColor(phase);
  const model = info.model ?? '---';
  const endpointUrl = info.endpoint ?? null;
  const startTime = info.startTime ?? status.startTime;
  const elapsed = formatElapsed(startTime);
  const isRunning = phase.toLowerCase() === 'running';
  const isCompleted = phase.toLowerCase() === 'completed' || phase.toLowerCase() === 'succeeded';

  // Extract metrics from status.summary (completed) or status.liveSummary (running)
  const summary = status.liveSummary ?? status.summary ?? {};
  const throughput = summary.throughput_rps ?? info.throughputRps ?? null;
  const ttftAvg = summary.ttft_avg_ms ?? null;
  const ttftP99 = summary.ttft_p99_ms ?? null;
  const latP99 = summary.latency_p99_ms ?? info.latencyP99Ms ?? null;
  const latAvg = summary.latency_avg_ms ?? null;
  const errorRate = summary.error_rate ?? null;
  const errorCount = errorRate != null ? fmtNumber(errorRate, 1) + '%' : null;

  // Metrics from results (nested under .metrics in the CR status)
  const rawResults = status.results ?? null;
  const results = rawResults?.metrics ?? rawResults ?? null;
  const outputTokenThroughputAvg = results?.output_token_throughput?.avg ?? summary.output_token_throughput_tps ?? null;
  // Current throughput: prefer live current-window value when available;
  // fall back to avg so completed/archived jobs still populate the headline.
  const outputTokenThroughputCurrent = summary.output_token_throughput_tps ?? outputTokenThroughputAvg;

  // SLO source of truth: config endpoint may wrap benchmark or unwrap it
  // (summary-fallback path). Check both shapes; fall back to {} when unset.
  const slos = jobConfig?.spec?.benchmark?.slos ?? jobConfig?.spec?.slos ?? {};
  const sloTtft = slos.time_to_first_token;
  const sloLatency = slos.request_latency;
  const sloThroughput = slos.request_throughput;

  const conditions = status.conditions ?? [];
  const rawPhases = status.phases ?? {};
  const pods = job?.pods ?? [];
  const jobError = info.error ?? status.error ?? null;

  // Build latency histogram from completed results if available
  const latencyHistogram = (() => {
    const buckets = job?.status?.results?.latency_histogram ?? job?.status?.results?.histograms?.request_latency ?? null;
    if (!buckets || !Array.isArray(buckets) || buckets.length === 0) return null;
    return {
      labels: buckets.map((b) => (typeof b.le === 'number' ? (b.le * 1000).toFixed(0) + 'ms' : String(b.le))),
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

  const throughputChartOptions = {
    plugins: { legend: { display: false } },
    scales: {
      x: {
        ticks: { color: palette.overlay0, maxTicksLimit: 6, font: { size: 10 } },
        grid: { color: palette.surface0 },
      },
      y: {
        ticks: { color: palette.overlay0, font: { size: 10 } },
        grid: { color: palette.surface0 },
        title: { display: true, text: 'req/s', color: palette.overlay1, font: { size: 10 } },
      },
    },
  };

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

  function fileColor(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    if (ext === 'json' || ext === 'jsonl') return palette.mauve;
    if (ext === 'csv') return palette.teal;
    if (ext === 'txt' || ext === 'ansi') return palette.blue;
    return palette.overlay1;
  }

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
              ${elapsed && html`<span class="text-dim" style="font-size: var(--font-size-sm)">${elapsed}</span>`}
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
                  : null
              }
            </div>
            <div class="text-dim" style="font-size: var(--font-size-sm); margin-top: var(--space-1)">
              ${namespace} · ${model}
              ${endpointUrl && html` · <span style="color: ${palette.blue}">${endpointUrl}</span>`}
            </div>
          </div>
          ${isRunning && info?.source !== 'archived' && html`
            <button
              class="btn btn-danger"
              onclick=${handleCancel}
              style=${'background: ' + colors.error + '22; color: ' + colors.error + '; border: 1px solid ' + colors.error + '44; padding: var(--space-2) var(--space-4); border-radius: var(--radius-md); cursor: pointer; font-size: var(--font-size-sm)'}
              data-testid="job-detail-cancel"
            >
              Cancel
            </button>
          `}
        </div>
      </div>

      ${info?.source === 'archived' && html`
        <div class="banner banner-info" style="margin-bottom: var(--space-3); padding: var(--space-3); border-radius: var(--radius-sm); background: var(--color-overlay-0, rgba(255,255,255,0.04)); color: var(--color-text-dim)">
          This run's Kubernetes resource has been deleted. Showing archived
          results from the results volume.
        </div>
      `}

      <!-- Conditions -->
      ${conditions.length > 0 && html`
        <div style="margin-bottom: var(--space-4)">
          <${Conditions} conditions=${conditions} />
        </div>
      `}

      <!-- Error banner -->
      ${jobError && html`
        <div class="card" style="border-color: ${colors.error}44; color: ${colors.error}; margin-bottom: var(--space-4)">
          <strong>Error:</strong> ${jobError}
        </div>
      `}

      <!-- Hero strip: SLO-aware health + elapsed/ETA + active-phase progress -->
      <div style="margin-bottom: var(--space-4)">
        <${HeroStrip} info=${info} status=${status} config=${jobConfig} />
      </div>

      <!-- KPI row -->
      <div class="kpi-row" style="margin-bottom: var(--space-6)">
        <${KpiCard}
          label="Throughput"
          value=${fmtRps(throughput) ?? '---'}
          unit=${throughput != null ? 'req/s' : ''}
          color=${colors.phaseRunning}
          sub=${throughput != null ? `avg ${fmtRps(throughput)} req/s` : null}
          rawValue=${throughput}
          slo=${sloThroughput != null ? { threshold: sloThroughput, compare: 'gt' } : null}
          points=${kpiSeries.throughput}
        />
        <${KpiCard}
          label="Token Throughput"
          value=${outputTokenThroughputCurrent != null ? fmtNum(outputTokenThroughputCurrent, 0) : '---'}
          unit=${outputTokenThroughputCurrent != null ? 'tok/s' : ''}
          color=${palette.sapphire}
          sub=${outputTokenThroughputAvg != null ? `avg ${fmtNum(outputTokenThroughputAvg, 0)} tok/s` : null}
          points=${kpiSeries.token_throughput}
        />
        <${KpiCard}
          label="TTFT"
          value=${ttftP99 != null ? fmtMs(ttftP99) : (ttftAvg != null ? fmtMs(ttftAvg) : '---')}
          unit=${(ttftP99 ?? ttftAvg) != null ? 'ms' : ''}
          color=${palette.teal}
          sub=${ttftAvg != null ? `avg ${fmtNum(ttftAvg, 0)} ms` : null}
          rawValue=${ttftP99 ?? ttftAvg}
          slo=${sloTtft != null ? { threshold: sloTtft, compare: 'lt' } : null}
          points=${kpiSeries.ttft}
        />
        <${KpiCard}
          label="Latency P99"
          value=${latP99 != null ? fmtNumber(latP99, 0) : '---'}
          unit=${latP99 != null ? 'ms' : ''}
          color=${palette.peach}
          sub=${latAvg != null ? `avg ${fmtNum(latAvg, 0)} ms` : null}
          rawValue=${latP99}
          slo=${sloLatency != null ? { threshold: sloLatency, compare: 'lt' } : null}
          points=${kpiSeries.latency}
        />
        <${KpiCard}
          label="Errors"
          value=${errorCount ?? '---'}
          color=${errorCount ? colors.error : colors.textMuted}
          points=${kpiSeries.errors}
        />
        <!-- Feature 5: Token Efficiency -->
        ${isCompleted && results && html`<${TokenEfficiencyCard} results=${results} info=${info} />`}
      </div>

      <!-- Two-column split -->
      <div class="detail-split">
        <!-- Left: Phase progress + pods -->
        <div>
          ${Object.keys(rawPhases).length > 0 && html`
            <div style="margin-bottom: var(--space-4)">
              <${PhaseCards} phases=${rawPhases} />
            </div>
          `}

          ${pods.length > 0 && info?.source !== 'archived' && html`
            <div class="card" data-testid="job-detail-pods">
              <div class="card-title">Pods</div>
              <${PodsBar} pods=${pods} />
            </div>
          `}
        </div>

        <!-- Right: Charts -->
        <div>
          <div class="card" style="margin-bottom: var(--space-4)">
            <div class="card-title">Live Throughput</div>
            ${chartData
              ? html`<${ChartWrapper} type="line" data=${chartData} options=${throughputChartOptions} height=${200} />`
              : html`<div class="text-dim" style="padding: var(--space-4); text-align: center">Waiting for data...</div>`
            }
          </div>

          ${isCompleted && latencyHistogram && html`
            <div class="card">
              <div class="card-title">Latency Distribution</div>
              <${ChartWrapper} type="bar" data=${latencyHistogram} options=${histogramOptions} height=${200} />
            </div>
          `}
        </div>
      </div>

      <!-- Feature 6: SLA Compliance (completed only) -->
      ${isCompleted && html`<${SLACompliance} results=${results} summary=${summary} />`}

      <!-- Server Metrics (completed only, when available) -->
      ${isCompleted && serverMetrics && html`<${ServerMetricsSection} serverMetrics=${serverMetrics} />`}

      <!-- Job Configuration (always shown if available) -->
      ${jobConfig && html`<${JobConfigSection} config=${jobConfig} />`}

      <!-- Feature 8: Run Metadata (completed only) -->
      ${isCompleted && html`<${RunMetadata} status=${status} results=${results} info=${info} />`}

      <!-- Per-Record Analysis from profile_export.jsonl -->
      ${isCompleted && jsonlRecords && html`<${PerRecordAnalysis} records=${jsonlRecords} />`}

      <!-- Feature 3: Concurrency vs Throughput (completed only) -->
      ${isCompleted && html`<${ConcurrencyThroughputChart} status=${status} />`}

      <!-- Latency percentile chart (completed only) -->
      ${isCompleted && results && html`<${LatencyPercentileChart} results=${results} />`}

      <!-- Feature 4: ISL Distribution (completed only) -->
      ${isCompleted && results && html`<${ISLDistributionChart} results=${results} />`}

      <!-- Full metrics breakdown (completed only) -->
      ${isCompleted && results && html`<${MetricsTable} results=${results} />`}

      <!-- Artifacts -->
      ${files.length > 0 && html`
        <div class="card" style="margin-top: var(--space-4)">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--space-3); flex-wrap: wrap; gap: var(--space-2)">
            <div class="card-title" style="margin: 0">Artifacts</div>
            <div style="display: flex; gap: var(--space-2)">
              ${hasExportFile && html`
                <button
                  onclick=${exportJson}
                  style=${'background: ' + palette.teal + '22; color: ' + palette.teal + '; border: 1px solid ' + palette.teal + '44; padding: var(--space-1) var(--space-3); border-radius: var(--radius-md); cursor: pointer; font-size: var(--font-size-sm)'}
                >
                  Export JSON
                </button>
              `}
              <button
                onclick=${downloadAll}
                style=${'background: ' + palette.blue + '22; color: ' + palette.blue + '; border: 1px solid ' + palette.blue + '44; padding: var(--space-1) var(--space-3); border-radius: var(--radius-md); cursor: pointer; font-size: var(--font-size-sm)'}
              >
                Download All
              </button>
            </div>
          </div>
          <div style="display: flex; flex-direction: column; gap: var(--space-1)">
            ${files.map(f => {
              const ext = f.name.split('.').pop().toLowerCase();
              const previewable = PREVIEWABLE.has(ext);
              return html`
                <div
                  key=${f.name}
                  onclick=${() => openFile(f.name)}
                  style=${'display: flex; justify-content: space-between; align-items: center; padding: var(--space-2) var(--space-3); background: ' + palette.base + '; border-radius: var(--radius-sm); cursor: pointer; transition: background 0.15s'}
                  onmouseenter=${e => { e.currentTarget.style.background = palette.surface0; }}
                  onmouseleave=${e => { e.currentTarget.style.background = palette.base; }}
                >
                  <span style=${'font-size: var(--font-size-sm); color: ' + fileColor(f.name)}>${f.name}</span>
                  <div style="display: flex; align-items: center; gap: var(--space-2)">
                    ${previewable && html`<span style=${'font-size: var(--font-size-xs); color: ' + palette.overlay0 + '; font-style: italic'}>preview</span>`}
                    <span class="text-dim" style="font-size: var(--font-size-xs)">${humanSize(f.size_bytes)}</span>
                  </div>
                </div>
              `;
            })}
          </div>
        </div>
      `}
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
