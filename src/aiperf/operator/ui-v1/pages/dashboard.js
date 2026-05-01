import { html } from 'htm/preact';
import { useState, useEffect } from 'preact/hooks';
import { api, poll } from '../lib/api.js';
import { jobs, clusterInfo } from '../lib/state.js';
import { phaseColor, modelColor, palette, colors } from '../lib/theme.js';
import { buildJobPath, navigate } from '../lib/router.js';
import { KpiCard } from '../components/kpi-card.js';
import { ChartWrapper } from '../components/chart-wrapper.js';
import { NsPill, ModelPill } from '../components/pills.js';
import { RelativeTime } from '../components/time.js';
import { LoadingPanel } from '../components/spinner.js';
import { fmtNumber, fmtInt, fmtThroughput, fmtLatencyStr } from '../lib/format.js';

function findBest(jobList, field) {
  let best = null;
  let bestName = null;
  for (const job of jobList) {
    const phase = (job.phase ?? '').toLowerCase();
    if (phase !== 'completed' && phase !== 'succeeded') continue;
    const val = job[field] ?? null;
    if (val != null && (best === null || val > best)) {
      best = val;
      bestName = job.name;
    }
  }
  return { value: best, name: bestName };
}

function findMin(jobList, field) {
  let best = null;
  let bestName = null;
  for (const job of jobList) {
    const phase = (job.phase ?? '').toLowerCase();
    if (phase !== 'completed' && phase !== 'succeeded') continue;
    const val = job[field] ?? null;
    if (val != null && (best === null || val < best)) {
      best = val;
      bestName = job.name;
    }
  }
  return { value: best, name: bestName };
}

// --- Section 1: StatusBar ---
//
// Material-style KPI tiles mirroring the cluster banner layout: each
// tile gets an outlined SVG icon, label, big number, and a sub-line of
// chips/dots. The full bar is one card with a flex grid that wraps to
// 2 columns on narrow screens.

function StatusIcon({ kind }) {
  // Tiny outlined SVGs tuned to 24\u00d724, 1.8 stroke \u2014 same visual idiom as
  // the ClusterStatsBanner. ``aria-hidden`` because the surrounding tile
  // already carries the label text for screen readers.
  if (kind === 'running') {
    return html`
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <circle cx="12" cy="12" r="9" />
        <polygon points="10,8 16,12 10,16" fill="currentColor" stroke="none" />
      </svg>
    `;
  }
  if (kind === 'completed') {
    return html`
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <circle cx="12" cy="12" r="9" />
        <polyline points="8,12.5 11,15.5 16,9.5" />
      </svg>
    `;
  }
  if (kind === 'failed') {
    return html`
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M10.3 3.5 L1.5 19 a2 2 0 0 0 1.7 3 h17.6 a2 2 0 0 0 1.7 -3 L13.7 3.5 a2 2 0 0 0 -3.4 0 z" />
        <line x1="12" y1="10" x2="12" y2="14" />
        <circle cx="12" cy="17.2" r="0.6" fill="currentColor" stroke="none" />
      </svg>
    `;
  }
  if (kind === 'best') {
    return html`
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M8 21 h8" />
        <path d="M12 17 v4" />
        <path d="M7 4 h10 v3 a5 5 0 0 1 -10 0 z" />
        <path d="M17 5 h3 v2 a3 3 0 0 1 -3 3" />
        <path d="M7 5 h-3 v2 a3 3 0 0 0 3 3" />
        <path d="M9 13 a4 4 0 0 0 6 0" />
      </svg>
    `;
  }
  if (kind === 'cluster') {
    return html`
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <rect x="3" y="4" width="18" height="5" rx="1.2" />
        <rect x="3" y="10" width="18" height="5" rx="1.2" />
        <rect x="3" y="16" width="18" height="4" rx="1.2" />
        <circle cx="6.5" cy="6.5" r="0.8" fill="currentColor" stroke="none" />
        <circle cx="6.5" cy="12.5" r="0.8" fill="currentColor" stroke="none" />
        <circle cx="6.5" cy="18" r="0.8" fill="currentColor" stroke="none" />
      </svg>
    `;
  }
  return null;
}

function StatusBar({ allJobs, cluster, best }) {
  const running = allJobs.filter(j => {
    const p = (j.phase ?? '').toLowerCase();
    return p === 'running' || p === 'initializing' || p === 'pending';
  }).length;
  const completed = allJobs.filter(j => {
    const p = (j.phase ?? '').toLowerCase();
    return p === 'completed' || p === 'succeeded';
  }).length;
  const failed = allJobs.filter(j => {
    const p = (j.phase ?? '').toLowerCase();
    return p === 'failed' || p === 'error';
  }).length;
  const total = allJobs.length;
  const failPct = total > 0 ? (failed / total) * 100 : 0;
  const gpusUsed = cluster?.gpus_used ?? null;
  const gpusTotal = cluster?.gpus ?? null;
  const nodes = cluster?.nodes ?? null;

  return html`
    <div class="status-bar status-bar--tiles" data-testid="dashboard-status-bar">
      <div class=${'status-tile' + (running > 0 ? ' status-tile--live' : '')} title="Jobs currently running, initializing, or pending">
        <div class="status-tile__icon status-tile__icon--accent"><${StatusIcon} kind="running" /></div>
        <div class="status-tile__body">
          <div class="status-tile__label">Running</div>
          <div class="status-tile__value">
            <span class=${'status-tile__num' + (running > 0 ? ' status-tile__num--accent' : ' status-tile__num--total')}>${fmtInt(running)}</span>
            ${running > 0 && html`<span class="status-tile__live-dot" aria-hidden="true"></span>`}
          </div>
          <div class="status-tile__sub">${total > 0 ? `of ${total} total` : 'no jobs yet'}</div>
        </div>
      </div>

      <div class="status-tile" title="Completed benchmark runs">
        <div class="status-tile__icon status-tile__icon--ok"><${StatusIcon} kind="completed" /></div>
        <div class="status-tile__body">
          <div class="status-tile__label">Completed</div>
          <div class="status-tile__value">
            <span class="status-tile__num status-tile__num--ok">${fmtInt(completed)}</span>
          </div>
          <div class="status-tile__sub">${total > 0 ? `${Math.round((completed / total) * 100)}% of total` : '\u2014'}</div>
        </div>
      </div>

      <div class="status-tile" title="Jobs that ended in a failure phase">
        <div class=${'status-tile__icon ' + (failed > 0 ? 'status-tile__icon--bad' : 'status-tile__icon--neutral')}><${StatusIcon} kind="failed" /></div>
        <div class="status-tile__body">
          <div class="status-tile__label">Failed</div>
          <div class="status-tile__value">
            <span class=${'status-tile__num' + (failed > 0 ? ' status-tile__num--bad' : ' status-tile__num--total')}>${fmtInt(failed)}</span>
          </div>
          <div class="status-tile__sub">
            ${failed > 0
              ? html`<span class="status-tile__chip status-tile__chip--bad">${failPct.toFixed(1)}% fail rate</span>`
              : 'no failures'}
          </div>
        </div>
      </div>

      ${gpusTotal != null && html`
        <div class="status-tile" title="GPUs in use across the cluster">
          <div class="status-tile__icon status-tile__icon--neutral"><${StatusIcon} kind="cluster" /></div>
          <div class="status-tile__body">
            <div class="status-tile__label">Cluster</div>
            <div class="status-tile__value">
              <span class="status-tile__num status-tile__num--total">${fmtInt(gpusUsed ?? 0)}</span>
              <span class="status-tile__num-sep">/</span>
              <span class="status-tile__num status-tile__num--total">${fmtInt(gpusTotal)}</span>
              <span class="status-tile__unit">GPUs</span>
            </div>
            <div class="status-tile__sub">${nodes != null ? `across ${fmtInt(nodes)} node${nodes === 1 ? '' : 's'}` : ''}</div>
          </div>
        </div>
      `}

      ${best?.value != null && html`
        <div class="status-tile" title="Highest request throughput observed across all completed jobs">
          <div class="status-tile__icon status-tile__icon--gold"><${StatusIcon} kind="best" /></div>
          <div class="status-tile__body">
            <div class="status-tile__label">Best Run</div>
            <div class="status-tile__value">
              <span class="status-tile__num status-tile__num--gold">${fmtThroughput(best.value)}</span>
              <span class="status-tile__unit">req/s</span>
            </div>
            <div class="status-tile__sub" style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis">
              ${best.name ?? '\u2014'}
            </div>
          </div>
        </div>
      `}
    </div>
  `;
}

// --- Section 2: ThroughputLatencyScatter ---

const AXIS_MODES = {
  tps_p99: { xField: 'throughputRps', yField: 'latencyP99Ms', xLabel: 'Throughput (req/s)', yLabel: 'Latency P99 (ms)' },
  tps_ttft: { xField: 'throughputRps', yField: 'ttftMs', xLabel: 'Throughput (req/s)', yLabel: 'TTFT (ms)' },
  tokps_p99: { xField: 'tokenThroughput', yField: 'latencyP99Ms', xLabel: 'Token Throughput (tok/s)', yLabel: 'Latency P99 (ms)' },
};

const quadrantPlugin = {
  id: 'quadrantLabels',
  afterDraw(chart) {
    const { ctx, chartArea: { left, right, top, bottom } } = chart;
    const midX = (left + right) / 2;

    ctx.save();
    ctx.font = '11px Inter, system-ui, sans-serif';
    ctx.fillStyle = palette.overlay0 + '60';
    ctx.textAlign = 'center';

    ctx.fillText('High Throughput, Low Latency', (midX + right) / 2, top + 16);
    ctx.fillText('Low Throughput, High Latency', (left + midX) / 2, bottom - 8);

    ctx.restore();
  },
};

// Chart.js UMD loads via a plain <script> tag while this module is imported
// asynchronously; at module-evaluation time, window.Chart may or may not exist.
// Register lazily from inside the component instead.
function ensureQuadrantPluginRegistered() {
  if (window.Chart && !window._quadrantPluginRegistered) {
    window.Chart.register(quadrantPlugin);
    window._quadrantPluginRegistered = true;
  }
}

function ThroughputLatencyScatter({ completedJobs }) {
  const [axisMode, setAxisMode] = useState('tps_p99');
  const [logScale, setLogScale] = useState(false);

  ensureQuadrantPluginRegistered();

  if (!completedJobs || completedJobs.length === 0) return null;

  const mode = AXIS_MODES[axisMode];
  const points = completedJobs.filter(
    j => j[mode.xField] != null && j[mode.yField] != null,
  );
  if (points.length === 0) return null;

  const modelGroups = {};
  for (const job of points) {
    const m = job.model ?? 'unknown';
    if (!modelGroups[m]) modelGroups[m] = [];
    modelGroups[m].push(job);
  }

  const datasets = Object.entries(modelGroups).map(([model, mjobs]) => ({
    label: model,
    data: mjobs.map(j => ({
      x: j[mode.xField],
      y: j[mode.yField],
      jobName: j.name,
    })),
    backgroundColor: modelColor(model) + 'cc',
    borderColor: modelColor(model),
    borderWidth: 1.5,
    pointRadius: 7,
    pointHoverRadius: 10,
  }));

  const scaleType = logScale ? 'logarithmic' : 'linear';
  const chartOptions = {
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: ctx => {
            const pt = ctx.raw;
            const xUnit = mode.xLabel.includes('tok/s') ? 'tok/s' : 'req/s';
            const yUnit = 'ms';
            return [
              `${ctx.dataset.label}${pt.jobName ? ' · ' + pt.jobName : ''}`,
              `${fmtNumber(pt.x, 1)} ${xUnit}, ${fmtNumber(pt.y, 0)} ${yUnit}`,
            ];
          },
        },
      },
      quadrantLabels: { enabled: true },
    },
    scales: {
      x: {
        type: scaleType,
        title: { display: true, text: mode.xLabel, color: palette.overlay1, font: { size: 11 } },
        ticks: { color: palette.muted, font: { size: 10 } },
        grid: { color: palette.border + '60' },
      },
      y: {
        type: scaleType,
        title: { display: true, text: mode.yLabel, color: palette.overlay1, font: { size: 11 } },
        ticks: { color: palette.muted, font: { size: 10 } },
        grid: { color: palette.border + '60' },
      },
    },
  };

  const models = Object.keys(modelGroups);

  return html`
    <div class="card" style="margin-bottom: var(--space-6)">
      <div class="scatter-header">
        <div style="display:flex;flex-direction:column;gap:2px;min-width:0">
          <div class="card-title" style="margin:0">Performance Scatter</div>
          <div style="font-size:18px;font-weight:600;color:${palette.text};line-height:1.2">Throughput vs Latency</div>
        </div>
        <div class="axis-toggles">
          <button class="nav-tab${axisMode === 'tps_p99' ? ' active' : ''}" onclick=${() => setAxisMode('tps_p99')}>TPS / P99</button>
          <button class="nav-tab${axisMode === 'tps_ttft' ? ' active' : ''}" onclick=${() => setAxisMode('tps_ttft')}>TPS / TTFT</button>
          <button class="nav-tab${axisMode === 'tokps_p99' ? ' active' : ''}" onclick=${() => setAxisMode('tokps_p99')}>Tok/s / P99</button>
          <button class="nav-tab${logScale ? ' active' : ''}" onclick=${() => setLogScale(!logScale)}>Log</button>
        </div>
      </div>
      <${ChartWrapper}
        type="scatter"
        data=${{ datasets }}
        options=${chartOptions}
        height=${280}
      />
      ${models.length > 1 ? html`
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:8px;padding:0 4px">
          ${models.map(m => html`
            <div key=${m} style="display:flex;align-items:center;gap:4px;font-size:11px;color:${palette.sub}">
              <span style="width:8px;height:8px;border-radius:50%;background:${modelColor(m)};display:inline-block"></span>
              ${m}
            </div>
          `)}
        </div>
      ` : null}
    </div>
  `;
}

// --- Main Dashboard ---

/**
 * Build a metrics map from per-job summaries fetched in parallel.
 * Single leaderboard call for throughput ranking, then fetch summaries
 * only for jobs that appear in the leaderboard (completed + have results).
 */
function enrichJobsFromSummaries(jobList, summaryMap) {
  return jobList.map(j => {
    const id = j.jobId ?? j.name;
    const s = summaryMap[id];
    if (!s) return j;
    return {
      ...j,
      throughputRps: j.throughputRps ?? s.throughputRps ?? null,
      latencyP99Ms: j.latencyP99Ms ?? s.latencyP99Ms ?? null,
      ttftMs: j.ttftMs ?? s.ttftMs ?? null,
      tokenThroughput: j.tokenThroughput ?? s.tokenThroughput ?? null,
    };
  });
}

export function Dashboard() {
  const [localJobs, setLocalJobs] = useState(jobs.value);
  const [cluster, setCluster] = useState(clusterInfo.value);
  const [clusterError, setClusterError] = useState(false);
  const [summaryMap, setSummaryMap] = useState({});
  // Block the dashboard body behind a spinner until the first /jobs
  // fetch returns. Without this, an empty cluster shows the entire
  // dashboard skeleton in its empty-state form before the first poll
  // resolves, which reads as "no data" rather than "still loading".
  const [firstJobsLoad, setFirstJobsLoad] = useState(jobs.value.length === 0);
  const [jobsError, setJobsError] = useState(null);

  useEffect(() => {
    const ac = new AbortController();
    poll(async () => {
      try {
        const data = await api.listJobs();
        const list = data?.jobs ?? [];
        jobs.value = list;
        setLocalJobs(list);
        setJobsError(null);
      } catch (err) {
        if (firstJobsLoad) setJobsError(err?.message ?? String(err));
        throw err;
      } finally {
        setFirstJobsLoad(false);
      }
    }, 5000, ac.signal);
    poll(async () => {
      try {
        const data = await api.getCluster();
        clusterInfo.value = data;
        setCluster(data);
        setClusterError(false);
      } catch (_e) { setClusterError(true); }
    }, 10000, ac.signal);
    // Single leaderboard call to discover which jobs have results,
    // then fetch per-job summaries for the top entries
    poll(async () => {
      try {
        const lb = await api.getLeaderboard('request_throughput', 'avg');
        const entries = lb?.entries ?? [];
        if (entries.length === 0) return;
        // Fetch summaries in parallel for all leaderboard entries
        const results = await Promise.allSettled(
          entries.map(e =>
            api.getJobSummary(e.namespace, e.job_id).then(s => ({ id: e.job_id, ns: e.namespace, summary: s }))
          )
        );
        const newEntries = {};
        for (const r of results) {
          if (r.status !== 'fulfilled') continue;
          const { id, summary: s } = r.value;
          newEntries[id] = {
            throughputRps: s?.request_throughput?.avg ?? null,
            latencyP99Ms: s?.request_latency?.p99 ?? null,
            ttftMs: s?.time_to_first_token?.avg ?? null,
            tokenThroughput: s?.output_token_throughput?.avg ?? null,
          };
        }
        setSummaryMap((prev) => ({ ...prev, ...newEntries }));
      } catch (_e) { /* not available yet */ }
    }, 15000, ac.signal);
    return () => ac.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const allJobs = enrichJobsFromSummaries(localJobs, summaryMap);
  const running = allJobs.filter(j => { const p = (j.phase ?? '').toLowerCase(); return p === 'running' || p === 'initializing' || p === 'pending'; });
  const completed = allJobs.filter(j => { const p = (j.phase ?? '').toLowerCase(); return p === 'completed' || p === 'succeeded'; });
  const failed = allJobs.filter(j => { const p = (j.phase ?? '').toLowerCase(); return p === 'failed' || p === 'error'; });

  const best = findBest(allJobs, 'throughputRps');
  const bestTtft = findMin(allJobs, 'ttftMs');
  const bestTokenTps = findBest(allJobs, 'tokenThroughput');

  const top5 = [...completed].sort((a, b) => (b.throughputRps ?? 0) - (a.throughputRps ?? 0)).slice(0, 5);
  const maxThroughput = top5.reduce((mx, j) => Math.max(mx, j.throughputRps ?? 0), 0) || 1;
  const maxLatency = top5.reduce((mx, j) => Math.max(mx, j.latencyP99Ms ?? 0), 0) || 1;

  if (firstJobsLoad) {
    return html`
      <div class="dashboard" data-testid="page-dashboard">
        <div class="card">
          <${LoadingPanel} label="Loading dashboard…" testid="dashboard-loading" />
        </div>
      </div>
    `;
  }

  if (jobsError) {
    return html`
      <div class="dashboard" data-testid="page-dashboard">
        <div class="card" style="border-color: var(--error); color: var(--error)" data-testid="dashboard-jobs-error">
          <div style="font-weight:600;margin-bottom:4px">Failed to load jobs</div>
          <div style="font-size:var(--font-size-sm);margin-bottom:8px">${jobsError}</div>
          <div style="font-size:var(--font-size-sm);color:var(--muted)">
            Check that the operator is reachable (try <code>aiperf kube status</code>) and that your kubeconfig context targets the right cluster.
          </div>
        </div>
      </div>
    `;
  }

  const noJobsAtAll = allJobs.length === 0;

  return html`
    <div class="dashboard" data-testid="page-dashboard">
      ${clusterError && html`<div class="cluster-warning-banner" title="The /cluster endpoint failed. GPU/node counts and topology may not reflect the live cluster.">Cluster endpoint unavailable — GPU/node counts may be stale. Check operator logs with <code>aiperf kube logs operator</code>.</div>`}

      ${noJobsAtAll ? html`
        <div class="empty-state card" style="text-align:center;padding:var(--space-6)" data-testid="dashboard-empty">
          <div style="font-size:18px;font-weight:600;margin-bottom:8px">No benchmarks yet</div>
          <p class="text-dim" style="margin:0 0 12px 0">
            Submit your first benchmark to see throughput, latency, and TTFT here.
          </p>
          <p class="text-dim" style="font-size:var(--font-size-sm);margin:0">
            Start one with <code>aiperf kube run --model &lt;model&gt; --url &lt;endpoint&gt;</code>,
            or scaffold a manifest with <code>aiperf kube init</code>.
          </p>
        </div>
      ` : html`
      <${StatusBar} allJobs=${allJobs} cluster=${cluster} best=${best} />

      <${ThroughputLatencyScatter} completedJobs=${completed} />

      <!-- Section 3: Metric cards -->
      <div class="metrics-row">
        <${KpiCard} label="Running" value=${running.length} color=${palette.blue} />
        <${KpiCard} label="Completed" value=${completed.length} color=${palette.green} />
        <${KpiCard} label="Peak Throughput" value=${best.value != null ? fmtThroughput(best.value) : '---'} unit=${best.value != null ? 'req/s' : ''} color=${palette.accent} sub=${best.name ?? ''} />
        <${KpiCard} label="Best TTFT" value=${bestTtft.value != null ? fmtNumber(bestTtft.value, 0) : '---'} unit=${bestTtft.value != null ? 'ms' : ''} color=${palette.cyan} sub=${bestTtft.name ?? ''} />
        <${KpiCard} label="Token Throughput" value=${bestTokenTps.value != null ? fmtInt(bestTokenTps.value) : '---'} unit=${bestTokenTps.value != null ? 'tok/s' : ''} color=${palette.amber} sub=${bestTokenTps.name ?? ''} />
      </div>
      <div class="text-dim" style="font-size:11px;margin-top:-8px;margin-bottom:var(--space-4);padding:0 4px">
        <span title="Time To First Token: latency from request send to first streamed token (lower is better)">TTFT</span> = time to first token,
        <span title="Inter-Token Latency: average time between successive output tokens">ITL</span> = inter-token latency,
        <span title="99th-percentile end-to-end request latency (lower is better)">P99</span> = 99th-percentile latency.
      </div>

      <!-- Section 4: Active Jobs -->
      <div class="section-header" style="margin-top:var(--space-6)">
        <span class="section-title">Active Jobs</span>
        <span class="text-dim" style="font-size: var(--font-size-sm)">
          ${running.length} job${running.length !== 1 ? 's' : ''}
        </span>
      </div>

      ${running.length === 0
        ? html`
          <div class="empty-state card">
            <p class="text-dim" style="margin:0">
              ${completed.length > 0
                ? html`No active jobs. ${completed.length} completed run${completed.length === 1 ? '' : 's'} below — start another with <code>aiperf kube run</code>.`
                : html`No active jobs. Start a benchmark with <code>aiperf kube run</code>.`}
            </p>
          </div>
        `
        : running.map(job => {
            const phase = job.phase ?? 'Unknown';
            const pct = Math.round(job.progressPercent ?? 0);
            const color = phaseColor(phase);
            const startTime = job.startTime;
            const workersReady = job.workersReady ?? 0;
            const workersTotal = job.workersTotal ?? 0;
            const showWorkers = workersTotal > 0;
            const errPctValue = job.errorRate != null ? job.errorRate * 100 : null;
            const errColor = errPctValue == null
              ? palette.muted
              : errPctValue >= 5 ? palette.red
              : errPctValue >= 1 ? palette.amber
              : palette.green;
            const liveMetrics = [
              { label: 'TTFT', value: job.ttftMs, fmt: v => fmtNumber(v, 0), unit: 'ms', help: 'Time To First Token (avg) — latency from request send to first streamed token' },
              { label: 'OutTok', value: job.outputTokenThroughputTps, fmt: v => fmtInt(v), unit: 'tok/s', help: 'Output token throughput — tokens generated per second across all in-flight requests' },
              { label: 'P99', value: job.latencyP99Ms, fmt: v => fmtNumber(v, 0), unit: 'ms', help: '99th-percentile end-to-end request latency' },
              { label: 'ITL', value: job.interTokenLatencyMs, fmt: v => fmtNumber(v, 1), unit: 'ms', help: 'Inter-Token Latency — average time between successive output tokens' },
              { label: 'Reqs', value: job.totalRequests, fmt: v => fmtInt(v), unit: '', help: 'Total requests issued so far in this run' },
            ].filter(m => m.value != null);

            const goToJob = () => navigate(buildJobPath(job));
            return html`
              <div
                key=${job.namespace + '/' + job.name}
                class="job-card"
                role="button"
                tabindex="0"
                aria-label=${'Open job ' + job.namespace + '/' + job.name}
                onclick=${goToJob}
                onkeydown=${e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); goToJob(); } }}
                onfocus=${e => { e.currentTarget.style.outline = '2px solid ' + palette.accent; e.currentTarget.style.outlineOffset = '2px'; }}
                onblur=${e => { e.currentTarget.style.outline = ''; e.currentTarget.style.outlineOffset = ''; }}
                style="cursor:pointer;margin-bottom:var(--space-3)"
              >
                <div style="display:grid;grid-template-columns:1fr auto;gap:8px;align-items:start">
                  <div>
                    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
                      <div class="job-indicator running"></div>
                      <span class="job-name">${job.name}</span>
                      <span class="job-badge running">${phase}</span>
                      ${job.currentPhase ? html`
                        <span class="job-subphase" title="Current benchmark phase">${job.currentPhase}</span>
                      ` : null}
                      <${NsPill} ns=${job.namespace} onClick=${ns => navigate('/jobs?ns=' + encodeURIComponent(ns))} testId=${'dashboard-active-ns-' + (job.namespace ?? '')} />
                      ${job.model && html`<${ModelPill} model=${job.model} testId=${'dashboard-active-model-' + (job.namespace ?? '')} />`}
                    </div>
                    <div class="text-dim" style="font-size:var(--font-size-sm);margin-top:4px;display:flex;gap:8px;flex-wrap:wrap;align-items:center">
                      ${startTime ? html`<${RelativeTime} ts=${startTime} mode="elapsed" />` : null}
                      ${showWorkers ? html`
                        <span title="Workers ready / total">\u00b7
                          <span style="color:${workersReady === workersTotal ? palette.green : palette.amber}">${workersReady}/${workersTotal}</span> workers
                        </span>
                      ` : null}
                    </div>
                  </div>
                  <div style="text-align:right">
                    ${job.throughputRps != null ? html`
                      <div style="font-size:24px;font-weight:700;color:${palette.text};line-height:1">${fmtThroughput(job.throughputRps)}</div>
                      <div style="font-size:11px;color:${palette.muted}">req/s</div>
                    ` : null}
                  </div>
                </div>
                ${liveMetrics.length > 0 || errPctValue != null ? html`
                  <div class="live-metric-strip" data-testid="dashboard-active-metrics">
                    ${liveMetrics.map(m => html`
                      <div class="live-metric" key=${m.label} title=${m.help + (m.unit ? ' (' + m.unit + ')' : '')}>
                        <span class="live-metric-label">${m.label}</span>
                        <span class="live-metric-value">${m.fmt(m.value)}</span>
                        ${m.unit ? html`<span class="live-metric-unit">${m.unit}</span>` : null}
                      </div>
                    `)}
                    ${errPctValue != null ? html`
                      <div class="live-metric" title=${'Errored requests as % of total — ' + (errPctValue >= 5 ? 'high error rate, investigate' : errPctValue >= 1 ? 'elevated error rate' : 'within tolerance')}>
                        <span class="live-metric-label">Err</span>
                        <span class="live-metric-value" style="color:${errColor}">
                          ${errPctValue >= 5 ? html`<span aria-label="high">! </span>` : null}${fmtNumber(errPctValue, errPctValue < 1 ? 2 : 1)}%
                        </span>
                      </div>
                    ` : null}
                  </div>
                ` : null}
                ${pct > 0 ? html`
                  <div class="progress-track" style="margin-top:8px">
                    <div class="progress-fill" style=${'width:' + pct + '%;background:' + color} />
                  </div>
                ` : null}
              </div>
            `;
          })
      }

      <!-- Section 5: Failed Jobs -->
      ${failed.length > 0 ? html`
        <div class="section-header" style="margin-top:var(--space-6)">
          <span class="section-title" style="color:${palette.red}">Failed Jobs</span>
          <span class="text-dim" style="font-size:var(--font-size-sm)">${failed.length}</span>
        </div>
        ${failed.map(job => {
          const goToFailed = () => navigate(buildJobPath(job));
          return html`
            <div
              key=${job.namespace + '/' + job.name}
              class="job-card"
              role="button"
              tabindex="0"
              aria-label=${'Open failed job ' + job.namespace + '/' + job.name}
              onclick=${goToFailed}
              onkeydown=${e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); goToFailed(); } }}
              onfocus=${e => { e.currentTarget.style.outline = '2px solid ' + palette.red; e.currentTarget.style.outlineOffset = '2px'; }}
              onblur=${e => { e.currentTarget.style.outline = ''; e.currentTarget.style.outlineOffset = ''; }}
              style="cursor:pointer;margin-bottom:var(--space-3);border-color:${palette.red}44"
            >
              <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
                <div class="job-indicator failed"></div>
                <span class="job-name">${job.name}</span>
                <span class="job-badge failed">${job.phase ?? 'Failed'}</span>
                <${NsPill} ns=${job.namespace} onClick=${ns => navigate('/jobs?ns=' + encodeURIComponent(ns))} testId=${'dashboard-failed-ns-' + (job.namespace ?? '')} />
              </div>
              ${job.error ? html`
                <div title=${job.error} style="font-size:var(--font-size-sm);color:${palette.red};margin-top:4px;word-break:break-word;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden">${job.error}</div>
              ` : null}
            </div>
          `;
        })}
      ` : null}

      <!-- Section 6: Leaderboard Preview -->
      ${top5.length > 0 ? html`
        <div class="section-header" style="margin-top:var(--space-6)">
          <div class="section-title">Leaderboard</div>
          <button class="nav-tab" onclick=${() => navigate('/leaderboard')} style="font-size:12px;padding:4px 10px;">View All \u2192</button>
        </div>
        <table class="compare-table">
          <thead>
            <tr>
              <th style="width:40px;text-align:right">#</th>
              <th>Configuration</th>
              <th style="width:200px">Throughput</th>
              <th style="width:200px">Latency P99</th>
              <th style="text-align:right">TTFT</th>
            </tr>
          </thead>
          <tbody>
            ${top5.map((job, i) => {
              const tpsVal = job.throughputRps ?? 0;
              const latVal = job.latencyP99Ms ?? 0;
              const tpsPct = maxThroughput > 0 ? (tpsVal / maxThroughput) * 100 : 0;
              const latPct = maxLatency > 0 ? (latVal / maxLatency) * 100 : 0;
              const mColor = modelColor(job.model);
              const goToLb = () => navigate(buildJobPath(job));

              return html`
                <tr
                  key=${job.namespace + '/' + job.name}
                  role="button"
                  tabindex="0"
                  aria-label=${'Open job ' + job.namespace + '/' + job.name + ', rank ' + (i + 1)}
                  onclick=${goToLb}
                  onkeydown=${e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); goToLb(); } }}
                  onfocus=${e => { e.currentTarget.style.outline = '2px solid ' + palette.accent; e.currentTarget.style.outlineOffset = '-2px'; }}
                  onblur=${e => { e.currentTarget.style.outline = ''; e.currentTarget.style.outlineOffset = ''; }}
                  style="cursor:pointer"
                >
                  <td><span class="rank${i === 0 ? ' gold' : ''}">${i + 1}</span></td>
                  <td>
                    <div class="model-cell">
                      <span class="model-color" style="background:${mColor}"></span>
                      <span
                        class="model-name"
                        title=${(job.model ?? job.name) + ' — ' + job.namespace + '/' + job.name}
                        style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:280px;display:inline-block;vertical-align:middle"
                      >${job.model ?? job.name}</span>
                    </div>
                  </td>
                  <td>
                    <div class="bar-cell">
                      <div class="inline-bar">
                        <div class="inline-bar-fill" style="width:${tpsPct}%;background:${palette.accent}"></div>
                      </div>
                      <span class="bar-val">${fmtThroughput(tpsVal)} req/s</span>
                    </div>
                  </td>
                  <td>
                    <div class="bar-cell">
                      <div class="inline-bar">
                        <div class="inline-bar-fill" style="width:${latPct}%;background:${palette.cyan}"></div>
                      </div>
                      <span class="bar-val">${fmtNumber(latVal, 0)} ms</span>
                    </div>
                  </td>
                  <td style="text-align:right;font-variant-numeric:tabular-nums">${job.ttftMs != null ? fmtNumber(job.ttftMs, 0) + ' ms' : '---'}</td>
                </tr>
              `;
            })}
          </tbody>
        </table>
      ` : null}
      `}
    </div>
  `;
}
