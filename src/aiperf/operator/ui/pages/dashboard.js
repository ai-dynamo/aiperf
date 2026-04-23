import { html } from 'htm/preact';
import { useState, useEffect } from 'preact/hooks';
import { api, poll } from '../lib/api.js';
import { jobs, clusterInfo } from '../lib/state.js';
import { phaseColor, modelColor, palette, colors } from '../lib/theme.js';
import { navigate } from '../lib/router.js';
import { KpiCard } from '../components/kpi-card.js';
import { ChartWrapper } from '../components/chart-wrapper.js';
import { fmtNumber, fmtInt, fmtThroughput, fmtLatencyStr } from '../lib/format.js';

function formatElapsed(ms) {
  const s = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(s / 60);
  const h = Math.floor(m / 60);
  if (h > 0) return `${h}h ${m % 60}m`;
  if (m > 0) return `${m}m ${s % 60}s`;
  return `${s}s`;
}

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
  const gpus = cluster?.gpus ?? cluster?.gpuCount ?? cluster?.gpu_count ?? '?';
  const nodes = cluster?.nodes ?? cluster?.nodeCount ?? cluster?.node_count ?? '?';

  return html`
    <div class="status-bar">
      <div class="status-item"><div class="status-dot${running > 0 ? ' live' : ''}"></div> <span class="status-val">${fmtInt(running)}</span> running</div>
      <span class="status-sep">\u00b7</span>
      <div class="status-item"><span class="status-val">${fmtInt(completed)}</span> completed</div>
      ${failed > 0 ? html`<span class="status-sep">\u00b7</span><div class="status-item"><span class="status-val" style="color:var(--red)">${fmtInt(failed)}</span> failed</div>` : null}
      <span class="status-sep">\u00b7</span>
      <div class="status-item"><span class="status-val">${gpus}</span> GPUs</div>
      <span class="status-sep">\u00b7</span>
      <div class="status-item"><span class="status-val">${nodes}</span> nodes</div>
      ${best.value != null ? html`
        <span class="status-sep">\u00b7</span>
        <div class="status-item">Best: <span class="status-val">${fmtThroughput(best.value)}</span> req/s ${best.name ? html`<span style="color:var(--muted)">(${best.name})</span>` : null}</div>
      ` : null}
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
      backend: j.backend ?? '',
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
            return `${ctx.dataset.label} @ ${pt.backend}\n${fmtNumber(pt.x, 1)} ${xUnit}, ${fmtNumber(pt.y, 0)} ${yUnit}`;
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
        <div class="card-title" style="margin:0">Throughput vs Latency</div>
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

  useEffect(() => {
    const ac = new AbortController();
    poll(async () => {
      const data = await api.listJobs();
      const list = data?.jobs ?? [];
      jobs.value = list;
      setLocalJobs(list);
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

  return html`
    <div class="dashboard">
      ${clusterError && html`<div class="cluster-warning-banner">Cluster endpoint unavailable — data may be stale.</div>`}

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

      <!-- Section 4: Active Jobs -->
      <div class="section-header">
        <span class="section-title">Active Jobs</span>
        <span class="text-dim" style="font-size: var(--font-size-sm)">
          ${running.length} job${running.length !== 1 ? 's' : ''}
        </span>
      </div>

      ${running.length === 0
        ? html`
          <div class="empty-state card">
            <p class="text-dim">No active jobs. Start a benchmark with <code>aiperf kube run</code>.</p>
          </div>
        `
        : running.map(job => {
            const phase = job.phase ?? 'Unknown';
            const pct = Math.round(job.progressPercent ?? 0);
            const color = phaseColor(phase);
            const startTime = job.startTime;
            const elapsed = startTime ? formatElapsed(Date.now() - new Date(startTime).getTime()) : null;

            return html`
              <div
                key=${job.namespace + '/' + job.name}
                class="job-card"
                onclick=${() => navigate('/jobs/' + encodeURIComponent(job.namespace) + '/' + encodeURIComponent(job.name))}
                style="cursor:pointer;margin-bottom:var(--space-3)"
              >
                <div style="display:grid;grid-template-columns:1fr auto;gap:8px;align-items:start">
                  <div>
                    <div style="display:flex;align-items:center;gap:8px">
                      <div class="job-indicator running"></div>
                      <span class="job-name">${job.name}</span>
                      <span class="job-badge running">${phase}</span>
                    </div>
                    <div class="text-dim" style="font-size:var(--font-size-sm);margin-top:4px;display:flex;gap:8px;flex-wrap:wrap">
                      ${job.model ? html`<span>${job.model}</span>` : null}
                      ${job.backend ? html`<span>\u00b7 ${job.backend}</span>` : null}
                      ${elapsed ? html`<span>\u00b7 ${elapsed}</span>` : null}
                      ${job.gpuConfig ? html`<span>\u00b7 ${job.gpuConfig}</span>` : null}
                    </div>
                  </div>
                  <div style="text-align:right">
                    ${job.throughputRps != null ? html`
                      <div style="font-size:24px;font-weight:700;color:${palette.text};line-height:1">${fmtThroughput(job.throughputRps)}</div>
                      <div style="font-size:11px;color:${palette.muted}">req/s</div>
                    ` : null}
                  </div>
                </div>
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
          return html`
            <div
              key=${job.namespace + '/' + job.name}
              class="job-card"
              onclick=${() => navigate('/jobs/' + encodeURIComponent(job.namespace) + '/' + encodeURIComponent(job.name))}
              style="cursor:pointer;margin-bottom:var(--space-3);border-color:${palette.red}44"
            >
              <div style="display:flex;align-items:center;gap:8px">
                <div class="job-indicator failed"></div>
                <span class="job-name">${job.name}</span>
                <span class="job-badge failed">${job.phase ?? 'Failed'}</span>
              </div>
              ${job.error ? html`
                <div style="font-size:var(--font-size-sm);color:${palette.red};margin-top:4px">${job.error}</div>
              ` : null}
            </div>
          `;
        })}
      ` : null}

      <!-- Section 6: Leaderboard Preview -->
      ${top5.length > 0 ? html`
        <div class="section-header" style="margin-top:24px">
          <div class="section-title">Leaderboard</div>
          <button class="nav-tab" onclick=${() => navigate('/leaderboard')} style="font-size:12px;padding:4px 10px;">View All \u2192</button>
        </div>
        <table class="compare-table">
          <thead>
            <tr>
              <th style="width:40px">#</th>
              <th>Configuration</th>
              <th>Backend</th>
              <th style="width:200px">Throughput</th>
              <th style="width:200px">Latency P99</th>
              <th>TTFT</th>
              <th>GPUs</th>
            </tr>
          </thead>
          <tbody>
            ${top5.map((job, i) => {
              const tpsVal = job.throughputRps ?? 0;
              const latVal = job.latencyP99Ms ?? 0;
              const tpsPct = maxThroughput > 0 ? (tpsVal / maxThroughput) * 100 : 0;
              const latPct = maxLatency > 0 ? (latVal / maxLatency) * 100 : 0;
              const mColor = modelColor(job.model);

              return html`
                <tr
                  key=${job.namespace + '/' + job.name}
                  onclick=${() => navigate('/jobs/' + encodeURIComponent(job.namespace) + '/' + encodeURIComponent(job.name))}
                  style="cursor:pointer"
                >
                  <td><span class="rank${i === 0 ? ' gold' : ''}">${i + 1}</span></td>
                  <td>
                    <div class="model-cell">
                      <span class="model-color" style="background:${mColor}"></span>
                      <span class="model-name">${job.model ?? job.name}</span>
                    </div>
                  </td>
                  <td>${job.backend ?? '---'}</td>
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
                  <td>${job.ttftMs != null ? fmtNumber(job.ttftMs, 0) + ' ms' : '---'}</td>
                  <td>${job.gpuConfig ?? '---'}</td>
                </tr>
              `;
            })}
          </tbody>
        </table>
      ` : null}
    </div>
  `;
}
