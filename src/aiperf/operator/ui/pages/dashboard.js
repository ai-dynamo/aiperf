// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Dashboard — CONSOLE redesign (ported from
 * ``tools/operator-ui-console-preview.html``).
 *
 * Layout:
 *   <section class="fleet-status">   — aggregate verdict across all live runs
 *   <section class="meter-bank">     — 5 KPI slots (cluster-wide)
 *   <Panel title="SCATTER · TPS × P99">  — chart
 *   <Panel title="ACTIVE RUNS">      — one row per Running job
 *   <Panel title="LEADERBOARD">      — top 4 completed jobs
 *   <div class="console-footnote">   — failed-run strip, if any
 *
 * Data wiring unchanged: jobs list + per-job summaries + cluster info, all
 * fetched via ``api`` helpers with the existing ``poll(...)`` signatures.
 * ``data-testid`` attributes preserved so the 45 e2e tests still pass
 * (``page-dashboard``, ``dashboard-hero``, ``kpi-running``, ``kpi-completed``,
 *  ``kpi-peak-throughput``, ``kpi-best-ttft``, ``kpi-token-throughput``).
 */

import { html } from 'htm/preact';
import { useState, useEffect } from 'preact/hooks';
import { api, poll } from '../lib/api.js';
import { jobs, clusterInfo } from '../lib/state.js';
import { modelColor, palette } from '../lib/theme.js';
import { navigate } from '../lib/router.js';
import { KpiCard } from '../components/kpi-card.js';
import { ChartWrapper } from '../components/chart-wrapper.js';
import { fmtNumber, fmtInt, fmtThroughput } from '../lib/format.js';
import { applyChartTheme } from '../lib/chart-theme.js';

/* ─────────────────────────── helpers ─────────────────────────── */

function formatElapsed(ms) {
  const s = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(s / 60);
  const h = Math.floor(m / 60);
  const d = Math.floor(h / 24);
  if (d > 0) return `${d}d ${h % 24}h`;
  if (h > 0) return `${h}h ${m % 60}m`;
  if (m > 0) return `${m}m ${s % 60}s`;
  return `${s}s`;
}

function etaFromPhase(phase) {
  if (!phase) return null;
  const total =
    phase.total_expected_requests ??
    phase.expected_requests ??
    phase.requestsTotal ??
    phase.requests_total ??
    null;
  const completed =
    phase.final_requests_completed ??
    phase.requestsCompleted ??
    phase.requests_completed ??
    phase.completed ??
    0;
  const startNs = phase.start_ns;
  if (!total || !startNs || completed <= 0) return null;
  const elapsedSec = (Date.now() - Number(startNs) / 1e6) / 1000;
  if (elapsedSec <= 0) return null;
  const rate = completed / elapsedSec;
  if (rate <= 0) return null;
  return Math.max(0, (total - completed) / rate);
}

function pickActivePhase(phaseMap) {
  const arr = Object.entries(phaseMap ?? {}).map(([name, p]) => ({ ...p, name }));
  const completedOf = p =>
    p.final_requests_completed ?? p.requestsCompleted ?? p.requests_completed ?? p.completed ?? 0;
  const done = p => p.complete === true;
  const explicit = arr.filter(p => p.active && !done(p));
  if (explicit.length > 0) {
    explicit.sort((a, b) => completedOf(b) - completedOf(a));
    return explicit[0];
  }
  const inProgress = arr.filter(p => !done(p) && completedOf(p) > 0);
  if (inProgress.length > 0) {
    inProgress.sort((a, b) => completedOf(b) - completedOf(a));
    return inProgress[0];
  }
  const pending = arr.filter(p => !done(p));
  return pending[0] ?? null;
}

function phaseProgressPct(phase) {
  if (!phase) return null;
  const total =
    phase.total_expected_requests ??
    phase.expected_requests ??
    phase.requestsTotal ??
    phase.requests_total ??
    null;
  const completed =
    phase.final_requests_completed ??
    phase.requestsCompleted ??
    phase.requests_completed ??
    phase.completed ??
    0;
  if (total && total > 0) return Math.min(100, (completed / total) * 100);
  if (phase.requestsProgressPercent != null) {
    return Math.min(100, Math.max(0, Number(phase.requestsProgressPercent)));
  }
  return null;
}

/** Evaluate one run's SLO status from its liveSummary + spec.benchmark.slos.
 *  Returns ``'ok' | 'warn' | 'err' | 'idle'``.
 */
function classifyRun(run) {
  const summary = run.liveSummary ?? run.summary ?? null;
  const slos = run.slos ?? null;
  if (!summary) return 'idle';
  if (!slos || Object.keys(slos).length === 0) return 'idle';
  let status = 'ok';
  for (const [key, thr] of Object.entries(slos)) {
    const probe =
      key === 'time_to_first_token'
        ? summary.ttft_p99_ms ?? summary.ttft_avg_ms
        : key === 'request_latency'
        ? summary.latency_p99_ms ?? summary.latency_avg_ms
        : key === 'inter_token_latency'
        ? summary.itl_p99_ms ?? summary.itl_avg_ms
        : null;
    if (probe == null || thr == null) continue;
    if (probe > thr) return 'err';
    if (probe > thr * 0.85) status = 'warn';
  }
  return status;
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

/* ─────────────── inline components (CONSOLE) ──────────────── */

/**
 * Fleet status hero — aggregate verdict across 0..N live runs.
 *
 * Verdict label logic:
 *   - 0 live runs  → "IDLE"
 *   - any failed   → "N FAILED"
 *   - all SLOs pass or no SLOs declared → "ALL ON TARGET" / "N LIVE"
 *   - some slipping → "X OF Y ON TARGET"
 *
 * The body sentence always names the running job(s) so e2e tests can look
 * for specific names ("live-run" etc.) inside the ``dashboard-hero``
 * wrapper. When all SLOs pass it includes the literal "On target" so the
 * case-sensitive ``to_contain_text("On target")`` assertion matches.
 */
function FleetStatus({ liveRuns, failedRuns, cluster, liveDetails, enabled }) {
  if (enabled === false) return null;
  const liveCount = liveRuns.length;
  const failCount = failedRuns.length;

  // Merge each run with its live detail (spec.benchmark.slos + liveSummary).
  const enriched = liveRuns.map(run => {
    const detail = liveDetails[`${run.namespace}/${run.name}`];
    const slos = detail?.slos ?? null;
    const summary = detail?.status?.liveSummary ?? detail?.status?.summary ?? null;
    return { ...run, slos, liveSummary: summary };
  });

  const perRunStatus = enriched.map(classifyRun);
  const errRuns = enriched.filter((_, i) => perRunStatus[i] === 'err');
  const warnRuns = enriched.filter((_, i) => perRunStatus[i] === 'warn');
  const okRuns = enriched.filter((_, i) => perRunStatus[i] === 'ok');

  const anyDeclared = enriched.some(r => r.slos && Object.keys(r.slos ?? {}).length > 0);

  let verdict;
  let glyphKind;
  if (liveCount === 0) {
    verdict = 'IDLE';
    glyphKind = 'idle';
  } else if (errRuns.length > 0) {
    verdict = `${errRuns.length} FAILED`;
    glyphKind = 'err';
  } else if (warnRuns.length > 0) {
    verdict = `${okRuns.length} OF ${liveCount} ON TARGET`;
    glyphKind = 'warn';
  } else if (anyDeclared && okRuns.length === liveCount) {
    verdict = 'ALL ON TARGET';
    glyphKind = 'ok';
  } else {
    verdict = liveCount === 1 ? '1 LIVE' : `${liveCount} LIVE`;
    glyphKind = 'ok';
  }

  // Namespaces in play
  const nsSet = new Set(liveRuns.map(r => r.namespace).filter(Boolean));
  const nsList = Array.from(nsSet);

  // Aggregate throughput (sum across live runs' summaries)
  let fleetRps = 0;
  let rpsKnown = false;
  let worstP99 = null;
  for (const run of enriched) {
    const s = run.liveSummary;
    if (s?.throughput_rps != null) {
      fleetRps += Number(s.throughput_rps);
      rpsKnown = true;
    }
    if (s?.latency_p99_ms != null) {
      worstP99 = worstP99 == null ? s.latency_p99_ms : Math.max(worstP99, s.latency_p99_ms);
    }
  }

  // Body sentence — always names the running job(s) so tests can find them.
  const nameList = liveRuns.map(r => r.name);
  const names = nameList.length <= 2
    ? nameList.join(' and ')
    : `${nameList.slice(0, 2).join(', ')}, and ${nameList.length - 2} more`;
  let bodyParts = [];
  if (liveCount === 0) {
    bodyParts.push('No benchmarks running right now.');
    if (cluster) {
      const g = cluster.gpus ?? cluster.gpuCount ?? cluster.gpu_count ?? null;
      const n = cluster.nodes ?? cluster.nodeCount ?? cluster.node_count ?? null;
      if (g != null || n != null) {
        bodyParts.push(`Cluster ready: ${g ?? '?'} GPUs across ${n ?? '?'} nodes.`);
      }
    }
  } else {
    const nsText = nsList.length === 0
      ? ''
      : nsList.length === 1
      ? ` in namespace ${nsList[0]}`
      : ` across ${nsList.slice(0, 2).join(' and ')}${nsList.length > 2 ? ` and ${nsList.length - 2} more` : ''}`;
    bodyParts.push(`Running ${liveCount === 1 ? '1 benchmark' : `${liveCount} benchmarks`}${nsText}: ${names}.`);
    if (rpsKnown) {
      bodyParts.push(`Aggregate throughput ${fleetRps.toFixed(1)} req/s.`);
    }
    if (errRuns.length > 0) {
      const firstErr = errRuns[0];
      bodyParts.push(`${errRuns.length} run${errRuns.length > 1 ? 's' : ''} violating SLOs (${firstErr.name}).`);
    } else if (warnRuns.length > 0) {
      const firstWarn = warnRuns[0];
      bodyParts.push(`${warnRuns.length} run${warnRuns.length > 1 ? 's' : ''} slipping toward SLO thresholds (${firstWarn.name}).`);
    } else if (anyDeclared && okRuns.length === liveCount) {
      bodyParts.push('On target across all declared SLOs.');
    } else if (!anyDeclared) {
      bodyParts.push('no SLOs declared — no live judgment.');
    }
  }
  const body = bodyParts.join(' ');

  const gpus = cluster?.gpus ?? cluster?.gpuCount ?? cluster?.gpu_count ?? null;
  const gpuCapacity = cluster?.gpuCapacity ?? cluster?.gpu_capacity ?? null;
  const nodes = cluster?.nodes ?? cluster?.nodeCount ?? cluster?.node_count ?? null;
  const totalSlos = enriched.reduce((n, r) => n + Object.keys(r.slos ?? {}).length, 0);
  const passedSlos = totalSlos - errRuns.length - warnRuns.length;

  return html`
    <div class="fleet-status reveal reveal-1" data-testid="dashboard-hero">
      <div class="fleet-head">
        <div class="fleet-label-row">
          <span class="console-label console-label--amber">▸ FLEET</span>
          <span class="console-label">STATUS</span>
          ${liveCount > 0 && html`<span class="console-label">▸ ${liveCount} LIVE · ${nsList.length || 0} NAMESPACE${nsList.length === 1 ? '' : 'S'}</span>`}
        </div>
        <h1 class="fleet-verdict">
          ${verdict}
          <span class=${'mark-glyph' + (glyphKind === 'warn' ? ' mark-glyph--warn' : glyphKind === 'err' ? ' mark-glyph--err' : glyphKind === 'idle' ? ' mark-glyph--idle' : '')}></span>
        </h1>
        <p class="fleet-body">${body}</p>
        <div class="fleet-chips">
          ${liveCount > 0 && html`<span class="fleet-chip fleet-chip--live">${liveCount} RUNNING</span>`}
          ${warnRuns.length > 0 && html`<span class="fleet-chip fleet-chip--warn">${warnRuns.length} SLIPPING</span>`}
          ${errRuns.length > 0 && html`<span class="fleet-chip fleet-chip--err">${errRuns.length} VIOLATED</span>`}
          ${failCount > 0 && html`<span class="fleet-chip fleet-chip--err">${failCount} FAILED</span>`}
          ${gpus != null && html`<span class="fleet-chip"><i class="ph ph-lightning"></i>${gpus}${gpuCapacity ? ` / ${gpuCapacity}` : ''} GPUs</span>`}
          ${nodes != null && html`<span class="fleet-chip"><i class="ph ph-stack"></i>${nodes} NODES</span>`}
          ${totalSlos > 0 && html`<span class="fleet-chip"><i class="ph ph-target"></i>${Math.max(0, passedSlos)} / ${totalSlos} SLOs</span>`}
        </div>
      </div>

      <div class="fleet-meters">
        <div class="fleet-meter">
          <span class="console-label">FLEET R/S</span>
          <span class="fleet-meter-val fleet-meter-val--amber">${rpsKnown ? fleetRps.toFixed(1) : '—'}</span>
        </div>
        <div class="fleet-meter">
          <span class="console-label">WORST P99</span>
          <span class=${'fleet-meter-val' + (errRuns.length ? ' fleet-meter-val--red' : '')}>
            ${worstP99 != null ? fmtNumber(worstP99, 0) : '—'}${worstP99 != null && html`<span class="unit">ms</span>`}
          </span>
        </div>
      </div>

      <div class="fleet-phases">
        <span class="console-label">CLUSTER LOAD</span>
        ${gpus != null && html`
          <div class="console-phase-line">
            <span class="console-phase-label">GPUs</span>
            <div class="console-phase-meter">
              <div class=${'console-phase-fill ' + (gpuCapacity ? 'console-phase-fill--active' : 'console-phase-fill--done')}
                   style=${'width: ' + (gpuCapacity ? Math.min(100, (gpus / gpuCapacity) * 100) : 100) + '%'}></div>
            </div>
            <span class="console-phase-count">${gpus}${gpuCapacity ? `/${gpuCapacity}` : ''}</span>
          </div>
        `}
        ${nodes != null && html`
          <div class="console-phase-line">
            <span class="console-phase-label">NODES</span>
            <div class="console-phase-meter">
              <div class="console-phase-fill console-phase-fill--done" style="width: 100%"></div>
            </div>
            <span class="console-phase-count">${nodes}</span>
          </div>
        `}
        <div class="console-phase-line">
          <span class="console-phase-label">SLOs</span>
          <div class="console-phase-meter">
            <div class=${'console-phase-fill ' + (errRuns.length > 0 ? 'console-phase-fill--active' : 'console-phase-fill--done')}
                 style=${'width: ' + (totalSlos > 0 ? (Math.max(0, passedSlos) / totalSlos) * 100 : 100) + '%'}></div>
          </div>
          <span class="console-phase-count">${totalSlos > 0 ? `${Math.max(0, passedSlos)}/${totalSlos}` : '—'}</span>
        </div>
      </div>
    </div>
  `;
}

/**
 * One row in the Active Runs panel.
 *
 * Props:
 *   - run: flat job info dict (name, namespace, model, concurrency, phase, ...)
 *   - detail: optional ``/api/v1/jobs/:ns/:name`` response (status.phases, liveSummary)
 */
function ActiveRunRow({ run, detail }) {
  const status = detail?.status ?? null;
  const summary = status?.liveSummary ?? status?.summary ?? null;
  const phaseMap = status?.phases ?? {};
  const active = pickActivePhase(phaseMap);
  const pct = phaseProgressPct(active);
  const eta = etaFromPhase(active);

  const slos = detail?.slos ?? null;
  const verdictKind = classifyRun({ ...run, liveSummary: summary, slos });
  const verdictLabel =
    verdictKind === 'err' ? 'SLO VIOLATED'
    : verdictKind === 'warn' ? 'SLO SLIPPING'
    : verdictKind === 'ok' ? 'ON TARGET'
    : 'LIVE';
  const verdictClass =
    verdictKind === 'err' ? 'verdict--err'
    : verdictKind === 'warn' ? 'verdict--warn'
    : verdictKind === 'ok' ? 'verdict--ok'
    : 'verdict--idle';

  const rps = summary?.throughput_rps;
  const ttft = summary?.ttft_p99_ms ?? summary?.ttft_avg_ms;
  const p99 = summary?.latency_p99_ms ?? summary?.latency_avg_ms;
  const rpsClass = verdictKind === 'warn' || verdictKind === 'err' ? 'is-amber' : '';
  const ttftClass = verdictKind === 'err' ? 'is-red' : verdictKind === 'warn' ? 'is-amber' : '';

  const completedCount = active?.final_requests_completed ?? active?.requestsCompleted ?? active?.completed ?? 0;
  const totalCount = active?.total_expected_requests ?? active?.expected_requests ?? null;

  const conc = run.concurrency ?? run.maxConcurrency ?? null;
  const concStr = conc != null ? `CONC ${conc}` : '';
  const subParts = [run.namespace, run.model, concStr].filter(Boolean).map(s => String(s).toUpperCase());

  return html`
    <div
      class="run-row"
      onclick=${() => navigate('/jobs/' + encodeURIComponent(run.namespace) + '/' + encodeURIComponent(run.name))}
      data-testid=${'run-row-' + run.namespace + '-' + run.name}
    >
      <span class="run-dot running"></span>
      <div class="run-name">${run.name}<small>${subParts.join(' · ')}</small></div>
      <span class=${'verdict ' + verdictClass}>${verdictLabel}</span>
      <div class="run-stats">
        <div class="run-stat">
          <span class="console-label">R/S</span>
          <span class=${'run-stat-val ' + rpsClass}>${rps != null ? fmtNumber(rps, 1) : '—'}</span>
        </div>
        <div class="run-stat">
          <span class="console-label">TTFT</span>
          <span class=${'run-stat-val ' + ttftClass}>
            ${ttft != null ? fmtNumber(ttft, 0) : '—'}${ttft != null && html`<small>ms</small>`}
          </span>
        </div>
        <div class="run-stat">
          <span class="console-label">P99</span>
          <span class="run-stat-val">
            ${p99 != null ? fmtNumber(p99, 0) : '—'}${p99 != null && html`<small>ms</small>`}
          </span>
        </div>
      </div>
      <div class="run-prog">
        <div class="run-prog-meta">
          <span>${(active?.name ?? run.phase ?? 'RUNNING').toUpperCase()} · ${pct != null ? fmtNumber(pct, 0) + '%' : '—'}</span>
          <span>${fmtInt(completedCount)}${totalCount ? ` / ${fmtInt(totalCount)}` : ''}</span>
        </div>
        <div class="run-prog-track">
          <div class="run-prog-fill" style=${'width: ' + (pct ?? 0) + '%'}></div>
        </div>
      </div>
      <div class="run-eta"><small>ETA</small>${eta != null ? formatElapsed(eta * 1000) : '—'}</div>
      <i class="ph ph-arrow-right arrow"></i>
    </div>
  `;
}

/** Leaderboard row on the dashboard preview. */
function LeaderboardRow({ entry, rank, isTop, maxThroughput }) {
  const strokeColor = isTop ? 'var(--amber)' : 'var(--paper-faint)';
  const val = entry.throughputRps ?? 0;
  const w = maxThroughput > 0 ? Math.max(8, (val / maxThroughput) * 112) : 112;
  const yEnd = Math.max(4, 28 - (w / 112) * 24);
  const mColor = modelColor(entry.model);
  return html`
    <div
      class=${'ldr-row' + (isTop ? ' ldr-row--top' : '')}
      onclick=${() => navigate('/jobs/' + encodeURIComponent(entry.namespace) + '/' + encodeURIComponent(entry.name))}
    >
      <div class="ldr-rank">${String(rank).padStart(2, '0')}</div>
      <div class="ldr-name">
        ${entry.name}
        <small>${entry.model ?? '—'}${entry.concurrency ? ` · conc ${entry.concurrency}` : ''}</small>
      </div>
      <div class="ldr-config">${entry.backend ?? '—'}${entry.gpuConfig ? ` · ${entry.gpuConfig}` : ''}</div>
      <svg class="ldr-sparkline" viewBox="0 0 120 30" preserveAspectRatio="none">
        <polyline fill="none" stroke=${strokeColor} stroke-width=${isTop ? 1.6 : 1.4}
          points=${`0,22 15,20 30,18 45,${Math.max(6, yEnd + 8)} 60,${Math.max(5, yEnd + 5)} 75,${Math.max(5, yEnd + 3)} 90,${Math.max(4, yEnd + 2)} 105,${Math.max(4, yEnd + 1)} 120,${yEnd}`}/>
        <circle cx="120" cy=${yEnd} r=${isTop ? 2 : 1.8} fill=${strokeColor}/>
      </svg>
      <div class="ldr-val">${fmtNumber(val, 1)}<span class="unit">r/s</span></div>
    </div>
  `;
}

/** Rack panel wrapper — title, optional subtitle, optional action buttons. */
function Panel({ title, subtitle, actions, children, reveal }) {
  return html`
    <section class=${'console-panel' + (reveal ? ` reveal ${reveal}` : '')}>
      <div class="console-panel-header">
        <div class="console-panel-title">
          ${title}
          ${subtitle && html`<small>${subtitle}</small>`}
        </div>
        ${actions && html`<div class="console-panel-actions">${actions}</div>`}
      </div>
      <div class=${'console-panel-body' + (children && children.isChart ? ' console-panel-body--chart' : '')}>
        ${children}
      </div>
    </section>
  `;
}

/* ──────────────────── scatter chart ───────────────────── */

const AXIS_MODES = {
  tps_p99: { xField: 'throughputRps', yField: 'latencyP99Ms', xLabel: 'Throughput (req/s)', yLabel: 'Latency P99 (ms)' },
  tps_ttft: { xField: 'throughputRps', yField: 'ttftMs', xLabel: 'Throughput (req/s)', yLabel: 'TTFT (ms)' },
  tokps_p99: { xField: 'tokenThroughput', yField: 'latencyP99Ms', xLabel: 'Token Throughput (tok/s)', yLabel: 'Latency P99 (ms)' },
};

function ThroughputLatencyScatter({ completedJobs }) {
  const [axisMode, setAxisMode] = useState('tps_p99');
  const [logScale, setLogScale] = useState(false);

  if (!completedJobs || completedJobs.length === 0) {
    return html`
      <div style="padding: 60px 20px; text-align: center; font-family: var(--f-mono); font-size: 11px; color: var(--paper-faint); letter-spacing: 0.16em; text-transform: uppercase;">
        Awaiting completed runs
      </div>
    `;
  }

  const mode = AXIS_MODES[axisMode];
  const points = completedJobs.filter(j => j[mode.xField] != null && j[mode.yField] != null);
  if (points.length === 0) {
    return html`
      <div style="padding: 60px 20px; text-align: center; font-family: var(--f-mono); font-size: 11px; color: var(--paper-faint); letter-spacing: 0.16em; text-transform: uppercase;">
        No ${mode.xLabel} × ${mode.yLabel} pairs yet
      </div>
    `;
  }

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
    backgroundColor: modelColor(model),
    borderColor: modelColor(model),
    borderWidth: 1.5,
    pointRadius: 8,
    pointHoverRadius: 12,
  }));

  const scaleType = logScale ? 'logarithmic' : 'linear';
  const chartOptions = applyChartTheme({
    plugins: {
      legend: {
        position: 'top',
        align: 'end',
        labels: {
          usePointStyle: true,
          pointStyle: 'rect',
          boxWidth: 10,
          padding: 18,
          color: 'var(--paper-dim)',
          font: { family: "'JetBrains Mono', monospace", size: 10, weight: '600' },
        },
      },
      tooltip: {
        backgroundColor: 'rgba(14, 16, 20, 0.98)',
        borderColor: 'rgba(255, 159, 28, 0.32)',
        borderWidth: 1,
        padding: 12,
        cornerRadius: 0,
        titleFont: { family: "'JetBrains Mono', monospace", size: 11, weight: '700' },
        bodyFont: { family: "'JetBrains Mono', monospace", size: 10 },
        displayColors: false,
        callbacks: {
          label: ctx => {
            const pt = ctx.raw;
            const xUnit = mode.xLabel.includes('Token') ? 'tok/s' : 'req/s';
            const yUnit = 'ms';
            return `${ctx.dataset.label} @ ${pt.backend || '—'}: ${fmtNumber(pt.x, 1)} ${xUnit} · ${fmtNumber(pt.y, 0)} ${yUnit}`;
          },
        },
      },
    },
    scales: {
      x: {
        type: scaleType,
        title: {
          display: true,
          text: mode.xLabel.toUpperCase(),
          color: 'rgba(244,240,225,0.36)',
          font: { family: "'JetBrains Mono', monospace", size: 10, weight: '700' },
          padding: { top: 12 },
        },
        grid: { color: 'rgba(244,240,225,0.06)', drawTicks: false },
        border: { display: false },
        ticks: { padding: 8, color: 'rgba(244,240,225,0.36)', font: { family: "'JetBrains Mono', monospace", size: 10 } },
      },
      y: {
        type: scaleType,
        title: {
          display: true,
          text: mode.yLabel.toUpperCase(),
          color: 'rgba(244,240,225,0.36)',
          font: { family: "'JetBrains Mono', monospace", size: 10, weight: '700' },
          padding: { bottom: 12 },
        },
        grid: { color: 'rgba(244,240,225,0.06)', drawTicks: false },
        border: { display: false },
        ticks: { padding: 8, color: 'rgba(244,240,225,0.36)', font: { family: "'JetBrains Mono', monospace", size: 10 } },
      },
    },
  });

  return html`
    <${ChartWrapper}
      type="scatter"
      data=${{ datasets }}
      options=${chartOptions}
      height=${360}
    />
  `;
}

/* ─────────────────────── main page ────────────────────── */

export function Dashboard() {
  const [localJobs, setLocalJobs] = useState(jobs.value);
  const [cluster, setCluster] = useState(clusterInfo.value);
  const [clusterError, setClusterError] = useState(false);
  const [summaryMap, setSummaryMap] = useState({});
  const [liveDetails, setLiveDetails] = useState({}); // key: "<ns>/<name>" -> { status, slos }
  const [axisMode, setAxisMode] = useState('tps_p99');

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
    poll(async () => {
      try {
        const lb = await api.getLeaderboard('request_throughput', 'avg');
        const entries = lb?.entries ?? [];
        if (entries.length === 0) return;
        const results = await Promise.allSettled(
          entries.map(e =>
            api.getJobSummary(e.namespace, e.job_id).then(s => ({ id: e.job_id, summary: s }))
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
        setSummaryMap(prev => ({ ...prev, ...newEntries }));
      } catch (_e) { /* leaderboard may not exist yet */ }
    }, 15000, ac.signal);
    return () => ac.abort();
  }, []);

  const allJobs = enrichJobsFromSummaries(localJobs, summaryMap);
  const running = allJobs.filter(j => {
    if (j.source === 'archived') return false;
    const p = (j.phase ?? '').toLowerCase();
    return p === 'running' || p === 'initializing' || p === 'pending';
  });
  const completed = allJobs.filter(j => {
    const p = (j.phase ?? '').toLowerCase();
    return p === 'completed' || p === 'succeeded';
  });
  const failed = allJobs.filter(j => {
    const p = (j.phase ?? '').toLowerCase();
    return p === 'failed' || p === 'error';
  });

  // Fetch per-run detail + config for every live run — gated by identity so
  // we don't kick off a fresh fetch on every parent render.
  const liveKeys = running.map(r => `${r.namespace}/${r.name}`).join('|');
  useEffect(() => {
    if (running.length === 0) {
      setLiveDetails({});
      return;
    }
    const ac = new AbortController();
    for (const run of running) {
      const key = `${run.namespace}/${run.name}`;
      poll(async () => {
        try {
          const [detail, cfg] = await Promise.all([
            api.getJob(run.namespace, run.name).catch(() => null),
            api.getJobConfig(run.namespace, run.name).catch(() => null),
          ]);
          const slos = cfg?.spec?.benchmark?.slos ?? cfg?.spec?.slos ?? null;
          setLiveDetails(prev => ({
            ...prev,
            [key]: { status: detail?.status ?? null, slos },
          }));
        } catch (_e) { /* transient */ }
      }, 5000, ac.signal);
    }
    return () => ac.abort();
  }, [liveKeys]);

  const best = findBest(allJobs, 'throughputRps');
  const bestTtft = findMin(allJobs, 'ttftMs');
  const bestTokenTps = findBest(allJobs, 'tokenThroughput');

  // Leaderboard top 4 (completed, sorted by throughput)
  const top4 = [...completed]
    .filter(j => j.throughputRps != null)
    .sort((a, b) => (b.throughputRps ?? 0) - (a.throughputRps ?? 0))
    .slice(0, 4);
  const maxTop4Throughput = top4.reduce((mx, j) => Math.max(mx, j.throughputRps ?? 0), 0) || 1;

  const axisButtons = html`
    <button class=${axisMode === 'tps_p99' ? 'is-active' : ''} onclick=${() => setAxisMode('tps_p99')}>TPS · P99</button>
    <button class=${axisMode === 'tps_ttft' ? 'is-active' : ''} onclick=${() => setAxisMode('tps_ttft')}>TPS · TTFT</button>
    <button class=${axisMode === 'tokps_p99' ? 'is-active' : ''} onclick=${() => setAxisMode('tokps_p99')}>TOK · P99</button>
  `;

  return html`
    <div class="page-dashboard" data-testid="page-dashboard">
      ${clusterError && html`
        <div class="cluster-warning-banner" style="margin-bottom: 16px;">
          Cluster endpoint unavailable — data may be stale.
        </div>
      `}

      <${FleetStatus}
        liveRuns=${running}
        failedRuns=${failed}
        cluster=${cluster}
        liveDetails=${liveDetails}
        enabled=${running.length > 0}
      />

      <!-- METER BANK — 5 KPI slots (test-ids preserved) -->
      <section class="meter-bank reveal reveal-2">
        <${KpiCard}
          label="Running"
          value=${running.length}
          icon="ph-play"
          tone=${running.length > 0 ? 'amber' : undefined}
          sub=${running.length === 1 && running[0].startTime
            ? formatElapsed(Date.now() - new Date(running[0].startTime).getTime()) + ' elapsed'
            : (running.length === 0 ? 'idle' : `${running.length} live`)}
        />
        <${KpiCard}
          label="Completed"
          value=${completed.length}
          icon="ph-check"
          sub=${failed.length > 0
            ? html`${completed.length} succ · <span style="color: var(--red)">${failed.length} fail</span>`
            : completed.length + ' in 7d'}
        />
        <${KpiCard}
          label="Peak Throughput"
          value=${best.value != null ? fmtThroughput(best.value) : '---'}
          unit=${best.value != null ? 'req/s' : ''}
          tone=${best.value != null ? 'amber' : undefined}
          icon="ph-trend-up"
          sub=${best.name ? html`<strong>${best.name}</strong>` : ''}
        />
        <${KpiCard}
          label="Best TTFT"
          value=${bestTtft.value != null ? fmtNumber(bestTtft.value, 0) : '---'}
          unit=${bestTtft.value != null ? 'ms' : ''}
          icon="ph-timer"
          sub=${bestTtft.name ? html`<strong>${bestTtft.name}</strong>` : ''}
        />
        <${KpiCard}
          label="Token Throughput"
          value=${bestTokenTps.value != null ? fmtInt(bestTokenTps.value) : '---'}
          unit=${bestTokenTps.value != null ? 'tok/s' : ''}
          icon="ph-activity"
          sub=${bestTokenTps.name ? html`<strong>${bestTokenTps.name}</strong>` : ''}
        />
      </section>

      <!-- SCATTER panel -->
      <${Panel}
        title="SCATTER · TPS × P99"
        subtitle="all stored runs"
        actions=${axisButtons}
        reveal="reveal-3"
      >
        <div class="console-panel-body--chart" style="height: 400px">
          <${ThroughputLatencyScatter} completedJobs=${completed} />
        </div>
      </${Panel}>

      <!-- ACTIVE RUNS panel -->
      <${Panel}
        title="ACTIVE RUNS"
        subtitle=${running.length > 0 ? `${running.length} live · ${new Set(running.map(r => r.namespace)).size} namespace${new Set(running.map(r => r.namespace)).size === 1 ? '' : 's'}` : 'idle'}
        reveal="reveal-4"
      >
        ${running.length === 0
          ? html`<div class="run-list-empty">No active runs · start one with <code style="color: var(--amber); font-family: var(--f-mono)">aiperf kube run</code></div>`
          : html`
              <div class="run-list">
                ${running.map(run => html`
                  <${ActiveRunRow}
                    key=${run.namespace + '/' + run.name}
                    run=${run}
                    detail=${liveDetails[`${run.namespace}/${run.name}`]}
                  />
                `)}
              </div>
            `}
      </${Panel}>

      <!-- LEADERBOARD panel -->
      ${top4.length > 0 && html`
        <${Panel}
          title="LEADERBOARD · REQ/S"
          subtitle=${`top ${top4.length} past 7d`}
          actions=${html`<button onclick=${() => navigate('/leaderboard')}>VIEW ALL →</button>`}
          reveal="reveal-5"
        >
          <div>
            ${top4.map((entry, i) => html`
              <${LeaderboardRow}
                key=${entry.namespace + '/' + entry.name}
                entry=${entry}
                rank=${i + 1}
                isTop=${i === 0}
                maxThroughput=${maxTop4Throughput}
              />
            `)}
          </div>
        </${Panel}>
      `}

      ${failed.length > 0 && html`
        <div class="console-footnote" data-testid="dashboard-failed-strip">
          <span>${failed.length} RUN FAILED · <span class="red">${failed.map(f => f.name).slice(0, 3).join(' · ')}</span></span>
          <button onclick=${() => navigate('/jobs')}>▸ INVESTIGATE</button>
        </div>
      `}
    </div>
  `;
}
