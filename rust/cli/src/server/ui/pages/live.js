// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Live in-flight view: subscribes to the orchestrator's `/api/live` SSE stream
// (the running child's heartbeat NDJSON, tailed server-side) and renders the
// current run's progress, counters, latency percentiles, and a live throughput/
// latency chart that grows as each heartbeat snapshot arrives. Idle when no run
// is in flight; reconnects automatically (EventSource) when a run starts.

import { html } from 'htm/preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import { KpiCard } from '../components/kpi-card.js';
import { ChartWrapper, CHART_PALETTE } from '../components/chart-wrapper.js';
import { fmtNumber, fmtInt, DASH } from '../lib/format.js';

const MAX_POINTS = 240; // ~ a few minutes of heartbeats

// Pull a percentile off a heartbeat sketch. The live heartbeat keys percentiles by
// the BARE number ("50", "99") — unlike the /summary projection's "pNN" — so accept
// either ("p99" -> "99").
function pct(sketch, key) {
  if (!sketch || !sketch.percentiles) return undefined;
  const p = sketch.percentiles;
  const bare = String(key).replace(/^p/, '');
  return p[bare] ?? p[key];
}

export function Live() {
  // status: 'connecting' | 'live' | 'idle' | 'error'
  const [status, setStatus] = useState('connecting');
  const [run, setRun] = useState(null); // { id, label, artifact_dir }
  const [hb, setHb] = useState(null); // latest heartbeat event
  const [, forceTick] = useState(0); // repaint on new chart points
  const series = useRef([]); // { t, completed, throughput, ttftP99, latP99 }
  const t0 = useRef(null);
  const prev = useRef(null); // previous heartbeat for rate derivation

  useEffect(() => {
    series.current = [];
    t0.current = null;
    prev.current = null;
    const es = new EventSource('/api/live');

    es.addEventListener('run', (e) => {
      const r = JSON.parse(e.data);
      setRun(r);
      setHb(null);
      setStatus('live');
      series.current = [];
      t0.current = null;
      prev.current = null;
    });

    es.addEventListener('idle', () => {
      setStatus('idle');
      setRun(null);
    });

    es.addEventListener('heartbeat', (e) => {
      const h = JSON.parse(e.data);
      setHb(h);
      setStatus('live');
      // Derive an instantaneous throughput (Δcompleted / Δt) between snapshots.
      const tSec = (h.observed_at_ns ?? 0) / 1e9;
      if (t0.current == null) t0.current = tSec;
      let throughput = null;
      if (prev.current) {
        const dt = tSec - prev.current.t;
        const dc = (h.counters?.completed ?? 0) - prev.current.completed;
        if (dt > 0.001) throughput = dc / dt;
      }
      prev.current = { t: tSec, completed: h.counters?.completed ?? 0 };
      series.current.push({
        t: +(tSec - t0.current).toFixed(1),
        completed: h.counters?.completed ?? 0,
        throughput,
        ttftP99: pct(h.ttft_ms, 'p99') ?? null,
        latP99: pct(h.latency_ms, 'p99') ?? null,
      });
      if (series.current.length > MAX_POINTS) series.current.shift();
      forceTick((n) => n + 1);
    });

    es.onopen = () => setStatus((s) => (s === 'error' ? 'connecting' : s));
    es.onerror = () => setStatus((s) => (s === 'live' ? 'live' : 'error'));

    return () => es.close();
  }, []);

  const counters = hb?.counters ?? {};
  const sat = hb?.saturation ?? {};
  const issued = counters.issued ?? 0;
  const completed = counters.completed ?? 0;
  const errored = counters.errored ?? 0;
  const inFlight = sat.in_flight ?? Math.max(0, issued - completed);
  const progress = issued > 0 ? Math.min(100, (100 * completed) / issued) : 0;
  const latest = series.current[series.current.length - 1];

  const dotClass =
    status === 'live' ? 'live-dot live' : status === 'idle' ? 'live-dot idle' : 'live-dot';
  const statusLabel =
    status === 'live'
      ? 'LIVE'
      : status === 'idle'
        ? 'IDLE'
        : status === 'error'
          ? 'RECONNECTING…'
          : 'CONNECTING…';

  return html`
    <div class="page">
      <div class="page-head">
        <div class="live-status">
          <span class=${dotClass}></span>
          <h1 class="live-title">${statusLabel}</h1>
        </div>
        ${run
          ? html`<div class="live-run-id">
              ${run.label} · <code>${run.artifact_dir}</code>
            </div>`
          : html`<div class="dim">No run in flight — start
              <code>aiperf profile --serve</code> and it appears here live.</div>`}
      </div>

      ${run &&
      html`
        <div class="card live-progress-card">
          <div class="live-progress-head">
            <span class="mono">${fmtInt(completed)} / ${fmtInt(issued)} requests</span>
            <span class="dim">${progress.toFixed(0)}%</span>
          </div>
          <div class="progress-track">
            <div class="progress-fill" style=${`width:${progress}%`}></div>
          </div>
          <div class="live-counter-row">
            <${Counter} label="in flight" value=${fmtInt(inFlight)} />
            <${Counter} label="completed" value=${fmtInt(completed)} accent />
            <${Counter} label="issued" value=${fmtInt(issued)} />
            <${Counter} label="errored" value=${fmtInt(errored)} err=${errored > 0} />
            <${Counter} label="req/s" value=${latest?.throughput != null ? fmtNumber(latest.throughput, 0) : DASH} />
          </div>
        </div>

        <div class="kpi-grid">
          <${KpiCard} icon="timer" label="TTFT p50" unit="ms" value=${fmtMs(pct(hb?.ttft_ms, 'p50'))} sub=${`p99 ${fmtMs(pct(hb?.ttft_ms, 'p99'))}`} />
          <${KpiCard} icon="clock" label="ITL p50" unit="ms" value=${fmtMs(pct(hb?.itl_ms, 'p50'))} sub=${`p99 ${fmtMs(pct(hb?.itl_ms, 'p99'))}`} />
          <${KpiCard} icon="speed" label="Latency p50" unit="ms" value=${fmtMs(pct(hb?.latency_ms, 'p50'))} sub=${`p99 ${fmtMs(pct(hb?.latency_ms, 'p99'))}`} />
          <${KpiCard} icon="tokens" label="Samples" value=${fmtInt(hb?.latency_ms?.count)} sub="completed reqs measured" />
        </div>

        <div class="chart-grid">
          <div class="card">
            <div class="card-title">Completed over time</div>
            <${ChartWrapper}
              type="line"
              height=${260}
              data=${completedChart(series.current)}
              options=${lineOpts('requests', 'seconds')}
            />
          </div>
          <div class="card">
            <div class="card-title">Latency p99 over time</div>
            <${ChartWrapper}
              type="line"
              height=${260}
              data=${latencyChart(series.current)}
              options=${lineOpts('ms', 'seconds')}
            />
          </div>
        </div>
      `}
    </div>
  `;
}

function Counter({ label, value, accent, err }) {
  const cls = 'live-counter' + (accent ? ' accent' : '') + (err ? ' err' : '');
  return html`<div class=${cls}><div class="lc-value">${value}</div><div class="lc-label">${label}</div></div>`;
}

function fmtMs(v) {
  return v == null ? DASH : fmtNumber(v, v < 1 ? 4 : 2);
}

function completedChart(pts) {
  return {
    labels: pts.map((p) => p.t),
    datasets: [
      {
        label: 'completed',
        data: pts.map((p) => p.completed),
        borderColor: CHART_PALETTE[0],
        backgroundColor: 'rgba(118,185,0,0.12)',
        fill: true,
        tension: 0.25,
        pointRadius: 0,
        borderWidth: 2,
      },
    ],
  };
}

function latencyChart(pts) {
  return {
    labels: pts.map((p) => p.t),
    datasets: [
      {
        label: 'request latency p99',
        data: pts.map((p) => p.latP99),
        borderColor: CHART_PALETTE[3] ?? '#c774e8',
        backgroundColor: 'transparent',
        tension: 0.25,
        pointRadius: 0,
        borderWidth: 2,
        spanGaps: true,
      },
      {
        label: 'TTFT p99',
        data: pts.map((p) => p.ttftP99),
        borderColor: CHART_PALETTE[2] ?? '#4a9eff',
        backgroundColor: 'transparent',
        tension: 0.25,
        pointRadius: 0,
        borderWidth: 2,
        spanGaps: true,
      },
    ],
  };
}

function lineOpts(yTitle, xTitle) {
  return {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    interaction: { intersect: false, mode: 'index' },
    scales: {
      x: { title: { display: true, text: xTitle } },
      y: { title: { display: true, text: yTitle }, beginAtZero: true },
    },
  };
}
