// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * RIGHT INSPECTOR — the contextual diagnostics pane on the right of the Flight
 * Deck.
 *
 * Content swaps based on what the main viewport is showing:
 *
 *   · viewKind=run       → selected run's SLOs, phase stack, workers, error
 *   · viewKind=overview  → cluster health block, GPU aggregate, alerts
 *   · (other)            → cluster health block
 *
 * The inspector never drives network traffic on its own — it reads from
 * signals populated by ``app.js`` + the RUN view. This keeps the pane fast to
 * mount/unmount and side-effect-free.
 */

import { html } from 'htm/preact';
import { useEffect, useState } from 'preact/hooks';
import { jobs, clusterInfo } from '../lib/state.js';
import { api, poll } from '../lib/api.js';
import { fmtInt, fmtNumber, fmtDuration, fmtPercent } from '../lib/format.js';

/* ──────────────────────── helpers ───────────────────────── */

function findJob(ns, name) {
  if (!ns || !name) return null;
  return (jobs.value ?? []).find(j => j.namespace === ns && j.name === name) ?? null;
}

function phaseBucket(phase) {
  const p = (phase ?? '').toLowerCase();
  if (p === 'running' || p === 'initializing' || p === 'pending') return 'live';
  if (p === 'failed'  || p === 'error')                            return 'fault';
  if (p === 'completed' || p === 'succeeded')                      return 'passed';
  return 'other';
}

function useRunDetail(ns, name, enabled) {
  const [detail, setDetail] = useState(null);
  const [config, setConfig] = useState(null);
  useEffect(() => {
    if (!enabled || !ns || !name) { setDetail(null); setConfig(null); return; }
    const ac = new AbortController();
    poll(async () => {
      try {
        const [d, c] = await Promise.all([
          api.getJob(ns, name).catch(() => null),
          api.getJobConfig(ns, name).catch(() => null),
        ]);
        setDetail(d);
        setConfig(c);
      } catch (_e) { /* transient */ }
    }, 4000, ac.signal);
    return () => ac.abort();
  }, [ns, name, enabled]);
  return { detail, config };
}

function slosFromConfig(config) {
  return config?.spec?.benchmark?.slos ?? config?.spec?.slos ?? null;
}

function evalSlo(slo, metrics) {
  if (!slo || typeof slo !== 'object') return [];
  const out = [];
  for (const [key, threshold] of Object.entries(slo)) {
    const probe =
      key === 'time_to_first_token'   ? metrics.ttft_p99_ms ?? metrics.ttft_avg_ms :
      key === 'request_latency'       ? metrics.latency_p99_ms ?? metrics.latency_avg_ms :
      key === 'inter_token_latency'   ? metrics.itl_p99_ms ?? metrics.itl_avg_ms :
      key === 'request_throughput'    ? metrics.throughput_rps :
      null;
    out.push({
      key,
      threshold,
      probe,
      kind: probe == null ? 'idle'
          : (key === 'request_throughput' ? probe >= threshold : probe <= threshold) ? 'pass' : 'fail',
    });
  }
  return out;
}

/* ────────────────────── run inspector ────────────────────── */

function RunInspector({ ns, name, open }) {
  const job = findJob(ns, name);
  const { detail, config } = useRunDetail(ns, name, open);

  const summary = detail?.status?.liveSummary ?? detail?.status?.summary ?? {};
  const phases = detail?.status?.phases ?? {};
  const slos = slosFromConfig(config);
  const sloResults = slos ? evalSlo(slos, summary) : [];
  const phaseEntries = Object.entries(phases);
  const bucket = phaseBucket(job?.phase);

  const elapsed = job?.startTime ? (Date.now() - new Date(job.startTime).getTime()) / 1000 : null;

  return html`
    <div class="insp-scroll">
      <div class="insp-hero">
        <div class="insp-hero-label">
          <span class=${'insp-hero-dot insp-hero-dot--' + bucket}></span>
          ${(job?.phase ?? 'UNKNOWN').toUpperCase()}
        </div>
        <div class="insp-hero-name">${name}</div>
        <div class="insp-hero-meta">
          <span>${ns}</span>
          ${job?.model && html`<span>${job.model.split('/').pop()}</span>`}
          ${job?.concurrency != null && html`<span>CONC ${job.concurrency}</span>`}
        </div>
        ${elapsed != null && html`
          <div class="insp-hero-elapsed">
            <span class="insp-hero-elapsed-label">ELAPSED</span>
            <span class="insp-hero-elapsed-val">${fmtDuration(elapsed)}</span>
          </div>
        `}
      </div>

      ${sloResults.length > 0 && html`
        <section class="insp-block">
          <div class="insp-block-head">
            <span class="insp-block-dot"></span>
            SLO GATES
          </div>
          <ul class="insp-slo-list">
            ${sloResults.map(r => html`
              <li key=${r.key} class=${'insp-slo insp-slo--' + r.kind}>
                <span class="insp-slo-key">${r.key.replace(/_/g, ' ')}</span>
                <span class="insp-slo-bar">
                  <span class="insp-slo-bar-thr"></span>
                  <span class="insp-slo-bar-fill" style=${'width: ' + (
                    r.probe == null ? 0 :
                    Math.max(6, Math.min(100, (Number(r.probe) / Number(r.threshold)) * 100))
                  ) + '%'}></span>
                </span>
                <span class="insp-slo-val">${r.probe != null ? fmtNumber(r.probe, 0) : '—'}<small>/${fmtNumber(r.threshold, 0)}</small></span>
              </li>
            `)}
          </ul>
        </section>
      `}

      ${phaseEntries.length > 0 && html`
        <section class="insp-block">
          <div class="insp-block-head">
            <span class="insp-block-dot"></span>
            PHASE STACK
          </div>
          <ol class="insp-phase-stack">
            ${phaseEntries.map(([pname, p]) => {
              const total = p.total_expected_requests ?? p.expected_requests ?? p.requests_total ?? null;
              const done  = p.final_requests_completed ?? p.requestsCompleted ?? p.requests_completed ?? p.completed ?? 0;
              const pct = total && total > 0 ? Math.min(100, (done / total) * 100) : (p.complete ? 100 : 0);
              const kind = p.complete ? 'done' : p.active ? 'active' : p.grace ? 'grace' : 'pending';
              return html`
                <li key=${pname} class=${'insp-phase insp-phase--' + kind}>
                  <div class="insp-phase-head">
                    <span class="insp-phase-name">${pname}</span>
                    <span class="insp-phase-pct">${fmtPercent(pct, 0)}</span>
                  </div>
                  <div class="insp-phase-track"><div class="insp-phase-fill" style=${'width: ' + pct + '%'}></div></div>
                  <div class="insp-phase-meta">${fmtInt(done)}${total ? ' / ' + fmtInt(total) : ''}</div>
                </li>
              `;
            })}
          </ol>
        </section>
      `}

      <section class="insp-block">
        <div class="insp-block-head">
          <span class="insp-block-dot"></span>
          TELEMETRY
        </div>
        <div class="insp-tele-grid">
          <div class="insp-tele">
            <span class="insp-tele-label">THROUGHPUT</span>
            <span class="insp-tele-val">${summary.throughput_rps != null ? fmtNumber(summary.throughput_rps, 1) : '—'}<small>r/s</small></span>
          </div>
          <div class="insp-tele">
            <span class="insp-tele-label">TTFT P99</span>
            <span class="insp-tele-val">${summary.ttft_p99_ms != null ? fmtInt(summary.ttft_p99_ms) : '—'}<small>ms</small></span>
          </div>
          <div class="insp-tele">
            <span class="insp-tele-label">LATENCY P99</span>
            <span class="insp-tele-val">${summary.latency_p99_ms != null ? fmtInt(summary.latency_p99_ms) : '—'}<small>ms</small></span>
          </div>
          <div class="insp-tele">
            <span class="insp-tele-label">ITL P99</span>
            <span class="insp-tele-val">${summary.itl_p99_ms != null ? fmtNumber(summary.itl_p99_ms, 1) : '—'}<small>ms</small></span>
          </div>
          <div class="insp-tele">
            <span class="insp-tele-label">REQS</span>
            <span class="insp-tele-val">${summary.total_requests != null ? fmtInt(summary.total_requests) : '—'}</span>
          </div>
          <div class="insp-tele">
            <span class="insp-tele-label">ERR %</span>
            <span class=${'insp-tele-val' + (summary.error_rate > 0 ? ' is-red' : '')}>
              ${summary.error_rate != null ? fmtNumber(summary.error_rate, 2) : '—'}<small>%</small>
            </span>
          </div>
        </div>
      </section>

      ${job?.error && html`
        <section class="insp-block insp-block--fault">
          <div class="insp-block-head">
            <span class="insp-block-dot insp-block-dot--red"></span>
            FAULT
          </div>
          <pre class="insp-fault">${job.error}</pre>
        </section>
      `}

      ${job?.workersTotal > 0 && html`
        <section class="insp-block">
          <div class="insp-block-head">
            <span class="insp-block-dot"></span>
            WORKERS
          </div>
          <div class="insp-workers">
            <span class="insp-workers-val">${job.workersReady ?? 0}<small>/ ${job.workersTotal}</small></span>
            <span class="insp-workers-label">READY</span>
          </div>
        </section>
      `}
    </div>
  `;
}

/* ──────────────────────── fleet inspector ───────────────────────── */

function FleetInspector() {
  const list = jobs.value ?? [];
  const ci = clusterInfo.value;

  const live = list.filter(j => phaseBucket(j.phase) === 'live');
  const fault = list.filter(j => phaseBucket(j.phase) === 'fault');
  const gpus = ci?.gpus ?? ci?.gpuCount ?? ci?.gpu_count ?? null;
  const gpuCap = ci?.gpuCapacity ?? ci?.gpu_capacity ?? null;
  const nodes = ci?.nodes ?? ci?.nodeCount ?? ci?.node_count ?? null;

  let sumRps = 0, rpsKnown = false, worstP99 = null;
  for (const j of live) {
    if (j.throughputRps != null) { sumRps += j.throughputRps; rpsKnown = true; }
    if (j.latencyP99Ms != null) worstP99 = worstP99 == null ? j.latencyP99Ms : Math.max(worstP99, j.latencyP99Ms);
  }

  const alerts = [];
  if (fault.length > 0) alerts.push({ kind: 'err', msg: `${fault.length} run${fault.length > 1 ? 's' : ''} in FAULT state` });
  if (worstP99 != null && worstP99 > 500) alerts.push({ kind: 'warn', msg: `P99 headroom exhausted (${fmtInt(worstP99)} ms)` });
  if (gpus != null && gpuCap && gpus / gpuCap > 0.85) alerts.push({ kind: 'warn', msg: `Cluster ≥ 85% utilised (${gpus}/${gpuCap} GPUs)` });
  if (alerts.length === 0) alerts.push({ kind: 'ok', msg: 'All subsystems nominal' });

  return html`
    <div class="insp-scroll">
      <div class="insp-hero">
        <div class="insp-hero-label">
          <span class=${'insp-hero-dot insp-hero-dot--' + (fault.length > 0 ? 'fault' : live.length > 0 ? 'live' : 'passed')}></span>
          FLEET
        </div>
        <div class="insp-hero-name">${live.length} LIVE</div>
        <div class="insp-hero-meta">
          <span>${fault.length} FAULT</span>
          <span>${list.length} TRACKED</span>
        </div>
      </div>

      <section class="insp-block">
        <div class="insp-block-head">
          <span class="insp-block-dot"></span>
          AGGREGATE
        </div>
        <div class="insp-tele-grid">
          <div class="insp-tele">
            <span class="insp-tele-label">FLEET R/S</span>
            <span class="insp-tele-val is-amber">${rpsKnown ? fmtNumber(sumRps, 0) : '—'}</span>
          </div>
          <div class="insp-tele">
            <span class="insp-tele-label">WORST P99</span>
            <span class=${'insp-tele-val' + (worstP99 != null && worstP99 > 500 ? ' is-red' : '')}>
              ${worstP99 != null ? fmtInt(worstP99) : '—'}<small>ms</small>
            </span>
          </div>
          <div class="insp-tele">
            <span class="insp-tele-label">GPUs</span>
            <span class="insp-tele-val">${gpus != null ? fmtInt(gpus) : '—'}${gpuCap ? html`<small>/${fmtInt(gpuCap)}</small>` : null}</span>
          </div>
          <div class="insp-tele">
            <span class="insp-tele-label">NODES</span>
            <span class="insp-tele-val">${nodes != null ? fmtInt(nodes) : '—'}</span>
          </div>
        </div>
      </section>

      <section class="insp-block">
        <div class="insp-block-head">
          <span class="insp-block-dot"></span>
          ALERTS
        </div>
        <ul class="insp-alert-list">
          ${alerts.map((a, i) => html`
            <li key=${i} class=${'insp-alert insp-alert--' + a.kind}>
              <span class="insp-alert-tag">${a.kind.toUpperCase()}</span>
              ${a.msg}
            </li>
          `)}
        </ul>
      </section>

      ${fault.length > 0 && html`
        <section class="insp-block insp-block--fault">
          <div class="insp-block-head">
            <span class="insp-block-dot insp-block-dot--red"></span>
            FAULT ROLL
          </div>
          <ul class="insp-fault-list">
            ${fault.slice(0, 5).map(j => html`
              <li key=${j.namespace + '/' + j.name}>
                <span class="insp-fault-name">${j.name}</span>
                <span class="insp-fault-ns">${j.namespace}</span>
                ${j.error && html`<span class="insp-fault-msg">${j.error}</span>`}
              </li>
            `)}
          </ul>
        </section>
      `}
    </div>
  `;
}

/* ─────────────────────────── main ────────────────────────── */

export function RightInspector({ viewKind, runParams, open, onToggle }) {
  return html`
    <aside
      class=${'insp' + (open ? '' : ' insp--collapsed')}
      aria-label="Inspector"
      data-testid="right-inspector"
      data-open=${open}
    >
      <div class="insp-head">
        <button
          class="insp-collapse"
          onclick=${onToggle}
          title=${open ? 'Collapse inspector (Ctrl+I)' : 'Expand inspector (Ctrl+I)'}
          aria-label=${open ? 'Collapse inspector' : 'Expand inspector'}
        >
          <i class=${'ph ' + (open ? 'ph-caret-double-right' : 'ph-caret-double-left')}></i>
        </button>
        ${open && html`
          <div class="insp-title">
            <span class="insp-title-caret">◂</span>
            ${viewKind === 'run' ? 'RUN INSPECTOR' : 'FLEET INSPECTOR'}
          </div>
        `}
      </div>
      ${open && (viewKind === 'run' && runParams
        ? html`<${RunInspector} ns=${runParams.ns} name=${runParams.name} open=${open} />`
        : html`<${FleetInspector} />`
      )}
    </aside>
  `;
}
