// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * RUN — main-viewport view when a specific run is pinned.
 *
 * Lives inside the persistent Flight Deck shell — the left rail still shows
 * every other run, the right inspector carries the per-run SLOs and phase
 * stack. So this view focuses on the headline + live meters + phase timeline
 * plus the richer diagnostic panes (conditions, pods, GPU telemetry).
 *
 * Layout, top to bottom:
 *   1. HEADER         — run name, phase tag, elapsed, phase ETA, CANCEL button
 *   2. CONDITIONS     — K8s condition badges (ConfigValid, WorkersReady, …)
 *   3. METER BAY      — R/S · TTFT · P99 · TOKEN/S · RELIABILITY
 *   4. PHASE TIMELINE — swimlane of warmup / main / cooldown
 *   5. PODS           — horizontal dots + names of worker pods
 *   6. GPU TELEMETRY  — one card per (endpoint, GPU idx), DCGM primaries + extras
 *   7. RESULTS        — downloadable artifacts (per-file + bundle zip)
 *   8. CONFIG         — collapsible <details> with JSON config
 */

import { html } from 'htm/preact';
import { useEffect, useState } from 'preact/hooks';
import { api, poll } from '../lib/api.js';
import { jobs } from '../lib/state.js';
import { navigate } from '../lib/router.js';
import { fmtBytes, fmtDuration, fmtInt, fmtNumber, fmtPercent } from '../lib/format.js';

function phaseBucket(phase) {
  const p = (phase ?? '').toLowerCase();
  if (p === 'running' || p === 'initializing' || p === 'pending') return 'live';
  if (p === 'failed' || p === 'error')                              return 'fault';
  if (p === 'completed' || p === 'succeeded')                       return 'passed';
  return 'other';
}

function pickActive(phases) {
  const arr = Object.entries(phases ?? {}).map(([name, p]) => ({ ...p, name }));
  const done = p => p.complete === true;
  const completed = p => p.final_requests_completed ?? p.requestsCompleted ?? p.requests_completed ?? p.completed ?? 0;
  const explicit = arr.filter(p => p.active && !done(p));
  if (explicit.length) return explicit.sort((a, b) => completed(b) - completed(a))[0];
  const inProgress = arr.filter(p => !done(p) && completed(p) > 0);
  if (inProgress.length) return inProgress.sort((a, b) => completed(b) - completed(a))[0];
  return arr.find(p => !done(p)) ?? null;
}

function phasePct(p) {
  if (!p) return 0;
  const total = p.total_expected_requests ?? p.expected_requests ?? p.requests_total ?? null;
  const done = p.final_requests_completed ?? p.requestsCompleted ?? p.requests_completed ?? p.completed ?? 0;
  if (total && total > 0) return Math.min(100, (done / total) * 100);
  if (p.complete) return 100;
  if (p.requestsProgressPercent != null) return Math.min(100, Math.max(0, Number(p.requestsProgressPercent)));
  return 0;
}

function phaseEta(p) {
  if (!p || !p.start_ns) return null;
  const total = p.total_expected_requests ?? p.expected_requests ?? p.requests_total ?? null;
  const done = p.final_requests_completed ?? p.requestsCompleted ?? p.requests_completed ?? p.completed ?? 0;
  if (!total || done <= 0) return null;
  const elapsed = (Date.now() - Number(p.start_ns) / 1e6) / 1000;
  const rate = elapsed > 0 ? done / elapsed : 0;
  return rate > 0 ? (total - done) / rate : null;
}

/* ────────────────────────── results pane ─────────────────────── */

/** File-extension → kind (drives the icon + color of the left-column chip). */
function fileKind(name) {
  const ext = name.toLowerCase().split('.').pop();
  if (['json', 'jsonl'].includes(ext))     return 'json';
  if (['csv', 'tsv'].includes(ext))        return 'csv';
  if (ext === 'parquet')                    return 'parquet';
  if (['yaml', 'yml'].includes(ext))        return 'yaml';
  if (['log', 'txt', 'ansi'].includes(ext)) return 'log';
  if (['html', 'htm'].includes(ext))        return 'html';
  if (['png', 'jpg', 'jpeg', 'svg', 'webp'].includes(ext)) return 'image';
  return 'bin';
}

/** A results pane — lists downloadable artifacts for a completed run, with a
 *  "Bundle ZIP" primary action. Polls once on mount and whenever the run key
 *  changes — the file list rarely mutates after a run finishes, so no
 *  continuous polling.
 */
function ResultsPane({ ns, name }) {
  const [state, setState] = useState({ kind: 'loading' });

  useEffect(() => {
    let cancel = false;
    setState({ kind: 'loading' });
    api.listJobFiles(ns, name)
      .then(r => { if (!cancel) setState({ kind: 'ok', files: r?.files ?? [] }); })
      .catch(err => {
        if (cancel) return;
        // 404 = no PVC directory for this run yet; anything else is a real error.
        if (/404/.test(err.message)) setState({ kind: 'none' });
        else setState({ kind: 'err', msg: err.message });
      });
    return () => { cancel = true; };
  }, [ns, name]);

  if (state.kind === 'loading') {
    return html`
      <section class="run-results" data-testid="run-results">
        <header class="slab-head slab-head--flush">
          <div class="slab-head-title"><span class="slab-head-caret">▸</span> RESULTS</div>
          <div class="slab-head-meta">SCANNING PVC…</div>
        </header>
      </section>
    `;
  }

  if (state.kind === 'err') {
    return html`
      <section class="run-results run-results--err" data-testid="run-results">
        <header class="slab-head slab-head--flush">
          <div class="slab-head-title"><span class="slab-head-caret">▸</span> RESULTS</div>
          <div class="slab-head-meta is-red">FETCH FAILED</div>
        </header>
        <div class="run-results-err">${state.msg}</div>
      </section>
    `;
  }

  if (state.kind === 'none') {
    return html`
      <section class="run-results run-results--empty" data-testid="run-results">
        <header class="slab-head slab-head--flush">
          <div class="slab-head-title"><span class="slab-head-caret">▸</span> RESULTS</div>
          <div class="slab-head-meta">NO ARTIFACTS YET</div>
        </header>
        <div class="run-results-empty">
          Files will appear here once the run completes and its output is
          archived to the operator PVC.
        </div>
      </section>
    `;
  }

  const files = state.files;
  const totalBytes = files.reduce((s, f) => s + (f.size_bytes ?? 0), 0);

  return html`
    <section class="run-results" data-testid="run-results">
      <header class="slab-head slab-head--flush">
        <div class="slab-head-title">
          <span class="slab-head-caret">▸</span> RESULTS
        </div>
        <div class="slab-head-meta">
          ${files.length} FILE${files.length === 1 ? '' : 'S'} · ${fmtBytes(totalBytes)}
        </div>
      </header>

      ${files.length === 0
        ? html`<div class="run-results-empty">Archive directory exists but contains no files yet.</div>`
        : html`
          <div class="run-results-actions">
            <a
              class="run-results-bundle"
              href=${api.resultBundleUrl(ns, name)}
              download
              data-testid="run-results-bundle"
            >
              <i class="ph ph-file-zip"></i>
              Download bundle
              <small>${fmtBytes(totalBytes)}</small>
            </a>
          </div>
          <ol class="run-results-list">
            ${files.map(f => html`
              <li
                key=${f.name}
                class="run-results-row"
                data-testid=${'run-results-row-' + f.name}
              >
                <span class=${'run-results-kind run-results-kind--' + fileKind(f.name)}>
                  ${fileKind(f.name).toUpperCase()}
                </span>
                <span class="run-results-name">${f.name}</span>
                <span class="run-results-size">${fmtBytes(f.size_bytes)}</span>
                ${f.compressed && html`<span class="run-results-zst" title="Stored zstd-compressed; decompressed on download">zst</span>`}
                <a
                  class="run-results-get"
                  href=${api.resultFileUrl(ns, name, f.name)}
                  download
                  aria-label=${'Download ' + f.name}
                >
                  <i class="ph ph-download-simple"></i>
                </a>
              </li>
            `)}
          </ol>
        `}
    </section>
  `;
}

/* ──────────────────── headline cancel button ─────────────────── */

function CancelButton({ ns, name, bucket, onCancel }) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  if (bucket !== 'live') return null;
  return html`
    <button
      class="run-cancel"
      disabled=${busy}
      onclick=${async () => {
        if (!confirm(`Cancel run "${name}"?`)) return;
        setBusy(true); setErr(null);
        try { await api.cancelJob(ns, name); onCancel?.(); }
        catch (e) { setErr(e.message); }
        finally { setBusy(false); }
      }}
      data-testid="run-cancel"
    >
      <i class="ph ph-x-circle"></i>
      ${busy ? 'CANCELLING…' : err ? 'RETRY' : 'CANCEL'}
    </button>
  `;
}

/* ─────────────────────── conditions strip ───────────────────── */

const CONDITION_LABELS = {
  ConfigValid:        'Config',
  EndpointReachable:  'Endpoint',
  PreflightPassed:    'Preflight',
  ResourcesCreated:   'Resources',
  WorkersReady:       'Workers',
  BenchmarkRunning:   'Benchmark',
  ResultsAvailable:   'Results',
};

function conditionKind(c) {
  const status = (c.status ?? '').toLowerCase();
  const reason = (c.reason ?? '').toLowerCase();
  if (status === 'true') return 'pass';
  if (reason.includes('progress') || reason.includes('waiting') || reason.includes('starting')) return 'progress';
  if (status === 'false') return 'fail';
  return 'idle';
}

function ConditionsStrip({ conditions }) {
  if (!conditions || conditions.length === 0) return null;
  return html`
    <section class="run-conditions" data-testid="run-conditions" aria-label="Conditions">
      ${conditions.map(c => {
        const label = CONDITION_LABELS[c.type] ?? c.type;
        const kind = conditionKind(c);
        const title = c.message ? `${c.type}: ${c.message}` : c.type;
        return html`
          <span
            key=${c.type}
            class=${'cond cond--' + kind}
            title=${title}
          >
            <span class="cond-dot"></span>
            ${label}
          </span>
        `;
      })}
    </section>
  `;
}

/* ─────────────────────────── pods bar ───────────────────────── */

function podKind(pod) {
  const phase = (pod.phase ?? '').toLowerCase();
  if (phase === 'failed' || phase === 'error') return 'fault';
  if (pod.ready) return 'ready';
  if (phase === 'running') return 'starting';
  return 'pending';
}

function truncPodName(name, max = 24) {
  if (!name) return '—';
  if (name.length <= max) return name;
  return '…' + name.slice(-(max - 1));
}

function PodsBar({ pods }) {
  if (!pods || pods.length === 0) return null;
  const ready = pods.filter(p => p.ready).length;
  const restarts = pods.reduce((s, p) => s + (p.restarts ?? 0), 0);
  return html`
    <section class="run-pods" data-testid="run-pods">
      <header class="slab-head slab-head--flush">
        <div class="slab-head-title">
          <span class="slab-head-caret">▸</span>
          WORKERS
        </div>
        <div class="slab-head-meta">
          <span class=${ready === pods.length ? 'is-green' : 'is-amber'}>${ready}/${pods.length} READY</span>
          ${restarts > 0 && html`<span class="is-amber"> · ${restarts} RESTART${restarts === 1 ? '' : 'S'}</span>`}
        </div>
      </header>
      <div class="run-pods-body">
        <div class="run-pods-dots" aria-label="Pod status dots">
          ${pods.map(p => html`
            <span
              key=${p.name}
              class=${'pod pod--' + podKind(p)}
              title=${`${p.name} (${p.phase ?? 'unknown'}${p.ready ? ', ready' : ''}${p.restarts ? `, ${p.restarts} restarts` : ''})`}
            ></span>
          `)}
        </div>
        <div class="run-pods-names">
          ${pods.map(p => html`
            <span key=${p.name} class=${'pod-name pod-name--' + podKind(p)} title=${p.name}>
              ${truncPodName(p.name)}
            </span>
          `)}
        </div>
      </div>
    </section>
  `;
}

/* ───────────────────────── GPU telemetry ────────────────────── */

const GPU_PRIMARY_TAGS = [
  { match: 'gpu_power_usage',  label: 'POWER' },
  { match: 'gpu_utilization',  label: 'UTIL' },
  { match: 'gpu_temperature',  label: 'TEMP' },
  { match: 'gpu_memory_used',  label: 'MEM' },
];

function parseGpuHeader(header) {
  if (!header || typeof header !== 'string') return null;
  const parts = header.split(' | ').map(s => s.trim());
  if (parts.length < 4) return null;
  const [metricName, endpoint, gpuText, ...modelParts] = parts;
  const m = /GPU\s+(\d+)/i.exec(gpuText);
  return {
    metricName,
    endpoint,
    gpuIndex: m ? parseInt(m[1], 10) : 0,
    model: modelParts.join(' | '),
  };
}

function gpuBaseName(tag) {
  if (!tag) return '';
  const cut = tag.indexOf('_dcgm_');
  return cut > 0 ? tag.slice(0, cut) : tag;
}

function groupGpuMetrics(metrics) {
  const groups = new Map();
  for (const r of metrics ?? []) {
    const info = parseGpuHeader(r.header);
    if (!info) continue;
    const key = `${info.endpoint}::${info.gpuIndex}`;
    if (!groups.has(key)) {
      groups.set(key, { endpoint: info.endpoint, gpuIndex: info.gpuIndex, model: info.model, metrics: [] });
    }
    groups.get(key).metrics.push({ ...r, baseName: gpuBaseName(r.tag), shortHeader: info.metricName });
  }
  return [...groups.values()].sort((a, b) =>
    a.endpoint.localeCompare(b.endpoint) || a.gpuIndex - b.gpuIndex);
}

function fmtGpuValue(metric) {
  const v = metric?.current ?? metric?.avg ?? null;
  if (v == null || typeof v !== 'number' || !isFinite(v)) return ['—', ''];
  const body = Math.abs(v) >= 1000 ? fmtInt(Math.round(v)) : fmtNumber(v, 1);
  return [body, metric.unit ?? ''];
}

function GpuTelemetry({ metrics }) {
  const gpus = groupGpuMetrics(metrics);
  if (gpus.length === 0) return null;
  return html`
    <section class="run-gpu" data-testid="run-gpu">
      <header class="slab-head slab-head--flush">
        <div class="slab-head-title">
          <span class="slab-head-caret">▸</span>
          GPU TELEMETRY
        </div>
        <div class="slab-head-meta">${gpus.length} GPU${gpus.length === 1 ? '' : 'S'}</div>
      </header>
      <div class="run-gpu-grid">
        ${gpus.map(gpu => {
          const primary = GPU_PRIMARY_TAGS.map(p => ({
            p, m: gpu.metrics.find(m => m.baseName === p.match || m.tag?.startsWith(p.match + '_')),
          }));
          const others = gpu.metrics.filter(m =>
            !GPU_PRIMARY_TAGS.some(p => m.baseName === p.match || m.tag?.startsWith(p.match + '_'))
          );
          return html`
            <article key=${gpu.endpoint + '::' + gpu.gpuIndex} class="run-gpu-card">
              <div class="run-gpu-head">
                <span class="run-gpu-idx">GPU ${gpu.gpuIndex}</span>
                <span class="run-gpu-endpoint">${gpu.endpoint}</span>
                ${gpu.model && html`<span class="run-gpu-model">${gpu.model}</span>`}
              </div>
              <div class="run-gpu-primary">
                ${primary.map(({ p, m }) => {
                  const [body, unit] = fmtGpuValue(m);
                  return html`
                    <div key=${p.match} class="run-gpu-tile">
                      <span class="run-gpu-tile-label">${p.label}</span>
                      <span class="run-gpu-tile-val">${body}${unit && html`<small> ${unit}</small>`}</span>
                    </div>
                  `;
                })}
              </div>
              ${others.length > 0 && html`
                <table class="run-gpu-extra">
                  <tbody>
                    ${others.map(m => {
                      const [body, unit] = fmtGpuValue(m);
                      return html`
                        <tr key=${m.tag}>
                          <td>${m.shortHeader ?? m.baseName}</td>
                          <td>${body}${unit ? ' ' + unit : ''}</td>
                        </tr>
                      `;
                    })}
                  </tbody>
                </table>
              `}
            </article>
          `;
        })}
      </div>
    </section>
  `;
}

/* ─────────────────────── reliability meter ──────────────────── */

/** Replacement for the vanilla REQUESTS meter. When SLOs are declared and
 *  `goodput_count` is on the summary, reports *SLO violations* (total − goodput).
 *  Otherwise falls back to total requests + error rate, which the old
 *  `ReliabilityTile` on the legacy Job Detail page showed. */
function ReliabilityMeter({ summary, slosDeclared }) {
  if (!summary || typeof summary !== 'object') {
    return html`<${RunMeter} label="REQUESTS" value="—" tone="dim" />`;
  }
  const total = summary.total_requests;
  const goodput = summary.goodput_count;
  const errorCount = summary.error_count
    ?? (total != null && summary.error_rate != null
        ? Math.round((summary.error_rate / 100) * total)
        : null);

  if (slosDeclared && total != null && goodput != null) {
    const failed = Math.max(0, Math.round(total - goodput));
    const pct = total > 0 ? Math.max(0, 100 - (failed / total) * 100) : null;
    return html`<${RunMeter}
      label="GOODPUT"
      value=${fmtInt(failed)}
      unit=${pct != null ? fmtNumber(pct, 1) + '% pass' : 'failed'}
      tone=${failed === 0 ? 'green' : 'amber'}
    />`;
  }

  if (total != null) {
    const errUnit = errorCount != null
      ? errorCount === 0 ? 'no errors' : `${fmtInt(errorCount)} err`
      : 'no errors';
    return html`<${RunMeter}
      label="REQUESTS"
      value=${fmtInt(total)}
      unit=${errUnit}
      tone=${(errorCount ?? 0) > 0 ? 'red' : 'green'}
    />`;
  }

  return html`<${RunMeter} label="REQUESTS" value="—" tone="dim" />`;
}

/* ──────────────────────── the view ────────────────────────── */

export function Run({ ns, name }) {
  const [detail, setDetail] = useState(null);
  const [config, setConfig] = useState(null);

  useEffect(() => {
    setDetail(null); setConfig(null);
    const ac = new AbortController();
    poll(async () => {
      try {
        const [d, c] = await Promise.all([
          api.getJob(ns, name).catch(() => null),
          api.getJobConfig(ns, name).catch(() => null),
        ]);
        setDetail(d); setConfig(c);
      } catch (_e) { /* transient */ }
    }, 4000, ac.signal);
    return () => ac.abort();
  }, [ns, name]);

  const job = (jobs.value ?? []).find(j => j.namespace === ns && j.name === name) ?? null;
  const status = detail?.status;
  const pods = detail?.pods ?? [];
  const conditions = status?.conditions ?? [];
  const gpuMetrics = status?.metrics ?? status?.liveMetrics ?? [];
  const slos = config?.spec?.benchmark?.slos ?? config?.spec?.slos ?? null;
  const slosDeclared = !!(slos && typeof slos === 'object' && Object.keys(slos).length > 0);
  const summary = status?.liveSummary ?? status?.summary ?? {};
  const phases = status?.phases ?? {};
  const phaseEntries = Object.entries(phases);
  const active = pickActive(phases);
  const bucket = phaseBucket(job?.phase);
  const elapsed = job?.startTime ? (Date.now() - new Date(job.startTime).getTime()) / 1000 : null;
  const eta = phaseEta(active);
  const rps = summary.throughput_rps;
  const ttft = summary.ttft_p99_ms ?? summary.ttft_avg_ms;
  const p99 = summary.latency_p99_ms ?? summary.latency_avg_ms;
  const tokps = summary.output_token_throughput ?? job?.tokenThroughput;
  const totalReq = summary.total_requests;
  const errRate = summary.error_rate ?? 0;  if (!job && !detail) {
    return html`
      <div class="v-run v-run--loading" data-testid="page-job-detail">
        <div class="run-404">
          <div class="run-404-glyph"><span class="ph ph-magnifying-glass"></span></div>
          <div class="run-404-title">LOCATING ${name.toUpperCase()}</div>
          <div class="run-404-meta">namespace ${ns}</div>
        </div>
      </div>
    `;
  }

  return html`
    <div class=${'v-run v-run--' + bucket} data-testid="page-job-detail">
      <!-- 1. HEADER -->
      <header class="run-header">
        <div class="run-header-title">
          <button class="run-header-back" onclick=${() => navigate('/overview')} title="Back to overview" aria-label="Back">
            <i class="ph ph-arrow-left"></i>
          </button>
          <div>
            <div class="run-header-eyebrow">
              <span class=${'run-header-phase run-header-phase--' + bucket}>
                ${(job?.phase ?? 'UNKNOWN').toUpperCase()}
              </span>
              <span class="run-header-ns">${ns}</span>
              ${job?.model && html`<span class="run-header-model">${job.model}</span>`}
            </div>
            <h1 class="run-header-name">${name}</h1>
            <div class="run-header-sub">
              ${job?.endpoint && html`<span><i class="ph ph-globe"></i> ${job.endpoint}</span>`}
              ${job?.concurrency != null && html`<span>CONC ${job.concurrency}</span>`}
              ${job?.backend && html`<span>${job.backend}</span>`}
              ${job?.gpuConfig && html`<span>${job.gpuConfig}</span>`}
            </div>
          </div>
        </div>
        <div class="run-header-clocks">
          <div class="run-clock">
            <span class="run-clock-label">ELAPSED</span>
            <span class="run-clock-val">${elapsed != null ? fmtDuration(elapsed) : '—'}</span>
          </div>
          <div class="run-clock">
            <span class="run-clock-label">PHASE ETA</span>
            <span class=${'run-clock-val' + (eta != null ? '' : ' is-dim')}>${eta != null ? fmtDuration(eta) : '—'}</span>
          </div>
          <${CancelButton} ns=${ns} name=${name} bucket=${bucket} />
        </div>
      </header>

      <!-- 2. CONDITIONS -->
      <${ConditionsStrip} conditions=${conditions} />

      <!-- 3. METER BAY -->
      <section class="run-meters">
        <${RunMeter} label="THROUGHPUT" value=${rps != null ? fmtNumber(rps, 1) : '—'} unit="req/s" tone=${rps != null ? 'amber' : 'dim'} />
        <${RunMeter} label="TTFT P99"   value=${ttft != null ? fmtInt(ttft) : '—'}     unit="ms"    tone="paper" />
        <${RunMeter} label="LATENCY P99" value=${p99 != null ? fmtInt(p99) : '—'}      unit="ms"    tone=${p99 != null && p99 > 500 ? 'red' : 'paper'} />
        <${RunMeter} label="TOKEN/S"    value=${tokps != null ? fmtInt(tokps) : '—'}   unit="tok/s" tone=${tokps != null ? 'amber' : 'dim'} />
        <${ReliabilityMeter} summary=${summary} slosDeclared=${slosDeclared} />
      </section>

      <!-- 3. PHASES SWIMLANE -->
      ${phaseEntries.length > 0 && html`
        <section class="run-phases">
          <header class="slab-head slab-head--flush">
            <div class="slab-head-title">
              <span class="slab-head-caret">▸</span>
              PHASE TIMELINE
            </div>
            <div class="slab-head-meta">
              ${phaseEntries.filter(([, p]) => p.complete).length} / ${phaseEntries.length} COMPLETE
            </div>
          </header>
          <div class="run-phase-lanes">
            ${phaseEntries.map(([pname, p]) => {
              const pct = phasePct(p);
              const kind = p.complete ? 'done' : p.active ? 'active' : p.grace ? 'grace' : 'pending';
              const total = p.total_expected_requests ?? p.expected_requests ?? p.requests_total ?? null;
              const done = p.final_requests_completed ?? p.requestsCompleted ?? p.requests_completed ?? p.completed ?? 0;
              return html`
                <div key=${pname} class=${'run-lane run-lane--' + kind}>
                  <div class="run-lane-label">
                    <span class="run-lane-status"></span>
                    ${pname.toUpperCase()}
                  </div>
                  <div class="run-lane-track">
                    <div class="run-lane-fill" style=${'width: ' + pct + '%'}></div>
                    <div class="run-lane-meta">
                      <span>${fmtInt(done)}${total ? ' / ' + fmtInt(total) : ''}</span>
                      <span>${fmtPercent(pct, 0)}</span>
                    </div>
                  </div>
                </div>
              `;
            })}
          </div>
        </section>
      `}

      <!-- 5. PODS -->
      <${PodsBar} pods=${pods} />

      <!-- 6. GPU TELEMETRY -->
      <${GpuTelemetry} metrics=${gpuMetrics} />

      <!-- 7. RESULTS -->
      <${ResultsPane} ns=${ns} name=${name} />

      <!-- 8. CONFIG (collapsed by default) -->
      ${config && html`
        <details class="run-config">
          <summary>
            <span class="slab-head-caret">▸</span>
            CONFIG · ${config.source ?? 'spec'}
            <span class="run-config-hint">expand</span>
          </summary>
          <pre class="run-config-body">${JSON.stringify(config.spec ?? config, null, 2)}</pre>
        </details>
      `}
    </div>
  `;
}

function RunMeter({ label, value, unit, tone }) {
  return html`
    <div class=${'run-meter run-meter--' + (tone ?? 'paper')}>
      <div class="run-meter-label">${label}</div>
      <div class="run-meter-value">${value}${unit && html`<span class="run-meter-unit">${unit}</span>`}</div>
    </div>
  `;
}
