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
import { useEffect, useRef, useState } from 'preact/hooks';
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
 *
 *  When ``epoch`` is non-null, reads files from the pinned historical run
 *  dir via ``/api/v1/results/<ns>/<name>/runs/<epoch>``; download URLs
 *  follow the same pattern so historical bundles stay addressable.
 */
function ResultsPane({ ns, name, epoch }) {
  const [state, setState] = useState({ kind: 'loading' });

  useEffect(() => {
    let cancel = false;
    setState({ kind: 'loading' });
    api.listJobFiles(ns, name, epoch)
      .then(r => { if (!cancel) setState({ kind: 'ok', files: r?.files ?? [] }); })
      .catch(err => {
        if (cancel) return;
        // 404 = no PVC directory for this run yet; anything else is a real error.
        if (/404/.test(err.message)) setState({ kind: 'none' });
        else setState({ kind: 'err', msg: err.message });
      });
    return () => { cancel = true; };
  }, [ns, name, epoch]);

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
              href=${api.resultBundleUrl(ns, name, epoch)}
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
                  href=${api.resultFileUrl(ns, name, f.name, epoch)}
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

/* ──────────────────── run-history dropdown ─────────────────── */

/** Format an epoch-seconds string as ``YYYY-MM-DD HH:MM:SS UTC`` for option
 *  labels. Falls back to the raw epoch string when parsing fails so the user
 *  never sees a silent "Invalid Date". */
function fmtRunLabel(entry, isLatest) {
  const seconds = Number(entry.mtime_epoch ?? entry.epoch);
  let label = entry.epoch;
  if (Number.isFinite(seconds) && seconds > 0) {
    const d = new Date(seconds * 1000);
    if (!isNaN(d.getTime())) {
      const pad = n => String(n).padStart(2, '0');
      label = `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} `
            + `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())} UTC`;
    }
  }
  return isLatest ? `${label} (latest)` : label;
}

/** Dropdown that pins the RESULTS pane to a historical run dir.
 *
 *  Fetches ``/api/v1/results/<ns>/<name>/runs`` on mount; renders a
 *  ``<select class="run-history-select">`` when at least one run exists.
 *  Value ``latest`` stays on ``latest.txt`` (stable URLs); any other value
 *  is an epoch string and routes to ``#/run/<ns>/<name>/runs/<epoch>``.
 */
function RunHistoryDropdown({ ns, name, selectedEpoch }) {
  const [state, setState] = useState({ kind: 'loading' });

  useEffect(() => {
    let cancel = false;
    setState({ kind: 'loading' });
    api.listRuns(ns, name)
      .then(({ runs, latestEpoch }) => {
        if (cancel) return;
        setState({ kind: 'ok', runs, latestEpoch });
      })
      .catch(err => {
        if (!cancel) setState({ kind: 'err', msg: err.message });
      });
    return () => { cancel = true; };
  }, [ns, name]);

  if (state.kind !== 'ok') return null;
  const { runs } = state;
  if (runs.length === 0) return null;

  const value = selectedEpoch ?? 'latest';
  return html`
    <label class="run-history" data-testid="run-history">
      <span class="run-history-label">RUN</span>
      <select
        class="run-history-select"
        value=${value}
        onchange=${(e) => {
          const v = e.target.value;
          if (v === 'latest') navigate(`/run/${encodeURIComponent(ns)}/${encodeURIComponent(name)}`);
          else navigate(`/run/${encodeURIComponent(ns)}/${encodeURIComponent(name)}/runs/${encodeURIComponent(v)}`);
        }}
        data-testid="run-history-select"
      >
        <option value="latest">Latest run</option>
        ${runs.map(r => html`
          <option key=${r.epoch} value=${r.epoch}>${fmtRunLabel(r, r.is_latest)}</option>
        `)}
      </select>
    </label>
  `;
}

/** Dropdown that jumps to the run-diff compare view. Populated from the same
 *  ``listRuns`` endpoint as ``RunHistoryDropdown``, with the currently-pinned
 *  run excluded (diffing a run against itself is meaningless). The default
 *  "Compare with…" option is a placeholder; picking any epoch navigates to
 *  ``#/compare/<ns>/<name>/<currentEpoch>/<selectedEpoch>``.
 *
 *  When ``selectedEpoch`` is null (viewing "latest"), the dropdown uses the
 *  reported ``latestEpoch`` as the A-side of the compare URL so the compare
 *  view always has two concrete epoch anchors — the #/compare route shape
 *  requires both. If no ``latestEpoch`` is available (edge case: only one
 *  run exists and ``listRuns`` hasn't reported an epoch for it), the
 *  dropdown hides itself rather than link to an unresolvable URL.
 */
function CompareWithDropdown({ ns, name, selectedEpoch }) {
  const [state, setState] = useState({ kind: 'loading' });

  useEffect(() => {
    let cancel = false;
    setState({ kind: 'loading' });
    api.listRuns(ns, name)
      .then(({ runs, latestEpoch }) => {
        if (cancel) return;
        setState({ kind: 'ok', runs, latestEpoch });
      })
      .catch(err => {
        if (!cancel) setState({ kind: 'err', msg: err.message });
      });
    return () => { cancel = true; };
  }, [ns, name]);

  if (state.kind !== 'ok') return null;
  const { runs, latestEpoch } = state;
  const currentEpoch = selectedEpoch ?? latestEpoch;
  if (!currentEpoch || runs.length < 2) return null;

  const others = runs.filter(r => r.epoch !== currentEpoch);
  if (others.length === 0) return null;

  return html`
    <label class="run-compare" data-testid="run-compare">
      <span class="run-compare-label">COMPARE</span>
      <select
        class="run-compare-select"
        value=""
        onchange=${(e) => {
          const v = e.target.value;
          if (!v) return;
          const nsSeg = encodeURIComponent(ns);
          const nameSeg = encodeURIComponent(name);
          const a = encodeURIComponent(currentEpoch);
          const b = encodeURIComponent(v);
          window.location.hash = `#/compare/${nsSeg}/${nameSeg}/${a}/${b}`;
        }}
        data-testid="run-compare-select"
      >
        <option value="">Compare with…</option>
        ${others.map(r => html`
          <option key=${r.epoch} value=${r.epoch}>${fmtRunLabel(r, r.is_latest)}</option>
        `)}
      </select>
    </label>
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

/* ──────────────────── re-launch button ─────────────────── */

/** Minimal YAML serializer — AIPerfJob specs only. Handles strings, numbers,
 *  bools, null, lists, objects. Quotes strings that contain YAML-significant
 *  characters. Not a full emitter. */
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
        // first line gets the dash, subsequent lines stay indented by 2
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

function suggestRetryName(orig) {
  if (!orig) return 'run-retry';
  const d = new Date();
  const pad = n => String(n).padStart(2, '0');
  const stamp = `${String(d.getFullYear()).slice(2)}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}`;
  // Strip any prior -retry-YYMMDD-HHMM suffix so repeat relaunches don't stack.
  const base = orig.replace(/-retry-\d{6}-\d{4}$/, '');
  return `${base}-retry-${stamp}`;
}

function RelaunchButton({ ns, name, config }) {
  const spec = config?.spec;
  if (!spec || Object.keys(spec).length === 0) return null;
  return html`
    <button
      class="run-relaunch"
      onclick=${() => {
        const manifest = {
          apiVersion: config.apiVersion ?? 'aiperf.nvidia.com/v1alpha1',
          kind: config.kind ?? 'AIPerfJob',
          metadata: {
            name: suggestRetryName(name),
            namespace: ns,
          },
          spec,
        };
        const yaml = serializeYaml(manifest) + '\n';
        try {
          sessionStorage.setItem('aiperf.launch.prefill', JSON.stringify({
            yaml,
            sourceNs: ns,
            sourceName: name,
            at: Date.now(),
          }));
        } catch (_e) { /* quota/private-mode — fall through to navigate */ }
        navigate('/launch');
      }}
      data-testid="run-relaunch"
      title="Copy this run's config into the Launch editor"
    >
      <i class="ph ph-arrow-counter-clockwise"></i>
      RE-LAUNCH
    </button>
  `;
}

/* ─────────────────────── live logs pane ────────────────────── */

const LOGS_MAX_LINES = 2000;

function LogsPane({ ns, name, pods }) {
  const podList = (pods ?? []).filter(p => p?.name);
  const [selectedPod, setSelectedPod] = useState(null);
  const [tailLines, setTailLines] = useState(200);
  const [follow, setFollow] = useState(true);
  const [tail, setTail] = useState([]);
  const [err, setErr] = useState(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const bufRef = useRef([]);
  const bodyRef = useRef(null);
  const autoScrollRef = useRef(true);

  // Auto-select first pod; re-align when pod list changes.
  useEffect(() => {
    if (podList.length === 0) { setSelectedPod(null); return; }
    if (!selectedPod || !podList.find(p => p.name === selectedPod)) {
      const pod = podList[0];
      setSelectedPod(pod.name);
      // default follow=ON iff pod is Running
      setFollow((pod.phase ?? '').toLowerCase() === 'running');
    }
  }, [podList.map(p => p.name).join('|')]);

  useEffect(() => { autoScrollRef.current = autoScroll; }, [autoScroll]);

  // Stream lifecycle: reset buffer + (re)open on any dep change.
  useEffect(() => {
    if (!selectedPod) return;
    bufRef.current = [];
    setTail([]);
    setErr(null);
    setAutoScroll(true);
    autoScrollRef.current = true;

    const ac = new AbortController();
    const clampedTail = Math.max(1, Math.min(5000, Number(tailLines) || 200));

    const appendText = (text) => {
      if (!text) return;
      const lines = text.split('\n');
      // trailing empty string from split('\n') drops a pure-newline chunk's tail
      if (lines.length && lines[lines.length - 1] === '') lines.pop();
      if (lines.length === 0) return;
      const next = bufRef.current.concat(lines);
      const overflow = next.length - LOGS_MAX_LINES;
      bufRef.current = overflow > 0 ? next.slice(overflow) : next;
      setTail(bufRef.current.slice());
    };

    (async () => {
      try {
        if (follow) {
          const res = await api.getJobLogs(ns, name, {
            pod: selectedPod, follow: true, tailLines: clampedTail, signal: ac.signal,
          });
          const reader = res.body?.getReader();
          if (!reader) {
            const text = await res.text();
            appendText(text);
            return;
          }
          const decoder = new TextDecoder();
          let leftover = '';
          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            const chunk = leftover + decoder.decode(value, { stream: true });
            const lastNl = chunk.lastIndexOf('\n');
            if (lastNl === -1) { leftover = chunk; continue; }
            appendText(chunk.slice(0, lastNl + 1));
            leftover = chunk.slice(lastNl + 1);
          }
          if (leftover) appendText(leftover + '\n');
        } else {
          const text = await api.getJobLogs(ns, name, {
            pod: selectedPod, follow: false, tailLines: clampedTail, signal: ac.signal,
          });
          appendText(text);
        }
      } catch (e) {
        if (ac.signal.aborted) return;
        if (/\b404\b/.test(e.message)) setErr('Pod not found (it may have been evicted).');
        else setErr(e.message);
      }
    })();

    return () => ac.abort();
  }, [ns, name, selectedPod, follow, tailLines]);

  // Auto-scroll to bottom on new data, unless user scrolled up.
  useEffect(() => {
    const el = bodyRef.current;
    if (!el) return;
    if (autoScrollRef.current) el.scrollTop = el.scrollHeight;
  }, [tail]);

  const onScroll = () => {
    const el = bodyRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.clientHeight - el.scrollTop <= 20;
    if (atBottom && !autoScrollRef.current) setAutoScroll(true);
    else if (!atBottom && autoScrollRef.current) setAutoScroll(false);
  };

  const jumpToLatest = () => {
    const el = bodyRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    setAutoScroll(true);
  };

  const header = (meta) => html`
    <header class="slab-head slab-head--flush">
      <div class="slab-head-title"><span class="slab-head-caret">▸</span> LOGS</div>
      <div class="slab-head-meta">${meta}</div>
    </header>
  `;

  if (podList.length === 0) {
    return html`
      <section class="run-logs" id="run-logs" data-testid="run-logs">
        ${header('NO PODS YET')}
        <div class="run-logs-empty">No pods yet — logs will appear here once workers are scheduled.</div>
      </section>
    `;
  }

  return html`
    <section class="run-logs" id="run-logs" data-testid="run-logs">
      ${header(`${tail.length} LINE${tail.length === 1 ? '' : 'S'}${follow ? ' · LIVE' : ''}`)}
      <div class="run-logs-controls">
        <select
          class="run-logs-pod"
          value=${selectedPod ?? ''}
          onchange=${e => setSelectedPod(e.target.value)}
          data-testid="run-logs-pod"
        >
          ${podList.map(p => html`
            <option key=${p.name} value=${p.name}>
              ${truncPodName(p.name, 40)} · ${(p.phase ?? 'unknown').toLowerCase()}
            </option>
          `)}
        </select>
        <button
          class=${'run-logs-follow' + (follow ? ' is-active' : '')}
          onclick=${() => setFollow(f => !f)}
          data-testid="run-logs-follow"
          title=${follow ? 'Pause streaming (static tail)' : 'Resume live follow'}
        >
          <i class=${'ph ' + (follow ? 'ph-pause' : 'ph-play')}></i>
          ${follow ? 'FOLLOW' : 'PAUSED'}
        </button>
        <label class="run-logs-tail-lbl">
          TAIL
          <input
            class="run-logs-tail"
            type="number"
            min="1"
            max="5000"
            value=${tailLines}
            onchange=${e => {
              const v = Math.max(1, Math.min(5000, parseInt(e.target.value, 10) || 200));
              setTailLines(v);
            }}
            data-testid="run-logs-tail"
          />
        </label>
      </div>
      <pre class="run-logs-body" ref=${bodyRef} onscroll=${onScroll} data-testid="run-logs-body">${tail.join('\n')}${err && html`<div class="run-logs-error">${err}</div>`}</pre>
      ${!autoScroll && html`
        <button class="run-logs-jump" onclick=${jumpToLatest} data-testid="run-logs-jump">
          <i class="ph ph-arrow-down"></i>
          JUMP TO LATEST
        </button>
      `}
    </section>
  `;
}

/* ─────────────────────── identity strip ───────────────────── */

/** Dense grid of "who is this run" tiles. Renders only tiles whose value is
 *  known; unknowns are omitted rather than shown as "—", so the strip stays
 *  legible for partial metadata. MODEL and ENDPOINT span two columns because
 *  their values are long. */
/** Count of other runs in the same (namespace, model) cluster. Comparability
 *  is count-only — we never aggregate metrics across independent benchmarks. */
function siblingCount(job) {
  if (!job || !job.model || !job.namespace) return 0;
  const all = jobs.value ?? [];
  let n = 0;
  for (const r of all) {
    if (r.namespace === job.namespace && r.model === job.model && r.name !== job.name) n++;
  }
  return n;
}

function IdentityStrip({ job, config, summary: _summary }) {
  const spec = config?.spec ?? {};
  const bench = spec.benchmark ?? {};
  const input = spec.input ?? {};
  const synth = input.synthetic_tokens ?? {};

  const model = job?.model ?? spec.model ?? null;
  const endpoint = job?.endpoint ?? spec.endpoint ?? null;
  const backend = job?.backend ?? spec.backend ?? null;
  const mode = bench.mode ?? spec.mode ?? null;
  const concurrency = job?.concurrency ?? bench.concurrency ?? null;
  const isl = synth.input_length ?? null;
  const osl = synth.output_length ?? null;
  const requests = bench.request_count ?? bench.number_of_requests ?? null;
  const durationRaw = bench.duration_secs ?? bench.duration ?? null;
  const gpus = job?.gpuConfig ?? null;

  const tiles = [];
  if (model != null)     tiles.push({ label: 'MODEL',          value: model });
  if (endpoint != null)  tiles.push({ label: 'ENDPOINT',       value: endpoint });
  if (backend != null)   tiles.push({ label: 'BACKEND',        value: backend });
  if (mode != null)      tiles.push({ label: 'BENCHMARK MODE', value: String(mode) });
  if (concurrency != null) tiles.push({ label: 'CONCURRENCY',  value: fmtInt(concurrency) });
  if (isl != null || osl != null) {
    tiles.push({
      label: 'ISL / OSL',
      value: `${isl ?? '—'} / ${osl ?? '—'}`,
    });
  }
  if (requests != null)  tiles.push({ label: 'REQUESTS',       value: fmtInt(requests) });
  if (durationRaw != null) {
    const secs = typeof durationRaw === 'number' ? durationRaw : Number(durationRaw);
    if (isFinite(secs)) tiles.push({ label: 'DURATION', value: fmtDuration(secs) });
  }
  if (gpus != null)      tiles.push({ label: 'GPUs',           value: gpus });

  if (tiles.length === 0) return null;

  const sibN = model != null ? siblingCount(job) : -1;
  const clusterKey = model != null ? `${job?.namespace ?? ''} · ${model}` : null;

  // The two data-testid="run-identity-sibling" rendering paths below mirror
  // the legacy sibling logic — exactly one of (link / empty placeholder)
  // is rendered per call. Tests inspect the testid, not its count > 1.
  return html`
    <section class="run-identity" data-testid="run-identity" aria-label="Run identity">
      ${tiles.map((t, i) => html`
        ${i > 0 && html`<span key=${'sep-' + t.label} class="run-identity-sep"></span>`}
        <div key=${t.label} class="run-identity-item">
          <span class="run-identity-label">${t.label}</span>
          <span class="run-identity-value">${t.value}</span>
        </div>
      `)}
      ${sibN > 0 && html`
        <span key="sib-sep" class="run-identity-sep"></span>
        <a
          key="siblings"
          class="run-identity-item run-identity-item--sibling"
          href=${'/compare?cluster=' + encodeURIComponent(clusterKey)}
          onclick=${(e) => { e.preventDefault(); navigate('/compare?cluster=' + encodeURIComponent(clusterKey)); }}
          data-testid="run-identity-sibling"
          title=${`Compare runs in ${clusterKey}`}
        >
          <span class="run-identity-label">${sibN} COMPARABLE RUN${sibN === 1 ? '' : 'S'}</span>
          <span class="run-identity-value">
            ${clusterKey}
            <i class="ph ph-arrow-right run-identity-sibling-arrow"></i>
          </span>
        </a>
      `}
      ${sibN === 0 && html`
        <span key="sib-sep" class="run-identity-sep"></span>
        <div
          key="siblings"
          class="run-identity-item run-identity-item--sibling-empty"
          data-testid="run-identity-sibling"
          aria-disabled="true"
        >
          <span class="run-identity-label">NO COMPARABLE RUNS</span>
          <span class="run-identity-value">${clusterKey}</span>
        </div>
      `}
    </section>
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

/** Per-type wording when a condition is negative (status === 'False').
 *  Without this flip, the strip/callout shows the *type* label ("Succeeded",
 *  "Ready") next to a red swatch — misleading because the label reads
 *  positive even though the condition failed. */
const CONDITION_FALSE_LABELS = {
  ConfigValid:        'Config invalid',
  EndpointReachable:  'Endpoint unreachable',
  PreflightPassed:    'Preflight failed',
  ResourcesCreated:   'Resources missing',
  WorkersReady:       'Workers not ready',
  BenchmarkRunning:   'Not running',
  ResultsAvailable:   'No results',
  Succeeded:          'Failed',
  Ready:              'Not ready',
};

function conditionKind(c) {
  const status = (c.status ?? '').toLowerCase();
  const reason = (c.reason ?? '').toLowerCase();
  if (status === 'true') return 'pass';
  if (reason.includes('progress') || reason.includes('waiting') || reason.includes('starting')) return 'progress';
  if (status === 'false') return 'fail';
  return 'idle';
}

function conditionLabel(c) {
  const status = (c.status ?? '').toLowerCase();
  if (status === 'false') {
    return CONDITION_FALSE_LABELS[c.type]
      ?? ('Not ' + (CONDITION_LABELS[c.type] ?? c.type).toLowerCase());
  }
  return CONDITION_LABELS[c.type] ?? c.type;
}

function ConditionsStrip({ conditions }) {
  if (!conditions || conditions.length === 0) return null;
  return html`
    <section class="run-conditions" data-testid="run-conditions" aria-label="Conditions">
      ${conditions.map(c => {
        const label = conditionLabel(c);
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
    return html`<${RunKpi} label="REQUESTS" value="—" unit="" tone="" />`;
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
    return html`<${RunKpi}
      label="GOODPUT"
      primary=${pct != null ? fmtNumber(pct, 1) + '% pass' : 'failed'}
      value=${fmtInt(failed)}
      unit="failed"
      tone=${failed === 0 ? 'good' : 'warn'}
    />`;
  }

  if (total != null) {
    const errUnit = errorCount != null
      ? errorCount === 0 ? 'no errors' : `${fmtInt(errorCount)} err`
      : 'no errors';
    return html`<${RunKpi}
      label="REQUESTS"
      primary=${errUnit}
      value=${fmtInt(total)}
      unit=""
      tone=${(errorCount ?? 0) > 0 ? 'bad' : 'good'}
    />`;
  }

  return html`<${RunKpi} label="REQUESTS" value="—" unit="" tone="" />`;
}

/* ─────────────────────── live sparklines ──────────────────── */

const MAX_SAMPLES = 60;  // ~4 min at 4 s/sample

const SPARK_SPECS = [
  { key: 'rps',   label: 'THROUGHPUT', unit: 'r/s',  color: 'var(--amber)',  digits: 1 },
  { key: 'p99',   label: 'LATENCY P99', unit: 'ms',  color: 'var(--cyan)',   digits: 0 },
  { key: 'ttft',  label: 'TTFT P99',    unit: 'ms',  color: 'var(--paper)',  digits: 0 },
  { key: 'tokps', label: 'TOKEN/S',     unit: 'tok/s', color: 'var(--green)', digits: 0 },
];

function cssVar(v) {
  if (typeof v !== 'string' || !v.startsWith('var(')) return v;
  const name = v.slice(4, -1).trim();
  if (typeof window === 'undefined') return '#76b900';
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#76b900';
}

function sparkPath(pts) {
  if (!pts || pts.length < 2) return '';
  const ys = pts.map(p => p.y);
  const min = Math.min(...ys);
  const max = Math.max(...ys);
  const range = max - min || 1;
  const W = 160, H = 32;
  const stepX = pts.length > 1 ? W / (pts.length - 1) : 0;
  return pts.map((p, i) => {
    const x = (i * stepX).toFixed(2);
    const y = (H - ((p.y - min) / range) * H).toFixed(2);
    return `${x},${y}`;
  }).join(' ');
}

function SparkTile({ spec, samples }) {
  const pts = samples
    .map((s, i) => ({ x: i, y: s[spec.key] }))
    .filter(p => p.y != null && isFinite(p.y));
  const last = pts.length > 0 ? pts[pts.length - 1].y : null;
  const color = cssVar(spec.color);
  const path = sparkPath(pts);

  return html`
    <div class="run-spark-tile">
      <span class="run-spark-tile-label">${spec.label}</span>
      <span class="run-spark-tile-value" style=${'color: ' + color}>
        ${last != null ? fmtNumber(last, spec.digits) : '—'}
        <small> ${spec.unit}</small>
      </span>
      ${pts.length < 2
        ? html`<svg class="run-spark-tile-svg" viewBox="0 0 160 32" preserveAspectRatio="none"></svg>`
        : html`
          <svg class="run-spark-tile-svg" viewBox="0 0 160 32" preserveAspectRatio="none">
            <polyline fill="none" stroke=${color} stroke-width="1.4" points=${path}/>
          </svg>
        `}
    </div>
  `;
}

/* ─────────────────── latency-over-time chart ─────────────────── */

const LATENCY_CHART_MAX_POINTS = 10000;

/** Pull end-to-end latency in ms out of a single ``profile_export.jsonl``
 *  record. The per-request stream stores metric values as
 *  ``metrics.request_latency = {value, unit}`` where ``unit`` is usually
 *  ``"ns"`` (raw) but some transports emit ``"s"``. Returns null when the
 *  metric is missing or the record is an error. */
function recordLatencyMs(rec) {
  if (!rec || rec.error != null) return null;
  const m = rec.metrics?.request_latency;
  if (!m || m.value == null) return null;
  const v = Number(m.value);
  if (!isFinite(v)) return null;
  const unit = m.unit ?? 'ns';
  if (unit === 'ns') return v / 1e6;
  if (unit === 'us') return v / 1e3;
  if (unit === 'ms') return v;
  if (unit === 's')  return v * 1e3;
  return v;  // unknown unit — show raw value so it at least plots
}

/** Stride-sample a latency array down to ``max`` points by keeping every Nth
 *  entry. Stride sampling (vs random) preserves monotonic request-index
 *  order, which is what the x-axis encodes. */
function strideSample(values, max) {
  if (values.length <= max) return values;
  const stride = Math.ceil(values.length / max);
  const out = [];
  for (let i = 0; i < values.length; i += stride) out.push(values[i]);
  return out;
}

/** Line chart of end-to-end latency (ms) vs request index.
 *
 *  Fetches ``profile_export.jsonl`` once on mount (and on ns/name/epoch
 *  change); stride-samples above 10k points; renders directly with
 *  ``window.Chart`` (vendored UMD). On any render failure, falls back to
 *  a plain text badge so a single broken run never blanks the whole view.
 */
function LatencyTimelineChart({ ns, name, epoch }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);
  const [state, setState] = useState({ kind: 'loading' });

  useEffect(() => {
    let cancel = false;
    setState({ kind: 'loading' });
    api.fetchRunRequests(ns, name, epoch)
      .then(({ records, skipped }) => {
        if (cancel) return;
        if (skipped) { setState({ kind: 'skip', msg: skipped }); return; }
        const ms = records
          .map(recordLatencyMs)
          .filter(v => v != null);
        if (ms.length === 0) { setState({ kind: 'skip', msg: 'no latency data' }); return; }
        setState({ kind: 'ok', sampled: strideSample(ms, LATENCY_CHART_MAX_POINTS), total: ms.length });
      })
      .catch(err => {
        if (!cancel) setState({ kind: 'err', msg: err.message });
      });
    return () => { cancel = true; };
  }, [ns, name, epoch]);

  useEffect(() => {
    if (state.kind !== 'ok') return;
    if (!canvasRef.current || !window.Chart) return;

    const points = state.sampled.map((y, x) => ({ x, y }));
    try {
      chartRef.current = new window.Chart(canvasRef.current, {
        type: 'line',
        data: {
          datasets: [{
            label: 'end-to-end latency (ms)',
            data: points,
            borderColor: 'rgba(126, 234, 255, 0.9)',
            backgroundColor: 'rgba(126, 234, 255, 0.15)',
            borderWidth: 1,
            pointRadius: 0,
            fill: false,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          parsing: false,
          plugins: { legend: { display: false } },
          scales: {
            x: {
              type: 'linear',
              title: { display: true, text: 'request index' },
              ticks: { font: { family: "'JetBrains Mono', monospace", size: 10 } },
            },
            y: {
              title: { display: true, text: 'latency (ms)' },
              beginAtZero: true,
              ticks: { font: { family: "'JetBrains Mono', monospace", size: 10 } },
            },
          },
        },
      });
    } catch (err) {
      setState({ kind: 'err', msg: `chart render failed: ${err.message}` });
    }

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
        chartRef.current = null;
      }
    };
  }, [state]);

  const header = (meta) => html`
    <header class="slab-head slab-head--flush">
      <div class="slab-head-title"><span class="slab-head-caret">▸</span> LATENCY TIMELINE</div>
      <div class="slab-head-meta">${meta}</div>
    </header>
  `;

  if (state.kind === 'loading') {
    return html`<section class="run-latency-chart" data-testid="run-latency-chart">${header('LOADING…')}</section>`;
  }
  if (state.kind === 'skip') {
    return html`
      <section class="run-latency-chart" data-testid="run-latency-chart">
        ${header(state.msg.toUpperCase())}
      </section>
    `;
  }
  if (state.kind === 'err') {
    return html`
      <section class="run-latency-chart" data-testid="run-latency-chart">
        ${header('CHART UNAVAILABLE')}
        <div class="run-latency-chart-err">${state.msg}</div>
      </section>
    `;
  }

  const { sampled, total } = state;
  const metaText = total > sampled.length
    ? `${fmtInt(sampled.length)} / ${fmtInt(total)} REQUESTS · SAMPLED`
    : `${fmtInt(total)} REQUESTS`;
  return html`
    <section class="run-latency-chart" data-testid="run-latency-chart">
      ${header(metaText)}
      <div class="card chart-box" style="height: 300px;">
        <canvas ref=${canvasRef}></canvas>
      </div>
    </section>
  `;
}

/* ─────────────────────── fault callout ──────────────────────── */

/** Prominent red callout shown only when the phase bucket is `fault`.
 *  Surfaces the first False condition + any Failed/Error pod so the operator
 *  doesn't have to scroll to Conditions + Pods to form a mental model. */
function FaultCallout({ bucket, conditions, pods }) {
  if (bucket !== 'fault') return null;
  const falseCond = (conditions ?? []).find(c => c.status === 'False');
  const failedPod = (pods ?? []).find(p => {
    const ph = (p.phase ?? '').toLowerCase();
    return ph === 'failed' || ph === 'error';
  });
  if (!falseCond && !failedPod) return null;

  const condLabel = falseCond ? conditionLabel(falseCond).toUpperCase() : null;
  const term = failedPod?.lastState?.terminated;

  return html`
    <section class="run-fault-callout" data-testid="run-fault-callout" aria-label="Run fault details">
      ${falseCond && html`
        <div class="run-fault-headline">
          <i class="ph ph-warning-octagon"></i>
          <div class="run-fault-headline-body">
            <span class="run-fault-headline-type">${condLabel}</span>
            ${falseCond.message && html`<span class="run-fault-headline-msg">${falseCond.message}</span>`}
          </div>
        </div>
      `}
      ${failedPod && html`
        <div class="run-fault-pod">
          <span class="run-fault-pod-label">FAILING POD</span>
          <span class="run-fault-pod-name">${failedPod.name}</span>
          ${term?.reason && html`<span class="run-fault-pod-reason">${term.reason}</span>`}
          ${term?.message && html`<span class="run-fault-pod-msg">${term.message}</span>`}
        </div>
      `}
      <a
        class="run-fault-footer"
        href="#run-events"
        onclick=${(e) => {
          const el = document.getElementById('run-events');
          if (el) { e.preventDefault(); el.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
        }}
      >See EVENTS below for more</a>
    </section>
  `;
}

/* ─────────────────────── events pane ────────────────────────── */

function relTime(iso) {
  if (!iso) return '—';
  const t = new Date(iso).getTime();
  if (!isFinite(t)) return '—';
  const s = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

function EventsPane({ ns, name }) {
  const [state, setState] = useState({ kind: 'loading' });
  const [filter, setFilter] = useState('all');
  const [refreshed, setRefreshed] = useState(null);

  useEffect(() => {
    let cancel = false;
    setState({ kind: 'loading' });
    const ac = new AbortController();
    const fetchOnce = async () => {
      try {
        const r = await api.getJobEvents(ns, name);
        if (cancel) return;
        const events = Array.isArray(r) ? r : (r?.events ?? []);
        setState({ kind: 'ok', events });
        setRefreshed(Date.now());
      } catch (err) {
        if (cancel) return;
        if (/\b404\b/.test(err.message)) setState({ kind: 'none' });
        else setState({ kind: 'err', msg: err.message });
        setRefreshed(Date.now());
      }
    };
    poll(fetchOnce, 15000, ac.signal);
    return () => { cancel = true; ac.abort(); };
  }, [ns, name]);

  const header = (meta) => html`
    <header class="slab-head slab-head--flush">
      <div class="slab-head-title"><span class="slab-head-caret">▸</span> EVENTS</div>
      <div class="slab-head-meta">${meta}</div>
    </header>
  `;

  if (state.kind === 'loading') {
    return html`<section class="run-events" id="run-events" data-testid="run-events">${header('LOADING…')}</section>`;
  }
  if (state.kind === 'none') {
    return html`
      <section class="run-events" id="run-events" data-testid="run-events">
        ${header('NO EVENTS')}
        <div class="run-events-empty">No events recorded for this run.</div>
      </section>
    `;
  }
  if (state.kind === 'err') {
    return html`
      <section class="run-events run-events--err" id="run-events" data-testid="run-events">
        ${header('FETCH FAILED')}
        <ol class="run-events-list">
          <li class="run-events-row run-events-row--err">
            <span class="run-events-type run-events-type--warn">ERR</span>
            <span class="run-events-msg">${state.msg}</span>
          </li>
        </ol>
      </section>
    `;
  }

  const events = state.events ?? [];
  const shown = filter === 'warn' ? events.filter(e => e.type === 'Warning') : events;
  const metaText = `${events.length} TOTAL${refreshed != null ? ' · ' + relTime(new Date(refreshed).toISOString()) : ''}`;

  return html`
    <section class="run-events" id="run-events" data-testid="run-events">
      <header class="slab-head slab-head--flush">
        <div class="slab-head-title"><span class="slab-head-caret">▸</span> EVENTS</div>
        <div class="slab-head-meta">
          <span class="run-events-filter-group">
            <button
              class=${'run-events-filter' + (filter === 'all' ? ' is-active' : '')}
              onclick=${() => setFilter('all')}
            >ALL</button>
            <button
              class=${'run-events-filter' + (filter === 'warn' ? ' is-active' : '')}
              onclick=${() => setFilter('warn')}
            >WARN</button>
          </span>
          <span class="run-events-meta-count">${metaText}</span>
        </div>
      </header>
      ${shown.length === 0
        ? html`<div class="run-events-empty">No ${filter === 'warn' ? 'warning ' : ''}events.</div>`
        : html`
          <ol class="run-events-list">
            ${shown.map((e, i) => {
              const isWarn = e.type === 'Warning';
              const ts = e.lastTimestamp ?? e.firstTimestamp;
              const obj = e.involvedObject ?? {};
              return html`
                <li key=${(e.reason ?? '') + '-' + (ts ?? i)} class="run-events-row">
                  <span class=${'run-events-type ' + (isWarn ? 'run-events-type--warn' : 'run-events-type--dim')}>
                    ${(e.type ?? 'Normal').toUpperCase()}
                  </span>
                  <span class="run-events-reason">${e.reason ?? ''}</span>
                  <span class="run-events-msg">${e.message ?? ''}</span>
                  <span class="run-events-meta">
                    ${obj.kind ?? ''}${obj.name ? '/' + obj.name : ''}
                    ${e.count > 1 ? ' · ×' + e.count : ''}
                    ${ts ? ' · ' + relTime(ts) : ''}
                  </span>
                </li>
              `;
            })}
          </ol>
        `}
    </section>
  `;
}

/* ──────────────────────── the view ────────────────────────── */

export function Run({ ns, name, epoch }) {
  const [detail, setDetail] = useState(null);
  const [config, setConfig] = useState(null);
  const [samples, setSamples] = useState([]);
  const samplesKeyRef = useRef('');

  useEffect(() => {
    setDetail(null); setConfig(null); setSamples([]);
    samplesKeyRef.current = ns + '/' + name;
    const ac = new AbortController();
    poll(async () => {
      try {
        const [d, c] = await Promise.all([
          api.getJob(ns, name).catch(() => null),
          api.getJobConfig(ns, name).catch(() => null),
        ]);
        setDetail(d); setConfig(c);
        const s = d?.status?.liveSummary ?? d?.status?.summary ?? null;
        if (s && samplesKeyRef.current === ns + '/' + name) {
          setSamples(prev => {
            const next = [...prev, {
              t: Date.now(),
              rps:   s.throughput_rps ?? null,
              p99:   s.latency_p99_ms ?? s.latency_avg_ms ?? null,
              ttft:  s.ttft_p99_ms ?? s.ttft_avg_ms ?? null,
              tokps: s.output_token_throughput ?? null,
            }];
            return next.length > MAX_SAMPLES ? next.slice(-MAX_SAMPLES) : next;
          });
        }
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

  if (!job && !detail) {
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

  const heroVerdict =
    bucket === 'fault' ? 'error' :
    bucket === 'live'  ? 'ok'    :
    bucket === 'passed'? 'ok'    :
                         'idle';
  const heroDotClass    = `run-hero-dot run-hero-dot--${heroVerdict}`;
  const heroBorderClass = `run-hero run-hero--${heroVerdict}`;
  const heroLabel =
    bucket === 'fault'  ? 'Failed'  :
    bucket === 'live'   ? 'Healthy' :
    bucket === 'passed' ? 'Completed' :
                          'Idle';
  const heroReasons = [
    job?.model && `model: ${job.model}`,
    pods.length > 0 && `${pods.filter(p => (p.phase || '').toLowerCase() === 'running').length}/${pods.length} pods running`,
  ].filter(Boolean).join(' · ');
  const phasePctNum = phasePct(active);
  const phaseFillClass =
    active?.complete ? 'run-hero-phase-fill run-hero-phase-fill--done' :
                       'run-hero-phase-fill';
  const phaseTotal = active?.total_expected_requests ?? active?.expected_requests ?? active?.requests_total ?? null;
  const phaseDone  = active?.final_requests_completed ?? active?.requestsCompleted ?? active?.requests_completed ?? active?.completed ?? 0;

  return html`
    <div class=${'v-run v-run--' + bucket} data-testid="page-job-detail">

      <!-- 1. HERO -->
      <section class=${heroBorderClass}>
        <div class="run-hero-health">
          <div class=${heroDotClass}></div>
          <div>
            <div class="run-hero-label">${heroLabel}</div>
            ${heroReasons && html`<div class="run-hero-reasons">${heroReasons}</div>`}
          </div>
        </div>
        <div class="run-hero-clock">
          <div class="run-hero-clock-line">
            <span class="run-hero-clock-label">Elapsed</span>
            <span class=${'run-hero-clock-val' + (elapsed != null ? '' : ' run-hero-clock-val--dim')}>
              ${elapsed != null ? fmtDuration(elapsed) : '—'}
            </span>
          </div>
          <div class="run-hero-clock-line">
            <span class="run-hero-clock-label">Phase ETA</span>
            <span class=${'run-hero-clock-val' + (eta != null ? '' : ' run-hero-clock-val--dim')}>
              ${eta != null ? fmtDuration(eta) : '—'}
            </span>
          </div>
        </div>
        <div class="run-hero-phase">
          <div class="run-hero-phase-head">
            <span class=${'run-hero-phase-name' + (active ? '' : ' run-hero-phase-name--idle')}>
              ${active?.name ?? 'idle'}
            </span>
            <span class="run-hero-phase-pct">${fmtPercent(phasePctNum, 0)}</span>
          </div>
          <div class="run-hero-phase-track">
            <div class=${phaseFillClass} style=${'width: ' + phasePctNum + '%'}></div>
          </div>
          <div class="run-hero-phase-sub">
            ${fmtInt(phaseDone)}${phaseTotal ? ' / ' + fmtInt(phaseTotal) : ''} requests
          </div>
        </div>
      </section>

      <!-- 1a. ACTIONS (cancel / relaunch / history / compare) -->
      <div class="run-hero-actions">
        <${CancelButton}        ns=${ns} name=${name} bucket=${bucket} />
        <${RelaunchButton}      ns=${ns} name=${name} config=${config} />
        <${RunHistoryDropdown}  ns=${ns} name=${name} selectedEpoch=${epoch} />
        <${CompareWithDropdown} ns=${ns} name=${name} selectedEpoch=${epoch} />
      </div>

      <!-- 1b. IDENTITY -->
      <${IdentityStrip} job=${job} config=${config} summary=${summary} />

      <!-- 1c. FAULT CALLOUT -->
      <${FaultCallout} bucket=${bucket} conditions=${conditions} pods=${pods} />

      <!-- 2. CONDITIONS -->
      <${ConditionsStrip} conditions=${conditions} />

      <!-- 3. KPIs (replaces METER BAY) -->
      <section class="run-kpis">
        <${RunKpi} label="Throughput"  primary=""            value=${rps  != null ? fmtNumber(rps, 1) : '—'} unit="req/s" tone=${rps  != null ? 'good' : ''} />
        <${RunKpi} label="TTFT p99"    primary="first token" value=${ttft != null ? fmtInt(ttft)     : '—'} unit="ms"    tone="" />
        <${RunKpi} label="Latency p99" primary="end-to-end"  value=${p99  != null ? fmtInt(p99)      : '—'} unit="ms"    tone=${p99 != null && p99 > 500 ? 'warn' : ''} />
        <${RunKpi} label="Token/s"     primary="output"      value=${tokps != null ? fmtInt(tokps)   : '—'} unit="tok/s" tone=${tokps != null ? 'good' : ''} />
        <${ReliabilityMeter} summary=${summary} slosDeclared=${slosDeclared} />
      </section>

      <!-- 3b. LIVE SPARKLINES -->
      ${bucket === 'live' && html`
        <section class="run-sparks" data-testid="run-sparks" aria-label="Live metric sparklines">
          ${SPARK_SPECS.map(spec => html`
            <${SparkTile} key=${spec.key} spec=${spec} samples=${samples} />
          `)}
        </section>
      `}

      <!-- 3c. REQUEST-LATENCY TIMELINE (completed runs) -->
      ${bucket !== 'live' && html`
        <${LatencyTimelineChart} ns=${ns} name=${name} epoch=${epoch} />
      `}

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

      <!-- 5b. EVENTS -->
      <${EventsPane} ns=${ns} name=${name} />

      <!-- 5c. LOGS -->
      <${LogsPane} ns=${ns} name=${name} pods=${pods} />

      <!-- 6. GPU TELEMETRY -->
      <${GpuTelemetry} metrics=${gpuMetrics} />

      <!-- 7. RESULTS -->
      <${ResultsPane} ns=${ns} name=${name} epoch=${epoch} />

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

function RunKpi({ label, primary, value, unit, tone }) {
  const cls = 'run-kpi' + (tone ? ' run-kpi--' + tone : '');
  return html`
    <div class=${cls}>
      <div class="run-kpi-head">
        <div class="run-kpi-label-block">
          <span>${label}</span>
          ${primary && html`<span class="run-kpi-primary-stat">${primary}</span>`}
        </div>
      </div>
      <div class="run-kpi-big">
        <span class="run-kpi-big-val">${value}</span>
        ${unit && html`<span class="run-kpi-big-unit">${unit}</span>`}
      </div>
    </div>
  `;
}
