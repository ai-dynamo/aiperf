// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Logs tab for DiagnosticsPanel — ports the data hook + render of
 * ``components/logs-pane.js``, minus the outer card chrome (Panel
 * supplies it now). Streams ``/api/v1/jobs/<ns>/<name>/logs`` for the
 * selected pod with a 2000-line rolling buffer, sticky auto-scroll, and
 * a tail-size override. Network is gated on ``active`` so the hidden
 * tab does not stream.
 */

import { html } from 'htm/preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import { api } from '../lib/api.js';

const LOGS_MAX_LINES = 2000;

// Container names that are sidecar-ish noise; skipped when picking the
// default container so the user lands on the workload container by default.
// Mirrors ``_default_container`` in ``jobs_logs.py``.
const SIDECAR_NAMES = new Set(['event-bus', 'results', 'istio-proxy']);

// Display order for control-plane container chips. Containers not listed
// here are appended at the end in the order returned by the API.
const CONTROL_PLANE_ORDER = [
  'control-plane',
  'dataset-manager',
  'timing-manager',
  'records-manager',
  'api',
  'gpu-telemetry-manager',
  'server-metrics-manager',
  'event-bus-proxy',
  'results-sidecar',
];

// Words that should stay all-caps when title-casing container names.
const ACRONYMS = new Set(['api', 'gpu', 'cpu']);

function titleCaseContainer(name) {
  if (!name) return '';
  return name.split(/[-_]/).map(w => {
    if (!w) return '';
    if (ACRONYMS.has(w.toLowerCase())) return w.toUpperCase();
    return w[0].toUpperCase() + w.slice(1).toLowerCase();
  }).join(' ');
}

function isControllerPod(pod) {
  return (pod?.containers ?? []).includes('control-plane');
}

function orderControlPlane(containers) {
  const present = new Set(containers);
  const ordered = CONTROL_PLANE_ORDER.filter(c => present.has(c));
  const known = new Set(ordered);
  const extras = containers.filter(c => !known.has(c));
  return [...ordered, ...extras];
}

function pickDefaultContainer(containers) {
  if (!containers || containers.length === 0) return null;
  for (const c of containers) {
    if (!SIDECAR_NAMES.has(c)) return c;
  }
  return containers[0];
}

function truncPodName(name, max = 24) {
  if (!name) return '—';
  if (name.length <= max) return name;
  return '…' + name.slice(-(max - 1));
}

export function LogsTab({ ns, name, pods, kind = 'job', active }) {
  const podList = (pods ?? []).filter(p => p?.name);
  const controllerPod = podList.find(isControllerPod) ?? null;
  const workerPods = podList.filter(p => !isControllerPod(p));
  const cpContainers = controllerPod ? orderControlPlane(controllerPod.containers ?? []) : [];

  const [selectedPod, setSelectedPod] = useState(null);
  const [selectedContainer, setSelectedContainer] = useState(null);
  const [tailLines, setTailLines] = useState(200);
  const [follow, setFollow] = useState(true);
  const [tail, setTail] = useState([]);
  const [err, setErr] = useState(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const bufRef = useRef([]);
  const bodyRef = useRef(null);
  const autoScrollRef = useRef(true);

  // Auto-select on first load and re-align when pod list changes. Prefer
  // the controller pod with its first control-plane container; fall back
  // to the first worker pod.
  useEffect(() => {
    if (podList.length === 0) { setSelectedPod(null); setSelectedContainer(null); return; }
    if (selectedPod && podList.find(p => p.name === selectedPod)) return;
    if (controllerPod) {
      setSelectedPod(controllerPod.name);
      setSelectedContainer(cpContainers[0] ?? null);
      setFollow((controllerPod.phase ?? '').toLowerCase() === 'running');
      return;
    }
    const w = workerPods[0];
    setSelectedPod(w.name);
    setSelectedContainer(pickDefaultContainer(w.containers ?? []));
    setFollow((w.phase ?? '').toLowerCase() === 'running');
  }, [podList.map(p => p.name).join('|')]);

  // Reset / re-align container selection whenever the pod (or its container
  // list) changes. Keeps the picker showing a valid container at all times.
  const selectedPodObj = podList.find(p => p.name === selectedPod) ?? null;
  const containerList = selectedPodObj?.containers ?? [];
  const containerKey = containerList.join('|');
  useEffect(() => {
    if (containerList.length === 0) {
      setSelectedContainer(null);
      return;
    }
    if (!selectedContainer || !containerList.includes(selectedContainer)) {
      setSelectedContainer(pickDefaultContainer(containerList));
    }
  }, [selectedPod, containerKey]);

  useEffect(() => { autoScrollRef.current = autoScroll; }, [autoScroll]);

  // Stream lifecycle: reset buffer + (re)open on any dep change. Gated on
  // ``active`` so a hidden Logs tab does not hold a streaming connection.
  useEffect(() => {
    if (!active) return;
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
        const fetchLogs = kind === 'sweep' ? api.getSweepLogs : api.getJobLogs;
        if (follow) {
          const res = await fetchLogs(ns, name, {
            pod: selectedPod, container: selectedContainer ?? undefined,
            follow: true, tailLines: clampedTail, signal: ac.signal,
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
          const text = await fetchLogs(ns, name, {
            pod: selectedPod, container: selectedContainer ?? undefined,
            follow: false, tailLines: clampedTail, signal: ac.signal,
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
  }, [ns, name, selectedPod, selectedContainer, follow, tailLines, kind, active]);

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

  if (podList.length === 0) {
    return html`
      <div class="diag-tab-body run-logs" data-testid="run-logs">
        <div style="display:flex; justify-content:flex-end; align-items:center; gap:8px; flex-wrap:wrap">
          <div style="font-size:var(--font-xs); color:var(--muted); font-family:var(--font-mono)">no pods yet</div>
        </div>
        <div class="empty">No pods yet — logs will appear here once workers are scheduled.</div>
      </div>
    `;
  }

  const selectedIsController = selectedPodObj && isControllerPod(selectedPodObj);
  const workerContainerList = !selectedIsController ? containerList : [];

  return html`
    <div class="diag-tab-body run-logs" data-testid="run-logs">
      <div class="run-logs-head">
        <div class="run-logs-actions">
          <button type="button"
            class=${'btn' + (follow ? ' btn--primary' : ' btn--ghost')}
            onclick=${() => setFollow(f => !f)}
            data-testid="run-logs-follow"
            title=${follow ? 'Pause streaming' : 'Resume live follow'}
          >
            ${follow ? 'Following' : 'Paused'}
          </button>
          <label class="run-logs-tail">
            Tail
            <input
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
          <span class="run-logs-meta">
            ${tail.length} line${tail.length === 1 ? '' : 's'}${follow ? ' · live' : ''}
          </span>
        </div>
      </div>

      <div class="logs-picker">
        ${controllerPod && cpContainers.length > 0 && html`
          <div class="logs-picker-row">
            <span class="logs-picker-label">Control plane</span>
            <div class="cp-chips" role="tablist" aria-label="Control plane services">
              ${cpContainers.map(c => {
                const activeChip = selectedPod === controllerPod.name && selectedContainer === c;
                return html`
                  <button
                    key=${c}
                    type="button"
                    class=${'cp-chip' + (activeChip ? ' cp-chip--active' : '')}
                    role="tab"
                    aria-selected=${activeChip}
                    data-testid=${'cp-chip-' + c}
                    onclick=${() => {
                      setSelectedPod(controllerPod.name);
                      setSelectedContainer(c);
                    }}
                  >
                    ${titleCaseContainer(c)}
                  </button>
                `;
              })}
            </div>
          </div>
        `}

        ${workerPods.length > 0 && html`
          <div class="logs-picker-row">
            <span class="logs-picker-label">Worker</span>
            <select
              class="ui-select ui-select--rounded"
              value=${selectedIsController ? '' : (selectedPod ?? '')}
              onchange=${e => {
                const wp = workerPods.find(p => p.name === e.target.value);
                if (!wp) return;
                setSelectedPod(wp.name);
                setSelectedContainer(pickDefaultContainer(wp.containers ?? []));
              }}
              data-testid="run-logs-pod"
            >
              <option value="" disabled hidden>Select a worker pod…</option>
              ${workerPods.map(p => html`
                <option key=${p.name} value=${p.name}>
                  ${truncPodName(p.name, 40)} · ${(p.phase ?? 'unknown').toLowerCase()}
                </option>
              `)}
            </select>
            ${!selectedIsController && workerContainerList.length > 1 && html`
              <select
                class="ui-select ui-select--rounded ui-select--sm"
                value=${selectedContainer ?? ''}
                onchange=${e => setSelectedContainer(e.target.value)}
                title="Select container to tail"
                data-testid="run-logs-container"
              >
                ${workerContainerList.map(c => html`
                  <option key=${c} value=${c}>${c}</option>
                `)}
              </select>
            `}
          </div>
        `}
      </div>

      <pre class="run-logs-body" ref=${bodyRef} onscroll=${onScroll} data-testid="run-logs-body">${tail.join('\n')}</pre>
      ${err && html`<div class="run-logs-error">${err}</div>`}
      ${!autoScroll && html`
        <button type="button" class="btn btn--ghost run-logs-jump" onclick=${jumpToLatest} data-testid="run-logs-jump">
          Jump to latest
        </button>
      `}
    </div>
  `;
}
