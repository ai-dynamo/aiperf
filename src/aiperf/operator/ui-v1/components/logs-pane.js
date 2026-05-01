// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Live pod-logs pane — copied from ``operator/ui/views/run.js::LogsPane``
 * unchanged. Streams ``/api/v1/jobs/<ns>/<name>/logs?pod=...&follow=1`` for
 * the selected pod, with a 2000-line rolling buffer, sticky auto-scroll,
 * and a tail-size override.
 */

import { html } from 'htm/preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import { api } from '../lib/api.js';

const LOGS_MAX_LINES = 2000;

function truncPodName(name, max = 24) {
  if (!name) return '—';
  if (name.length <= max) return name;
  return '…' + name.slice(-(max - 1));
}

export function LogsPane({ ns, name, pods }) {
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

  if (podList.length === 0) {
    return html`
      <section class="run-logs" id="run-logs" data-testid="run-logs">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; flex-wrap:wrap">
          <div class="run-logs-title">Logs</div>
          <div style="font-size:var(--font-xs); color:var(--muted); font-family:var(--font-mono)">no pods yet</div>
        </div>
        <div class="empty">No pods yet — logs will appear here once workers are scheduled.</div>
      </section>
    `;
  }

  return html`
    <section class="run-logs" id="run-logs" data-testid="run-logs">
      <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; flex-wrap:wrap">
        <div class="run-logs-title">Logs</div>
        <div class="run-logs-controls">
          <select
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
            class=${'btn' + (follow ? ' btn--primary' : ' btn--ghost')}
            onclick=${() => setFollow(f => !f)}
            data-testid="run-logs-follow"
            title=${follow ? 'Pause streaming' : 'Resume live follow'}
          >
            ${follow ? 'Following' : 'Paused'}
          </button>
          <label style="display:inline-flex; align-items:center; gap:4px; font-size:var(--font-xs); color:var(--muted)">
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
              style="width:64px"
            />
          </label>
          <span style="font-size:var(--font-xs); color:var(--muted); font-family:var(--font-mono)">
            ${tail.length} line${tail.length === 1 ? '' : 's'}${follow ? ' · live' : ''}
          </span>
        </div>
      </div>
      <pre class="run-logs-body" ref=${bodyRef} onscroll=${onScroll} data-testid="run-logs-body">${tail.join('\n')}</pre>
      ${err && html`<div class="run-logs-error">${err}</div>`}
      ${!autoScroll && html`
        <button class="btn btn--ghost run-logs-jump" onclick=${jumpToLatest} data-testid="run-logs-jump">
          Jump to latest
        </button>
      `}
    </section>
  `;
}
