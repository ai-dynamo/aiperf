// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * LOG STRIP — bottom terminal-style event feed.
 *
 * Collapsed by default (grid row height = 0). Toggle via Ctrl+` or the terminal
 * button in the top rail. When open, shows a reverse-chronological list of
 * life-cycle events derived from the `jobs` signal — phase transitions (Pending
 * → Running → Completed/Failed), worker-ready transitions, SLO violations.
 *
 * Derivation is purely client-side: we diff successive snapshots of the jobs
 * list and emit events when a tracked field changes. This keeps the log cheap
 * (no backend streaming) and always in sync with what the left rail shows.
 */

import { html } from 'htm/preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import { jobs } from '../lib/state.js';
import { navigate } from '../lib/router.js';

const MAX_EVENTS = 80;
const pad = n => String(n).padStart(2, '0');

function fmtTs(ts) {
  const d = new Date(ts);
  return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`;
}

function phaseBucket(phase) {
  const p = (phase ?? '').toLowerCase();
  if (p === 'running' || p === 'initializing' || p === 'pending') return 'live';
  if (p === 'failed'  || p === 'error')                            return 'fault';
  if (p === 'completed' || p === 'succeeded')                      return 'passed';
  return 'other';
}

/** Diff successive `jobs` snapshots, synthesizing events. */
function useEventFeed() {
  const [events, setEvents] = useState([]);
  const prevRef = useRef(new Map());

  useEffect(() => {
    const unsubscribe = jobs.subscribe((list) => {
      const now = Date.now();
      const prev = prevRef.current;
      const next = new Map();
      const fresh = [];

      for (const j of list ?? []) {
        const key = `${j.namespace}/${j.name}`;
        next.set(key, { phase: j.phase, workersReady: j.workersReady, workersTotal: j.workersTotal });

        const p = prev.get(key);
        if (!p) {
          // first time we see the run — only emit if actively running or failed
          const b = phaseBucket(j.phase);
          if (b === 'live' || b === 'fault') {
            fresh.push({
              ts: now, kind: b === 'fault' ? 'fault' : 'info',
              ns: j.namespace, name: j.name,
              msg: b === 'fault' ? 'discovered in FAULT state' : 'run detected',
            });
          }
          continue;
        }

        if (p.phase !== j.phase) {
          const to = (j.phase ?? '').toLowerCase();
          const b = phaseBucket(j.phase);
          fresh.push({
            ts: now,
            kind: b === 'fault' ? 'fault' : b === 'passed' ? 'pass' : 'info',
            ns: j.namespace, name: j.name,
            msg: `phase ▸ ${to}`,
          });
        }
        if (p.workersReady !== j.workersReady && j.workersTotal > 0) {
          fresh.push({
            ts: now, kind: 'dim',
            ns: j.namespace, name: j.name,
            msg: `workers ${j.workersReady}/${j.workersTotal}`,
          });
        }
      }
      prevRef.current = next;
      if (fresh.length > 0) {
        setEvents(prev => [...fresh.reverse(), ...prev].slice(0, MAX_EVENTS));
      }
    });
    return unsubscribe;
  }, []);

  return events;
}

export function LogStrip({ open, onClose }) {
  const events = useEventFeed();
  const [filter, setFilter] = useState('all');

  const visible = events.filter(e => {
    if (filter === 'all') return true;
    if (filter === 'fault') return e.kind === 'fault';
    if (filter === 'pass') return e.kind === 'pass';
    return e.kind !== 'dim';
  });

  return html`
    <section class=${'log-strip' + (open ? ' log-strip--open' : '')} aria-label="Event log" data-testid="log-strip" aria-hidden=${!open}>
      <div class="log-strip-head">
        <div class="log-strip-title">
          <span class="log-strip-caret">▸</span>
          EVENT LOG
          <span class="log-strip-count">${events.length}</span>
        </div>
        <div class="log-strip-filters" role="tablist">
          ${['all', 'events', 'fault', 'pass'].map(k => html`
            <button
              key=${k}
              class=${'log-strip-filter' + (filter === k ? ' is-active' : '')}
              onclick=${() => setFilter(k)}
              role="tab"
              aria-selected=${filter === k}
            >${k.toUpperCase()}</button>
          `)}
        </div>
        <button class="log-strip-close" onclick=${onClose} title="Close (Ctrl + grave)" aria-label="Close log">
          <i class="ph ph-x"></i>
        </button>
      </div>
      <div class="log-strip-body">
        ${visible.length === 0
          ? html`<div class="log-strip-empty">no events — waiting for fleet activity</div>`
          : html`
              <ol class="log-strip-list">
                ${visible.map((e, i) => html`
                  <li
                    key=${e.ts + '-' + i}
                    class=${'log-event log-event--' + e.kind}
                    onclick=${() => navigate('/run/' + encodeURIComponent(e.ns) + '/' + encodeURIComponent(e.name))}
                  >
                    <span class="log-event-ts">${fmtTs(e.ts)}</span>
                    <span class="log-event-tag">${e.kind === 'fault' ? 'FAIL' : e.kind === 'pass' ? 'PASS' : e.kind === 'dim' ? 'INFO' : 'EVT'}</span>
                    <span class="log-event-src">${e.ns}/${e.name}</span>
                    <span class="log-event-msg">${e.msg}</span>
                  </li>
                `)}
              </ol>
            `}
      </div>
    </section>
  `;
}
