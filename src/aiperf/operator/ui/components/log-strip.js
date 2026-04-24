// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * LOG STRIP — always-on bottom event feed.
 *
 * Persistent (not collapsible) 180px-tall strip at the bottom of the
 * Workbench shell. Streams derived lifecycle events — diffs successive
 * ``jobs`` snapshots and emits events on phase transitions, worker-ready
 * changes, fault discovery.
 *
 * Derivation is purely client-side: no backend streaming. First snapshot
 * primes the state; thereafter only transitions produce entries.
 */

import { html } from 'htm/preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import { jobs } from '../lib/state.js';
import { navigate } from '../lib/router.js';

const MAX_EVENTS = 120;
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

function useEventFeed() {
  const [events, setEvents] = useState([]);
  const prevRef = useRef(null);  // null on first snapshot — don't emit

  useEffect(() => {
    const unsubscribe = jobs.subscribe((list) => {
      const now = Date.now();
      const next = new Map();
      const fresh = [];
      const prev = prevRef.current;

      for (const j of list ?? []) {
        const key = `${j.namespace}/${j.name}`;
        next.set(key, { phase: j.phase, workersReady: j.workersReady, workersTotal: j.workersTotal });

        if (prev === null) continue;  // first snapshot — prime, don't emit
        const p = prev.get(key);
        if (!p) {
          const b = phaseBucket(j.phase);
          fresh.push({
            ts: now, kind: b === 'fault' ? 'fault' : b === 'live' ? 'info' : 'dim',
            ns: j.namespace, name: j.name,
            msg: b === 'fault' ? 'discovered in FAULT state'
               : b === 'live'  ? 'new run detected'
               : 'new run archived',
          });
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

export function LogStrip() {
  const events = useEventFeed();
  const [filter, setFilter] = useState('all');

  const visible = events.filter(e => {
    if (filter === 'all') return true;
    if (filter === 'fault') return e.kind === 'fault';
    if (filter === 'pass') return e.kind === 'pass';
    return e.kind !== 'dim';
  });

  return html`
    <section class="log-strip" aria-label="Event log" data-testid="log-strip">
      <div class="log-strip-head">
        <div class="log-strip-title">
          <span class="log-strip-caret">▸</span>
          EVENTS
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
      </div>
      <div class="log-strip-body">
        ${visible.length === 0
          ? html`<div class="log-strip-empty">no events yet — waiting for run activity</div>`
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
