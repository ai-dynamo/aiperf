// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * LogStrip — always-on bottom event feed.
 *
 * Persistent strip at the bottom of the shell. Streams derived lifecycle
 * events — diffs successive ``jobs`` snapshots and emits events on phase
 * transitions, worker-ready changes, and error discovery.
 *
 * Derivation is purely client-side: no backend streaming. First snapshot
 * primes the state; thereafter only transitions produce entries.
 *
 * Copied verbatim from ``operator/ui/components/log-strip.js``. The only
 * adaptation is the click navigate path (``/jobs/<ns>/<name>`` instead of
 * the legacy ``/ns/<ns>/run/<name>``) so clicks resolve in ui-v1.
 */

import { html } from 'htm/preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import { jobs } from '../lib/state.js';
import { navigate, buildJobPath } from '../lib/router.js';

const MAX_EVENTS = 120;
const pad = n => String(n).padStart(2, '0');

function fmtTs(ts) {
  const d = new Date(ts);
  return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`;
}

// Map a job phase string to a severity bucket used by the filter UI and
// to choose the entry's color class.
function phaseSeverity(phase) {
  const p = (phase ?? '').toLowerCase();
  if (p === 'failed' || p === 'error')                return 'error';
  if (p === 'running' || p === 'initializing' || p === 'pending') return 'info';
  if (p === 'completed' || p === 'succeeded')         return 'info';
  return 'info';
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
          const sev = phaseSeverity(j.phase);
          fresh.push({
            ts: now,
            severity: sev,
            cat: 'phase',
            ns: j.namespace, name: j.name,
            msg: sev === 'error'
              ? `${j.namespace}/${j.name} discovered in error state`
              : `${j.namespace}/${j.name} new run detected`,
          });
          continue;
        }

        if (p.phase !== j.phase) {
          const sev = phaseSeverity(j.phase);
          fresh.push({
            ts: now,
            severity: sev,
            cat: 'phase',
            ns: j.namespace, name: j.name,
            msg: `${j.namespace}/${j.name} phase ▸ ${(j.phase ?? '').toLowerCase()}`,
          });
        }
        if (p.workersReady !== j.workersReady && j.workersTotal > 0) {
          fresh.push({
            ts: now,
            severity: 'info',
            cat: 'worker',
            ns: j.namespace, name: j.name,
            msg: `${j.namespace}/${j.name} workers ${j.workersReady}/${j.workersTotal}`,
          });
        }
      }
      prevRef.current = next;
      if (fresh.length > 0) {
        // Append new events at the tail so the strip reads oldest → newest;
        // trim from the head when we exceed MAX_EVENTS so the most recent
        // ones stay on screen.
        setEvents(prev => [...prev, ...fresh].slice(-MAX_EVENTS));
      }
    });
    return unsubscribe;
  }, []);

  return events;
}

// Default-visible row count when the strip is collapsed. Keep small — the
// strip lives at the bottom of every page and competes for vertical real
// estate with the page body. Users who want history click "Show all".
const COLLAPSED_ROWS = 5;

export function LogStrip() {
  const events = useEventFeed();
  const [filter, setFilter] = useState('all');
  const [collapsed, setCollapsed] = useState(true);
  const bodyRef = useRef(null);
  const stickyRef = useRef(true);

  const counts = {
    all: events.length,
    warn: events.filter(e => e.severity === 'warn').length,
    error: events.filter(e => e.severity === 'error').length,
  };

  const filtered = events.filter(e => {
    if (filter === 'all') return true;
    return e.severity === filter;
  });
  // When collapsed, show only the last N entries so the strip stays a tidy
  // 5-line peek; expanding reveals the full bounded window (up to MAX_EVENTS).
  const visible = collapsed ? filtered.slice(-COLLAPSED_ROWS) : filtered;
  const hiddenCount = filtered.length - visible.length;

  // Auto-scroll the strip to the bottom on each new event so the freshest
  // line stays in view. Skip when the user has scrolled up (>32 px from the
  // bottom) so reading older entries isn't interrupted. Re-snap when toggling
  // collapse so the latest event lands in view in either mode.
  useEffect(() => {
    const el = bodyRef.current;
    if (!el || !stickyRef.current) return;
    el.scrollTop = el.scrollHeight;
  }, [visible.length, collapsed]);

  function onScroll(e) {
    const el = e.currentTarget;
    stickyRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 32;
  }

  const filters = [
    { key: 'all',   label: 'All' },
    { key: 'warn',  label: 'Warn' },
    { key: 'error', label: 'Error' },
  ];

  return html`
    <section
      class=${'log-strip' + (collapsed ? ' log-strip--collapsed' : '')}
      aria-label="Event log"
      data-testid="log-strip"
    >
      <div class="log-strip-head">
        <div class="log-strip-title">Event Log</div>
        <div class="log-strip-filters" role="tablist">
          ${filters.map(f => html`
            <button
              key=${f.key}
              type="button"
              class=${'log-strip-filter' + (filter === f.key ? ' log-strip-filter--active' : '')}
              role="tab"
              aria-selected=${filter === f.key}
              onclick=${() => setFilter(f.key)}
            >
              ${f.label}
              <span class="log-strip-filter-count">${counts[f.key]}</span>
            </button>
          `)}
        </div>
        <button
          type="button"
          class="log-strip-toggle"
          data-testid="log-strip-toggle"
          aria-expanded=${!collapsed}
          onclick=${() => setCollapsed(v => !v)}
          title=${collapsed
            ? (hiddenCount > 0 ? `Show all ${filtered.length} events (${hiddenCount} hidden)` : 'Expand event log')
            : 'Collapse to last ' + COLLAPSED_ROWS + ' events'}
        >
          ${collapsed
            ? (hiddenCount > 0 ? `Show all (${filtered.length})` : 'Expand')
            : 'Collapse'}
        </button>
      </div>
      <div class="log-strip-body" ref=${bodyRef} onscroll=${onScroll}>
        ${visible.map((e, i) => {
          const sevClass = e.severity === 'error' ? ' log-strip-entry--error'
                         : e.severity === 'warn'  ? ' log-strip-entry--warn'
                         : '';
          const jobPath = buildJobPath({ namespace: e.ns, name: e.name });
          return html`
            <a
              key=${e.ts + '-' + i}
              class=${'log-strip-entry' + sevClass}
              href=${jobPath}
              onclick=${event => {
                event.preventDefault();
                navigate(jobPath);
              }}
            >
              <span class="ts">${fmtTs(e.ts)}</span>
              <span class=${'log-strip-cat log-strip-cat--' + e.cat}>${e.cat}</span>
              <span>${e.msg}</span>
            </a>
          `;
        })}
      </div>
    </section>
  `;
}
