// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Events tab for DiagnosticsPanel — ports the data hook + render of
 * ``components/events-pane.js``, minus the outer card chrome (Panel
 * supplies it now). Polls ``/api/v1/jobs/<ns>/<name>/events`` every 15s
 * with an All / Warn filter and sticky-bottom auto-scroll. Network is
 * gated on ``active`` so the hidden tab does not poll.
 */

import { html } from 'htm/preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import { api, poll } from '../lib/api.js';

const pad = n => String(n).padStart(2, '0');

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

function fmtTs(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (!isFinite(d.getTime())) return '—';
  return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`;
}

// Map a k8s event reason to a chip-color tone. Colors follow pod lifecycle:
// blue=admission, cyan=in-progress fetch, green=ready/running, pink=container
// shell created, peach=intentional teardown, amber=warning, red=hard failure.
function eventCatTone(reason, type) {
  const r = (reason ?? '').toLowerCase();
  if (!r) return type === 'Warning' ? 'warning' : 'normal';

  if (/(backoff|oom|evict|preempt)/.test(r)) return 'error';
  if (r.startsWith('failed') || r.endsWith('failed')) return 'error';
  if (/(error|invalid)/.test(r)) return 'error';

  if (/^(killing|stopping|drain)/.test(r)) return 'killing';

  if (r === 'unhealthy' || r === 'probewarning') return 'warn';

  if (r === 'scheduled') return 'scheduled';
  if (r.startsWith('pulling')) return 'pulling';
  if (r.startsWith('pulled')) return 'pulled';
  if (r === 'created' || r === 'sandboxchanged') return 'created';
  if (r === 'started' || r === 'running' || r.startsWith('noderead') || r.startsWith('successful')) return 'started';

  return type === 'Warning' ? 'warning' : 'normal';
}

export function EventsTab({ ns, name, kind = 'job', active }) {
  const [state, setState] = useState({ kind: 'loading' });
  const [filter, setFilter] = useState('all');
  const [refreshed, setRefreshed] = useState(null);
  const listRef = useRef(null);
  // Track whether the user is "stuck to bottom" so we only auto-scroll when
  // they haven't manually scrolled up to read older entries. Anything past
  // 32px from the bottom is treated as "user is reading" and we leave the
  // scroll position alone.
  const stickyRef = useRef(true);

  useEffect(() => {
    if (!active) return;
    let cancel = false;
    setState({ kind: 'loading' });
    const ac = new AbortController();
    const fetchOnce = async () => {
      try {
        const r = kind === 'sweep'
          ? await api.getSweepEvents(ns, name)
          : await api.getJobEvents(ns, name);
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
  }, [ns, name, kind, active]);

  const headerRow = (meta, extras) => html`
    <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; flex-wrap:wrap">
      <div style="display:flex; gap:8px; align-items:center; font-size:var(--font-xs); color:var(--muted); font-family:var(--font-mono); margin-left:auto">
        ${extras}
        <span>${meta}</span>
      </div>
    </div>
  `;

  // Pre-compute the sorted+filtered list before the early returns so the
  // auto-scroll useEffect below can sit above any conditional ``return`` and
  // still have stable hook ordering across renders.
  const okEvents = state.kind === 'ok' ? (state.events ?? []) : [];
  const sortedEvents = [...okEvents].sort((a, b) => {
    const ta = new Date(a.last_timestamp ?? a.first_timestamp ?? 0).getTime();
    const tb = new Date(b.last_timestamp ?? b.first_timestamp ?? 0).getTime();
    return (isFinite(ta) ? ta : 0) - (isFinite(tb) ? tb : 0);
  });
  const shown = filter === 'warn' ? sortedEvents.filter(e => e.type === 'Warning') : sortedEvents;

  // Auto-scroll the list to the bottom whenever the visible event count
  // grows AND the user hasn't scrolled up to read older entries. We snap on
  // shown.length / refreshed because each poll either appends or replaces;
  // either way the bottom anchor is what the user wants to see. Hoisted above
  // the early returns so the hook call order stays stable across states.
  useEffect(() => {
    if (state.kind !== 'ok') return;
    const el = listRef.current;
    if (!el || !stickyRef.current) return;
    el.scrollTop = el.scrollHeight;
  }, [state.kind, shown.length, refreshed]);

  if (state.kind === 'loading') {
    return html`<div class="diag-tab-body run-events" data-testid="run-events">${headerRow('loading…', null)}</div>`;
  }
  if (state.kind === 'none') {
    return html`
      <div class="diag-tab-body run-events" data-testid="run-events">
        ${headerRow('no events', null)}
        <div class="empty">No events recorded for this run.</div>
      </div>
    `;
  }
  if (state.kind === 'err') {
    return html`
      <div class="diag-tab-body run-events run-events--err" data-testid="run-events">
        ${headerRow('fetch failed', null)}
        <div class="run-events-list">
          <div class="run-event run-event--error">${state.msg}</div>
        </div>
      </div>
    `;
  }

  const events = state.events ?? [];
  const metaText = `${events.length} total${refreshed != null ? ' · ' + relTime(new Date(refreshed).toISOString()) : ''}`;
  const filterControls = html`
    <span style="display:inline-flex; gap:4px">
      <button type="button"
        class=${'btn btn--ghost' + (filter === 'all' ? ' btn--primary' : '')}
        style="font-size:10px; padding:2px 8px"
        onclick=${() => setFilter('all')}
      >All</button>
      <button type="button"
        class=${'btn btn--ghost' + (filter === 'warn' ? ' btn--primary' : '')}
        style="font-size:10px; padding:2px 8px"
        onclick=${() => setFilter('warn')}
      >Warn</button>
    </span>
  `;

  function onScroll(e) {
    const el = e.currentTarget;
    stickyRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 32;
  }

  return html`
    <div class="diag-tab-body run-events" data-testid="run-events">
      ${headerRow(metaText, filterControls)}
      ${shown.length === 0
        ? html`<div class="empty">No ${filter === 'warn' ? 'warning ' : ''}events.</div>`
        : html`
          <div class="run-events-list" ref=${listRef} onscroll=${onScroll}>
            ${shown.map((e, i) => {
              const isWarn = e.type === 'Warning';
              const tone = isWarn ? 'warn' : '';
              const ts = e.last_timestamp ?? e.first_timestamp;
              const obj = e.involved_object ?? {};
              const catTone = eventCatTone(e.reason, e.type);
              const reason = e.reason ?? (isWarn ? 'warning' : 'event');
              return html`
                <div key=${(e.reason ?? '') + '-' + (ts ?? i)} class=${'run-event' + (tone ? ' run-event--' + tone : '')}>
                  <span class="run-event-ts" title=${ts ? relTime(ts) : ''}>${fmtTs(ts)}</span>
                  <span class=${'run-event-cat run-event-cat--' + catTone}>${reason}</span>
                  ${e.message ? html`<span>${e.message}</span>` : ''}
                  ${obj.kind ? html` <span style="color:var(--dim)">· ${obj.kind}${obj.name ? '/' + obj.name : ''}</span>` : ''}
                  ${e.count > 1 ? html` <span style="color:var(--dim)">· ×${e.count}</span>` : ''}
                </div>
              `;
            })}
          </div>
        `}
    </div>
  `;
}
