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
import { api, poll, httpStatusOf } from '../lib/api.js';

const pad = n => String(n).padStart(2, '0');

/**
 * Turn a failed events fetch into copy that says what actually happened.
 *
 * A 404 used to render as "No events recorded for this run." — the exact
 * conflation the backend goes out of its way to avoid: routers/jobs.py:795
 * returns 200 with an empty list for an archived run whose CR is gone,
 * precisely so a missing CR does not masquerade as "no events". If this
 * endpoint answers 404, the route or the job name is wrong, and telling the
 * reader their run has no events sends them after the wrong problem.
 *
 * @param {Error & {status?: number, url?: string}} err
 * @returns {string}
 */
function describeEventsError(err) {
  const status = httpStatusOf(err);
  if (status === 404) {
    const where = err.url ? ` (${err.url})` : '';
    return `Events endpoint returned 404${where}. A run with no events returns `
      + 'an empty list, so this is a missing route or job name, not an empty '
      + 'event stream.';
  }
  if (status != null) return err.message;
  return `Could not reach the events endpoint: ${err.message}`;
}

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
        setState({ kind: 'ok', events, okAt: Date.now() });
        setRefreshed(Date.now());
      } catch (err) {
        if (cancel) return;
        // Carry the last good snapshot forward. Replacing it with the error
        // text destroys the events the reader was already looking at and
        // implies the run produced none, when in fact it produced these and
        // the refresh is what failed.
        setState(prev => ({
          kind: 'err',
          msg: describeEventsError(err),
          events: prev.events ?? null,
          okAt: prev.okAt ?? null,
        }));
        setRefreshed(Date.now());
      }
    };
    poll(fetchOnce, 15000, ac.signal);
    return () => { cancel = true; ac.abort(); };
  }, [ns, name, kind, active]);

  const headerRow = (meta, extras) => html`
    <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; flex-wrap:wrap">
      <div style="display:flex; gap:8px; align-items:center; font-size:var(--font-size-xs); color:var(--muted); font-family:var(--font-mono); margin-left:auto">
        ${extras}
        <span>${meta}</span>
      </div>
    </div>
  `;

  // Pre-compute the sorted+filtered list before the early returns so the
  // auto-scroll useEffect below can sit above any conditional ``return`` and
  // still have stable hook ordering across renders.
  // ``state.events`` survives a failed refresh, so the list below renders the
  // last good snapshot in both the 'ok' and 'err' states; the error strip
  // above it is what marks the difference.
  const okEvents = state.events ?? [];
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
    return html`
      <div class="diag-tab-body run-events" data-testid="run-events">
        ${headerRow('loading…', null)}
        <div class="empty" data-testid="run-events-loading">Loading events…</div>
      </div>
    `;
  }
  const failed = state.kind === 'err';
  const hasPriorEvents = Array.isArray(state.events) && state.events.length > 0;
  // Nothing was ever loaded and the fetch failed: there is no list to show, so
  // the error IS the content. Distinct from a run that genuinely has no events,
  // which arrives as kind='ok' with an empty array.
  if (failed && !hasPriorEvents) {
    return html`
      <div class="diag-tab-body run-events run-events--err" data-testid="run-events">
        ${headerRow('fetch failed', null)}
        <div class="run-events-list">
          <div class="run-event run-event--error" data-testid="run-events-error">${state.msg}</div>
        </div>
      </div>
    `;
  }

  const events = state.events ?? [];
  const metaText = `${events.length} total${failed ? ' · stale' : ''}`
    + `${refreshed != null ? ' · ' + relTime(new Date(refreshed).toISOString()) : ''}`;
  // One modifier, never both. style.css defines .btn--ghost (6161) after
  // .btn--primary (6155), so a button carrying both keeps ghost's transparent
  // background while primary's `color: var(--bg)` paints the label the same
  // shade as the panel behind it. The selected filter came out invisible and
  // the unselected one looked selected -- the reader cannot tell which subset
  // of events they are looking at. Matches diagnostics-logs-tab.js:238.
  const filterButton = (key, label) => html`
    <button type="button"
      class=${'btn ' + (filter === key ? 'btn--primary' : 'btn--ghost')}
      style="font-size: var(--font-size-xs); padding:2px 8px"
      aria-pressed=${filter === key}
      data-testid=${'run-events-filter-' + key}
      onclick=${() => setFilter(key)}
    >${label}</button>
  `;
  const filterControls = html`
    <span style="display:inline-flex; gap:4px">
      ${filterButton('all', 'All')}
      ${filterButton('warn', 'Warn')}
    </span>
  `;

  function onScroll(e) {
    const el = e.currentTarget;
    stickyRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 32;
  }

  const staleAgo = state.okAt != null
    ? Math.max(0, Math.round((Date.now() - state.okAt) / 1000))
    : null;

  return html`
    <div class=${'diag-tab-body run-events' + (failed ? ' run-events--err' : '')} data-testid="run-events">
      ${headerRow(metaText, filterControls)}
      ${failed && html`
        <div class="run-event run-event--error" data-testid="run-events-stale">
          Refresh failed${staleAgo != null ? `; showing events from ${staleAgo}s ago` : ''} — ${state.msg}
        </div>
      `}
      ${shown.length === 0
        ? html`<div class="empty">${filter === 'warn'
            ? 'No warning events.'
            : 'No events recorded for this run.'}</div>`
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
