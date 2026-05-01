// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * K8s events pane — copied from ``operator/ui/views/run.js::EventsPane``
 * unchanged. Polls ``/api/v1/jobs/<ns>/<name>/events`` every 15s and shows
 * the last batch with an All / Warn filter and a relative-time timestamp.
 */

import { html } from 'htm/preact';
import { useEffect, useState } from 'preact/hooks';
import { api, poll } from '../lib/api.js';

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

export function EventsPane({ ns, name }) {
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

  const headerRow = (meta, extras) => html`
    <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; flex-wrap:wrap">
      <div class="run-events-title">Events</div>
      <div style="display:flex; gap:8px; align-items:center; font-size:var(--font-xs); color:var(--muted); font-family:var(--font-mono)">
        ${extras}
        <span>${meta}</span>
      </div>
    </div>
  `;

  if (state.kind === 'loading') {
    return html`<section class="run-events" id="run-events" data-testid="run-events">${headerRow('loading…', null)}</section>`;
  }
  if (state.kind === 'none') {
    return html`
      <section class="run-events" id="run-events" data-testid="run-events">
        ${headerRow('no events', null)}
        <div class="empty">No events recorded for this run.</div>
      </section>
    `;
  }
  if (state.kind === 'err') {
    return html`
      <section class="run-events run-events--err" id="run-events" data-testid="run-events">
        ${headerRow('fetch failed', null)}
        <div class="run-events-list">
          <div class="run-event run-event--error">${state.msg}</div>
        </div>
      </section>
    `;
  }

  const events = state.events ?? [];
  const shown = filter === 'warn' ? events.filter(e => e.type === 'Warning') : events;
  const metaText = `${events.length} total${refreshed != null ? ' · ' + relTime(new Date(refreshed).toISOString()) : ''}`;
  const filterControls = html`
    <span style="display:inline-flex; gap:4px">
      <button
        class=${'btn btn--ghost' + (filter === 'all' ? ' btn--primary' : '')}
        style="font-size:10px; padding:2px 8px"
        onclick=${() => setFilter('all')}
      >All</button>
      <button
        class=${'btn btn--ghost' + (filter === 'warn' ? ' btn--primary' : '')}
        style="font-size:10px; padding:2px 8px"
        onclick=${() => setFilter('warn')}
      >Warn</button>
    </span>
  `;

  return html`
    <section class="run-events" id="run-events" data-testid="run-events">
      ${headerRow(metaText, filterControls)}
      ${shown.length === 0
        ? html`<div class="empty">No ${filter === 'warn' ? 'warning ' : ''}events.</div>`
        : html`
          <div class="run-events-list">
            ${shown.map((e, i) => {
              const isWarn = e.type === 'Warning';
              const tone = isWarn ? 'warn' : '';
              const ts = e.lastTimestamp ?? e.firstTimestamp;
              const obj = e.involvedObject ?? {};
              return html`
                <div key=${(e.reason ?? '') + '-' + (ts ?? i)} class=${'run-event' + (tone ? ' run-event--' + tone : '')}>
                  <span class="run-event-ts">${ts ? relTime(ts) : '—'}</span>
                  <strong>${e.reason ?? ''}</strong>
                  ${e.message ? ' · ' + e.message : ''}
                  ${obj.kind ? html` <span style="color:var(--dim)">· ${obj.kind}${obj.name ? '/' + obj.name : ''}</span>` : ''}
                  ${e.count > 1 ? html` <span style="color:var(--dim)">· ×${e.count}</span>` : ''}
                </div>
              `;
            })}
          </div>
        `}
    </section>
  `;
}
