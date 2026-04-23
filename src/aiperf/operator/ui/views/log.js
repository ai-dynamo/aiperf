// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * LOG — historical run log, grouped by day.
 *
 * Lives in the main viewport. The bottom strip (LogStrip) carries live-diff
 * events since page load; this view shows the durable archive: every run's
 * final outcome, its duration, and a clickable link to the RUN view.
 */

import { html } from 'htm/preact';
import { useMemo } from 'preact/hooks';
import { jobs } from '../lib/state.js';
import { navigate } from '../lib/router.js';
import { fmtDuration, fmtInt, fmtNumber } from '../lib/format.js';

function phaseKind(phase) {
  const p = (phase ?? '').toLowerCase();
  if (['running', 'initializing', 'pending'].includes(p)) return 'live';
  if (['failed', 'error'].includes(p)) return 'fault';
  if (['completed', 'succeeded'].includes(p)) return 'passed';
  return 'other';
}

function dayKey(ts) {
  if (!ts) return 'unknown';
  return new Date(ts).toISOString().slice(0, 10);
}

export function Log() {
  const list = jobs.value ?? [];

  const grouped = useMemo(() => {
    const g = new Map();
    const sorted = [...list].sort((a, b) => new Date(b.completionTime ?? b.created ?? 0) - new Date(a.completionTime ?? a.created ?? 0));
    for (const j of sorted) {
      const k = dayKey(j.completionTime ?? j.created);
      (g.get(k) ?? g.set(k, []).get(k)).push(j);
    }
    return [...g.entries()];
  }, [list]);

  return html`
    <div class="v-log" data-testid="page-history">
      <header class="v-head">
        <div class="v-head-title">
          <span class="v-head-caret">▸</span>
          <h1>RUN LOG</h1>
        </div>
        <div class="v-head-meta">${list.length} RUNS TRACKED</div>
      </header>

      ${grouped.length === 0
        ? html`<div class="v-log-empty">NO HISTORY YET</div>`
        : grouped.map(([day, runs]) => html`
            <section key=${day} class="v-log-day">
              <header class="v-log-day-head">
                <span class="v-log-day-caret">▸</span>
                ${day.replace(/-/g, '.')}
                <span class="v-log-day-count">${runs.length}</span>
              </header>
              <ol class="v-log-list">
                ${runs.map(j => {
                  const kind = phaseKind(j.phase);
                  const dur = j.startTime && j.completionTime
                    ? (new Date(j.completionTime) - new Date(j.startTime)) / 1000
                    : null;
                  const ts = new Date(j.completionTime ?? j.created);
                  return html`
                    <li key=${j.namespace + '/' + j.name}>
                      <button
                        class=${'v-log-row v-log-row--' + kind}
                        onclick=${() => navigate('/run/' + encodeURIComponent(j.namespace) + '/' + encodeURIComponent(j.name))}
                      >
                        <span class="v-log-time">${String(ts.getUTCHours()).padStart(2,'0')}:${String(ts.getUTCMinutes()).padStart(2,'0')}</span>
                        <span class=${'v-log-dot v-log-dot--' + kind}></span>
                        <span class="v-log-phase">${(j.phase ?? '—').toUpperCase()}</span>
                        <span class="v-log-name">${j.name}</span>
                        <span class="v-log-ns">${j.namespace}</span>
                        <span class="v-log-dur">${dur != null ? fmtDuration(dur) : '—'}</span>
                        <span class="v-log-rps">
                          ${j.throughputRps != null ? html`${fmtNumber(j.throughputRps, 0)}<small> r/s</small>` : '—'}
                        </span>
                        <span class="v-log-p99">
                          ${j.latencyP99Ms != null ? html`${fmtInt(j.latencyP99Ms)}<small> ms</small>` : '—'}
                        </span>
                        <i class="ph ph-arrow-right"></i>
                      </button>
                    </li>
                  `;
                })}
              </ol>
            </section>
          `)
      }
    </div>
  `;
}
