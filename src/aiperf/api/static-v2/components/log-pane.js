// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Scrolling log pane with severity coloring and a minimal filter.
 *
 * Every entry carries a ``severity`` (info/warn/error) plus optional
 * ``category`` tag (phase/worker/records/…). Warnings and errors are
 * color-coded; the filter chips let a customer narrow to just warnings
 * or errors during a noisy run.
 */

import { html } from 'htm/preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import { logs } from '../lib/state.js';

const LEVELS = [
  { key: 'all',   label: 'all' },
  { key: 'warn',  label: 'warn+' },
  { key: 'error', label: 'error' },
];

function passes(entry, level) {
  if (level === 'all') return true;
  if (level === 'warn') return entry.severity === 'warn' || entry.severity === 'error';
  if (level === 'error') return entry.severity === 'error';
  return true;
}

export function LogPane() {
  const entries = logs.value;
  const [level, setLevel] = useState('all');
  const containerRef = useRef(null);

  const visible = entries.filter(e => passes(e, level));

  useEffect(() => {
    const el = containerRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [visible.length]);

  const counts = {
    all: entries.length,
    warn: entries.filter(e => e.severity === 'warn' || e.severity === 'error').length,
    error: entries.filter(e => e.severity === 'error').length,
  };

  return html`
    <div class="card">
      <div class="log-head">
        <div class="card-title" style="margin-bottom: 0">Log</div>
        <div class="log-filters">
          ${LEVELS.map((l) => html`
            <button
              key=${l.key}
              class=${'log-filter' + (level === l.key ? ' log-filter--active' : '')}
              onclick=${() => setLevel(l.key)}
              title=${l.key === 'all' ? 'show all entries' :
                     l.key === 'warn' ? 'show warnings and errors' :
                     'show only errors'}>
              ${l.label}
              <span class="log-filter-count">${counts[l.key]}</span>
            </button>
          `)}
        </div>
      </div>
      <div class="log-pane" ref=${containerRef}>
        ${visible.length === 0
          ? html`<div class="empty" style="padding: var(--space-4)">
              ${level === 'all' ? 'No events yet.' : 'No ' + level + ' entries.'}
            </div>`
          : visible.map((e, i) => html`
            <div class=${'log-entry log-entry--' + (e.severity ?? 'info')} key=${i}>
              <span class="ts">${e.ts}</span>
              ${e.category && html`<span class="log-cat log-cat--${e.category}">${e.category}</span>`}
              <span class="log-msg">${e.message}</span>
            </div>
          `)}
      </div>
    </div>
  `;
}
