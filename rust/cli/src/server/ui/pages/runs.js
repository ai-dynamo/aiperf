// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Runs landing page: a sortable, filterable cross-run table. Reads the shared
// ``runs``/``meta`` signals (polled by the app shell every 5s), so newly
// completed runs in a live session appear here automatically. Each row exposes
// a compare checkbox feeding the Compare tray, the four headline metrics, and
// source/success badges. Clicking a row opens the run detail page.

import { html } from 'htm/preact';
import { useState, useMemo } from 'preact/hooks';
import { navigate } from '../lib/router.js';
import { runs, meta, compareSel, toggleCompare } from '../lib/state.js';
import { HEADLINE, fmtMetric, fmtInt } from '../lib/format.js';

// Sortable columns: label + source + success + the four headline metrics.
const SORT_COLS = [
  { key: 'label', label: 'Run', numeric: false },
  { key: 'source', label: 'Source', numeric: false },
  { key: 'success', label: 'Status', numeric: false },
  ...HEADLINE.map((h) => ({ key: h.tag, label: h.short, numeric: true, headline: true })),
];

function sortValue(run, key) {
  if (key === 'label') return run.label ?? run.id ?? '';
  if (key === 'source') return run.source ?? '';
  if (key === 'success') return run.success ? 1 : 0;
  return run.headline?.[key] ?? null;
}

function Badge({ kind, text }) {
  return html`<span class=${'badge badge-' + kind}>${text}</span>`;
}

export function Runs() {
  const allRuns = runs.value;
  const m = meta.value;
  const sel = compareSel.value;

  const [filter, setFilter] = useState('');
  const [sortKey, setSortKey] = useState('label');
  const [sortDir, setSortDir] = useState('asc');

  const rows = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    const filtered = allRuns.filter((r) => {
      if (!needle) return true;
      return (
        String(r.label ?? '').toLowerCase().includes(needle) ||
        String(r.id ?? '').toLowerCase().includes(needle) ||
        String(r.artifact_dir ?? '').toLowerCase().includes(needle) ||
        String(r.sweep_id ?? '').toLowerCase().includes(needle)
      );
    });
    const dir = sortDir === 'asc' ? 1 : -1;
    return [...filtered].sort((a, b) => {
      const av = sortValue(a, sortKey);
      const bv = sortValue(b, sortKey);
      if (typeof av === 'string' || typeof bv === 'string') {
        return String(av ?? '').localeCompare(String(bv ?? '')) * dir;
      }
      if (av == null && bv == null) return 0;
      if (av == null) return 1; // nulls last
      if (bv == null) return -1;
      return (av - bv) * dir;
    });
  }, [allRuns, filter, sortKey, sortDir]);

  function onSort(key, defaultDir) {
    if (key === sortKey) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir(defaultDir);
    }
  }

  const arrow = (key) => (key === sortKey ? (sortDir === 'asc' ? ' ▲' : ' ▼') : '');

  return html`
    <div class="page">
      <div class="page-head">
        <h1 class="page-title">Runs</h1>
        <div class="meta-strip">
          <span class="meta-item"><span class="dim">runs</span> ${fmtInt(allRuns.length, '0')}</span>
          <span class="meta-item"><span class="dim">live</span> ${fmtInt(m?.session_runs, '0')}</span>
          <span class="meta-item" title=${m?.results_root ?? ''}>
            <span class="dim">root</span> <code>${m?.results_root ?? '(session only)'}</code>
          </span>
        </div>
      </div>

      <div class="toolbar">
        <input
          class="filter-input"
          type="text"
          placeholder="Filter by label, id, dir, sweep…"
          value=${filter}
          onInput=${(e) => setFilter(e.target.value)}
        />
        <label class="sort-select">
          <span class="dim">sort</span>
          <select
            value=${sortKey}
            onChange=${(e) => {
              const key = e.target.value;
              const col = SORT_COLS.find((c) => c.key === key);
              setSortKey(key);
              setSortDir(col && col.numeric ? 'desc' : 'asc');
            }}
          >
            ${SORT_COLS.map((c) => html`<option value=${c.key}>${c.label}</option>`)}
          </select>
        </label>
        <button
          class="btn"
          onClick=${() => setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))}
          title="Toggle sort direction"
        >
          ${sortDir === 'asc' ? '↑ asc' : '↓ desc'}
        </button>
      </div>

      ${allRuns.length === 0
        ? html`<div class="empty">No runs discovered yet. Complete a run or point the server at a results root.</div>`
        : html`
            <div class="table-scroll card">
              <table class="data-table runs-table">
                <thead>
                  <tr>
                    <th class="pick-col"></th>
                    ${SORT_COLS.map(
                      (c) => html`
                        <th
                          key=${c.key}
                          class=${'sortable' + (c.numeric ? ' num' : '')}
                          onClick=${() => onSort(c.key, c.numeric ? 'desc' : 'asc')}
                          title=${c.headline ? c.label : ''}
                        >
                          ${c.label}${arrow(c.key)}
                        </th>
                      `,
                    )}
                    <th>Artifacts</th>
                  </tr>
                </thead>
                <tbody>
                  ${rows.map((r) => {
                    const checked = sel.includes(r.id);
                    return html`
                      <tr
                        key=${r.id}
                        class=${'clickable' + (checked ? ' sel' : '')}
                        onClick=${() => navigate('/runs/' + encodeURIComponent(r.id))}
                      >
                        <td class="pick-col" onClick=${(e) => e.stopPropagation()}>
                          <input
                            type="checkbox"
                            checked=${checked}
                            title="Add to compare"
                            onChange=${() => toggleCompare(r.id)}
                          />
                        </td>
                        <td class="run-name">
                          <div class="run-label">${r.label ?? r.id}</div>
                          ${r.sweep_id &&
                          html`<div class="dim caption">sweep ${r.sweep_id}${r.trial != null ? ' · trial ' + r.trial : ''}</div>`}
                        </td>
                        <td>
                          <${Badge}
                            kind=${r.source === 'session' ? 'session' : 'disk'}
                            text=${r.source ?? 'disk'}
                          />
                        </td>
                        <td>
                          <${Badge} kind=${r.success ? 'ok' : 'fail'} text=${r.success ? 'ok' : 'fail'} />
                        </td>
                        ${HEADLINE.map(
                          (h) => html`<td class="num" key=${h.tag}>${fmtMetric(r.headline?.[h.tag])}</td>`,
                        )}
                        <td class="dim caption artifact-cell" title=${r.artifact_dir ?? ''}>
                          <code>${r.artifact_dir ?? '—'}</code>
                        </td>
                      </tr>
                    `;
                  })}
                </tbody>
              </table>
            </div>
          `}

      ${sel.length > 0 &&
      html`
        <div class="compare-tray">
          <span>${sel.length} run${sel.length === 1 ? '' : 's'} selected for comparison</span>
          <div class="tray-actions">
            <button class="btn" onClick=${() => (compareSel.value = [])}>Clear</button>
            <button
              class="btn btn-primary"
              disabled=${sel.length < 2}
              onClick=${() => navigate('/compare')}
            >
              Compare →
            </button>
          </div>
        </div>
      `}
    </div>
  `;
}
