// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { useState, useMemo, useEffect, useRef } from 'preact/hooks';
import { phaseColor, palette } from '../lib/theme.js';
import { fmtNumber, fmtThroughput } from '../lib/format.js';
import { navigate } from '../lib/router.js';
import { NsPill } from './pills.js';
import { RelativeTime } from './time.js';

const COLUMNS = [
  { key: 'name', label: 'Name', alwaysVisible: true },
  { key: 'namespace', label: 'Namespace' },
  { key: 'phase', label: 'Phase' },
  { key: 'workers', label: 'Workers', numeric: true },
  { key: 'progress', label: 'Progress' },
  { key: 'throughput', label: 'Throughput', numeric: true },
  { key: 'latency', label: 'Latency', numeric: true },
  { key: 'age', label: 'Age' },
];

// localStorage key for hidden-column user preference. Shared across
// every JobTable instance so toggling on /jobs also affects the
// children table on /sweeps/<ns>/<name>; matches the way users
// expect "I hid latency, leave it hidden" to behave globally.
const HIDDEN_COLS_STORAGE_KEY = 'aiperf-ui-v1.job-table.hidden-cols';
const NUMERIC_SORT_KEYS = new Set(['workers', 'progress', 'throughput', 'latency', 'age']);

function finiteNumber(value) {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value !== 'string' || value.trim() === '') return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function loadHiddenCols() {
  if (typeof localStorage === 'undefined') return new Set();
  try {
    const raw = localStorage.getItem(HIDDEN_COLS_STORAGE_KEY);
    if (!raw) return new Set();
    const parsed = JSON.parse(raw);
    return new Set(Array.isArray(parsed) ? parsed : []);
  } catch {
    return new Set();
  }
}

function saveHiddenCols(set) {
  if (typeof localStorage === 'undefined') return;
  try {
    localStorage.setItem(HIDDEN_COLS_STORAGE_KEY, JSON.stringify([...set]));
  } catch { /* quota / private mode — silent */ }
}

// API returns AIPerfJobInfo with flat camelCase fields:
// name, namespace, phase, workersReady, workersTotal, progressPercent,
// throughputRps, latencyP99Ms, created, model, endpoint, currentPhase, error
function jobValue(job, key) {
  switch (key) {
    case 'name': return job.name ?? '';
    case 'namespace': return job.namespace ?? '';
    case 'phase': return job.phase ?? '';
    case 'workers': return job.workersTotal ?? null;
    case 'progress': return job.progressPercent ?? null;
    case 'throughput': return job.throughputRps ?? null;
    case 'latency': return job.latencyP99Ms ?? null;
    case 'age': return job.created ? new Date(job.created).getTime() : null;
    default: return '';
  }
}

export function JobTable({ jobs, onRowClick, filter, onNamespaceClick, sort, onSortChange }) {
  const [internalSort, setInternalSort] = useState({ key: 'age', dir: -1 });
  const [hoverCol, setHoverCol] = useState(null);
  const [hiddenCols, setHiddenCols] = useState(loadHiddenCols);
  const [pickerOpen, setPickerOpen] = useState(false);
  const pickerRef = useRef(null);
  const controlled = !!(sort && onSortChange);
  const activeSort = controlled ? sort : internalSort;
  const sortKey = activeSort.key;
  const sortDir = Number(activeSort.dir) || 1;

  // Persist column-visibility selection across navigations and reloads.
  useEffect(() => {
    saveHiddenCols(hiddenCols);
  }, [hiddenCols]);

  // Click-outside / Escape closes the picker. Only attached when open
  // so we're not running global listeners for every JobTable instance
  // when the picker isn't visible.
  useEffect(() => {
    if (!pickerOpen) return undefined;
    function onDocMouseDown(e) {
      if (pickerRef.current && !pickerRef.current.contains(e.target)) {
        setPickerOpen(false);
      }
    }
    function onKey(e) {
      if (e.key === 'Escape') setPickerOpen(false);
    }
    document.addEventListener('mousedown', onDocMouseDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDocMouseDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [pickerOpen]);

  function toggleColumn(key) {
    setHiddenCols((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  function showAllColumns() {
    setHiddenCols(new Set());
  }

  const visibleColumns = COLUMNS.filter((c) => c.alwaysVisible || !hiddenCols.has(c.key));
  const hiddenCount = COLUMNS.filter((c) => !c.alwaysVisible && hiddenCols.has(c.key)).length;

  function toggleSort(key) {
    const next = (sortKey === key)
      ? { key, dir: -sortDir }
      : { key, dir: 1 };
    if (controlled) onSortChange(next);
    else setInternalSort(next);
  }

  const filtered = filter && filter.length > 0
    ? (jobs ?? []).filter((j) => {
        const phase = (j.phase ?? '').toLowerCase();
        return filter.map((f) => f.toLowerCase()).includes(phase);
      })
    : (jobs ?? []);

  const sorted = [...filtered].sort((a, b) => {
    let av = jobValue(a, sortKey);
    let bv = jobValue(b, sortKey);
    if (NUMERIC_SORT_KEYS.has(sortKey)) {
      av = finiteNumber(av);
      bv = finiteNumber(bv);
    }
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    if (av < bv) return -sortDir;
    if (av > bv) return sortDir;
    return 0;
  });

  // Feature 9: Compute max throughput for relative bar sizing
  const maxThroughput = useMemo(() => {
    let max = 0;
    for (const j of (jobs ?? [])) {
      const val = finiteNumber(j.throughputRps) ?? 0;
      if (val > max) max = val;
    }
    return max;
  }, [jobs]);

  function renderSortIcon(key) {
    if (sortKey !== key) return html`<span class="sort-icon sort-icon--none">\u2195</span>`;
    return sortDir === 1
      ? html`<span class="sort-icon sort-icon--asc">\u2191</span>`
      : html`<span class="sort-icon sort-icon--desc">\u2193</span>`;
  }

  function renderPhase(phase) {
    const color = phaseColor(phase);
    return html`
      <span class="phase-badge" style=${'background: ' + color + '22; color: ' + color + '; border-color: ' + color + '44'}>
        ${phase || 'Unknown'}
      </span>
    `;
  }

  function renderProgress(job) {
    const pct = job.progressPercent;
    if (pct == null) return html`<span class="text-dim">---</span>`;
    const rounded = Math.round(pct);
    return html`
      <div class="progress-cell">
        <div class="progress-track">
          <div class="progress-fill" style=${'width: ' + rounded + '%'} />
        </div>
        <span class="progress-label">${rounded}%</span>
      </div>
    `;
  }

  // Feature 9: Throughput with inline relative bar
  function renderThroughput(job) {
    const val = job.throughputRps;
    const numericVal = finiteNumber(val);
    if (numericVal == null) return html`<span class="text-dim">---</span>`;

    const phase = (job.phase ?? '').toLowerCase();
    const isComplete = phase === 'completed' || phase === 'succeeded';
    const pct = maxThroughput > 0 ? (numericVal / maxThroughput) * 100 : 0;

    return html`
      <div style="display: flex; align-items: center; justify-content: flex-end; gap: var(--space-2); min-width: 120px">
        ${isComplete && maxThroughput > 0 && html`
          <div
            style=${'flex: 1; height: 4px; background: ' + palette.surface0 + '; border-radius: 2px; overflow: hidden; min-width: 40px'}
          >
            <div
              style=${'height: 100%; width: ' + pct.toFixed(1) + '%; background: ' + palette.blue + '; border-radius: 2px; transition: width 0.3s'}
            />
          </div>
        `}
        <span style="white-space: nowrap; min-width: 60px; text-align: right">${fmtThroughput(val)} req/s</span>
      </div>
    `;
  }

  function renderLatency(job) {
    const val = finiteNumber(job.latencyP99Ms);
    if (val == null) return html`<span class="text-dim">---</span>`;
    if (val > 1000) return html`<span>${fmtNumber(val / 1000, 1)} s</span>`;
    return html`<span>${fmtNumber(val, 0)} ms</span>`;
  }

  function renderWorkers(job) {
    const ready = job.workersReady ?? 0;
    const total = job.workersTotal ?? 0;
    if (total === 0) return html`<span class="text-dim">---</span>`;
    return html`<span>${ready}/${total}</span>`;
  }

  // Data-driven cell renderer. Keeping each branch in one place lets the
  // header (visibleColumns.map) and body share identical column ordering
  // and lets the column-picker hide arbitrary subsets without dropping
  // any <td> count vs <th> count.
  function renderCell(job, key) {
    switch (key) {
      case 'name':
        return html`
          <td key=${key} class="job-table-td job-table-name">
            ${job.name}
            ${job.sweepName && html`
              <div class="text-dim" style="font-size:11px;font-style:italic;margin-top:2px">
                <a href=${`#/sweeps/${encodeURIComponent(job.namespace)}/${encodeURIComponent(job.sweepName)}`}
                   data-testid="job-row-sweep-link"
                   onclick=${e => { e.stopPropagation(); navigate(`/sweeps/${encodeURIComponent(job.namespace)}/${encodeURIComponent(job.sweepName)}`); e.preventDefault(); }}>
                  ↳ sweep: ${job.sweepName}
                </a>
                ${job.variationLabel && html`<span> · ${job.variationLabel}</span>`}
                ${job.trialIndex != null && html`<span> · trial ${job.trialIndex}</span>`}
              </div>
            `}
          </td>
        `;
      case 'namespace':
        return html`
          <td key=${key} class="job-table-td">
            <${NsPill} ns=${job.namespace} onClick=${onNamespaceClick} testId=${'job-row-ns-' + (job.namespace ?? '')} />
          </td>
        `;
      case 'phase':
        return html`<td key=${key} class="job-table-td">${renderPhase(job.phase)}</td>`;
      case 'workers':
        return html`<td key=${key} class="job-table-td" style="text-align: right">${renderWorkers(job)}</td>`;
      case 'progress':
        return html`<td key=${key} class="job-table-td">${renderProgress(job)}</td>`;
      case 'throughput':
        return html`<td key=${key} class="job-table-td" style="text-align: right">${renderThroughput(job)}</td>`;
      case 'latency':
        return html`<td key=${key} class="job-table-td" style="text-align: right">${renderLatency(job)}</td>`;
      case 'age':
        return html`<td key=${key} class="job-table-td text-dim"><${RelativeTime} ts=${job.created} /></td>`;
      default:
        return html`<td key=${key} class="job-table-td"></td>`;
    }
  }

  // Picker is rendered even when the table itself is empty so users can
  // still adjust visibility before any data lands.
  function renderColumnPicker() {
    const togglable = COLUMNS.filter((c) => !c.alwaysVisible);
    return html`
      <div ref=${pickerRef} style="position: relative">
        <button
          type="button"
          onclick=${() => setPickerOpen((v) => !v)}
          data-testid="job-table-columns-btn"
          aria-haspopup="true"
          aria-expanded=${pickerOpen}
          title="Show or hide columns"
          style=${'display: inline-flex; align-items: center; gap: var(--space-2);'
            + ' padding: var(--space-2) var(--space-3);'
            + ' background: var(--bg-card); border: 1px solid '
            + (hiddenCount > 0 ? 'var(--accent)' : 'var(--border)') + ';'
            + ' border-radius: var(--radius-sm);'
            + ' color: ' + (hiddenCount > 0 ? 'var(--accent)' : 'var(--sub)') + ';'
            + ' font-size: var(--font-size-xs); cursor: pointer'}
        >
          Columns${hiddenCount > 0 ? ` (${hiddenCount} hidden)` : ''}
          <span style="font-size: 10px; opacity: 0.7">${pickerOpen ? '▲' : '▼'}</span>
        </button>
        ${pickerOpen && html`
          <div
            data-testid="job-table-columns-picker"
            style=${'position: absolute; top: calc(100% + 4px); right: 0;'
              + ' z-index: 50; min-width: 180px;'
              + ' background: var(--bg-card); border: 1px solid var(--border);'
              + ' border-radius: var(--radius); padding: var(--space-2);'
              + ' box-shadow: 0 8px 24px rgba(0,0,0,0.4);'
              + ' display: flex; flex-direction: column; gap: 2px'}
          >
            ${COLUMNS.map((col) => {
              const checked = col.alwaysVisible || !hiddenCols.has(col.key);
              const disabled = !!col.alwaysVisible;
              return html`
                <label
                  key=${col.key}
                  style=${'display: flex; align-items: center; gap: var(--space-2);'
                    + ' padding: var(--space-1) var(--space-2);'
                    + ' border-radius: var(--radius-sm);'
                    + ' cursor: ' + (disabled ? 'default' : 'pointer') + ';'
                    + ' color: var(--text); font-size: var(--font-size-sm);'
                    + ' opacity: ' + (disabled ? '0.6' : '1')}
                >
                  <input
                    type="checkbox"
                    checked=${checked}
                    disabled=${disabled}
                    onchange=${() => !disabled && toggleColumn(col.key)}
                    style="accent-color: var(--accent)"
                  />
                  <span>${col.label}</span>
                  ${disabled && html`<span class="text-dim" style="font-size: var(--font-size-xs); margin-left: auto">required</span>`}
                </label>
              `;
            })}
            ${hiddenCount > 0 && html`
              <button
                type="button"
                onclick=${showAllColumns}
                data-testid="job-table-columns-reset"
                style=${'margin-top: var(--space-2); padding: var(--space-1) var(--space-2);'
                  + ' background: transparent; border: 1px solid var(--border);'
                  + ' border-radius: var(--radius-sm); color: var(--sub);'
                  + ' font-size: var(--font-size-xs); cursor: pointer'}
              >
                Show all columns
              </button>
            `}
          </div>
        `}
      </div>
    `;
  }

  if (sorted.length === 0) {
    return html`
      <div>
        <div style="display: flex; justify-content: flex-end; margin-bottom: var(--space-2)">
          ${renderColumnPicker()}
        </div>
        <div class="job-table-empty"><p>No jobs found</p></div>
      </div>
    `;
  }

  return html`
    <div>
      <div style="display: flex; justify-content: flex-end; margin-bottom: var(--space-2)">
        ${renderColumnPicker()}
      </div>
      <div class="job-table-wrapper">
        <table class="job-table">
          <thead>
            <tr>
              ${visibleColumns.map(
                (col) => {
                  const isHover = hoverCol === col.key;
                  const thStyle = [
                    'cursor: pointer',
                    'user-select: none',
                    col.numeric ? 'text-align: right' : '',
                    isHover ? 'background: rgba(255,255,255,0.06)' : '',
                  ].filter(Boolean).join('; ');
                  return html`
                  <th
                    key=${col.key}
                    class="job-table-th"
                    role="columnheader"
                    tabindex="0"
                    onkeydown=${(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleSort(col.key); } }}
                    onclick=${() => toggleSort(col.key)}
                    onmouseenter=${() => setHoverCol(col.key)}
                    onmouseleave=${() => setHoverCol(null)}
                    style=${thStyle}
                    data-testid=${'col-header-' + col.key}
                  >
                    ${col.label} ${renderSortIcon(col.key)}
                  </th>
                `;
                },
              )}
            </tr>
          </thead>
          <tbody data-testid="job-table">
            ${sorted.map((job) => html`
              <tr
                key=${job.namespace + '/' + job.name}
                class="job-table-row"
                role="row"
                tabindex=${onRowClick ? '0' : undefined}
                onkeydown=${(e) => { if (onRowClick && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); onRowClick(job); } }}
                onclick=${() => onRowClick && onRowClick(job)}
                style=${onRowClick ? 'cursor: pointer' : ''}
                data-testid=${'job-row-' + (job.namespace ?? '') + '-' + (job.name ?? '')}
              >
                ${visibleColumns.map((col) => renderCell(job, col.key))}
              </tr>
            `)}
          </tbody>
        </table>
      </div>
    </div>
  `;
}
