// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { useState, useMemo, useEffect, useRef } from 'preact/hooks';
import { phaseColor, palette } from '../lib/theme.js';
import { fmtMilliseconds, fmtReqPerSecond } from '../lib/format.js';
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

function isTerminalSuccess(phase) {
  const p = (phase ?? '').toLowerCase();
  return p === 'completed' || p === 'succeeded';
}

/**
 * Key identifying runs whose throughputs may be compared against each other.
 *
 * Namespace plus target model is the coarsest grouping that is still defensible
 * -- runs of different models are not on a common scale at all. It does not
 * make every member of a group a controlled comparison (ISL/OSL and endpoint
 * can still differ), which is why the bar is a within-group relative hint and
 * never a cross-group one.
 */
function comparabilityKey(job) {
  // NUL separator: neither a namespace nor a model name may contain it, so
  // ("a b", "") and ("a", "b") cannot collide into one group.
  return (job.namespace ?? '') + '\u0000' + (job.model ?? '');
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

  // Per-group scale for the relative throughput bar.
  //
  // A bar is a ratio, and a ratio between two AIPerfJob runs only means
  // something when the runs are comparable. Each AIPerfJob is its own
  // experiment -- a different model, prompt length, or concurrency changes the
  // achievable req/s by orders of magnitude -- so a table-wide maximum drew a
  // 3%-full bar next to a full one for two benchmarks that were never
  // measuring the same thing, and read as "this run is 30x worse".
  //
  // Group by (namespace, model) instead: on /jobs that separates unrelated
  // workloads, and on the sweep children table every child shares both, so the
  // bar keeps doing its actual job of ranking one sweep's operating points.
  // Only terminal runs with a finite throughput are counted, matching the only
  // rows that draw a bar.
  const throughputScales = useMemo(() => {
    const scales = new Map();
    for (const j of (jobs ?? [])) {
      if (!isTerminalSuccess(j.phase)) continue;
      const val = finiteNumber(j.throughputRps);
      if (val == null) continue;
      const key = comparabilityKey(j);
      const entry = scales.get(key);
      if (entry) {
        entry.count += 1;
        if (val > entry.max) entry.max = val;
      } else {
        scales.set(key, { max: val, count: 1 });
      }
    }
    return scales;
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
    const pct = finiteNumber(job.progressPercent);
    if (pct == null) return html`<span class="text-dim">---</span>`;
    // Math.round carries 99.5 and above up to 100, so a job with a few
    // hundred requests still outstanding rendered a full bar and a "100%"
    // label. 100% is the one value a reader acts on -- it is the difference
    // between "wait" and "go look at the results" -- so it is reserved for a
    // reported progress that already reached it, and anything short of that
    // tops out at 99. Same rule, same reasoning, as components/phase-bar.js:24-33.
    // The same number is the bar's CSS width, so it is also clamped into
    // [0, 100] -- an over- or under-reported percent must not draw a bar
    // longer than its track or a negative one.
    const rounded = pct >= 100 ? 100 : Math.max(0, Math.min(99, Math.round(pct)));
    if ((job.phase ?? '').toLowerCase() !== 'running') {
      return html`<span class="progress-label">${rounded}%</span>`;
    }
    return html`
      <div class="progress-cell">
        <div class="progress-track">
          <div class="progress-fill" style=${'width: ' + rounded + '%'} />
        </div>
        <span class="progress-label">${rounded}%</span>
      </div>
    `;
  }

  // Feature 9: Throughput with inline relative bar.
  //
  // Everything downstream uses the *parsed* number, never the raw field.
  // `fmtNumber` (lib/format.js:45) rejects anything whose typeof is not
  // 'number' and returns its '---' fallback, so handing it a string-typed
  // throughput printed the cell as missing while `finiteNumber` had already
  // accepted the same value for sorting, for `throughputScales`, and for
  // drawing this row's bar. A present value must never render as absent.
  function renderThroughput(job) {
    const numericVal = finiteNumber(job.throughputRps);
    if (numericVal == null) return html`<span class="text-dim">---</span>`;

    // The bar is drawn only against peers it is legitimate to compare with:
    // the same namespace and model, terminal, and at least one other such run
    // (a group of one has nothing to be relative to and would always render
    // full, which reads as "fastest" rather than "only").
    const scale = throughputScales.get(comparabilityKey(job));
    const comparable = isTerminalSuccess(job.phase) && scale != null
      && scale.count > 1 && scale.max > 0;
    const pct = comparable ? Math.min(100, (numericVal / scale.max) * 100) : 0;
    const barTitle = comparable
      ? `${pct.toFixed(0)}% of the fastest of ${scale.count} completed ${job.model || 'model'} runs in ${job.namespace}`
      : '';

    return html`
      <div style="display: flex; align-items: center; justify-content: flex-end; gap: var(--space-2); min-width: 120px">
        ${comparable && html`
          <div
            title=${barTitle}
            style=${'flex: 1; height: 4px; background: ' + palette.surface0 + '; border-radius: 2px; overflow: hidden; min-width: 40px'}
          >
            <div
              style=${'height: 100%; width: ' + pct.toFixed(1) + '%; background: ' + palette.blue + '; border-radius: 2px; transition: width 0.3s'}
            />
          </div>
        `}
        <span style="white-space: nowrap; min-width: 60px; text-align: right">${fmtReqPerSecond(numericVal)} req/s</span>
      </div>
    `;
  }

  function renderLatency(job) {
    const val = finiteNumber(job.latencyP99Ms);
    if (val == null) return html`<span class="text-dim">---</span>`;
    return html`<span>${fmtMilliseconds(val)} ms</span>`;
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
              <div class="job-sweep-context">
                <a href=${`#/sweeps/${encodeURIComponent(job.namespace)}/${encodeURIComponent(job.sweepName)}`}
                   data-testid="job-row-sweep-link"
                   onclick=${e => { e.stopPropagation(); navigate(`/sweeps/${encodeURIComponent(job.namespace)}/${encodeURIComponent(job.sweepName)}`); e.preventDefault(); }}>
                  ↳ sweep: ${job.sweepName}
                </a>
                ${job.variationLabel && html`<div class="job-variation-context">${job.variationLabel}</div>`}
                ${job.trialIndex != null && html`<div class="job-variation-context">trial ${job.trialIndex}</div>`}
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
          <span style="font-size: var(--font-size-xs); opacity: 0.7">${pickerOpen ? '▲' : '▼'}</span>
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
