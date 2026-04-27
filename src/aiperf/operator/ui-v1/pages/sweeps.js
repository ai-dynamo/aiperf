import { html } from 'htm/preact';
import { useState, useEffect, useMemo } from 'preact/hooks';
import { api, poll } from '../lib/api.js';
import { sweeps } from '../lib/state.js';
import { navigate } from '../lib/router.js';
import { palette, phaseColor } from '../lib/theme.js';

const FILTERS = [
  { label: 'All', value: null },
  { label: 'Running', value: ['running', 'aggregating'] },
  { label: 'Completed', value: ['succeeded'] },
  { label: 'Failed', value: ['failed', 'partiallyfailed', 'cancelled'] },
];

const COLUMNS = [
  { key: 'name', label: 'Name' },
  { key: 'namespace', label: 'Namespace' },
  { key: 'phase', label: 'Phase' },
  { key: 'progress', label: 'Progress' },
  { key: 'failed', label: 'Failed' },
  { key: 'variations', label: 'Variations' },
  { key: 'model', label: 'Model' },
  { key: 'source', label: 'Source' },
  { key: 'age', label: 'Age' },
  { key: 'epochs', label: '' },
];

function sweepValue(s, key) {
  switch (key) {
    case 'name': return s.name ?? '';
    case 'namespace': return s.namespace ?? '';
    case 'phase': return s.phase ?? '';
    case 'progress': return s.completed_runs ?? 0;
    case 'failed': return s.failed_runs ?? 0;
    case 'variations': return s.total_variations ?? 0;
    case 'model': return s.model ?? '';
    case 'source': return s.source ?? '';
    case 'age': return s.age_seconds ?? 0;
    default: return '';
  }
}

function formatAge(s) {
  if (s == null) return '---';
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s/60)}m`;
  if (s < 86400) return `${Math.floor(s/3600)}h`;
  return `${Math.floor(s/86400)}d`;
}

function renderPhase(phase) {
  const color = phaseColor(phase);
  return html`
    <span class="phase-badge" style=${'background: ' + color + '22; color: ' + color + '; border-color: ' + color + '44'}>
      ${phase || 'Unknown'}
    </span>
  `;
}

function renderSource(source) {
  return html`<span class="text-dim" style=${`font-size:11px;padding:1px 6px;border:1px solid ${palette.surface0};border-radius:6px`}>${source}</span>`;
}

export function Sweeps() {
  const [list, setList] = useState(sweeps.value);
  const [activeFilter, setActiveFilter] = useState(null);
  const [searchText, setSearchText] = useState('');
  const [sortKey, setSortKey] = useState('age');
  const [sortDir, setSortDir] = useState(-1);

  useEffect(() => {
    const ac = new AbortController();
    poll(async () => {
      const data = await api.listSweeps();
      const next = data?.sweeps ?? [];
      sweeps.value = next;
      setList(next);
    }, 5000, ac.signal);
    return () => ac.abort();
  }, []);

  function toggleSort(key) {
    if (!key) return;
    if (sortKey === key) setSortDir(d => -d);
    else { setSortKey(key); setSortDir(1); }
  }

  function renderSortIcon(key) {
    if (!key) return null;
    if (sortKey !== key) return html`<span class="sort-icon sort-icon--none">↕</span>`;
    return sortDir === 1
      ? html`<span class="sort-icon sort-icon--asc">↑</span>`
      : html`<span class="sort-icon sort-icon--desc">↓</span>`;
  }

  const filtered = useMemo(() => {
    let r = list;
    if (activeFilter) r = r.filter(s => activeFilter.includes((s.phase ?? '').toLowerCase()));
    if (searchText) {
      const q = searchText.toLowerCase();
      r = r.filter(s =>
        (s.name ?? '').toLowerCase().includes(q) ||
        (s.namespace ?? '').toLowerCase().includes(q)
      );
    }
    return r;
  }, [list, activeFilter, searchText]);

  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      const av = sweepValue(a, sortKey);
      const bv = sweepValue(b, sortKey);
      if (av < bv) return -sortDir;
      if (av > bv) return sortDir;
      return 0;
    });
  }, [filtered, sortKey, sortDir]);

  function rowClick(s) {
    navigate(`/sweeps/${encodeURIComponent(s.namespace)}/${encodeURIComponent(s.name)}`);
  }

  return html`
    <div class="sweeps-page" data-testid="page-sweeps">
      <div class="section-header">
        <div class="filter-tabs">
          ${FILTERS.map(f => html`
            <button
              key=${f.label}
              class=${'filter-tab' + (activeFilter === f.value ? ' filter-tab--active' : '')}
              onclick=${() => setActiveFilter(f.value)}
            >
              ${f.label}
              ${f.value === null
                ? html`<span class="filter-tab-count">${list.length}</span>`
                : html`<span class="filter-tab-count">
                    ${list.filter(s => f.value.includes((s.phase ?? '').toLowerCase())).length}
                  </span>`}
            </button>
          `)}
        </div>
        <span class="text-dim" style="font-size: var(--font-size-sm)">
          ${filtered.length} of ${list.length} sweep${list.length !== 1 ? 's' : ''}
        </span>
      </div>

      <div style="display: flex; gap: var(--space-3); margin-bottom: var(--space-4); flex-wrap: wrap; align-items: center">
        <input
          type="text"
          placeholder="Search name or namespace..."
          value=${searchText}
          oninput=${e => setSearchText(e.target.value)}
          style=${`flex: 1; min-width: 150px; padding: var(--space-2) var(--space-3);
                   background: ${palette.mantle}; border: 1px solid ${palette.surface0};
                   border-radius: var(--radius-md); color: ${palette.text};
                   font-size: var(--font-size-sm)`}
        />
      </div>

      ${sorted.length === 0
        ? html`<div class="job-table-empty"><p>No sweeps found</p></div>`
        : html`
          <div class="job-table-wrapper">
            <table class="job-table" data-testid="sweep-table">
              <thead>
                <tr>
                  ${COLUMNS.map(col => html`
                    <th key=${col.key}
                        class="job-table-th"
                        onclick=${() => toggleSort(col.key)}
                        data-testid=${'col-header-' + col.key}>
                      ${col.label} ${renderSortIcon(col.key)}
                    </th>
                  `)}
                </tr>
              </thead>
              <tbody>
                ${sorted.map(s => {
                  const detailUrl = `/sweeps/${encodeURIComponent(s.namespace)}/${encodeURIComponent(s.name)}`;
                  return html`
                    <tr key=${`${s.namespace}/${s.name}`}
                        class="job-table-row"
                        onclick=${() => rowClick(s)}
                        style="cursor: pointer"
                        data-testid=${'sweep-row-' + (s.namespace ?? '') + '-' + (s.name ?? '')}>
                      <td class="job-table-td job-table-name">${s.name}</td>
                      <td class="job-table-td text-dim">${s.namespace}</td>
                      <td class="job-table-td">${renderPhase(s.phase)}</td>
                      <td class="job-table-td">${s.completed_runs} / ${s.total_variations || '?'}</td>
                      <td class="job-table-td"
                          style=${s.failed_runs > 0 ? `color:${palette.red}` : ''}>
                        ${s.failed_runs}
                      </td>
                      <td class="job-table-td">${s.total_variations}</td>
                      <td class="job-table-td text-dim">${s.model ?? '---'}</td>
                      <td class="job-table-td">${renderSource(s.source)}</td>
                      <td class="job-table-td text-dim">${formatAge(s.age_seconds)}</td>
                      <td class="job-table-td">
                        <a href=${`#${detailUrl}`}
                           title="View run history"
                           onclick=${e => { e.stopPropagation(); navigate(detailUrl); e.preventDefault(); }}
                           style=${`color:${palette.overlay0};text-decoration:none`}>↻</a>
                      </td>
                    </tr>
                  `;
                })}
              </tbody>
            </table>
          </div>
        `}
    </div>
  `;
}
