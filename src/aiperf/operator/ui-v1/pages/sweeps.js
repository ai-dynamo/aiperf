import { html } from 'htm/preact';
import { useState, useEffect, useMemo } from 'preact/hooks';
import { api, poll } from '../lib/api.js';
import { sweeps } from '../lib/state.js';
import { navigate } from '../lib/router.js';
import { palette } from '../lib/theme.js';

const FILTERS = [
  { label: 'All', value: null },
  { label: 'Running', value: ['running', 'aggregating'] },
  { label: 'Completed', value: ['succeeded'] },
  { label: 'Failed', value: ['failed', 'partiallyfailed', 'cancelled'] },
];

export function Sweeps() {
  const [list, setList] = useState(sweeps.value);
  const [activeFilter, setActiveFilter] = useState(null);
  const [searchText, setSearchText] = useState('');

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

      <table class="data-table" data-testid="sweep-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Namespace</th>
            <th>Phase</th>
            <th>Progress</th>
            <th>Failed</th>
            <th>Variations</th>
            <th>Model</th>
            <th>Source</th>
            <th>Age</th>
          </tr>
        </thead>
        <tbody>
          ${filtered.map(s => html`
            <tr key=${`${s.namespace}/${s.name}`} onclick=${() => rowClick(s)} style="cursor: pointer">
              <td>${s.name}</td>
              <td class="text-dim">${s.namespace}</td>
              <td><${PhasePill} phase=${s.phase} /></td>
              <td>${s.completed_runs} / ${s.total_variations || '?'}</td>
              <td style=${`color: ${s.failed_runs > 0 ? palette.red : 'inherit'}`}>${s.failed_runs}</td>
              <td>${s.total_variations}</td>
              <td class="text-dim">${s.model ?? '—'}</td>
              <td><${SourceChip} source=${s.source} /></td>
              <td class="text-dim">${formatAge(s.age_seconds)}</td>
            </tr>
          `)}
        </tbody>
      </table>
    </div>
  `;
}

function PhasePill({ phase }) {
  const p = (phase ?? '').toLowerCase();
  let bg = palette.surface0;
  if (['running', 'aggregating'].includes(p)) bg = palette.blue ?? '#4ea1ff';
  else if (p === 'succeeded') bg = palette.green ?? '#4caf50';
  else if (['failed', 'cancelled', 'partiallyfailed'].includes(p)) bg = palette.red ?? '#e53935';
  return html`<span style=${`background:${bg};color:white;padding:2px 8px;border-radius:8px;font-size:11px`}>${phase ?? 'Unknown'}</span>`;
}

function SourceChip({ source }) {
  return html`<span class="text-dim" style="font-size:11px;padding:1px 6px;border:1px solid ${palette.surface0};border-radius:6px">${source}</span>`;
}

function formatAge(s) {
  if (s == null) return '—';
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s/60)}m`;
  if (s < 86400) return `${Math.floor(s/3600)}h`;
  return `${Math.floor(s/86400)}d`;
}
