import { html } from 'htm/preact';
import { useState, useEffect, useMemo } from 'preact/hooks';
import { api, poll } from '../lib/api.js';
import { jobs } from '../lib/state.js';
import { navigate, query, setQuery } from '../lib/router.js';
import { palette } from '../lib/theme.js';
import { JobTable } from '../components/job-table.js';

const FILTERS = [
  { label: 'All', value: null },
  { label: 'Running', value: ['running', 'initializing'] },
  { label: 'Completed', value: ['completed', 'succeeded'] },
  { label: 'Failed', value: ['failed', 'error'] },
];

const PHASE_BY_KEY = Object.fromEntries(
  FILTERS.filter(f => f.value).map(f => [f.label.toLowerCase(), f.value])
);

function parseSort(s) {
  if (!s) return { key: 'age', dir: -1 };
  const [key, dir] = s.split(':');
  return { key: key || 'age', dir: dir === 'asc' ? 1 : -1 };
}

function formatSort(sort) {
  return `${sort.key}:${sort.dir === 1 ? 'asc' : 'desc'}`;
}

export function Jobs() {
  const [localJobs, setLocalJobs] = useState(jobs.value);

  // URL-driven filter state
  const q = query.value;
  const phaseKey = q.phase ?? null;
  const activeFilter = phaseKey ? (PHASE_BY_KEY[phaseKey] ?? null) : null;
  const ns = q.ns ?? '';
  const modelFilter = q.model ?? '';
  const endpointFilter = q.endpoint ?? '';
  const sort = parseSort(q.sort);

  // Search text is local; debounced into ?q= so typing doesn't spam history
  const urlQ = q.q ?? '';
  const [searchText, setSearchText] = useState(urlQ);
  useEffect(() => {
    if (searchText !== urlQ) setSearchText(urlQ);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [urlQ]);
  useEffect(() => {
    const t = setTimeout(() => {
      if (searchText !== urlQ) setQuery({ q: searchText });
    }, 200);
    return () => clearTimeout(t);
  }, [searchText, urlQ]);

  useEffect(() => {
    const ac = new AbortController();
    poll(
      async () => {
        const data = await api.listJobs();
        const list = data?.jobs ?? [];
        jobs.value = list;
        setLocalJobs(list);
      },
      5000,
      ac.signal,
    );
    return () => ac.abort();
  }, []);

  const models = useMemo(() => {
    const set = new Set(localJobs.map(j => j.model).filter(Boolean));
    return [...set].sort();
  }, [localJobs]);

  const endpoints = useMemo(() => {
    const set = new Set(localJobs.map(j => j.endpoint).filter(Boolean));
    return [...set].sort();
  }, [localJobs]);

  const filtered = useMemo(() => {
    let result = localJobs;
    if (activeFilter) {
      result = result.filter(j => activeFilter.includes((j.phase ?? '').toLowerCase()));
    }
    if (ns) {
      result = result.filter(j => (j.namespace ?? '') === ns);
    }
    if (searchText) {
      const qLower = searchText.toLowerCase();
      result = result.filter(j =>
        (j.name ?? '').toLowerCase().includes(qLower) ||
        (j.namespace ?? '').toLowerCase().includes(qLower),
      );
    }
    if (modelFilter) {
      result = result.filter(j => j.model === modelFilter);
    }
    if (endpointFilter) {
      result = result.filter(j => j.endpoint === endpointFilter);
    }
    return result;
  }, [localJobs, activeFilter, ns, searchText, modelFilter, endpointFilter]);

  function handleRowClick(job) {
    navigate('/jobs/' + encodeURIComponent(job.namespace ?? 'default') + '/' + encodeURIComponent(job.name ?? ''));
  }

  function clearFilters() {
    setSearchText('');
    setQuery({ q: undefined, ns: undefined, phase: undefined, model: undefined, endpoint: undefined });
  }

  const hasFilters = searchText || ns || modelFilter || endpointFilter || activeFilter;

  return html`
    <div class="jobs-page" data-testid="page-jobs">
      <div class="section-header">
        <div class="filter-tabs">
          ${FILTERS.map((f) => {
            const key = f.value ? f.label.toLowerCase() : null;
            const active = (phaseKey ?? null) === key;
            return html`
              <button
                key=${f.label}
                class=${'filter-tab' + (active ? ' filter-tab--active' : '')}
                onclick=${() => setQuery({ phase: key })}
              >
                ${f.label}
                ${f.value === null
                  ? html`<span class="filter-tab-count">${localJobs.length}</span>`
                  : html`<span class="filter-tab-count">
                      ${localJobs.filter((j) => f.value.includes((j.phase ?? '').toLowerCase())).length}
                    </span>`
                }
              </button>
            `;
          })}
        </div>
        <span class="text-dim" style="font-size: var(--font-size-sm)">
          ${filtered.length} of ${localJobs.length} job${localJobs.length !== 1 ? 's' : ''}
        </span>
      </div>

      <!-- Filter bar -->
      <div style="display: flex; gap: var(--space-3); margin-bottom: var(--space-4); flex-wrap: wrap; align-items: center">
        <input
          type="text"
          placeholder="Search name..."
          value=${searchText}
          oninput=${e => setSearchText(e.target.value)}
          style=${'flex: 1; min-width: 150px; padding: var(--space-2) var(--space-3); background: ' + palette.mantle + '; border: 1px solid ' + palette.surface0 + '; border-radius: var(--radius-md); color: ' + palette.text + '; font-size: var(--font-size-sm)'}
        />
        ${ns && html`
          <span
            class="meta-pill meta-pill--clickable"
            style=${'background:' + palette.teal + '22;color:' + palette.teal + ';border-color:' + palette.teal + '55'}
            title=${'Namespace filter: ' + ns + ' (click to clear)'}
            onclick=${() => setQuery({ ns: undefined })}
            data-testid="ns-filter-chip"
          >
            <span class="meta-pill__prefix">ns</span>${ns}
            <span style="margin-left:4px;opacity:0.7">×</span>
          </span>
        `}
        ${models.length > 1 && html`
          <select
            value=${modelFilter}
            onchange=${e => setQuery({ model: e.target.value })}
            style=${'padding: var(--space-2) var(--space-3); background: ' + palette.mantle + '; border: 1px solid ' + palette.surface0 + '; border-radius: var(--radius-md); color: ' + palette.text + '; font-size: var(--font-size-sm)'}
          >
            <option value="">All Models</option>
            ${models.map(m => html`<option key=${m} value=${m}>${m}</option>`)}
          </select>
        `}
        ${endpoints.length > 1 && html`
          <select
            value=${endpointFilter}
            onchange=${e => setQuery({ endpoint: e.target.value })}
            style=${'padding: var(--space-2) var(--space-3); background: ' + palette.mantle + '; border: 1px solid ' + palette.surface0 + '; border-radius: var(--radius-md); color: ' + palette.text + '; font-size: var(--font-size-sm)'}
          >
            <option value="">All Endpoints</option>
            ${endpoints.map(e => html`<option key=${e} value=${e}>${e}</option>`)}
          </select>
        `}
        ${hasFilters && html`
          <button
            onclick=${clearFilters}
            style=${'padding: var(--space-2) var(--space-3); background: transparent; border: 1px solid ' + palette.surface0 + '; border-radius: var(--radius-md); color: ' + palette.overlay0 + '; cursor: pointer; font-size: var(--font-size-sm)'}
          >
            Clear
          </button>
        `}
      </div>

      <${JobTable}
        jobs=${filtered}
        onRowClick=${handleRowClick}
        sort=${sort}
        onSortChange=${next => setQuery({ sort: formatSort(next) })}
        onNamespaceClick=${nsClicked => setQuery({ ns: nsClicked, q: undefined })}
      />
    </div>
  `;
}
