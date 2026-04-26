import { html } from 'htm/preact';
import { useState, useEffect, useMemo } from 'preact/hooks';
import { api, poll } from '../lib/api.js';
import { jobs } from '../lib/state.js';
import { navigate } from '../lib/router.js';
import { palette } from '../lib/theme.js';
import { JobTable } from '../components/job-table.js';

const FILTERS = [
  { label: 'All', value: null },
  { label: 'Running', value: ['running', 'initializing'] },
  { label: 'Completed', value: ['completed', 'succeeded'] },
  { label: 'Failed', value: ['failed', 'error'] },
  { label: 'Archived', value: 'archived', source: true },
];

export function Jobs() {
  const [localJobs, setLocalJobs] = useState(jobs.value);
  const [activeFilter, setActiveFilter] = useState(null);
  const [searchText, setSearchText] = useState('');
  const [modelFilter, setModelFilter] = useState('');
  const [endpointFilter, setEndpointFilter] = useState('');

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

  // Extract unique models and endpoints for filter dropdowns
  const models = useMemo(() => {
    const set = new Set(localJobs.map(j => j.model).filter(Boolean));
    return [...set].sort();
  }, [localJobs]);

  const endpoints = useMemo(() => {
    const set = new Set(localJobs.map(j => j.endpoint).filter(Boolean));
    return [...set].sort();
  }, [localJobs]);

  // Apply all filters
  const filtered = useMemo(() => {
    let result = localJobs;

    // Phase filter
    if (activeFilter) {
      if (activeFilter === 'archived') {
        result = result.filter(j => j.source === 'archived');
      } else {
        result = result.filter(j => activeFilter.includes((j.phase ?? '').toLowerCase()));
      }
    }

    // Text search (name, namespace)
    if (searchText) {
      const q = searchText.toLowerCase();
      result = result.filter(j =>
        (j.name ?? '').toLowerCase().includes(q) ||
        (j.namespace ?? '').toLowerCase().includes(q),
      );
    }

    // Model filter
    if (modelFilter) {
      result = result.filter(j => j.model === modelFilter);
    }

    // Endpoint filter
    if (endpointFilter) {
      result = result.filter(j => j.endpoint === endpointFilter);
    }

    return result;
  }, [localJobs, activeFilter, searchText, modelFilter, endpointFilter]);

  function handleRowClick(job) {
    navigate('/jobs/' + encodeURIComponent(job.namespace ?? 'default') + '/' + encodeURIComponent(job.name ?? ''));
  }

  function clearFilters() {
    setSearchText('');
    setModelFilter('');
    setEndpointFilter('');
    setActiveFilter(null);
  }

  const hasFilters = searchText || modelFilter || endpointFilter || activeFilter;

  return html`
    <div class="jobs-page" data-testid="page-jobs">
      <div class="section-header">
        <div class="filter-tabs">
          ${FILTERS.map((f) => html`
            <button
              key=${f.label}
              class=${'filter-tab' + (activeFilter === f.value ? ' filter-tab--active' : '')}
              onclick=${() => setActiveFilter(f.value)}
            >
              ${f.label}
              ${f.value === null
                ? html`<span class="filter-tab-count">${localJobs.length}</span>`
                : f.source
                ? html`<span class="filter-tab-count">
                    ${localJobs.filter((j) => j.source === f.value).length}
                  </span>`
                : html`<span class="filter-tab-count">
                    ${localJobs.filter((j) => f.value.includes((j.phase ?? '').toLowerCase())).length}
                  </span>`
              }
            </button>
          `)}
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
        ${models.length > 1 && html`
          <select
            value=${modelFilter}
            onchange=${e => setModelFilter(e.target.value)}
            style=${'padding: var(--space-2) var(--space-3); background: ' + palette.mantle + '; border: 1px solid ' + palette.surface0 + '; border-radius: var(--radius-md); color: ' + palette.text + '; font-size: var(--font-size-sm)'}
          >
            <option value="">All Models</option>
            ${models.map(m => html`<option key=${m} value=${m}>${m}</option>`)}
          </select>
        `}
        ${endpoints.length > 1 && html`
          <select
            value=${endpointFilter}
            onchange=${e => setEndpointFilter(e.target.value)}
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
      />
    </div>
  `;
}
