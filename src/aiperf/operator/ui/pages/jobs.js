// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { useState, useEffect, useMemo } from 'preact/hooks';
import { api, poll } from '../lib/api.js';
import { jobs, dedupeByNsName, freshness } from '../lib/state.js';
import { buildJobPath, navigate, query, setQuery } from '../lib/router.js';
import { palette } from '../lib/theme.js';
import { JobTable } from '../components/job-table.js';
import { FreshnessPill, StaleBanner } from '../components/freshness.js';
import { LoadingPanel } from '../components/spinner.js';

const FILTERS = [
  { label: 'All', value: null },
  { label: 'Running', value: ['running', 'initializing'] },
  { label: 'Completed', value: ['completed', 'succeeded'] },
  { label: 'Failed', value: ['failed', 'error', 'cancelled'] },
];

const PHASE_BY_KEY = Object.fromEntries(
  FILTERS.filter(f => f.value).map(f => [f.label.toLowerCase(), f.value])
);

function parseSort(s) {
  if (!s) return { key: 'age', dir: -1 };
  const [key, dir] = s.split(':');
  return { key: key || 'age', dir: dir === 'asc' ? 1 : -1 };
}

// Tighten generic API errors into something actionable. The api lib throws
// `API <status>: <body>` on HTTP errors and a TypeError-shaped string on
// network failures; we only re-shape when we recognize the pattern.
function describeLoadError(raw) {
  const s = String(raw ?? '');
  if (/API 404/.test(s)) return 'jobs endpoint not found — operator API may be older than this UI build';
  if (/API 401|API 403/.test(s)) return 'no permission to list jobs — check RBAC for the operator service account';
  if (/API 503|API 502|API 504/.test(s)) return 'operator unreachable — try `kubectl -n aiperf-operator get pods`';
  if (/Failed to fetch|NetworkError|ECONNREFUSED/i.test(s)) return 'network error reaching operator API — port-forward may have dropped';
  return s;
}

function formatSort(sort) {
  return `${sort.key}:${sort.dir === 1 ? 'asc' : 'desc'}`;
}

export function Jobs() {
  const [localJobs, setLocalJobs] = useState(jobs.value);
  // ``jobs`` is a global signal — when the user navigates here from
  // another page that already populated it, skip the loading skeleton.
  // Otherwise show one until the first fetch resolves so the empty state
  // can't be confused with "no jobs exist".
  const [firstLoad, setFirstLoad] = useState(jobs.value.length === 0);
  const [loadError, setLoadError] = useState(null);
  // Last successful refresh timestamp — surfaced as "updated Ns ago" so
  // the user knows whether they're staring at fresh data or a hung poll.
  const [lastUpdated, setLastUpdated] = useState(null);
  const [tickNow, setTickNow] = useState(Date.now());
  const jobsFreshness = freshness.value.jobs ?? null;

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
        try {
          const data = await api.listJobs();
          const list = dedupeByNsName(data?.jobs ?? []);
          jobs.value = list;
          setLocalJobs(list);
          setLoadError(null);
          setLastUpdated(Date.now());
        } catch (err) {
          if (firstLoad) setLoadError(describeLoadError(err?.message ?? err));
          throw err;
        } finally {
          setFirstLoad(false);
        }
      },
      5000,
      ac.signal,
      { source: 'jobs' },
    );
    return () => ac.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Re-render the "updated Ns ago" label every second without re-fetching.
  useEffect(() => {
    const id = setInterval(() => setTickNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  const updatedAgo = lastUpdated
    ? Math.max(0, Math.round((tickNow - lastUpdated) / 1000))
    : null;

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
    navigate(buildJobPath(job));
  }

  function clearFilters() {
    setSearchText('');
    // Intentionally does NOT clear ?sort= — sort is a view preference, not a
    // filter, and resetting it on "Clear filters" was reported as surprising.
    setQuery({ q: undefined, ns: undefined, phase: undefined, model: undefined, endpoint: undefined });
  }

  // Keyboard-activated chip removal: Enter/Space match native button behavior.
  function chipKeyHandler(onActivate) {
    return (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        onActivate();
      }
    };
  }

  const hasFilters = searchText || ns || modelFilter || endpointFilter || activeFilter;

  return html`
    <div class="jobs-page" data-testid="page-jobs">
      <div class="section-header">
        <div class="filter-tabs" role="tablist" aria-label="Filter jobs by phase">
          ${FILTERS.map((f) => {
            const key = f.value ? f.label.toLowerCase() : null;
            const active = (phaseKey ?? null) === key;
            return html`
              <button type="button"
                key=${f.label}
                role="tab"
                aria-pressed=${active}
                aria-selected=${active}
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
        <span class="text-dim" style="font-size: var(--font-size-sm); display: inline-flex; align-items: center; gap: var(--space-2)" aria-live="polite" aria-atomic="true">
          <span>${filtered.length} of ${localJobs.length} job${localJobs.length !== 1 ? 's' : ''}</span>
          ${updatedAgo != null && html`
            <span
              class="text-dim"
              style="font-size: var(--font-size-xs); opacity: 0.75"
              title=${'Auto-refreshes every 5s · last fetch ' + new Date(lastUpdated).toLocaleTimeString()}
              data-testid="jobs-last-updated"
            >· updated ${updatedAgo}s ago</span>
          `}
          ${jobsFreshness && html`<${FreshnessPill} source=${jobsFreshness} compact=${true} />`}
        </span>
      </div>

      <${StaleBanner} source=${jobsFreshness} label="Jobs list" />

      <!-- Filter bar -->
      <div class="jobs-filter-bar">
        <div class="jobs-search-field">
          <input
            type="text"
            placeholder="Search name..."
            aria-label="Search jobs by name or namespace"
            value=${searchText}
            oninput=${e => setSearchText(e.target.value)}
            onkeydown=${e => { if (e.key === 'Enter' && searchText !== urlQ) { e.preventDefault(); setQuery({ q: searchText }); } }}
            style=${'width: 100%; padding: var(--space-2) ' + (searchText ? '28px' : 'var(--space-3)') + ' var(--space-2) var(--space-3); background: ' + palette.mantle + '; border: 1px solid ' + palette.surface0 + '; border-radius: var(--radius-md); color: ' + palette.text + '; font-size: var(--font-size-sm)'}
          />
          ${searchText && html`
            <button
              type="button"
              aria-label="Clear search"
              onclick=${() => setSearchText('')}
              data-testid="search-clear"
              style=${'position: absolute; right: 6px; background: transparent; border: 0; color: ' + palette.overlay0 + '; cursor: pointer; font-size: var(--font-size-base); line-height: 1; padding: 2px 6px'}
            >×</button>
          `}
        </div>
        ${ns && html`
          <span
            class="meta-pill meta-pill--clickable"
            role="button"
            tabindex="0"
            aria-label=${'Remove namespace filter ' + ns}
            style=${'background:' + palette.teal + '22;color:' + palette.teal + ';border-color:' + palette.teal + '55'}
            title=${'Namespace filter: ' + ns + ' (click to clear)'}
            onclick=${() => setQuery({ ns: undefined })}
            onkeydown=${chipKeyHandler(() => setQuery({ ns: undefined }))}
            data-testid="ns-filter-chip"
          >
            <span class="meta-pill__prefix">ns</span>${ns}
            <span style="margin-left:4px;opacity:0.7" aria-hidden="true">×</span>
          </span>
        `}
        ${modelFilter && html`
          <span
            class="meta-pill meta-pill--clickable"
            role="button"
            tabindex="0"
            aria-label=${'Remove model filter ' + modelFilter}
            style=${'background:' + palette.mauve + '22;color:' + palette.mauve + ';border-color:' + palette.mauve + '55'}
            title=${'Model filter: ' + modelFilter + ' (click to clear)'}
            onclick=${() => setQuery({ model: undefined })}
            onkeydown=${chipKeyHandler(() => setQuery({ model: undefined }))}
            data-testid="model-filter-chip"
          >
            <span class="meta-pill__prefix">model</span>${modelFilter}
            <span style="margin-left:4px;opacity:0.7" aria-hidden="true">×</span>
          </span>
        `}
        ${endpointFilter && html`
          <span
            class="meta-pill meta-pill--clickable"
            role="button"
            tabindex="0"
            aria-label=${'Remove endpoint filter ' + endpointFilter}
            style=${'background:' + palette.peach + '22;color:' + palette.peach + ';border-color:' + palette.peach + '55'}
            title=${'Endpoint filter: ' + endpointFilter + ' (click to clear)'}
            onclick=${() => setQuery({ endpoint: undefined })}
            onkeydown=${chipKeyHandler(() => setQuery({ endpoint: undefined }))}
            data-testid="endpoint-filter-chip"
          >
            <span class="meta-pill__prefix">endpoint</span>${endpointFilter}
            <span style="margin-left:4px;opacity:0.7" aria-hidden="true">×</span>
          </span>
        `}
        ${models.length > 1 && html`
          <select
            class="ui-select"
            value=${modelFilter}
            onchange=${e => setQuery({ model: e.target.value })}
          >
            <option value="">All Models</option>
            ${models.map(m => html`<option key=${m} value=${m}>${m}</option>`)}
          </select>
        `}
        ${endpoints.length > 1 && html`
          <select
            class="ui-select"
            value=${endpointFilter}
            onchange=${e => setQuery({ endpoint: e.target.value })}
          >
            <option value="">All Endpoints</option>
            ${endpoints.map(e => html`<option key=${e} value=${e}>${e}</option>`)}
          </select>
        `}
        ${hasFilters && html`
          <button type="button"
            onclick=${clearFilters}
            class="jobs-filter-clear"
          >
            Clear
          </button>
        `}
      </div>

      ${firstLoad && html`
        <div class="card">
          <${LoadingPanel} label="Loading jobs…" testid="jobs-loading" />
        </div>
      `}

      ${loadError && html`
        <div class="card" style="border-color: var(--error); color: var(--error)" data-testid="jobs-error">
          Failed to load jobs: ${loadError}
        </div>
      `}

      ${!firstLoad && !loadError && filtered.length === 0 && localJobs.length === 0 && html`
        <div class="card" data-testid="jobs-empty-real" style="text-align: center; padding: var(--space-6)">
          <p style=${'color:' + palette.text + ';margin:0 0 var(--space-2) 0'}>No jobs yet.</p>
          <p class="text-dim" style="margin:0;font-size:var(--font-size-sm)">
            Create one with <code>aiperf kube apply -f job.yaml</code> or <code>aiperf kube generate</code>.
          </p>
        </div>
      `}

      ${!firstLoad && !loadError && filtered.length === 0 && localJobs.length > 0 && html`
        <div class="card" data-testid="jobs-empty-filtered" style="text-align: center; padding: var(--space-6)">
          <p style=${'color:' + palette.text + ';margin:0 0 var(--space-3) 0'}>No jobs match these filters.</p>
          <button type="button"
            onclick=${clearFilters}
            style=${'padding: var(--space-2) var(--space-4); background: ' + palette.surface0 + '; border: 1px solid ' + palette.surface1 + '; border-radius: var(--radius-md); color: ' + palette.text + '; cursor: pointer; font-size: var(--font-size-sm)'}
          >
            Clear filters
          </button>
        </div>
      `}

      ${!firstLoad && filtered.length > 0 && html`<${JobTable}
        jobs=${filtered}
        onRowClick=${handleRowClick}
        sort=${sort}
        onSortChange=${next => setQuery({ sort: formatSort(next) })}
        onNamespaceClick=${nsClicked => setQuery({ ns: nsClicked, q: undefined })}
      />`}
    </div>
  `;
}
