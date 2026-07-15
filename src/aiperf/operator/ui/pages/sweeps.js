// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { useState, useEffect, useMemo } from 'preact/hooks';
import { api, poll } from '../lib/api.js';
import { sweeps, dedupeByNsName, freshness } from '../lib/state.js';
import { navigate, query, setQuery } from '../lib/router.js';
import { palette, phaseColor } from '../lib/theme.js';
import { NsPill, ModelPill } from '../components/pills.js';
import { RelativeTime } from '../components/time.js';
import { FreshnessPill, StaleBanner } from '../components/freshness.js';
import { LoadingPanel } from '../components/spinner.js';

const FILTERS = [
  { label: 'All', value: null },
  { label: 'Running', value: ['running', 'aggregating'] },
  { label: 'Completed', value: ['succeeded'] },
  { label: 'Failed', value: ['failed', 'partiallyfailed', 'cancelled'] },
];

const PHASE_BY_KEY = Object.fromEntries(
  FILTERS.filter(f => f.value).map(f => [f.label.toLowerCase(), f.value])
);

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
const NUMERIC_SORT_KEYS = new Set(['progress', 'failed', 'variations', 'age']);

function finiteNumber(value) {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value !== 'string' || value.trim() === '') return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function sweepAlias(s, snakeKey, camelKey) {
  return s[snakeKey] ?? s[camelKey];
}

function sweepValue(s, key) {
  switch (key) {
    case 'name': return s.name ?? '';
    case 'namespace': return s.namespace ?? '';
    case 'phase': return s.phase ?? '';
    case 'progress': return sweepAlias(s, 'completed_runs', 'completedRuns') ?? 0;
    case 'failed': return sweepAlias(s, 'failed_runs', 'failedRuns') ?? 0;
    case 'variations': return sweepAlias(s, 'total_variations', 'totalVariations') ?? 0;
    case 'model': return s.model ?? '';
    case 'source': return s.source ?? '';
    // Negate so descending sort (dir=-1, the default) yields newest first.
    // The API only ships ``age_seconds`` (seconds-since-creation), where
    // *smaller* = newer; without the flip, dir=-1 would list oldest first
    // and disagree with the JobTable's ``case 'age'`` which keys on a
    // monotonically-increasing ``created`` timestamp.
    case 'age': return -(s.age_seconds ?? s.ageSeconds ?? 0);
    default: return '';
  }
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

// Tighten generic API errors into something actionable. The api lib throws
// `API <status>: <body>` on HTTP errors; we only re-shape when we recognize
// the pattern, so unknown errors still surface verbatim.
function describeLoadError(raw) {
  const s = String(raw ?? '');
  if (/API 404/.test(s)) return 'sweeps endpoint not found — operator API may be older than this UI build';
  if (/API 401|API 403/.test(s)) return 'no permission to list sweeps — check RBAC for the operator service account';
  if (/API 503|API 502|API 504/.test(s)) return 'operator unreachable — try `kubectl -n aiperf-operator get pods`';
  if (/Failed to fetch|NetworkError|ECONNREFUSED/i.test(s)) return 'network error reaching operator API — port-forward may have dropped';
  return s;
}

function parseSort(s) {
  if (!s) return { key: 'age', dir: -1 };
  const [key, dir] = s.split(':');
  return { key: key || 'age', dir: dir === 'asc' ? 1 : -1 };
}

function formatSort(sort) {
  return `${sort.key}:${sort.dir === 1 ? 'asc' : 'desc'}`;
}

export function Sweeps() {
  const [list, setList] = useState(sweeps.value);
  // ``sweeps`` is a global signal; show the loading skeleton only when
  // we haven't fetched yet (no other page populated it). Otherwise the
  // empty list reads as "no sweeps found" instead of "still fetching".
  const [firstLoad, setFirstLoad] = useState(sweeps.value.length === 0);
  const [loadError, setLoadError] = useState(null);
  // Last successful refresh — surfaced as "updated Ns ago" so users know
  // whether they're looking at a fresh snapshot or a stalled poll.
  const [lastUpdated, setLastUpdated] = useState(null);
  const [tickNow, setTickNow] = useState(Date.now());
  const sweepsFreshness = freshness.value.sweeps ?? null;

  const q = query.value;
  const phaseKey = q.phase ?? null;
  const activeFilter = phaseKey ? (PHASE_BY_KEY[phaseKey] ?? null) : null;
  const ns = q.ns ?? '';
  const sort = parseSort(q.sort);

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
    poll(async () => {
      try {
        const data = await api.listSweeps();
        const next = dedupeByNsName(data?.sweeps ?? []);
        sweeps.value = next;
        setList(next);
        setLoadError(null);
        setLastUpdated(Date.now());
      } catch (err) {
        if (firstLoad) setLoadError(describeLoadError(err?.message ?? err));
        throw err;
      } finally {
        setFirstLoad(false);
      }
    }, 5000, ac.signal, { source: 'sweeps' });
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

  function toggleSort(key) {
    if (!key) return;
    const next = (sort.key === key)
      ? { key, dir: -sort.dir }
      : { key, dir: 1 };
    setQuery({ sort: formatSort(next) });
  }

  function renderSortIcon(key) {
    if (!key) return null;
    if (sort.key !== key) return html`<span class="sort-icon sort-icon--none">↕</span>`;
    return sort.dir === 1
      ? html`<span class="sort-icon sort-icon--asc">↑</span>`
      : html`<span class="sort-icon sort-icon--desc">↓</span>`;
  }

  const filtered = useMemo(() => {
    let r = list;
    if (activeFilter) r = r.filter(s => activeFilter.includes((s.phase ?? '').toLowerCase()));
    if (ns) r = r.filter(s => (s.namespace ?? '') === ns);
    if (searchText) {
      const qLower = searchText.toLowerCase();
      r = r.filter(s =>
        (s.name ?? '').toLowerCase().includes(qLower) ||
        (s.namespace ?? '').toLowerCase().includes(qLower) ||
        (s.model ?? '').toLowerCase().includes(qLower)
      );
    }
    return r;
  }, [list, activeFilter, ns, searchText]);

  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      let av = sweepValue(a, sort.key);
      let bv = sweepValue(b, sort.key);
      if (NUMERIC_SORT_KEYS.has(sort.key)) {
        av = finiteNumber(av);
        bv = finiteNumber(bv);
      }
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (av < bv) return -sort.dir;
      if (av > bv) return sort.dir;
      return 0;
    });
  }, [filtered, sort.key, sort.dir]);

  function rowClick(s) {
    navigate(`/sweeps/${encodeURIComponent(s.namespace)}/${encodeURIComponent(s.name)}`);
  }

  function clearFilters() {
    setSearchText('');
    // Intentionally does NOT clear ?sort= — sort is a view preference, not a
    // filter, and resetting it on "Clear filters" was reported as surprising.
    setQuery({ q: undefined, ns: undefined, phase: undefined });
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

  const hasFilters = Boolean(searchText || ns || activeFilter);

  return html`
    <div class="sweeps-page" data-testid="page-sweeps">
      <div class="section-header">
        <div class="filter-tabs" role="tablist" aria-label="Filter sweeps by phase">
          ${FILTERS.map(f => {
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
                  ? html`<span class="filter-tab-count">${list.length}</span>`
                  : html`<span class="filter-tab-count">
                      ${list.filter(s => f.value.includes((s.phase ?? '').toLowerCase())).length}
                    </span>`}
              </button>
            `;
          })}
        </div>
        <span class="text-dim" style="font-size: var(--font-size-sm); display: inline-flex; align-items: center; gap: var(--space-2)" aria-live="polite" aria-atomic="true">
          <span>${filtered.length} of ${list.length} sweep${list.length !== 1 ? 's' : ''}</span>
          ${updatedAgo != null && html`
            <span
              class="text-dim"
              style="font-size: var(--font-size-xs); opacity: 0.75"
              title=${'Auto-refreshes every 5s · last fetch ' + new Date(lastUpdated).toLocaleTimeString()}
              data-testid="sweeps-last-updated"
            >· updated ${updatedAgo}s ago</span>
          `}
          ${sweepsFreshness && html`<${FreshnessPill} source=${sweepsFreshness} compact=${true} />`}
        </span>
      </div>

      <${StaleBanner} source=${sweepsFreshness} label="Sweeps list" />

      <div style="display: flex; gap: var(--space-3); margin-bottom: var(--space-4); flex-wrap: wrap; align-items: center">
        <div style="position: relative; flex: 1; min-width: 150px; display: flex; align-items: center">
          <input
            type="text"
            placeholder="Search name, namespace, or model..."
            aria-label="Search sweeps by name, namespace, or model"
            value=${searchText}
            oninput=${e => setSearchText(e.target.value)}
            onkeydown=${e => { if (e.key === 'Enter' && searchText !== urlQ) { e.preventDefault(); setQuery({ q: searchText }); } }}
            style=${`width: 100%; padding: var(--space-2) ${searchText ? '28px' : 'var(--space-3)'} var(--space-2) var(--space-3);
                     background: ${palette.mantle}; border: 1px solid ${palette.surface0};
                     border-radius: var(--radius-md); color: ${palette.text};
                     font-size: var(--font-size-sm)`}
          />
          ${searchText && html`
            <button
              type="button"
              aria-label="Clear search"
              onclick=${() => setSearchText('')}
              data-testid="search-clear"
              style=${'position: absolute; right: 6px; background: transparent; border: 0; color: ' + palette.overlay0 + '; cursor: pointer; font-size: 16px; line-height: 1; padding: 2px 6px'}
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
      </div>

      ${firstLoad
        ? html`<div class="card"><${LoadingPanel} label="Loading sweeps…" testid="sweeps-loading" /></div>`
        : loadError
          ? html`<div class="card" style="border-color: var(--error); color: var(--error)" data-testid="sweeps-error">Failed to load sweeps: ${loadError}</div>`
          : sorted.length === 0
            ? (hasFilters
              ? html`<div class="job-table-empty" data-testid="sweeps-empty-filtered" style="text-align:center;padding:var(--space-6)">
                  <p style=${'color:' + palette.text + ';margin:0 0 var(--space-3) 0'}>No sweeps match these filters.</p>
                  <button type="button"
                    onclick=${clearFilters}
                    style=${'padding: var(--space-2) var(--space-4); background: ' + palette.surface0 + '; border: 1px solid ' + palette.surface1 + '; border-radius: var(--radius-md); color: ' + palette.text + '; cursor: pointer; font-size: var(--font-size-sm)'}
                  >
                    Clear filters
                  </button>
                </div>`
              : html`<div class="job-table-empty" data-testid="sweeps-empty-real" style="text-align:center;padding:var(--space-6)">
                  <p style=${'color:' + palette.text + ';margin:0 0 var(--space-2) 0'}>No sweeps yet.</p>
                  <p class="text-dim" style="margin:0;font-size:var(--font-size-sm)">
                    Create one with <code>aiperf kube apply -f sweep.yaml</code>.
                  </p>
                </div>`)
            : html`
          <div class="job-table-wrapper">
            <table class="job-table" data-testid="sweep-table">
              <thead>
                <tr>
                  ${COLUMNS.map(col => html`
                    <th key=${col.key}
                        class="job-table-th"
                        role="columnheader"
                        tabindex="0"
                        onkeydown=${(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleSort(col.key); } }}
                        onclick=${() => toggleSort(col.key)}
                        data-testid=${'col-header-' + col.key}>
                      ${col.label} ${renderSortIcon(col.key)}
                    </th>
                  `)}
                </tr>
              </thead>
              <tbody>
                ${sorted.map(s => {
                  const completedRuns = finiteNumber(sweepAlias(s, 'completed_runs', 'completedRuns')) ?? 0;
                  const failedRuns = finiteNumber(sweepAlias(s, 'failed_runs', 'failedRuns')) ?? 0;
                  const totalVariations = finiteNumber(sweepAlias(s, 'total_variations', 'totalVariations'));
                  const ageSeconds = finiteNumber(s.age_seconds ?? s.ageSeconds);
                  const detailUrl = `/sweeps/${encodeURIComponent(s.namespace)}/${encodeURIComponent(s.name)}`;
                  return html`
                    <tr key=${`${s.namespace}/${s.name}`}
                        class="job-table-row"
                        role="row"
                        tabindex="0"
                        onkeydown=${(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); rowClick(s); } }}
                        onclick=${() => rowClick(s)}
                        style="cursor: pointer"
                        data-testid=${'sweep-row-' + (s.namespace ?? '') + '-' + (s.name ?? '')}>
                      <td class="job-table-td job-table-name">${s.name}</td>
                      <td class="job-table-td">
                        <${NsPill} ns=${s.namespace} onClick=${nsClicked => setQuery({ ns: nsClicked, q: undefined })} testId=${'sweep-row-ns-' + (s.namespace ?? '')} />
                      </td>
                      <td class="job-table-td">${renderPhase(s.phase)}</td>
                      <td class="job-table-td" title=${(() => {
                        if (totalVariations == null) return '';
                        if (completedRuns > totalVariations) return `${completedRuns} child runs completed across all run epochs; current epoch declares ${totalVariations} variations. Stale children from prior epochs are being counted — server-side rollup should be filtering by current runEpoch.`;
                        return '';
                      })()}>
                        ${(() => {
                          if (totalVariations == null || totalVariations === 0) return html`${completedRuns} / ?`;
                          // Clamp the visible numerator so users don't see "5 / 3"
                          // when the rollup picks up stale children from prior
                          // run-epochs. The hover-title surfaces the raw numbers.
                          const shown = Math.min(completedRuns, totalVariations);
                          const surplus = completedRuns > totalVariations ? completedRuns - totalVariations : 0;
                          return html`${shown} / ${totalVariations}${surplus > 0
                            ? html`<span class="text-dim" style="margin-left:4px;font-size:11px"> (+${surplus} stale)</span>`
                            : null}`;
                        })()}
                      </td>
                      <td class="job-table-td"
                          style=${failedRuns > 0 ? `color:${palette.red}` : ''}>
                        ${failedRuns}
                      </td>
                      <td class="job-table-td">${totalVariations ?? html`<span class="text-dim">—</span>`}</td>
                      <td class="job-table-td">
                        ${s.model
                          ? html`<${ModelPill} model=${s.model} testId=${'sweep-row-model-' + s.model} />`
                          : html`<span class="text-dim">—</span>`}
                      </td>
                      <td class="job-table-td">${renderSource(s.source)}</td>
                      <td class="job-table-td text-dim">
                        <${RelativeTime} seconds=${ageSeconds} />
                      </td>
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
