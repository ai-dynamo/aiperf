// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * ARCHIVE — cross-namespace past-runs browser.
 *
 * Mounted at ``/archive``. Unlike the legacy view (``/ns/:ns/archive``)
 * this page takes ALL namespaces by default and exposes a namespace
 * filter dropdown; the active namespace is mirrored to ``?ns=<value>``.
 *
 * Data sources merged (deduped by ``namespace + '/' + name``):
 *   - ``api.listJobs()`` — live + PVC-archive union (already deduped
 *     server-side by ``operator.job_union.list_all_jobs``). Polled at 5s.
 *   - ``api.listResults()`` — pure PVC scan keyed by ``(ns, job_id)``.
 *     Adds nothing the union endpoint already returns, but provides a
 *     belt-and-suspenders fallback when the union response is degraded.
 *     Entries lack phase/throughput/latency/start_time, so we synthesize
 *     ``phase = 'completed'`` (files exist => the run finished) and let
 *     the sort comparators treat missing metrics as ``-Infinity``.
 */

import { html } from 'htm/preact';
import { useEffect, useMemo, useState } from 'preact/hooks';
import { api, poll } from '../lib/api.js';
import { jobs, dedupeByNsName } from '../lib/state.js';
import { navigate, query, setQuery, buildJobPath } from '../lib/router.js';
import { palette, phaseColor } from '../lib/theme.js';

const BUCKETS = [
  { key: 'all', label: 'All', match: () => true },
  { key: 'live', label: 'Live', match: j => ['running', 'initializing', 'pending'].includes((j.phase ?? '').toLowerCase()) },
  { key: 'pass', label: 'Passed', match: j => ['completed', 'succeeded'].includes((j.phase ?? '').toLowerCase()) },
  { key: 'fault', label: 'Failed', match: j => ['failed', 'error'].includes((j.phase ?? '').toLowerCase()) },
];

const SORTS = [
  { key: 'newest', label: 'Newest first' },
  { key: 'oldest', label: 'Oldest first' },
  { key: 'rps', label: 'Throughput (desc)' },
  { key: 'p99', label: 'P99 latency (desc)' },
  { key: 'dur', label: 'Duration (desc)' },
];

function phaseBucket(phase) {
  const p = (phase ?? '').toLowerCase();
  if (p === 'running' || p === 'initializing' || p === 'pending') return 'live';
  if (p === 'failed' || p === 'error') return 'fault';
  if (p === 'completed' || p === 'succeeded') return 'passed';
  return 'other';
}

function statusTone(phase) {
  const b = phaseBucket(phase);
  if (b === 'fault') return 'bad';
  if (b === 'live') return 'info';
  if (b === 'passed') return 'good';
  return 'neutral';
}

function statusLabel(phase) {
  const b = phaseBucket(phase);
  if (b === 'fault') return 'Failed';
  if (b === 'live') return 'Running';
  if (b === 'passed') return 'Passed';
  return phase ? String(phase) : '—';
}

function relAge(ts) {
  if (!ts) return '—';
  const s = Math.floor((Date.now() - new Date(ts).getTime()) / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h`;
  return `${Math.floor(h / 24)}d`;
}

function jobStartMs(j) {
  return j.startTime ? new Date(j.startTime).getTime() : 0;
}

function jobDurationSec(j) {
  if (j.startTime && j.completionTime) {
    return (new Date(j.completionTime) - new Date(j.startTime)) / 1000;
  }
  if (j.startTime) return (Date.now() - new Date(j.startTime).getTime()) / 1000;
  return 0;
}

function compareJobs(sort) {
  const num = v => (v == null ? -Infinity : Number(v));
  switch (sort) {
    case 'oldest': return (a, b) => jobStartMs(a) - jobStartMs(b);
    case 'rps': return (a, b) => num(b.throughputRps) - num(a.throughputRps);
    case 'p99': return (a, b) => num(b.latencyP99Ms) - num(a.latencyP99Ms);
    case 'dur': return (a, b) => jobDurationSec(b) - jobDurationSec(a);
    case 'newest':
    default: return (a, b) => jobStartMs(b) - jobStartMs(a);
  }
}

// Shape a /results JobEntry into the same shape the live ``jobs`` signal uses,
// so the merged list can pass through one set of helpers. The /results
// endpoint only carries (namespace, job_id, model, endpoint, file_count,
// total_size_bytes) — no phase/timing/metrics — so we synthesize what we can:
//   * ``phase = 'completed'`` (the entry exists because files are on disk)
//   * ``startTime/completionTime = null`` (not in payload; sort to bottom)
//   * ``throughputRps/latencyP99Ms = null`` (sort to -Infinity)
function archiveEntryToJobLike(entry) {
  return {
    namespace: entry.namespace,
    name: entry.job_id,
    phase: 'completed',
    model: entry.model ?? null,
    endpoint: entry.endpoint ?? null,
    startTime: null,
    completionTime: null,
    throughputRps: null,
    latencyP99Ms: null,
    source: 'archive',
  };
}

// Status chip — small inline pill matching the colour used in JobTable's
// phase-badge so the same job presents identically across pages.
function StatusChip({ phase }) {
  const tone = statusTone(phase);
  const label = statusLabel(phase);
  const color = tone === 'bad' ? palette.red
    : tone === 'good' ? palette.green
    : tone === 'info' ? palette.blue
    : palette.subtext0;
  // phaseColor() falls back when ``phase`` doesn't map cleanly; use the
  // tone-derived colour above for archive entries with synthetic phases.
  const final = phase ? phaseColor(phase) : color;
  return html`
    <span
      class="phase-badge"
      style=${'background: ' + final + '22; color: ' + final + '; border-color: ' + final + '44'}
    >${label}</span>
  `;
}

export function Archive() {
  const [bucket, setBucket] = useState('all');
  const [q, setQ] = useState('');
  const [sort, setSort] = useState('newest');
  const [liveJobs, setLiveJobs] = useState(jobs.value);
  const [archived, setArchived] = useState([]);
  const [archivedError, setArchivedError] = useState(null);

  const ns = query.value.ns ?? '';

  // Poll listJobs at 5s — same cadence as pages/jobs.js. If the user landed
  // here cold (jobs.value empty), this fills the global signal so a later
  // navigation to /jobs starts populated too.
  useEffect(() => {
    const ac = new AbortController();
    poll(
      async () => {
        try {
          const data = await api.listJobs();
          const list = dedupeByNsName(data?.jobs ?? []);
          jobs.value = list;
          setLiveJobs(list);
        } catch {
          // Swallow poll errors — the merged list still renders archived
          // entries below; surfacing a hard error here would hide them.
        }
      },
      5000,
      ac.signal,
    );
    return () => ac.abort();
  }, []);

  // /results is one-shot: it only changes when a run finishes and writes its
  // artifacts, which the live-jobs poll already surfaces via the union path.
  useEffect(() => {
    api.listResults()
      .then(resp => setArchived(resp?.jobs ?? []))
      .catch(err => setArchivedError(err?.message ?? String(err)));
  }, []);

  // Merge: live entries take precedence, then any archive-only entries
  // whose (ns, job_id) is not already in the live union.
  const merged = useMemo(() => {
    const liveKeys = new Set(liveJobs.map(j => `${j.namespace ?? 'default'}/${j.name ?? ''}`));
    const out = [...liveJobs];
    for (const entry of archived) {
      const key = `${entry.namespace ?? 'default'}/${entry.job_id ?? ''}`;
      if (!entry.job_id) continue;
      if (liveKeys.has(key)) continue;
      out.push(archiveEntryToJobLike(entry));
    }
    return out;
  }, [liveJobs, archived]);

  // Namespace-scoped list — counts and rows reflect this. (Does not apply
  // the search filter, so users see what's in the namespace before typing.)
  const nsList = useMemo(
    () => (ns ? merged.filter(j => (j.namespace ?? '') === ns) : merged),
    [merged, ns],
  );

  const namespaces = useMemo(() => {
    const set = new Set(merged.map(j => j.namespace).filter(Boolean));
    return [...set].sort();
  }, [merged]);

  const cur = BUCKETS.find(b => b.key === bucket) ?? BUCKETS[0];

  const filtered = useMemo(() => {
    let r = nsList.filter(cur.match);
    if (q) {
      const needle = q.toLowerCase();
      r = r.filter(j => (j.name ?? '').toLowerCase().includes(needle)
        || (j.model ?? '').toLowerCase().includes(needle));
    }
    return [...r].sort(compareJobs(sort));
  }, [nsList, bucket, q, sort]);

  const bucketCount = key => nsList.filter(BUCKETS.find(b => b.key === key).match).length;

  const shownCount = filtered.length;
  const hiddenCount = Math.max(0, nsList.length - shownCount);
  const liveShown = filtered.filter(j => phaseBucket(j.phase) === 'live').length;
  const passedShown = filtered.filter(j => phaseBucket(j.phase) === 'passed').length;
  const faultShown = filtered.filter(j => phaseBucket(j.phase) === 'fault').length;

  const inputStyle = 'padding: var(--space-2) var(--space-3); background: '
    + palette.mantle + '; border: 1px solid ' + palette.surface0
    + '; border-radius: var(--radius-md); color: ' + palette.text
    + '; font-size: var(--font-size-sm)';

  return html`
    <div class="archive-page" data-testid="page-archive">
      <div class="section-header" style="margin-bottom: var(--space-4)">
        <span class="section-title">Archive</span>
        <span class="text-dim" style="font-size: var(--font-size-sm)" data-testid="arch-summary">
          <span>Shown ${shownCount}</span>
          <span style="margin-left: var(--space-3)">Hidden ${hiddenCount}</span>
          <span style="margin-left: var(--space-3)">Running ${liveShown}</span>
          <span style="margin-left: var(--space-3)">Passed ${passedShown}</span>
          <span style="margin-left: var(--space-3)">Failed ${faultShown}</span>
        </span>
      </div>

      <div class="filter-tabs" role="tablist" aria-label="Filter archive by phase" style="margin-bottom: var(--space-3)">
        ${BUCKETS.map(b => {
          const active = bucket === b.key;
          return html`
            <button
              key=${b.key}
              role="tab"
              aria-selected=${active}
              class=${'filter-tab' + (active ? ' filter-tab--active' : '')}
              data-testid=${'tab-' + b.key}
              onclick=${() => setBucket(b.key)}
            >
              ${b.label}
              <span class="filter-tab-count">${bucketCount(b.key)}</span>
            </button>
          `;
        })}
      </div>

      <div style="display: flex; gap: var(--space-3); margin-bottom: var(--space-4); flex-wrap: wrap; align-items: center">
        <input
          type="text"
          value=${q}
          oninput=${e => setQ(e.target.value)}
          placeholder="filter name / model…"
          aria-label="Search archive by name or model"
          data-testid="arch-search"
          style=${inputStyle + '; flex: 1; min-width: 200px'}
        />
        <select
          class="ui-select"
          value=${ns}
          onchange=${e => setQuery({ ns: e.target.value || undefined })}
          data-testid="arch-ns"
          aria-label="Filter by namespace"
        >
          <option value="">All namespaces</option>
          ${namespaces.map(n => html`<option key=${n} value=${n}>${n}</option>`)}
        </select>
        <select
          class="ui-select"
          value=${sort}
          onchange=${e => setSort(e.target.value)}
          data-testid="arch-sort"
          aria-label="Sort runs"
        >
          ${SORTS.map(s => html`<option key=${s.key} value=${s.key}>${s.label}</option>`)}
        </select>
      </div>

      ${archivedError && html`
        <div class="card" style=${'border-color:' + palette.amber + '; color:' + palette.amber + '; margin-bottom: var(--space-3)'}>
          Archive scan failed: ${archivedError}. Showing live jobs only.
        </div>
      `}

      ${filtered.length === 0
        ? html`
          <div class="card" data-testid="arch-empty" style="text-align: center; padding: var(--space-6)">
            <p class="text-dim" style="margin: 0">
              ${ns
                ? `No matches in namespace ${ns}${q ? ' for "' + q + '"' : ''} — try another namespace${q ? ' or clear the search' : ''}.`
                : `No matches${q ? ' for "' + q + '"' : ''} — try changing the filter${q ? ' or clearing the search' : ''}.`}
            </p>
          </div>`
        : html`
          <div class="card" style="padding: 0">
            ${filtered.map(j => {
              const modelShort = j.model ? String(j.model).split('/').pop() : '—';
              const ageTs = j.completionTime ?? j.startTime ?? j.created;
              return html`
                <div
                  key=${(j.namespace ?? 'default') + '/' + (j.name ?? '')}
                  class="archive-row"
                  data-testid=${'arch-row-' + (j.namespace ?? 'default') + '-' + (j.name ?? '')}
                  onclick=${() => navigate(buildJobPath(j))}
                  style=${'display: grid; grid-template-columns: minmax(160px, 2fr) minmax(120px, 1fr) 90px 110px; gap: var(--space-3); align-items: center; padding: var(--space-3); border-bottom: 1px solid ' + palette.surface0 + '; cursor: pointer'}
                >
                  <div>
                    <div style="font-weight: 600; color: var(--text)">${j.name}</div>
                    <div class="text-dim" style="font-size: var(--font-size-xs)">${j.namespace ?? 'default'}</div>
                  </div>
                  <div class="text-dim" style="font-family: var(--font-mono); font-size: var(--font-size-xs)" title=${j.model ?? ''}>${modelShort}</div>
                  <div class="text-dim" style="font-size: var(--font-size-xs)" title=${ageTs ?? ''}>${relAge(ageTs)}</div>
                  <div><${StatusChip} phase=${j.phase} /></div>
                </div>
              `;
            })}
          </div>
        `}
    </div>
  `;
}
