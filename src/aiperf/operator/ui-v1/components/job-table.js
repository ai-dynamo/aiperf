import { html } from 'htm/preact';
import { useState, useMemo } from 'preact/hooks';
import { phaseColor, palette } from '../lib/theme.js';
import { fmtNumber, fmtThroughput } from '../lib/format.js';

const COLUMNS = [
  { key: 'name', label: 'Name' },
  { key: 'namespace', label: 'Namespace' },
  { key: 'phase', label: 'Phase' },
  { key: 'workers', label: 'Workers' },
  { key: 'progress', label: 'Progress' },
  { key: 'throughput', label: 'Throughput' },
  { key: 'latency', label: 'Latency' },
  { key: 'age', label: 'Age' },
];

function relativeAge(ts) {
  if (!ts) return '---';
  const ms = Date.now() - new Date(ts).getTime();
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h`;
  return `${Math.floor(h / 24)}d`;
}

// API returns AIPerfJobInfo with flat camelCase fields:
// name, namespace, phase, workersReady, workersTotal, progressPercent,
// throughputRps, latencyP99Ms, created, model, endpoint, currentPhase, error
function jobValue(job, key) {
  switch (key) {
    case 'name': return job.name ?? '';
    case 'namespace': return job.namespace ?? '';
    case 'phase': return job.phase ?? '';
    case 'workers': return job.workersTotal ?? 0;
    case 'progress': return job.progressPercent ?? 0;
    case 'throughput': return job.throughputRps ?? 0;
    case 'latency': return job.latencyP99Ms ?? 0;
    case 'age': return job.created ? new Date(job.created).getTime() : 0;
    default: return '';
  }
}

export function JobTable({ jobs, onRowClick, filter }) {
  const [sortKey, setSortKey] = useState('age');
  const [sortDir, setSortDir] = useState(-1);

  function toggleSort(key) {
    if (sortKey === key) {
      setSortDir((d) => -d);
    } else {
      setSortKey(key);
      setSortDir(1);
    }
  }

  const filtered = filter && filter.length > 0
    ? (jobs ?? []).filter((j) => {
        const phase = (j.phase ?? '').toLowerCase();
        return filter.map((f) => f.toLowerCase()).includes(phase);
      })
    : (jobs ?? []);

  const sorted = [...filtered].sort((a, b) => {
    const av = jobValue(a, sortKey);
    const bv = jobValue(b, sortKey);
    if (av < bv) return -sortDir;
    if (av > bv) return sortDir;
    return 0;
  });

  // Feature 9: Compute max throughput for relative bar sizing
  const maxThroughput = useMemo(() => {
    let max = 0;
    for (const j of (jobs ?? [])) {
      const val = j.throughputRps ?? 0;
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

  function renderSourceBadge(job) {
    if (job.source === 'archived') {
      return html`<span class="text-dim" style="margin-left: var(--space-2); font-size: var(--font-size-xs)" title="No Kubernetes resource; showing PVC results">archived</span>`;
    }
    if (job.source === 'both') {
      return html`<span class="text-dim" style="margin-left: var(--space-2); font-size: var(--font-size-xs)" title="Live CR + persisted results">live+archived</span>`;
    }
    return null;
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
    if (val == null) return html`<span class="text-dim">---</span>`;

    const phase = (job.phase ?? '').toLowerCase();
    const isComplete = phase === 'completed' || phase === 'succeeded';
    const pct = maxThroughput > 0 ? (val / maxThroughput) * 100 : 0;

    return html`
      <div style="display: flex; align-items: center; gap: var(--space-2); min-width: 120px">
        <span style="white-space: nowrap; min-width: 60px">${fmtThroughput(val)} req/s</span>
        ${isComplete && maxThroughput > 0 && html`
          <div
            style=${'flex: 1; height: 4px; background: ' + palette.surface0 + '; border-radius: 2px; overflow: hidden; min-width: 40px'}
          >
            <div
              style=${'height: 100%; width: ' + pct.toFixed(1) + '%; background: ' + palette.blue + '; border-radius: 2px; transition: width 0.3s'}
            />
          </div>
        `}
      </div>
    `;
  }

  function renderLatency(job) {
    const val = job.latencyP99Ms;
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

  if (sorted.length === 0) {
    return html`<div class="job-table-empty"><p>No jobs found</p></div>`;
  }

  return html`
    <div class="job-table-wrapper">
      <table class="job-table">
        <thead>
          <tr>
            ${COLUMNS.map(
              (col) => html`
                <th key=${col.key} class="job-table-th" onclick=${() => toggleSort(col.key)} data-testid=${'col-header-' + col.key}>
                  ${col.label} ${renderSortIcon(col.key)}
                </th>
              `,
            )}
          </tr>
        </thead>
        <tbody data-testid="job-table">
          ${sorted.map((job) => html`
            <tr
              key=${job.namespace + '/' + job.name}
              class="job-table-row"
              onclick=${() => onRowClick && onRowClick(job)}
              style=${onRowClick ? 'cursor: pointer' : ''}
              data-testid=${'job-row-' + (job.namespace ?? '') + '-' + (job.name ?? '')}
            >
              <td class="job-table-td job-table-name">${job.name}</td>
              <td class="job-table-td text-dim">${job.namespace}</td>
              <td class="job-table-td">${renderPhase(job.phase)}${renderSourceBadge(job)}</td>
              <td class="job-table-td">${renderWorkers(job)}</td>
              <td class="job-table-td">${renderProgress(job)}</td>
              <td class="job-table-td">${renderThroughput(job)}</td>
              <td class="job-table-td">${renderLatency(job)}</td>
              <td class="job-table-td text-dim">${relativeAge(job.created)}</td>
            </tr>
          `)}
        </tbody>
      </table>
    </div>
  `;
}
