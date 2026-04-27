import { html } from 'htm/preact';
import { useState, useEffect, useMemo } from 'preact/hooks';
import { api, poll } from '../lib/api.js';
import { palette, phaseColor } from '../lib/theme.js';
import { KpiCard } from '../components/kpi-card.js';
import { Conditions } from '../components/conditions.js';
import { JobTable } from '../components/job-table.js';
import { CellsChart } from '../components/cells-chart.js';
import { CellsTable } from '../components/cells-table.js';
import { EpochSelector } from '../components/epoch-selector.js';
import { navigate } from '../lib/router.js';

const TERMINAL = new Set(['succeeded', 'failed', 'cancelled', 'partiallyfailed']);
const RUNNING_PHASES = new Set(['pending', 'running', 'aggregating']);
const DEFAULT_METRIC = 'request_throughput';
const DEFAULT_STAT = 'avg';
const STATS = ['avg', 'p50', 'p90', 'p95', 'p99', 'min', 'max'];

function formatAge(s) {
  if (s == null) return null;
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  if (s < 86400) {
    const h = Math.floor(s / 3600);
    return `${h}h ${Math.floor((s % 3600) / 60)}m`;
  }
  return `${Math.floor(s / 86400)}d`;
}

export function SweepDetail({ namespace, name, epoch }) {
  const [detail, setDetail] = useState(null);
  const [cells, setCells] = useState(null);
  const [epochs, setEpochs] = useState([]);
  const [archivedChildren, setArchivedChildren] = useState(null);
  const [view, setView] = useState('chart');
  const [metric, setMetric] = useState(DEFAULT_METRIC);
  const [stat, setStat] = useState(DEFAULT_STAT);
  const [error, setError] = useState(null);

  useEffect(() => {
    const ac = new AbortController();
    let stopped = false;
    async function tick() {
      try {
        const d = await api.getSweep(namespace, name, epoch);
        if (!stopped) setDetail(d);
        const phase = (d?.sweep?.phase ?? '').toLowerCase();
        if (TERMINAL.has(phase)) ac.abort();
      } catch (e) {
        if (!stopped) setError(String(e));
      }
    }
    poll(tick, 5000, ac.signal);
    return () => { stopped = true; ac.abort(); };
  }, [namespace, name, epoch]);

  useEffect(() => {
    let cancelled = false;
    api.getSweepCells(namespace, name, epoch)
      .then(d => { if (!cancelled) setCells(d); })
      .catch(e => { if (!cancelled) setError(String(e)); });
    return () => { cancelled = true; };
  }, [namespace, name, epoch]);

  useEffect(() => {
    let cancelled = false;
    api.getSweepEpochs(namespace, name)
      .then(d => { if (!cancelled) setEpochs(d.epochs ?? []); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [namespace, name]);

  useEffect(() => {
    if (epoch === undefined) {
      setArchivedChildren(null);
      return;
    }
    if (detail?.children && detail.children.length > 0) {
      setArchivedChildren(null);
      return;
    }
    let cancelled = false;
    api.getSweepChildren(namespace, name, epoch)
      .then(d => { if (!cancelled) setArchivedChildren(d?.children ?? []); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [namespace, name, epoch, detail]);

  function pickEpoch(next) {
    if (next === undefined) {
      navigate(`/sweeps/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}`);
    } else {
      navigate(`/sweeps/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}/runs/${encodeURIComponent(next)}`);
    }
  }

  const childRows = useMemo(() => {
    const live = detail?.children ?? [];
    if (epoch !== undefined && live.length === 0 && archivedChildren) {
      return archivedChildren;
    }
    return live;
  }, [detail, epoch, archivedChildren]);
  const childRowsAreArchived =
    epoch !== undefined && (detail?.children ?? []).length === 0 && !!archivedChildren;

  const metricNames = useMemo(() => {
    const set = new Set();
    for (const c of (cells?.cells ?? [])) {
      for (const m of Object.keys(c.metrics ?? {})) set.add(m);
    }
    return [...set].sort();
  }, [cells]);

  if (error) {
    return html`
      <div data-testid="page-sweep-detail">
        <div class="card" style=${`border-color:${palette.red}44;color:${palette.red}`}>
          <strong>Error:</strong> ${error}
        </div>
      </div>
    `;
  }
  if (!detail) {
    return html`<div data-testid="page-sweep-detail" class="text-dim" style="padding:var(--space-6)">Loading…</div>`;
  }

  const s = detail.sweep;
  const status = detail.status ?? {};
  const conditions = status.conditions ?? [];
  const currentCell = status.currentCell;
  const phase = s.phase ?? 'Unknown';
  const phaseClr = phaseColor(phase);
  const isRunning = RUNNING_PHASES.has(phase.toLowerCase());

  return html`
    <div class="sweep-detail" data-testid="page-sweep-detail">
      <!-- Header -->
      <div class="card" style="margin-bottom: var(--space-4)">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:var(--space-3)">
          <div>
            <div style="display:flex;align-items:center;gap:var(--space-3);flex-wrap:wrap">
              <h2 style="margin:0;font-size:var(--font-size-lg)">${s.name}</h2>
              <span class="phase-badge" style=${'background: ' + phaseClr + '22; color: ' + phaseClr + '; border-color: ' + phaseClr + '44'}>
                ${phase}
              </span>
              ${formatAge(s.age_seconds) && html`<span class="text-dim" style="font-size:var(--font-size-sm)">${formatAge(s.age_seconds)}</span>`}
              ${isRunning && html`
                <span style=${`display:inline-flex;align-items:center;gap:var(--space-1);font-size:var(--font-size-xs);color:${palette.green}`}>
                  <span style=${`display:inline-block;width:8px;height:8px;border-radius:50%;background:${palette.green};animation:pulse 1.5s ease-in-out infinite`}></span>
                  Live
                </span>
              `}
              <${EpochSelector} epochs=${epochs} current=${epoch} onPick=${pickEpoch} />
            </div>
            <div class="text-dim" style="font-size:var(--font-size-sm);margin-top:var(--space-1)">
              ${s.namespace}${s.model ? ' · ' + s.model : ''}
              ${html` · <span style=${`color:${palette.overlay1};font-size:var(--font-size-xs);padding:1px 6px;border:1px solid ${palette.surface0};border-radius:6px`}>${s.source}</span>`}
            </div>
            ${currentCell && html`
              <p class="text-dim" style="margin:var(--space-1) 0 0 0;font-size:var(--font-size-sm)">
                running variation ${currentCell.variationIndex ?? '?'}/${s.total_variations}${currentCell.trial != null ? ` · trial ${currentCell.trial}` : ''}
              </p>
            `}
          </div>
        </div>
      </div>

      ${conditions.length > 0 && html`
        <div style="margin-bottom: var(--space-4)">
          <${Conditions} conditions=${conditions} />
        </div>
      `}

      <!-- KPI row -->
      <div class="kpi-row" style="margin-bottom: var(--space-6)">
        <${KpiCard}
          label="Variations"
          value=${s.total_variations}
          color=${palette.blue}
        />
        <${KpiCard}
          label="Completed"
          value=${s.completed_runs}
          color=${palette.green}
        />
        <${KpiCard}
          label="Failed"
          value=${s.failed_runs}
          color=${s.failed_runs > 0 ? palette.red : palette.overlay1}
        />
        <${KpiCard}
          label="Total runs"
          value=${(s.completed_runs ?? 0) + (s.failed_runs ?? 0)}
          color=${palette.mauve}
        />
      </div>

      <!-- Cells panel -->
      <div class="card" style="margin-bottom: var(--space-4)">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:var(--space-3);flex-wrap:wrap;margin-bottom:var(--space-3)">
          <div class="card-title" style="margin:0">Cells</div>
          <div style="display:flex;gap:var(--space-2);align-items:center;flex-wrap:wrap">
            <select
              value=${metric}
              onchange=${e => setMetric(e.target.value)}
              style=${`padding:var(--space-1) var(--space-2);background:${palette.mantle};border:1px solid ${palette.surface0};border-radius:var(--radius-sm);color:${palette.text};font-size:var(--font-size-sm)`}
            >
              ${metricNames.length === 0
                ? html`<option value=${DEFAULT_METRIC}>${DEFAULT_METRIC}</option>`
                : metricNames.map(m => html`<option key=${m} value=${m}>${m}</option>`)}
            </select>
            <select
              value=${stat}
              onchange=${e => setStat(e.target.value)}
              style=${`padding:var(--space-1) var(--space-2);background:${palette.mantle};border:1px solid ${palette.surface0};border-radius:var(--radius-sm);color:${palette.text};font-size:var(--font-size-sm)`}
            >
              ${STATS.map(s2 => html`<option key=${s2} value=${s2}>${s2}</option>`)}
            </select>
            <div class="filter-tabs" style="margin:0">
              <button
                class=${'filter-tab' + (view === 'chart' ? ' filter-tab--active' : '')}
                onclick=${() => setView('chart')}
              >Chart</button>
              <button
                class=${'filter-tab' + (view === 'table' ? ' filter-tab--active' : '')}
                onclick=${() => setView('table')}
              >Table</button>
            </div>
          </div>
        </div>
        ${view === 'chart'
          ? html`<${CellsChart}
              dimensions=${cells?.dimensions ?? []}
              cells=${cells?.cells ?? []}
              metric=${metric}
              stat=${stat} />`
          : html`<${CellsTable}
              dimensions=${cells?.dimensions ?? []}
              cells=${cells?.cells ?? []}
              metric=${metric}
              stat=${stat}
              onCellClick=${c => c.children?.[0] && navigate(`/jobs/${encodeURIComponent(c.children[0].namespace)}/${encodeURIComponent(c.children[0].name)}`)} />`}
      </div>

      <!-- Children -->
      <div class="card" data-testid="sweep-detail-children">
        <div class="card-title">Children (${childRows.length})</div>
        ${childRows.length === 0
          ? html`<div class="text-dim" style="padding:var(--space-3) 0">No children persisted for this epoch yet.</div>`
          : childRowsAreArchived
            ? html`
                <table class="job-table" data-testid="sweep-detail-archived-children">
                  <thead>
                    <tr>
                      <th class="job-table-th">Name</th>
                      <th class="job-table-th">Namespace</th>
                      <th class="job-table-th">Variation</th>
                      <th class="job-table-th">Trial</th>
                      <th class="job-table-th">Run epoch</th>
                    </tr>
                  </thead>
                  <tbody>
                    ${childRows.map(c => html`
                      <tr
                        key=${c.namespace + '/' + c.name + '/' + (c.childRunEpoch ?? '')}
                        class="job-table-row"
                        style="cursor:pointer"
                        onclick=${() => navigate(`/jobs/${encodeURIComponent(c.namespace)}/${encodeURIComponent(c.name)}/runs/${encodeURIComponent(c.childRunEpoch)}`)}
                      >
                        <td class="job-table-td">${c.name}</td>
                        <td class="job-table-td">${c.namespace}</td>
                        <td class="job-table-td">${c.variationLabel || c.variationIndex}</td>
                        <td class="job-table-td">${c.trialIndex ?? '---'}</td>
                        <td class="job-table-td text-dim">${c.childRunEpoch}</td>
                      </tr>
                    `)}
                  </tbody>
                </table>
              `
            : html`<${JobTable} jobs=${childRows} onRowClick=${j =>
                navigate(`/jobs/${encodeURIComponent(j.namespace)}/${encodeURIComponent(j.name)}`)} />`
        }
      </div>
    </div>
  `;
}
