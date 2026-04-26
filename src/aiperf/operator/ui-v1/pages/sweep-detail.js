import { html } from 'htm/preact';
import { useState, useEffect, useMemo } from 'preact/hooks';
import { api, poll } from '../lib/api.js';
import { palette } from '../lib/theme.js';
import { KpiCard } from '../components/kpi-card.js';
import { Conditions } from '../components/conditions.js';
import { JobTable } from '../components/job-table.js';
import { CellsChart } from '../components/cells-chart.js';
import { CellsTable } from '../components/cells-table.js';
import { navigate } from '../lib/router.js';

const TERMINAL = new Set(['succeeded', 'failed', 'cancelled', 'partiallyfailed']);
const DEFAULT_METRIC = 'request_throughput';
const DEFAULT_STAT = 'avg';

export function SweepDetail({ namespace, name }) {
  const [detail, setDetail] = useState(null);
  const [cells, setCells] = useState(null);
  const [view, setView] = useState('chart');
  const [metric, setMetric] = useState(DEFAULT_METRIC);
  const [stat, setStat] = useState(DEFAULT_STAT);
  const [error, setError] = useState(null);

  useEffect(() => {
    const ac = new AbortController();
    let stopped = false;
    async function tick() {
      try {
        const d = await api.getSweep(namespace, name);
        if (!stopped) setDetail(d);
        const phase = (d?.sweep?.phase ?? '').toLowerCase();
        if (TERMINAL.has(phase)) ac.abort();
      } catch (e) {
        if (!stopped) setError(String(e));
      }
    }
    poll(tick, 5000, ac.signal);
    return () => { stopped = true; ac.abort(); };
  }, [namespace, name]);

  useEffect(() => {
    let cancelled = false;
    api.getSweepCells(namespace, name)
      .then(d => { if (!cancelled) setCells(d); })
      .catch(e => { if (!cancelled) setError(String(e)); });
    return () => { cancelled = true; };
  }, [namespace, name]);

  const childRows = useMemo(() => detail?.children ?? [], [detail]);
  const metricNames = useMemo(() => {
    const set = new Set();
    for (const c of (cells?.cells ?? [])) {
      for (const m of Object.keys(c.metrics ?? {})) set.add(m);
    }
    return [...set].sort();
  }, [cells]);

  if (error) {
    return html`<div data-testid="page-sweep-detail" class="error-banner">${error}</div>`;
  }
  if (!detail) {
    return html`<div data-testid="page-sweep-detail" class="text-dim">Loading…</div>`;
  }

  const s = detail.sweep;
  const status = detail.status ?? {};
  const conditions = status.conditions ?? [];
  const currentCell = status.currentCell;

  return html`
    <div class="sweep-detail" data-testid="page-sweep-detail">
      <header class="page-header">
        <h2>${s.name} <span class="text-dim">${s.namespace}</span></h2>
        <div style="display:flex;gap:var(--space-3);align-items:center">
          <span style=${`background:${pillColor(s.phase)};color:white;padding:2px 10px;border-radius:8px;font-size:12px`}>${s.phase}</span>
          <span class="text-dim">model: ${s.model ?? '—'}</span>
          <span class="text-dim">${s.source}</span>
        </div>
        ${currentCell && html`
          <p class="text-dim">running variation ${currentCell.variationIndex ?? '?'}/${s.total_variations} ${currentCell.trial != null ? `trial ${currentCell.trial}` : ''}</p>
        `}
      </header>

      <section class="kpi-row" style="display:grid;grid-template-columns:repeat(4,1fr);gap:var(--space-3)">
        <${KpiCard} label="Variations" value=${s.total_variations} />
        <${KpiCard} label="Completed" value=${s.completed_runs} />
        <${KpiCard} label="Failed" value=${s.failed_runs} />
        <${KpiCard} label="Total runs" value=${s.completed_runs + s.failed_runs} />
      </section>

      <section style="margin-top:var(--space-4)">
        <${Conditions} conditions=${conditions} />
      </section>

      <section style="margin-top:var(--space-4)">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:var(--space-2)">
          <h3>Cells</h3>
          <div style="display:flex;gap:var(--space-2);align-items:center">
            <select value=${metric} onchange=${e => setMetric(e.target.value)}>
              ${metricNames.length === 0
                ? html`<option value=${DEFAULT_METRIC}>${DEFAULT_METRIC}</option>`
                : metricNames.map(m => html`<option key=${m} value=${m}>${m}</option>`)}
            </select>
            <select value=${stat} onchange=${e => setStat(e.target.value)}>
              ${['avg','p50','p90','p95','p99','min','max'].map(s2 =>
                html`<option key=${s2} value=${s2}>${s2}</option>`)}
            </select>
            <button class=${'filter-tab' + (view === 'chart' ? ' filter-tab--active' : '')}
                    onclick=${() => setView('chart')}>Chart</button>
            <button class=${'filter-tab' + (view === 'table' ? ' filter-tab--active' : '')}
                    onclick=${() => setView('table')}>Table</button>
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
      </section>

      <section style="margin-top:var(--space-4)">
        <h3>Children</h3>
        <${JobTable} jobs=${childRows} onRowClick=${j =>
          navigate(`/jobs/${encodeURIComponent(j.namespace)}/${encodeURIComponent(j.name)}`)} />
      </section>
    </div>
  `;
}

function pillColor(phase) {
  const p = (phase ?? '').toLowerCase();
  if (['running', 'aggregating'].includes(p)) return palette.blue ?? '#4ea1ff';
  if (p === 'succeeded') return palette.green ?? '#4caf50';
  if (['failed', 'cancelled', 'partiallyfailed'].includes(p)) return palette.red ?? '#e53935';
  return palette.surface0;
}
