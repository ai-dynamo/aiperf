import { html } from 'htm/preact';
import { useState, useEffect, useMemo } from 'preact/hooks';
import { api, poll } from '../lib/api.js';
import { palette, phaseColor } from '../lib/theme.js';
import { sweeps as sweepsSignal } from '../lib/state.js';
import { KpiCard } from '../components/kpi-card.js';
import { Conditions } from '../components/conditions.js';
import { JobTable } from '../components/job-table.js';
import { CellsChart } from '../components/cells-chart.js';
import { CellsTable } from '../components/cells-table.js';
import { VariationsTable } from '../components/variations-table.js';
import { VariationsChart } from '../components/variations-chart.js';
import { VariationsPareto } from '../components/variations-pareto.js';
import { EpochSelector } from '../components/epoch-selector.js';
import { NsPill, ModelPill } from '../components/pills.js';
import { RelativeTime } from '../components/time.js';
import { LoadingPanel } from '../components/spinner.js';
import { fmtNumber } from '../lib/format.js';
import { buildJobPath, navigate, query, setQuery } from '../lib/router.js';

const TERMINAL = new Set(['succeeded', 'failed', 'cancelled', 'partiallyfailed']);
const RUNNING_PHASES = new Set(['pending', 'running', 'aggregating']);

const HEADLINE_METRICS = [
  { key: 'request_throughput',      stat: 'avg', label: 'Req throughput',      unit: 'req/s' },
  { key: 'output_token_throughput', stat: 'avg', label: 'Output tok/s',        unit: 'tok/s' },
  { key: 'total_token_throughput',  stat: 'avg', label: 'Total tok/s',         unit: 'tok/s' },
  { key: 'request_latency',         stat: 'p50', label: 'Req latency p50',     unit: 'ms'    },
  { key: 'request_latency',         stat: 'p99', label: 'Req latency p99',     unit: 'ms'    },
  { key: 'time_to_first_token',     stat: 'p50', label: 'TTFT p50',            unit: 'ms'    },
  { key: 'time_to_first_token',     stat: 'p99', label: 'TTFT p99',            unit: 'ms'    },
  { key: 'inter_token_latency',     stat: 'avg', label: 'ITL avg',             unit: 'ms'    },
];

const DEFAULT_CHART_METRIC_KEY = 'output_token_throughput.avg';

// Mirror the axis presets from the legacy ui's ``analysis.js`` so the
// pareto UX feels identical: pick from a short list of well-known
// throughput-vs-latency pairs rather than freeform x/y selectors.
const PARETO_AXES = [
  {
    key: 'tps_p99',
    label: 'req/s × lat p99',
    x: { key: 'request_throughput',      stat: 'avg', label: 'Throughput',       unit: 'req/s' },
    y: { key: 'request_latency',         stat: 'p99', label: 'Latency P99',      unit: 'ms'    },
    yIsSmallerBetter: true,
  },
  {
    key: 'tps_ttft',
    label: 'req/s × TTFT',
    x: { key: 'request_throughput',      stat: 'avg', label: 'Throughput',       unit: 'req/s' },
    y: { key: 'time_to_first_token',     stat: 'avg', label: 'TTFT',             unit: 'ms'    },
    yIsSmallerBetter: true,
  },
  {
    key: 'tok_p99',
    label: 'tok/s × lat p99',
    x: { key: 'output_token_throughput', stat: 'avg', label: 'Token Throughput', unit: 'tok/s' },
    y: { key: 'request_latency',         stat: 'p99', label: 'Latency P99',      unit: 'ms'    },
    yIsSmallerBetter: true,
  },
];
const DEFAULT_PARETO_AXIS_KEY = 'tps_p99';

/** Compute population mean / std / cv across an array of numbers. */
function meanStd(values) {
  const filtered = values.filter(v => typeof v === 'number' && Number.isFinite(v));
  if (filtered.length === 0) return null;
  const n = filtered.length;
  const mean = filtered.reduce((a, b) => a + b, 0) / n;
  if (n < 2) return { mean, std: 0, cv: null, n };
  const variance = filtered.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  const std = Math.sqrt(variance);
  const cv = mean !== 0 ? std / Math.abs(mean) : null;
  return { mean, std, cv, n };
}

function fmtKpi(value, unit) {
  if (value == null) return '---';
  if (unit === 'req/s' || unit === 'tok/s') return fmtNumber(value, 0);
  if (unit === 'ms') return fmtNumber(value, value < 1 ? 3 : 1);
  return fmtNumber(value, 3);
}

// "Similar sweeps" chip — sweep-level mirror of the job-detail
// ``SimilarRunsLink`` (same namespace AND same model, excluding the current
// sweep). Count-only — never aggregate metrics across independent sweeps.
// Clicking jumps to ``/sweeps?ns=<namespace>`` filtered to the namespace,
// where the user can pick another to compare side-by-side. No new backend
// route required: derived purely from the existing ``sweeps`` signal.
function SimilarSweepsLink({ namespace, model, currentName }) {
  if (!namespace || !model) return null;
  const all = sweepsSignal.value ?? [];
  let n = 0;
  for (const r of all) {
    if (r.namespace === namespace && r.model === model && r.name !== currentName) n++;
  }
  if (n === 0) return null;

  const onClick = (e) => {
    e.preventDefault();
    navigate('/sweeps?ns=' + encodeURIComponent(namespace));
  };

  return html`
    <a
      href=${'#/sweeps?ns=' + encodeURIComponent(namespace)}
      onclick=${onClick}
      data-testid="sweep-detail-similar-sweeps"
      title=${`Browse the other ${n} sweep${n === 1 ? '' : 's'} on model "${model}" in namespace "${namespace}"`}
      style=${'display: inline-flex; align-items: center; gap: var(--space-1);'
        + ' padding: 2px var(--space-2);'
        + ' border-radius: 999px;'
        + ' font-size: var(--font-size-xs);'
        + ' font-weight: 600;'
        + ' background: ' + palette.accent + '14;'
        + ' color: ' + palette.accent + ';'
        + ' border: 1px solid ' + palette.accent + '33;'
        + ' text-decoration: none;'
        + ' cursor: pointer'}
    >
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <rect x="3" y="3" width="7" height="7" rx="1" />
        <rect x="14" y="3" width="7" height="7" rx="1" />
        <rect x="3" y="14" width="7" height="7" rx="1" />
        <rect x="14" y="14" width="7" height="7" rx="1" />
      </svg>
      <span>+${n} similar sweep${n === 1 ? '' : 's'}</span>
      <span aria-hidden="true" style="opacity: 0.7; font-size: 10px">→</span>
    </a>
  `;
}

export function SweepDetail({ namespace, name, epoch }) {
  const [detail, setDetail] = useState(null);
  const [cells, setCells] = useState(null);
  const [epochs, setEpochs] = useState([]);
  const [archivedChildren, setArchivedChildren] = useState(null);
  const [childSummaries, setChildSummaries] = useState({});
  // URL-driven view state: ?metric= and ?axis= persist the chart-metric and
  // pareto-axis selectors so deep-links and reloads keep the chosen view.
  // Default values are elided from the URL to avoid noise.
  const urlMetric = query.value.metric ?? DEFAULT_CHART_METRIC_KEY;
  const urlAxis = query.value.axis ?? DEFAULT_PARETO_AXIS_KEY;
  const [chartMetricKey, setChartMetricKey] = useState(urlMetric);
  const [paretoAxisKey, setParetoAxisKey] = useState(urlAxis);
  useEffect(() => {
    if (chartMetricKey !== urlMetric) setChartMetricKey(urlMetric);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [urlMetric]);
  useEffect(() => {
    if (paretoAxisKey !== urlAxis) setParetoAxisKey(urlAxis);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [urlAxis]);
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
      .catch(() => {});
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

  // The variation manifest lives on ``status.aggregate.children`` once the
  // sweep-controller has patched it. The on-disk ``children.json`` is
  // wrapped as ``{sweep_run_epoch, children: [...]}`` and embedded
  // verbatim, so normalize either shape (object envelope or bare array).
  const manifest = useMemo(() => {
    const raw = detail?.status?.aggregate?.children;
    if (Array.isArray(raw)) return raw;
    if (raw && Array.isArray(raw.children)) return raw.children;
    return [];
  }, [detail]);

  // Fetch each child's status.summary once per manifest snapshot, so the
  // chart and table share one set of network requests instead of duplicating.
  useEffect(() => {
    if (manifest.length === 0) {
      setChildSummaries({});
      return;
    }
    let cancelled = false;
    Promise.all(
      manifest.map(c =>
        api.getJob(c.namespace ?? namespace, c.name)
          .then(d => [c.name, d?.status?.summary ?? d?.status?.results?.metrics ?? null])
          .catch(() => [c.name, null])
      )
    ).then(pairs => {
      if (cancelled) return;
      setChildSummaries(Object.fromEntries(pairs));
    });
    return () => { cancelled = true; };
  }, [namespace, JSON.stringify(manifest.map(c => c.name))]);

  // Group manifest entries by variation_index and compute mean/std/cv per
  // headline metric across the available trials. ``perMetric`` is keyed
  // ``"<key>.<stat>"`` so a metric+stat selector can index it directly.
  const variations = useMemo(() => {
    if (manifest.length === 0) return [];
    const groups = new Map();
    for (const c of manifest) {
      const idx = c.variation_index ?? 0;
      if (!groups.has(idx)) {
        groups.set(idx, {
          variation_index: idx,
          label: c.variation_label ?? '',
          n_total: 0,
          summaries: [],
        });
      }
      const g = groups.get(idx);
      g.n_total += 1;
      const summary = childSummaries[c.name];
      if (summary) g.summaries.push(summary);
    }
    return [...groups.values()]
      .sort((a, b) => a.variation_index - b.variation_index)
      .map(g => {
        const perMetric = {};
        for (const m of HEADLINE_METRICS) {
          const values = g.summaries.map(s => s?.[m.key]?.[m.stat]).filter(x => x != null);
          const r = meanStd(values);
          perMetric[m.key + '.' + m.stat] = r ?? { mean: null, std: null, cv: null, n: 0 };
        }
        return {
          variation_index: g.variation_index,
          label: g.label,
          n_trials: g.summaries.length,
          n_total: g.n_total,
          perMetric,
        };
      });
  }, [manifest, childSummaries]);

  // Per-metric series used by the chart: one point per variation, with
  // ``mean`` + ``std`` for the error band.
  const chartMetric = useMemo(() => {
    const m = HEADLINE_METRICS.find(x => x.key + '.' + x.stat === chartMetricKey)
      ?? HEADLINE_METRICS[0];
    const series = variations.map(v => {
      const r = v.perMetric?.[m.key + '.' + m.stat];
      return {
        variation_index: v.variation_index,
        label: v.label,
        mean: r?.mean ?? null,
        std: r?.std ?? 0,
        cv: r?.cv ?? null,
        n: r?.n ?? 0,
      };
    });
    return { meta: m, series };
  }, [variations, chartMetricKey]);

  const paretoAxis = useMemo(() =>
    PARETO_AXES.find(a => a.key === paretoAxisKey) ?? PARETO_AXES[0]
  , [paretoAxisKey]);

  // Headline KPI extraction: pick the *peak* mean across variations for
  // throughput, and the *minimum* mean across variations for latency. CV
  // shown on the card is the variation that produced the peak/min.
  const headlineKpis = useMemo(() => {
    const out = [];
    const pick = (key, stat, label, unit, mode) => {
      const points = variations
        .map(v => ({ v, r: v.perMetric?.[key + '.' + stat] }))
        .filter(p => p.r?.mean != null);
      if (points.length === 0) return;
      points.sort((a, b) => mode === 'max' ? b.r.mean - a.r.mean : a.r.mean - b.r.mean);
      const top = points[0];
      out.push({
        label,
        unit,
        value: top.r.mean,
        cv: top.r.cv,
        variation: top.v.label || `v${top.v.variation_index}`,
      });
    };
    pick('output_token_throughput', 'avg', 'Peak output tok/s',  'tok/s', 'max');
    pick('request_throughput',      'avg', 'Peak req/s',         'req/s', 'max');
    pick('time_to_first_token',     'p50', 'Best TTFT p50',      'ms',    'min');
    pick('request_latency',         'p99', 'Best req lat p99',   'ms',    'min');
    return out;
  }, [variations]);

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
    return html`<div data-testid="page-sweep-detail"><${LoadingPanel} label=${'Loading sweep ' + namespace + '/' + name + '…'} /></div>`;
  }

  const s = detail.sweep;
  const status = detail.status ?? {};
  const conditions = status.conditions ?? [];
  const currentCell = status.currentCell;
  const phase = s.phase ?? 'Unknown';
  const phaseClr = phaseColor(phase);
  const phaseLower = phase.toLowerCase();
  const isRunning = RUNNING_PHASES.has(phaseLower);
  const isCompleted = phaseLower === 'succeeded' || phaseLower === 'completed';
  const isFailed = phaseLower === 'failed';
  const isPartiallyFailed = phaseLower === 'partiallyfailed';
  const isCancelled = phaseLower === 'cancelled';
  // Show legacy /cells panel only when the new manifest path has nothing
  // to render — avoids a confusing "No cells completed yet." card sitting
  // next to a populated VariationsTable.
  const hasManifest = manifest.length > 0;

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
              <${NsPill} ns=${s.namespace} onClick=${ns => navigate('/sweeps?ns=' + encodeURIComponent(ns))} testId="sweep-detail-ns-pill" />
              ${s.model && html`<${ModelPill} model=${s.model} testId="sweep-detail-model-pill" />`}
              ${s.model && s.model !== '---' && html`<${SimilarSweepsLink} namespace=${s.namespace} model=${s.model} currentName=${s.name} />`}
              ${s.age_seconds != null && html`<${RelativeTime} seconds=${s.age_seconds} mode="elapsed" className="text-dim" />`}
              ${isRunning
                ? html`
                  <span style=${`display:inline-flex;align-items:center;gap:var(--space-1);font-size:var(--font-size-xs);color:${palette.green}`}>
                    <span style=${`display:inline-block;width:8px;height:8px;border-radius:50%;background:${palette.green};animation:pulse 1.5s ease-in-out infinite`}></span>
                    Live
                  </span>
                `
                : isCompleted
                  ? html`<span style=${'font-size:var(--font-size-xs);color:' + palette.green + ';opacity:0.7'}>Completed</span>`
                  : isFailed
                    ? html`<span style=${'font-size:var(--font-size-xs);color:' + palette.red + ';opacity:0.85'} title="Sweep failed before completing — see conditions for the underlying reason.">Failed</span>`
                    : isCancelled
                      ? html`<span style=${'font-size:var(--font-size-xs);color:' + palette.overlay1 + ';opacity:0.85'} title="Sweep was cancelled before completion — KPIs reflect partial data.">Cancelled</span>`
                      : isPartiallyFailed
                        ? html`<span style=${'font-size:var(--font-size-xs);color:' + palette.red + ';opacity:0.85'} title="Sweep finished but some variations failed — KPIs reflect surviving data.">Partially failed</span>`
                        : null
              }
              <${EpochSelector} epochs=${epochs} current=${epoch} onPick=${pickEpoch} />
            </div>
            <div class="text-dim" style="font-size:var(--font-size-sm);margin-top:var(--space-1)">
              <span style=${`color:${palette.overlay1};font-size:var(--font-size-xs);padding:1px 6px;border:1px solid ${palette.surface0};border-radius:6px`}>${s.source}</span>
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
          <${Conditions} conditions=${conditions.length > 8 ? conditions.slice(-8) : conditions} />
          ${conditions.length > 8 && html`
            <div class="text-dim" style="font-size:var(--font-size-xs);margin-top:var(--space-1);padding-left:var(--space-2)">
              Showing 8 most recent of ${conditions.length} conditions.
            </div>
          `}
        </div>
      `}

      <!-- KPI row: progress (left) + headline performance (right) -->
      <div class="kpi-row" style="margin-bottom: var(--space-4)">
        ${(() => {
          const totalVariations = s.total_variations ?? 0;
          const completed = s.completed_runs ?? 0;
          const failed = s.failed_runs ?? 0;
          const denom = completed + failed;
          const pct = totalVariations > 0
            ? Math.min(100, Math.round((denom / totalVariations) * 100))
            : (denom > 0 ? 100 : 0);
          return html`
            <${KpiCard}
              label="Variations"
              value=${totalVariations}
              color=${palette.blue}
              icon="trending-up"
              tone="accent"
              progress=${pct}
              progressTone=${isRunning ? 'live' : 'accent'}
            />
          `;
        })()}
        <${KpiCard}
          label="Completed"
          value=${`${s.completed_runs ?? 0}/${(s.completed_runs ?? 0) + (s.failed_runs ?? 0)}`}
          color=${palette.green}
          icon="check"
          tone="ok"
        />
        <${KpiCard}
          label="Failed"
          value=${s.failed_runs}
          color=${s.failed_runs > 0 ? palette.red : palette.overlay1}
          icon="errors"
          tone=${s.failed_runs > 0 ? 'bad' : 'neutral'}
        />
        ${headlineKpis.map((k, i) => {
          const iconByLabel = {
            'Peak output tok/s': 'trending-up',
            'Peak req/s':        'speed',
            'Best TTFT p50':     'clock',
            'Best req lat p99':  'timer',
          };
          const isLeadThroughput = i === 0;
          const tone = isLeadThroughput
            ? (isRunning ? 'live' : 'gold')
            : (k.unit === 'ms' ? 'ok' : 'accent');
          return html`
            <${KpiCard}
              key=${k.label}
              label=${k.label}
              value=${fmtKpi(k.value, k.unit)}
              unit=${k.unit}
              color=${palette.peach}
              icon=${iconByLabel[k.label] ?? 'trophy'}
              tone=${tone}
              sub=${html`<span class="text-dim" style="font-size:var(--font-size-xs)">${k.variation}${k.cv != null ? ` · cv ${(k.cv * 100).toFixed(1)}%` : ''}</span>`}
            />
          `;
        })}
      </div>

      <!-- Per-variation curve + table (driven by the inline aggregate manifest) -->
      ${hasManifest && html`
        <div class="card" style="margin-bottom: var(--space-4)" data-testid="sweep-detail-variations">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:var(--space-3);flex-wrap:wrap;margin-bottom:var(--space-3)">
            <div class="card-title" style="margin:0">Variation curve</div>
            <select
              value=${chartMetricKey}
              onchange=${e => setQuery({ metric: e.target.value === DEFAULT_CHART_METRIC_KEY ? undefined : e.target.value })}
              data-testid="variations-chart-metric"
              style=${`padding:var(--space-1) var(--space-2);background:${palette.mantle};border:1px solid ${palette.surface0};border-radius:var(--radius-sm);color:${palette.text};font-size:var(--font-size-sm)`}
            >
              ${HEADLINE_METRICS.map(m => html`
                <option key=${m.key + '.' + m.stat} value=${m.key + '.' + m.stat}>
                  ${m.label} (${m.unit})
                </option>
              `)}
            </select>
          </div>
          <${VariationsChart}
            variations=${chartMetric.series}
            metricLabel=${chartMetric.meta.label}
            unit=${chartMetric.meta.unit}
          />
          <div style="margin-top: var(--space-3); overflow-x: auto">
            <${VariationsTable}
              variations=${variations}
              headlineMetrics=${HEADLINE_METRICS}
            />
          </div>
        </div>

        <!-- Pareto frontier — mirrors the legacy ui's analysis.js: pick
             one of a few preset axis pairs and we sweep the points
             monotonically (sort by x asc, track best y, push improvements)
             to draw the frontier as a dashed line. -->
        <div class="card" style="margin-bottom: var(--space-4)" data-testid="sweep-detail-pareto">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:var(--space-3);flex-wrap:wrap;margin-bottom:var(--space-3)">
            <div class="card-title" style="margin:0">Pareto · ${paretoAxis.x.label} × ${paretoAxis.y.label}</div>
            <div class="filter-tabs" role="tablist" aria-label="Pareto axis selector" style="margin:0">
              ${PARETO_AXES.map(a => html`
                <button
                  key=${a.key}
                  role="tab"
                  aria-pressed=${paretoAxisKey === a.key}
                  aria-selected=${paretoAxisKey === a.key}
                  title=${a.x.label + ' (' + a.x.unit + ') × ' + a.y.label + ' (' + a.y.unit + ')'}
                  class=${'filter-tab' + (paretoAxisKey === a.key ? ' filter-tab--active' : '')}
                  onclick=${() => setQuery({ axis: a.key === DEFAULT_PARETO_AXIS_KEY ? undefined : a.key })}
                  data-testid=${'pareto-axis-' + a.key}
                >${a.label}</button>
              `)}
            </div>
          </div>
          <${VariationsPareto}
            variations=${variations}
            xMetric=${paretoAxis.x}
            yMetric=${paretoAxis.y}
            yIsSmallerBetter=${paretoAxis.yIsSmallerBetter}
          />
        </div>
      `}

      <!-- Legacy server-computed Cells panel — only when the new manifest
           path has no data, e.g. older sweeps that never carried the
           inline aggregate. -->
      ${!hasManifest && cells && html`
        <div class="card" style="margin-bottom: var(--space-4)">
          <div class="card-title">Cells</div>
          <${CellsChart}
            dimensions=${cells?.dimensions ?? []}
            cells=${cells?.cells ?? []}
            metric="request_throughput"
            stat="avg" />
          <div style="margin-top: var(--space-3)">
            <${CellsTable}
              dimensions=${cells?.dimensions ?? []}
              cells=${cells?.cells ?? []}
              metric="request_throughput"
              stat="avg"
              onCellClick=${c => c.children?.[0] && navigate(buildJobPath(c.children[0]))} />
          </div>
        </div>
      `}

      <!-- Children -->
      <div class="card" data-testid="sweep-detail-children">
        <div class="card-title" style="display:flex;align-items:center;gap:var(--space-2);flex-wrap:wrap">
          <span>Children (${childRows.length})</span>
          ${childRowsAreArchived && html`
            <span
              title="These runs are from a prior sweep epoch — re-running the sweep will produce a new set."
              style=${`font-size:var(--font-size-xs);font-weight:normal;padding:1px 6px;border:1px solid ${palette.surface1};border-radius:6px;color:${palette.overlay1};background:${palette.surface0}33`}
            >
              archived epoch ${epoch}
            </span>
          `}
        </div>
        ${childRows.length === 0
          ? phaseLower === 'pending'
            ? html`<div class="text-dim" style="padding:var(--space-3) 0" data-testid="sweep-detail-children-pending">
                Sweep is being initialized — children will appear here shortly.
              </div>`
            : html`<div class="text-dim" style="padding:var(--space-3) 0">No children persisted for this epoch yet.</div>`
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
                navigate(buildJobPath(j))} />`
        }
      </div>
    </div>
  `;
}
