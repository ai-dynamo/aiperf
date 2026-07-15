// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { useState, useEffect, useMemo } from 'preact/hooks';
import { api, poll } from '../lib/api.js';
import { palette, phaseColor } from '../lib/theme.js';
import { sweeps as sweepsSignal, freshness, clearFreshnessSource } from '../lib/state.js';
import { FreshnessPill, StaleBanner } from '../components/freshness.js';
import { KpiCard } from '../components/kpi-card.js';
import { Conditions } from '../components/conditions.js';
import { DiagnosticsPanel } from '../components/diagnostics-panel.js';
import { JobTable } from '../components/job-table.js';
import { LiveVariationsCard } from '../components/live-variations-card.js';
import { SweepLiveTrialBoard } from '../components/sweep-live-trial-board.js';
import { SweepWinnerSummary } from '../components/sweep-winner-summary.js';
import { ArtifactsCard } from '../components/artifacts-card.js';
import { CellsChart } from '../components/cells-chart.js';
import { CellsTable } from '../components/cells-table.js';
import { VariationsTable } from '../components/variations-table.js';
import { VariationsChart } from '../components/variations-chart.js';
import { VariationsPareto } from '../components/variations-pareto.js';
import { EpochSelector } from '../components/epoch-selector.js';
import { NsPill, ModelPill } from '../components/pills.js';
import { RelativeTime } from '../components/time.js';
import { LoadingPanel } from '../components/spinner.js';
import { fmtBytes, fmtNumber } from '../lib/format.js';
import { buildJobPath, navigate, query, setQuery } from '../lib/router.js';
import { buildSweepVariations, pickSweepWinner, resolveSweepManifest, shouldShowSweepDiagnostics, sweepPhaseMode } from './sweep-detail-helpers.js';

// ``archived`` is included so polling stops for sweeps whose live CR
// has been deleted but whose aggregate.json is still served from the
// PVC (see ``sweep_union.py`` lines 152/291). Without it the page
// would tick the API forever for a sweep that's already gone.
const TERMINAL = new Set(['succeeded', 'failed', 'cancelled', 'partiallyfailed', 'archived']);
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
    y: { key: 'time_to_first_token',     stat: 'p50', label: 'TTFT',             unit: 'ms'    },
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
  const [artifactFiles, setArtifactFiles] = useState([]);
  const [artifactFilesLoaded, setArtifactFilesLoaded] = useState(false);
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
  // Mirrors job-detail's ``liveStale``: flips true when a poll throws
  // after the first successful detail load. Lets the header indicator
  // downgrade Live → Stale without nuking the rest of the page on a
  // transient operator restart or port-forward blip.
  const [liveStale, setLiveStale] = useState(false);
  const sweepFreshness = freshness.value['sweep-detail'] ?? null;
  const status = detail?.status ?? {};
  const latestPersistedSweepEpoch = epochs.find(e => e?.isLatest)?.epoch ?? epochs[0]?.epoch;
  const resolvedEpoch = epoch
    ?? (status.runEpoch != null ? String(status.runEpoch) : null)
    ?? (latestPersistedSweepEpoch != null ? String(latestPersistedSweepEpoch) : null);

  useEffect(() => {
    const ac = new AbortController();
    let stopped = false;
    setDetail(null);
    setError(null);
    setLiveStale(false);
    let firstLoadDone = false;
    clearFreshnessSource('sweep-detail');
    async function tick({ stopFreshness }) {
      try {
        const d = await api.getSweep(namespace, name, epoch);
        if (stopped) return;
        setDetail(d);
        setError(null);
        setLiveStale(false);
        firstLoadDone = true;
        const phase = (d?.sweep?.phase ?? '').toLowerCase();
        if (TERMINAL.has(phase)) {
          stopFreshness('terminal');
          ac.abort();
        }
      } catch (e) {
        if (stopped) return;
        if (!firstLoadDone) {
          // Only the first-load failure replaces the page — once we have
          // detail rendered, downgrade subsequent transport errors to a
          // header Stale indicator so users keep their context.
          setError(String(e));
        } else {
          setLiveStale(true);
        }
        throw e;
      }
    }
    poll(tick, 5000, ac.signal, { source: 'sweep-detail' });
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
    const ac = new AbortController();
    setArtifactFiles([]);
    if (resolvedEpoch == null) {
      setArtifactFilesLoaded(true);
      return () => ac.abort();
    }
    setArtifactFilesLoaded(false);
    fetch(api.sweepArtifactListUrl(namespace, name, resolvedEpoch), { signal: ac.signal })
      .then(r => r.ok ? r.json() : null)
      .then(d => {
        if (ac.signal.aborted) return;
        setArtifactFiles(d?.files ?? []);
        setArtifactFilesLoaded(true);
      })
      .catch(() => {
        if (ac.signal.aborted) return;
        setArtifactFiles([]);
        setArtifactFilesLoaded(true);
      });
    return () => ac.abort();
  }, [namespace, name, resolvedEpoch]);

  // Fetch the children manifest from /sweeps/<ns>/<name>/children. Live
  // (sweep-controller alive, but ``status.aggregate.children`` not yet
  // patched) and archived (post-TTL) sweeps both flow through the same
  // endpoint; the operator picks the right source. Skip when the CR
  // already exposes children via ``detail.children`` to avoid duplicate
  // network calls.
  useEffect(() => {
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
  // When the CR-side aggregate is empty (mid-run, sweep-controller hasn't
  // patched yet), fall back to ``archivedChildren`` — the operator's
  // ``/sweeps/<ns>/<name>/children`` endpoint synthesizes a live manifest
  // from labelled AIPerfJob CRs in that case, so the live-variations
  // rollup card has data to render immediately as children appear.
  const manifest = useMemo(
    () => resolveSweepManifest({ detail, archivedChildren }),
    [detail, archivedChildren],
  );

  // Fetch each child's status (summary + phase + progressPercent) after every
  // sweep-detail poll. Child names can remain stable while phases and metrics
  // move Pending -> Running -> Succeeded, so tying this only to the manifest
  // snapshot would leave the live trial board stale.
  useEffect(() => {
    if (manifest.length === 0) {
      setChildSummaries({});
      return;
    }
    let cancelled = false;
    Promise.all(
      manifest.map(c =>
        api.getJob(c.namespace ?? namespace, c.name)
          .then(d => [c.name, {
            summary: d?.status?.summary ?? d?.status?.results?.metrics ?? null,
            phase: d?.status?.phase ?? d?.job?.phase ?? null,
            progressPercent: d?.status?.progressPercent ?? d?.job?.progressPercent ?? d?.progressPercent ?? null,
          }])
          .catch(() => [c.name, { summary: null, phase: 'Unknown', progressPercent: null }])
      )
    ).then(pairs => {
      if (cancelled) return;
      setChildSummaries(Object.fromEntries(pairs));
    });
    return () => { cancelled = true; };
  }, [detail, namespace, JSON.stringify(manifest.map(c => c.name))]);

  // Group manifest entries by variation_index and compute mean/std/cv per
  // headline metric across the available trials. ``perMetric`` is keyed
  // ``"<key>.<stat>"`` so a metric+stat selector can index it directly.
  const variations = useMemo(() => buildSweepVariations({
    manifest,
    childSummaries,
    cells,
    headlineMetrics: HEADLINE_METRICS,
  }), [manifest, childSummaries, cells]);

  // Per-metric series used by the chart: one point per variation, with
  // ``mean`` + ``std`` for the error band.
  const chartMetric = useMemo(() => {
    const m = HEADLINE_METRICS.find(x => x.key + '.' + x.stat === chartMetricKey)
      ?? HEADLINE_METRICS[0];
    const metricKey = m.key + '.' + m.stat;
    const series = variations.map(v => {
      const r = v.perMetric?.[metricKey];
      return {
        variation_index: v.variation_index,
        label: v.label,
        mean: r?.mean ?? null,
        std: r?.std ?? 0,
        cv: r?.cv ?? null,
        n: r?.n ?? 0,
      };
    });
    return { meta: m, metricKey, series };
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
  const conditions = status.conditions ?? [];
  const pods = detail.pods ?? [];
  const currentCell = status.currentCell;
  const phase = s.phase ?? 'Unknown';
  const phaseClr = phaseColor(phase);
  const phaseLower = phase.toLowerCase();
  const isRunning = RUNNING_PHASES.has(phaseLower);
  // ``archived`` covers responses sourced purely from the PVC aggregate
  // (CR has been deleted, or a non-latest epoch was requested) — see
  // ``sweep_union._record_from_archived_doc``. Treat as a successful
  // completion so headline KPI tones, progress bars, and any
  // ``isCompleted``-gated UI render the same way as a live ``Succeeded``
  // CR; the alternative hides a finished sweep behind a phase string the
  // page never generated itself.
  const isCompleted = phaseLower === 'succeeded'
    || phaseLower === 'completed'
    || phaseLower === 'archived';
  const isFailed = phaseLower === 'failed';
  const isPartiallyFailed = phaseLower === 'partiallyfailed';
  const isCancelled = phaseLower === 'cancelled';
  // Show legacy /cells panel only when the new manifest path has nothing
  // to render — avoids a confusing "No cells completed yet." card sitting
  // next to a populated VariationsTable.
  const hasManifest = manifest.length > 0;
  const phaseMode = sweepPhaseMode(phase);
  const isLiveMode = phaseMode === 'live';
  const isTerminalMode = phaseMode === 'terminal';
  const winner = pickSweepWinner({ variations, metricKey: chartMetric.metricKey });

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
              ${sweepFreshness && html`<${FreshnessPill} source=${sweepFreshness} compact=${true} />`}
              ${isRunning
                ? liveStale
                  ? html`
                    <span
                      title="Live updates paused — operator API is not responding. Retrying in the background; numbers shown are from the last successful poll."
                      data-testid="sweep-detail-live-stale"
                      style=${`display:inline-flex;align-items:center;gap:var(--space-1);font-size:var(--font-size-xs);color:${palette.amber}`}
                    >
                      <span style=${`display:inline-block;width:8px;height:8px;border-radius:50%;background:${palette.amber}`}></span>
                      Stale
                    </span>
                  `
                  : html`
                    <span
                      data-testid="sweep-detail-live"
                      style=${`display:inline-flex;align-items:center;gap:var(--space-1);font-size:var(--font-size-xs);color:${palette.green}`}
                    >
                      <span style=${`display:inline-block;width:8px;height:8px;border-radius:50%;background:${palette.green};animation:pulse 1.5s ease-in-out infinite`}></span>
                      Live
                    </span>
                  `
                : isCompleted
                  ? phaseLower === 'archived'
                    ? html`<span
                        title="Sweep finished and the live CR has been archived — values shown come from the persisted aggregate."
                        style=${'font-size:var(--font-size-xs);color:' + palette.subtext0 + ';opacity:0.85'}
                      >Archived</span>`
                    : html`<span style=${'font-size:var(--font-size-xs);color:' + palette.green + ';opacity:0.7'}>Completed</span>`
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

      <${StaleBanner} source=${sweepFreshness} label="Sweep detail" />

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
        ${(() => {
          // ``cancelled`` is a separate terminal bucket from ``failed`` on AIPerfSweep.
          // ``s.cancelled_runs`` arrives via the extended SweepRecord schema;
          // tolerate older API responses where the field is absent.
          const failed = s.failed_runs ?? 0;
          const cancelled = s.cancelled_runs ?? 0;
          const nonSuccess = failed + cancelled;
          const completed = s.completed_runs ?? 0;
          return html`
            <${KpiCard}
              label="Completed"
              value=${`${completed}/${completed + nonSuccess}`}
              color=${palette.green}
              icon="check"
              tone="ok"
            />
            <${KpiCard}
              label="Failed"
              value=${nonSuccess}
              color=${nonSuccess > 0 ? palette.red : palette.overlay1}
              icon="errors"
              tone=${nonSuccess > 0 ? 'bad' : 'neutral'}
              sub=${cancelled > 0 ? `${cancelled} cancelled` : undefined}
            />
          `;
        })()}
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

      ${isTerminalMode && html`
        <${SweepWinnerSummary} winner=${winner} metric=${chartMetric.meta} />
      `}

      <div style="margin-bottom: var(--space-4)">
        <${ArtifactsCard}
          files=${artifactFiles}
          filesLoaded=${artifactFilesLoaded}
          namespace=${namespace}
          name=${name}
          epoch=${resolvedEpoch}
          resolvedEpoch=${resolvedEpoch}
          isCompleted=${isCompleted}
          isRunning=${isRunning}
          api=${api}
          fmtBytes=${fmtBytes}
          title="Aggregate Artifacts"
          testIdPrefix="sweep-detail-aggregate-artifacts"
          cardTestId="sweep-detail-aggregate-artifacts-card"
          bundleUrl=${resolvedEpoch != null ? api.sweepArtifactBundleUrl(namespace, name, resolvedEpoch) : null}
          quickExportUrl=${resolvedEpoch != null ? api.sweepProfileExportUrl(namespace, name, resolvedEpoch, 'json') : null}
          quickExportLabel="Export JSON"
          showIndividualDownloadAll=${true}
          fileUrl=${fileName => resolvedEpoch != null
            ? api.sweepArtifactFileUrl(namespace, name, resolvedEpoch, fileName)
            : null}
          emptyMessages=${{
            waiting: 'Waiting for a sweep epoch before showing aggregate artifacts.',
            completed: 'No aggregate artifacts available for this sweep epoch.',
            running: 'No aggregate artifacts yet.',
            available: 'No aggregate artifacts available for this sweep epoch.',
            unavailable: 'No aggregate artifacts available for this sweep epoch.',
          }}
          emptyDetails=${{
            waiting: 'This page requires a pinned sweep epoch before it will fetch aggregate artifacts, so the sweep summary and results cannot drift to different runs.',
            completed: 'The sweep completed but no aggregate artifacts were uploaded — check the operator logs or the sweep-controller pod for this epoch.',
            running: 'Aggregate files appear here once the sweep-controller writes and uploads the sweep aggregate bundle.',
            unavailable: 'Aggregate artifacts will appear here after the sweep starts producing output.',
          }}
        />
      </div>

      ${isLiveMode && hasManifest && html`
        <${SweepLiveTrialBoard}
          manifest=${manifest}
          childSummaries=${childSummaries}
        />
      `}

      <!-- Per-variation curve + table (driven by the inline aggregate manifest) -->
      ${hasManifest && html`
        <div class="card" style="margin-bottom: var(--space-4)" data-testid="sweep-detail-variations">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:var(--space-3);flex-wrap:wrap;margin-bottom:var(--space-3)">
            <div class="card-title" style="margin:0">Variation curve</div>
            <select
              class="ui-select"
              value=${chartMetricKey}
              onchange=${e => setQuery({ metric: e.target.value === DEFAULT_CHART_METRIC_KEY ? undefined : e.target.value })}
              data-testid="variations-chart-metric"
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
                <button type="button"
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

      <!-- Live Variations (per-variation rollup, replaces empty state of cells/headlines during a live sweep) -->
      ${!hasManifest ? null : html`
        <${LiveVariationsCard} manifest=${manifest} childData=${childSummaries} />
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
                        key=${c.namespace + '/' + c.name + '/' + (c.childRunEpoch ?? c.child_run_epoch ?? '')}
                        class="job-table-row"
                        role="row"
                        tabindex="0"
                        style="cursor:pointer"
                        onkeydown=${(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); navigate(buildJobPath({ ...c, childRunEpoch: c.childRunEpoch ?? c.child_run_epoch })); } }}
                        onclick=${() => navigate(buildJobPath({ ...c, childRunEpoch: c.childRunEpoch ?? c.child_run_epoch }))}
                      >
                        <td class="job-table-td">${c.name}</td>
                        <td class="job-table-td">${c.namespace}</td>
                        <td class="job-table-td">${c.variationLabel ?? c.variation_label ?? c.variationIndex ?? c.variation_index ?? '---'}</td>
                        <td class="job-table-td">${c.trialIndex ?? c.trial_index ?? '---'}</td>
                        <td class="job-table-td text-dim">${c.childRunEpoch ?? c.child_run_epoch ?? '---'}</td>
                      </tr>
                    `)}
                  </tbody>
                </table>
              `
            : html`<${JobTable} jobs=${childRows} onRowClick=${j =>
                navigate(buildJobPath(j))} />`
        }
      </div>

      <!-- Events / Logs / Conditions / Pods (tabbed) -->
      ${shouldShowSweepDiagnostics(phase) && html`
        <div style="margin-top: var(--space-4)">
          <${DiagnosticsPanel}
            ns=${namespace}
            name=${name}
            kind="sweep"
            conditions=${conditions}
            pods=${pods}
            mode="live"
            archived=${false}
            eventCount=${null}
            logSeverityCounts=${null}
            conditionWarnCount=${(conditions || []).filter(c => c.status !== 'True').length}
            podCrashCount=${(pods || []).filter(p => /crashloop/i.test(p.reason || '')).length} />
        </div>
      `}
    </div>
  `;
}
