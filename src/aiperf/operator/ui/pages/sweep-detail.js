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
import { fmtBytes, fmtMilliseconds, fmtNumber, fmtReqPerSecond } from '../lib/format.js';
import { buildJobPath, navigate, query, setQuery } from '../lib/router.js';
import { buildSweepVariations, isVariationFeasible, pickObjectiveWinner, pickSweepWinner, resolveSweepManifest, shouldShowSweepDiagnostics, sweepPhaseMode } from './sweep-detail-helpers.js';
import { RelaunchButton } from '../components/relaunch-button.js';

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
  if (unit === 'req/s') return fmtReqPerSecond(value);
  if (unit === 'tok/s') return fmtNumber(value, 0);
  if (unit === 'ms') return fmtMilliseconds(value);
  return fmtNumber(value, 3);
}

// Similar-sweep link — sweep-level mirror of the job-detail
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
      class="sweep-detail-similar-link"
    >
      ${n} similar sweep${n === 1 ? '' : 's'}
    </a>
  `;
}

export function SweepDetail({ namespace, name, epoch }) {
  const [detail, setDetail] = useState(null);
  const [cells, setCells] = useState(null);
  // `epochs` is paired with the sweep it was fetched for. Clearing it in an
  // effect is not enough: state updates land on the NEXT render, so the first
  // render after navigating to another sweep still sees the previous sweep's
  // epochs and resolves `resolvedEpoch` from them.
  const [epochs, setEpochs] = useState([]);
  const [epochsFor, setEpochsFor] = useState(null);
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
  const [sweepConfig, setSweepConfig] = useState(null);
  // Mirrors job-detail's ``liveStale``: flips true when a poll throws
  // after the first successful detail load. Lets the header indicator
  // downgrade Live → Stale without nuking the rest of the page on a
  // transient operator restart or port-forward blip.
  const [liveStale, setLiveStale] = useState(false);
  const sweepFreshness = freshness.value['sweep-detail'] ?? null;
  const status = detail?.status ?? {};
  // Only trust `epochs` when they were fetched for THIS sweep. Without the
  // guard, navigating gemma-conc2 -> cp-sweep -> gemma-bo4 made each page
  // request its predecessor's epoch
  // (/sweeps/gemma-bo4/epochs/<cp-sweep-epoch>/artifacts -> 404), and the
  // artifacts card rendered "No aggregate artifacts available for this sweep
  // epoch" for a sweep whose artifacts exist.
  const epochsAreForThisSweep = epochsFor === `${namespace}/${name}`;
  const latestPersistedSweepEpoch = epochsAreForThisSweep
    ? (epochs.find(e => e?.isLatest)?.epoch ?? epochs[0]?.epoch)
    : undefined;
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
    setSweepConfig(null);
    api.getSweepConfig(namespace, name, epoch)
      .then(d => { if (!cancelled) setSweepConfig(d); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [namespace, name, epoch]);

  useEffect(() => {
    let cancelled = false;
    // Clear FIRST. `resolvedEpoch` falls back to the newest entry of `epochs`
    // when `detail` is null, and `detail` IS null immediately after navigating
    // to another sweep. Leaving the previous sweep's epochs in state made the
    // new sweep resolve to a FOREIGN epoch and fetch
    // /sweeps/<new>/epochs/<old-epoch>/artifacts, which 404s -- the artifacts
    // card then rendered "No aggregate artifacts available for this sweep
    // epoch" for a sweep whose artifacts exist. Observed navigating
    // gemma-conc2 -> cp-sweep -> gemma-bo4: each sweep requested its
    // predecessor's epoch.
    const key = `${namespace}/${name}`;
    api.getSweepEpochs(namespace, name)
      .then(d => { if (!cancelled) { setEpochs(d.epochs ?? []); setEpochsFor(key); } })
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
  // network calls. That skip is only safe because `detail.children` entries
  // carry `variationValues` themselves (AIPerfJobInfo, populated from the
  // `aiperf.nvidia.com/variation-values` annotation on the live CR and from
  // the sweep epoch's children.json on the archived half). It always fires on
  // a live sweep -- the sweeps router builds `detail.children` from
  // `list_all_jobs` -- so anything the manifest carries but AIPerfJobInfo does
  // not is unreachable in exactly the case it was written for.
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
        api.getJob(c.namespace ?? namespace, c.name, c.childRunEpoch ?? c.child_run_epoch ?? null)
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
  }, [detail, namespace, JSON.stringify(manifest.map(c => [c.name, c.childRunEpoch ?? c.child_run_epoch]))]);

  // Group manifest entries by variation_index and compute mean/std/cv per
  // headline metric across the available trials. ``perMetric`` is keyed
  // ``"<key>.<stat>"`` so a metric+stat selector can index it directly.
  // `statusRuns` is the GAP-FILLER for the swept parameter values, not their
  // only source: `indexVariationValues` reads the children manifest first and
  // consults `status.runs[]` only for variation indices the manifest did not
  // cover. The manifest copy is fresher (it exists while a child is still
  // running; `status.runs[]` is appended only at terminal) and less lossy
  // (2048-byte annotation budget vs 256 for status). Both still matter:
  // without values from one of them, adaptive variations render as opaque
  // `search_iter_NNNN` with no indication of what was actually tried.
  const statusRuns = detail?.status?.runs ?? null;
  const variations = useMemo(() => buildSweepVariations({
    manifest,
    childSummaries,
    cells,
    statusRuns,
    headlineMetrics: HEADLINE_METRICS,
  }), [manifest, childSummaries, cells, statusRuns]);

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
        // Carried so VariationsChart can tick the x axis with what each point
        // actually tried ("concurrency=17") instead of the adaptive planner's
        // cell id ("search_iter_0008"). The chart already prefers this field;
        // dropping it here silently made that fall back to the opaque id.
        valuesLabel: v.valuesLabel ?? null,
        mean: r?.mean ?? null,
        // Null, never 0. `meanStd` returns null for n<2 because one
        // observation does not estimate spread, and `VariationsChart` documents
        // null as "not measured" (`variations-chart.js:24-25`). Coercing it
        // back to zero here would re-assert "measured, and every trial landed
        // on the same number" for every single-trial variation.
        std: r?.std ?? null,
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
  //
  // Each tile also carries the SLA regime its number belongs to. The extremum
  // is deliberately taken over ALL variations, feasible or not -- see
  // `kpiSlaAnnotation` for why the tiles are labelled rather than filtered --
  // and the verdict comes from the same `isVariationFeasible` the winner card
  // uses, so the two surfaces cannot disagree about who passes.
  const headlineKpis = useMemo(() => {
    const claim = slaClaimState({
      slaFilters: detail?.spec_summary?.sla_filters ?? null,
      // Null, not 0, when there is no search summary: absent evidence about how
      // many filters the search applied must not read as "it applied none".
      slaFilterCount: detail?.search_summary
        ? detail.search_summary.sla_filter_count
        : null,
      variations,
    });
    const feasibleIndexes = claim.state === 'active'
      ? new Set(
          variations
            .filter(v => isVariationFeasible(v, claim.filters))
            .map(v => v.variation_index),
        )
      : null;
    const out = [];
    const pick = (key, stat, label, unit, mode) => {
      const points = variations
        .map(v => ({ v, r: v.perMetric?.[key + '.' + stat] }))
        .filter(p => p.r?.mean != null);
      if (points.length === 0) return;
      points.sort((a, b) => mode === 'max' ? b.r.mean - a.r.mean : a.r.mean - b.r.mean);
      const top = points[0];
      const attribution = kpiVariationAttribution(top.v);
      const feasible = feasibleIndexes
        ? feasibleIndexes.has(top.v.variation_index)
        : null;
      const altPoint = feasible === false
        ? points.find(p => feasibleIndexes.has(p.v.variation_index)) ?? null
        : null;
      const sla = kpiSlaAnnotation({
        claim,
        feasible,
        observedText: claim.filters.map(f => {
          const observed = top.v.perMetric?.[slaFilterMetricKey(f)]?.mean;
          const tag = f.metricTag ?? f.metric_tag;
          const metric = HEADLINE_METRICS.find(item => item.key === tag);
          const shown = Number.isFinite(observed) ? fmtKpi(observed, metric?.unit) : 'no value';
          return `${tag} ${f.stat ?? 'p95'} = ${shown}`;
        }),
        alternative: altPoint
          ? {
              valueText: `${fmtKpi(altPoint.r.mean, unit)} ${unit}`,
              attribution: kpiVariationAttribution(altPoint.v).text,
            }
          : null,
        attributionTitle: attribution.title,
      });
      out.push({
        label,
        unit,
        value: top.r.mean,
        cv: top.r.cv,
        variation: attribution.text,
        variationTitle: sla.title ?? attribution.title,
        slaNote: sla.note,
        slaNoteTone: sla.noteTone,
        slaTone: sla.tone,
      });
    };
    pick('output_token_throughput', 'avg', 'Peak output tok/s',  'tok/s', 'max');
    pick('request_throughput',      'avg', 'Peak req/s',         'req/s', 'max');
    pick('time_to_first_token',     'p50', 'Best TTFT p50',      'ms',    'min');
    pick('request_latency',         'p99', 'Best req lat p99',   'ms',    'min');
    return out;
  }, [variations, detail]);

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
  const liveCaption = currentCellCaption({
    currentCell,
    phaseMode,
    valuesLabel: variations.find(v => v.variation_index === currentCell?.variationIndex)?.valuesLabel,
  });
  // The winner is a property of the SWEEP, not of whatever series the chart is
  // currently showing. When the spec declares objectives (adaptive_search), rank
  // by that objective and drop SLA-infeasible variations; only fall back to the
  // chart-metric ranking for generator sweeps that declare no objective.
  // Previously this always ranked by `chartMetric.metricKey`, so selecting
  // "ITL avg" in the chart silently re-crowned the winner as the lowest-ITL
  // variation -- and even on the default metric it ignored `slaFilters`
  // entirely, promoting points that breached the constraint.
  const specSummary = detail.spec_summary ?? null;
  const presentation = sweepPresentationModel({
    phaseMode,
    sweepType: specSummary?.sweep_type,
    hasPlannerVerdict: Array.isArray(detail.search_summary?.best_trials)
      && detail.search_summary.best_trials.length > 0,
    isFailed,
    isCancelled,
  });
  const hasMultipleObjectives = Array.isArray(specSummary?.objectives)
    && specSummary.objectives.length > 1;
  const objectiveWinner = pickObjectiveWinner({
    variations,
    objectives: specSummary?.objectives,
    slaFilters: specSummary?.sla_filters,
  });
  const winner = objectiveWinner ?? pickSweepWinner({ variations, metricKey: chartMetric.metricKey });
  // Label the card with the metric the winner was actually chosen on, which is
  // the objective for adaptive sweeps and only coincidentally the chart metric.
  const winnerMetricMeta = winner
    ? (HEADLINE_METRICS.find(m => `${m.key}.${m.stat}` === winner.metricKey) ?? chartMetric.meta)
    : chartMetric.meta;

  return html`
    <div class="sweep-detail" data-testid="page-sweep-detail">
      <!-- Header -->
      <header class="sweep-detail-header" data-testid="sweep-detail-header">
        <div class="sweep-detail-header__identity">
          <div>
            <div style="display:flex;align-items:center;gap:var(--space-3);flex-wrap:wrap">
              <h1 class="sweep-detail-header__title">${s.name}</h1>
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
                    : null
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
              <span
                title=${sweepSourceTitle(s.source) ?? undefined}
                data-testid="sweep-detail-source"
                class="sweep-detail-source"
              >${s.source}</span>
            </div>
            ${liveCaption && html`
              <p class="text-dim" style="margin:var(--space-1) 0 0 0;font-size:var(--font-size-sm)" data-testid="sweep-detail-current-cell">
                ${liveCaption}
              </p>
            `}
          </div>
        </div>
        ${sweepConfig?.spec && html`
          <div style="display: flex; flex-direction: column; align-items: flex-end; gap: var(--space-1); margin-left: auto; align-self: flex-start">
            <${RelaunchButton} namespace=${namespace} name=${s.name} config=${sweepConfig} />
          </div>
        `}
      </header>

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
          const card = variationsCardModel({
            sweepType: specSummary?.sweep_type,
            totalVariations: s.total_variations,
            observedVariations: variations.length,
            phaseMode,
            // Numerator and denominator must be in the SAME unit. Both counters
            // are RUNS (variations x trials), so the denominator is
            // `status.maxTotalRuns` -- written at create time as
            // `n_variations * max_trials` (`handlers/sweep/create.py:194`) --
            // and not `totalVariations`, against which a 3-trial sweep would
            // report 100% a third of the way through. Older archives predate
            // the key, so fall back to the variation count they do carry.
            //
            // `cancelled_runs` counts toward "finished": it is a separate
            // terminal bucket from `failed_runs` on AIPerfSweep, and omitting
            // it left the bar permanently short on any sweep with a cancelled
            // child, implying work still in flight that nothing will ever do.
            finishedRuns: (s.completed_runs ?? 0) + (s.failed_runs ?? 0) + (s.cancelled_runs ?? 0),
            plannedRuns: status.maxTotalRuns || s.total_variations,
            // Lets the subtitle say "converged early" instead of the neutral
            // "stopped early", but ONLY when the planner actually reported
            // convergence. Null (no search summary, or a cancelled run) keeps
            // the weaker wording rather than crediting a convergence that did
            // not happen.
            stopKind: detail.search_summary?.stop_kind ?? null,
          });
          return html`
            <${KpiCard}
              label="Variations"
              value=${card.value}
              sub=${card.sub ?? undefined}
              title=${card.title ?? undefined}
              progress=${card.progress}
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
          const planned = status.maxTotalRuns || s.total_variations || 0;
          const nonSuccessCard = nonSuccessCardModel({
            failedRuns: failed,
            cancelledRuns: cancelled,
          });
          // The denominator here is runs that have FINISHED, not runs planned,
          // so a clean sweep always reads "N/N" and the tile looks like a
          // completion ratio it is not. The number is right; only its referent
          // was unstated. Spell it out on hover rather than widening the tile.
          const completedTitle = `${completed} of the ${completed + nonSuccess} run`
            + `${completed + nonSuccess === 1 ? '' : 's'} that have finished so far succeeded`
            + (planned > 0 ? `; the sweep plans ${planned} in total.` : '.');
          return html`
            <${KpiCard}
              label="Completed"
              value=${`${completed}/${completed + nonSuccess}`}
              title=${completedTitle}
              color=${palette.green}
              icon="check"
              tone="ok"
            />
            <${KpiCard}
              label=${nonSuccessCard.label}
              value=${nonSuccessCard.value}
              color=${nonSuccessCard.tone === 'bad' ? palette.red : palette.overlay1}
              icon="errors"
              tone=${nonSuccessCard.tone}
              sub=${nonSuccessCard.sub ?? undefined}
              title=${nonSuccessCard.title ?? undefined}
            />
          `;
        })()}
        ${headlineKpis.map((k, i) => {
          const noteColor = k.slaNoteTone === 'bad' ? palette.amber : palette.green;
          return html`
            <${KpiCard}
              key=${k.label}
              label=${k.label}
              value=${fmtKpi(k.value, k.unit)}
              unit=${k.unit}
              title=${k.variationTitle ?? undefined}
              sub=${html`
                <span
                  class="text-dim"
                  style="font-size:var(--font-size-xs);display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
                  data-testid=${'kpi-attribution-' + i}
                >${k.variation}${k.cv != null ? ` · cv ${(k.cv * 100).toFixed(1)}%` : ''}</span>
                ${k.slaNote && html`
                  <span
                    style=${'font-size:var(--font-size-xs);display:block;overflow:hidden;'
                      + 'text-overflow:ellipsis;white-space:nowrap;color:' + noteColor}
                    data-testid=${'kpi-sla-note-' + i}
                  >${k.slaNote}</span>
                `}
              `}
            />
          `;
        })}
      </div>

      <div class="sweep-detail-decision">
        ${presentation.kind === 'study' && html`
          <${SweepStudyPanel}
            presentation=${presentation}
            leader=${winner}
            metric=${winnerMetricMeta}
            phase=${phase}
          />
        `}

        ${presentation.kind === 'unavailable' && html`
          <${SweepStudyPanel}
            presentation=${presentation}
            leader=${null}
            metric=${winnerMetricMeta}
            phase=${phase}
          />
        `}

        ${presentation.showsPlannerVerdict && html`
          <${SweepWinnerSummary}
            winner=${winner}
            metric=${winnerMetricMeta}
            search=${detail.search_summary ?? null}
          />
        `}
      </div>

      ${isLiveMode && hasManifest && html`
        <${SweepLiveTrialBoard}
          manifest=${manifest}
          childSummaries=${childSummaries}
          statusRuns=${statusRuns}
        />
      `}

      <!-- Per-variation curve + table (driven by the inline aggregate manifest) -->
      ${hasManifest && html`
        <section class="card sweep-detail-analysis" data-testid="sweep-detail-variations">
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
        </section>

        ${hasMultipleObjectives && html`
        <section class="card sweep-detail-analysis" data-testid="sweep-detail-pareto">
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
        </section>
        `}
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

      <!-- Live Variations is an in-flight monitor. Once aggregate results are
           available, the variation curve and its table are the canonical view. -->
      ${isLiveMode && hasManifest && html`
        <${LiveVariationsCard} manifest=${manifest} childData=${childSummaries} statusRuns=${statusRuns} />
      `}

      <!-- Children -->
      <section class="card sweep-detail-children" data-testid="sweep-detail-children">
        <div class="card-title" style="display:flex;align-items:center;gap:var(--space-2);flex-wrap:wrap">
          <span>Children (${childRows.length})</span>
          ${childRowsAreArchived && html`
            <span
              title="These runs are from a prior sweep epoch — re-running the sweep will produce a new set."
              class="sweep-detail-archive-note"
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
      </section>

      ${(resolvedEpoch != null || artifactFiles.length > 0) && html`
      <section class="sweep-detail-artifacts">
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
          title="Aggregate artifacts"
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
      </section>
      `}

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

// ---------------------------------------------------------------------------
// aiperf:sweep-detail-pure:begin
//
// Pure presentation helpers, dependency-free BY CONTRACT: nothing in this block
// may reference an import, a constant declared elsewhere in this module, or an
// `html` template. `tests/unit/ui/test_operator_sweep_detail_presentation.py`
// slices the block out by these sentinel comments and evaluates it in bare
// node, which is the only way to get behavioural (not source-grep) coverage of
// page-level logic -- the page itself imports browser import-map specifiers
// (`htm/preact`) that node cannot resolve, and there is no bundler in the repo.
//
// The block sits at the END of the module (function declarations hoist, so the
// component above can still call these) because
// `tests/unit/ui/test_operator_performance_static.py` pins an allowlisted
// JSON.stringify by LINE NUMBER at `pages/sweep-detail.js:329`; inserting
// helpers above the component would shift that line and break an unrelated
// baseline.
// ---------------------------------------------------------------------------

/**
 * Caption describing the variation currently executing, or null when there is
 * nothing live to say.
 *
 * `status.currentCell` is written by the sweep-controller at the START of every
 * child (`sweep_controller/k8s_executor.py:453` -> `status_writer.current_cell`)
 * and is never cleared: no writer in `sweep_controller/status_writer.py` or
 * `operator/handlers/sweep/` removes the key, and the terminal writer
 * (`status_writer.aggregation_complete`) only merge-patches sibling fields. The
 * last value therefore outlives the sweep, and a finished `Succeeded` sweep kept
 * advertising "running variation 13/22". A terminal object must not render live
 * affordances -- a stale progress claim is strictly worse than no claim, because
 * the reader cannot tell it is stale. So the caption is gated on the live phase
 * mode rather than on the mere presence of `currentCell`.
 *
 * The `N/M` form is dropped as well. `currentCell.variationIndex` is a 0-based
 * IDENTIFIER (it is what names the child `<sweep>-v13`; `SweepVariation.index`
 * comes from `enumerate` in `config/sweep/expand.py:269` and from `self._iter`
 * starting at 0 in `search_planner/optuna_planner.py:225`), not a count of work
 * done. Rendering `13/22` asserts a completion ratio the numerator does not
 * measure, and is simultaneously off by one read as an ordinal. The header's
 * Variations tile already owns progress; this line owns identity, so it keeps
 * the raw index and adds what the variation is actually trying.
 */
export function currentCellCaption({ currentCell, phaseMode, valuesLabel }) {
  if (phaseMode !== 'live' || !currentCell) return null;
  const idx = currentCell.variationIndex;
  const head = idx == null ? 'running a variation' : `running variation ${idx}`;
  const descriptor = valuesLabel || currentCell.label || null;
  const trial = currentCell.trial;
  return head
    + (descriptor ? ` · ${descriptor}` : '')
    + (trial != null ? ` · trial ${trial}` : '');
}

/**
 * Select the sweep surface from lifecycle evidence, not from the incidental
 * presence of a best-trial array. A planner may emit intermediate data while
 * it is still evaluating points, and that data is useful as a current leader
 * but is not a final recommendation.
 */
export function sweepPresentationModel({
  phaseMode,
  sweepType,
  hasPlannerVerdict,
  isFailed,
  isCancelled,
}) {
  if (sweepType !== 'adaptive_search') {
    return { kind: 'variations', showsPlannerVerdict: false, leaderLabel: null };
  }
  if (phaseMode === 'live') {
    return { kind: 'study', showsPlannerVerdict: false, leaderLabel: 'Current leader' };
  }
  if (phaseMode === 'terminal' && !isFailed && !isCancelled && hasPlannerVerdict) {
    return { kind: 'result', showsPlannerVerdict: true, leaderLabel: null };
  }
  return { kind: 'unavailable', showsPlannerVerdict: false, leaderLabel: null };
}

// Sweep types whose declared variation count is a CEILING, not a measurement.
// `operator/handlers/sweep/create.py:287` sets `n_variations =
// sweep.max_iterations` for `AdaptiveSearchSweep` and says so in its own
// docstring ("don't know the final variation count up front -- only an upper
// bound"). Every generator type (grid/zip/scenarios/sobol/lhs) instead gets
// `len(expand_sweep(...))`, which is exact.
const BOUNDED_VARIATION_SWEEP_TYPES = new Set(['adaptive_search']);

export function isBoundedVariationCount(sweepType) {
  return BOUNDED_VARIATION_SWEEP_TYPES.has(String(sweepType ?? ''));
}

/**
 * Presentation model for the "Variations" KPI tile.
 *
 * Two different quantities get conflated here if you are not careful:
 *   - `totalVariations` (`status.totalVariations`) is EXACT for grid-family
 *     sweeps and an UPPER BOUND for `adaptive_search` (see
 *     `BOUNDED_VARIATION_SWEEP_TYPES`).
 *   - `observedVariations` is how many distinct variation indices the children
 *     manifest actually produced -- always a measurement.
 *
 * Never render a bound as a measurement. On the converged `gemma-bo4` sweep the
 * tile read "VARIATIONS 22" with a part-filled bar next to "COMPLETED 14/14",
 * which parses as "8 runs are missing" when in fact the search deliberately
 * stopped at 14. So once a measurement exists, bounded sweeps show it and demote
 * the ceiling to the subtitle.
 *
 * The subtitle distinguishes "converged early" from "stopped early" using
 * `stopKind`, projected from the artifact's `convergence_reason` onto
 * `SweepDetailResponse.search_summary`. Only a verdict of `converged` earns the
 * stronger word; a cancelled or incomplete run keeps the neutral "stopped
 * early", and an absent verdict also keeps it. Callers that cannot supply
 * `stopKind` get the neutral wording, so the claim never outruns the evidence.
 */
export function variationsCardModel({
  sweepType,
  totalVariations,
  observedVariations,
  phaseMode,
  finishedRuns,
  plannedRuns,
  stopKind = null,
}) {
  const declared = Number(totalVariations) > 0 ? Number(totalVariations) : 0;
  const observed = Number(observedVariations) > 0 ? Number(observedVariations) : 0;
  const done = Number(finishedRuns) > 0 ? Number(finishedRuns) : 0;
  const planned = Number(plannedRuns) > 0 ? Number(plannedRuns) : 0;
  const pct = planned > 0
    ? Math.min(100, Math.round((done / planned) * 100))
    : (done > 0 ? 100 : 0);

  if (!isBoundedVariationCount(sweepType)) {
    return {
      value: declared,
      sub: null,
      progress: pct,
      title: planned > 0 ? `${done} of ${planned} planned runs have finished.` : null,
    };
  }

  const terminal = phaseMode === 'terminal';
  let sub;
  if (observed === 0) sub = declared > 0 ? `up to ${declared}` : null;
  else if (!terminal) sub = declared > 0 ? `of up to ${declared}` : null;
  else if (declared > observed) {
    const verb = stopKind === 'converged' ? 'converged' : 'stopped';
    sub = `${verb} early · limit ${declared}`;
  }
  // declared === observed does NOT imply the cap was reached. An archived
  // adaptive sweep records totalVariations as the count it actually ran, losing
  // the original maxIterations, so a run that converged at 14 below a limit of
  // 22 arrives here indistinguishable from one that exhausted 14. Saying "hit
  // limit 14" then asserts a cap that never existed, and directly contradicts a
  // planner verdict of `converged`. Trust the verdict over the arithmetic.
  else if (stopKind === 'converged') sub = 'converged';
  else sub = declared > 0 ? `hit limit ${declared}` : null;

  return {
    value: observed > 0 ? observed : declared,
    sub,
    // A determinate bar's contract is "this fraction of a known total is done".
    // An adaptive sweep has no known total, so the bar is shown only while the
    // run is live -- where "at most this much left" is genuinely useful -- and
    // dropped at terminal, where the observed count IS the total and a
    // part-filled bar would read as truncation.
    progress: terminal ? null : pct,
    title: declared > 0
      ? `Adaptive search stops when its convergence rule fires, so ${declared} `
        + 'is the configured maximum iteration count, not a target.'
        + (observed > 0 ? ` ${observed} variation${observed === 1 ? '' : 's'} ran.` : '')
      : null,
  };
}

/**
 * Attribution line for a headline KPI tile: which variation produced the peak.
 *
 * Adaptive planners label variations `search_iter_NNNN`
 * (`search_planner/optuna_planner.py:226`), so the tiles read "PEAK REQ/S 419 /
 * search_iter_0005" -- an id that answers "which cell?" but not the question the
 * tile actually poses, which is "at what setting?". `buildSweepVariations`
 * already attaches `valuesLabel` ("concurrency=309") from `status.runs[].values`
 * (`sweep-detail-helpers.js:197-205`); the winner card and variations table lead
 * with it, and these tiles were the last place still leading with the raw id.
 *
 * Progressive disclosure: the tile surface carries the meaning, and the opaque
 * planner id stays one hover away because it is the cell identity used in
 * artifact paths -- it must remain recoverable, just not lead.
 *
 * The fallback path shortens dotted dimension paths the same way
 * `formatVariationValues` does, because a grid sweep's `variation_label` is the
 * full path (`phases.profiling.concurrency=64`) and that is 31 characters into
 * a ~134px tile caption. The prefix is identical on every variation of a sweep,
 * so it carries no information the reader can act on while costing the part
 * that does. The unshortened label stays in the tooltip.
 */
export function kpiVariationAttribution(variation) {
  const idx = variation?.variation_index;
  const rawIdentity = variation?.label || (idx != null ? `v${idx}` : '');
  const identity = shortenDimensionPaths(rawIdentity);
  const meaning = variation?.valuesLabel || '';
  const text = meaning || identity || '---';
  const title = meaning && rawIdentity && meaning !== rawIdentity
    ? `${meaning} · ${rawIdentity}`
    : (text === '---' ? null : (rawIdentity || text));
  return { text, title };
}

/**
 * Reduce `a.b.c=1, d.e=2` to `c=1, e=2`.
 *
 * Only touches the part before an `=`, so a value that itself contains dots (a
 * float, a model path) is left alone.
 */
function shortenDimensionPaths(label) {
  const raw = String(label ?? '');
  if (!raw.includes('.')) return raw;
  return raw
    .split(',')
    .map(part => {
      const eq = part.indexOf('=');
      if (eq < 0) return part;
      const key = part.slice(0, eq);
      const leaf = key.trim().split('.').pop();
      if (!leaf) return part;
      return (key.startsWith(' ') ? ' ' : '') + leaf + part.slice(eq);
    })
    .join(',');
}

// Comparators as declared on `sweep.slaFilters[*].op`, rendered for humans.
// This is a PRESENTATION map only -- the feasibility verdict itself always
// comes from `isVariationFeasible` in sweep-detail-helpers.js, so the two can
// never disagree about who passes.
const SLA_OP_SYMBOLS = { lt: '<', le: '≤', gt: '>', ge: '≥' };

/**
 * The `perMetric` key a filter is evaluated against.
 *
 * The `?? 'p95'` default MUST match `isVariationFeasible`
 * (`sweep-detail-helpers.js:367`); reading a different stat here would quote an
 * observed number that had nothing to do with the verdict shown beside it.
 */
export function slaFilterMetricKey(filter) {
  const tag = filter?.metricTag ?? filter?.metric_tag;
  return tag ? `${tag}.${filter?.stat ?? 'p95'}` : null;
}

/** Render one filter as `time_to_first_token p99 < 500`. */
export function formatSlaFilter(filter) {
  const tag = filter?.metricTag ?? filter?.metric_tag;
  const symbol = SLA_OP_SYMBOLS[filter?.op];
  if (!tag || !symbol || typeof filter?.threshold !== 'number') return null;
  return `${tag} ${filter?.stat ?? 'p95'} ${symbol} ${filter.threshold}`;
}

/**
 * What, if anything, this page is entitled to say about SLA feasibility.
 *
 *   `off`         - the sweep declared no constraints. Every `feasible` flag in
 *                   the API response is then VACUOUSLY true (see
 *                   `SweepSearchSummary.sla_filter_count`,
 *                   `routers/sweeps_models.py:401-412`), so the tiles must say
 *                   nothing at all rather than advertise an SLA nobody set.
 *   `unevaluable` - constraints existed but this page cannot check them: either
 *                   `spec_summary.sla_filters` is absent (legitimately null on
 *                   archives written before the field), or a constrained metric
 *                   is not among `HEADLINE_METRICS` so no variation carries it.
 *                   The tiles then degrade to their unannotated form and SAY SO
 *                   on hover -- silently rendering the unfiltered peak while the
 *                   winner card below is feasibility-filtered is the exact
 *                   contradiction this whole model exists to remove.
 *   `active`      - constraints exist and every constrained metric is measured
 *                   on at least one variation, so a per-variation verdict is
 *                   real evidence.
 *
 * `slaFilterCount` is authoritative when present: it is what the SEARCH
 * actually applied at trial-scoring time, read from the artifact. A spec that
 * declares filters against a search that applied none did not constrain
 * anything, and calling a point "infeasible" would contradict the run's own
 * scoring. Pass null when there is no `search_summary` to consult.
 */
export function slaClaimState({ slaFilters, slaFilterCount, variations }) {
  const filters = (Array.isArray(slaFilters) ? slaFilters : [])
    .filter(f => formatSlaFilter(f) != null);
  const applied = Number(slaFilterCount);
  if (Number.isFinite(applied) && applied === 0) {
    return { state: 'off', filters: [], declaredCount: 0 };
  }
  if (filters.length === 0) {
    const declared = Number.isFinite(applied) && applied > 0 ? applied : 0;
    return declared > 0
      ? { state: 'unevaluable', filters: [], declaredCount: declared, reason: 'definitions' }
      : { state: 'off', filters: [], declaredCount: 0 };
  }
  const measurable = filters.every(f => {
    const key = slaFilterMetricKey(f);
    return (variations ?? []).some(v => Number.isFinite(v?.perMetric?.[key]?.mean));
  });
  return measurable
    ? { state: 'active', filters, declaredCount: filters.length }
    : { state: 'unevaluable', filters, declaredCount: filters.length, reason: 'unmeasured' };
}

/**
 * Annotate a headline KPI tile with the SLA regime its number belongs to.
 *
 * The tile keeps the TRUE extremum. Filtering the tiles down to feasible points
 * was the other candidate fix and was rejected: "what is the most this server
 * can do if I ignore latency" is a real engineering question, the variations
 * table directly below is itself unfiltered, and a filtered tile would have to
 * change its own semantics between sweeps depending on whether feasibility
 * happened to be checkable -- which is precisely the silent, undeclared
 * switching the defect was about. What was actually wrong was that a peak
 * belonging to a rejected operating point sat unlabelled, in a larger font,
 * beside a feasibility-filtered winner 2.2x smaller, with nothing on the page
 * saying the two numbers were answers to different questions.
 *
 * So: the number stays, and the tile states which regime it is in. An
 * infeasible peak also drops its gold "award" tone for a warning tone and names
 * the feasible alternative inline, so the two numbers cannot be read as
 * competing claims about the same question.
 *
 * @param {object} args
 * @param {object} args.claim - Output of `slaClaimState`.
 * @param {boolean|null} args.feasible - Verdict for the attributed variation,
 *   from `isVariationFeasible`; null when `claim.state` is not `active`.
 * @param {string[]} args.observedText - One pre-formatted
 *   `"time_to_first_token p99 = 18,380.0"` per constrained metric, measured on
 *   the attributed variation. The limit itself is not repeated here; the
 *   sentence already carries it.
 * @param {{valueText: string, attribution: string}|null} args.alternative -
 *   Best SLA-feasible point for this same metric, when the tile's own point is
 *   infeasible and a feasible one exists.
 * @param {string|null} args.attributionTitle - Existing hover text to extend.
 * @returns {{note: string|null, noteTone: ('bad'|'ok')|null, tone: string|null,
 *   title: string|null}}
 */
export function kpiSlaAnnotation({ claim, feasible, observedText, alternative, attributionTitle }) {
  const head = attributionTitle ? `${attributionTitle}. ` : '';
  const state = claim?.state ?? 'off';
  if (state === 'off') {
    return { note: null, noteTone: null, tone: null, title: attributionTitle ?? null };
  }
  if (state === 'unevaluable') {
    const n = claim?.declaredCount ?? 0;
    const why = claim?.reason === 'unmeasured'
      ? 'the constrained metric is not among the metrics this page collects'
      : 'this sweep record does not carry their definitions';
    return {
      note: null,
      noteTone: null,
      tone: null,
      title: head
        + `This sweep applied ${n} SLA constraint${n === 1 ? '' : 's'}, but `
        + `${why}, so this tile is NOT filtered for feasibility.`,
    };
  }
  const constraints = claim.filters.map(formatSlaFilter).join('; ');
  if (feasible === false) {
    // Scoped to "of the variations charted here" on purpose. This walks the
    // page's own per-variation means, whereas the winner card ranks the
    // planner's recorded per-trial objective values; the two are normally the
    // same number but are not the same measurement, so an unqualified "the best
    // SLA-feasible value" would be a second unlabelled headline of exactly the
    // kind this annotation exists to prevent.
    const alt = alternative
      ? ` Of the variations charted here the best SLA-feasible value is `
        + `${alternative.valueText} at ${alternative.attribution}; see the `
        + 'winner summary for the sweep\'s own verdict.'
      : ' No variation charted here satisfied the SLA on this metric.';
    return {
      note: 'breaches SLA',
      noteTone: 'bad',
      tone: 'warn',
      title: head
        + 'This is the extremum across EVERY variation, including points the '
        + `constrained search rejected. It breaches the sweep's SLA (${constraints})`
        + (observedText?.length ? `: observed ${observedText.join('; ')}.` : '.')
        + alt,
    };
  }
  return {
    note: 'meets SLA',
    noteTone: 'ok',
    tone: null,
    title: head + `Satisfies the sweep's SLA (${constraints}).`,
  };
}

/**
 * Presentation model for the non-success tile.
 *
 * `cancelled` is a distinct terminal bucket from `failed` on AIPerfSweep -- the
 * schema says so explicitly (`routers/sweeps_models.py:80-88`: "Kept separate
 * from `failed_runs` so user-cancelled children are not counted as failures").
 * The tile nonetheless summed them into one red "FAILED" number, so a sweep the
 * user cancelled on purpose reported failures it never had.
 *
 * Principle: a label is an assertion about what the number means. Cancellation
 * is an operator action, not a defect, and colouring it red invites someone to
 * go hunting for a root cause that does not exist. The failure count therefore
 * owns the tile only when there are failures; a purely cancelled sweep relabels
 * and de-escalates, and a mixed one keeps failures as the headline with
 * cancellations demoted to the subtitle. The total that finished remains
 * derivable from the Completed tile's denominator.
 */
export function nonSuccessCardModel({ failedRuns, cancelledRuns }) {
  const failed = Number(failedRuns) > 0 ? Number(failedRuns) : 0;
  const cancelled = Number(cancelledRuns) > 0 ? Number(cancelledRuns) : 0;
  if (failed === 0 && cancelled > 0) {
    return {
      label: 'Cancelled',
      value: cancelled,
      tone: 'neutral',
      sub: null,
      title: `${cancelled} run${cancelled === 1 ? ' was' : 's were'} cancelled. `
        + 'Cancellation is an operator action, not a failure.',
    };
  }
  return {
    label: 'Failed',
    value: failed,
    tone: failed > 0 ? 'bad' : 'neutral',
    sub: cancelled > 0 ? `+${cancelled} cancelled` : null,
    title: cancelled > 0
      ? `${failed} run${failed === 1 ? '' : 's'} failed; ${cancelled} `
        + `${cancelled === 1 ? 'was' : 'were'} cancelled and ${cancelled === 1 ? 'is' : 'are'} `
        + 'counted separately.'
      : null,
  };
}

/**
 * Explain the bare `live` / `archived` / `both` provenance chip in the header.
 *
 * `SweepSummary.source` (`routers/sweeps_models.py:75`) is an internal
 * discriminator that the page renders verbatim as a chip with no surrounding
 * label. `both` in particular is unreadable from the outside -- it means the
 * live CR and a persisted aggregate were merged by `sweep_union._merge`
 * (`sweep_union.py:274-310`), with the live CR authoritative for every numeric
 * counter and the archive backfilling only identity fields. Provenance belongs
 * on the page, but a reader should not have to grep the operator to decode a
 * six-letter token, so the meaning goes on hover.
 */
export function sweepSourceTitle(source) {
  switch (String(source ?? '')) {
    case 'live':
      return 'Read from the live AIPerfSweep resource in the cluster.';
    case 'archived':
      return 'Read from the persisted aggregate on the operator\'s PVC; '
        + 'the live resource is gone.';
    case 'both':
      return 'Live resource and persisted aggregate both exist. Counters come '
        + 'from the live resource; the archive supplies artifacts and any '
        + 'identity fields the live resource omits.';
    default:
      return null;
  }
}

// aiperf:sweep-detail-pure:end

function SweepStudyPanel({ presentation, leader, metric, phase }) {
  if (presentation.kind === 'study') {
    return html`
      <section class="sweep-study-status" data-testid="sweep-study-status">
        <div>
          <div class="sweep-study-status__eyebrow">Adaptive search</div>
          <h3>Optimization study in progress</h3>
          <p>The planner is still sampling the search space. This readout is provisional, not a final recommendation.</p>
        </div>
        <div class="sweep-study-status__readout">
          <div class="sweep-study-status__eyebrow">${presentation.leaderLabel}</div>
          ${leader
            ? html`
              <div class="sweep-study-status__leader">${leader.valuesLabel || leader.label || `variation ${leader.variation_index}`}</div>
              <div class="sweep-study-status__metric">
                ${metric.label} · ${fmtKpi(leader.mean, metric.unit)} ${metric.unit}
              </div>
            `
            : html`<div class="sweep-study-status__metric">Awaiting the first completed observation.</div>`}
        </div>
      </section>
    `;
  }

  if (presentation.kind === 'unavailable') {
    const stoppedWithoutVerdict = phase === 'Failed' || phase === 'Cancelled';
    return html`
      <section class="sweep-study-status sweep-study-status--muted" data-testid="sweep-study-no-verdict">
        <div>
          <div class="sweep-study-status__eyebrow">Adaptive search</div>
          <h3>No final recommendation</h3>
          <p>${stoppedWithoutVerdict
            ? `${phase} sweeps do not publish a final operating point.`
            : 'The sweep ended without a planner verdict. Review the trial history and search artifact before choosing an operating point.'}</p>
        </div>
      </section>
    `;
  }

  return null;
}
