// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { useState, useEffect, useRef, useMemo } from 'preact/hooks';
import { api } from '../lib/api.js';
import { palette, modelColor } from '../lib/theme.js';
import { query, setQuery } from '../lib/router.js';
import { ChartWrapper } from '../components/chart-wrapper.js';
import { LoadingPanel, Spinner } from '../components/spinner.js';
import { fmtMilliseconds, fmtNumber, fmtReqPerSecond } from '../lib/format.js';
import { CHART_TYPOGRAPHY } from '../lib/typography.js';
import { applyJobFilters, extractFacets, extractCrossFacets, FILTER_NONE } from './compare-filters.js';
export { applyJobFilters, extractFacets, extractCrossFacets, FILTER_NONE };

// Metrics where lower is better
const LOWER_IS_BETTER = new Set([
  'request_latency',
  'time_to_first_token',
  'inter_token_latency',
]);

const JOB_COLORS = [
  palette.mauve,
  palette.blue,
  palette.green,
  palette.peach,
  palette.pink,
  palette.teal,
  palette.sapphire,
  palette.yellow,
];

/**
 * Best value per comparable group, for one metric row of the compare table.
 *
 * "Best" ranks the members of a set against each other, which requires them to
 * be alternatives. Selecting a 1B and a 70B run puts both in the table -- a
 * legitimate side-by-side reading of the numbers -- but marking the 1B run's
 * throughput as the winner reports its size, not its quality. So the marker is
 * scoped to the group it can speak for: same namespace, same model.
 *
 * Groups with a single run get no marker at all: crowning the only candidate
 * in a group is a ranking with nothing to rank.
 *
 * @param {string} metric metric name, for direction lookup
 * @param {Record<string, number|null>} values metric value keyed by "<ns>/<job>"
 * @param {Record<string, string>} clusterByKey comparable-group key per run key
 * @returns {Map<string, number>} group key -> best value in that group
 */
function bestValuePerCluster(metric, values, clusterByKey) {
  const lowerIsBetter = LOWER_IS_BETTER.has(metric);
  const counts = new Map();
  for (const cluster of Object.values(clusterByKey)) {
    counts.set(cluster, (counts.get(cluster) ?? 0) + 1);
  }
  const best = new Map();
  for (const [key, value] of Object.entries(values ?? {})) {
    if (value == null) continue;
    const cluster = clusterByKey[key];
    if (cluster == null || (counts.get(cluster) ?? 0) < 2) continue;
    const current = best.get(cluster);
    if (current == null || (lowerIsBetter ? value < current : value > current)) {
      best.set(cluster, value);
    }
  }
  return best;
}

/**
 * Explain what the green "best" markers rank against. Prose lives here rather
 * than inside the ``htm`` template literal.
 */
function tableScopeNote(clusterByKey) {
  const counts = new Map();
  for (const cluster of Object.values(clusterByKey ?? {})) {
    counts.set(cluster, (counts.get(cluster) ?? 0) + 1);
  }
  const rankable = [...counts.values()].filter((n) => n >= 2).length;
  const groups = counts.size;
  if (rankable === 0) {
    return `These ${groups} runs are each the only one of their namespace x model, so nothing is marked best - there is no like-for-like comparison to make.`;
  }
  return `Green marks the best value within each namespace x model group (${groups} groups here). Runs of different models are shown side by side but never ranked against each other.`;
}

function formatNum(v, unit = '') {
  if (v == null) return '\u2014';
  if (typeof v !== 'number') return String(v);
  if (unit === 'ms') return fmtMilliseconds(v);
  if (unit === 'req/s' || unit === 'req/s/GPU') return fmtReqPerSecond(v);
  // Tiny per-GPU values (e.g. 0.04 req/s/GPU on 1024 GPUs) round to "0.000"
  // at 3 decimals. Bump precision when the magnitude requires it.
  const abs = Math.abs(v);
  if (abs > 0 && abs < 0.01) return fmtNumber(v, 5);
  if (abs > 0 && abs < 1) return fmtNumber(v, 4);
  return fmtNumber(v, 3);
}

function formatAxisTick(value, unit) {
  const number = Number(value);
  return Number.isFinite(number) ? formatNum(number, unit) : String(value);
}

// Human-friendly metric labels for tooltip hints. Falls back to the
// machine name when nothing matches so a missing entry just hides the
// hint rather than breaking layout.
const METRIC_DESCRIPTIONS = {
  request_throughput: 'Successful requests completed per second.',
  request_latency: 'End-to-end request latency (request sent \u2192 final token received).',
  output_token_throughput: 'Output tokens generated per second across the whole run.',
  output_token_throughput_per_user: 'Output tokens per second observed by a single user/session.',
  total_token_throughput: 'Input + output tokens processed per second.',
  time_to_first_token: 'Time from request sent to the first streamed token.',
  inter_token_latency: 'Time between successive streamed output tokens.',
  output_sequence_length: 'Number of output tokens per response.',
  input_sequence_length: 'Number of input tokens per request.',
};

// Look up one (metric, stat) row from the pivoted compare response.
function findEntry(entries, metric, stat) {
  if (!Array.isArray(entries)) return null;
  return entries.find((e) => e.metric === metric && e.stat === stat) ?? null;
}

function finiteMetricValue(value) {
  if (value == null) return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

// Hardware colors — points on the InferenceX-style Pareto are colored by
// GPU family rather than per-job, so identical accelerators cluster
// visually. Falls back to a stable hash-based color for unknowns.
const GPU_FAMILY_COLORS = {
  B200: palette.green,
  GB200: palette.teal,
  H200: palette.blue,
  H100: palette.sapphire,
  GH200: palette.lavender,
  A100: palette.peach,
  L40S: palette.yellow,
  L40: palette.yellow,
  L4: palette.maroon,
  A10G: palette.maroon,
  MI300X: palette.red,
  MI325X: palette.red,
};

// Pull a short family tag (e.g. "H100") out of a DCGM model string like
// "NVIDIA H100 80GB HBM3" or "NVIDIA H200". Returns null when nothing
// matches so callers can fall back to the raw name.
function gpuFamily(name) {
  if (!name || typeof name !== 'string') return null;
  const upper = name.toUpperCase();
  const known = [
    'GB200', 'B200', 'GH200', 'H200', 'H100', 'A100',
    'L40S', 'L40', 'L4', 'A10G', 'MI325X', 'MI300X',
  ];
  for (const tag of known) {
    if (upper.includes(tag)) return tag;
  }
  return null;
}

function gpuColor(name) {
  const fam = gpuFamily(name);
  if (fam && GPU_FAMILY_COLORS[fam]) return GPU_FAMILY_COLORS[fam];
  // Stable hash → palette pick for unknown GPUs.
  let h = 0;
  for (let i = 0; i < (name || '').length; i++) h = (h * 31 + name.charCodeAt(i)) | 0;
  return JOB_COLORS[Math.abs(h) % JOB_COLORS.length];
}

// Identity of a comparable group: same namespace, same model. Declared here
// because both the InferenceX scatters and the Pareto Lab key off it.
const clusterKeyOf = (ns, model) => `${ns || 'unknown'} · ${model || 'unknown'}`;

// Build (x, y, label, color) points for a two-metric scatter. ``yPerGpu``
// flag divides the Y value by the run's gpu_count (skipping runs that
// lack telemetry). Color is keyed by GPU family so identical hardware
// clusters on the chart.
function buildScatterPoints(entries, x, y, displayKeys, splitKey, meta, yPerGpu) {
  const xEntry = findEntry(entries, x.metric, x.stat);
  const yEntry = findEntry(entries, y.metric, y.stat);
  if (!xEntry || !yEntry) return [];
  const points = [];
  for (let i = 0; i < displayKeys.length; i++) {
    const key = displayKeys[i];
    const xValue = finiteMetricValue(xEntry.values?.[key]);
    const yValue = finiteMetricValue(yEntry.values?.[key]);
    if (xValue == null || yValue == null) continue;
    const m = (meta && meta[key]) || {};
    const gpuCount = Number(m.gpu_count) || 0;
    if (yPerGpu && gpuCount <= 0) continue;
    const yScaled = yPerGpu ? yValue / gpuCount : yValue;
    if (!Number.isFinite(yScaled)) continue;
    const gpuName = m.gpu_name || null;
    const color = gpuName ? gpuColor(gpuName) : JOB_COLORS[i % JOB_COLORS.length];
    const { ns, jobId } = splitKey(key);
    const model = m.model || 'unknown';
    points.push({
      x: xValue,
      y: yScaled,
      key,
      jobName: jobId,
      ns: ns || 'unknown',
      model,
      // Two runs are alternatives for one another only when they benchmark
      // the same model in the same namespace; see clusterFrontiers below.
      clusterKey: clusterKeyOf(ns, model),
      gpuCount,
      gpuName,
      gpuFamily: gpuFamily(gpuName) || gpuName || 'unknown',
      color,
    });
  }
  return points;
}

// Compute the Pareto-non-dominated subset of points, sorted by x ascending
// for clean line drawing. ``larger`` flags say which direction is "better"
// on each axis. O(n^2) \u2014 n is bounded by the number of selected runs.
function paretoFrontier(points, xLargerBetter, yLargerBetter) {
  if (!points || points.length < 2) return points ? points.slice() : [];
  const dominates = (a, b) => {
    const xBeat = xLargerBetter ? a.x >= b.x : a.x <= b.x;
    const yBeat = yLargerBetter ? a.y >= b.y : a.y <= b.y;
    const strict = a.x !== b.x || a.y !== b.y;
    return xBeat && yBeat && strict;
  };
  return points
    .filter((p) => !points.some((q) => q !== p && dominates(q, p)))
    .slice()
    .sort((a, b) => a.x - b.x);
}

/**
 * Group points by (namespace x model) and compute a frontier inside each group.
 *
 * A Pareto frontier is a claim that the runs on it are the non-dominated
 * *alternatives* for one decision. That claim only holds within a comparable
 * set: a 1B model will dominate a 70B model on every throughput axis without
 * being a substitute for it, so a single frontier drawn over a mixed selection
 * reports a scale difference as if it were an efficiency result. This is the
 * same grouping the Pareto Lab on this page already applies
 * (``buildLabDatasets``); the InferenceX scatters above it were the outlier.
 *
 * Singleton clusters yield no frontier -- one point is not a frontier, and
 * connecting it to a point from a different model would be exactly the
 * cross-run comparison we are avoiding.
 */
function clusterFrontiers(points, xLargerBetter, yLargerBetter) {
  const groups = new Map();
  for (const p of points) {
    const ck = p.clusterKey ?? 'unknown';
    if (!groups.has(ck)) groups.set(ck, { clusterKey: ck, model: p.model, points: [] });
    groups.get(ck).points.push(p);
  }
  const out = [];
  for (const group of groups.values()) {
    if (group.points.length < 2) continue;
    const frontier = paretoFrontier(group.points, xLargerBetter, yLargerBetter);
    if (frontier.length >= 2) {
      out.push({ clusterKey: group.clusterKey, model: group.model, frontier });
    }
  }
  return out;
}

// "Pareto Lab" axis presets \u2014 ported from operator/ui/views/analysis.js. Each
// preset names an (x, y) metric/stat pair and the optimization direction so
// per-cluster frontiers know which way is "better." Same three presets as
// the legacy ui so users muscle-memory translates between the two.
const LAB_AXES = [
  {
    key: 'tps_p99',
    label: 'Throughput \u00d7 Latency p99',
    x: { metric: 'request_throughput', stat: 'avg', label: 'Throughput', unit: 'req/s', largerBetter: true },
    y: { metric: 'request_latency', stat: 'p99', label: 'Latency p99', unit: 'ms', largerBetter: false },
  },
  {
    key: 'tps_ttft',
    label: 'Throughput \u00d7 TTFT',
    x: { metric: 'request_throughput', stat: 'avg', label: 'Throughput', unit: 'req/s', largerBetter: true },
    y: { metric: 'time_to_first_token', stat: 'avg', label: 'TTFT', unit: 'ms', largerBetter: false },
  },
  {
    key: 'tok_p99',
    label: 'Token Throughput \u00d7 Latency p99',
    x: { metric: 'output_token_throughput', stat: 'avg', label: 'Token Throughput', unit: 'tok/s', largerBetter: true },
    y: { metric: 'request_latency', stat: 'p99', label: 'Latency p99', unit: 'ms', largerBetter: false },
  },
];

const shortModel = (m) => (m ? String(m).split('/').pop() : 'unknown');
const MUTED_CLUSTER_COLOR = palette.overlay0;



// Build (x, y) points for the lab Pareto, tagging each with its (ns, model)
// cluster so the renderer can group + color per cluster instead of per-job.
// Mirrors the analysis.js cluster-then-frontier shape rather than
// buildScatterPoints' GPU-family-coloring shape.
function buildLabPoints(entries, axis, displayKeys, splitKey, meta) {
  const xEntry = findEntry(entries, axis.x.metric, axis.x.stat);
  const yEntry = findEntry(entries, axis.y.metric, axis.y.stat);
  if (!xEntry || !yEntry) return [];
  const points = [];
  for (const key of displayKeys) {
    const xValue = finiteMetricValue(xEntry.values?.[key]);
    const yValue = finiteMetricValue(yEntry.values?.[key]);
    if (xValue == null || yValue == null) continue;
    const { ns, jobId } = splitKey(key);
    const model = meta[key]?.model || 'unknown';
    points.push({
      x: xValue,
      y: yValue,
      key,
      jobName: jobId,
      ns: ns || 'unknown',
      model,
      clusterKey: clusterKeyOf(ns, model),
    });
  }
  return points;
}

// Build chart.js datasets for the clustered Pareto: one scatter dataset per
// active cluster (model-colored), plus a per-cluster dashed frontier line
// when the cluster has \u22652 points. Singletons render in a muted color so
// they're visible but visually de-emphasized.
function buildLabDatasets(points, axis, activeClusters) {
  const groups = {};
  for (const p of points) {
    if (activeClusters && !activeClusters.has(p.clusterKey)) continue;
    (groups[p.clusterKey] ??= { ns: p.ns, model: p.model, points: [] }).points.push(p);
  }
  const datasets = [];
  for (const [ck, grp] of Object.entries(groups)) {
    const isSingleton = grp.points.length < 2;
    const color = isSingleton ? MUTED_CLUSTER_COLOR : modelColor(grp.model);
    datasets.push({
      label: ck,
      data: grp.points,
      backgroundColor: color,
      borderColor: color,
      borderWidth: 1.4,
      pointRadius: 7,
      pointHoverRadius: 11,
      showLine: false,
      order: 1,
    });
    if (!isSingleton) {
      const frontier = paretoFrontier(grp.points, axis.x.largerBetter, axis.y.largerBetter);
      if (frontier.length >= 2) {
        datasets.push({
          label: `${ck} \u00b7 frontier`,
          data: frontier.map((p) => ({ x: p.x, y: p.y, jobName: p.jobName, clusterKey: ck })),
          borderColor: color,
          backgroundColor: color,
          borderWidth: 1.6,
          borderDash: [4, 4],
          showLine: true,
          pointRadius: 0,
          pointHoverRadius: 0,
          fill: false,
          order: 2,
          legend: false,
        });
      }
    }
  }
  return datasets;
}

/**
 * Prose for what the dashed lines do and do not assert.
 *
 * Kept as a plain function rather than inline markup so the sentences live
 * outside the ``htm`` template literal.
 */
function frontierNote(points, frontiers) {
  if (!points || points.length === 0) return null;
  const clusters = new Set(points.map((p) => p.clusterKey ?? 'unknown')).size;
  if (frontiers.length === 0) {
    return clusters > 1
      ? `No frontier drawn: the ${points.length} plotted runs span ${clusters} namespace x model groups, none with two or more runs. A frontier only means something among runs that are alternatives for the same deployment.`
      : 'Add another run for a frontier - a Pareto needs at least two comparable points.';
  }
  if (frontiers.length > 1) {
    return `${frontiers.length} frontiers, one per namespace x model. Runs of different models are not substitutes for one another, so they are never joined into a single frontier.`;
  }
  return null;
}

function buildLabOptions(axis) {
  return {
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: palette.mantle,
        titleColor: palette.text,
        bodyColor: palette.text,
        borderColor: palette.surface0,
        borderWidth: 1,
        callbacks: {
          label: (ctx) => {
            const p = ctx.raw;
            const xs = `${axis.x.label}: ${formatNum(p.x, axis.x.unit)}${axis.x.unit ? ' ' + axis.x.unit : ''}`;
            const ys = `${axis.y.label}: ${formatNum(p.y, axis.y.unit)}${axis.y.unit ? ' ' + axis.y.unit : ''}`;
            const ck = p.clusterKey || ctx.dataset.label || '';
            const head = p.jobName ? `${ck} \u00b7 ${p.jobName}` : ck;
            return head ? `${head} \u2014 ${xs}, ${ys}` : `${xs}, ${ys}`;
          },
        },
      },
    },
    scales: {
      x: {
        type: 'linear',
        title: {
          display: true,
          text: axis.x.unit ? `${axis.x.label} (${axis.x.unit})` : axis.x.label,
          color: palette.overlay1,
          font: { size: CHART_TYPOGRAPHY.AXIS_LABEL },
        },
        grid: { color: palette.surface0 + '40' },
        ticks: { color: palette.overlay0, font: { size: CHART_TYPOGRAPHY.AXIS_TICK }, callback: value => formatAxisTick(value, axis.x.unit) },
      },
      y: {
        type: 'linear',
        title: {
          display: true,
          text: axis.y.unit ? `${axis.y.label} (${axis.y.unit})` : axis.y.label,
          color: palette.overlay1,
          font: { size: CHART_TYPOGRAPHY.AXIS_LABEL },
        },
        grid: { color: palette.surface0 + '40' },
        ticks: { color: palette.overlay0, font: { size: CHART_TYPOGRAPHY.AXIS_LABEL }, callback: value => formatAxisTick(value, axis.y.unit) },
      },
    },
  };
}

// Build a chart.js scatter config from a point set + one frontier line per
// comparable (namespace x model) cluster.
function buildScatterChart(points, frontiers, x, y) {
  const datasets = [
    {
      label: 'runs',
      data: points,
      pointBackgroundColor: points.map((p) => p.color),
      pointBorderColor: points.map((p) => p.color),
      pointRadius: 7,
      pointHoverRadius: 11,
      showLine: false,
      order: 1,
      legend: false,
    },
  ];
  const multiCluster = (frontiers ?? []).length > 1;
  for (const { clusterKey, model, frontier } of frontiers ?? []) {
    // Color by model when several frontiers share the canvas, so two dashed
    // lines can't be mistaken for one frontier over the whole selection.
    const color = multiCluster ? modelColor(model) : palette.mauve;
    datasets.push({
      label: `${clusterKey} · frontier`,
      data: frontier.map((p) => ({ x: p.x, y: p.y, jobName: p.jobName, clusterKey })),
      borderColor: color,
      backgroundColor: color,
      borderWidth: 1.6,
      borderDash: [4, 4],
      showLine: true,
      pointRadius: 0,
      pointHoverRadius: 0,
      fill: false,
      order: 2,
      legend: false,
    });
  }

  const options = {
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: palette.mantle,
        titleColor: palette.text,
        bodyColor: palette.text,
        borderColor: palette.surface0,
        borderWidth: 1,
        callbacks: {
          label: (ctx) => {
            const p = ctx.raw;
            const xs = `${x.label}: ${formatNum(p.x, x.unit)}${x.unit ? ' ' + x.unit : ''}`;
            const ys = `${y.label}: ${formatNum(p.y, y.unit)}${y.unit ? ' ' + y.unit : ''}`;
            const head = p.jobName || '';
            const hw = p.gpuCount && p.gpuFamily
              ? ` \u00b7 ${p.gpuCount}\u00d7 ${p.gpuFamily}`
              : '';
            // Name the comparable group on every point: which frontier a point
            // belongs to is not inferable from position alone once more than
            // one is drawn.
            const cluster = p.clusterKey ? ` \u00b7 ${p.clusterKey}` : '';
            return head ? `${head}${cluster}${hw} \u2014 ${xs}, ${ys}` : `${xs}, ${ys}`;
          },
        },
      },
    },
    scales: {
      x: {
        type: 'linear',
        title: {
          display: true,
          text: x.unit ? `${x.label} (${x.unit})` : x.label,
          color: palette.overlay1,
          font: { size: CHART_TYPOGRAPHY.AXIS_LABEL },
        },
        grid: { color: palette.surface0 + '40' },
        ticks: { color: palette.overlay0, font: { size: CHART_TYPOGRAPHY.AXIS_TICK }, callback: value => formatAxisTick(value, x.unit) },
      },
      y: {
        type: 'linear',
        title: {
          display: true,
          text: y.unit ? `${y.label} (${y.unit})` : y.label,
          color: palette.overlay1,
          font: { size: CHART_TYPOGRAPHY.AXIS_LABEL },
        },
        grid: { color: palette.surface0 + '40' },
        ticks: { color: palette.overlay0, font: { size: CHART_TYPOGRAPHY.AXIS_LABEL }, callback: value => formatAxisTick(value, y.unit) },
      },
    },
  };

  return { datasets, options };
}

export function Compare() {
  const [storedJobs, setStoredJobs] = useState([]);
  const [jobsLoading, setJobsLoading] = useState(true);
  const [jobsError, setJobsError] = useState(null);

  const [search, setSearch] = useState('');
  // selectedKeys are composite "<namespace>/<job_id>" strings — this is the key
  // format used by the backend's compare response (_pivot_compare_rows).
  const [selectedKeys, setSelectedKeys] = useState([]);

  const [compareData, setCompareData] = useState(null);
  const [comparing, setComparing] = useState(false);
  const [compareError, setCompareError] = useState(null);

  // Chip-strip overflow collapse: above 6 selections we hide the tail behind
  // a "+N more" pill so the strip doesn't wrap into a wall of chips.
  const [chipsExpanded, setChipsExpanded] = useState(false);

  const [nsFilter, setNsFilter] = useState(new Set());
  const [modelFilter, setModelFilter] = useState(new Set());
  const [endpointFilter, setEndpointFilter] = useState(new Set());
  // Per-dimension overflow-collapse (mirrors chipsExpanded for the
  // selection chip-strip below; each filter row collapses past 6 chips).
  const [facetExpanded, setFacetExpanded] = useState({ ns: false, model: false, endpoint: false });

  // Pareto Lab state — ported from operator/ui/views/analysis.js. The axis
  // preset is the active (x, y) metric pair; ``labActiveClusters = null``
  // means "show all clusters" (default), and a populated Set means the
  // user has toggled chip-bar visibility for individual clusters.
  const [labAxisKey, setLabAxisKey] = useState('tps_p99');
  const [labActiveClusters, setLabActiveClusters] = useState(null);

  // ``?cluster=<ns> · <model>`` deep-link from the job-detail "+N similar
  // runs" chip. Ports the legacy ui's IdentityStrip → Compare flow (see
  // src/aiperf/operator/ui/views/run.js::SimilarRunsLink). The /results
  // endpoint now carries ``model`` directly on each entry (sourced
  // server-side from ``job_spec.json``), so we filter ``storedJobs``
  // without any client-side cross-reference. Comparability is count-only
  // — we never aggregate metrics across independent benchmarks.
  const deepLinkAppliedRef = useRef(false);

  // When the deep-link lands, we hold onto the originating cluster label so
  // the UI can show "Comparing all runs in cluster <ns> · <model>" context
  // (matched > 0) or "no stored runs" guidance (matched === 0). Cleared the
  // moment the user edits selection, so the URL and visible state never lie
  // to each other.
  const [activeClusterLabel, setActiveClusterLabel] = useState(null);
  const [unmatchedClusterLabel, setUnmatchedClusterLabel] = useState(null);

  let firstJobsLoadDone = false;

  useEffect(() => {
    let stopped = false;
    setJobsLoading(true);
    api
      .listResults()
      .then((resp) => {
        if (stopped) return;
        setStoredJobs(resp?.jobs ?? resp ?? []);
        setJobsError(null);
        setJobsLoading(false);
        firstJobsLoadDone = true;
      })
      .catch((err) => {
        if (stopped) return;
        setJobsError(err.message);
        setJobsLoading(false);
      });
    return () => { stopped = true; };
  }, []);

  // Apply the ?cluster= deep-link once stored-jobs lands.
  useEffect(() => {
    if (deepLinkAppliedRef.current) return;
    if (jobsLoading) return;
    const cluster = query.value.cluster;
    if (!cluster) return;
    // Same separator the legacy ui writes: "<ns> · <model>" (spaced
    // middle-dot). The router hands the value back already URL-decoded.
    const sep = ' · ';
    const idx = cluster.indexOf(sep);
    if (idx < 0) {
      deepLinkAppliedRef.current = true;
      return;
    }
    const ns = cluster.slice(0, idx);
    const model = cluster.slice(idx + sep.length);
    if (!ns || !model) {
      deepLinkAppliedRef.current = true;
      return;
    }
    const matches = storedJobs
      .filter((j) => j.namespace === ns && j.model === model)
      .map((j) => compositeKey(j));
    deepLinkAppliedRef.current = true;
    if (matches.length === 0) {
      // Tell the user their deep-link wasn't ignored — the cluster simply
      // has no stored runs yet. Banner renders above the normal empty-state.
      setUnmatchedClusterLabel(cluster);
      return;
    }
    setActiveClusterLabel(cluster);
    setNsFilter(new Set([ns]));
    setModelFilter(new Set([model]));
    setSelectedKeys(matches);
    if (matches.length >= 2) {
      // Auto-trigger the compare so the user lands on a populated view
      // instead of having to click Compare. Defer to next tick so the
      // setSelectedKeys above flushes before runCompareWithKeys runs.
      // Track the timeout so the effect's cleanup can cancel it if the
      // component unmounts before the tick fires (avoids setState on
      // unmounted component during fast nav-away/nav-back).
      const id = setTimeout(() => runCompareWithKeys(matches), 0);
      return () => clearTimeout(id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [storedJobs, jobsLoading]);

  // Extracted so the deep-link effect can drive the API call without
  // depending on the local handleCompare closure (which reads state).
  let firstCompareLoadDone = false;
  async function runCompareWithKeys(keys) {
    if (keys.length < 2) return;
    setComparing(true);
    setCompareError(null);
    setCompareData(null);
    try {
      const resp = await api.compareJobs(keys);
      setCompareData(resp);
      firstCompareLoadDone = true;
    } catch (err) {
      setCompareError(err.message);
    } finally {
      setComparing(false);
    }
  }

  function compositeKey(job) {
    const id = job.job_id ?? '';
    const ns = job.namespace ?? '';
    return ns ? `${ns}/${id}` : id;
  }

  function splitKey(key) {
    const idx = key.indexOf('/');
    return idx < 0 ? { ns: '', jobId: key } : { ns: key.slice(0, idx), jobId: key.slice(idx + 1) };
  }

  // Drop the deep-link pill + URL ?cluster= param the moment the user takes
  // any selection action. Otherwise the pill claims "Comparing all runs in
  // cluster X" while the actual selection has been edited — confusing, and
  // a back/refresh would re-apply the stale cluster.
  function clearDeepLinkContext() {
    if (activeClusterLabel) setActiveClusterLabel(null);
    if (unmatchedClusterLabel) setUnmatchedClusterLabel(null);
    if (query.value.cluster) setQuery({ cluster: '' });
  }

  function toggleJob(key) {
    clearDeepLinkContext();
    setCompareData(null);
    setCompareError(null);
    setSelectedKeys((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key],
    );
  }

  function clearSelection() {
    clearDeepLinkContext();
    setSelectedKeys([]);
    setCompareData(null);
    setCompareError(null);
  }

  // Quick-pick: take the most recent N completed runs (by start_time desc,
  // falling back to original list order). Bridges the cold-start gap between
  // "0 selected" and "see Pareto" — without it, first-time users have to hunt
  // for jobs by name before they can see anything.
  function selectRecent(n) {
    clearDeepLinkContext();
    setCompareData(null);
    setCompareError(null);
    const sorted = [...filtered].sort((a, b) => {
      const ta = a?.start_time ? Date.parse(a.start_time) : 0;
      const tb = b?.start_time ? Date.parse(b.start_time) : 0;
      return tb - ta;
    });
    const picks = sorted.slice(0, n).map(compositeKey).filter(Boolean);
    if (picks.length >= 2) setSelectedKeys(picks);
  }

  function toggleFacet(setFn, value) {
    setFn((prev) => {
      const next = new Set(prev);
      if (next.has(value)) next.delete(value); else next.add(value);
      return next;
    });
  }

  function clearFilters() {
    setNsFilter(new Set());
    setModelFilter(new Set());
    setEndpointFilter(new Set());
    setSearch('');
  }

  function toggleFacetExpanded(dim) {
    setFacetExpanded((prev) => ({ ...prev, [dim]: !prev[dim] }));
  }

  async function handleCompare() {
    if (selectedKeys.length < 2) return;
    setComparing(true);
    setCompareError(null);
    setCompareData(null);
    try {
      const resp = await api.compareJobs(selectedKeys);
      setCompareData(resp);
      firstCompareLoadDone = true;
    } catch (err) {
      setCompareError(err.message);
    } finally {
      setComparing(false);
    }
  }

  const normalizedSearch = search.trim();
  const facets = useMemo(
    () => extractCrossFacets(storedJobs, { nsFilter, modelFilter, endpointFilter, search: normalizedSearch }),
    [storedJobs, nsFilter, modelFilter, endpointFilter, normalizedSearch],
  );
  const filtered = useMemo(
    () => applyJobFilters(storedJobs, { nsFilter, modelFilter, endpointFilter, search: normalizedSearch }),
    [storedJobs, nsFilter, modelFilter, endpointFilter, normalizedSearch],
  );
  const anyFilterActive = nsFilter.size > 0 || modelFilter.size > 0 || endpointFilter.size > 0 || normalizedSearch.length > 0;

  const entries = compareData?.entries ?? [];
  // Display keys match composite values-map keys from the backend.
  const displayKeys = selectedKeys;

  // Build chart data: grouped bars per metric, one dataset per job
  const chartData = (() => {
    if (entries.length === 0) return null;
    const metrics = entries.map((e) => e.metric + (e.stat ? ' (' + e.stat + ')' : ''));
    const datasets = displayKeys.map((key, idx) => ({
      label: splitKey(key).jobId,
      data: entries.map((e) => e.values?.[key] ?? null),
      backgroundColor: JOB_COLORS[idx % JOB_COLORS.length] + 'cc',
      borderColor: JOB_COLORS[idx % JOB_COLORS.length],
      borderWidth: 1,
    }));
    return { labels: metrics, datasets };
  })();

  // InferenceX-style charts. Built only when the relevant metrics are
  // present for at least two runs — non-streaming runs lack
  // ``output_token_throughput_per_user`` and will silently drop out.
  // Y-axis throughput is normalized per GPU using meta.gpu_count from
  // the run's telemetry payload (``profile_export_aiperf.json``'s
  // ``telemetry_data.endpoints[..].gpus``). Runs without GPU telemetry
  // get dropped from the per-GPU charts and surfaced in a "no telemetry"
  // note rather than silently misrepresenting throughput.
  const meta = compareData?.meta ?? {};
  // Comparable-group key per selected run, shared by the metrics table's
  // "best" markers and the scatter frontiers below.
  const tableClusterByKey = Object.fromEntries(
    displayKeys.map((key) => [key, clusterKeyOf(splitKey(key).ns, meta[key]?.model)]),
  );
  const paretoAxes = {
    x: {
      metric: 'output_token_throughput_per_user',
      stat: 'avg',
      label: 'Per-user output throughput',
      unit: 'tok/s',
      largerBetter: true,
    },
    y: {
      metric: 'output_token_throughput',
      stat: 'avg',
      label: 'Output token throughput / GPU',
      unit: 'tok/s/GPU',
      largerBetter: true,
    },
  };
  const paretoPoints = buildScatterPoints(
    entries, paretoAxes.x, paretoAxes.y, displayKeys, splitKey, meta, true,
  );
  const paretoFrontiers = clusterFrontiers(
    paretoPoints, paretoAxes.x.largerBetter, paretoAxes.y.largerBetter,
  );
  const paretoChart = paretoPoints.length >= 1
    ? buildScatterChart(paretoPoints, paretoFrontiers, paretoAxes.x, paretoAxes.y)
    : null;

  const latThruAxes = {
    x: {
      metric: 'request_latency',
      stat: 'p99',
      label: 'Request latency p99',
      unit: 'ms',
      largerBetter: false,
    },
    y: {
      metric: 'request_throughput',
      stat: 'avg',
      label: 'Request throughput / GPU',
      unit: 'req/s/GPU',
      largerBetter: true,
    },
  };
  const latThruPoints = buildScatterPoints(
    entries, latThruAxes.x, latThruAxes.y, displayKeys, splitKey, meta, true,
  );
  const latThruFrontiers = clusterFrontiers(
    latThruPoints, latThruAxes.x.largerBetter, latThruAxes.y.largerBetter,
  );
  const latThruChart = latThruPoints.length >= 1
    ? buildScatterChart(latThruPoints, latThruFrontiers, latThruAxes.x, latThruAxes.y)
    : null;

  // GPU-family legend rows for the chart cards: one entry per distinct
  // family represented in the current Pareto point set.
  const paretoGpuLegend = (() => {
    const seen = new Map();
    for (const p of paretoPoints) {
      const fam = p.gpuFamily || 'unknown';
      if (!seen.has(fam)) seen.set(fam, { family: fam, color: p.color });
    }
    return Array.from(seen.values());
  })();
  const latThruGpuLegend = (() => {
    const seen = new Map();
    for (const p of latThruPoints) {
      const fam = p.gpuFamily || 'unknown';
      if (!seen.has(fam)) seen.set(fam, { family: fam, color: p.color });
    }
    return Array.from(seen.values());
  })();

  // Count of selected runs whose telemetry didn't expose any GPU at all —
  // used for the "N omitted (no GPU telemetry)" note.
  const runsMissingGpuTelemetry = displayKeys.filter((k) => {
    const c = Number(meta[k]?.gpu_count) || 0;
    return c <= 0;
  }).length;

  // Pareto Lab: cluster-grouped (ns × model) scatter with per-cluster
  // dashed frontiers, axis-preset switcher, and chip-bar visibility
  // toggles. Mirrors the operator/ui ``views/analysis.js`` Pareto.
  const labAxis = LAB_AXES.find((a) => a.key === labAxisKey) || LAB_AXES[0];
  const labPoints = buildLabPoints(entries, labAxis, displayKeys, splitKey, meta);
  const labClusterGroups = (() => {
    const g = {};
    for (const p of labPoints) {
      (g[p.clusterKey] ??= { ns: p.ns, model: p.model, points: [] }).points.push(p);
    }
    return g;
  })();
  const labAllClusterKeys = Object.keys(labClusterGroups);
  const labActiveSet = labActiveClusters ?? new Set(labAllClusterKeys);
  const labDatasets = buildLabDatasets(labPoints, labAxis, labActiveSet);
  const labOptions = buildLabOptions(labAxis);
  const toggleLabCluster = (ck) => {
    setLabActiveClusters((prev) => {
      const base = prev ?? new Set(labAllClusterKeys);
      const next = new Set(base);
      if (next.has(ck)) next.delete(ck); else next.add(ck);
      return next;
    });
  };

  const chartOptions = {
    plugins: {
      legend: {
        display: true,
        labels: { color: palette.overlay0, font: { size: CHART_TYPOGRAPHY.AXIS_LABEL } },
      },
    },
    scales: {
      x: {
        ticks: { color: palette.overlay0, font: { size: CHART_TYPOGRAPHY.AXIS_TICK }, maxRotation: 30 },
        grid: { color: palette.surface0 + '40' },
      },
      y: {
        ticks: { color: palette.overlay0, font: { size: CHART_TYPOGRAPHY.AXIS_LABEL } },
        grid: { color: palette.surface0 + '40' },
      },
    },
  };

  const FACET_COLLAPSE_AT = 6;
  const FACET_VISIBLE_WHEN_COLLAPSED = 5;
  const renderFacetRow = (label, dim, facetMap, filterSet, setFilterFn) => {
    const entries = Array.from(facetMap.entries()).sort((a, b) => b[1] - a[1]);
    if (entries.length === 0) return null;
    const expanded = facetExpanded[dim];
    const collapsed = entries.length > FACET_COLLAPSE_AT && !expanded;
    const visible = collapsed ? entries.slice(0, FACET_VISIBLE_WHEN_COLLAPSED) : entries;
    const overflow = entries.length - visible.length;
    return html`
      <div style="margin-bottom: var(--space-2)" data-testid=${'compare-facet-' + dim}>
        <div style="font-size: var(--font-size-xs); color: var(--overlay0); margin-bottom: var(--space-1)">${label}</div>
        <div style="display: flex; flex-wrap: wrap; gap: var(--space-1)">
          ${visible.map(([value, count]) => {
            const on = filterSet.has(value);
            const display = value === FILTER_NONE ? '(none)' : value;
            return html`
              <span
                key=${value}
                onclick=${() => toggleFacet(setFilterFn, value)}
                onkeydown=${(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    toggleFacet(setFilterFn, value);
                  }
                }}
                role="button"
                tabindex="0"
                aria-pressed=${on}
                title=${value === FILTER_NONE ? '(no value)' : value}
                style=${'display: inline-flex; align-items: center; gap: var(--space-1); padding: 0 0 2px; border: 0; border-bottom: 1px solid; border-radius: 0; font-size: var(--font-size-xs); cursor: pointer; max-width: 100%; min-width: 0;'
                  + (on
                    ? ' background: transparent; color: var(--text); border-color: var(--accent);'
                    : ' background: transparent; color: var(--subtext0); border-color: var(--surface1);')}
              >
                <span style="font-family: var(--font-mono); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0">${display}</span>
                <span style="opacity: 0.6; flex-shrink: 0">· ${count}</span>
              </span>
            `;
          })}
          ${(collapsed || (expanded && entries.length > FACET_COLLAPSE_AT)) && html`
            <span
              onclick=${() => toggleFacetExpanded(dim)}
              onkeydown=${(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  toggleFacetExpanded(dim);
                }
              }}
              role="button"
              tabindex="0"
              data-testid=${'compare-facet-toggle-' + dim}
              style="display: inline-flex; align-items: center; padding: 0 0 2px; border-radius: 0; font-size: var(--font-size-xs); cursor: pointer; background: transparent; color: var(--subtext0); border: 0; border-bottom: 1px solid var(--surface1)"
            >
              ${collapsed ? '+' + overflow + ' more' : 'Show less'}
            </span>
          `}
        </div>
      </div>
    `;
  };

  return html`
    <div class="compare-page" data-testid="page-compare">
      <div style="display: grid; grid-template-columns: 320px 1fr; gap: var(--space-4); align-items: start">

        <!-- Left: Job selector -->
        <div class="card">
          <div class="card-title" style="margin-bottom: var(--space-3)">Select Jobs</div>

          <input
            type="text"
            class="metric-selector-select"
            placeholder="Search jobs…"
            value=${search}
            oninput=${(e) => setSearch(e.target.value)}
            style="width: 100%; margin-bottom: var(--space-3)"
          />

          ${renderFacetRow('Namespace', 'ns', facets.ns, nsFilter, setNsFilter)}
          ${renderFacetRow('Model', 'model', facets.model, modelFilter, setModelFilter)}
          ${renderFacetRow('Endpoint', 'endpoint', facets.endpoint, endpointFilter, setEndpointFilter)}
          ${anyFilterActive && html`
            <div style="margin-bottom: var(--space-3)">
              <span
                onclick=${clearFilters}
                onkeydown=${(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    clearFilters();
                  }
                }}
                role="button"
                tabindex="0"
                data-testid="compare-clear-filters"
                style="font-size: var(--font-size-xs); color: var(--subtext0); cursor: pointer; text-decoration: underline; text-decoration-style: dotted"
              >Clear filters</span>
            </div>
          `}

          ${!jobsLoading && storedJobs.length >= 2 && selectedKeys.length === 0 && html`
            <div style="display: flex; gap: var(--space-2); flex-wrap: wrap; margin-bottom: var(--space-3)" data-testid="compare-quick-pick">
              <span class="text-dim" style="font-size: var(--font-size-xs); align-self: center">Quick pick:</span>
              <button type="button"
                onclick=${() => selectRecent(Math.min(3, storedJobs.length))}
                disabled=${storedJobs.length < 2}
                style="padding: 0 0 2px; border-radius: 0; border: 0; border-bottom: 1px solid var(--surface1); background: transparent; color: var(--subtext1); font-size: var(--font-size-xs); cursor: pointer"
                title="Pick the 3 most recent completed runs"
              >Last 3</button>
              ${storedJobs.length >= 5 && html`
                <button type="button"
                  onclick=${() => selectRecent(5)}
                  style="padding: 0 0 2px; border-radius: 0; border: 0; border-bottom: 1px solid var(--surface1); background: transparent; color: var(--subtext1); font-size: var(--font-size-xs); cursor: pointer"
                  title="Pick the 5 most recent completed runs"
                >Last 5</button>
              `}
            </div>
          `}

          ${jobsLoading && html`
            <${LoadingPanel} label="Loading jobs…" inline=${true} testid="compare-jobs-loading" />
          `}

          ${jobsError && html`
            <div style="color: var(--error); font-size: var(--font-size-sm)">${jobsError}</div>
          `}

          ${!jobsLoading && storedJobs.length === 0 && html`
            <div class="text-dim" style="padding: var(--space-3); text-align: center; font-size: var(--font-size-sm)">
              No completed jobs found.
            </div>
          `}

          ${!jobsLoading && filtered.length === 0 && storedJobs.length > 0 && html`
            <div class="text-dim" style="padding: var(--space-3); text-align: center; font-size: var(--font-size-sm)">
              No completed jobs match these filters.
            </div>
          `}

          <div style="max-height: 320px; overflow-y: auto" data-testid="compare-select">
            ${filtered.map((job) => {
              const jobId = job.job_id ?? '';
              const ns = job.namespace ?? '';
              const key = compositeKey(job);
              const isChecked = selectedKeys.includes(key);
              return html`
                <label
                  key=${key}
                  style=${'display: flex; align-items: flex-start; gap: var(--space-2); padding: var(--space-2) var(--space-1); cursor: pointer; border-radius: var(--radius-sm);' + (isChecked ? ' background: var(--surface0);' : '')}
                >
                  <input
                    type="checkbox"
                    checked=${isChecked}
                    onchange=${() => toggleJob(key)}
                    style="margin-top: 2px; accent-color: var(--accent)"
                  />
                  <div>
                    <div style="font-size: var(--font-size-sm); font-family: var(--font-mono)">${jobId}</div>
                    <div style="font-size: var(--font-size-xs); color: var(--overlay0)">${ns}</div>
                  </div>
                </label>
              `;
            })}
          </div>

          <div style="margin-top: var(--space-3); display: flex; gap: var(--space-2)">
            <button type="button"
              onclick=${handleCompare}
              disabled=${selectedKeys.length < 2 || comparing}
              style=${'flex: 1; padding: var(--space-2) var(--space-3); border-radius: var(--radius-sm); border: 1px solid; font-size: var(--font-size-sm); cursor: pointer;'
                + (selectedKeys.length >= 2 && !comparing
                  ? ' background: var(--accent); color: var(--base); border-color: var(--accent); font-weight: 600;'
                  : ' background: var(--surface0); color: var(--overlay0); border-color: var(--surface1); cursor: not-allowed;')}
            >
              ${comparing
                ? html`<span style="display: inline-flex; align-items: center; gap: var(--space-2)"><${Spinner} size=${12} thickness=${1.5} color="var(--overlay0)" />Comparing…</span>`
                : `Compare (${selectedKeys.length})`}
            </button>
            ${selectedKeys.length > 0 && html`
              <button type="button"
                onclick=${clearSelection}
                style="padding: var(--space-2) var(--space-3); border-radius: var(--radius-sm); border: 1px solid var(--surface1); background: transparent; color: var(--subtext0); font-size: var(--font-size-sm); cursor: pointer"
              >
                Clear
              </button>
            `}
          </div>
        </div>

        <!-- Right: Results -->
        <div>
          ${activeClusterLabel && html`
            <div
              data-testid="compare-cluster-pill"
              style="display: inline-flex; align-items: center; gap: var(--space-2); padding: 0 0 3px; margin-bottom: var(--space-3); border-radius: 0; background: transparent; border: 0; border-bottom: 1px solid var(--accent); color: var(--subtext1); font-size: var(--font-size-xs)"
            >
              <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: var(--accent)"></span>
              <span>Comparing all runs in cluster <span style="font-family: var(--font-mono); color: var(--text)">${activeClusterLabel}</span></span>
              <span
                onclick=${clearSelection}
                onkeydown=${(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    clearSelection();
                  }
                }}
                role="button"
                tabindex="0"
                aria-label="Clear deep-link cluster filter"
                title="Clear cluster filter and selection"
                style="cursor: pointer; opacity: 0.7; padding: 0 var(--space-1); border-left: 1px solid var(--surface1); margin-left: var(--space-1); line-height: 1; outline-offset: 2px"
              >✕</span>
            </div>
          `}

          ${unmatchedClusterLabel && html`
            <div
              class="card"
              data-testid="compare-cluster-unmatched"
              style="border-color: var(--peach); margin-bottom: var(--space-3); padding: var(--space-3); display: flex; align-items: flex-start; gap: var(--space-3)"
            >
              <div style="flex: 1; font-size: var(--font-size-sm)">
                <div style="color: var(--peach); font-weight: 600; margin-bottom: var(--space-1)">Cluster has no stored runs</div>
                <div class="text-dim">
                  Cluster <span style="font-family: var(--font-mono); color: var(--text)">${unmatchedClusterLabel}</span> has no stored runs yet — submit one to compare.
                </div>
              </div>
              <span
                onclick=${() => { setUnmatchedClusterLabel(null); if (query.value.cluster) setQuery({ cluster: '' }); }}
                role="button"
                tabindex="0"
                aria-label="Dismiss cluster banner"
                title="Dismiss"
                style="cursor: pointer; opacity: 0.7; padding: 0 var(--space-1); line-height: 1; color: var(--subtext0)"
              >✕</span>
            </div>
          `}

          ${selectedKeys.length > 0 && (() => {
            // Collapse threshold: keep 5 inline + "+N more" pill once we have
            // 7+ selections. At 6 we still render every chip — going from
            // 5 chips to "5 + (+1 more)" would be net-noisier than just
            // showing the sixth.
            const COLLAPSE_AT = 6;
            const VISIBLE_WHEN_COLLAPSED = 5;
            const collapsed = selectedKeys.length > COLLAPSE_AT && !chipsExpanded;
            const visibleKeys = collapsed
              ? selectedKeys.slice(0, VISIBLE_WHEN_COLLAPSED)
              : selectedKeys;
            const overflowCount = selectedKeys.length - visibleKeys.length;
            return html`
              <div style="display: flex; flex-wrap: wrap; gap: var(--space-2); margin-bottom: var(--space-4)" data-testid="compare-chips-overflow">
                ${visibleKeys.map((key) => html`
                  <span
                    key=${key}
                    style="display: inline-flex; align-items: center; gap: var(--space-1); padding: 0 0 2px; border-radius: 0; font-size: var(--font-size-xs); font-family: var(--font-mono); background: transparent; color: var(--subtext1); border: 0; border-bottom: 1px solid var(--surface1)"
                  >
                    ${splitKey(key).jobId}
                    ${(() => {
                      const m = meta[key];
                      if (!m || !m.gpu_count || m.gpu_count <= 0) return null;
                      const fam = gpuFamily(m.gpu_name) || m.gpu_name || 'GPU';
                      return html`
                        <span style="font-family: var(--font-mono); opacity: 0.85; padding-left: var(--space-1); border-left: 1px solid currentColor; line-height: 1">
                          ${m.gpu_count}× ${fam}
                        </span>
                      `;
                    })()}
                    <span
                      onclick=${() => toggleJob(key)}
                      onkeydown=${(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          toggleJob(key);
                        }
                      }}
                      title=${'Remove ' + splitKey(key).jobId + ' from comparison'}
                      aria-label=${'Remove ' + splitKey(key).jobId + ' from comparison'}
                      role="button"
                      tabindex="0"
                      style="cursor: pointer; opacity: 0.7; font-size: var(--font-size-sm); padding: 0 var(--space-1); margin-left: var(--space-1); border-left: 1px solid currentColor; line-height: 1; outline-offset: 2px"
                    >✕</span>
                  </span>
                `)}
                ${(collapsed || (chipsExpanded && selectedKeys.length > COLLAPSE_AT)) && html`
                  <span
                    onclick=${() => setChipsExpanded((v) => !v)}
                    onkeydown=${(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        setChipsExpanded((v) => !v);
                      }
                    }}
                    role="button"
                    tabindex="0"
                    data-testid="compare-chips-toggle"
                    aria-label=${collapsed ? 'Show ' + overflowCount + ' more selected runs' : 'Show fewer selected runs'}
                    title=${collapsed ? 'Show ' + overflowCount + ' more' : 'Show fewer'}
                    style="display: inline-flex; align-items: center; padding: 0 0 2px; border-radius: 0; font-size: var(--font-size-xs); font-family: var(--font-mono); cursor: pointer; background: transparent; color: var(--subtext0); border: 0; border-bottom: 1px solid var(--surface1); outline-offset: 2px"
                  >
                    ${collapsed ? '+' + overflowCount + ' more' : 'Show less'}
                  </span>
                `}
              </div>
            `;
          })()}

          ${compareError && html`
            <div class="card" style="border-color: var(--error); color: var(--error); margin-bottom: var(--space-4)">
              Compare failed: ${compareError}
            </div>
          `}

          ${!compareData && !comparing && selectedKeys.length === 0 && storedJobs.length === 0 && !jobsLoading && html`
            <div class="card empty-state">
              <p class="text-dim">No completed jobs yet. Run an AIPerfJob to populate the comparison list.</p>
            </div>
          `}

          ${!compareData && !comparing && selectedKeys.length === 0 && storedJobs.length > 0 && html`
            <div class="card empty-state">
              <p class="text-dim">Select 2 or more jobs from the list to compare them.</p>
            </div>
          `}

          ${!compareData && !comparing && selectedKeys.length === 1 && html`
            <div class="card empty-state">
              <p class="text-dim">Select at least one more job — comparison needs 2 or more runs.</p>
            </div>
          `}

          ${comparing && html`
            <div class="card">
              <${LoadingPanel} label="Running comparison…" testid="compare-running" />
            </div>
          `}

          ${compareData && !comparing && entries.length === 0 && html`
            <div class="card empty-state" data-testid="compare-no-entries">
              <p class="text-dim">No comparable metrics returned for the selected runs. They may be from different endpoints, or their results haven't been fully exported yet.</p>
            </div>
          `}

          ${compareData && !comparing && entries.length > 0 && html`
            <!-- Metrics table -->
            <div class="card" style="margin-bottom: var(--space-4)">
              <div class="card-title">Metric Comparison</div>
              ${new Set(Object.values(tableClusterByKey)).size > 1 && html`
                <div class="text-dim" style="font-size: var(--font-size-xs); margin-bottom: var(--space-2)" data-testid="compare-table-scope-note">
                  ${tableScopeNote(tableClusterByKey)}
                </div>
              `}
              <div style="overflow-x: auto">
                <table style="width: 100%; border-collapse: collapse; font-size: var(--font-size-sm)">
                  <thead>
                    <tr style="color: var(--subtext0); border-bottom: 1px solid var(--surface1)">
                      <th style="text-align: left; padding: var(--space-2) var(--space-3)">Metric</th>
                      ${displayKeys.map((key) => html`
                        <th
                          key=${key}
                          title=${key + ' · ' + tableClusterByKey[key]}
                          style="text-align: right; padding: var(--space-2) var(--space-3); color: var(--subtext1)"
                        >
                          <div>${splitKey(key).jobId}</div>
                          <div style="font-weight: normal; font-size: var(--font-size-xs); color: var(--overlay0)">
                            ${splitKey(key).ns || 'unknown'} · ${shortModel(meta[key]?.model)}
                          </div>
                        </th>
                      `)}
                    </tr>
                  </thead>
                  <tbody>
                    ${entries.map((entry) => {
                      const bestByCluster = bestValuePerCluster(
                        entry.metric, entry.values ?? {}, tableClusterByKey,
                      );
                      return html`
                        <tr key=${entry.metric + entry.stat} style="border-bottom: 1px solid var(--surface0)">
                          <td style="padding: var(--space-2) var(--space-3)" title=${METRIC_DESCRIPTIONS[entry.metric] || entry.metric}>
                            <div style="font-family: var(--font-mono); font-size: var(--font-size-xs)">${entry.metric}</div>
                            ${entry.stat && html`<div style="font-size: var(--font-size-xs); color: var(--overlay0)">${entry.stat}${entry.unit ? ' · ' + entry.unit : ''}</div>`}
                          </td>
                          ${displayKeys.map((key) => {
                            const val = entry.values?.[key] ?? null;
                            const isBest = val != null
                              && val === bestByCluster.get(tableClusterByKey[key]);
                            return html`
                              <td
                                key=${key}
                                style=${'text-align: right; padding: var(--space-2) var(--space-3); font-weight: 600;'
                                  + (isBest ? ' color: ' + palette.green : ' color: var(--text)')}
                              >
                                ${formatNum(val, entry.unit)}
                              </td>
                            `;
                          })}
                        </tr>
                      `;
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            <!-- Bar chart -->
            ${chartData && html`
              <div class="card">
                <div class="card-title">Visual Comparison</div>
                <${ChartWrapper} type="bar" data=${chartData} options=${chartOptions} height=${300} />
              </div>
            `}

            <!-- InferenceX-style Pareto: per-user vs per-GPU output throughput -->
            ${paretoChart && html`
              <div class="card" style="margin-top: var(--space-4)" data-testid="compare-pareto-throughput">
                <div class="card-title">Throughput Pareto · per-user × per-GPU</div>
                <${ChartWrapper} type="scatter" data=${{ datasets: paretoChart.datasets }} options=${paretoChart.options} height=${340} />
                ${frontierNote(paretoPoints, paretoFrontiers) && html`
                  <div class="text-dim" style="margin-top: var(--space-2); font-size: var(--font-size-xs)" data-testid="compare-pareto-throughput-note">
                    ${frontierNote(paretoPoints, paretoFrontiers)}
                  </div>
                `}
                ${paretoGpuLegend.length > 1 && html`
                  <div style="margin-top: var(--space-2); display: flex; flex-wrap: wrap; gap: var(--space-3); font-size: var(--font-size-xs)">
                    ${paretoGpuLegend.map((g) => html`
                      <span key=${g.family} style="display: inline-flex; align-items: center; gap: var(--space-1)">
                        <span style=${'display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: ' + g.color}></span>
                        <span style="color: var(--subtext0)">${g.family}</span>
                      </span>
                    `)}
                  </div>
                `}
                ${paretoPoints.length < displayKeys.length && html`
                  <div class="text-dim" style="margin-top: var(--space-2); font-size: var(--font-size-xs)">
                    ${displayKeys.length - paretoPoints.length} run(s) omitted — missing ${paretoAxes.x.metric} or GPU telemetry
                    ${runsMissingGpuTelemetry > 0
                      ? html` (${runsMissingGpuTelemetry} have no DCGM data — verify <code style="font-family: var(--font-mono)">dcgm-exporter</code> sidecar is enabled and the controller pod's GPU telemetry config exposes it)`
                      : ''}
                  </div>
                `}
              </div>
            `}

            <!-- InferenceX-style latency × throughput tradeoff -->
            ${latThruChart && html`
              <div class="card" style="margin-top: var(--space-4)" data-testid="compare-pareto-latency">
                <div class="card-title">Latency × Throughput · request p99 × req/s/GPU</div>
                <${ChartWrapper} type="scatter" data=${{ datasets: latThruChart.datasets }} options=${latThruChart.options} height=${340} />
                ${frontierNote(latThruPoints, latThruFrontiers) && html`
                  <div class="text-dim" style="margin-top: var(--space-2); font-size: var(--font-size-xs)" data-testid="compare-pareto-latency-note">
                    ${frontierNote(latThruPoints, latThruFrontiers)}
                  </div>
                `}
                ${latThruGpuLegend.length > 1 && html`
                  <div style="margin-top: var(--space-2); display: flex; flex-wrap: wrap; gap: var(--space-3); font-size: var(--font-size-xs)">
                    ${latThruGpuLegend.map((g) => html`
                      <span key=${g.family} style="display: inline-flex; align-items: center; gap: var(--space-1)">
                        <span style=${'display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: ' + g.color}></span>
                        <span style="color: var(--subtext0)">${g.family}</span>
                      </span>
                    `)}
                  </div>
                `}
                ${latThruPoints.length < displayKeys.length && html`
                  <div class="text-dim" style="margin-top: var(--space-2); font-size: var(--font-size-xs)">
                    ${displayKeys.length - latThruPoints.length} run(s) omitted — missing latency, throughput, or GPU telemetry
                    ${runsMissingGpuTelemetry > 0
                      ? html` (${runsMissingGpuTelemetry} have no DCGM data — verify <code style="font-family: var(--font-mono)">dcgm-exporter</code> sidecar is enabled and the controller pod's GPU telemetry config exposes it)`
                      : ''}
                  </div>
                `}
              </div>
            `}

            <!-- Pareto Lab — clustered (ns × model) frontiers with axis switcher.
                 Ported from operator/ui/views/analysis.js. -->
            ${labAllClusterKeys.length > 0 && html`
              <div class="card" style="margin-top: var(--space-4)" data-testid="compare-pareto-lab">
                <div style="display: flex; justify-content: space-between; align-items: center; gap: var(--space-3); flex-wrap: wrap; margin-bottom: var(--space-3)">
                  <div class="card-title" style="margin: 0">Pareto Lab · ${labAxis.label}</div>
                  <div style="display: inline-flex; gap: var(--space-4); flex-wrap: wrap" data-testid="compare-pareto-lab-axes">
                    ${LAB_AXES.map((a) => {
                      const isActive = a.key === labAxisKey;
                      return html`
                        <button type="button"
                          key=${a.key}
                          onclick=${() => setLabAxisKey(a.key)}
                          title=${a.label}
                          style=${'padding: var(--space-1) 0; border: 0; border-bottom: 2px solid; border-radius: 0; font-size: var(--font-size-xs); font-family: var(--font-mono); cursor: pointer;'
                            + (isActive
                              ? ' background: transparent; color: ' + palette.text + '; border-color: ' + palette.accent + ';'
                              : ' background: transparent; color: ' + palette.subtext0 + '; border-color: transparent;')}
                        >
                          ${a.key}
                        </button>
                      `;
                    })}
                  </div>
                </div>
                ${labAllClusterKeys.length > 1 && html`
                  <div style="display: flex; flex-wrap: wrap; gap: var(--space-4); margin-bottom: var(--space-3)" data-testid="compare-pareto-lab-clusters">
                    ${labAllClusterKeys.map((ck) => {
                      const grp = labClusterGroups[ck];
                      const isSingleton = grp.points.length < 2;
                      const on = labActiveSet.has(ck);
                      const dot = isSingleton ? MUTED_CLUSTER_COLOR : modelColor(grp.model);
                      return html`
                        <button type="button"
                          key=${ck}
                          onclick=${() => toggleLabCluster(ck)}
                          title=${ck + (isSingleton ? ' (singleton)' : '')}
                          style=${'display: inline-flex; align-items: center; gap: var(--space-1); padding: 0; border: 0; border-radius: 0; font-size: var(--font-size-xs); cursor: pointer;'
                            + (on
                              ? ' background: transparent; color: ' + palette.text + ';'
                              : ' background: transparent; color: ' + palette.overlay0 + '; opacity: 0.6;')}
                        >
                          <span style=${'display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: ' + dot + (on ? '' : '; opacity: 0.5')}></span>
                          <span style="font-family: var(--font-mono)">${grp.ns} · ${shortModel(grp.model)}</span>
                          <span style="opacity: 0.6">· ${grp.points.length}${isSingleton ? ' · singleton' : ''}</span>
                        </button>
                      `;
                    })}
                  </div>
                `}
                ${labDatasets.length === 0
                  ? html`
                    <div class="text-dim" style="padding: var(--space-4); text-align: center; font-size: var(--font-size-sm)">
                      No clusters visible. Toggle a series above to bring runs back into the chart.
                    </div>
                  `
                  : html`<${ChartWrapper} type="scatter" data=${{ datasets: labDatasets }} options=${labOptions} height=${360} />`}
                ${labPoints.length < displayKeys.length && html`
                  <div class="text-dim" style="margin-top: var(--space-2); font-size: var(--font-size-xs)">
                    ${displayKeys.length - labPoints.length} run(s) omitted — missing ${labAxis.x.metric} or ${labAxis.y.metric} for the selected stats.
                  </div>
                `}
              </div>
            `}
          `}
        </div>
      </div>
    </div>
  `;
}
