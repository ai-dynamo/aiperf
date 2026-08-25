// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const RUNNING_PHASES = new Set(['pending', 'running', 'aggregating']);
const CHILD_RUNNING_PHASES = new Set(['profiling', 'processing', 'running', 'aggregating']);
const CHILD_PENDING_PHASES = new Set(['pending', 'queued', 'initializing', '']);
const TERMINAL_PHASES = new Set(['succeeded', 'completed', 'archived', 'failed', 'partiallyfailed', 'cancelled']);
const SUCCEEDED_PHASES = new Set(['succeeded', 'completed', 'archived']);
const FAILED_PHASES = new Set(['failed', 'partiallyfailed']);
const CANCELLED_PHASES = new Set(['cancelled']);

const DEFAULT_HEADLINE_METRICS = [
  { key: 'request_throughput', stat: 'avg', label: 'Req throughput', unit: 'req/s' },
  { key: 'output_token_throughput', stat: 'avg', label: 'Output tok/s', unit: 'tok/s' },
  { key: 'total_token_throughput', stat: 'avg', label: 'Total tok/s', unit: 'tok/s' },
  { key: 'request_latency', stat: 'p50', label: 'Req latency p50', unit: 'ms' },
  { key: 'request_latency', stat: 'p99', label: 'Req latency p99', unit: 'ms' },
  { key: 'time_to_first_token', stat: 'p50', label: 'TTFT p50', unit: 'ms' },
  { key: 'time_to_first_token', stat: 'p99', label: 'TTFT p99', unit: 'ms' },
  { key: 'inter_token_latency', stat: 'avg', label: 'ITL avg', unit: 'ms' },
];

function pick(obj, keys) {
  for (const key of keys) {
    if (obj?.[key] != null) return obj[key];
  }
  return null;
}

function normalizePhase(phase) {
  return (phase ?? '').toString().toLowerCase();
}

export function sweepPhaseMode(phase) {
  const normalized = normalizePhase(phase);
  if (RUNNING_PHASES.has(normalized)) return 'live';
  if (TERMINAL_PHASES.has(normalized)) return 'terminal';
  return 'unknown';
}

export function childSweepState(phase) {
  const normalized = normalizePhase(phase);
  if (CHILD_PENDING_PHASES.has(normalized)) return 'pending';
  if (CHILD_RUNNING_PHASES.has(normalized)) return 'running';
  if (SUCCEEDED_PHASES.has(normalized)) return 'succeeded';
  if (FAILED_PHASES.has(normalized)) return 'failed';
  if (CANCELLED_PHASES.has(normalized)) return 'cancelled';
  return 'unknown';
}

export function isHigherBetterMetric(metricKey) {
  const normalized = (metricKey ?? '').toString().toLowerCase();
  return !(
    normalized.includes('latency') ||
    normalized.includes('ttft') ||
    normalized.includes('time_to_first_token') ||
    normalized.includes('inter_token_latency')
  );
}

/**
 * Mean, standard deviation and coefficient of variation across trials.
 *
 * `std` and `cv` are BOTH null for a single trial. One observation does not
 * estimate spread, and `std: 0` is not the same claim as `std: null`: zero
 * asserts "measured, and every trial landed on the same number", which is a
 * reproducibility result nobody produced. Null says "unmeasured", and callers
 * must render it as such rather than coercing it back to 0.
 *
 * The discipline was already applied on either side of this function and only
 * `std` was left out: `cv` has always been null for n<2 (the variations table
 * prints `cv ---`), and `sweep-live-trial-board.js` says "1 trial, spread
 * unknown" rather than `±0`. `variations-chart.js` draws no band where this is
 * null, for the same reason its tooltip already refuses to print `±0`.
 */
function meanStd(values) {
  const filtered = values.filter(v => typeof v === 'number' && Number.isFinite(v));
  if (filtered.length === 0) return null;
  const n = filtered.length;
  const mean = filtered.reduce((a, b) => a + b, 0) / n;
  if (n < 2) return { mean, std: null, cv: null, n };
  const variance = filtered.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  const std = Math.sqrt(variance);
  const cv = mean !== 0 ? std / Math.abs(mean) : null;
  return { mean, std, cv, n };
}

function metricValue(summary, metric) {
  const value = summary?.[metric.key]?.[metric.stat];
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function indexCells(cells) {
  const out = new Map();
  for (const cell of cells?.cells ?? []) {
    const idx = pick(cell, ['variation_index', 'variationIndex']);
    if (idx == null) continue;
    out.set(Number(idx), cell);
  }
  return out;
}

/**
 * Pick the variation manifest from the first populated source.
 *
 * Priority, highest first:
 *   1. `detail.status.aggregate.children` — the sweep-controller's own
 *      manifest, embedded on the CR. Bare-array or `{children: [...]}` envelope.
 *   2. `archivedChildren` — `GET /sweeps/{ns}/{name}/children`, fetched only
 *      when `detail.children` is empty (see the skip in sweep-detail.js).
 *   3. `detail.children` — `AIPerfJobInfo` rows from `list_all_jobs`. Despite
 *      being last, this is the source that actually renders a live sweep,
 *      because the router populates it for every sweep that has children, which
 *      suppresses source 2 and precedes source 1 being patched.
 *
 * Every source must therefore carry `variation_values`/`variationValues`; a
 * field present on only the higher-priority ones is dead on a live sweep.
 */
export function resolveSweepManifest({ detail, archivedChildren }) {
  const raw = detail?.status?.aggregate?.children;
  if (Array.isArray(raw) && raw.length > 0) return raw;
  if (raw && Array.isArray(raw.children) && raw.children.length > 0) {
    return raw.children;
  }
  if (Array.isArray(archivedChildren) && archivedChildren.length > 0) {
    return archivedChildren;
  }
  if (Array.isArray(detail?.children) && detail.children.length > 0) {
    return detail.children;
  }
  return [];
}

/**
 * The scalar swept parameters of a variation, as `[{path, leaf, value}]`.
 *
 * The single implementation of "describe a variation by its swept values".
 * Everything user-facing is an adapter over this list: `formatVariationValues`
 * joins it into a display string, `components/live-variations-helpers.js`
 * `parseVariationValues` maps it to chips, and
 * `components/sweep-winner-summary-helpers.js` `formatTrialValues` delegates to
 * the former. There were three parallel implementations and they disagreed --
 * on the truncation marker, on nested objects, on nulls, on lists, and on
 * unparseable strings -- so a guard added to one silently did not apply to the
 * others. Every guard now lives here exactly once.
 *
 * Adaptive planners label variations `search_iter_NNNN` -- that string is the
 * cell identity used for artifact paths, so it can't be renamed, but on its own
 * it says nothing about what was actually tried. The parameter values are what
 * a reader wants: `concurrency=17`, not `search_iter_0008`.
 *
 * Accepts the JSON-string form the CR carries (`status.runs[].values`, the
 * children manifest's `variation_values`, and the child CR's
 * `aiperf.nvidia.com/variation-values` annotation all use it) or an
 * already-parsed object. Dotted dimension paths are shortened to their leaf
 * (`phases.profiling.concurrency` -> `concurrency`) because the prefix is
 * identical across every variation and only costs width.
 *
 * Returns `[]` -- never a partial descriptor -- for each of:
 *
 *   - Unparseable input. The writer only ever emits `orjson.dumps(...)`
 *     (sweep_controller/k8s_executor.py:188-192), so a string that fails
 *     `JSON.parse` is a corrupted or hand-edited annotation, not a descriptor.
 *     A JSON object truncated mid-encode would otherwise surface its fragment
 *     (`{"phases.profiling.conc`) as the page headline, because callers LEAD
 *     with this label and demote the real `variation_label` beneath it.
 *   - The writer-side truncation marker (`__aiperf_truncated__`, emitted when
 *     the encoded values exceed the annotation/status byte budget). Rendering
 *     `limitBytes=256, reason=...` as if it were a swept parameter would be
 *     worse than showing the planner id.
 *   - Null values. A key whose value is absent describes nothing; emitting
 *     `concurrency=` or `concurrency=undefined` is the half-formed label this
 *     helper exists to avoid.
 *   - Nested objects and lists. Neither has a compact honest rendering on one
 *     line, and an elided blob (`tuning={...}`) looks authoritative while
 *     saying nothing.
 *
 * Same rule as the Python download-progress renderer,
 * `kubernetes/results.py:_cell_values`, which already returned `""` for all
 * four while claiming parity with a JS helper that did not.
 */
export function sweptValueEntries(values) {
  let parsed = values;
  if (typeof parsed === 'string') {
    try {
      parsed = JSON.parse(parsed);
    } catch {
      return [];
    }
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return [];
  if (parsed.__aiperf_truncated__) return [];
  const entries = [];
  for (const [path, value] of Object.entries(parsed)) {
    if (value == null || typeof value === 'object') continue;
    const text = String(path);
    // `filter(Boolean)` so a trailing-dot path (`phases.profiling.`) yields
    // `profiling` rather than falling through to the whole dotted path and
    // printing a dangling `phases.profiling.=17`.
    const leaf = text.split('.').filter(Boolean).pop() || text;
    entries.push({ path: text, leaf, value });
  }
  return entries;
}

/**
 * The swept values as one display string, e.g. `concurrency=17`.
 *
 * Returns null -- so callers can fall back to the raw `variation_label` with
 * `?? label` -- whenever `sweptValueEntries` finds nothing trustworthy to show.
 * That fallback is the whole contract: a non-null return is a promise that the
 * string describes the operating point.
 */
export function formatVariationValues(values) {
  const entries = sweptValueEntries(values);
  if (entries.length === 0) return null;
  return entries.map(entry => `${entry.leaf}=${entry.value}`).join(', ');
}

/**
 * Index `status.runs[]` by variation index so swept values can be attached.
 *
 * Keyed by `index` (the variation index) rather than `childName` so it still
 * matches when multiRun produces several trials per variation, which share one
 * set of values.
 */
function indexRunValues(statusRuns) {
  const byIndex = new Map();
  for (const run of statusRuns ?? []) {
    const idx = Number(run?.index);
    if (!Number.isFinite(idx) || byIndex.has(idx)) continue;
    const formatted = formatVariationValues(run?.values);
    if (formatted) byIndex.set(idx, { values: run.values, valuesLabel: formatted });
  }
  return byIndex;
}

/** Index the children manifest's own `variation_values` by variation index. */
function indexManifestValues(manifest) {
  const byIndex = new Map();
  for (const child of manifest ?? []) {
    const idx = Number(pick(child, ['variation_index', 'variationIndex']) ?? 0);
    if (!Number.isFinite(idx) || byIndex.has(idx)) continue;
    const raw = pick(child, ['variation_values', 'variationValues']);
    const formatted = formatVariationValues(raw);
    if (formatted) byIndex.set(idx, { values: raw, valuesLabel: formatted });
  }
  return byIndex;
}

/**
 * Resolve `{values, valuesLabel}` per variation index from every source that
 * can carry it, so all sweep surfaces describe a variation identically.
 *
 * The manifest is consulted first and `status.runs[]` only fills gaps. Both
 * ultimately derive from the same `aiperf.nvidia.com/variation-values`
 * annotation, so they cannot disagree — but the manifest copy is both fresher
 * and less lossy:
 *
 *   - Freshness: `status.runs[]` is appended only once a child reaches a
 *     terminal phase (operator/handlers/sweep/child_rollup.py:153-154), while
 *     the manifest carries values for children that are still running
 *     (routers/_sweeps_live.children_manifest_from_live_aiperfjobs).
 *   - Fidelity: the manifest/annotation budget is 2048 bytes
 *     (VARIATION_VALUES_MAX_ANNOTATION_BYTES) against 256 for `status.runs[]`
 *     (_STATUS_VARIATION_VALUES_MAX_BYTES), so the status copy degrades to a
 *     truncation marker sooner.
 *
 * Entries that format to nothing are omitted rather than stored as empty, so a
 * caller's `?? null` fallback to the raw label always fires.
 */
export function indexVariationValues({ manifest, statusRuns }) {
  const merged = indexManifestValues(manifest);
  for (const [idx, entry] of indexRunValues(statusRuns)) {
    if (!merged.has(idx)) merged.set(idx, entry);
  }
  return merged;
}

export function buildSweepVariations({
  manifest,
  childSummaries,
  cells,
  statusRuns,
  headlineMetrics = DEFAULT_HEADLINE_METRICS,
}) {
  if (!manifest || manifest.length === 0) return [];
  const cellsByIndex = indexCells(cells);
  const runValuesByIndex = indexVariationValues({ manifest, statusRuns });
  const groups = new Map();
  for (const c of manifest) {
    const idx = Number(pick(c, ['variation_index', 'variationIndex']) ?? 0);
    if (!groups.has(idx)) {
      groups.set(idx, {
        variation_index: idx,
        label: pick(c, ['variation_label', 'variationLabel']) ?? '',
        n_total: 0,
        summaries: [],
      });
    }
    const group = groups.get(idx);
    group.n_total += 1;
    const summary = childSummaries?.[c.name]?.summary ?? null;
    if (summary) group.summaries.push(summary);
  }
  for (const group of groups.values()) {
    if (group.summaries.length === 0) {
      const cell = cellsByIndex.get(group.variation_index);
      if (cell?.metrics) group.summaries.push(cell.metrics);
    }
  }
  return [...groups.values()]
    .sort((a, b) => a.variation_index - b.variation_index)
    .map(group => {
      const perMetric = {};
      for (const metric of headlineMetrics) {
        const values = group.summaries
          .map(summary => metricValue(summary, metric))
          .filter(value => value != null);
        perMetric[metric.key + '.' + metric.stat] = meanStd(values) ?? { mean: null, std: null, cv: null, n: 0 };
      }
      const swept = runValuesByIndex.get(group.variation_index) ?? null;
      return {
        variation_index: group.variation_index,
        label: group.label,
        // What the variation actually tried, e.g. "concurrency=17". Null when
        // the CR carried no values (older archives, or a sweep shape that does
        // not stamp them); callers fall back to `label`.
        valuesLabel: swept?.valuesLabel ?? null,
        values: swept?.values ?? null,
        n_trials: group.summaries.length,
        n_total: group.n_total,
        perMetric,
      };
    });
}

/**
 * Group manifest children into per-variation rows for the live trial board.
 *
 * Rows carry `valuesLabel` on the same contract as `buildSweepVariations`:
 * the swept parameters when they are known, null otherwise, never an empty or
 * partial string. Callers lead with it and demote `label` — an adaptive
 * planner's `search_iter_NNNN` is an identifier, not a description.
 */
export function buildTrialBoardRows({ manifest, childSummaries, statusRuns }) {
  if (!manifest || manifest.length === 0) return [];
  const valuesByIndex = indexVariationValues({ manifest, statusRuns });
  const groups = new Map();
  for (const child of manifest) {
    const variationIndex = Number(pick(child, ['variation_index', 'variationIndex']) ?? 0);
    if (!groups.has(variationIndex)) {
      const swept = valuesByIndex.get(variationIndex) ?? null;
      groups.set(variationIndex, {
        variation_index: variationIndex,
        label: pick(child, ['variation_label', 'variationLabel']) ?? '',
        valuesLabel: swept?.valuesLabel ?? null,
        values: swept?.values ?? null,
        trials: [],
      });
    }
    const summary = childSummaries?.[child.name] ?? {};
    const phase = pick(summary, ['phase']) ?? pick(child, ['phase']) ?? pick(child, ['status']);
    groups.get(variationIndex).trials.push({
      trial_index: Number(pick(child, ['trial_index', 'trialIndex']) ?? 0),
      name: child.name,
      namespace: child.namespace,
      phase,
      state: childSweepState(phase),
      progressPercent: pick(summary, ['progressPercent']) ?? pick(child, ['progress_percent', 'progressPercent']) ?? null,
      summary: pick(summary, ['summary']) ?? null,
    });
  }
  return [...groups.values()]
    .sort((a, b) => a.variation_index - b.variation_index)
    .map(group => ({
      ...group,
      trials: group.trials.sort((a, b) => a.trial_index - b.trial_index),
    }));
}

// SLA operators as declared on `sweep.slaFilters[*].op`.
const SLA_COMPARATORS = {
  lt: (v, t) => v < t,
  le: (v, t) => v <= t,
  gt: (v, t) => v > t,
  ge: (v, t) => v >= t,
};

/**
 * Resolve the objective the sweep itself declared, as a chart-style metric key.
 *
 * Returns null for generator sweep types (grid/zip/scenarios/sobol/...), which
 * have no objective — those keep the metric-ranked winner.
 */
export function objectiveMetricKey(objectives) {
  const first = (objectives ?? [])[0];
  if (!first?.metric) return null;
  return {
    metricKey: `${first.metric}.${first.stat ?? 'avg'}`,
    higherIsBetter: (first.direction ?? 'maximize') === 'maximize',
  };
}

/**
 * True when a variation satisfies every configured SLA filter.
 *
 * A variation missing the constrained metric entirely is treated as infeasible:
 * an unmeasured constraint is not a satisfied one, and calling it the winner
 * would advertise an operating point nobody demonstrated is servable.
 */
export function isVariationFeasible(variation, slaFilters) {
  for (const f of slaFilters ?? []) {
    const tag = f?.metricTag ?? f?.metric_tag;
    if (!tag) continue;
    const cmp = SLA_COMPARATORS[f?.op];
    const threshold = f?.threshold;
    if (!cmp || typeof threshold !== 'number') continue;
    const observed = variation?.perMetric?.[`${tag}.${f.stat ?? 'p95'}`]?.mean;
    if (typeof observed !== 'number' || !Number.isFinite(observed)) return false;
    if (!cmp(observed, threshold)) return false;
  }
  return true;
}

/**
 * Winner by the sweep's declared objective, restricted to SLA-feasible points.
 *
 * This is deliberately independent of the chart's metric selector. Ranking by
 * the selected chart metric made the headline "winner" change as soon as a user
 * clicked a different series, and ignoring `slaFilters` surfaced the highest
 * raw throughput even when it blew the latency constraint by 25x — on
 * gemma-bo4 that was concurrency 309 at TTFT p99 18.5s rather than the real
 * answer of 17.
 *
 * Returns null when the sweep declared no objective, so callers can fall back.
 */
export function pickObjectiveWinner({ variations, objectives, slaFilters }) {
  const objective = objectiveMetricKey(objectives);
  if (!objective) return null;
  const { metricKey, higherIsBetter } = objective;
  let winner = null;
  let feasibleCount = 0;
  for (const variation of variations ?? []) {
    if (!isVariationFeasible(variation, slaFilters)) continue;
    feasibleCount += 1;
    const metric = variation?.perMetric?.[metricKey];
    const mean = metric?.mean;
    if (typeof mean !== 'number' || !Number.isFinite(mean)) continue;
    if (!winner || (higherIsBetter ? mean > winner.mean : mean < winner.mean)) {
      winner = {
        variation_index: variation.variation_index,
        label: variation.label,
        valuesLabel: variation.valuesLabel ?? null,
        metricKey,
        mean,
        cv: metric.cv ?? null,
        n: metric.n ?? null,
        higherIsBetter,
        fromObjective: true,
      };
    }
  }
  if (winner) {
    winner.feasibleCount = feasibleCount;
    winner.constrained = (slaFilters ?? []).length > 0;
  }
  return winner;
}

export function pickSweepWinner({ variations, metricKey = 'output_token_throughput.avg' }) {
  const higherIsBetter = isHigherBetterMetric(metricKey);
  let winner = null;
  for (const variation of variations ?? []) {
    const metric = variation?.perMetric?.[metricKey];
    const mean = metric?.mean;
    if (typeof mean !== 'number' || !Number.isFinite(mean)) continue;
    if (!winner || (higherIsBetter ? mean > winner.mean : mean < winner.mean)) {
      winner = {
        variation_index: variation.variation_index,
        label: variation.label,
        valuesLabel: variation.valuesLabel ?? null,
        metricKey,
        mean,
        cv: metric.cv ?? null,
        n: metric.n ?? null,
        higherIsBetter,
      };
    }
  }
  return winner;
}

export function shouldShowSweepDiagnostics(phase) {
  return sweepPhaseMode(phase) === 'live';
}
