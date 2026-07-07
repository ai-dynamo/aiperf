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

export function buildSweepVariations({
  manifest,
  childSummaries,
  cells,
  headlineMetrics = DEFAULT_HEADLINE_METRICS,
}) {
  if (!manifest || manifest.length === 0) return [];
  const cellsByIndex = indexCells(cells);
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
      return {
        variation_index: group.variation_index,
        label: group.label,
        n_trials: group.summaries.length,
        n_total: group.n_total,
        perMetric,
      };
    });
}

export function buildTrialBoardRows({ manifest, childSummaries }) {
  if (!manifest || manifest.length === 0) return [];
  const groups = new Map();
  for (const child of manifest) {
    const variationIndex = Number(pick(child, ['variation_index', 'variationIndex']) ?? 0);
    if (!groups.has(variationIndex)) {
      groups.set(variationIndex, {
        variation_index: variationIndex,
        label: pick(child, ['variation_label', 'variationLabel']) ?? '',
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
