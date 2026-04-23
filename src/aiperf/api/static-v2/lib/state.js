// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Global dashboard state held in Preact signals.
 *
 * Phases are keyed by their *actual* backend phase name — fixing the v1
 * behavior where every non-warmup phase was bucketed under "profiling",
 * which silently lost per-phase progress for multi-phase runs.
 */

import { signal } from '@preact/signals';
import { fmtClock } from './format.js';
import { pushSample } from './timeseries.js';

/** 'disconnected' | 'connecting' | 'connected' */
export const connection = signal('disconnected');

/** Parsed /api/config payload (or null). */
export const config = signal(null);

/** Map of phaseName → PhaseStats (whatever the backend sends, plus some derived fields). */
export const phases = signal({});

/** Records manager state (processing_stats / all_records_received). */
export const records = signal({
  successRecords: 0,
  errorRecords: 0,
  finalRequestsCompleted: null,
  startNs: null,
  endNs: null,
  active: false,
  complete: false,
});

/** Map of workerId → WorkerInfo. */
export const workers = signal({});

/** Server metrics endpoint_summaries (array of { endpoint, metrics: [...] }). */
export const serverMetrics = signal([]);

/** Realtime benchmark-side metrics (list of MetricResult from the message bus). */
export const realtimeMetrics = signal([]);

/** Realtime GPU telemetry metrics (list of MetricResult, grouped per-GPU in the renderer). */
export const telemetryMetrics = signal([]);

/** Rolling per-metric time series keyed by metric tag. Feeds sparklines and
 *  the live throughput-vs-latency chart. Updated in ``ws-dispatch.js`` every
 *  time a ``realtime_metrics`` message arrives.
 *
 *  Shape: { [metricTag]: [{t: number, values: {current, avg, p99, ...}}, ...] }
 */
export const timeseries = signal({});

/** Append a batch of MetricResults into the rolling window (called from dispatch). */
export function recordTimeseriesSample(metrics) {
  if (!Array.isArray(metrics) || metrics.length === 0) return;
  const t = Date.now();
  const next = { ...timeseries.value };
  for (const m of metrics) {
    if (!m?.tag) continue;
    const prev = next[m.tag] ?? [];
    const values = {};
    for (const stat of ['current', 'avg', 'p50', 'p90', 'p99', 'max', 'min']) {
      const v = m[stat];
      if (typeof v === 'number' && isFinite(v)) values[stat] = v;
    }
    // Only record the sample if at least one stat is meaningful.
    if (Object.keys(values).length > 0) {
      next[m.tag] = pushSample(prev, { t, values });
    }
  }
  timeseries.value = next;
}

/** Bounded log: newest last, capped at MAX_LOG_ENTRIES.
 *  Each entry is { ts: 'HH:MM:SS', severity: 'info'|'warn'|'error', message, category? } */
export const logs = signal([]);

export const MAX_LOG_ENTRIES = 60;

/** Append an entry to the log. Accepts either a plain string (info) or an
 *  object with { severity, message, category }. */
export function log(entry) {
  const normalized = typeof entry === 'string'
    ? { severity: 'info', message: entry }
    : { severity: entry.severity ?? 'info', message: entry.message, category: entry.category };
  const next = [...logs.value, { ts: fmtClock(), ...normalized }];
  if (next.length > MAX_LOG_ENTRIES) next.splice(0, next.length - MAX_LOG_ENTRIES);
  logs.value = next;
}

/** Timestamp (ms epoch) when the current run started. Set when the first
 *  credit_phase_start lands so ETA / elapsed are relative to the run,
 *  not the page load. */
export const runStartedAt = signal(null);

export function markRunStarted() {
  if (runStartedAt.value == null) runStartedAt.value = Date.now();
}

/** Clear all live state (called on WebSocket disconnect). */
export function resetLiveState() {
  phases.value = {};
  records.value = {
    successRecords: 0,
    errorRecords: 0,
    finalRequestsCompleted: null,
    startNs: null,
    endNs: null,
    active: false,
    complete: false,
  };
  workers.value = {};
  serverMetrics.value = [];
  realtimeMetrics.value = [];
  telemetryMetrics.value = [];
  timeseries.value = {};
  runStartedAt.value = null;
}
