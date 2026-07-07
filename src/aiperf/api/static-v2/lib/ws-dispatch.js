// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Dispatch incoming WebSocket messages to the right state signal.
 *
 * Contract: every message has a `type` field. Unknown types are logged
 * once (debug) and ignored.
 *
 * Unlike v1, phases are keyed by their *actual* phase name rather than
 * collapsed into warmup/profiling buckets.
 */

import {
  phases, records, workerGroups, serverMetrics,
  realtimeMetrics, telemetryMetrics,
  recordTimeseriesSample,
  markRunStarted,
  log,
} from './state.js';

/** Merge a per-phase stats update into the phases map. */
function applyPhase(name, stats, patch = {}) {
  const prev = phases.value[name] ?? {};
  const prevTerminal = Boolean(prev.complete || prev.failed || prev.requests_end_ns);
  if (prevTerminal) return;

  const merged = { ...prev, ...stats, ...patch, name };
  // Derived flags to drive badge/bar state.
  merged.failed = Boolean(patch.failed || stats?.failed);
  merged.complete = Boolean(stats?.requests_end_ns) && !merged.failed;
  merged.active = Boolean(stats?.start_ns) && !stats?.requests_end_ns && !merged.failed;
  merged.grace = Boolean(stats?.timeout_triggered || stats?.grace_period_timeout_triggered)
    && !merged.complete && !merged.failed;
  phases.value = { ...phases.value, [name]: merged };
}

/** Apply a subset of processing_stats fields to the records signal. */
function applyRecords(patch) {
  records.value = { ...records.value, ...patch };
}

/** Normalize the wire-format ``endpoint_summaries`` (a dict keyed by endpoint
 *  name, each value carrying ``metrics: dict[name, {type, unit, series:[{stats}]}]``)
 *  into the array shape the v2 ``<ServerMetrics>`` component consumes:
 *  ``[{endpoint, metrics: [{name, value, unit}, ...]}, ...]``. The component
 *  shows one representative value per metric — first series, ``stats.avg``
 *  (or the only stats field that's set, e.g. ``rate`` for counters). */
function isRecord(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function finiteNumber(value) {
  return typeof value === 'number' && isFinite(value) ? value : null;
}

function firstFiniteStat(stats) {
  return finiteNumber(stats.avg) ?? finiteNumber(stats.rate) ?? finiteNumber(stats.value);
}

function copyFiniteStats(stats) {
  const out = {};
  for (const key of ['avg', 'min', 'max', 'p99', 'p90', 'p50']) {
    const value = finiteNumber(stats?.[key]);
    if (value != null) out[key] = value;
  }
  return out;
}

export function normalizeEndpointSummaries(summaries) {
  if (!isRecord(summaries)) return [];
  return Object.entries(summaries).flatMap(([endpoint, body]) => {
    if (!isRecord(body) || !isRecord(body.metrics)) return [];
    const metrics = Object.entries(body.metrics).flatMap(([name, m]) => {
      if (!isRecord(m) || !Array.isArray(m.series)) return [];
      const series = m.series.find((sample) => firstFiniteStat(sample?.stats) != null);
      if (!series) return [];
      return [{
        name,
        value: firstFiniteStat(series.stats),
        unit: m.unit ?? null,
        ...copyFiniteStats(series.stats),
      }];
    });
    return metrics.length ? [{ endpoint, metrics }] : [];
  });
}

const unknownMessageTypesLogged = new Set();
let phaseWsUpdateSeen = false;
let serverMetricsWsUpdateSeen = false;

export function hasPhaseWsUpdate() {
  return phaseWsUpdateSeen;
}

export function hasServerMetricsWsUpdate() {
  return serverMetricsWsUpdateSeen;
}

function shortGroupId(id) {
  if (!id) return '';
  const parts = id.split('-');
  return parts.length <= 2 ? id : parts.slice(-2).join('-');
}

/** Derive in-flight tasks from a wire WorkerTaskStats dict.
 *  Server-side ``task_stats.in_progress`` is a non-serialized @property
 *  (total - completed - failed), so the wire payload never carries it —
 *  recompute it client-side from the serialized counters. */
function inFlightTasks(ts) {
  return Math.max(0, (ts?.total ?? 0) - (ts?.completed ?? 0) - (ts?.failed ?? 0));
}

/** Replace one group entry from a WorkerGroupStatsMessage. */
function applyGroupStats(msg) {
  const groupId = msg.group_id ?? msg.service_id;
  if (!groupId) return;
  const children = {};
  for (const [wid, status] of Object.entries(msg.worker_statuses ?? {})) {
    const ts = (msg.worker_task_stats ?? {})[wid] ?? {};
    const wh = (msg.worker_health ?? {})[wid] ?? null;
    children[wid] = {
      id: wid,
      status,
      startupState: (msg.worker_startup_states ?? {})[wid] ?? null,
      inFlight: inFlightTasks(ts),
      completed: ts.completed ?? 0,
      failed: ts.failed ?? 0,
      total: ts.total ?? 0,
      cpu: wh?.cpu_usage ?? null,
      memory: wh?.memory_usage ?? null,
    };
  }
  const group = {
    id: groupId,
    status: msg.status ?? 'idle',
    startupState: msg.startup_state ?? null,
    declaredWorkers: msg.declared_workers ?? 0,
    readyWorkers: msg.ready_workers ?? 0,
    inFlight: inFlightTasks(msg.task_stats),
    completed: msg.task_stats?.completed ?? 0,
    failed: msg.task_stats?.failed ?? 0,
    total: msg.task_stats?.total ?? 0,
    cpu: msg.health?.cpu_usage ?? null,
    memory: msg.health?.memory_usage ?? null,
    workers: children,
  };
  const prev = workerGroups.value[groupId];
  workerGroups.value = { ...workerGroups.value, [groupId]: group };
  // Surface group-level health transitions in the log so users notice
  // error / high_load rollups even when only the group row renders.
  if (prev && prev.status !== group.status) {
    if (group.status === 'error') {
      log({ severity: 'error', category: 'worker',
            message: `Group ${shortGroupId(groupId)} → error` });
    } else if (group.status === 'high_load') {
      log({ severity: 'warn', category: 'worker',
            message: `Group ${shortGroupId(groupId)} under high load` });
    }
  }
}

export function handleWsMessage(msg) {
  if (!msg || typeof msg !== 'object') return;
  const type = msg.type ?? msg.message_type;

  switch (type) {
    case 'subscribed':
      log(`Subscribed: ${(msg.message_types || []).join(', ')}`);
      return;

    case 'credit_phase_start':
    case 'credit_phase_progress':
    case 'credit_phase_sending_complete':
    case 'credit_phase_complete':
    case 'credit_phase_failed': {
      // Real aiperf server nests the phase name inside `stats.phase` (the
      // CreditPhaseStats model); our test harness sometimes passes it at
      // the top level. Check both.
      const stats = msg.stats ?? {};
      const name = msg.phase ?? msg.phase_name ?? msg.credit_phase
        ?? stats.phase ?? stats.phase_name ?? 'unknown';
      applyPhase(name, msg.stats ?? msg, type === 'credit_phase_failed' ? { failed: true } : {});
      if (type === 'credit_phase_start') {
        markRunStarted();
        log({ severity: 'info', category: 'phase', message: `Phase started: ${name}` });
      }
      if (type === 'credit_phase_sending_complete') {
        log({ severity: 'info', category: 'phase', message: `Sending complete: ${name}` });
      }
      if (type === 'credit_phase_complete') {
        log({ severity: 'info', category: 'phase', message: `Phase complete: ${name}` });
      }
      if (type === 'credit_phase_failed') {
        log({ severity: 'error', category: 'phase', message: `Phase failed: ${name}` });
      }
      phaseWsUpdateSeen = true;
      // Any grace-period transition is worth surfacing.
      const s = msg.stats ?? msg;
      if ((s?.timeout_triggered || s?.grace_period_timeout_triggered)
          && !(s?.requests_end_ns)) {
        log({ severity: 'warn', category: 'phase',
              message: `${name}: grace period triggered` });
      }
      return;
    }

    case 'processing_stats': {
      // RecordsProcessingStatsMessage carries the counters under
      // `processing_stats` on the wire; `stats` is kept as a compat fallback.
      const s = msg.processing_stats ?? msg.stats ?? msg;
      applyRecords({
        successRecords: Number(s.success_records) || 0,
        errorRecords: Number(s.error_records) || 0,
        finalRequestsCompleted: s.final_requests_completed != null
          ? Number(s.final_requests_completed) : records.value.finalRequestsCompleted,
        startNs: s.start_ns != null ? Number(s.start_ns) : records.value.startNs,
        active: true,
      });
      return;
    }

    case 'all_records_received': {
      // AllRecordsReceivedMessage carries the counters under
      // `final_processing_stats` on the wire; `stats` is a compat fallback.
      const s = msg.final_processing_stats ?? msg.stats ?? msg;
      applyRecords({
        successRecords: s.success_records != null
          ? Number(s.success_records) : records.value.successRecords,
        errorRecords: s.error_records != null
          ? Number(s.error_records) : records.value.errorRecords,
        finalRequestsCompleted: s.final_requests_completed != null
          ? Number(s.final_requests_completed) : records.value.finalRequestsCompleted,
        endNs: s.records_end_ns != null ? Number(s.records_end_ns) : null,
        active: false,
        complete: true,
      });
      log({ severity: 'info', category: 'records', message: 'All records received' });
      return;
    }

    case 'worker_group_stats':
      applyGroupStats(msg);
      return;

    case 'realtime_server_metrics':
      serverMetricsWsUpdateSeen = true;
      serverMetrics.value = normalizeEndpointSummaries(msg.endpoint_summaries);
      return;

    case 'realtime_metrics':
      if (Array.isArray(msg.metrics)) {
        realtimeMetrics.value = msg.metrics;
        recordTimeseriesSample(msg.metrics);
        // Visible-to-e2e diagnostic: log the count + first-tile primary once
        // per batch so a failing run can tell if metrics even arrive.
        if (msg.metrics.length > 0) {
          const first = msg.metrics[0];
          const firstValue = finiteNumber(first?.current) ?? finiteNumber(first?.avg);
          log({ severity: 'info', category: 'metrics',
                message: `realtime: ${msg.metrics.length} metrics (${first?.tag ?? '?'} ${firstValue ?? '?'})` });
        }
      }
      return;

    case 'realtime_telemetry_metrics':
      if (Array.isArray(msg.metrics)) telemetryMetrics.value = msg.metrics;
      return;

    default:
      if (type && !unknownMessageTypesLogged.has(type)) {
        unknownMessageTypesLogged.add(type);
        log({ severity: 'info', category: 'ws', message: `Unknown WS message type: ${type}` });
      }
      return;
  }
}

/** Bootstrap from a /api/progress response (handles mid-run page refresh). */
export function bootstrapProgress(data) {
  const phaseDict = data?.phases ?? {};
  for (const [name, stats] of Object.entries(phaseDict)) {
    if (!stats?.start_ns) continue;
    applyPhase(name, stats, {
      completed: stats.final_requests_completed ?? stats.requests_completed ?? 0,
    });
  }
}
