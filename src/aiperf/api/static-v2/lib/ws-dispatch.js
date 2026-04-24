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
  phases, records, workers, serverMetrics,
  realtimeMetrics, telemetryMetrics,
  recordTimeseriesSample,
  markRunStarted,
  log,
} from './state.js';

/** Merge a per-phase stats update into the phases map. */
function applyPhase(name, stats, patch = {}) {
  const prev = phases.value[name] ?? {};
  const merged = { ...prev, ...stats, ...patch, name };
  // Derived flags to drive badge/bar state.
  merged.active = Boolean(stats?.start_ns) && !stats?.requests_end_ns;
  merged.complete = Boolean(stats?.requests_end_ns);
  merged.grace = Boolean(stats?.timeout_triggered || stats?.grace_period_timeout_triggered)
    && !merged.complete;
  phases.value = { ...phases.value, [name]: merged };
}

/** Apply a subset of processing_stats fields to the records signal. */
function applyRecords(patch) {
  records.value = { ...records.value, ...patch };
}

/** Merge one or many worker updates into the workers signal. */
function applyWorkers(updates) {
  if (!updates) return;
  const next = { ...workers.value };
  for (const [wid, info] of Object.entries(updates)) {
    next[wid] = { ...(next[wid] ?? {}), ...info };
  }
  workers.value = next;
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
    case 'credit_phase_complete': {
      // Real aiperf server nests the phase name inside `stats.phase` (the
      // CreditPhaseStats model); our test harness sometimes passes it at
      // the top level. Check both.
      const stats = msg.stats ?? {};
      const name = msg.phase ?? msg.phase_name ?? msg.credit_phase
        ?? stats.phase ?? stats.phase_name ?? 'unknown';
      applyPhase(name, msg.stats ?? msg);
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
      const s = msg.stats ?? msg;
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
      const s = msg.stats ?? msg;
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

    case 'worker_health': {
      const wid = msg.worker_id ?? msg.id;
      if (!wid) return;
      const status = msg.status ?? 'idle';
      const prev = workers.value[wid];
      applyWorkers({ [wid]: {
        id: wid,
        status,
        startupState: msg.startup_state ?? null,
        inFlight: msg.in_flight ?? 0,
        completed: msg.completed ?? 0,
        failed: msg.failed ?? 0,
        cpu: msg.cpu_percent ?? null,
        memory: msg.memory_bytes ?? null,
      }});
      if (prev && prev.status !== status) {
        if (status === 'error') {
          log({ severity: 'error', category: 'worker',
                message: `Worker ${wid.split('-').slice(-1)[0]} → error` });
        } else if (status === 'high_load' && prev.status !== 'high_load') {
          log({ severity: 'warn', category: 'worker',
                message: `Worker ${wid.split('-').slice(-1)[0]} under high load` });
        }
      }
      return;
    }

    case 'worker_status_summary':
      applyWorkers(msg.workers ?? {});
      return;

    case 'realtime_server_metrics':
      if (Array.isArray(msg.endpoint_summaries)) {
        serverMetrics.value = msg.endpoint_summaries;
      }
      return;

    case 'realtime_metrics':
      if (Array.isArray(msg.metrics)) {
        realtimeMetrics.value = msg.metrics;
        recordTimeseriesSample(msg.metrics);
        // Visible-to-e2e diagnostic: log the count + first-tile primary once
        // per batch so a failing run can tell if metrics even arrive.
        if (msg.metrics.length > 0) {
          const first = msg.metrics[0];
          log({ severity: 'info', category: 'metrics',
                message: `realtime: ${msg.metrics.length} metrics (${first?.tag ?? '?'} ${first?.current ?? first?.avg ?? '?'})` });
        }
      }
      return;

    case 'realtime_telemetry_metrics':
      if (Array.isArray(msg.metrics)) telemetryMetrics.value = msg.metrics;
      return;

    default:
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
