// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { signal, computed } from '@preact/signals';

// Raw jobs list from /api/v1/jobs
export const jobs = signal([]);

// Raw sweeps list from /api/v1/sweeps
export const sweeps = signal([]);

// Currently selected job (for detail page)
export const selectedJob = signal(null);

// Cluster info from /api/v1/cluster
export const clusterInfo = signal(null);

// Global error message (displayed in top bar)
export const globalError = signal(null);

// Live data freshness by source name. Sources are stable strings such as
// "jobs", "sweeps", "cluster", "job-detail", and "sweep-detail".
export const freshness = signal({});

function nowMs() {
  return Date.now();
}

function existingFreshness(source) {
  return freshness.value[source] ?? {
    source,
    status: 'idle',
    intervalMs: null,
    lastAttemptAt: null,
    lastSuccessAt: null,
    lastError: null,
    reason: null,
  };
}

function setFreshnessSource(source, patch) {
  if (!source) return null;
  const next = { ...existingFreshness(source), ...patch, source };
  freshness.value = { ...freshness.value, [source]: next };
  return next;
}

/**
 * Record that a refresh is in flight.
 *
 * A source that has already failed and never succeeded stays ``failed``
 * while the retry runs, and keeps its ``lastError``. Reverting to
 * ``loading`` would claim we are still waiting on a first answer when we
 * already have one and it was a failure -- observed live as a "Jobs
 * Loading" pill sitting on screen through three consecutive 503s -- and
 * clearing the error dropped the only detail the pill tooltip could show.
 */
export function markFreshnessAttempt(source, intervalMs = null, at = nowMs()) {
  if (!source) return null;
  const prior = existingFreshness(source);
  const coldFailure = prior.lastSuccessAt == null && prior.status === 'failed';
  return setFreshnessSource(source, {
    status: prior.lastSuccessAt == null ? (coldFailure ? 'failed' : 'loading') : prior.status,
    intervalMs,
    lastAttemptAt: at,
    lastError: coldFailure ? prior.lastError : null,
    reason: null,
  });
}

export function markFreshnessSuccess(source, at = nowMs()) {
  if (!source) return null;
  return setFreshnessSource(source, {
    status: 'fresh',
    lastAttemptAt: at,
    lastSuccessAt: at,
    lastError: null,
    reason: null,
  });
}

/**
 * Record a failed refresh.
 *
 * Three outcomes, deliberately distinct:
 *   - ``failed``   -- never succeeded. There is no data behind this source,
 *                     so no "showing last-known data" claim is available.
 *   - ``stale``    -- succeeded before, one recent failure.
 *   - ``retrying`` -- succeeded before, past the poll failure threshold.
 */
export function markFreshnessFailure(source, error, at = nowMs(), retrying = false) {
  if (!source) return null;
  const prior = existingFreshness(source);
  return setFreshnessSource(source, {
    status: prior.lastSuccessAt == null ? 'failed' : retrying ? 'retrying' : 'stale',
    lastAttemptAt: at,
    lastError: String(error ?? 'refresh failed'),
    reason: null,
  });
}

export function markFreshnessStopped(source, reason = 'stopped', at = nowMs()) {
  if (!source) return null;
  return setFreshnessSource(source, {
    status: 'stopped',
    lastAttemptAt: at,
    lastError: null,
    reason,
  });
}

export function clearFreshnessSource(source) {
  if (!source || freshness.value[source] == null) return;
  const next = { ...freshness.value };
  delete next[source];
  freshness.value = next;
}

export const freshnessSources = computed(() =>
  Object.values(freshness.value).sort((a, b) => a.source.localeCompare(b.source)),
);

// Loading states
export const loading = signal({
  jobs: false,
  cluster: false,
  leaderboard: false,
  history: false,
});

// Derived: jobs indexed by "namespace/name" key.
// Note: /api/v1/jobs returns flat AIPerfJobInfo records, not raw CR objects.
export const jobsById = computed(() => {
  const map = {};
  for (const job of jobs.value) {
    const key = `${job.namespace ?? 'default'}/${job.name ?? ''}`;
    map[key] = job;
  }
  return map;
});

// Derived: running jobs only
export const runningJobs = computed(() =>
  jobs.value.filter((j) => {
    const phase = (j.phase ?? '').toLowerCase();
    return phase === 'running' || phase === 'initializing';
  }),
);

// Derived: completed jobs only
export const completedJobs = computed(() =>
  jobs.value.filter((j) => {
    const phase = (j.phase ?? '').toLowerCase();
    return phase === 'completed' || phase === 'succeeded';
  }),
);

// Derived: non-success terminal jobs (failed, error, cancelled).
// ``cancelled`` is a separate terminal phase from ``failed``
// — keep both rolled into this signal so dashboard "Failed" tabs still
// surface user-cancelled runs.
export const failedJobs = computed(() =>
  jobs.value.filter((j) => {
    const phase = (j.phase ?? '').toLowerCase();
    return phase === 'failed' || phase === 'error' || phase === 'cancelled';
  }),
);

/**
 * Update the loading state for a specific key.
 * @param {string} key
 * @param {boolean} value
 */
export function setLoading(key, value) {
  loading.value = { ...loading.value, [key]: value };
}

/**
 * Set a global error. Pass null to clear.
 * @param {string|null} message
 */
export function setError(message) {
  globalError.value = message;
}

/**
 * Frontend dedupe safety net for the live + archive union endpoints
 * (``/api/v1/jobs`` and ``/api/v1/sweeps``).
 *
 * The backend in :mod:`aiperf.operator.job_union` and
 * :mod:`aiperf.operator.sweep_union` already merges by ``(namespace, name)``
 * and tags overlap entries with ``source="both"``. This helper catches
 * any future regression on that path without papering over data: when
 * dupes exist, prefer the live-side entry (CR has authoritative phase /
 * worker / progress) and copy any non-null fields from the archive
 * sibling so we don't drop summary-derived columns.
 *
 * @template T
 * @param {T[] | null | undefined} rows - Raw list from the API.
 * @returns {T[]} Deduped list, stably ordered by first appearance.
 */
export function dedupeByNsName(rows) {
  if (!Array.isArray(rows)) return [];
  const seen = new Map();
  const order = [];
  for (const row of rows) {
    if (row == null || typeof row !== 'object') continue;
    const ns = row.namespace ?? 'default';
    const name = row.name ?? '';
    if (!name) continue;
    const key = ns + '/' + name;
    const prior = seen.get(key);
    if (!prior) {
      seen.set(key, row);
      order.push(key);
      continue;
    }
    // Pick the live-side (or "both") entry as the base; backfill any
    // null fields from the archived sibling so columns like throughput /
    // latency that the CR is silent about don't disappear.
    const liveLike = (s) => s === 'live' || s === 'both';
    const base = liveLike(prior.source) ? prior
                : liveLike(row.source) ? row
                : prior;
    const other = base === prior ? row : prior;
    const merged = { ...base };
    for (const k of Object.keys(other)) {
      if (merged[k] == null && other[k] != null) merged[k] = other[k];
    }
    if (base === row) merged.source = 'both';
    seen.set(key, merged);
  }
  return order.map((k) => seen.get(k));
}
