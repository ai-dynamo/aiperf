// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  clearFreshnessSource,
  markFreshnessAttempt,
  markFreshnessFailure,
  markFreshnessStopped,
  markFreshnessSuccess,
  setError,
} from './state.js';

const BASE = '/api/v1';

// Number of consecutive `poll()` failures before we surface the
// app-level "Operator API unreachable" banner. Two ticks dampens
// transient blips (one bad request, one operator-pod restart) while
// still flagging a real outage within ~6-10s for typical poll cadences.
const POLL_FAIL_THRESHOLD = 2;

export const DASHBOARD_MUTATIONS_ENABLED = false;
export const DASHBOARD_MUTATIONS_DISABLED_MESSAGE = 'Dashboard mutating actions are disabled because the browser app has no safe bearer-token delivery path. Use aiperf kube or kubectl for create/cancel operations.';

function dashboardMutationDisabled() {
  throw new Error(DASHBOARD_MUTATIONS_DISABLED_MESSAGE);
}

// Count of `poll()` instances currently reporting unhealthy. The banner
// stays up while ≥1 poller is failing; clears once every poller has had
// a clean tick. Module-scope so all poll() instances share the gate.
let _unhealthyPollers = 0;

/**
 * Low-level fetch wrapper. Throws on non-2xx.
 * @param {string} path - API path
 * @param {RequestInit} [opts] - Fetch options
 * @returns {Promise<any>}
 */
async function apiFetch(path, opts = {}) {
  const resp = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...opts.headers },
    ...opts,
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => resp.statusText);
    throw new Error(`API ${resp.status}: ${text}`);
  }
  if (resp.status === 204) return null;
  return resp.json();
}

// Jobs
export const api = {
  /** List all AIPerfJob resources */
  listJobs() {
    return apiFetch('/jobs');
  },

  /** List all AIPerfSweep records (live + archived) */
  listSweeps() {
    return apiFetch('/sweeps');
  },

  /** Get a single job by namespace and name (optional epoch) */
  getJob(ns, name, epoch) {
    const q = epoch ? `?epoch=${encodeURIComponent(epoch)}` : '';
    return apiFetch(`/jobs/${encodeURIComponent(ns)}/${encodeURIComponent(name)}${q}`);
  },

  /**
   * List the persisted run epochs for a job. Each entry carries:
   * { epoch, isLatest, mtimeEpoch, fileCount, status, startedAt, endedAt }
   * where status is one of running/succeeded/failed/cancelled/unknown.
   */
  getJobEpochs(ns, name) {
    return apiFetch(`/jobs/${encodeURIComponent(ns)}/${encodeURIComponent(name)}/epochs`);
  },

  /** Get a sweep, optionally a specific epoch */
  getSweep(ns, name, epoch) {
    const q = epoch ? `?epoch=${encodeURIComponent(epoch)}` : '';
    return apiFetch(`/sweeps/${encodeURIComponent(ns)}/${encodeURIComponent(name)}${q}`);
  },

  /** List sweep epochs */
  getSweepEpochs(ns, name) {
    return apiFetch(`/sweeps/${encodeURIComponent(ns)}/${encodeURIComponent(name)}/epochs`);
  },

  /** Per-cell aggregates, optional epoch */
  getSweepCells(ns, name, epoch) {
    const q = epoch ? `?epoch=${encodeURIComponent(epoch)}` : '';
    return apiFetch(`/sweeps/${encodeURIComponent(ns)}/${encodeURIComponent(name)}/cells${q}`);
  },

  /** Per-epoch children manifest */
  getSweepChildren(ns, name, epoch) {
    const q = epoch ? `?epoch=${encodeURIComponent(epoch)}` : '';
    return apiFetch(`/sweeps/${encodeURIComponent(ns)}/${encodeURIComponent(name)}/children${q}`);
  },

  /** Get K8s events for an AIPerfSweep (CR + sweep-controller pod). */
  getSweepEvents(ns, name) {
    return apiFetch(
      `/sweeps/${encodeURIComponent(ns)}/${encodeURIComponent(name)}/events`,
    );
  },

  /** Fetch logs for an AIPerfSweep's sweep-controller pod.
   *
   *  Same shape as ``getJobLogs`` — non-follow returns text, follow returns the
   *  raw ``Response`` whose ``body.getReader()`` streams chunks. The pod
   *  defaults to the JobSet's single controller replica; pass ``opts.pod`` to
   *  override (e.g. when reading a previous restart's tail).
   */
  getSweepLogs(ns, name, opts) {
    return getSweepLogs(ns, name, opts);
  },

  /** Create an AIPerfJob from a parsed manifest object.
   *  Disabled in the browser app because protected mutating routes require a
   *  bearer token that should not be embedded in static JavaScript.
   */
  createJob(_manifest) {
    return dashboardMutationDisabled();
  },

  /** Cancel a running job */
  cancelJob(_ns, _name) {
    return dashboardMutationDisabled();
  },

  /** Get cluster-level info */
  getCluster() {
    return apiFetch('/cluster');
  },

  /** Leaderboard analytics */
  getLeaderboard(metric = 'request_throughput', stat = 'avg', limit = 20) {
    // Backend default is 20. Callers that filter client-side (e.g.
    // pages/leaderboard.js) should pass ``limit=1000`` so matching runs
    // ranked below 20 aren't silently absent from the filtered view.
    const params = new URLSearchParams({ metric, stat, limit: String(limit) });
    return apiFetch(`/analytics/leaderboard?${params}`);
  },

  /** History analytics */
  getHistory(metric = 'request_throughput', stat = 'avg') {
    const params = new URLSearchParams({ metric, stat });
    return apiFetch(`/analytics/history?${params}`);
  },

  /** Compare multiple jobs */
  compareJobs(jobIds) {
    const params = new URLSearchParams();
    for (const id of jobIds) params.append('jobs', id);
    return apiFetch(`/analytics/compare?${params}`);
  },

  /** Single job analytics summary */
  getJobSummary(ns, jobId) {
    return apiFetch(
      `/analytics/summary/${encodeURIComponent(ns)}/${encodeURIComponent(jobId)}`,
    );
  },

  /** List stored/completed jobs */
  listResults() {
    return apiFetch('/results');
  },

  /** Get original CR config for a job */
  getJobConfig(ns, jobId, epoch = null) {
    const path = `/config/${encodeURIComponent(ns)}/${encodeURIComponent(jobId)}`;
    if (epoch && epoch !== 'latest') {
      return apiFetch(`${path}?epoch=${encodeURIComponent(epoch)}`);
    }
    return apiFetch(path);
  },

  /** Get the full job index */
  getIndex() {
    return apiFetch('/index');
  },

  /** Get K8s events for a job (involvedObject=AIPerfJob + owned pods). */
  getJobEvents(ns, name) {
    return apiFetch(
      `/jobs/${encodeURIComponent(ns)}/${encodeURIComponent(name)}/events`,
    );
  },

  /** Stream a run's per-request export (``profile_export.jsonl``) as a list
   *  of JSON-decoded record objects.
   *
   *  Returns ``{records, skipped}`` so the caller can distinguish an
   *  intentionally-skipped run (no per-request data, file too big, transport
   *  failure) from a successful empty response.
   *
   *  Size cap: 200 MB (via Content-Length) to keep the browser responsive
   *  on huge runs.
   */
  async fetchRunRequests(ns, jobId, epoch = null) {
    const nsSeg = encodeURIComponent(ns);
    const idSeg = encodeURIComponent(jobId);
    const file = 'profile_export.jsonl';
    const url = epoch && epoch !== 'latest'
      ? `${BASE}/results/${nsSeg}/${idSeg}/runs/${encodeURIComponent(epoch)}/${file}`
      : `${BASE}/results/${nsSeg}/${idSeg}/${file}`;

    let resp;
    try {
      resp = await fetch(url, { headers: { Accept: 'application/x-ndjson, text/plain' } });
    } catch (err) {
      return { records: [], skipped: `fetch failed: ${err.message}` };
    }
    if (resp.status === 404) return { records: [], skipped: 'no per-request data' };
    if (!resp.ok) return { records: [], skipped: `API ${resp.status}` };

    const lenHeader = resp.headers.get('Content-Length');
    const size = lenHeader != null ? Number(lenHeader) : null;
    if (size != null && size > 200 * 1024 * 1024) {
      return { records: [], skipped: `file too large (${Math.round(size / 1024 / 1024)} MB)` };
    }

    const text = await resp.text();
    const records = [];
    for (const line of text.split('\n')) {
      const s = line.trim();
      if (!s) continue;
      try {
        records.push(JSON.parse(s));
      } catch (_e) { /* skip malformed line */ }
    }
    return { records, skipped: null };
  },

  /** Fetch a single run's exported summary JSON for a given epoch.
   *
   *  Hits ``/results/<ns>/<jobId>/runs/<epoch>/profile_export_aiperf.json``.
   *  Throws ``Error`` with ``.status`` attached on non-2xx so callers can
   *  distinguish 404 (no summary on disk for that epoch) from transport
   *  failures.
   */
  async fetchRunSummary(ns, jobId, epoch) {
    const nsSeg = encodeURIComponent(ns);
    const idSeg = encodeURIComponent(jobId);
    const epSeg = encodeURIComponent(epoch);
    const url = `${BASE}/results/${nsSeg}/${idSeg}/runs/${epSeg}/profile_export_aiperf.json`;
    const resp = await fetch(url);
    if (!resp.ok) {
      const err = new Error(`fetchRunSummary ${ns}/${jobId}/${epoch}: ${resp.status}`);
      err.status = resp.status;
      throw err;
    }
    return resp.json();
  },

  /** Fetch pod logs for a job. See ``getJobLogs`` below. */
  getJobLogs(ns, name, opts) {
    return getJobLogs(ns, name, opts);
  },

  /** Build a URL for the full job-results bundle as a single zip.
   *
   *  Hits ``/results/<ns>/<jobId>/runs/<epoch>.zip``. The backend rejects an
   *  unpinned "latest" bundle because latest can move while the archive is being
   *  streamed, so callers must pass a concrete epoch.
   */
  resultBundleUrl(ns, jobId, epoch = null) {
    if (!epoch || epoch === 'latest') {
      throw new Error('resultBundleUrl requires a concrete run epoch');
    }
    const nsSeg = encodeURIComponent(ns);
    const idSeg = encodeURIComponent(jobId);
    return `${BASE}/results/${nsSeg}/${idSeg}/runs/${encodeURIComponent(epoch)}.zip`;
  },

  sweepArtifactListUrl(ns, sweepName, epoch) {
    const nsSeg = encodeURIComponent(ns);
    const sweepSeg = encodeURIComponent(sweepName);
    const epSeg = encodeURIComponent(epoch);
    return `${BASE}/sweeps/${nsSeg}/${sweepSeg}/epochs/${epSeg}/artifacts`;
  },

  sweepArtifactBundleUrl(ns, sweepName, epoch) {
    const nsSeg = encodeURIComponent(ns);
    const sweepSeg = encodeURIComponent(sweepName);
    const epSeg = encodeURIComponent(epoch);
    return `${BASE}/sweeps/${nsSeg}/${sweepSeg}/epochs/${epSeg}/artifacts.zip`;
  },

  sweepArtifactFileUrl(ns, sweepName, epoch, filename) {
    const nsSeg = encodeURIComponent(ns);
    const sweepSeg = encodeURIComponent(sweepName);
    const epSeg = encodeURIComponent(epoch);
    const fileSeg = filename.split('/').map(encodeURIComponent).join('/');
    return `${BASE}/sweeps/${nsSeg}/${sweepSeg}/epochs/${epSeg}/artifacts/${fileSeg}`;
  },

  sweepProfileExportUrl(ns, sweepName, epoch, format = 'json') {
    const nsSeg = encodeURIComponent(ns);
    const sweepSeg = encodeURIComponent(sweepName);
    const epSeg = encodeURIComponent(epoch);
    const formatSeg = encodeURIComponent(format);
    return `${BASE}/sweeps/${nsSeg}/${sweepSeg}/epochs/${epSeg}/artifacts/profile_export?format=${formatSeg}`;
  },
};

/**
 * Pod logs fetcher with optional follow streaming. Response in non-follow
 * mode is a string of raw text; in follow mode it's the raw ``Response``
 * so the caller can pump ``response.body.getReader()`` for live updates.
 *
 * Defined outside ``api`` so it can return either a Response or text
 * without forcing an extra branch in every callsite.
 *
 * @param {string} ns
 * @param {string} name
 * @param {{pod: string, container?: string, follow?: boolean, tailLines?: number, signal?: AbortSignal}} opts
 * @returns {Promise<string|Response>}
 */
async function getJobLogs(ns, name, opts) {
  const { pod, container, follow, tailLines, signal } = opts ?? {};
  const params = new URLSearchParams();
  if (pod) params.set('pod', pod);
  if (container) params.set('container', container);
  if (follow) params.set('follow', '1');
  if (tailLines != null) params.set('tail_lines', String(tailLines));
  const url = `${BASE}/jobs/${encodeURIComponent(ns)}/${encodeURIComponent(name)}/logs?${params}`;
  const resp = await fetch(url, { headers: { Accept: 'text/plain' }, signal });
  if (!resp.ok) {
    const text = await resp.text().catch(() => resp.statusText);
    throw new Error(`API ${resp.status}: ${text}`);
  }
  if (follow) return resp;
  return resp.text();
}

/**
 * Sweep-controller pod log fetcher. Same shape as ``getJobLogs`` but rooted at
 * ``/api/v1/sweeps/<ns>/<name>/logs``. ``pod`` is optional; the operator
 * defaults to the JobSet's running controller replica when omitted.
 *
 * @param {string} ns
 * @param {string} name
 * @param {{pod?: string, container?: string, follow?: boolean, tailLines?: number, signal?: AbortSignal}} opts
 * @returns {Promise<string|Response>}
 */
async function getSweepLogs(ns, name, opts) {
  const { pod, container, follow, tailLines, signal } = opts ?? {};
  const params = new URLSearchParams();
  if (pod) params.set('pod', pod);
  if (container) params.set('container', container);
  if (follow) params.set('follow', '1');
  if (tailLines != null) params.set('tail_lines', String(tailLines));
  const url = `${BASE}/sweeps/${encodeURIComponent(ns)}/${encodeURIComponent(name)}/logs?${params}`;
  const resp = await fetch(url, { headers: { Accept: 'text/plain' }, signal });
  if (!resp.ok) {
    const text = await resp.text().catch(() => resp.statusText);
    throw new Error(`API ${resp.status}: ${text}`);
  }
  if (follow) return resp;
  return resp.text();
}

/**
 * Polling helper. Calls fn() immediately, then every intervalMs.
 * Stops when the AbortSignal is aborted.
 *
 * Tracks consecutive failures per poll instance: after
 * ``POLL_FAIL_THRESHOLD`` failures in a row, raises the app-level
 * "Operator API unreachable" banner via ``setError``. The banner
 * clears once every active poll instance has had at least one
 * successful tick (so a single recovering endpoint doesn't hide a
 * separate one that's still 5xx-ing).
 *
 * Per-page error UX (richer messages, first-load blocks) still works
 * because pages can wrap fn() with their own try/catch + state.
 *
 * @param {(context: {stopFreshness: (reason?: string) => void, source: string|null}) => Promise<void>} fn
 *   Async function to call on each tick
 * @param {number} intervalMs - Polling interval in milliseconds
 * @param {AbortSignal} abortSignal - Stop polling when this fires
 * @param {{source?: string}} [options] - Optional named freshness source
 * @returns {void}
 */
export function poll(fn, intervalMs, abortSignal, options = {}) {
  if (abortSignal.aborted) return;

  const source = options.source ?? null;
  let handle = null;
  let consecutiveFailures = 0;
  let countedAsUnhealthy = false;
  let stoppedFreshness = false;

  function markHealthy(at) {
    consecutiveFailures = 0;
    if (source && !stoppedFreshness && !abortSignal.aborted) markFreshnessSuccess(source, at);
    if (countedAsUnhealthy) {
      countedAsUnhealthy = false;
      _unhealthyPollers = Math.max(0, _unhealthyPollers - 1);
      if (_unhealthyPollers === 0) setError(null);
    }
  }

  function markFailure(err, at) {
    consecutiveFailures += 1;
    const retrying = consecutiveFailures >= POLL_FAIL_THRESHOLD;
    if (source && !stoppedFreshness) {
      markFreshnessFailure(source, err?.message ?? err, at, retrying);
    }
    if (retrying && !countedAsUnhealthy) {
      countedAsUnhealthy = true;
      _unhealthyPollers += 1;
      setError('Operator API unreachable — live data is paused. Retrying…');
    }
  }

  function stopFreshness(reason = 'stopped') {
    if (!source || abortSignal.aborted) return;
    stoppedFreshness = true;
    markFreshnessStopped(source, reason, Date.now());
  }

  async function tick() {
    if (abortSignal.aborted) return;
    const attemptAt = Date.now();
    if (source && !stoppedFreshness) markFreshnessAttempt(source, intervalMs, attemptAt);
    try {
      await fn({ stopFreshness, source });
      if (!abortSignal.aborted) markHealthy(Date.now());
    } catch (err) {
      if (!abortSignal.aborted) markFailure(err, Date.now());
    }
    if (!abortSignal.aborted) {
      handle = setTimeout(tick, intervalMs);
    }
  }

  abortSignal.addEventListener('abort', () => {
    if (handle !== null) clearTimeout(handle);
    if (source && !stoppedFreshness) clearFreshnessSource(source);
    if (countedAsUnhealthy) {
      countedAsUnhealthy = false;
      _unhealthyPollers = Math.max(0, _unhealthyPollers - 1);
      if (_unhealthyPollers === 0) setError(null);
    }
  }, { once: true });

  tick();
}
