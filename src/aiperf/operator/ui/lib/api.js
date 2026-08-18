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

// Leading sentence of the app-level banner. Load-bearing: the browser e2e
// suites (tests/unit/operator/ui_e2e) assert on this exact phrase, and the
// results-server's archived-mode stubs are documented against it.
const UNREACHABLE_PREFIX = 'Operator API unreachable — live data is paused. Retrying…';

// Cap on the appended failure detail so a chatty 500 body cannot push the rest
// of the app off the screen.
const MAX_BANNER_DETAIL_CHARS = 160;

/**
 * One-line, length-capped rendering of whatever ``fn`` threw.
 * @param {unknown} err
 * @returns {string|null} null when the throw carried no usable text.
 */
function bannerDetail(err) {
  const raw = err == null ? '' : String(err.message ?? err);
  const text = raw.replace(/\s+/g, ' ').trim();
  if (!text) return null;
  return text.length > MAX_BANNER_DETAIL_CHARS
    ? text.slice(0, MAX_BANNER_DETAIL_CHARS - 1) + '…'
    : text;
}

/**
 * Banner text for a poller that has crossed the failure threshold.
 *
 * Without the appended detail, an RBAC 403, a 503 from a rolling operator
 * restart, and a genuinely unreachable pod all render identically, so the
 * reader has nothing to act on and no reason to suspect the diagnosis is
 * wrong. The status code is the cheapest possible correction to that.
 *
 * @param {unknown} err
 * @returns {string}
 */
function unreachableBannerText(err) {
  const detail = bannerDetail(err);
  return detail ? `${UNREACHABLE_PREFIX} (last error: ${detail})` : UNREACHABLE_PREFIX;
}

export const DASHBOARD_MUTATIONS_ENABLED = true;

const _TOKEN_KEY = 'aiperf.mutating.token';

export function getSessionToken() {
  try { return sessionStorage.getItem(_TOKEN_KEY) ?? null; }
  catch (_) { return null; }
}

export function setSessionToken(token) {
  try { sessionStorage.setItem(_TOKEN_KEY, token); } catch (_) {}
}

export function clearSessionToken() {
  try { sessionStorage.removeItem(_TOKEN_KEY); } catch (_) {}
}

class TokenRequiredError extends Error {
  constructor() {
    super('Bearer token required. Enter your AIPERF_OPERATOR_MUTATING_ROUTES_TOKEN to continue.');
    this.code = 'TOKEN_REQUIRED';
  }
}

export function isTokenRequiredError(err) {
  return err?.code === 'TOKEN_REQUIRED';
}

async function mutatingFetch(path, opts = {}) {
  const token = getSessionToken();
  if (!token) throw new TokenRequiredError();
  const resp = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}`, ...opts.headers },
    ...opts,
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => resp.statusText);
    if (resp.status === 401) clearSessionToken();
    throw httpError(resp, `${BASE}${path}`, text);
  }
  if (resp.status === 204) return null;
  return resp.json();
}

// Count of `poll()` instances currently reporting unhealthy. The banner
// stays up while ≥1 poller is failing; clears once every poller has had
// a clean tick. Module-scope so all poll() instances share the gate.
let _unhealthyPollers = 0;

/**
 * Build the Error thrown for a non-2xx response.
 *
 * The message shape (``API <status>: <body>``) is unchanged, but the status,
 * the URL that failed, and the raw body are attached as own properties.
 * Callers used to recover the status by running ``/\b404\b/`` over the
 * message, which also matches a 404 that merely appears inside a 500's body
 * — the exact "a failed fetch rendered as an empty result" trap. Read
 * ``err.status`` instead; ``httpStatusOf`` handles the mixed case.
 *
 * @param {Response} resp
 * @param {string} url - Fully-qualified URL that was requested.
 * @param {string} body - Response body text (or statusText if unreadable).
 * @returns {Error & {status: number, url: string, body: string}}
 */
function httpError(resp, url, body) {
  const err = new Error(`API ${resp.status}: ${body}`);
  err.status = resp.status;
  err.url = url;
  err.body = body;
  return err;
}

/**
 * Status code for an error raised by this module, or ``null`` when the
 * failure never reached an HTTP response (DNS, offline, CORS, abort).
 *
 * ``null`` is a meaningful answer: "the request did not complete" is a
 * different claim from "the server answered 404", and UI copy that conflates
 * them sends the reader after the wrong problem.
 *
 * @param {unknown} err
 * @returns {number|null}
 */
export function httpStatusOf(err) {
  if (err && typeof err.status === 'number') return err.status;
  return null;
}

/**
 * Low-level fetch wrapper. Throws on non-2xx.
 * @param {string} path - API path
 * @param {RequestInit} [opts] - Fetch options
 * @returns {Promise<any>}
 */
async function apiFetch(path, opts = {}) {
  const url = `${BASE}${path}`;
  const resp = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...opts.headers },
    ...opts,
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => resp.statusText);
    throw httpError(resp, url, text);
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

  /** Create an AIPerfJob from a parsed manifest object. Requires a bearer token
   *  stored in sessionStorage (enter once via the UI token prompt). */
  createJob(manifest) {
    return mutatingFetch('/jobs', { method: 'POST', body: JSON.stringify({ manifest }) });
  },

  /** Cancel a running job. Requires a bearer token (see createJob). */
  cancelJob(ns, name) {
    return mutatingFetch(`/jobs/${encodeURIComponent(ns)}/${encodeURIComponent(name)}/cancel`, { method: 'POST' });
  },

  /** Create an AIPerfSweep from a parsed manifest object. Requires a bearer token. */
  createSweep(manifest) {
    return mutatingFetch('/sweeps', { method: 'POST', body: JSON.stringify({ manifest }) });
  },

  /** Fetch the full raw spec of an AIPerfSweep for re-launch prefill. */
  getSweepConfig(ns, name, epoch = null) {
    const params = new URLSearchParams();
    if (epoch) params.set('epoch', String(epoch));
    const query = params.size ? `?${params}` : '';
    return apiFetch(`/sweeps/${encodeURIComponent(ns)}/${encodeURIComponent(name)}/config${query}`);
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
  getHistory(metric = 'request_throughput', stat = 'avg', { namespace = '', model = '', endpoint = '', limit = 10000 } = {}) {
    const params = new URLSearchParams({ metric, stat, limit: String(limit) });
    if (namespace) params.set('namespace', namespace);
    if (model) params.set('model', model);
    if (endpoint) params.set('endpoint', endpoint);
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

  /** All scatter metrics for the dashboard chart (single SQLite query) */
  getScatterData() {
    return apiFetch('/analytics/scatter');
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
  async fetchRunRequests(ns, jobId, epoch = null, filename = 'profile_export.jsonl') {
    const nsSeg = encodeURIComponent(ns);
    const idSeg = encodeURIComponent(jobId);
    const file = encodeURIComponent(filename);
    const url = epoch && epoch !== 'latest'
      ? `${BASE}/results/${nsSeg}/${idSeg}/runs/${encodeURIComponent(epoch)}/${file}`
      : `${BASE}/results/${nsSeg}/${idSeg}/${file}`;

    let resp;
    try {
      resp = await fetch(url, { headers: { Accept: 'application/x-ndjson, text/plain' } });
    } catch (err) {
      return { records: [], skipped: `fetch failed: ${err.message}` };
    }
    // Keep the status in the reason. "no per-request data" alone reads as a
    // statement about the run; the reader cannot tell it apart from a wrong
    // URL or a deleted epoch without knowing an HTTP 404 produced it.
    if (resp.status === 404) return { records: [], skipped: 'no per-request data (API 404)' };
    if (!resp.ok) return { records: [], skipped: `API ${resp.status} fetching ${file}` };

    const lenHeader = resp.headers.get('Content-Length');
    const size = lenHeader != null ? Number(lenHeader) : null;
    if (size != null && size > 200 * 1024 * 1024) {
      return { records: [], skipped: `file too large (${Math.round(size / 1024 / 1024)} MB)` };
    }

    const reader = resp.body?.getReader();
    if (!reader) return { records: [], skipped: 'response body unavailable' };
    const decoder = new TextDecoder();
    const MAX_DECODED_BYTES = 200 * 1024 * 1024;
    let decodedBytes = 0;
    let pending = '';
    const records = [];
    async function addLines(text, final = false) {
      pending += text;
      const lines = pending.split('\n');
      pending = final ? '' : lines.pop();
      for (const line of lines) {
      const s = line.trim();
      if (!s) continue;
      try {
        records.push(JSON.parse(s));
      } catch (_e) { /* skip malformed line */ }
      }
    }
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        decodedBytes += value.byteLength;
        if (decodedBytes > MAX_DECODED_BYTES) {
          await reader.cancel();
          return { records: [], skipped: 'file too large (over 200 MB)' };
        }
        await addLines(decoder.decode(value, { stream: true }));
      }
      await addLines(decoder.decode(), true);
    } catch (err) {
      return { records: [], skipped: `read failed: ${err.message}` };
    }
    return { records, skipped: null };
  },

  /** Fetch a single run's exported summary JSON for a given epoch.
   *
   *  Hits the run-specific profile-export alias so custom prefixes work.
   *  Throws ``Error`` with ``.status`` attached on non-2xx so callers can
   *  distinguish 404 (no summary on disk for that epoch) from transport
   *  failures.
   */
  async fetchRunSummary(ns, jobId, epoch) {
    const nsSeg = encodeURIComponent(ns);
    const idSeg = encodeURIComponent(jobId);
    const epSeg = encodeURIComponent(epoch);
    const url = `${BASE}/results/${nsSeg}/${idSeg}/runs/${epSeg}/profile_export`;
    const resp = await fetch(url);
    if (!resp.ok) {
      const err = new Error(`fetchRunSummary ${ns}/${jobId}/${epoch}: ${resp.status}`);
      err.status = resp.status;
      err.url = url;
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
    throw httpError(resp, url, text);
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
    throw httpError(resp, url, text);
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
  let pendingWakeup = false;

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
      setError(unreachableBannerText(err));
    }
  }

  function stopFreshness(reason = 'stopped') {
    if (!source || abortSignal.aborted) return;
    stoppedFreshness = true;
    markFreshnessStopped(source, reason, Date.now());
  }

  async function tick() {
    if (abortSignal.aborted) return;
    if (document.hidden) {
      pendingWakeup = true;
      return;
    }
    const ms = typeof intervalMs === 'function' ? intervalMs() : intervalMs;
    const attemptAt = Date.now();
    if (source && !stoppedFreshness) markFreshnessAttempt(source, ms, attemptAt);
    try {
      await fn({ stopFreshness, source });
      if (!abortSignal.aborted) markHealthy(Date.now());
    } catch (err) {
      if (!abortSignal.aborted) markFailure(err, Date.now());
    }
    if (!abortSignal.aborted) {
      handle = setTimeout(tick, ms);
    }
  }

  function onVisibilityChange() {
    if (document.visibilityState === 'visible' && pendingWakeup) {
      pendingWakeup = false;
      if (handle !== null) clearTimeout(handle);
      tick();
    }
  }

  document.addEventListener('visibilitychange', onVisibilityChange);

  abortSignal.addEventListener('abort', () => {
    if (handle !== null) clearTimeout(handle);
    document.removeEventListener('visibilitychange', onVisibilityChange);
    if (source && !stoppedFreshness) clearFreshnessSource(source);
    if (countedAsUnhealthy) {
      countedAsUnhealthy = false;
      _unhealthyPollers = Math.max(0, _unhealthyPollers - 1);
      if (_unhealthyPollers === 0) setError(null);
    }
  }, { once: true });

  tick();
}
