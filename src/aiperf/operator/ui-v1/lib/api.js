const BASE = '/api/v1';

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

  /** List the persisted run epochs for a job */
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

  /** Create an AIPerfJob from a parsed manifest object. POSTs the manifest
   *  wrapped under {manifest: ...} as the operator API expects. Returns the
   *  created object's {namespace, name} on success; throws ``Error("API <n>: <body>")``
   *  on non-2xx so callers can extract status + body.detail.
   */
  createJob(manifest) {
    return apiFetch('/jobs', { method: 'POST', body: JSON.stringify({ manifest }) });
  },

  /** Cancel a running job */
  cancelJob(ns, name) {
    return apiFetch(
      `/jobs/${encodeURIComponent(ns)}/${encodeURIComponent(name)}/cancel`,
      { method: 'POST' },
    );
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
  getJobConfig(ns, jobId) {
    return apiFetch(
      `/config/${encodeURIComponent(ns)}/${encodeURIComponent(jobId)}`,
    );
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
 * Polling helper. Calls fn() immediately, then every intervalMs.
 * Stops when the AbortSignal is aborted.
 *
 * @param {() => Promise<void>} fn - Async function to call on each tick
 * @param {number} intervalMs - Polling interval in milliseconds
 * @param {AbortSignal} abortSignal - Stop polling when this fires
 * @returns {void}
 */
export function poll(fn, intervalMs, abortSignal) {
  if (abortSignal.aborted) return;

  let handle = null;

  async function tick() {
    if (abortSignal.aborted) return;
    try {
      await fn();
    } catch (_err) {
      // Caller should handle errors inside fn; we don't crash the poll loop
    }
    if (!abortSignal.aborted) {
      handle = setTimeout(tick, intervalMs);
    }
  }

  abortSignal.addEventListener('abort', () => {
    if (handle !== null) clearTimeout(handle);
  }, { once: true });

  tick();
}
