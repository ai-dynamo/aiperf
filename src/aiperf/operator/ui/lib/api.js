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

/**
 * Pod log fetch. Non-follow returns text; follow=true returns the Response
 * so the caller can reader.read() the chunked body itself. 404 throws like
 * the other helpers. Accepts an AbortSignal for stream cleanup.
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

// Jobs
export const api = {
  /** List all AIPerfJob resources */
  listJobs() {
    return apiFetch('/jobs');
  },

  /** Get a single job by namespace and name */
  getJob(ns, name) {
    return apiFetch(`/jobs/${encodeURIComponent(ns)}/${encodeURIComponent(name)}`);
  },

  /** Cancel a running job */
  cancelJob(ns, name) {
    return apiFetch(
      `/jobs/${encodeURIComponent(ns)}/${encodeURIComponent(name)}/cancel`,
      { method: 'POST' },
    );
  },

  /** Create a new AIPerfJob CR from a manifest dict. Returns {namespace, name, uid}. */
  createJob(manifest) {
    return apiFetch('/jobs', {
      method: 'POST',
      body: JSON.stringify({ manifest }),
    });
  },

  /** Get cluster-level info */
  getCluster() {
    return apiFetch('/cluster');
  },

  /** Leaderboard analytics */
  getLeaderboard(metric = 'request_throughput', stat = 'avg') {
    const params = new URLSearchParams({ metric, stat });
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

  /** List individual result files for a job on the PVC.
   *
   * When ``epoch`` is omitted/null/``'latest'``, hits the stable
   * ``/results/<ns>/<jobId>`` endpoint (follows ``latest.txt``). When a
   * specific epoch string is passed, hits the pinned historical endpoint
   * ``/results/<ns>/<jobId>/runs/<epoch>``.
   */
  listJobFiles(ns, jobId, epoch = null) {
    const nsSeg = encodeURIComponent(ns);
    const idSeg = encodeURIComponent(jobId);
    if (epoch && epoch !== 'latest') {
      return apiFetch(`/results/${nsSeg}/${idSeg}/runs/${encodeURIComponent(epoch)}`);
    }
    return apiFetch(`/results/${nsSeg}/${idSeg}`);
  },

  /** Build an absolute-ish URL for a single result file (for anchor href
   *  downloads). Passing an epoch routes through the historical endpoint.
   */
  resultFileUrl(ns, jobId, filename, epoch = null) {
    const nsSeg = encodeURIComponent(ns);
    const idSeg = encodeURIComponent(jobId);
    const fileSeg = encodeURIComponent(filename);
    if (epoch && epoch !== 'latest') {
      return `${BASE}/results/${nsSeg}/${idSeg}/runs/${encodeURIComponent(epoch)}/${fileSeg}`;
    }
    return `${BASE}/results/${nsSeg}/${idSeg}/${fileSeg}`;
  },

  /** Build an absolute-ish URL for the full job bundle as a single zip.
   *  Passing an epoch targets the historical bundle.
   */
  resultBundleUrl(ns, jobId, epoch = null) {
    const nsSeg = encodeURIComponent(ns);
    const idSeg = encodeURIComponent(jobId);
    if (epoch && epoch !== 'latest') {
      return `${BASE}/results/${nsSeg}/${idSeg}/runs/${encodeURIComponent(epoch)}.zip`;
    }
    return `${BASE}/results/${nsSeg}/${idSeg}.zip`;
  },

  /** List every historical run dir for ``<ns>/<jobId>``, newest first.
   *
   * Returns ``{runs, latestEpoch}``. A 404 is treated as "no runs yet"
   * and yields an empty list — this lets the UI render the detail page
   * for a brand-new CR with no archive tree without surfacing an error.
   */
  async listRuns(ns, jobId) {
    const resp = await fetch(
      `${BASE}/results/${encodeURIComponent(ns)}/${encodeURIComponent(jobId)}/runs`,
    );
    if (resp.status === 404) return { runs: [], latestEpoch: null };
    if (!resp.ok) {
      const text = await resp.text().catch(() => resp.statusText);
      throw new Error(`API ${resp.status}: ${text}`);
    }
    const body = await resp.json();
    return { runs: body.runs ?? [], latestEpoch: body.latest_epoch ?? null };
  },

  /** Fetch per-request records from ``profile_export.jsonl`` for a run.
   *
   *  ``profile_export_aiperf.json`` is the aggregate summary (percentile
   *  stats with ``unit`` keys); the per-request stream lives in
   *  ``profile_export.jsonl`` — one JSON object per line. Each record has
   *  ``metrics.request_latency = {value: <ns>, unit: "ns"|"s"}`` plus
   *  ``metadata`` timing fields.
   *
   *  Returns ``{records, skipped}`` where ``skipped`` is a reason string when
   *  the file is too large or missing — callers render a graceful fallback
   *  without showing an error. Passing ``epoch`` pins to a historical run;
   *  otherwise reads the latest-run file via ``latest.txt``.
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

  /** Fetch the top-level summary metrics of a historical run's
   *  ``profile_export_aiperf.json``. Used by the run-diff view to lay two
   *  runs' ``request_throughput`` / ``request_latency`` / etc. side-by-side.
   *
   *  Throws with a parseable ``status`` field on the error when the HTTP
   *  request fails so callers can distinguish 404 (run exists, no export)
   *  from transport errors. Caller is expected to handle the missing-export
   *  case gracefully (render "n/a" in the affected column).
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

  /** Get original CR config for a job */
  getJobConfig(ns, jobId) {
    return apiFetch(
      `/config/${encodeURIComponent(ns)}/${encodeURIComponent(jobId)}`,
    );
  },

  /** Get K8s events for a job (involvedObject=AIPerfJob + owned pods). */
  getJobEvents(ns, name) {
    return apiFetch(
      `/jobs/${encodeURIComponent(ns)}/${encodeURIComponent(name)}/events`,
    );
  },

  /** Get the full job index */
  getIndex() {
    return apiFetch('/index');
  },

  /** Fetch pod logs for a job. See `getJobLogs` above. */
  getJobLogs(ns, name, opts) {
    return getJobLogs(ns, name, opts);
  },
};

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
