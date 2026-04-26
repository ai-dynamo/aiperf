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
