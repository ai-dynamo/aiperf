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
