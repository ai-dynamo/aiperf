import { signal, computed } from '@preact/signals';

// Raw jobs list from /api/v1/jobs
export const jobs = signal([]);

// Currently selected job (for detail page)
export const selectedJob = signal(null);

// Cluster info from /api/v1/cluster
export const clusterInfo = signal(null);

// Global error message (displayed in top bar)
export const globalError = signal(null);

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

// Derived: failed jobs only
export const failedJobs = computed(() =>
  jobs.value.filter((j) => {
    const phase = (j.phase ?? '').toLowerCase();
    return phase === 'failed' || phase === 'error';
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
