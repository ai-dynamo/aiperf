import { signal } from '@preact/signals';

// Raw jobs list from /api/v1/jobs
export const jobs = signal([]);

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
