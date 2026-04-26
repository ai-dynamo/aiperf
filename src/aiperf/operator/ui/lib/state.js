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

// Launch view divergence: when the YAML's top-level ``namespace:`` field
// disagrees with the URL's ``:ns`` segment, this holds the offending YAML
// value (string); null otherwise. The top rail reads this to mark the
// namespace pill as bad. The launch view writes it on every keystroke
// (debounced) and clears it on unmount.
export const launchDivergence = signal(null);
