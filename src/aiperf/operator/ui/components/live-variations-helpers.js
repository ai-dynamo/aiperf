const PHASE_DONE = new Set(['Succeeded', 'Completed', 'Archived']);

export function trialContributesMetrics(phase) {
  return PHASE_DONE.has(phase);
}
