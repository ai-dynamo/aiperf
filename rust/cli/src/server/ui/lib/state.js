// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared app signals + a small summary cache. The runs list and meta are
// polled once at the shell level and every page reads them from here, so a
// live ``--serve`` session shows new runs without each page re-fetching.

import { signal } from '@preact/signals';
import { api } from './api.js';

/** Latest ``/api/runs`` payload (array). */
export const runs = signal([]);
/** Latest ``/api/meta`` payload (object or null). */
export const meta = signal(null);
/** App-level error banner text (null = healthy). */
export const globalError = signal(null);
/** Selected run ids for the Compare tray (array of id strings). */
export const compareSel = signal([]);

// ── Summary cache ─────────────────────────────────────────────────────────
// Compare/Leaderboard/Sweeps need many runs' projected summaries. Dedupe the
// in-flight promises so re-renders don't refetch, and drop rejected entries
// so a transient failure can be retried.
const summaryCache = new Map();

/** Fetch (and memoize) one run's ``/summary``. */
export function getSummary(id) {
  if (summaryCache.has(id)) return summaryCache.get(id);
  const p = api.summary(id).catch((err) => {
    summaryCache.delete(id);
    throw err;
  });
  summaryCache.set(id, p);
  return p;
}

/**
 * Load summaries for many run ids, tolerating individual failures.
 * @returns {Promise<Array<{id, summary}|null>>} nulls for runs that failed.
 */
export async function loadSummaries(ids) {
  const settled = await Promise.allSettled(ids.map((id) => getSummary(id)));
  return settled.map((r, i) =>
    r.status === 'fulfilled' ? { id: ids[i], summary: r.value } : null,
  );
}

/** Poll ``/api/meta`` + ``/api/runs`` once, updating the shared signals. */
export async function refreshRuns() {
  const [m, r] = await Promise.all([api.meta(), api.runs()]);
  meta.value = m;
  runs.value = Array.isArray(r) ? r : [];
  globalError.value = null;
}

/** Toggle a run id in the compare selection. */
export function toggleCompare(id) {
  const cur = compareSel.value;
  compareSel.value = cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id];
}
