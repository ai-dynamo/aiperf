// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Thin fetch layer over the native cross-run dashboard REST contract. Every
// call throws ``Error`` on non-2xx so pages can render an error strip and keep
// the shell mounted. The contract is fixed server-side — see the module docs
// in ``server/mod.rs``; this file only consumes it.

/**
 * Low-level JSON GET. Throws ``Error`` (message ``API <status>: <body>``) on
 * any non-2xx response so callers get a consistent failure shape.
 * @param {string} path
 * @returns {Promise<any>}
 */
async function getJson(path) {
  const resp = await fetch(path, { headers: { Accept: 'application/json' } });
  if (!resp.ok) {
    const text = await resp.text().catch(() => resp.statusText);
    throw new Error(`API ${resp.status}: ${text}`);
  }
  return resp.json();
}

export const api = {
  /** ``{ service, started_unix, results_root, session_runs }`` */
  meta() {
    return getJson('/api/meta');
  },

  /** Array of run summaries (see the RunSummary shape in the server). */
  runs() {
    return getJson('/api/runs');
  },

  /** The full ``native-v2.json`` report for one run. */
  run(id) {
    return getJson(`/api/runs/${encodeURIComponent(id)}`);
  },

  /** The convenient projected summary: ``{ run, headline, metrics }``. */
  summary(id) {
    return getJson(`/api/runs/${encodeURIComponent(id)}/summary`);
  },
};
