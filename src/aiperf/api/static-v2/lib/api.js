// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Minimal fetch wrapper for the three REST endpoints the dashboard needs. */

async function getJson(path, signal) {
  const resp = await fetch(path, { signal });
  if (!resp.ok) throw new Error(`${path} → HTTP ${resp.status}`);
  return resp.json();
}

export const api = {
  getConfig(signal)       { return getJson('/api/config', signal); },
  getProgress(signal)     { return getJson('/api/progress', signal); },
  getServerMetrics(signal){ return getJson('/api/server-metrics', signal); },
};
