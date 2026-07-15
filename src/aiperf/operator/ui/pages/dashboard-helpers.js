// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export function jobCreatedTs(job) {
  const raw = job?.created ?? job?.startTime ?? job?.completionTime ?? null;
  if (!raw) return 0;
  const t = new Date(raw).getTime();
  return Number.isFinite(t) ? t : 0;
}

export function isRecentJob(job) {
  const phase = (job?.phase ?? '').toLowerCase();
  return phase === 'completed' || phase === 'succeeded' || phase === 'failed' || phase === 'error';
}

export function recentJobs(jobList, limit = 5) {
  const top = [];

  for (const job of jobList ?? []) {
    if (!isRecentJob(job)) continue;
    const item = { j: job, ts: jobCreatedTs(job) };
    const idx = top.findIndex(existing => item.ts > existing.ts);
    if (idx === -1) {
      if (top.length < limit) top.push(item);
    } else {
      top.splice(idx, 0, item);
      if (top.length > limit) top.pop();
    }
  }

  return top.map(({ j }) => j);
}
