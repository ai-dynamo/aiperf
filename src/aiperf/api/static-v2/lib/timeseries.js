// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Bounded rolling-window time series for the live dashboard.
 *
 * Each entry is ``{ t: <ms epoch>, values: { <stat>: number, ... } }``.
 * Samples are appended every time a ``realtime_metrics`` WS message lands
 * (roughly once per second in production). The window is capped in both
 * point count and age so a long-running benchmark doesn't grow unbounded.
 */

export const MAX_POINTS = 120;         // ≈ 2 minutes at 1 Hz
export const MAX_AGE_MS = 5 * 60_000;  // 5 minutes hard ceiling regardless of rate

function finiteTimestamp(value) {
  return typeof value === 'number' && isFinite(value);
}

/** Append one sample, pruning anything older than the window. */
export function pushSample(series, sample) {
  const candidates = series.filter(s => finiteTimestamp(s.t));
  if (finiteTimestamp(sample?.t)) candidates.push(sample);

  const newest = candidates.reduce((mx, s) => Math.max(mx, s.t), -Infinity);
  if (!isFinite(newest)) return [];

  const cutoff = newest - MAX_AGE_MS;
  const next = candidates
    .filter(s => s.t >= cutoff)
    .sort((a, b) => a.t - b.t);
  if (next.length > MAX_POINTS) next.splice(0, next.length - MAX_POINTS);
  return next;
}

/** Extract a numeric series for a given stat key; useful for sparklines. */
export function pluck(series, statKey) {
  const out = [];
  for (const s of series) {
    const v = s.values?.[statKey];
    if (finiteTimestamp(s.t) && typeof v === 'number' && isFinite(v)) out.push({ t: s.t, v });
  }
  return out.sort((a, b) => a.t - b.t);
}

/** Min / max over the pluck()ed series, ignoring empty. */
export function extent(points) {
  if (points.length === 0) return { min: 0, max: 0 };
  let mn = Infinity, mx = -Infinity;
  for (const p of points) {
    if (p.v < mn) mn = p.v;
    if (p.v > mx) mx = p.v;
  }
  return { min: mn, max: mx };
}
