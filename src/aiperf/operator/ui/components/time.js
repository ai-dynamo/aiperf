// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { useEffect, useState } from 'preact/hooks';

/**
 * Format a duration in seconds as a compact relative string.
 * Examples: '12s', '4m', '2h', '5d'.
 *
 * @param {number|null|undefined} seconds
 * @returns {string}
 */
export function fmtRelativeSeconds(seconds) {
  if (seconds == null || !Number.isFinite(seconds)) return '---';
  const s = Math.max(0, Math.floor(seconds));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h`;
  return `${Math.floor(h / 24)}d`;
}

/**
 * Format a duration in seconds with two units of precision.
 * Examples: '45s', '5m 30s', '2h 15m', '3d'.
 *
 * @param {number|null|undefined} seconds
 * @returns {string}
 */
export function fmtElapsedSeconds(seconds) {
  if (seconds == null || !Number.isFinite(seconds)) return '---';
  const s = Math.max(0, Math.floor(seconds));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ${m % 60}m`;
  return `${Math.floor(h / 24)}d`;
}

/**
 * Format an ISO timestamp as a localised absolute string.
 *
 * @param {string|number|Date|null|undefined} ts
 * @returns {string}
 */
export function fmtAbsolute(ts) {
  if (!ts) return '';
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return String(ts);
  return d.toLocaleString();
}

/**
 * Render a compact relative-time string with a tooltip showing the absolute
 * timestamp on hover.
 *
 * Usage:
 *   <RelativeTime ts="2026-04-25T18:12:03Z" />              // since now
 *   <RelativeTime seconds={300} />                          // raw duration
 *   <RelativeTime ts={start} mode="elapsed" />              // 2 units of precision
 *   <RelativeTime ts={start} prefix="ago" />                // suffix like "ago"
 *
 * Props:
 *   ts:       string|number|Date  — anchor timestamp
 *   seconds:  number              — used when ts not given
 *   mode:     'short' | 'elapsed' (default 'short')
 *   suffix:   string              — appended after the value (e.g. 'ago')
 *   className: string
 */
export function RelativeTime({ ts, seconds, mode, suffix, className, title: titleProp }) {
  // Force a re-render on an adaptive cadence so "30s ago" text doesn't go stale.
  const [, setTick] = useState(0);
  let durationSeconds = seconds;
  if (durationSeconds == null && ts != null) {
    const t = new Date(ts).getTime();
    if (!Number.isNaN(t)) durationSeconds = (Date.now() - t) / 1000;
  }
  // Pick a re-render interval scaled to the magnitude of the offset:
  //   <60s -> 5s, <1h -> 30s, <1d -> 5min, otherwise stable (no tick).
  let intervalMs = 0;
  if (durationSeconds != null && Number.isFinite(durationSeconds)) {
    const abs = Math.abs(durationSeconds);
    if (abs < 60) intervalMs = 5_000;
    else if (abs < 3_600) intervalMs = 30_000;
    else if (abs < 86_400) intervalMs = 300_000;
  }
  // Only schedule a live timer when we're driving from a wall-clock anchor (ts);
  // raw `seconds` durations don't change on their own.
  const live = ts != null && intervalMs > 0;
  useEffect(() => {
    if (!live) return undefined;
    const id = setInterval(() => setTick((n) => (n + 1) | 0), intervalMs);
    return () => clearInterval(id);
  }, [live, intervalMs]);

  if (durationSeconds == null || !Number.isFinite(durationSeconds)) {
    return html`<span class=${className}>---</span>`;
  }
  const text = mode === 'elapsed'
    ? fmtElapsedSeconds(durationSeconds)
    : fmtRelativeSeconds(durationSeconds);
  const defaultTitle = ts != null ? fmtAbsolute(ts) : `${Math.floor(durationSeconds)}s`;
  const title = titleProp != null ? titleProp : defaultTitle;
  return html`
    <span class=${className} title=${title}>
      ${text}${suffix ? ' ' + suffix : ''}
    </span>
  `;
}
