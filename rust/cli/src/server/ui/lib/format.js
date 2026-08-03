// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Number/metric formatting helpers ported from the operator + static-v2 UIs,
// plus the headline-metric metadata this dashboard keys everything off of.
// Every formatter is null/NaN-safe and renders an em-dash for absent data so
// a run with no report still lays out cleanly.

export const DASH = '—';

/** Pick a decimal count that keeps tiny non-zero magnitudes legible. */
function magnitudeAwareDecimals(value, decimals) {
  const abs = Math.abs(value);
  if (abs === 0 || !isFinite(abs)) return decimals;
  if (abs < 0.01) return Math.max(decimals, 5);
  if (abs < 1) return Math.max(decimals, 4);
  return decimals;
}

/**
 * Format a number with comma grouping and magnitude-aware decimals.
 * @param {number|null|undefined} value
 * @param {number} [decimals=2]
 * @param {string} [fallback]
 */
export function fmtNumber(value, decimals = 2, fallback = DASH) {
  if (value == null || typeof value !== 'number' || !isFinite(value)) return fallback;
  const effective = magnitudeAwareDecimals(value, decimals);
  return value.toLocaleString('en-US', {
    minimumFractionDigits: effective,
    maximumFractionDigits: effective,
  });
}

/** Format an integer with comma grouping. */
export function fmtInt(value, fallback = DASH) {
  if (value == null || typeof value !== 'number' || !isFinite(value)) return fallback;
  return Math.round(value).toLocaleString('en-US');
}

/**
 * Adaptive metric display: large magnitudes round to integers with commas,
 * everything else keeps two significant decimals. Used across the tables.
 */
export function fmtMetric(value, fallback = DASH) {
  if (value == null || typeof value !== 'number' || !isFinite(value)) return fallback;
  return Math.abs(value) >= 1000 ? fmtInt(value) : fmtNumber(value, 2);
}

/** Compact time-since-epoch-seconds label (e.g. ``3m ago``, ``2h ago``). */
export function fmtAgo(unixSec) {
  if (unixSec == null || !isFinite(unixSec)) return DASH;
  const secs = Math.max(0, Date.now() / 1000 - unixSec);
  if (secs < 60) return `${Math.round(secs)}s ago`;
  if (secs < 3600) return `${Math.round(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.round(secs / 3600)}h ago`;
  return `${Math.round(secs / 86400)}d ago`;
}

// ── Headline metrics ──────────────────────────────────────────────────────
// The four values ``/api/runs`` carries on each ``headline`` object. Labels
// and "lower is better" direction drive the KPI tiles, sort menu, and
// leaderboard defaults. ``icon`` maps into the KpiCard icon registry.
export const HEADLINE = [
  {
    tag: 'output_token_throughput',
    label: 'Output Throughput',
    short: 'out tok/s',
    unit: 'tok/s',
    icon: 'trending-up',
    lowerBetter: false,
  },
  {
    tag: 'time_to_first_token',
    label: 'Time to First Token',
    short: 'TTFT',
    unit: 'ms',
    icon: 'clock',
    lowerBetter: true,
  },
  {
    tag: 'inter_token_latency',
    label: 'Inter-Token Latency',
    short: 'ITL',
    unit: 'ms',
    icon: 'timer',
    lowerBetter: true,
  },
  {
    tag: 'request_latency',
    label: 'Request Latency',
    short: 'req lat',
    unit: 'ms',
    icon: 'timer',
    lowerBetter: true,
  },
];

// Latency-family tags where a smaller number is better. Used to color
// "best" cells and orient leaderboard/pareto ranking when the metric name
// isn't in HEADLINE.
const LOWER_IS_BETTER_PREFIX = [
  'request_latency',
  'time_to_first_token',
  'time_to_second_token',
  'inter_token_latency',
  'inter_chunk_latency',
];

/** True when smaller is better for a metric tag (latency-like). */
export function isLowerBetter(tag) {
  if (!tag) return false;
  return LOWER_IS_BETTER_PREFIX.some((p) => tag.startsWith(p));
}

/** Human label for an arbitrary metric tag (title-cased words). */
export function prettyTag(tag) {
  if (!tag) return '';
  return String(tag)
    .split('_')
    .map((w) => (w.length <= 3 ? w.toUpperCase() : w[0].toUpperCase() + w.slice(1)))
    .join(' ');
}
