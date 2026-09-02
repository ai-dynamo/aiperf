// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Number formatting utilities for the operator UI.
 * All numeric displays should use these formatters for consistent, locale-aware output.
 */

/**
 * Pick a decimal count for a finite numeric magnitude, expanding precision
 * for tiny non-zero values so they don't collapse to a string of zeros.
 *
 * Bands (open intervals on |value|):
 *   (0, 0.01)  -> max(decimals, 5)
 *   [0.01, 1)  -> max(decimals, 4)
 *   otherwise  -> decimals
 *
 * Exact 0 (and Infinity / NaN, which the caller filters) honor the requested
 * decimals so "0.00" stays "0.00".
 * @param {number} value
 * @param {number} decimals
 * @returns {number}
 */
function magnitudeAwareDecimals(value, decimals) {
  const abs = Math.abs(value);
  if (abs === 0 || !isFinite(abs)) return decimals;
  if (abs < 0.01) return Math.max(decimals, 5);
  if (abs < 1) return Math.max(decimals, 4);
  return decimals;
}

/**
 * Drop zeros this formatter ADDED, never digits the caller asked for.
 *
 * `magnitudeAwareDecimals` widens the decimal count so a small value stays
 * visible, but it widens to a fixed floor rather than to "just enough". At
 * decimals=1 that turned 0.004 into "0.00400" -- three significant figures
 * invented from a one-significant-figure input, and trailing zeros after a
 * decimal point read as significant. The docstring's own motivating example is
 * the giveaway: 0.04 at decimals=2 is already "0.04" via toFixed, yet the
 * [0.01, 1) band fires and pads it to "0.0400".
 *
 * Trimming is floored at the caller's requested `decimals`, so a column asking
 * for 2 keeps "1.50" and stays decimal-aligned.
 */
function trimAddedZeros(text, requestedDecimals) {
  if (!text.includes('.')) return text;
  const [whole, frac] = text.split('.');
  let end = frac.length;
  while (end > requestedDecimals && frac[end - 1] === '0') end--;
  return end === 0 ? whole : `${whole}.${frac.slice(0, end)}`;
}

/**
 * Format a number with commas and fixed decimal places.
 *
 * For tiny non-zero values, the effective decimal count is expanded so a
 * per-GPU normalized throughput like 0.04 req/s/GPU doesn't render as
 * "0.00" at decimals=2. See {@link magnitudeAwareDecimals} for the bands.
 * @param {number|null|undefined} value
 * @param {number} decimals - Number of decimal places (default: 1)
 * @param {string} fallback - Fallback text for null/undefined (default: '---')
 * @returns {string}
 */
export function fmtNumber(value, decimals = 1, fallback = '---') {
  if (value == null) return fallback;
  if (typeof value !== 'number' || !isFinite(value)) return fallback;
  const effective = magnitudeAwareDecimals(value, decimals);
  const text = value.toLocaleString(undefined, {
    minimumFractionDigits: effective,
    maximumFractionDigits: effective,
  });
  return trimAddedZeros(text, decimals);
}

/**
 * Pick ONE decimal count for a whole table column.
 *
 * `fmtNumber` decides precision per value, which is right for a standalone KPI
 * and wrong for a column: a req/s column holding 0.5329 and 2 rendered them as
 * "0.5329" and "2", so the decimal point moved four places between adjacent
 * rows and the magnitudes stopped being visually comparable. Uniform decimals
 * per column is the standard fix -- it is what makes a column scannable.
 *
 * The count is the widest any single value needs, so the rule inherits
 * `magnitudeAwareDecimals`' guarantee that no non-zero value collapses to
 * "0.00". The cost is trailing zeros on the large values ("2.0000" beside
 * "0.5329"). That is the deliberate trade: within a column, trailing zeros
 * state the column's precision, whereas a wandering decimal point misstates
 * the magnitudes, which is the worse error.
 *
 * Non-finite and null entries are ignored; an empty or all-null column falls
 * back to `decimals`.
 * @param {Array<number|null|undefined>} values
 * @param {number} decimals - Baseline the column requests.
 * @returns {number}
 */
export function columnDecimals(values, decimals = 1) {
  let widest = decimals;
  for (const value of values ?? []) {
    if (typeof value !== 'number' || !isFinite(value)) continue;
    const needed = magnitudeAwareDecimals(value, decimals);
    if (needed > widest) widest = needed;
  }
  return widest;
}

/**
 * Format at EXACTLY `decimals` places -- no magnitude expansion, no trimming.
 *
 * Pair with {@link columnDecimals} to render a column. `fmtNumber` cannot be
 * reused here: it re-derives precision per value, so a single tiny cell would
 * widen past the column's count and re-introduce the ragged decimal point this
 * exists to remove.
 * @param {number|null|undefined} value
 * @param {number} decimals
 * @param {string} fallback
 * @returns {string}
 */
export function fmtFixed(value, decimals = 1, fallback = '---') {
  if (value == null) return fallback;
  if (typeof value !== 'number' || !isFinite(value)) return fallback;
  const places = Math.max(0, decimals);
  return value.toLocaleString(undefined, {
    minimumFractionDigits: places,
    maximumFractionDigits: places,
  });
}

/**
 * Format an integer with commas (no decimal places).
 * @param {number|null|undefined} value
 * @param {string} fallback
 * @returns {string}
 */
export function fmtInt(value, fallback = '---') {
  if (value == null) return fallback;
  if (typeof value !== 'number' || !isFinite(value)) return fallback;
  return Math.round(value).toLocaleString();
}

/**
 * Two-decimal readout that never reports a non-zero measurement as zero.
 *
 * The console pins rates and latencies to two decimals so one value keeps one
 * shape everywhere. At that width anything under 0.005 renders as "0.00", which
 * does not read as "small" -- it reads as "none", and a benchmark completing one
 * request every five minutes (0.0033 req/s) is not a benchmark producing
 * nothing. Widen only in that case, so the uniform width holds for every value
 * a reader would otherwise be comparing.
 * @param {number|null|undefined} value
 * @param {string} fallback
 * @returns {string}
 */
function fmtTwoDecimals(value, fallback = '---') {
  if (value == null) return fallback;
  if (typeof value !== 'number' || !isFinite(value)) return fallback;
  if (value !== 0 && Math.abs(value) < 0.005) return fmtNumber(value, 2, fallback);
  return fmtFixed(value, 2);
}

/**
 * Format request throughput as a comma-grouped, two-decimal readout.
 * @param {number|null|undefined} value
 * @returns {string}
 */
export function fmtThroughput(value) {
  return fmtTwoDecimals(value);
}

/**
 * Format a request rate as a comma-grouped, two-decimal readout.
 *
 * This explicit alias lets unit-aware callers state their intent instead of
 * relying on the older generic throughput name.
 * @param {number|null|undefined} value
 * @returns {string}
 */
export function fmtReqPerSecond(value) {
  return fmtTwoDecimals(value);
}

/**
 * Format a latency value in milliseconds with comma grouping and two decimals.
 * @param {number|null|undefined} ms
 * @returns {{ value: string, unit: string } | null}
 */
export function fmtLatency(ms) {
  if (ms == null || typeof ms !== 'number' || !isFinite(ms)) return null;
  return { value: fmtTwoDecimals(ms), unit: 'ms' };
}

/**
 * Format a latency number without its unit.
 * @param {number|null|undefined} ms
 * @param {string} fallback
 * @returns {string}
 */
export function fmtMilliseconds(ms, fallback = '---') {
  return fmtTwoDecimals(ms, fallback);
}

/**
 * Format a latency value as a simple string with unit.
 * @param {number|null|undefined} ms
 * @returns {string}
 */
export function fmtLatencyStr(ms) {
  const result = fmtLatency(ms);
  if (!result) return '---';
  return `${result.value} ${result.unit}`;
}

/**
 * Format a number with 3 decimal places (for precise metric displays).
 * @param {number|null|undefined} value
 * @param {string} fallback
 * @returns {string}
 */
export function fmtPrecise(value, fallback = '\u2014') {
  return fmtNumber(value, 3, fallback);
}

/**
 * Format a percentage value (e.g., 75.6%).
 * @param {number|null|undefined} value - Already in percent (0-100)
 * @param {number} decimals
 * @returns {string}
 */
export function fmtPercent(value, decimals = 1) {
  if (value == null || typeof value !== 'number' || !isFinite(value)) return '---';
  return fmtNumber(value, decimals) + '%';
}

/**
 * Format file size in human-readable form.
 * @param {number} bytes
 * @returns {string}
 */
export function fmtBytes(bytes) {
  if (bytes == null || typeof bytes !== 'number' || !isFinite(bytes) || bytes < 0) return '---';
  if (bytes < 1024) return fmtInt(bytes) + ' B';
  if (bytes < 1024 * 1024) return fmtNumber(bytes / 1024, 1) + ' KiB';
  return fmtNumber(bytes / (1024 * 1024), 1) + ' MiB';
}
