// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Number formatting utilities for the operator UI.
 * All numeric displays should use these formatters for consistent comma-separated output.
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
  return value.toLocaleString('en-US', {
    minimumFractionDigits: effective,
    maximumFractionDigits: effective,
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
  return Math.round(value).toLocaleString('en-US');
}

/**
 * Format a throughput value: X,XXX.X req/s
 * @param {number|null|undefined} value
 * @returns {string}
 */
export function fmtThroughput(value) {
  if (value == null) return '---';
  return fmtNumber(value, 1);
}

/**
 * Format a latency value in ms, or convert to seconds if > 1000ms.
 * @param {number|null|undefined} ms
 * @returns {{ value: string, unit: string } | null}
 */
export function fmtLatency(ms) {
  if (ms == null) return null;
  if (ms > 1000) return { value: fmtNumber(ms / 1000, 1), unit: 's' };
  return { value: fmtNumber(ms, 0, '---'), unit: 'ms' };
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
