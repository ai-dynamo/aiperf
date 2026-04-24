/**
 * Number formatting utilities for the operator UI.
 * All numeric displays should use these formatters for consistent comma-separated output.
 */

/**
 * Format a number with commas and fixed decimal places.
 * @param {number|null|undefined} value
 * @param {number} decimals - Number of decimal places (default: 1)
 * @param {string} fallback - Fallback text for null/undefined (default: '---')
 * @returns {string}
 */
export function fmtNumber(value, decimals = 1, fallback = '---') {
  if (value == null) return fallback;
  if (typeof value !== 'number' || !isFinite(value)) return String(value);
  return value.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
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
  if (typeof value !== 'number' || !isFinite(value)) return String(value);
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
  if (value == null) return '---';
  return fmtNumber(value, decimals) + '%';
}

/**
 * Format an elapsed duration given in **seconds** as "1d 2h 3m" / "4h 5m 6s" /
 * "7m 8s" / "9s". Mirrors the static-v2 dashboard formatter so both UIs show
 * identical duration strings.
 * @param {number|null|undefined} seconds
 * @returns {string}
 */
export function fmtDuration(seconds) {
  if (seconds == null || !isFinite(seconds)) return '---';
  const s = Math.max(0, Math.floor(seconds));
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (d > 0) return `${d}d ${h}h ${m}m`;
  if (h > 0) return `${h}h ${m}m ${sec}s`;
  if (m > 0) return `${m}m ${sec}s`;
  return `${sec}s`;
}

/**
 * Format file size in human-readable form.
 * @param {number} bytes
 * @returns {string}
 */
export function fmtBytes(bytes) {
  if (bytes == null || typeof bytes !== 'number' || !isFinite(bytes)) return '---';
  if (bytes < 1024) return fmtInt(bytes) + ' B';
  if (bytes < 1024 * 1024) return fmtNumber(bytes / 1024, 1) + ' KiB';
  return fmtNumber(bytes / (1024 * 1024), 1) + ' MiB';
}
