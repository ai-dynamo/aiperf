// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared number / time / bytes formatters.
 * Mirrors src/aiperf/operator/ui/lib/format.js so both dashboards format
 * identical values the same way.
 */

export function fmtNumber(value, decimals = 1, fallback = '---') {
  if (value == null) return fallback;
  if (typeof value !== 'number' || !isFinite(value)) return String(value);
  return value.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export function fmtInt(value, fallback = '---') {
  if (value == null) return fallback;
  if (typeof value !== 'number' || !isFinite(value)) return String(value);
  return Math.round(value).toLocaleString('en-US');
}

export function fmtPercent(value, decimals = 1) {
  if (value == null) return '---';
  return fmtNumber(value, decimals) + '%';
}

export function fmtBytes(bytes) {
  if (bytes == null || typeof bytes !== 'number' || !isFinite(bytes)) return '---';
  if (bytes < 1024) return fmtInt(bytes) + ' B';
  if (bytes < 1024 * 1024) return fmtNumber(bytes / 1024, 1) + ' KiB';
  if (bytes < 1024 * 1024 * 1024) return fmtNumber(bytes / (1024 * 1024), 1) + ' MiB';
  return fmtNumber(bytes / (1024 * 1024 * 1024), 2) + ' GiB';
}

/** Format an elapsed duration in seconds as "1d 2h 3m" / "4h 5m 6s" / "7m 8s" / "9s". */
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

/** Seconds since a backend-supplied ns timestamp. Returns null if not a number. */
export function secondsSinceNs(ns) {
  if (ns == null || !isFinite(Number(ns))) return null;
  return (Date.now() - Number(ns) / 1e6) / 1000;
}

/** HH:MM:SS in local time for log lines. */
export function fmtClock(date = new Date()) {
  return date.toLocaleTimeString();
}
