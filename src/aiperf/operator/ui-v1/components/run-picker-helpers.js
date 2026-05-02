// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Pure helpers for the RunPicker dropdown. Kept in a sibling module
// (no preact/htm imports) so they can be unit-tested via raw Node — see
// tests/unit/ui/test_operator_run_picker.py. The component lives in
// run-picker.js and re-exports these helpers as part of its public surface.

import { runHref } from '../lib/run-selector.js';

export function formatRelativeTime(unixSeconds, now) {
  if (unixSeconds == null) return '';
  const delta = Math.max(0, now - unixSeconds);
  if (delta < 60) return `${Math.floor(delta)}s ago`;
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  if (delta < 604800) return `${Math.floor(delta / 86400)}d ago`;
  return new Date(unixSeconds * 1000).toLocaleDateString([], {
    month: 'short', day: 'numeric',
  });
}

/**
 * Pure helper — returns the menu rows in newest-first order with ordinal
 * "Run N" labels (oldest = Run 1). Exported so unit tests can assert the
 * shape without needing a DOM.
 */
export function buildPickerRows({ namespace, name, epochs, current }) {
  const ascending = [...(epochs || [])].sort(
    (a, b) => (a?.mtimeEpoch ?? 0) - (b?.mtimeEpoch ?? 0)
  );
  // Ordinal: oldest = Run 1, newest = Run M.
  const ordinalByEpoch = new Map();
  ascending.forEach((e, i) => ordinalByEpoch.set(String(e.epoch), i + 1));

  const desc = [...ascending].reverse();
  return desc.map(e => {
    const epochStr = String(e.epoch);
    // When no epoch is pinned (current null/undefined), the latest row is
    // shown as selected — that's the row the page is implicitly rendering.
    const selected = current != null
      ? current === epochStr
      : Boolean(e.isLatest);
    return {
      epoch: epochStr,
      label: `Run ${ordinalByEpoch.get(epochStr)}`,
      status: e.status || 'unknown',
      isLatest: Boolean(e.isLatest),
      selected,
      href: e.isLatest ? runHref(namespace, name) : runHref(namespace, name, epochStr),
      startedAt: e.startedAt ?? null,
      mtimeEpoch: e.mtimeEpoch ?? null,
    };
  });
}

/**
 * Pure helper — returns ``{text, status, isLatest, notLatest, inert}`` describing
 * the collapsed button content, or ``null`` when the picker should not render.
 */
export function buildButtonLabel({ epochs, current, now }) {
  if (!epochs || epochs.length === 0) return null;

  const ascending = [...epochs].sort(
    (a, b) => (a?.mtimeEpoch ?? 0) - (b?.mtimeEpoch ?? 0)
  );
  const ordinalByEpoch = new Map();
  ascending.forEach((e, i) => ordinalByEpoch.set(String(e.epoch), i + 1));

  const latest = ascending[ascending.length - 1];
  const latestEpoch = latest ? String(latest.epoch) : null;
  const viewingLatest = current == null || current === latestEpoch;
  const inert = epochs.length === 1;

  if (viewingLatest && latest) {
    const ord = ordinalByEpoch.get(latestEpoch);
    const status = latest.status || 'unknown';
    const text = status === 'running'
      ? `Run ${ord} · running`
      : `Run ${ord} · ${formatRelativeTime(latest.endedAt ?? latest.startedAt ?? latest.mtimeEpoch, now)}`;
    return { text, status, isLatest: true, notLatest: false, inert };
  }

  // Viewing pinned older epoch (or stale/orphan).
  const found = ascending.find(e => String(e.epoch) === String(current));
  if (!found) {
    return {
      text: `Run ?(${current}) · unknown`,
      status: 'unknown',
      isLatest: false,
      notLatest: true,
      inert: false,
    };
  }
  const ord = ordinalByEpoch.get(String(found.epoch));
  const rel = formatRelativeTime(found.endedAt ?? found.startedAt ?? found.mtimeEpoch, now);
  return {
    text: `Run ${ord} · ${rel} · not latest`,
    status: found.status || 'unknown',
    isLatest: false,
    notLatest: true,
    inert,
  };
}
