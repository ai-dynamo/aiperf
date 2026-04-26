// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Per-namespace UI preferences persisted in ``localStorage``.
 *
 * Keys:
 *   ``aiperf.ui.lastNamespace``                    sticky last-used namespace
 *   ``aiperf.ui.ns.<ns>.pinnedRunNames``           pinned runs surfaced on overview
 *   ``aiperf.ui.ns.<ns>.lastLaunchTemplateId``     auto-loaded launch template
 *   ``aiperf.ui.ns.<ns>.overviewMetricKey``        chart series key on overview
 *
 * Best-effort: missing key returns the supplied default; quota / disabled
 * storage errors are swallowed so the UI never crashes on persistence.
 */

const LAST_NS_KEY = 'aiperf.ui.lastNamespace';

function nsKey(ns, key) {
  return `aiperf.ui.ns.${ns}.${key}`;
}

export function getNsPref(ns, key, fallback) {
  try {
    const raw = window.localStorage.getItem(nsKey(ns, key));
    if (raw == null) return fallback;
    return JSON.parse(raw);
  } catch (_e) {
    return fallback;
  }
}

export function setNsPref(ns, key, value) {
  try {
    window.localStorage.setItem(nsKey(ns, key), JSON.stringify(value));
  } catch (_e) {
    // quota / disabled storage / SecurityError — drop on the floor
  }
}

export function getLastNamespace() {
  try {
    const raw = window.localStorage.getItem(LAST_NS_KEY);
    return raw == null ? null : raw;
  } catch (_e) {
    return null;
  }
}

export function setLastNamespace(ns) {
  try {
    if (ns) window.localStorage.setItem(LAST_NS_KEY, ns);
  } catch (_e) { /* swallow */ }
}
