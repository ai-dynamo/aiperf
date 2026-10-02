// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Pure helpers for the Live Variations card.
 *
 * No `htm` import, so these can be exercised directly from Node -- the card
 * itself cannot, which is why anything worth testing lives here.
 */

import { sweptValueEntries } from '../pages/sweep-detail-helpers.js';

const PHASE_DONE = new Set(['Succeeded', 'Completed', 'Archived']);

export function trialContributesMetrics(phase) {
  return PHASE_DONE.has(phase);
}

/** Title-case a single token, preserving digits and inner punctuation. */
export function titleCase(token) {
  if (!token) return token;
  return token
    .split(/[_\-]+/)
    .filter(Boolean)
    .map(w => (w[0] ? w[0].toUpperCase() + w.slice(1) : w))
    .join(' ');
}

/**
 * The swept parameter values as `[{name, value}]` chips.
 *
 * The chip adapter over `sweptValueEntries`, which owns every guard: this
 * returns `[]` -- never a partial chip -- for unparseable input, the
 * `__aiperf_truncated__` marker, null values, and nested objects or lists.
 * Callers fall back to `parseVariationLabel`.
 *
 * Preferred over `parseVariationLabel` whenever it yields anything, because it
 * reads the structured values the sweep actually applied rather than
 * reverse-engineering them from a display string. It is also the only option
 * for adaptive sweeps, whose labels are `search_iter_NNNN` -- a planner counter
 * that carries no parameter information at all.
 *
 * Chips and `formatVariationValues`'s string differ only in presentation (a
 * title-cased name in its own element vs. `leaf=value` inline). They cannot
 * differ in WHICH parameters they show, which is what the three parallel
 * implementations kept getting wrong.
 */
export function parseVariationValues(values) {
  return sweptValueEntries(values).map(entry => ({
    name: titleCase(entry.leaf),
    value: String(entry.value),
  }));
}
