// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tone-driven color helpers for KPI tiles. Lives in its own module (no
 * htm/preact import) so the mapping can be unit-tested under plain Node
 * without the CDN-resolved preact runtime — the operator UI uses
 * importmap-based bare-spec resolution which is unavailable in tests.
 */

/**
 * Map a tile `tone` to sparkline stroke/fill so the live trend tracks the
 * same color signal the value number already carries.
 *
 * @param {('accent'|'warn'|'bad'|'ok'|'neutral'|null|undefined|string)} tone
 * @returns {{ stroke: string, fill: string }}
 */
// Mirrors the `.metric-val--<tone>` rules in style.css:971-977 exactly, which is
// what "tracks the same color signal the value number already carries" means.
// The previous mapping broke that contract three ways: `warn` drew RED, silently
// escalating caution to failure; `ok` drew grey, discarding a positive signal the
// number was making in green; and `default:` returned the positive accent, so ANY
// unrecognised or misspelled tone asserted "good" -- the one direction a fallback
// must never assert. Unknown now falls to the neutral sub colour: a fallback
// should make no claim.
const _SPARK_BY_TONE = {
  ok: { stroke: 'var(--green)', fill: 'var(--green-dim)' },
  warn: { stroke: 'var(--amber)', fill: 'rgba(255,193,7,0.15)' },
  gold: { stroke: 'var(--amber)', fill: 'rgba(255,193,7,0.15)' },
  bad: { stroke: 'var(--red)', fill: 'rgba(239,83,80,0.15)' },
  accent: { stroke: 'var(--accent)', fill: 'var(--accent-dim)' },
  live: { stroke: 'var(--accent)', fill: 'var(--accent-dim)' },
};

const _SPARK_NEUTRAL = { stroke: 'var(--sub)', fill: 'rgba(167,167,167,0.10)' };

export function sparkColors(tone) {
  return _SPARK_BY_TONE[tone] ?? _SPARK_NEUTRAL;
}
