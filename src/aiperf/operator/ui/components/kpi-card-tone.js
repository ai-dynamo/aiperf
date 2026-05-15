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
export function sparkColors(tone) {
  switch (tone) {
    case 'warn':
    case 'bad':
      return { stroke: 'var(--red)', fill: 'rgba(239,83,80,0.15)' };
    case 'ok':
    case 'neutral':
    case undefined:
    case null:
      return { stroke: 'var(--sub)', fill: 'rgba(167,167,167,0.10)' };
    default:
      return { stroke: 'var(--accent)', fill: 'var(--accent-dim)' };
  }
}
