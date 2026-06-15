# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared theoretical prefix-cache pre-pass for Weka traces (spec §5.5).

``hash_id_scope: "local"`` means one hash namespace per trace FILE, so a
block first sent by any conversation of a trace (root, subagent child, or
detected flat chain) is a cache hit when any other conversation of the same
trace re-sends it. This module computes those values over ONE shared
per-trace seen-set consumed in global time order; emission then looks them
up per ``(session_id, turn_index)`` instead of keeping per-conversation
seen-sets.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class MetricRecord:
    """One request's contribution to the per-trace shared seen-set."""

    sort_key: tuple[float, int, int, int]
    """(absolute_t, outer_idx, stream_idx, k) — deterministic global order."""
    session_id: str
    """Conversation the value is looked up under at emission time."""
    k: int
    """Turn index within that conversation."""
    hash_ids: list[int]
    """The request's input hash blocks."""


def compute_shared_prefix_cache_metrics(
    records: list[MetricRecord],
) -> dict[tuple[str, int], tuple[int, int]]:
    """{(session_id, k): (hit_blocks, total_blocks)} over ONE shared
    per-trace seen-set, consumed in global time order (spec §5.5)."""
    out: dict[tuple[str, int], tuple[int, int]] = {}
    seen: set[int] = set()
    for rec in sorted(records, key=lambda r: r.sort_key):
        hits = 0
        for hid in rec.hash_ids:
            if hid not in seen:
                break
            hits += 1
        out[(rec.session_id, rec.k)] = (hits, len(rec.hash_ids))
        seen.update(rec.hash_ids)
    return out
