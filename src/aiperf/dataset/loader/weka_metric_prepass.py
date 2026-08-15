# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared theoretical prefix-cache pre-pass for Weka traces (spec §5.5).

``hash_id_scope: "local"`` means one hash namespace per trace FILE, so a
block first sent by any conversation of a trace (root, subagent child, or
detected flat chain) is a cache hit when any other conversation of the same
trace re-sends it. This module computes those values over ONE shared
per-trace cache consumed in global arrival order under causal availability;
emission then looks them up per ``(session_id, turn_index)`` instead of
keeping per-conversation caches.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class MetricRecord:
    """One request's contribution to the per-trace shared cache."""

    sort_key: tuple[float, int, int, int]
    """(absolute_t, outer_idx, stream_idx, k) — deterministic global order."""
    session_id: str
    """Conversation the value is looked up under at emission time."""
    k: int
    """Turn index within that conversation."""
    hash_ids: list[int]
    """The request's input hash blocks."""
    api_time: float = 0.0
    """Recorded request duration, seconds. Blocks enter the cache at t+api_time.

    Defaults to 0.0 so a caller with no recorded duration degrades to the pure
    infinite-cache bound (blocks available the instant their producer arrives).
    """


def compute_shared_prefix_cache_metrics(
    records: list[MetricRecord],
) -> dict[tuple[str, int], tuple[int, int]]:
    """{(session_id, k): (hit_blocks, total_blocks)} over ONE shared per-trace
    cache, consumed in global arrival order under causal availability (spec §5.5).

    A block counts as a hit only once the request that produced it has FINISHED
    at or before the consumer's arrival -- see
    :func:`aiperf.dataset.graph.segment_trie.prefix_cache.compute_causal_prefix_hits`,
    which owns the accounting for every loader path.
    """
    from aiperf.dataset.graph.segment_trie.prefix_cache import (
        CausalRequest,
        compute_causal_prefix_hits,
    )

    ordered = sorted(records, key=lambda r: r.sort_key)
    hits = compute_causal_prefix_hits(
        CausalRequest(
            hash_ids=rec.hash_ids,
            start=rec.sort_key[0],
            end=rec.sort_key[0] + rec.api_time,
        )
        for rec in ordered
    )
    return {
        (rec.session_id, rec.k): (hit, len(rec.hash_ids))
        for rec, hit in zip(ordered, hits, strict=True)
    }
