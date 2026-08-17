# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared theoretical prefix-cache stamping for the segment trie.

``hash_id_scope: "local"`` means one hash namespace per trace file, so a block
first sent by ANY conversation of a trace (root, subagent child, or flat chain)
is a cache hit when any other conversation of the same trace re-sends it. The
counts are computed over ONE shared per-trace cache consumed in recorded global
ARRIVAL order under CAUSAL availability -- a block counts as a hit only once
the request that produced it has FINISHED (see
:func:`compute_causal_prefix_hits`) -- then stamped per node onto the NATIVE
``LlmNode.theoretical_prefix_cache_hit_blocks`` / ``_total_blocks`` fields
(``Turn`` naming) where the ``theoretical_prefix_cache`` results accumulator
reads them back (:func:`extract_prefix_cache_by_node` via the dataset manager).

Adapter-agnostic: any adapter that lowers through the segment trie (dynamo)
gets the stamping by calling :func:`stamp_theoretical_prefix_cache` on its
assembled node map. Nodes whose requests carry no hash blocks are left
unstamped (the accumulator treats a zero total as absent).

The seen-set stays per-trace even for ``hash_id_scope: "global"`` corpora:
recorded ``t`` is conversation-relative, so no cross-file clock exists to order
one trace's sends against another's. Under global scope the stamped hits are
therefore a LOWER BOUND -- a block another trace already sent still counts as a
miss here even though the server (infinite cache) would hit it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, MutableMapping, Sequence
from dataclasses import dataclass

import msgspec

from aiperf.dataset.graph.models import GraphRecord, LlmNode
from aiperf.dataset.graph.segment_trie.trie_content import (
    TrieNode,
    covered_block_count,
)


@dataclass(slots=True, frozen=True)
class CausalRequest:
    """One request's contribution to the causal infinite-cache accounting."""

    hash_ids: Sequence[int]
    """The request's input block hashes, in prompt order."""
    start: float
    """Recorded arrival time. Consumption order, and the availability deadline."""
    end: float
    """Recorded completion time. When this request's blocks enter the cache."""


def compute_causal_prefix_hits(requests: Iterable[CausalRequest]) -> list[int]:
    """Leading-run hit count per request, aligned to the INPUT order.

    Requests are consumed in ARRIVAL order (``start``), and a block only counts
    as a hit once the request that produced it has FINISHED: a block is
    available to a consumer iff some producer's ``end`` is at or before the
    consumer's ``start``. Under concurrency the two orderings genuinely differ
    (a request can complete long after a later-arriving one), and crediting a
    hit against a block that did not physically exist yet inflates the ceiling.

    Availability is the MINIMUM ``end`` over every producer of a block, so a
    slow first producer does not mask a fast later one. The sort is stable on
    ``start`` alone, so the caller's own ordering survives as the tiebreak.

    Degrades to the pure infinite-cache bound when durations are unknown: with
    ``end == start`` every block is available the instant its producer arrives.
    """
    items = list(requests)
    hits = [0] * len(items)
    available_at: dict[int, float] = {}
    for i in sorted(range(len(items)), key=lambda i: items[i].start):
        req = items[i]
        run = 0
        for hid in req.hash_ids:
            avail = available_at.get(hid)
            if avail is None or avail > req.start:
                break
            run += 1
        hits[i] = run
        for hid in req.hash_ids:
            prev = available_at.get(hid)
            if prev is None or req.end < prev:
                available_at[hid] = req.end
    return hits


def compute_shared_prefix_cache_counts(
    trie_nodes: Iterable[TrieNode],
    block_size: int,
) -> dict[str, tuple[int, int]]:
    """``{node_id: (hit_blocks, total_blocks)}`` over one shared seen-set.

    Nodes are consumed in recorded global ARRIVAL order (``request.t``, with
    the flattened recorded-order index as the deterministic tiebreak — the same
    ordering the interval-order rank uses) and credited under causal
    availability (:func:`compute_causal_prefix_hits`): a block is a hit only
    once the request that produced it has finished, on the RAW recorded clock.
    ``hit_blocks`` is the LEADING run of the node's ``hash_ids`` available at
    the node's own start (stop at the first miss); ``total_blocks`` is the
    COVERED-block count (``min(len(hash_ids), in // block_size)``) -- the blocks
    the node actually sends. A recording that hashes its partial tail carries
    one hash MORE than it emits, and counting that block would measure the
    theoretical-cache denominator over traffic that is never sent. Hits are
    clamped to the same bound for the same reason. Every block of every request
    enters the infinite cache regardless of hit position.
    """
    ordered = sorted(trie_nodes, key=lambda n: (n.request.t, n.order))
    hits = compute_causal_prefix_hits(
        CausalRequest(
            hash_ids=node.request.hash_ids,
            start=node.raw_start,
            end=node.raw_end,
        )
        for node in ordered
    )
    counts: dict[str, tuple[int, int]] = {}
    for node, hit in zip(ordered, hits, strict=True):
        total = covered_block_count(
            node.request.hash_ids, node.request.input_length, block_size
        )
        counts[node.node_id] = (min(hit, total), total)
    return counts


def stamp_theoretical_prefix_cache(
    llm_nodes: MutableMapping[str, LlmNode],
    trie_nodes: Iterable[TrieNode],
    block_size: int,
    on_counts: Callable[[int, int], None] | None = None,
) -> None:
    """Stamp per-node counts onto the native ``LlmNode`` fields in ``llm_nodes``.

    ``LlmNode`` is frozen, so each stamped node is REPLACED in the mapping
    (``msgspec.structs.replace``) with the counts on
    ``theoretical_prefix_cache_hit_blocks`` / ``_total_blocks``. Nodes with
    zero hash blocks are skipped; trie nodes without an assembled LlmNode
    (never the case today) are ignored.
    """
    for node_id, (hit_blocks, total_blocks) in compute_shared_prefix_cache_counts(
        trie_nodes, block_size
    ).items():
        if on_counts is not None:
            on_counts(hit_blocks, total_blocks)
        if total_blocks <= 0:
            continue
        llm = llm_nodes.get(node_id)
        if llm is None:
            continue
        llm_nodes[node_id] = msgspec.structs.replace(
            llm,
            theoretical_prefix_cache_hit_blocks=hit_blocks,
            theoretical_prefix_cache_total_blocks=total_blocks,
        )


def extract_prefix_cache_by_node(
    top_graph: GraphRecord,
) -> dict[str, list[int]]:
    """Collect every stamped node's prefix-cache counts as ``{node_id: [hit, total]}``.

    Reads the native ``theoretical_prefix_cache_hit_blocks`` / ``_total_blocks``
    fields off the graph. The key is the node id the graph dispatch path
    reports on the wire, which is what the ``theoretical_prefix_cache``
    accumulator recovers from each record's ``x_correlation_id``. Returns ``{}``
    when no node was stamped (a non-trie graph, or a trace whose requests
    carried no hash blocks).
    """
    out: dict[str, list[int]] = {}
    for node_id, node in top_graph.nodes.items():
        if (
            node.theoretical_prefix_cache_hit_blocks is None
            or node.theoretical_prefix_cache_total_blocks is None
        ):
            continue
        out[node_id] = [
            node.theoretical_prefix_cache_hit_blocks,
            node.theoretical_prefix_cache_total_blocks,
        ]
    return out


__all__ = [
    "CausalRequest",
    "compute_causal_prefix_hits",
    "compute_shared_prefix_cache_counts",
    "extract_prefix_cache_by_node",
    "stamp_theoretical_prefix_cache",
]
