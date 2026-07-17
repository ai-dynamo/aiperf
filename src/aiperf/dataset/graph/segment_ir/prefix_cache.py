# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared theoretical prefix-cache stamping for the segment-trie IR.

``hash_id_scope: "local"`` means one hash namespace per trace file, so a block
first sent by ANY conversation of a trace (root, subagent child, or flat chain)
is a cache hit when any other conversation of the same trace re-sends it. The
counts are computed over ONE shared per-trace seen-set consumed in recorded
global time order, then stamped per node onto the NATIVE
``LlmNode.theoretical_prefix_cache_hit_blocks`` / ``_total_blocks`` fields
(``Turn`` naming) where the ``theoretical_prefix_cache`` results accumulator
reads them back (:func:`extract_prefix_cache_by_node` via the dataset manager).

Adapter-agnostic: any adapter that lowers through the trie IR (weka, dynamo)
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

from collections.abc import Iterable, MutableMapping

import msgspec

from aiperf.dataset.graph.models import GraphRecord, LlmNode
from aiperf.dataset.graph.segment_ir.trie_content import TrieNode


def compute_shared_prefix_cache_counts(
    trie_nodes: Iterable[TrieNode],
) -> dict[str, tuple[int, int]]:
    """``{node_id: (hit_blocks, total_blocks)}`` over one shared seen-set.

    Nodes are consumed in recorded global time order (``request.t``, with the
    flattened recorded-order index as the deterministic tiebreak — the same
    ordering the interval-order rank uses). ``hit_blocks`` is the LEADING run of
    the node's ``hash_ids`` already in the cache (stop at the first miss);
    ``total_blocks`` is the full hash-id count. Every block of every request
    enters the infinite cache regardless of hit position.
    """
    out: dict[str, tuple[int, int]] = {}
    seen: set[int] = set()
    for node in sorted(trie_nodes, key=lambda n: (n.request.t, n.order)):
        hashes = node.request.hash_ids
        hits = 0
        for hid in hashes:
            if hid not in seen:
                break
            hits += 1
        out[node.node_id] = (hits, len(hashes))
        seen.update(hashes)
    return out


def stamp_theoretical_prefix_cache(
    llm_nodes: MutableMapping[str, LlmNode],
    trie_nodes: Iterable[TrieNode],
) -> None:
    """Stamp per-node counts onto the native ``LlmNode`` fields in ``llm_nodes``.

    ``LlmNode`` is frozen, so each stamped node is REPLACED in the mapping
    (``msgspec.structs.replace``) with the counts on
    ``theoretical_prefix_cache_hit_blocks`` / ``_total_blocks``. Nodes with
    zero hash blocks are skipped; trie nodes without an assembled LlmNode
    (never the case today) are ignored.
    """
    for node_id, (hit_blocks, total_blocks) in compute_shared_prefix_cache_counts(
        trie_nodes
    ).items():
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
        if not isinstance(node, LlmNode):
            continue
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
    "compute_shared_prefix_cache_counts",
    "extract_prefix_cache_by_node",
    "stamp_theoretical_prefix_cache",
]
