# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interval-order dependency-edge derivation for the segment trie.

Format-agnostic: given a set of nodes that each expose a recorded interval
(``raw_start`` / ``raw_end``), a warped ``start`` / ``end`` (the clock the
runtime replays), a time-consistent ``rank``, and a set of enclosing async
subtree ids, derive each node's incoming ``StaticEdge``s by the finished-before +
transitive-reduction frontier rule. Any adapter whose nodes expose that duck-typed
surface -- a ``node_id`` str, an int ``rank``, float ``start`` / ``end`` (the
idle-gap-warped clock the runtime replays), float ``raw_start`` / ``raw_end`` (the
RAW recorded who-finished-before-whom interval), and a ``frozenset[str]``
``async_ancestors`` (enclosing fire-and-forget subtree ids) -- can reuse this
(``trie_content.build_segment_trie`` is the caller today); the rule reads only those
attributes, never a recorded-trace type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiperf.common.constants import MICROS_PER_SECOND
from aiperf.dataset.graph.models import START_NODE_ID, StaticEdge

if TYPE_CHECKING:
    from aiperf.dataset.graph.segment_trie.trie_content import TrieNode


def compute_ranks(nodes: list) -> None:
    """Stamp each node's time-consistent global ``rank``.

    The rank is the node's index in the total order sorted by
    ``(start, end, node_id)`` -- a linear extension of raw finished-before
    (idle-warp monotonicity), so the interval-order edge rule is a strict partial
    order and its transitive reduction is always a DAG. Mutates ``node.rank``.
    """
    for i, node in enumerate(sorted(nodes, key=lambda n: (n.start, n.end, n.node_id))):
        node.rank = i


def _excluded_async(cand: Any, target: Any) -> bool:
    """``True`` when ``cand`` sits under an async boundary ``target`` does not share.

    A fire-and-forget (``async_launched``) subtree never AND-joins the scope that
    launched it: a candidate whose enclosing async-subtree ids are NOT a subset of
    the target's is excluded from the target's predecessor set before the frontier
    filter runs.
    """
    return not cand.async_ancestors <= target.async_ancestors


def build_interval_edges(nodes: list) -> dict[str, list[StaticEdge]]:
    """Derive every node's incoming interval-order edges globally.

    Semantics:

    * ``A -> B`` iff ``A`` finished before ``B`` started (RAW clock,
      ``A.raw_end <= B.raw_start``) AND ``rank(A) < rank(B)``.
    * Async exclusion (:func:`_excluded_async`) drops fire-and-forget children
      from the candidate set BEFORE the frontier filter.
    * Frontier (transitive reduction): keep only the MAXIMAL finished-before
      candidates -- drop ``c`` when another candidate ``d`` has ``c`` finished
      before ``d`` AND the covering edge ``c -> d`` actually exists, i.e. ``d``
      does not async-exclude ``c`` (``c -> d -> node``, so ``c`` is transitively
      covered). Without the exclusion check, a main-chain ``d`` outside ``c``'s
      async subtree would drop ``c`` while carrying no ``c -> d`` edge itself,
      losing the recorded ``c``-before-``node`` ordering inside the subtree.
    * Binding-cause delay: the latest-ending frontier predecessor (``max by
      .end``) carries the warped end-to-start gap; every other frontier
      predecessor is an AND-join wait (delay ``0.0``).
    * Empty frontier: the node roots at ``START`` at its own warped arrival offset.

    Complexity: the per-node frontier filter is ``O(|candidates|^2)`` worst case,
    so total is ``Theta(n^2)`` per node up to ``Theta(n^3)`` for a pathological
    wide fan-in (rare). A dropped ``c`` is always transitively covered: the
    dominating ``d`` carries a real ``c -> d`` edge (by induction on ``d``'s own
    candidate set, which includes ``c`` because ``d`` does not async-exclude it).
    """
    by_rank = sorted(nodes, key=lambda n: n.rank)
    out: dict[str, list[StaticEdge]] = {}
    for node in nodes:
        cands = [
            c
            for c in by_rank
            if c is not node
            and c.rank < node.rank
            and c.raw_end <= node.raw_start
            and not _excluded_async(c, node)
        ]
        if not cands:
            out[node.node_id] = [
                StaticEdge(
                    source=START_NODE_ID,
                    target=node.node_id,
                    min_start_delay_us=node.start * MICROS_PER_SECOND,
                )
            ]
            continue
        # Frontier = maximal finished-before candidates (drop c if the edge
        # c -> d exists for some later-ranked d in the candidate set). The
        # edge exists only when d does not async-exclude c -- a main-chain d
        # outside c's async subtree carries no c -> d edge, so it cannot cover c.
        frontier = [
            c
            for i, c in enumerate(cands)
            if not any(
                c.raw_end <= d.raw_start and not _excluded_async(c, d)
                for d in cands[i + 1 :]
            )
        ]
        binding = max(frontier, key=lambda c: c.end)
        out[node.node_id] = [
            StaticEdge(
                source=c.node_id,
                target=node.node_id,
                delay_after_predecessor_us=(
                    max(0.0, node.start - c.end) * MICROS_PER_SECOND
                    if c is binding
                    else 0.0
                ),
            )
            for c in frontier
        ]
    return out


def apply_start_anchors(
    nodes: list[TrieNode], edges_by_node: dict[str, list[StaticEdge]]
) -> None:
    """Replace an overlapped node's incoming edges with one start-anchored edge.

    For each node whose ``causal_parent_id`` names another node in the set
    and whose recorded start falls INSIDE that parent's recorded interval
    (``parent.raw_start <= node.raw_start < parent.raw_end``), the
    interval-order edges are replaced with a single
    ``StaticEdge(parent -> node, delay_after_predecessor_start_us=D)``
    where ``D`` is the warped start-to-start gap. The runtime schedules
    such a node at its parent's DISPATCH and gates it ``D`` later, so
    recorded mid-flight concurrency (subagent spawns, aux tool calls)
    tracks the parent causally instead of freezing to the recorded wall
    clock. Nodes whose causal parent had already finished keep their
    interval-order edges -- end-anchoring is correct there by construction.

    When the parent is a streaming request (``parent.request.ttft`` set) and the
    child was recorded at/after that first token (``node.raw_start -
    parent.raw_start >= ttft``), the edge additionally carries
    ``delay_after_predecessor_first_token_us = D - ttft*1e6`` so the runtime can
    re-anchor onto the parent's OBSERVED first token, falling back to
    dispatch + ``D`` when the parent terminates without streaming one. A
    pre-TTFT child (started before the recorded first token) or a non-streaming
    parent (``ttft is None``) keeps the pure dispatch anchor.
    """
    by_id = {n.node_id: n for n in nodes}
    for node in nodes:
        pid = node.causal_parent_id
        if pid is None:
            continue
        parent = by_id.get(pid)
        if parent is None or parent is node:
            continue
        if not (parent.raw_start <= node.raw_start < parent.raw_end):
            continue
        delay_us = max(0.0, node.start - parent.start) * MICROS_PER_SECOND
        ttft_s = parent.request.ttft
        first_token_delay_us = None
        if ttft_s is not None and (node.raw_start - parent.raw_start) >= ttft_s:
            first_token_delay_us = max(0.0, delay_us - ttft_s * MICROS_PER_SECOND)
        edges_by_node[node.node_id] = [
            StaticEdge(
                source=parent.node_id,
                target=node.node_id,
                delay_after_predecessor_start_us=delay_us,
                delay_after_predecessor_first_token_us=first_token_delay_us,
            )
        ]


__all__ = [
    "apply_start_anchors",
    "build_interval_edges",
    "compute_ranks",
]
