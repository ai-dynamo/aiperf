# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""t* frontier chop for a segment-trie ``ParsedGraph``.

Format-agnostic: operates purely on ``ParsedGraph`` topology, ``StaticEdge``s,
per-node ``arrival_offset_us``, and AND-fan-in ``inputs`` -- no recorded-trace
(weka/dynamo) types. Any adapter that emits the segment trie (a graph of
``LlmNode`` + ``StaticEdge`` with ``metadata["trie"]``) can be snapshotted here.
"""

from __future__ import annotations

import msgspec

from aiperf.dataset.graph.models import (
    START_NODE_ID,
    NodeUnion,
    ParsedGraph,
    StaticEdge,
)


def chop_trie_at_tstar(graph: ParsedGraph, t_star_us: int) -> ParsedGraph:
    """Chop a segment-trie ``ParsedGraph`` to its live frontier at ``t*``.

    The segment-trie graph is the trivial ``LlmNode`` + ``StaticEdge`` realization
    of one recorded trace; every node carries ``arrival_offset_us`` (its recorded
    ``t`` * 1e6). The snapshot chop:

    * Drops every node whose ``arrival_offset_us < t_star_us`` -- the pre-``t*``
      turns were already WARMED (primed), not PROFILED.
    * Re-roots each SURVIVING node that lost ALL its predecessors to the chop
      from ``START`` via a synthetic ``StaticEdge(START -> node)`` whose
      ``min_start_delay_us = arrival_offset_us - t_star_us`` -- the node's
      ABSOLUTE offset from the instance run-origin ``t*`` (the executor anchors
      this via ``absolute_start_offsets=True``). Inter-turn edges between two
      SURVIVING nodes are kept verbatim (whichever delay kind they carry --
      ``delay_after_predecessor_us`` end-to-start OR
      ``delay_after_predecessor_start_us`` dispatch-to-start -- is unchanged).
    * Leaves each surviving node's ``metadata["trie"]["prompt_segment_ids"]``
      UNCHANGED: the full pre-``t*`` prefix stays in the path so the worker still
      materializes the EXACT resume prompt (no truncation needed -- the dropped
      turns were dispatched during warmup and the server holds their KV).

    ``t_star_us <= 0`` => the graph is returned UNCHANGED (full t*=0 replay).
    """
    if t_star_us <= 0:
        return graph

    old_graph = graph.graph
    survivors = {
        nid: node
        for nid, node in old_graph.nodes.items()
        if (node.arrival_offset_us or 0) >= t_star_us
    }

    new_edges = _chop_edges(old_graph.edges, survivors, t_star_us)

    # Recompute each surviving node's AND-fan-in ``inputs`` against the chop: a
    # requirement on a DROPPED predecessor's ``{src}_out`` channel would deadlock
    # ``await_inputs`` (that channel is never written post-chop). Keep only
    # requirements whose source survives; a node re-rooted entirely from START
    # ends with empty ``inputs``.
    survivor_out_channels = {f"{nid}_out" for nid in survivors}
    rescoped = {
        nid: _chop_node_inputs(node, survivor_out_channels)
        for nid, node in survivors.items()
    }

    new_graph = msgspec.structs.replace(old_graph, nodes=rescoped, edges=new_edges)
    return msgspec.structs.replace(graph, graph=new_graph)


def _chop_edges(
    edges: list[StaticEdge], survivors: dict[str, NodeUnion], t_star_us: int
) -> list[StaticEdge]:
    """Recompute the chopped graph's edge set against the surviving frontier.

    An edge survives only when BOTH endpoints survive (or it roots an explicitly
    kept node at START). Each surviving node that lost ALL its predecessors to the
    chop is re-rooted from START at its t*-relative absolute offset, dropping any
    kept START edge for it (the builder rooted pre-t* roots at START with the
    recorded absolute offset; the frontier re-root replaces it with the
    t*-relative offset).
    """
    kept_edges: list[StaticEdge] = []
    has_surviving_pred: set[str] = set()
    for edge in edges:
        src, tgt = edge.source, edge.target
        if tgt not in survivors:
            continue
        if src == START_NODE_ID or src in survivors:
            kept_edges.append(edge)
            if src != START_NODE_ID:
                has_surviving_pred.add(tgt)

    new_edges = [
        e
        for e in kept_edges
        if not (e.source == START_NODE_ID and e.target not in has_surviving_pred)
    ]
    for nid, node in survivors.items():
        if nid not in has_surviving_pred:
            new_edges.append(
                StaticEdge(
                    source=START_NODE_ID,
                    target=nid,
                    min_start_delay_us=float((node.arrival_offset_us or 0) - t_star_us),
                )
            )
    return new_edges


def _chop_node_inputs(node: NodeUnion, survivor_out_channels: set[str]) -> NodeUnion:
    """Drop a surviving node's ``inputs`` requirements on dropped predecessors.

    Input-free nodes pass through untouched. An ``inputs`` list whose surviving
    subset equals the original is returned unchanged so the common
    no-chop-of-this-node case avoids a struct rebuild.
    """
    inputs = node.inputs
    if not inputs:
        return node
    kept = [req for req in inputs if req.channel in survivor_out_channels]
    if len(kept) == len(inputs):
        return node
    return msgspec.structs.replace(node, inputs=kept)


__all__ = ["chop_trie_at_tstar"]
