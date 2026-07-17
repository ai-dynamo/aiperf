# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""t* frontier chop for a segment-trie ``ParsedGraph``.

Format-agnostic: operates purely on ``ParsedGraph`` topology, ``StaticEdge``s,
per-node ``arrival_offset_us``, and AND-fan-in ``inputs`` -- no recorded-trace
(weka/dynamo) types. Any adapter that emits the segment-trie IR (a graph of
``LlmNode`` + ``StaticEdge`` with ``metadata["trie"]``) can be snapshotted here.
Also hosts the extended-warmup handoff chop (``chop_trie_at_frontier``).
"""

from __future__ import annotations

from typing import Any

import msgspec

from aiperf.dataset.graph.models import START_NODE_ID, ParsedGraph, StaticEdge


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
    edges: list[Any], survivors: dict[str, Any], t_star_us: int
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
        if not isinstance(edge, StaticEdge):
            continue
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


def _chop_node_inputs(node: Any, survivor_out_channels: set[str]) -> Any:
    """Drop a surviving node's ``inputs`` requirements on dropped predecessors.

    Non-``LlmNode`` (or input-free) nodes pass through untouched. An ``inputs``
    list whose surviving subset equals the original is returned unchanged so the
    common no-chop-of-this-node case avoids a struct rebuild.
    """
    inputs = getattr(node, "inputs", None)
    if not inputs:
        return node
    kept = [req for req in inputs if req.channel in survivor_out_channels]
    if len(kept) == len(inputs):
        return node
    return msgspec.structs.replace(node, inputs=kept)


def chop_trie_at_frontier(
    graph: ParsedGraph,
    *,
    t_star_us: float,
    executed: frozenset[str],
    return_wall_us: dict[str, float],
    drain_end_wall_us: float,
    residual_cap_us: float | None = None,
) -> ParsedGraph:
    """Chop a segment-trie ``ParsedGraph`` to its extended-warmup handoff frontier.

    The extended (cache-pressure) warmup replays the post-``t*`` remainder with
    zero idle delay; at drain, PROFILING must resume each chain at its first
    NOT-yet-executed node rather than re-firing from ``t*``. The chop:

    * Drops every node with ``arrival_offset_us < t_star_us`` (pre-``t*``
      history, primed by the boundary warmup) AND every node in ``executed``
      (dispatched-and-returned during warmup/pressure -- the server holds
      their KV).
    * Keeps inter-survivor edges verbatim (recorded pacing resumes past the
      frontier).
    * Re-roots each surviving node that lost ALL its real predecessors from
      ``START`` with ``min_start_delay_us`` set to its RESIDUAL delay: for each
      dropped predecessor edge, ``max(0, recorded_delay - max(0,
      drain_end_wall_us - return_wall_us[pred]))`` -- the recorded gap minus
      wall time already spent waiting for the drain;
      AND-fan-in takes the max across
      dropped predecessors, and the result is clamped to ``residual_cap_us``
      when set (graph edge delays
      are not bounded by the build-plane idle-gap warp when another stream's
      active interval covers the gap, so the cap is load-bearing). The
      recorded base uses ONLY end-anchored quantities
      (``delay_after_predecessor_us``, edge ``min_start_delay_us``): the
      ledger wall is the predecessor's RETURN, so a dispatch-anchored delay
      (``delay_after_predecessor_start_us`` / first-token) debited from a
      return-anchored elapsed would over-delay by the pred's live service
      time -- start-anchored edges contribute 0 (burst).
      A dropped predecessor with no recorded wall contributes 0
      (fire immediately -- consistent with compressed pressure pacing, where
      recorded leads are considered consumed).
    * Rescopes surviving nodes' AND-fan-in ``inputs`` exactly like
      :func:`chop_trie_at_tstar` (a requirement on a dropped predecessor's
      channel would deadlock ``await_inputs``). A survivor that KEEPS a
      surviving-pred edge but LOST a residual-carrying binding edge to the chop
      is NOT re-rooted; instead that dropped edge's residual is FOLDED into the
      node's ``min_start_delay_us`` (max-combined with any existing node value).
      Under ``absolute_start_offsets=True`` the executor anchors that node-level
      gate to the instance run-start -- the same anchor the re-root residuals
      use -- and max-combines it with the surviving edge gates
      (``_compute_firing_gate_us``), so the dropped binding gap survives instead
      of vanishing behind a zero-delay join edge.

    ``return_wall_us`` and ``drain_end_wall_us`` share one monotonic clock
    (the warmup strategy's ledger); only differences are meaningful.
    """
    old_graph = graph.graph
    survivors = {
        nid: node
        for nid, node in old_graph.nodes.items()
        if (node.arrival_offset_us or 0) >= t_star_us and nid not in executed
    }

    new_edges, kept_pred_residuals = _frontier_edges(
        old_graph.edges,
        survivors,
        return_wall_us=return_wall_us,
        drain_end_wall_us=drain_end_wall_us,
        residual_cap_us=residual_cap_us,
    )

    survivor_out_channels = {f"{nid}_out" for nid in survivors}
    rescoped: dict[str, Any] = {}
    for nid, node in survivors.items():
        node = _chop_node_inputs(node, survivor_out_channels)
        residual = kept_pred_residuals.get(nid)
        if residual is not None:
            # A dropped binding edge's residual must not vanish just because a
            # zero-delay join edge from a SURVIVING pred remains: fold it into
            # the node-level gate. Under the strategy's absolute_start_offsets
            # this anchors to the instance run-start -- the same anchor the
            # re-root residuals use -- and the executor max-combines it with the
            # surviving edge gates (models.py min_start_delay_us contract).
            node = msgspec.structs.replace(
                node,
                min_start_delay_us=max(node.min_start_delay_us or 0.0, residual),
            )
        rescoped[nid] = node

    new_graph = msgspec.structs.replace(old_graph, nodes=rescoped, edges=new_edges)
    return msgspec.structs.replace(graph, graph=new_graph)


def _frontier_edges(
    edges: list[Any],
    survivors: dict[str, Any],
    *,
    return_wall_us: dict[str, float],
    drain_end_wall_us: float,
    residual_cap_us: float | None = None,
) -> tuple[list[StaticEdge], dict[str, float]]:
    """Edge set for the handoff chop: keep inter-survivor, re-root at residuals.

    Mirrors :func:`_chop_edges` structurally; the ONLY divergence is the
    re-root offset: the t* chop rebases to the recorded absolute offset
    (``arrival - t*``) because nothing was replayed yet, while the frontier
    chop uses the residual-of-recorded-gap because pressure already consumed
    the recorded leads.

    Returns ``(edges, kept_pred_residuals)``. ``kept_pred_residuals`` maps a
    survivor id to the leftover residual of a dropped binding edge whose target
    ALSO retains a surviving-pred edge (so it is not re-rooted). Those residuals
    would otherwise be discarded; the caller folds them node-level so a dropped
    binding gap is not lost just because a zero-delay join edge from a surviving
    pred remains.
    """
    kept_edges: list[StaticEdge] = []
    has_surviving_pred: set[str] = set()
    residual_by_target: dict[str, float] = {}
    for edge in edges:
        if not isinstance(edge, StaticEdge):
            continue
        src, tgt = edge.source, edge.target
        if tgt not in survivors:
            continue
        if src == START_NODE_ID or src in survivors:
            kept_edges.append(edge)
            if src != START_NODE_ID:
                has_surviving_pred.add(tgt)
            continue
        # src was dropped (executed / pre-t* history): fold its recorded gap,
        # minus the wall time already waited since its return, into the
        # target's re-root offset. The executor gate takes the max of a node's
        # incoming delays, so AND-fan-in residuals max-combine here too.
        # END-anchored quantities only: the ledger wall is src's RETURN, so a
        # dispatch-anchored delay (start / first-token) debited from a
        # return-anchored elapsed would over-delay by src's live service time
        # -- those edges burst (residual 0).
        recorded_us = max(
            edge.delay_after_predecessor_us or 0.0,
            edge.min_start_delay_us or 0.0,
        )
        wall = return_wall_us.get(src)
        residual = (
            max(0.0, recorded_us - max(0.0, drain_end_wall_us - wall))
            if wall is not None
            else 0.0
        )
        if residual_cap_us is not None:
            residual = min(residual, residual_cap_us)
        residual_by_target[tgt] = max(residual_by_target.get(tgt, 0.0), residual)

    new_edges = [
        e
        for e in kept_edges
        if not (e.source == START_NODE_ID and e.target not in has_surviving_pred)
    ]
    kept_pred_residuals: dict[str, float] = {}
    for nid in survivors:
        if nid not in has_surviving_pred:
            new_edges.append(
                StaticEdge(
                    source=START_NODE_ID,
                    target=nid,
                    min_start_delay_us=residual_by_target.get(nid, 0.0),
                )
            )
        else:
            residual = residual_by_target.get(nid, 0.0)
            if residual > 0.0:
                kept_pred_residuals[nid] = residual
    return new_edges, kept_pred_residuals


__all__ = ["chop_trie_at_tstar", "chop_trie_at_frontier"]
