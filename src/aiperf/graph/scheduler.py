# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph-derived adjacency helper for the async-dataflow executor.

A pure adjacency lookup over the parsed graph's static edges; holds no mutable
per-trace state. Per-trace firing dedup lives on
`_TraceContext.scheduled_node_ids`.
"""

from __future__ import annotations

from collections import defaultdict

import msgspec

from aiperf.dataset.graph.models import (
    END_NODE_ID,
    START_NODE_ID,
    GraphRecord,
    StaticEdge,
)


class Scheduler:
    """Graph-derived adjacency helper for the async-dataflow executor.

    All state is derived from the parsed graph at construction time and is
    immutable across traces (no mutable per-trace fields). Per-trace state
    lives on `_TraceContext.scheduled_node_ids`; this class is shared across
    every trace in a phase.
    """

    __slots__ = (
        "_entry",
        "_start_anchored_succ",
        "_static_pred_edges",
        "_static_succ",
    )

    def __init__(self, graph: GraphRecord) -> None:
        _reject_start_anchored_start_edge(graph.edges)
        static_succ: dict[str, list[str]] = defaultdict(list)
        start_anchored_succ: dict[str, list[str]] = defaultdict(list)
        static_pred_edges: dict[str, list[StaticEdge]] = defaultdict(list)

        for edge in graph.edges:
            if edge.delay_after_predecessor_start_us is not None:
                start_anchored_succ[edge.source].append(edge.target)
            else:
                static_succ[edge.source].append(edge.target)
            static_pred_edges[edge.target].append(edge)

        _reject_unanchored_first_token_edge(static_pred_edges)
        _reject_mixed_anchor_fan_in(static_pred_edges)

        entry: list[str] = []
        seen: set[str] = set()
        for target in static_succ.get(START_NODE_ID, []):
            if target == END_NODE_ID or target in seen:
                continue
            seen.add(target)
            entry.append(target)

        self._static_succ: dict[str, list[str]] = dict(static_succ)
        self._start_anchored_succ: dict[str, list[str]] = dict(start_anchored_succ)
        self._static_pred_edges: dict[str, list[StaticEdge]] = dict(static_pred_edges)
        self._entry: list[str] = entry

    def entry_nodes(self) -> list[str]:
        """Return node ids that should fire at trace start (successors of START).

        END is suppressed.
        """
        return list(self._entry)

    def successors_after(self, node_id: str) -> list[str]:
        """Return successor node ids that should fire after `node_id` completes.

        Static successors via StaticEdge. Start-anchored StaticEdges
        (`delay_after_predecessor_start_us`) are EXCLUDED here; they are
        scheduled at `node_id`'s DISPATCH instead of its completion (see
        `start_anchored_successors`). END is suppressed.
        """
        return [t for t in self._static_succ.get(node_id, []) if t != END_NODE_ID]

    def incoming_static_edges(self, node_id: str) -> list[StaticEdge]:
        """Return StaticEdge objects targeting `node_id`.

        Used by edge-gate computation for `min_start_delay_us` /
        `delay_after_predecessor_us` / `delay_after_predecessor_start_us`.
        """
        return list(self._static_pred_edges.get(node_id, []))

    def start_anchored_successors(self, node_id: str) -> list[str]:
        """Successors wired via start-anchored edges. The executor schedules
        these at `node_id`'s DISPATCH (firing-gate clear), not its completion;
        they are deliberately absent from `successors_after` so a child that
        finishes before its still-running parent is not re-scheduled into the
        cycle guard."""
        return [
            t for t in self._start_anchored_succ.get(node_id, []) if t != END_NODE_ID
        ]


def _reject_start_anchored_start_edge(edges: list[StaticEdge]) -> None:
    """Reject an edge whose start anchor refers to the virtual START node."""
    for edge in edges:
        if (
            edge.source == START_NODE_ID
            and edge.delay_after_predecessor_start_us is not None
        ):
            raise NotImplementedError(
                f"edge {START_NODE_ID!r} -> {edge.target!r}: "
                "delay_after_predecessor_start_us is unsupported on the "
                "virtual START node because START is never dispatched. Use "
                "min_start_delay_us for a trace-entry delay."
            )


def _reject_unanchored_first_token_edge(
    static_pred_edges: dict[str, list[StaticEdge]],
) -> None:
    """Reject a first-token-anchored in-edge that carries no start anchor.

    ``TraceExecutor._apply_firing_delay`` blocks on
    ``ctx.first_token_event(edge.source).wait()`` for every incoming edge
    carrying ``delay_after_predecessor_first_token_us``, and that wait has no
    self-imposed bound (``AIPERF_GRAPH_EXECUTOR_WATCHDOG_TIMEOUT`` is unset by
    default). It is safe today only because the edge ALSO carries
    ``delay_after_predecessor_start_us``, which makes it start-anchored, which
    makes :func:`_reject_mixed_anchor_fan_in` guarantee it is its target's ONLY
    in-edge -- so the target is scheduled solely by
    ``start_anchored_successors(source)`` at the source's DISPATCH. The source
    is therefore already running, and ``_finalize_node``'s ``finally`` will set
    the latch on every exit path.

    Strip the start anchor and that whole chain lapses: the target becomes
    reachable from an unrelated predecessor while the first-token source may
    never run, and the wait deadlocks with no diagnostic. The sole producer
    (``interval_order.apply_start_anchors``) always co-emits both, so this
    gates a shape no shipped lowering can currently emit -- it exists to make a
    future lowering that breaks the pairing fail loudly at graph load instead
    of hanging a benchmark.

    A completion anchor (``delay_after_predecessor_us``) does NOT satisfy this:
    it leaves the edge classified as a completion edge, so the sole-in-edge
    guarantee still never applies.
    """
    for target, edges in static_pred_edges.items():
        for edge in edges:
            if edge.delay_after_predecessor_first_token_us is None:
                continue
            if edge.delay_after_predecessor_start_us is not None:
                continue
            raise NotImplementedError(
                f"node {target!r}: first-token-anchored edge "
                f"{edge.source!r} -> {target!r} carries "
                "delay_after_predecessor_first_token_us without "
                "delay_after_predecessor_start_us, which is unsupported. The "
                "executor waits on the source's first-token latch before "
                "computing this node's firing gate, and that wait is only "
                "guaranteed to be satisfiable because a start anchor forces "
                "the edge to be its target's ONLY in-edge (so the target is "
                "scheduled at the source's dispatch, with the source already "
                "running). Without the start anchor the target can be "
                "scheduled by an unrelated predecessor while the source never "
                "runs, and the wait deadlocks. Emit "
                "delay_after_predecessor_start_us on the same edge."
            )


def _reject_mixed_anchor_fan_in(
    static_pred_edges: dict[str, list[StaticEdge]],
) -> None:
    """Reject a start-anchored in-edge that is not its target's ONLY in-edge.

    The runtime half-supports ANY fan-in involving a start anchor: the start
    anchor schedules the target at its anchor parent's DISPATCH, so the target
    fires without waiting for its other predecessors (a completion
    predecessor's recorded ordering is silently ignored; a second start-anchor
    parent that has not yet dispatched is silently dropped by the firing gate)
    and, when the other predecessor later finishes/dispatches, it re-schedules
    the DONE target into the cycle guard (spurious "cycle detected"). No
    shipped lowering emits either shape (`apply_start_anchors` replaces a
    start-anchored node's WHOLE in-edge set with exactly one edge), so it is
    gated loudly at graph load instead.
    """
    for target, edges in static_pred_edges.items():
        start_anchored = [
            e for e in edges if e.delay_after_predecessor_start_us is not None
        ]
        if not start_anchored or len(edges) == 1:
            continue
        completion = [e for e in edges if e.delay_after_predecessor_start_us is None]
        if completion:
            shape = "mixed-anchor fan-in"
            detail = (
                f"start-anchored edge {start_anchored[0].source!r} -> {target!r} "
                f"(delay_after_predecessor_start_us) and completion edge "
                f"{completion[0].source!r} -> {target!r} arrive at the same node"
            )
        else:
            shape = "multi-start-anchored fan-in"
            detail = (
                f"start-anchored edges {start_anchored[0].source!r} -> {target!r} "
                f"and {start_anchored[1].source!r} -> {target!r} "
                f"(delay_after_predecessor_start_us) arrive at the same node"
            )
        raise NotImplementedError(
            f"node {target!r}: {shape} is unsupported: {detail}. "
            "The runtime schedules a start-anchored target at its anchor "
            "parent's DISPATCH, so the node would fire without waiting for "
            "its other predecessor and then be re-scheduled into the cycle "
            "guard when that predecessor finishes or dispatches. A "
            "start-anchored in-edge must be its target's ONLY in-edge."
        )


def collapse_leading_start_offsets(graph: GraphRecord) -> GraphRecord:
    """Zero the leading phase-start offsets (`--burst-phase-starts` collapse).

    The anchor carrier for a firing's t*-relative leading offset is the
    `min_start_delay_us` on its START in-edge: the trie build roots gap-started
    chains at START with the warped arrival offset
    (`interval_order.build_interval_edges`) and the t* snapshot chop re-roots
    every surviving frontier node at START with its `arrival - t*` offset
    (`snapshot_chop._chop_edges`). The executor gates on that edge field
    (`_compute_firing_gate_us`). Burst collapses ONLY those leading offsets --
    every node fires the instant its inputs are ready -- while non-START edges
    keep their recorded inter-turn pacing untouched. Node-level
    `min_start_delay_us` is collapsed too, but ONLY on a node with no non-START
    static in-edge (same leading-anchor semantics): a
    node-level delay on a node that HAS a real predecessor is mid-graph pacing,
    not a leading offset, and must survive burst.

    Pure function: returns a rebuilt `GraphRecord` (msgspec replace), never
    mutates. Identity-preserving for untouched nodes/edges.
    """
    has_real_pred = {
        edge.target for edge in graph.edges if edge.source != START_NODE_ID
    }
    new_edges = [
        msgspec.structs.replace(edge, min_start_delay_us=0.0)
        if edge.source == START_NODE_ID and edge.min_start_delay_us
        else edge
        for edge in graph.edges
    ]
    new_nodes = {
        nid: (
            msgspec.structs.replace(node, min_start_delay_us=0.0)
            if node.min_start_delay_us and nid not in has_real_pred
            else node
        )
        for nid, node in graph.nodes.items()
    }
    return msgspec.structs.replace(graph, nodes=new_nodes, edges=new_edges)
