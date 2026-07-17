# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lowering: expanded ``dag_jsonl`` trees -> unified-segment-store ParsedGraph.

The byte-parity core of the ``dag_jsonl`` graph adapter. Each
:class:`~aiperf.dataset.graph.adapters.dag_jsonl.tree.DagTree` becomes one
per-trace :class:`GraphRecord` (mirroring the native lowering's per-trace
graphs) against ONE shared content-addressed :class:`SegmentPool`:

* Every authored message is interned VERBATIM via ``SegmentPool.add_raw_message``
  (key order and extra keys preserved), parent-chained across the whole per-node
  walk so shared conversation prefixes dedup into a real KV-cache prefix.
* A node with message-context lineage carries an ordered assembly program
  (``metadata["trie"]["assembly"]``) interleaving interned static segments with
  ``{"s": {"src": <producer>}}`` live-reply slots -- the worker splices each slot
  as ``{"role": "assistant", "content": <captured text>}``, byte-matching the
  legacy captured-assistant message for text responses. Slot producers are
  stamped ``capture: true``; lineage-free nodes carry no assembly key (pure
  static -> bytes path at dispatch).
* Model / stream / token cap / tools ride the NATIVE node fields (``model`` /
  ``streaming`` / ``max_tokens`` / ``raw_tools``, Turn naming);
  ``extra_body`` carries the merged vendor keys
  (``**endpoint_extra, **turn extra``). The run's
  ``--extra-inputs`` (``endpoint_extra``) are folded at parse with the
  legacy precedence -- the later turn-``extra`` update wins on overlap -- and
  every node is stamped
  ``metadata["dispatch"]["endpoint_extra_applied"] = True`` so the worker skips
  its own ``endpoint.extra`` re-merge (parse-time folding is authoritative even
  when the run has no extras). Legacy-vs-graph wire parity is proven
  order-insensitively (canonical sorted-keys body comparison), so the legacy
  body KEY ORDER is not reproduced -- keys and values are.
* Firing topology: one completion-anchored ``StaticEdge`` per predecessor
  (authored turn delay in microseconds), ``START`` edges for predecessor-less
  nodes, ``END`` edges from sink nodes, and AND-fan-in ``ChannelRequirement``
  gates for SPAWN_JOIN inputs plus every spliced lineage producer (credits are
  minted only after the captures they splice have landed).
"""

from __future__ import annotations

from typing import Any

from aiperf.dataset.graph.adapters.dag_jsonl.tree import DagNodeSpec, DagTree
from aiperf.dataset.graph.models import (
    END_NODE_ID,
    START_NODE_ID,
    ChannelRequirement,
    ChannelSpec,
    GraphRecord,
    LlmNode,
    ParsedGraph,
    ProvenanceSpec,
    StaticEdge,
    TraceRecord,
)
from aiperf.dataset.graph.segment_ir.envelope import stamp_prompt_segment_ids
from aiperf.dataset.graph.segment_ir.pool import SegmentPool

__all__ = ["lower_dag_trees"]

_PROVENANCE = ProvenanceSpec(source="dag_jsonl", tool="aiperf-dag-jsonl-lowering/1")


def lower_dag_trees(
    trees: list[DagTree],
    *,
    default_model: str | None,
    run_streaming: bool,
    endpoint_extra: list[tuple[str, Any]] | None = None,
) -> ParsedGraph:
    """Lower expanded dag trees onto the unified interned segment store.

    One :class:`GraphRecord` per tree in ``parsed.graphs[trace_id]`` with a
    matching ``TraceRecord(id=trace_id, graph_ref=trace_id)``; ``parsed.graph``
    aliases the first tree's record. The lowering is a pure function of its
    input (content-addressed ids, no RNG, no clock), so dataset-plane and
    timing-plane re-parses stamp identical graphs. ``endpoint_extra`` (the run's
    ``--extra-inputs`` pairs) is folded into every node's ``extra_body``
    at the legacy precedence (see :func:`_lower_node`).
    """
    if not trees:
        raise NotImplementedError(
            "dag_jsonl workload: the file expands to zero root trees; "
            "there is nothing to lower"
        )
    pool = SegmentPool()
    graphs: dict[str, GraphRecord] = {}
    traces: list[TraceRecord] = []
    for tree in trees:
        graphs[tree.trace_id] = _lower_tree(
            pool,
            tree,
            default_model=default_model,
            run_streaming=run_streaming,
            endpoint_extra=endpoint_extra,
        )
        traces.append(TraceRecord(id=tree.trace_id, graph_ref=tree.trace_id))
    return ParsedGraph(
        graph=graphs[traces[0].id],
        graphs=graphs,
        traces=traces,
        segment_pool=pool,
    )


def _lower_tree(
    pool: SegmentPool,
    tree: DagTree,
    *,
    default_model: str | None,
    run_streaming: bool,
    endpoint_extra: list[tuple[str, Any]] | None,
) -> GraphRecord:
    capture_ids = {aid for spec in tree.nodes.values() for aid in spec.lineage}
    nodes: dict[str, LlmNode] = {}
    edges: list[StaticEdge] = []
    for spec in tree.nodes.values():
        nodes[spec.node_id] = _lower_node(
            pool,
            spec,
            tree,
            capture_ids,
            default_model=default_model,
            run_streaming=run_streaming,
            endpoint_extra=endpoint_extra,
        )
        if spec.predecessors:
            edges.extend(
                StaticEdge(
                    source=pred_id,
                    target=spec.node_id,
                    delay_after_predecessor_us=float(delay_us) if delay_us else None,
                )
                for pred_id, delay_us in spec.predecessors
            )
        else:
            edges.append(StaticEdge(source=START_NODE_ID, target=spec.node_id))
    sources = {edge.source for edge in edges}
    edges.extend(
        StaticEdge(source=nid, target=END_NODE_ID)
        for nid in nodes
        if nid not in sources
    )
    # Every LlmNode writes its response into a per-node ``{node_id}_out``
    # channel; the executor's channel store rejects writes to undeclared
    # channels, and successors read them only to enforce the AND-fan-in WAIT
    # (prompt content comes from the segment pool / dynamic slots, never the
    # channel value), so the default single-producer overwrite spec suffices.
    state = {f"{nid}_out": ChannelSpec() for nid in nodes}
    return GraphRecord(
        version="2.0",
        provenance=_PROVENANCE,
        state=state,
        nodes=nodes,
        edges=edges,
    )


def _lower_node(
    pool: SegmentPool,
    spec: DagNodeSpec,
    tree: DagTree,
    capture_ids: set[str],
    *,
    default_model: str | None,
    run_streaming: bool,
    endpoint_extra: list[tuple[str, Any]] | None,
) -> LlmNode:
    # The full conversation prefix is interned at build time: each lineage
    # ancestor contributes its verbatim authored messages plus one live-reply
    # slot, then the node's own messages follow. ``parent`` threads through the
    # WHOLE walk (slots intern nothing) so shared prefixes dedup across nodes.
    assembly: list[dict[str, Any]] = []
    parent: str | None = None
    for ancestor_id in spec.lineage:
        for message in tree.nodes[ancestor_id].turn.raw_messages:
            parent = pool.add_raw_message(message=message, parent_id=parent)
            assembly.append({"seg": parent})
        assembly.append({"s": {"src": ancestor_id}})
    for message in spec.turn.raw_messages:
        parent = pool.add_raw_message(message=message, parent_id=parent)
        assembly.append({"seg": parent})

    # Model / stream / token cap / tools ride the NATIVE node fields (Turn
    # naming); ``extra_body`` carries only the merged vendor keys.
    # Legacy merge: endpoint extras (--extra-inputs) update first, the turn's
    # own extras update last -- the turn value wins on an overlapping key.
    extra_body: dict[str, Any] = {}
    extra_body.update(endpoint_extra or [])
    extra_body.update(spec.turn.extra_body or {})

    # SPAWN_JOIN gates plus one completion gate per spliced lineage producer
    # (disjoint sets by construction: join leaves are spawned children, never
    # in the joining node's own message-context lineage).
    inputs = [
        ChannelRequirement(channel=f"{join_id}_out", count=1)
        for join_id in spec.join_inputs
    ]
    inputs.extend(
        ChannelRequirement(channel=f"{ancestor_id}_out", count=1)
        for ancestor_id in spec.lineage
    )

    node = LlmNode(
        prompt=[],
        output=f"{spec.node_id}_out",
        streaming=run_streaming,
        model=spec.turn.model or default_model,
        max_tokens=spec.turn.max_tokens,
        raw_tools=_effective_tools(spec, tree),
        extra_body=extra_body,
        inputs=inputs,
        arrival_offset_us=0,
        # Unconditional: parse-time folding owns the run's --extra-inputs even
        # when there are none, so the worker's endpoint.extra re-merge (which
        # would clobber turn-extra values) is always skipped for dag nodes.
        # The "dag" stamp carries the instance's legacy record identity; the
        # timing plane's re-parse reads it to stamp agent_depth /
        # parent_correlation_id on every credit (parse-plane metadata only --
        # the worker's store envelope never carries it).
        metadata={
            "dispatch": {"endpoint_extra_applied": True},
            "dag": {
                "agent_depth": spec.agent_depth,
                "parent_node": spec.parent_node_id,
            },
        },
    )
    segment_ids = [token["seg"] for token in assembly if "seg" in token]
    extra: dict[str, Any] = {}
    if spec.lineage:
        extra["assembly"] = assembly
    if spec.node_id in capture_ids:
        extra["capture"] = True
    return stamp_prompt_segment_ids(node, segment_ids, extra=extra or None)


def _effective_tools(spec: DagNodeSpec, tree: DagTree) -> list[dict[str, Any]] | None:
    """Own turn's ``raw_tools``, else the nearest lineage ancestor's (legacy
    parity: ``raw_tools`` is the lone per-turn field that walks history)."""
    if spec.turn.raw_tools is not None:
        return spec.turn.raw_tools
    for ancestor_id in reversed(spec.lineage):
        tools = tree.nodes[ancestor_id].turn.raw_tools
        if tools is not None:
            return tools
    return None
