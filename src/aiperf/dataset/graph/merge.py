# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merge per-source ParsedGraphs into one multi-graph workload.

Used by the dynamo per-session-tree merge and the streaming store build's
structural-sidecar merge.
"""

from __future__ import annotations

from collections.abc import Iterable

import msgspec

from aiperf.dataset.graph.models import GraphRecord, ParsedGraph, resolve_trace_graph
from aiperf.dataset.graph.segment_trie.pool import Segment, SegmentPool


class GraphMergeError(ValueError):
    """Raised when per-source ParsedGraphs cannot be merged into one workload."""


def merge_parsed_graphs(per_source: Iterable[ParsedGraph]) -> ParsedGraph:
    """Merge per-source ``ParsedGraph`` outputs into one MULTI-GRAPH workload.

    Trace sources are HETEROGENEOUS — every trace is an independent
    conversation whose topology (turn count, subagent fan-out) differs, so
    each trace keeps its OWN ``GraphRecord`` under ``ParsedGraph.graphs``
    keyed by its trace id, and that trace's ``graph_ref`` selects it. A
    single source may carry one trace (dynamo captures) or many
    (a multi-session dag_jsonl parse); either way each trace's graph is
    resolved from ITS source via ``resolve_trace_graph``, which also threads
    the per-trace graph through firing-plan / snapshot / conversation-source
    computation downstream.

    This function intentionally accepts any iterable and folds each
    ``ParsedGraph`` into the merged structures immediately. The HF path feeds it
    a pool-result generator, so the parent never holds the complete list of
    decoded per-trace graphs on top of the merged graph.
    """
    seen_trace_ids: set[str] = set()
    merged_traces = []
    merged_warmup_traces = []
    graphs: dict[str, GraphRecord] = {}
    merged_graph: GraphRecord | None = None
    # Segment trie: union every per-source pool's content-addressed
    # entries. Ids are content-addressed (blake2b over parent_id/role/tokens),
    # so the SAME id MUST carry identical content -- dedup is only correct under
    # that invariant. Stays None when no per-source graph carries a pool so the
    # merged graph carries none either.
    merged_segments: dict[str, Segment] | None = None

    for pg in per_source:
        if pg.segment_pool is not None:
            if merged_segments is None:
                merged_segments = {}
            # Fail loud on an id that maps to divergent content: two Segments
            # sharing an id but differing in value (Segment is a frozen
            # dataclass, so == is a value comparison) can only mean a
            # content-addressing / hash break, which would otherwise silently
            # keep whichever entry wins the union. This is a correctness
            # invariant, not a debug aid, so it is a real raise (not an assert).
            for sid, seg in pg.segment_pool.by_id.items():
                prior = merged_segments.get(sid)
                if prior is not None and prior != seg:
                    raise GraphMergeError(
                        f"segment id {sid!r} maps to divergent content across "
                        f"graph trace sources (content-addressing invariant "
                        f"broken): {prior!r} != {seg!r}"
                    )
                merged_segments[sid] = seg
        if merged_graph is None:
            # Frozen structs are immutable and never mutated below, so the
            # default single-graph slot can share the first source's graph directly.
            merged_graph = pg.graph

        # Key each trace's OWN topology by its trace id. A source is usually
        # one TraceRecord over one top-level graph (dynamo captures), but a
        # source may itself be multi-trace (a dag_jsonl file
        # with several independent root sessions streams ONE structural blob
        # carrying every trace); resolving through the trace's graph_ref keeps
        # each trace on its own graph instead of remapping all of them onto
        # the first tree's ``pg.graph``.
        for trace in pg.traces:
            if trace.id in seen_trace_ids:
                raise GraphMergeError(
                    f"duplicate trace id across graph trace sources: {trace.id!r}"
                )
            seen_trace_ids.add(trace.id)
            graphs[trace.id] = resolve_trace_graph(pg, trace)
            merged_traces.append(msgspec.structs.replace(trace, graph_ref=trace.id))
        for trace in pg.warmup_traces:
            if trace.id in seen_trace_ids:
                raise GraphMergeError(
                    f"duplicate warmup trace id across graph trace sources: {trace.id!r}"
                )
            seen_trace_ids.add(trace.id)
            graphs[trace.id] = resolve_trace_graph(pg, trace)
            merged_warmup_traces.append(
                msgspec.structs.replace(trace, graph_ref=trace.id)
            )

    if merged_graph is None or not merged_traces:
        raise GraphMergeError("graph merge produced zero ParsedGraph results")

    merged_traces.sort(key=lambda t: t.id)
    merged_warmup_traces.sort(key=lambda t: t.id)

    merged_pool = (
        SegmentPool(_by_id=merged_segments) if merged_segments is not None else None
    )

    return ParsedGraph(
        graph=merged_graph,
        graphs=graphs,
        traces=merged_traces,
        warmup_traces=merged_warmup_traces,
        segment_pool=merged_pool,
    )


__all__ = [
    "GraphMergeError",
    "merge_parsed_graphs",
]
