# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write/locate the content-free structural ``ParsedGraph`` sidecar.

The DatasetManager serializes the structural graph into the per-benchmark
sidecar directory (``aiperf_graph_meta_<id>/``; the unified segment store
lives in its own ``aiperf_graph_segments_<id>/``) on EVERY graph build route,
and advertises the written path on the graph-typed
``DatasetConfiguredNotification.client_metadata``. The sidecar is mandatory:
the TimingManager only ingests this artifact, from the broadcast path (it
never re-parses the workload and never re-derives the path from env
conventions), so a build that cannot land a catalog-consistent sidecar fails
the run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec

from aiperf.dataset.graph.codecs import (
    GRAPH_META_SIDECAR_FILENAME,
    encode_graph_meta_sidecar,
)
from aiperf.dataset.graph.graph_path_catalog import build_graph_path_catalog
from aiperf.dataset.graph.models import (
    GraphRecord,
    ParsedGraph,
    TraceRecord,
)

GRAPH_META_DIR_PREFIX = "aiperf_graph_meta_"

__all__ = [
    "catalogs_match",
    "sidecar_matches_index",
    "sidecar_path_for",
    "strip_replay_text",
    "write_graph_meta_sidecar",
]


def _strip_graph_node_content(graph: GraphRecord) -> GraphRecord:
    """Return a copy of ``graph`` with every ``LlmNode``'s bulk content emptied.

    Two dominant, schedule-plane-irrelevant fields are cleared:

    * ``LlmNode.prompt`` -- the resolved prompt content, inline megabytes/trace;
    * ``metadata["trie"]`` contents -> ``{}`` -- the ``prompt_segment_ids`` and
      any dynamic-slot ``assembly``/``capture`` keys (the worker reads all of
      those from the STORE, not the sidecar). The ``"trie"`` KEY is KEPT (empty)
      so ``agent_graph_replay._is_trie_graph`` still routes the trie frontier-chop.

    All other metadata, the native dispatch fields (model / max_tokens /
    raw_tools / extra_headers / theoretical prefix-cache counts),
    node ids/types, edges, and ``arrival_offset_us`` are
    preserved verbatim -- everything the schedule plane and the prefix-cache metric
    actually consume.
    """

    def _strip(node):
        metadata = node.metadata or {}
        new_metadata = {**metadata, "trie": {}} if "trie" in metadata else metadata
        return msgspec.structs.replace(node, prompt=[], metadata=new_metadata)

    return msgspec.structs.replace(
        graph, nodes={nid: _strip(n) for nid, n in graph.nodes.items()}
    )


def strip_replay_text(graph: ParsedGraph) -> ParsedGraph:
    """Return a content-free copy for the graph_meta sidecar.

    Every trace's ``replay_outputs`` is cleared. For the segment trie
    (``segment_pool is not None``) the pool content is emptied (the pool is kept
    NON-None so the loaded graph still reads as a segment-store-backed graph -- e.g.
    ``TraceExecutor``'s dispatch-failure sentinel writes gate on
    ``segment_pool is not None``) and each
    ``LlmNode``'s bulk content is stripped via :func:`_strip_graph_node_content`
    across the top graph and every per-trace graph: the
    inline ``prompt`` and the ``metadata["trie"]`` contents (``prompt_segment_ids``
    and any ``assembly``/``capture`` slot keys) -> ``{}`` (the ``"trie"`` marker
    key stays). Non-trie graphs carry
    no pool (``None``) and keep their small channel-ref prompts + metadata
    untouched. The native dispatch fields (model / max_tokens / raw_tools /
    extra_headers / theoretical prefix-cache counts), node ids/types, edges,
    and ``arrival_offset_us`` are preserved verbatim -- everything the TimingManager
    and the prefix-cache metric actually consume; the worker reads prompt content +
    ``prompt_segment_ids`` from the STORE, never the sidecar.
    """
    from aiperf.dataset.graph.segment_trie.pool import SegmentPool

    stripped: list[TraceRecord] = [
        msgspec.structs.replace(t, replay_outputs={}) for t in graph.traces
    ]
    if graph.segment_pool is None:
        return msgspec.structs.replace(graph, traces=stripped)

    return msgspec.structs.replace(
        graph,
        traces=stripped,
        segment_pool=SegmentPool(),
        graph=_strip_graph_node_content(graph.graph),
        graphs={k: _strip_graph_node_content(g) for k, g in graph.graphs.items()},
    )


def sidecar_path_for(base_path: Path, benchmark_id: str) -> Path:
    """Canonical sidecar path inside the per-benchmark sidecar dir."""
    return (
        Path(base_path)
        / f"{GRAPH_META_DIR_PREFIX}{benchmark_id}"
        / GRAPH_META_SIDECAR_FILENAME
    )


def write_graph_meta_sidecar(
    graph: ParsedGraph,
    *,
    base_path: Path,
    benchmark_id: str,
    source_fingerprint: dict[str, Any],
    schema_version: int,
) -> Path:
    """Encode and write the structural sidecar; return the written path."""
    out = sidecar_path_for(base_path, benchmark_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(
        encode_graph_meta_sidecar(
            strip_replay_text(graph),
            source_fingerprint=source_fingerprint,
            schema_version=schema_version,
        )
    )
    return out


def catalogs_match(
    graph: ParsedGraph, other_catalog: dict[str, dict[str, int]]
) -> bool:
    """True iff ``graph``'s build catalog equals ``other_catalog`` exactly."""
    return build_graph_path_catalog(graph) == other_catalog


def sidecar_matches_index(
    graph: ParsedGraph, index_offsets: dict[str, dict[int, Any]]
) -> bool:
    """Best-effort load-time check: per-trace sidecar ordinals == store index ordinals.

    The store index inner keys are node ordinals; compare the per-trace
    ordinal SETS. A mismatch means the sidecar's topology
    diverged from the stored envelopes -> the caller
    (``TimingManager._load_graph_sidecar``) raises ``InvalidStateError`` (the
    sidecar is mandatory; no re-parse fallback exists).
    """
    catalog = build_graph_path_catalog(graph)
    for trace_id, node_ordinals in catalog.items():
        store_ordinals = set(index_offsets.get(trace_id, {}))
        if set(node_ordinals.values()) - store_ordinals:
            return False
    return True
