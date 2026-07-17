# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stable node-id -> node-ordinal catalog for the graph store.

Every live producer lowers to a flat graph of ``LlmNode``s, so the catalog key
is simply the bare node id. Ordinals are assigned by
``flat_trie_ordinals`` (the SAME helper the build plane uses), so
build-time manifest ordinals and dispatch-time catalog ordinals are
byte-identical by construction and stable across runs.
"""

from __future__ import annotations

import dataclasses

from aiperf.dataset.graph.models import (
    ParsedGraph,
    TraceRecord,
)


@dataclasses.dataclass(frozen=True, slots=True)
class CatalogContext:
    """Immutable per-build catalog state threaded into :func:`node_ordinal_for`.

    Built once per parse via :func:`build_catalog_context`; passed explicitly
    to the resolver so there is NO module-level mutable cache (which would
    corrupt across concurrent traces).
    """

    # ``{trace_id: {node_id: node_ordinal}}`` -- the addressing map.
    catalog: dict[str, dict[str, int]]


def build_catalog_context(parsed: ParsedGraph) -> CatalogContext:
    """Build the :class:`CatalogContext` (the per-trace ordinal catalog)."""
    return CatalogContext(catalog=build_graph_path_catalog(parsed))


def build_graph_path_catalog(parsed: ParsedGraph) -> dict[str, dict[str, int]]:
    """Build ``{trace_id: {node_id: node_ordinal}}`` for every trace."""
    return {trace.id: _catalog_for_trace(parsed, trace) for trace in parsed.traces}


def _catalog_for_trace(parsed: ParsedGraph, trace: TraceRecord) -> dict[str, int]:
    """Assign a dense ordinal to each LlmNode id in the shared trie order.

    The build-plane unified store is keyed by
    :func:`flat_trie_ordinals` -- the SHARED build/schedule ordinal
    scheme. The dispatch adapter must resolve a fired node to the EXACT ordinal
    the store was written at, so this returns the same helper's output
    verbatim -- byte-identical to the build plane by construction, never a
    re-derived ordering.
    """
    # Imported locally to avoid a build<->schedule module import cycle.
    from aiperf.dataset.graph.segment_ir.store_builder import (
        flat_trie_ordinals,
    )

    return flat_trie_ordinals(parsed, trace)


def node_ordinal_for(
    context: CatalogContext,
    trace_id: str,
    node_key: str,
) -> int | None:
    """Resolve a fired node's build-time ordinal, or ``None`` when unknown."""
    return context.catalog.get(trace_id, {}).get(node_key)


__all__ = [
    "CatalogContext",
    "build_catalog_context",
    "build_graph_path_catalog",
    "node_ordinal_for",
]
