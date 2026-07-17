# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-trace single-graph projection for the weka graph-IR replay strategy.

Extracted from ``GraphIRReplayStrategy`` (no behavior change) to keep that file
under the file-size ergonomics ceiling. The runtime ``TraceExecutor`` (and the
``rewrite_for_snapshot`` / ``rewrite_for_warmup`` / ``compute_snapshot`` paths it
drives) read ``parsed.graph`` directly, so every trace must be projected onto a
single-graph ``ParsedGraph`` view whose ``.graph`` is the trace's OWN resolved
topology before it reaches the executor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec

from aiperf.dataset.graph.models import resolve_trace_graph

if TYPE_CHECKING:
    from aiperf.dataset.graph.models import ParsedGraph, TraceRecord

__all__ = ["parsed_for_trace"]


def parsed_for_trace(parsed: ParsedGraph, trace: TraceRecord) -> ParsedGraph:
    """Project ``parsed`` onto a single-graph view whose ``.graph`` is ``trace``'s.

    Single-graph workloads (``trace.graph_ref is None``) return ``parsed``
    unchanged -- ``resolve_trace_graph`` already yields the shared ``parsed.graph``,
    so the projection is a strict identity and the byte-unchanged single-file /
    hand-authored path is untouched.

    Multi-graph workloads (a heterogeneous directory / HuggingFace corpus of weka
    traces) keep each trace's own topology in ``parsed.graphs`` keyed by
    ``trace.graph_ref``; ``parsed.graph`` is only the FIRST file's graph. This
    returns a ``ParsedGraph`` whose ``.graph`` is this trace's resolved topology
    (``parsed.graphs[trace.graph_ref]``) so the executor and the t* rewrites
    operate on the CORRECT per-trace graph rather than the first file's graph.
    The ``traces`` list is narrowed to this one trace so the rewrite's
    matching-trace lookup resolves it; ``segment_pool`` and ``graphs`` ride along
    via ``replace``. The pool rides along as the segment-store-IR marker
    (``segment_pool is not None`` gates the executor's dispatch-failure
    sentinel writes) and for debug
    materialization via ``read_prompt_segment_ids`` -- NOT for inline node
    prompts (trie-route ``LlmNode``s carry ``prompt=[]``; the worker materializes
    from the mmap segment store).

    Without this projection a non-first trace runs the wrong topology -- the
    first file's ``parsed.graph`` -- instead of its own resolved one.
    """
    graph = resolve_trace_graph(parsed, trace)
    if graph is parsed.graph:
        return parsed
    return msgspec.structs.replace(parsed, graph=graph, traces=[trace])
