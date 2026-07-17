# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-trace PEAK context-length helpers for the recorded-trace adapters.

A later filter-then-cap trace selection pass drops traces whose peak context
(input + output tokens on the largest single request) exceeds
``--max-context-length``. These pure functions compute that peak straight off
each adapter's recorded schema -- no graph build, no tokenization -- so the
selector can screen a corpus cheaply.

Both helpers mirror the token accounting their trie builders already apply, so
the screened peak matches the value the built graph would carry:

- weka: :func:`weka_trace_peak_context` mirrors
  :func:`~aiperf.dataset.graph.adapters.weka.trie_build._build_llm_node`'s
  ``max_osl`` cap -- only TOP-LEVEL leaves are capped to ``min(out, max_osl)``;
  subagent-body leaves (any nesting depth) stay uncapped, matching
  ``_top_level_leaf_ids``.
- dynamo: :func:`dynamo_tree_peak_context` mirrors
  :func:`~aiperf.dataset.graph.adapters.dynamo.trie_lowering.dynamo_trie_nodes`'s
  per-record token read (``replay.input_length`` else ``input_tokens`` else 1,
  plus ``output_tokens`` or 0). ``max_osl`` does NOT cap dynamo output.
"""

from __future__ import annotations

from collections.abc import Iterable

from aiperf.dataset.graph.adapters.dynamo.trace_reader import AgentTraceRecord
from aiperf.dataset.graph.adapters.weka.trace_models import (
    WekaRequest,
    WekaSubagentEntry,
    WekaTrace,
)


def weka_trace_peak_context(trace: WekaTrace, *, max_osl: int | None) -> int:
    """Peak ``input_length + output`` over every leaf request in a weka trace.

    Top-level leaves (those directly in ``trace.requests``) use
    ``min(output_length, max_osl)`` when ``max_osl`` is set, matching the
    dispatch ``max_tokens`` cap the trie builder applies to top-level chain
    requests. Subagent-body leaves (inside any :class:`WekaSubagentEntry`, at
    any nesting depth) always use the raw recorded ``output_length``. Returns
    ``0`` for a trace with no leaf requests.
    """
    peak = 0
    for req in trace.requests:
        if isinstance(req, WekaSubagentEntry):
            peak = max(peak, _uncapped_leaf_peak(req.requests))
        else:
            output = (
                req.output_length
                if max_osl is None
                else min(req.output_length, max_osl)
            )
            peak = max(peak, req.input_length + output)
    return peak


def _uncapped_leaf_peak(requests: list[WekaRequest]) -> int:
    """Peak ``input_length + output_length`` over subagent-body leaves (uncapped).

    Recurses into nested :class:`WekaSubagentEntry` markers so a
    subagent-within-subagent body is screened at its raw recorded ``out``.
    """
    peak = 0
    for req in requests:
        if isinstance(req, WekaSubagentEntry):
            peak = max(peak, _uncapped_leaf_peak(req.requests))
        else:
            peak = max(peak, req.input_length + req.output_length)
    return peak


def dynamo_tree_peak_context(records_or_tree: Iterable[AgentTraceRecord]) -> int:
    """Peak ``input_length + output_tokens`` over a dynamo session-tree's records.

    ``records_or_tree`` is any iterable of :class:`AgentTraceRecord` making up a
    tree (a root session plus its ``parent_session_id`` descendants). Each
    record's input length is read exactly as the trie lowering reads it:
    ``request.replay.input_length`` when replay metadata is present, else
    ``request.input_tokens``, else ``1``; output uses ``request.output_tokens``
    or ``0``. ``max_osl`` does NOT cap dynamo output. Returns ``0`` for an empty
    iterable.
    """
    peak = 0
    for record in records_or_tree:
        req = record.request
        if req is not None and req.replay is not None:
            input_length = int(req.replay.input_length)
        elif req is not None and req.input_tokens:
            input_length = int(req.input_tokens)
        else:
            input_length = 1
        output_tokens = (
            int(req.output_tokens) if req is not None and req.output_tokens else 0
        )
        peak = max(peak, input_length + output_tokens)
    return peak


__all__ = [
    "dynamo_tree_peak_context",
    "weka_trace_peak_context",
]
