# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-trace PEAK context-length helpers for the recorded-trace adapters.

A later filter-then-cap trace selection pass drops traces whose peak context
(input + output tokens on the largest single request) exceeds
``--max-context-length``. These pure functions compute that peak straight off
each adapter's recorded schema -- no graph build, no tokenization -- so the
selector can screen a corpus cheaply.

The helper mirrors the token accounting the trie builder already applies, so
the screened peak matches the value the built graph would carry:

- dynamo: :func:`dynamo_tree_peak_context` mirrors
  :func:`~aiperf.dataset.graph.adapters.dynamo.trie_lowering.dynamo_trie_nodes`'s
  per-record token read (``replay.input_length`` else ``input_tokens`` else 1,
  plus ``output_tokens`` or 0). Output caps are applied during lowering.
"""

from __future__ import annotations

from collections.abc import Iterable

from aiperf.dataset.graph.adapters.dynamo.trace_reader import AgentTraceRecord


def _record_input_length(record: AgentTraceRecord) -> int:
    """The input length one record CONTRIBUTES, block-exact.

    Mirrors the trie lowering's per-record read (``replay.input_length`` else
    ``input_tokens`` else ``1``) and then applies the same block-alignment the
    trie applies at emission: only whole blocks are sent, so the ``in % bs``
    partial tail must not be screened against. A prompt shorter than one block
    keeps its recorded length -- the small-prompt fallback synthesizes it whole.
    The block size is the record's own recorded ``trace_block_size``; a record
    with no replay lowers through the virtual-hash fallback at
    :data:`~aiperf.dataset.graph.adapters.dynamo.trie_lowering.DEFAULT_VIRTUAL_BLOCK_SIZE`
    -- an approximation for that record only, since the real lowering resolves
    ONE block size per trace and a mixed trace would floor such a record at its
    sibling's recorded block size instead. Screening stays per-record and
    tokenizer-free by design.
    """
    from aiperf.dataset.graph.adapters.dynamo.trie_lowering import (
        DEFAULT_VIRTUAL_BLOCK_SIZE,
    )

    # No cycle: trie_content imports nothing from the adapters package. The
    # covered-count itself cannot be reused here -- screening deliberately never
    # parses ``input_sequence_hashes`` -- but the length rule is single-sourced.
    from aiperf.dataset.graph.segment_trie.trie_content import block_exact_length

    req = record.request
    if req is not None and req.replay is not None:
        input_length = int(req.replay.input_length)
        block_size = int(req.replay.trace_block_size) or DEFAULT_VIRTUAL_BLOCK_SIZE
    elif req is not None and req.input_tokens:
        input_length = int(req.input_tokens)
        block_size = DEFAULT_VIRTUAL_BLOCK_SIZE
    else:
        return 1
    return block_exact_length(input_length // block_size, input_length, block_size)


def dynamo_tree_peak_context(records_or_tree: Iterable[AgentTraceRecord]) -> int:
    """Peak ``input_length + output_tokens`` over a dynamo session-tree's records.

    ``records_or_tree`` is any iterable of :class:`AgentTraceRecord` making up a
    tree (a root session plus its ``parent_session_id`` descendants). Each
    record's input length is read exactly as the trie lowering reads it:
    ``request.replay.input_length`` when replay metadata is present, else
    ``request.input_tokens``, else ``1``; output uses ``request.output_tokens``
    or ``0``. Output caps are applied during lowering. Returns ``0`` for an empty
    iterable.
    """
    peak = 0
    for record in records_or_tree:
        req = record.request
        output_tokens = (
            int(req.output_tokens) if req is not None and req.output_tokens else 0
        )
        peak = max(peak, _record_input_length(record) + output_tokens)
    return peak


def dynamo_tree_peak_input(records_or_tree: Iterable[AgentTraceRecord]) -> int:
    """Return the largest EFFECTIVE (block-exact) input length in a Dynamo session tree."""
    return max((_record_input_length(r) for r in records_or_tree), default=0)


__all__ = [
    "dynamo_tree_peak_context",
    "dynamo_tree_peak_input",
]
