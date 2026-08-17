# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A turn that begins while everything else is still in flight roots at START.

``build_interval_edges`` gives a node whose candidate set is EMPTY a synthetic
``StaticEdge(START -> node, min_start_delay_us=node.start * 1e6)``. The candidate
set is "everything that FINISHED before this node started", so a turn issued
while a long request is still running has none -- and therefore carries its whole
arrival offset as a leading START delay that ``TraceExecutor`` parks on.

This is not a corner case: the production glm-5-2-fp8 capture carries leading
offsets up to 24.4s of exactly this shape, and until recently
``--burst-phase-starts`` silently failed to collapse them at t*=0 (the default
disposition) while the replay-wait advisory silently failed to report them.

The active-interval idle warp cannot shrink these either: the stretch before such
a node is BUSY (the long request is running), and the warp only collapses
stretches where the whole trace is idle. Passing a tiny
``--trace-idle-gap-cap-seconds`` therefore does NOT bound them -- pinned below,
because that misconception cost real debugging time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aiperf.dataset.graph.adapters.dynamo.trace import from_dynamo_trace
from aiperf.dataset.graph.models import START_NODE_ID

from .conftest import write_jsonl

T0 = 1_700_000_000_000
BLOCK_SIZE = 64
SESSION = "sess_gap"


def _record(
    rid: str, received_ms: int, end_ms: int, hashes: list[int]
) -> dict[str, Any]:
    """One ``request_end`` whose replay hashes span a 2-block, 128-token prompt."""
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": end_ms,
        "agent_context": {"session_id": SESSION, "trajectory_id": SESSION},
        "request": {
            "request_id": rid,
            "output_tokens": 8,
            "request_received_ms": received_ms,
            "total_time_ms": float(end_ms - received_ms),
            "replay": {
                "trace_block_size": BLOCK_SIZE,
                "input_length": 2 * BLOCK_SIZE,
                "input_sequence_hashes": hashes,
            },
        },
    }


def _overlapping_trace(tmp_path: Path) -> Path:
    """A: raw [0s, 20s] (long). B: raw [8s, 9s], issued while A is in flight.

    B ends FIRST, so the end-ordered turn indices are B=:0 and A=:1.
    """
    return write_jsonl(
        tmp_path / "gap_start.jsonl",
        [
            _record("req-a", T0, T0 + 20_000, [1001, 1002]),
            _record("req-b", T0 + 8_000, T0 + 9_000, [1001, 2002]),
        ],
    )


def _leading_offsets(graph) -> dict[str, float]:
    return {
        e.target: e.min_start_delay_us
        for e in graph.edges
        if e.source == START_NODE_ID and e.min_start_delay_us
    }


def test_turn_started_mid_flight_carries_its_arrival_as_a_leading_offset(
    tmp_path: Path,
) -> None:
    """B started 8s in with nothing finished, so it roots at START + 8s."""
    parsed = from_dynamo_trace(
        _overlapping_trace(tmp_path), content_root_seed=42, content_tokenizer="builtin"
    )

    assert _leading_offsets(parsed.graph) == {f"{SESSION}:0": 8_000_000.0}


def test_idle_gap_cap_does_not_shrink_a_busy_leading_offset(tmp_path: Path) -> None:
    """A 1s idle-gap cap leaves the 8s offset intact, because nothing was idle.

    ``ActiveIdleWarp`` collapses only stretches where NOTHING is running. The
    long request A blankets [0s, 20s], so the 8s before B is busy, not idle, and
    the cap correctly declines to cut it. Reaching for a smaller cap to shorten
    this is the wrong lever -- the whole point of the reworded advisory.
    """
    trace = _overlapping_trace(tmp_path)
    uncapped = from_dynamo_trace(
        trace, content_root_seed=42, content_tokenizer="builtin"
    )
    capped = from_dynamo_trace(
        trace,
        content_root_seed=42,
        content_tokenizer="builtin",
        idle_gap_cap_seconds=1.0,
    )

    assert _leading_offsets(capped.graph) == _leading_offsets(uncapped.graph)
    assert _leading_offsets(capped.graph) == {f"{SESSION}:0": 8_000_000.0}


def test_max_isl_rejections_are_named_in_the_empty_corpus_error(
    tmp_path: Path,
) -> None:
    """An empty corpus must name the knob that actually emptied it.

    ``--max-isl`` rejects a tree BEFORE the filter-then-cap generator yields, so
    those trees never reach ``SelectionStats``. The error therefore reported
    ``all 0 session-trees exceeded max_context_length=None`` -- pointing the
    operator at a knob that had rejected nothing, while ``max_isl`` had rejected
    everything.
    """
    from aiperf.dataset.graph.adapters.dynamo.trace_reader import (
        EmptyDynamoTraceError,
    )

    trace = _overlapping_trace(tmp_path)
    with pytest.raises(EmptyDynamoTraceError) as exc:
        from_dynamo_trace(
            trace,
            content_root_seed=42,
            content_tokenizer="builtin",
            max_isl=1,
        )

    message = str(exc.value)
    assert "max_isl=1" in message
    assert "all 0 session-trees" not in message
