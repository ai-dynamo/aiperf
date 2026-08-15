# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fidelity-parity coverage: recorded output-token pinning plus recorded edge delays warped through the shared ``ActiveIdleWarp``."""

from __future__ import annotations

from pathlib import Path

from aiperf.dataset.graph.adapters.dynamo.trace import from_dynamo_trace
from aiperf.dataset.graph.models import (
    GraphRecord,
    LlmNode,
    ParsedGraph,
    StaticEdge,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "dynamo_nested"

CADENCE_FIXTURE = "nested_2_level.jsonl.gz"

# Recorded timing lives on the EDGES (interval-order ``delay_after_predecessor_us``
# / START ``min_start_delay_us``), not on per-node ``pre_wait`` WaitSpecs.


def _llm_nodes(graph: GraphRecord) -> list[LlmNode]:
    return [n for n in graph.nodes.values() if isinstance(n, LlmNode)]


def _static_edges(pb: ParsedGraph) -> list[StaticEdge]:
    return [e for e in pb.graph.edges if isinstance(e, StaticEdge)]


# --- recorded output pinning (weka parity, always on) -------------------------


def test_output_cap_always_pinned_to_recorded() -> None:
    """Every node pins ``max_tokens`` to its recorded ``output_tokens`` by default (weka parity)."""
    # A recorded 0 upgrades to 1 (wire_output_cap): zero is not a legal wire cap.
    pb = from_dynamo_trace(FIXTURES / "nested_2_level.jsonl.gz")
    pinned = 0
    for node in _llm_nodes(pb.graph):
        cap = node.max_tokens
        recorded = node.expected.output_tokens if node.expected else None
        assert cap == (recorded if recorded else 1)
        if recorded:
            pinned += 1
    assert pinned >= 1


def test_output_cap_honors_synthesis_max_osl() -> None:
    """Dynamo lowers the shared synthesis OSL cap into max_tokens."""
    pb = from_dynamo_trace(FIXTURES / CADENCE_FIXTURE, max_osl=1)
    assert all((node.max_tokens or 0) <= 1 for node in _llm_nodes(pb.graph))


# --- idle-warped recorded timing on edges ------------------------------------


def test_default_replays_recorded_edge_delays() -> None:
    """Delays are never silently dropped: the default keeps the recorded gaps."""
    pb = from_dynamo_trace(FIXTURES / CADENCE_FIXTURE)
    delays = [e.delay_after_predecessor_us or 0.0 for e in _static_edges(pb)]
    assert any(d > 0.0 for d in delays), (
        "the default parse must keep the recorded end-to-start gaps on the "
        f"binding edges; got {delays}"
    )
    # Arrival stamping rides the same warped clock.
    assert any((n.arrival_offset_us or 0) > 0 for n in _llm_nodes(pb.graph))


def test_idle_gap_cap_none_keeps_raw_recorded_gaps() -> None:
    """An explicit ``idle_gap_cap_seconds=None`` disables the warp, matching a cap set above every recorded gap."""
    raw = from_dynamo_trace(FIXTURES / CADENCE_FIXTURE, idle_gap_cap_seconds=None)
    uncompressed = from_dynamo_trace(
        FIXTURES / CADENCE_FIXTURE, idle_gap_cap_seconds=1e9
    )
    raw_delays = [e.delay_after_predecessor_us or 0.0 for e in _static_edges(raw)]
    assert raw_delays == [
        e.delay_after_predecessor_us or 0.0 for e in _static_edges(uncompressed)
    ]
    assert any(d > 0.0 for d in raw_delays)


def test_idle_gap_cap_zero_collapses_recorded_gaps() -> None:
    """``idle_gap_cap_seconds=0.0`` compresses every idle gap, so all warped edge delays and arrival offsets hit zero."""
    pb = from_dynamo_trace(
        FIXTURES / CADENCE_FIXTURE,
        idle_gap_cap_seconds=0.0,
    )
    for e in _static_edges(pb):
        assert (e.delay_after_predecessor_us or 0.0) == 0.0
        assert (e.min_start_delay_us or 0.0) == 0.0
    for n in _llm_nodes(pb.graph):
        assert (n.arrival_offset_us or 0) == 0


def test_ignore_trace_delays_collapses_recorded_gaps() -> None:
    """Ignoring trace delays removes recorded idle time from the graph timeline."""
    pb = from_dynamo_trace(FIXTURES / CADENCE_FIXTURE, ignore_trace_delays=True)
    assert all((e.delay_after_predecessor_us or 0.0) == 0.0 for e in _static_edges(pb))


# --- DynamoTraceAdapter.parse knob sources -----------------------------------


def test_parse_forwards_ctx_idle_gap_cap() -> None:
    """The shared trace idle-gap cap reaches the dynamo build through ``GraphParseContext``."""
    from aiperf.dataset.graph.adapters.dynamo import trace as dt
    from aiperf.dataset.graph.parse_context import GraphParseContext

    pb = dt.DynamoTraceAdapter.parse(
        FIXTURES / CADENCE_FIXTURE,
        ctx=GraphParseContext(idle_gap_cap_seconds=0.0),
    )
    for e in _static_edges(pb):
        assert (e.delay_after_predecessor_us or 0.0) == 0.0
        assert (e.min_start_delay_us or 0.0) == 0.0
