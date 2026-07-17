# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TimingManager first-token-anchoring source-streaming advisory.

A post-TTFT-anchored ``StaticEdge`` releases its successor at the SOURCE node's
observed first token. That observation only exists when the source node itself
streams. Each graph node streams per its own recorded ``streaming`` mode
(per-request override), so the global ``--streaming`` flag does not govern
whether a first-token event is emitted. The advisory therefore warns iff a
first-token edge's SOURCE ``LlmNode`` carries ``streaming=False`` -- possible
only in hand-authored/degenerate graphs, since recorded corpora are consistent
by construction (the same recorded ttft drives both the edge and the node mode).
These tests exercise that source-node matrix; the advisory logs once via the
module logger (not ``self.warning``), so no lifecycle/logger init is needed.
"""

import logging

from aiperf.dataset.graph.models import (
    GraphRecord,
    LlmNode,
    ParsedGraph,
    StaticEdge,
    TraceRecord,
)
from aiperf.timing.manager import TimingManager

_ADVISORY_LOGGER = "aiperf.timing.manager"
_ADVISORY_NEEDLE = "first-token-anchored"


def _make_tm() -> TimingManager:
    """Bare TimingManager (bypassing service init).

    The advisory reads nothing off ``self`` -- it scans the passed graph and
    logs via the module logger -- so a raw ``__new__`` instance suffices.
    """
    return TimingManager.__new__(TimingManager)


def _graph_with_nodes(*, source_streaming: bool, first_token: bool) -> GraphRecord:
    """One edge ``a -> b`` with both ends materialized as ``LlmNode``s.

    ``first_token`` toggles post-TTFT anchoring on the edge; ``source_streaming``
    sets the source node ``a``'s recorded streaming mode.
    """
    edge = StaticEdge(
        source="a",
        target="b",
        delay_after_predecessor_start_us=1000.0,
        delay_after_predecessor_first_token_us=500.0 if first_token else None,
    )
    return GraphRecord(
        nodes={
            "a": LlmNode(prompt=[], output="a_out", streaming=source_streaming),
            "b": LlmNode(prompt=[], output="b_out"),
        },
        edges=[edge],
    )


def _advisory_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if _ADVISORY_NEEDLE in r.getMessage()]


def test_recorded_streaming_sources_are_silent(caplog):
    """Recorded corpus: first-token source streams -> no warning (global OFF)."""
    parsed = ParsedGraph(
        graph=_graph_with_nodes(source_streaming=True, first_token=True),
        traces=[TraceRecord(id="t-1", tags=["x"])],
    )
    tm = _make_tm()
    with caplog.at_level(logging.WARNING, logger=_ADVISORY_LOGGER):
        tm._advise_non_streaming_first_token_sources(parsed)
    assert _advisory_records(caplog) == []


def test_non_streaming_first_token_source_warns_once(caplog):
    """Degenerate corpus: first-token source has streaming=False -> one warning."""
    parsed = ParsedGraph(
        graph=_graph_with_nodes(source_streaming=False, first_token=True),
        traces=[TraceRecord(id="t-1", tags=["x"])],
    )
    tm = _make_tm()
    with caplog.at_level(logging.WARNING, logger=_ADVISORY_LOGGER):
        tm._advise_non_streaming_first_token_sources(parsed)
    records = _advisory_records(caplog)
    assert len(records) == 1, "expected exactly one source-streaming advisory warning"
    message = records[0].getMessage()
    assert "'a'" in message, "warning must name the offending source node id"
    assert "COMPLETION latch" in message, (
        "warning must describe the completion-latch fallback"
    )


def test_no_first_token_edges_is_silent(caplog):
    """No post-TTFT edge -> no first-token source -> silent even for streaming=False."""
    parsed = ParsedGraph(
        graph=_graph_with_nodes(source_streaming=False, first_token=False),
        traces=[TraceRecord(id="t-1", tags=["x"])],
    )
    tm = _make_tm()
    with caplog.at_level(logging.WARNING, logger=_ADVISORY_LOGGER):
        tm._advise_non_streaming_first_token_sources(parsed)
    assert _advisory_records(caplog) == []


def test_non_streaming_source_in_subgraph_warns(caplog):
    """Detection scans subgraph bodies too, not just the top-level graph."""
    parsed = ParsedGraph(
        graph=GraphRecord(),
        graphs={"body": _graph_with_nodes(source_streaming=False, first_token=True)},
        traces=[TraceRecord(id="t-1", tags=["x"])],
    )
    tm = _make_tm()
    with caplog.at_level(logging.WARNING, logger=_ADVISORY_LOGGER):
        tm._advise_non_streaming_first_token_sources(parsed)
    assert len(_advisory_records(caplog)) == 1


def test_multiple_non_streaming_sources_warn_once(caplog):
    """Once-per-run: multiple non-streaming first-token sources -> one warning."""
    graph = GraphRecord(
        nodes={
            "a": LlmNode(prompt=[], output="a_out", streaming=False),
            "b": LlmNode(prompt=[], output="b_out"),
            "c": LlmNode(prompt=[], output="c_out", streaming=False),
            "d": LlmNode(prompt=[], output="d_out"),
        },
        edges=[
            StaticEdge(
                source="a",
                target="b",
                delay_after_predecessor_start_us=1000.0,
                delay_after_predecessor_first_token_us=500.0,
            ),
            StaticEdge(
                source="c",
                target="d",
                delay_after_predecessor_start_us=1000.0,
                delay_after_predecessor_first_token_us=600.0,
            ),
        ],
    )
    parsed = ParsedGraph(graph=graph, traces=[TraceRecord(id="t-1", tags=["x"])])
    tm = _make_tm()
    with caplog.at_level(logging.WARNING, logger=_ADVISORY_LOGGER):
        tm._advise_non_streaming_first_token_sources(parsed)
    assert len(_advisory_records(caplog)) == 1
