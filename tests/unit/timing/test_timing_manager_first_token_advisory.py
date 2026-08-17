# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TimingManager advisory warning for non-streaming first-token-anchored edge sources."""

import logging
from collections.abc import Callable

import pytest
from pytest import param

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
    """Bare TimingManager instance that bypasses service initialization."""
    return TimingManager.__new__(TimingManager)


def _graph_with_nodes(*, source_streaming: bool, first_token: bool) -> GraphRecord:
    """One edge ``a -> b`` with both ends materialized as ``LlmNode``s."""
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


def _two_non_streaming_sources() -> GraphRecord:
    """Two independent first-token edges whose sources both have streaming off."""
    return GraphRecord(
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


def _advise(parsed: ParsedGraph, caplog: pytest.LogCaptureFixture) -> list[str]:
    """Run the advisory pass and return the advisory warning messages it emitted."""
    tm = _make_tm()
    with caplog.at_level(logging.WARNING, logger=_ADVISORY_LOGGER):
        tm._advise_non_streaming_first_token_sources(parsed)
    return [
        r.getMessage() for r in caplog.records if _ADVISORY_NEEDLE in r.getMessage()
    ]


@pytest.mark.parametrize(
    "build_parsed,expected_warnings",
    [
        param(
            lambda: ParsedGraph(
                graph=_graph_with_nodes(source_streaming=True, first_token=True),
                traces=[TraceRecord(id="t-1", tags=["x"])],
            ),
            0,
            id="recorded_streaming_source_is_silent",
        ),
        param(
            lambda: ParsedGraph(
                graph=_graph_with_nodes(source_streaming=False, first_token=False),
                traces=[TraceRecord(id="t-1", tags=["x"])],
            ),
            0,
            id="no_first_token_edge_is_silent_even_when_not_streaming",
        ),
        param(
            lambda: ParsedGraph(
                graph=_graph_with_nodes(source_streaming=False, first_token=True),
                traces=[TraceRecord(id="t-1", tags=["x"])],
            ),
            1,
            id="non_streaming_first_token_source_warns",
        ),
        param(
            lambda: ParsedGraph(
                graph=GraphRecord(),
                graphs={
                    "body": _graph_with_nodes(source_streaming=False, first_token=True)
                },
                traces=[TraceRecord(id="t-1", tags=["x"])],
            ),
            1,
            id="detection_scans_subgraph_bodies",
        ),
        param(
            lambda: ParsedGraph(
                graph=_two_non_streaming_sources(),
                traces=[TraceRecord(id="t-1", tags=["x"])],
            ),
            1,
            id="multiple_offending_sources_warn_once_per_run",
        ),
    ],
)  # fmt: skip
def test_advisory_warning_count_matches_graph_shape(
    build_parsed: Callable[[], ParsedGraph],
    expected_warnings: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Advisory fires once per run only when a first-token edge source is non-streaming."""
    assert len(_advise(build_parsed(), caplog)) == expected_warnings


def test_advisory_message_names_node_and_completion_latch_fallback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The advisory warning identifies the offending node and the fallback it triggers."""
    parsed = ParsedGraph(
        graph=_graph_with_nodes(source_streaming=False, first_token=True),
        traces=[TraceRecord(id="t-1", tags=["x"])],
    )
    messages = _advise(parsed, caplog)
    assert len(messages) == 1, "expected exactly one source-streaming advisory warning"
    assert "'a'" in messages[0], "warning must name the offending source node id"
    assert "COMPLETION latch" in messages[0], (
        "warning must describe the completion-latch fallback"
    )
