# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared agent-graph fixtures for building small ParsedGraph topologies."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from aiperf.dataset.graph.models import (
    ChannelRequirement,
    GraphRecord,
    LlmNode,
    ParsedGraph,
    StaticEdge,
    TraceRecord,
)

LlmNodeFactory = Callable[..., LlmNode]
ParsedGraphFactory = Callable[..., ParsedGraph]


def _make_llm_node(
    output: str,
    *,
    arrival_offset_us: int | None = None,
    min_start_delay_us: float | None = None,
    inputs: list[ChannelRequirement] | None = None,
) -> LlmNode:
    return LlmNode(
        prompt=[f"@{output}"],
        output=output,
        arrival_offset_us=arrival_offset_us,
        min_start_delay_us=min_start_delay_us,
        inputs=inputs or [],
    )


def _make_parsed_graph(
    nodes: dict[str, Any],
    edges: list[StaticEdge],
    *,
    state: dict[str, Any] | None = None,
    trace_id: str = "t",
) -> ParsedGraph:
    graph = GraphRecord(nodes=nodes, edges=edges, state=state or {})
    return ParsedGraph(graph=graph, traces=[TraceRecord(id=trace_id)])


@pytest.fixture
def llm_node() -> LlmNodeFactory:
    """Factory for an LlmNode whose prompt is the ``@<output>`` channel placeholder."""
    return _make_llm_node


@pytest.fixture
def parsed_graph() -> ParsedGraphFactory:
    """Factory for a single-trace ParsedGraph over a node/edge topology."""
    return _make_parsed_graph
