# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared builders for the graph parser tests."""

from __future__ import annotations

from pathlib import Path

from aiperf.dataset.graph.models import (
    GraphRecord,
    LlmNode,
    ParsedGraph,
    StaticEdge,
    TraceRecord,
)

DYNAMO_NESTED_FIXTURE = (
    Path(__file__).resolve().parent
    / "adapters"
    / "fixtures"
    / "dynamo_nested"
    / "nested_2_level.jsonl.gz"
)


def write_yaml(tmp_path: Path, text: str) -> Path:
    """Write ``text`` as the run's workload YAML and return its path."""
    p = tmp_path / "workload.yaml"
    p.write_text(text)
    return p


def llm_node(output: str = "o", prompt_text: str = "q") -> LlmNode:
    """A minimal single-user-turn LlmNode."""
    return LlmNode(prompt=[{"role": "user", "content": prompt_text}], output=output)


def single_edge_graph(edge: StaticEdge) -> GraphRecord:
    """Minimal two-LlmNode graph ('a' -> 'b') carrying a single edge under test."""
    nodes = {
        "a": LlmNode(prompt=["hi"], output="ra"),
        "b": LlmNode(prompt=["hi"], output="rb"),
    }
    return GraphRecord(nodes=nodes, edges=[edge], state={})


def parsed_with(graph: GraphRecord, trace_id: str = "t") -> ParsedGraph:
    """Wrap ``graph`` in a one-trace ParsedGraph."""
    return ParsedGraph(graph=graph, traces=[TraceRecord(id=trace_id)])
