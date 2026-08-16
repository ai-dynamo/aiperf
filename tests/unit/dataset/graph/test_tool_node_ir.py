# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ToolNode as a second graph node kind: union round-trip and store isolation."""

from __future__ import annotations

import msgspec

from aiperf.dataset.graph.models import (
    GraphRecord,
    LlmNode,
    NodeKind,
    NodeUnion,
    ParsedGraph,
    ToolNode,
    TraceRecord,
)
from aiperf.dataset.graph.segment_trie.store_builder import flat_trie_ordinals


def test_tool_node_reports_its_kind() -> None:
    node = ToolNode(commands=["echo hi"], output="t0_out")
    assert node.node_type is NodeKind.TOOL
    assert node.write_channels == ["t0_out"]


def test_tool_node_round_trips_through_the_tagged_union() -> None:
    """The codec must discriminate ToolNode from LlmNode on `node_type`."""
    node = ToolNode(commands=["ls -la", "cat x"], output="t0_out", timeout_s=30.0)
    raw = msgspec.json.encode(node)
    assert b'"node_type":"tool"' in raw
    assert msgspec.json.decode(raw, type=NodeUnion) == node


def test_llm_node_still_round_trips_after_the_union_widens() -> None:
    node = LlmNode(prompt=[], output="n0_out", max_tokens=7)
    raw = msgspec.json.encode(node)
    assert msgspec.json.decode(raw, type=NodeUnion) == node


def test_store_ordinals_skip_tool_nodes() -> None:
    """Tool nodes carry no prompt manifest, so they must not consume an ordinal.

    Build-plane and schedule-plane ordinals both come from this function; a tool
    node taking a slot would shift every later LlmNode's manifest key.
    """
    graph = GraphRecord(
        nodes={
            "n0": LlmNode(prompt=[], output="n0_out", arrival_offset_us=0),
            "t0": ToolNode(commands=["echo hi"], output="t0_out", arrival_offset_us=1),
            "n1": LlmNode(prompt=[], output="n1_out", arrival_offset_us=2),
        },
        edges=[],
    )
    parsed = ParsedGraph(graph=graph, traces=[TraceRecord(id="t")])
    ordinals = flat_trie_ordinals(parsed, parsed.traces[0])
    assert ordinals == {"n0": 0, "n1": 1}
