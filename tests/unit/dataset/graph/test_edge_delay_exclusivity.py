# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rule 54: an edge's end-anchored and start-anchored delays are exclusive."""

from __future__ import annotations

from aiperf.dataset.graph.models import (
    GraphRecord,
    LlmNode,
    ParsedGraph,
    StaticEdge,
    TraceRecord,
)
from aiperf.dataset.graph.validator import (
    _rule_54_edge_delay_exclusivity,
    validate,
)


def graph_with(edge: StaticEdge) -> GraphRecord:
    """Minimal two-LlmNode graph carrying a single edge under test."""
    nodes = {
        "a": LlmNode(prompt=["hi"], output="ra"),
        "b": LlmNode(prompt=["hi"], output="rb"),
    }
    return GraphRecord(nodes=nodes, edges=[edge], state={})


def test_edge_with_both_delay_fields_rejected():
    """Rule 54: delay_after_predecessor_us and delay_after_predecessor_start_us
    are mutually exclusive on one edge."""
    edge = StaticEdge(
        source="a",
        target="b",
        delay_after_predecessor_us=1000.0,
        delay_after_predecessor_start_us=1000.0,
    )
    issues = _rule_54_edge_delay_exclusivity(graph_with(edge))
    assert issues, "both-set edge must produce a validation issue"
    # T1-M1: the rule must also surface through the full validate() entrypoint,
    # not just the direct rule call (adapter-tests-skip-validator trap). Mirror
    # the Task 3 E2E test: wrap the graph in a ParsedGraph with a TraceRecord.
    parsed = ParsedGraph(graph=graph_with(edge), traces=[TraceRecord(id="t")])
    assert any(i.rule_id == "rule-54" for i in validate(parsed)), (
        "rule-54 must surface end-to-end via validate()"
    )


def test_edge_with_single_delay_field_passes():
    for kwargs in (
        {"delay_after_predecessor_us": 1000.0},
        {"delay_after_predecessor_start_us": 1000.0},
        {},
    ):
        edge = StaticEdge(source="a", target="b", **kwargs)
        assert _rule_54_edge_delay_exclusivity(graph_with(edge)) == []


def test_start_sourced_start_anchored_edge_rejected():
    """Rule 54: a START-sourced edge cannot be start-anchored -- the START
    pseudo-node never dispatches, so the target would be silently orphaned."""
    edge = StaticEdge(
        source="START", target="b", delay_after_predecessor_start_us=1000.0
    )
    issues = _rule_54_edge_delay_exclusivity(graph_with(edge))
    assert issues, "START-sourced start-anchored edge must produce an issue"
    assert all(i.rule_id == "rule-54" for i in issues)
    assert any("START" in i.message for i in issues)


def test_start_sourced_min_start_delay_only_passes():
    """A START-sourced edge with only min_start_delay_us is the correct way to
    express an absolute offset from trace start; rule 54 must accept it."""
    edge = StaticEdge(source="START", target="b", min_start_delay_us=1000.0)
    assert _rule_54_edge_delay_exclusivity(graph_with(edge)) == []
