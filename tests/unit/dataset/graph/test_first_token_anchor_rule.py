# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rule 55: a first-token-anchored edge must carry its dispatch fallback and a
real source."""

from __future__ import annotations

from aiperf.dataset.graph.models import (
    GraphRecord,
    LlmNode,
    ParsedGraph,
    StaticEdge,
    TraceRecord,
)
from aiperf.dataset.graph.validator import (
    _rule_55_first_token_anchor_shape,
    validate,
)


def graph_with(edge: StaticEdge) -> GraphRecord:
    """Minimal two-LlmNode graph carrying a single edge under test."""
    nodes = {
        "a": LlmNode(prompt=["hi"], output="ra"),
        "b": LlmNode(prompt=["hi"], output="rb"),
    }
    return GraphRecord(nodes=nodes, edges=[edge], state={})


def _rule_55_issues_via_validate(edge: StaticEdge) -> list:
    """Rule-55 issues produced by the FULL ``validate()`` entrypoint.

    Positive-path wiring proof: if the ``_rule_55...`` call were dropped from
    ``validator.validate``, the rule-level assertions would still pass while
    every violating graph sailed through the real entrypoint -- so the
    rejection tests must also assert through here.
    """
    parsed = ParsedGraph(graph=graph_with(edge), traces=[TraceRecord(id="t")])
    return [i for i in validate(parsed) if i.rule_id == "rule-55"]


def test_first_token_without_start_anchor_rejected():
    """Rule 55: a first-token anchor without its dispatch fallback
    (delay_after_predecessor_start_us) is rejected."""
    edge = StaticEdge(
        source="a",
        target="b",
        delay_after_predecessor_first_token_us=1000.0,
    )
    assert _rule_55_first_token_anchor_shape(graph_with(edge))
    assert _rule_55_issues_via_validate(edge), (
        "violating edge must yield rule-55 through validate() (wiring proof)"
    )


def test_first_token_with_end_anchor_rejected():
    """Rule 55: a first-token anchor must not combine with the completion
    anchor delay_after_predecessor_us."""
    edge = StaticEdge(
        source="a",
        target="b",
        delay_after_predecessor_us=1000.0,
        delay_after_predecessor_first_token_us=1000.0,
    )
    assert _rule_55_first_token_anchor_shape(graph_with(edge))
    assert _rule_55_issues_via_validate(edge), (
        "violating edge must yield rule-55 through validate() (wiring proof)"
    )


def test_first_token_from_start_source_rejected():
    """Rule 55: a START-sourced edge cannot be first-token-anchored -- the
    START pseudo-node never dispatches or streams a first token."""
    edge = StaticEdge(
        source="START",
        target="b",
        delay_after_predecessor_start_us=2000.0,
        delay_after_predecessor_first_token_us=1000.0,
    )
    issues = _rule_55_first_token_anchor_shape(graph_with(edge))
    assert issues, "START-sourced first-token-anchored edge must produce an issue"
    assert all(i.rule_id == "rule-55" for i in issues)
    assert any("START" in i.message for i in issues)
    validate_issues = _rule_55_issues_via_validate(edge)
    assert validate_issues, (
        "violating edge must yield rule-55 through validate() (wiring proof)"
    )
    assert any("START" in i.message for i in validate_issues)


def test_valid_first_token_edge_passes_full_validate():
    """A first-token anchor alongside its dispatch fallback on a real source is
    the correct shape; rule 55 must accept it, end-to-end via validate()."""
    edge = StaticEdge(
        source="a",
        target="b",
        delay_after_predecessor_start_us=3000.0,
        delay_after_predecessor_first_token_us=1000.0,
    )
    assert _rule_55_first_token_anchor_shape(graph_with(edge)) == []
    # Mirror test_edge_delay_exclusivity.py: the rule must also stay silent
    # through the full validate() entrypoint (adapter-tests-skip-validator trap).
    parsed = ParsedGraph(graph=graph_with(edge), traces=[TraceRecord(id="t")])
    assert not any(i.rule_id == "rule-55" for i in validate(parsed)), (
        "valid first-token edge must not trip rule-55 via validate()"
    )
