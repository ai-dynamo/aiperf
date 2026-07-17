# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validator rule coverage.

Covers: rule-1 iterative cycle detection at corpus scale (no RecursionError),
``validate()`` visiting every ``parsed.graphs`` entry, rule-56 dangling edge
endpoints, rule-57 non-finite delay values, and the rule-13 default-"manual"
provenance warning for adapter-emitted graphs.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.dataset.graph.models import (
    GraphRecord,
    LlmNode,
    ParsedGraph,
    ProvenanceSpec,
    StaticEdge,
    TraceRecord,
)
from aiperf.dataset.graph.validator import (
    ValidationSeverity,
    _rule_01_cycles,
    _rule_13_provenance_tool,
    _rule_56_edge_endpoints,
    _rule_57_finite_delays,
    validate,
)


def _node() -> LlmNode:
    return LlmNode(prompt=[{"role": "user", "content": "q"}], output="o")


def _chain_graph(n: int) -> GraphRecord:
    nodes = {f"n{i}": _node() for i in range(n)}
    edges = [StaticEdge(source="START", target="n0")]
    edges += [StaticEdge(source=f"n{i}", target=f"n{i + 1}") for i in range(n - 1)]
    edges.append(StaticEdge(source=f"n{n - 1}", target="END"))
    return GraphRecord(nodes=nodes, edges=edges)


class TestRule01IterativeCycles:
    """G8: rule-1 must not recurse -- recorded corpora exceed 100k-node chains."""

    def test_100k_node_chain_validates_without_recursion_error(self) -> None:
        assert _rule_01_cycles(_chain_graph(100_000)) == []

    def test_long_cycle_still_detected(self) -> None:
        n = 5_000
        graph = _chain_graph(n)
        edges = [*graph.edges, StaticEdge(source=f"n{n - 1}", target="n0")]
        import msgspec

        issues = _rule_01_cycles(msgspec.structs.replace(graph, edges=edges))
        assert [i.rule_id for i in issues] == ["rule-1"]

    def test_small_cycle_detected(self) -> None:
        graph = GraphRecord(
            nodes={"a": _node(), "b": _node()},
            edges=[
                StaticEdge(source="START", target="a"),
                StaticEdge(source="a", target="b"),
                StaticEdge(source="b", target="a"),
            ],
        )
        assert [i.rule_id for i in _rule_01_cycles(graph)] == ["rule-1"]

    def test_diamond_is_not_a_cycle(self) -> None:
        graph = GraphRecord(
            nodes={k: _node() for k in ("a", "b", "c", "d")},
            edges=[
                StaticEdge(source="START", target="a"),
                StaticEdge(source="a", target="b"),
                StaticEdge(source="a", target="c"),
                StaticEdge(source="b", target="d"),
                StaticEdge(source="c", target="d"),
                StaticEdge(source="d", target="END"),
            ],
        )
        assert _rule_01_cycles(graph) == []


class TestValidateVisitsAllGraphs:
    """G7: validate() must run every rule over parsed.graphs values too."""

    def test_cycle_in_secondary_graph_reported_with_graph_name(self) -> None:
        clean = _chain_graph(2)
        cyclic = GraphRecord(
            nodes={"a": _node(), "b": _node()},
            edges=[
                StaticEdge(source="START", target="a"),
                StaticEdge(source="a", target="b"),
                StaticEdge(source="b", target="a"),
            ],
        )
        parsed = ParsedGraph(
            graph=clean,
            graphs={"t1": clean, "t2": cyclic},
            traces=[
                TraceRecord(id="t1", graph_ref="t1"),
                TraceRecord(id="t2", graph_ref="t2"),
            ],
        )
        cycle_issues = [i for i in validate(parsed) if i.rule_id == "rule-1"]
        assert cycle_issues, "cycle in a parsed.graphs entry must surface"
        assert all(i.location.startswith("graphs[t2]") for i in cycle_issues)

    def test_aliased_main_graph_not_double_reported(self) -> None:
        # The native lowering aliases parsed.graph to the first graphs entry;
        # its issues must not be duplicated.
        cyclic = GraphRecord(
            nodes={"a": _node()},
            edges=[
                StaticEdge(source="START", target="a"),
                StaticEdge(source="a", target="a"),
            ],
        )
        parsed = ParsedGraph(
            graph=cyclic,
            graphs={"t1": cyclic},
            traces=[TraceRecord(id="t1", graph_ref="t1")],
        )
        assert len([i for i in validate(parsed) if i.rule_id == "rule-1"]) == 1

    def test_new_rules_run_via_validate(self) -> None:
        graph = GraphRecord(
            nodes={"a": _node()},
            edges=[
                StaticEdge(source="START", target="a"),
                StaticEdge(source="a", target="ghost"),
                StaticEdge(
                    source="a", target="END", delay_after_predecessor_us=float("inf")
                ),
            ],
        )
        parsed = ParsedGraph(graph=graph, traces=[TraceRecord(id="t")])
        rule_ids = {i.rule_id for i in validate(parsed)}
        assert {"rule-56", "rule-57"} <= rule_ids


class TestRule56EdgeEndpoints:
    """G9: edges must reference declared nodes or the matching sentinel."""

    @pytest.mark.parametrize(
        ("source", "target", "bad"),
        [
            param("START", "ghost", "ghost", id="dangling_target"),
            param("ghost", "END", "ghost", id="dangling_source"),
            param("END", "a", "END", id="end_as_source"),
            param("a", "START", "START", id="start_as_target"),
        ],
    )  # fmt: skip
    def test_dangling_endpoint_flagged(
        self, source: str, target: str, bad: str
    ) -> None:
        graph = GraphRecord(
            nodes={"a": _node()},
            edges=[StaticEdge(source=source, target=target)],
        )
        issues = _rule_56_edge_endpoints(graph)
        assert [i.rule_id for i in issues] == ["rule-56"]
        assert bad in issues[0].message

    def test_declared_endpoints_and_sentinels_pass(self) -> None:
        graph = GraphRecord(
            nodes={"a": _node(), "b": _node()},
            edges=[
                StaticEdge(source="START", target="a"),
                StaticEdge(source="a", target="b"),
                StaticEdge(source="b", target="END"),
            ],
        )
        assert _rule_56_edge_endpoints(graph) == []


class TestRule57FiniteDelays:
    """G3 (validator half): already-decoded graphs with non-finite delays are
    caught even though typed construction bypasses decode.py."""

    @pytest.mark.parametrize(
        "field",
        [
            param("delay_after_predecessor_us", id="completion"),
            param("min_start_delay_us", id="min_start"),
            param("delay_after_predecessor_start_us", id="start_anchor"),
        ],
    )  # fmt: skip
    def test_inf_edge_delay_flagged(self, field: str) -> None:
        graph = GraphRecord(
            nodes={"a": _node(), "b": _node()},
            edges=[StaticEdge(source="a", target="b", **{field: float("inf")})],
        )
        issues = _rule_57_finite_delays(graph)
        assert [i.rule_id for i in issues] == ["rule-57"]
        assert issues[0].location.endswith(field)

    def test_inf_node_min_start_delay_flagged(self) -> None:
        node = LlmNode(
            prompt=[{"role": "user", "content": "q"}],
            output="o",
            min_start_delay_us=float("inf"),
        )
        graph = GraphRecord(nodes={"a": node}, edges=[])
        issues = _rule_57_finite_delays(graph)
        assert [i.rule_id for i in issues] == ["rule-57"]
        assert issues[0].location == "graph.nodes.a.min_start_delay_us"

    def test_finite_delays_pass(self) -> None:
        graph = GraphRecord(
            nodes={"a": _node(), "b": _node()},
            edges=[
                StaticEdge(
                    source="a",
                    target="b",
                    delay_after_predecessor_us=1.0,
                    min_start_delay_us=2.0,
                )
            ],
        )
        assert _rule_57_finite_delays(graph) == []


class TestRule13DefaultManualWarning:
    """G11: an adapter-emitted graph still carrying the default tool 'manual'
    is treated as unstamped and warned about."""

    def test_adapter_source_with_default_manual_warns(self) -> None:
        graph = GraphRecord(
            provenance=ProvenanceSpec(source="weka_trace")  # tool defaults 'manual'
        )
        issues = _rule_13_provenance_tool(graph)
        assert [i.rule_id for i in issues] == ["rule-13"]
        assert issues[0].severity is ValidationSeverity.WARNING

    def test_adapter_source_with_stamped_tool_passes(self) -> None:
        graph = GraphRecord(
            provenance=ProvenanceSpec(source="weka_trace", tool="aiperf-weka-trie/1")
        )
        assert _rule_13_provenance_tool(graph) == []

    def test_adapter_source_with_empty_tool_still_errors(self) -> None:
        graph = GraphRecord(provenance=ProvenanceSpec(source="weka_trace", tool="  "))
        issues = _rule_13_provenance_tool(graph)
        assert [i.rule_id for i in issues] == ["rule-13"]
        assert issues[0].severity is ValidationSeverity.ERROR

    def test_hand_authored_with_manual_tool_passes(self) -> None:
        assert _rule_13_provenance_tool(GraphRecord()) == []
