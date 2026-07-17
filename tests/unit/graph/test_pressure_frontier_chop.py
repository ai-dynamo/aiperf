# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Handoff frontier chop (``chop_trie_at_frontier``) -- pure rewrite tests.

The extended-warmup handoff resumes PROFILING at each lane's pressure-stage
execution frontier instead of the original t* frontier. The chop drops every
node the pressure stage already executed (the server holds their KV), keeps
inter-survivor edges verbatim (recorded pacing resumes), and re-roots each
chain's frontier from START with a RESIDUAL delay: the recorded gap to the
next turn minus the wall-clock time already spent draining -- so the profiling
handoff ramps instead of bursting.
"""

from __future__ import annotations

import msgspec
import pytest
from pytest import param

from aiperf.dataset.graph.models import (
    ChannelRequirement,
    GraphRecord,
    LlmNode,
    ParsedGraph,
    StaticEdge,
    TraceRecord,
)
from aiperf.timing.snapshot_chop import chop_trie_at_frontier

_S = 1_000_000.0  # one second in microseconds


def _llm(arrival_us: int, inputs: list[ChannelRequirement] | None = None) -> LlmNode:
    return LlmNode(
        prompt=["p"],
        output="out",
        arrival_offset_us=arrival_us,
        inputs=list(inputs or []),
    )


def _parsed(nodes: dict[str, LlmNode], edges: list[StaticEdge]) -> ParsedGraph:
    graph = GraphRecord(nodes=nodes, edges=edges)
    return ParsedGraph(graph=graph, traces=[TraceRecord(id="t-1")])


def _chain() -> ParsedGraph:
    """START -> a(1s) -> b(2s) -> c(3s), 0.7s / 0.5s end-to-start gaps."""
    nodes = {
        "a": _llm(int(1 * _S)),
        "b": _llm(int(2 * _S)),
        "c": _llm(int(3 * _S)),
    }
    edges = [
        StaticEdge(source="START", target="a", min_start_delay_us=1 * _S),
        StaticEdge(source="a", target="b", delay_after_predecessor_us=0.7 * _S),
        StaticEdge(source="b", target="c", delay_after_predecessor_us=0.5 * _S),
    ]
    return _parsed(nodes, edges)


def test_chop_trie_at_frontier_executed_dropped_and_frontier_rerooted():
    """executed={a}: survivors {b, c}; b re-roots from START; b->c kept verbatim."""
    out = chop_trie_at_frontier(
        _chain(),
        t_star_us=0.0,
        executed=frozenset({"a"}),
        return_wall_us={"a": 100.0},
        drain_end_wall_us=100.0,
    )
    assert set(out.graph.nodes) == {"b", "c"}
    pairs = {(e.source, e.target) for e in out.graph.edges}
    assert pairs == {("START", "b"), ("b", "c")}
    bc = next(e for e in out.graph.edges if e.source == "b")
    assert bc.delay_after_predecessor_us == 0.5 * _S


@pytest.mark.parametrize(
    "elapsed_us,expected_residual_us",
    [
        param(0.3 * _S, 0.4 * _S, id="partial_credit"),
        param(0.7 * _S, 0.0, id="fully_elapsed"),
        param(2.0 * _S, 0.0, id="over_elapsed_clamps_to_zero"),
        param(0.0, 0.7 * _S, id="no_elapsed_full_recorded_delay"),
    ],
)  # fmt: skip
def test_chop_trie_at_frontier_residual_credits_drain_elapsed(
    elapsed_us: float, expected_residual_us: float
):
    """Residual = recorded gap minus wall time already waited since pred return."""
    drain_end = 1_000_000.0
    out = chop_trie_at_frontier(
        _chain(),
        t_star_us=0.0,
        executed=frozenset({"a"}),
        return_wall_us={"a": drain_end - elapsed_us},
        drain_end_wall_us=drain_end,
    )
    reroot = next(e for e in out.graph.edges if e.source == "START")
    assert reroot.target == "b"
    assert reroot.min_start_delay_us == pytest.approx(expected_residual_us)


def test_chop_trie_at_frontier_unanchored_frontier_bursts():
    """A dropped pred with no recorded return wall anchors nothing: residual 0."""
    out = chop_trie_at_frontier(
        _chain(),
        t_star_us=0.0,
        executed=frozenset({"a"}),
        return_wall_us={},
        drain_end_wall_us=500.0,
    )
    reroot = next(e for e in out.graph.edges if e.source == "START")
    assert reroot.min_start_delay_us == 0.0


def test_chop_trie_at_frontier_pre_tstar_dropped_without_execution():
    """t* drops pre-t* nodes even when not in executed; boundary wall anchors."""
    out = chop_trie_at_frontier(
        _chain(),
        t_star_us=1.5 * _S,  # a is pre-t* history; nothing executed in pressure
        executed=frozenset(),
        return_wall_us={"a": 900_000.0},  # priming return, merged by the caller
        drain_end_wall_us=1_000_000.0,
    )
    assert set(out.graph.nodes) == {"b", "c"}
    reroot = next(e for e in out.graph.edges if e.source == "START")
    # recorded 0.7s minus 0.1s elapsed since the priming return
    assert reroot.min_start_delay_us == pytest.approx(0.6 * _S)


def test_chop_trie_at_frontier_and_fan_in_inputs_rescoped():
    """A survivor's AND-fan-in requirement on a dropped pred's channel is removed."""
    nodes = {
        "a": _llm(0),
        "b": _llm(int(1 * _S)),
        "j": _llm(
            int(2 * _S),
            inputs=[
                ChannelRequirement(channel="a_out", count=1),
                ChannelRequirement(channel="b_out", count=1),
            ],
        ),
    }
    edges = [
        StaticEdge(source="START", target="a"),
        StaticEdge(source="START", target="b"),
        StaticEdge(source="a", target="j", delay_after_predecessor_us=0.2 * _S),
        StaticEdge(source="b", target="j", delay_after_predecessor_us=0.1 * _S),
    ]
    out = chop_trie_at_frontier(
        _parsed(nodes, edges),
        t_star_us=0.0,
        executed=frozenset({"a"}),
        return_wall_us={"a": 0.0},
        drain_end_wall_us=0.0,
    )
    j = out.graph.nodes["j"]
    assert [req.channel for req in j.inputs] == ["b_out"]
    # j keeps its surviving pred b (no re-root)...
    pairs = {(e.source, e.target) for e in out.graph.edges}
    assert ("b", "j") in pairs
    assert ("START", "j") not in pairs
    # ...but the dropped a->j binding residual (0.2s, zero drain elapsed) is
    # FOLDED into j's node-level gate instead of silently discarded.
    assert j.min_start_delay_us == pytest.approx(0.2 * _S)


def test_chop_trie_at_frontier_kept_pred_residual_debits_and_caps():
    """The folded node-level residual uses the same debit + cap math as re-roots."""
    nodes = {
        "a": _llm(0),
        "b": _llm(int(1 * _S)),
        "j": _llm(int(2 * _S)),
    }
    edges = [
        StaticEdge(source="START", target="a"),
        StaticEdge(source="START", target="b"),
        StaticEdge(source="a", target="j", delay_after_predecessor_us=300 * _S),
        StaticEdge(source="b", target="j", delay_after_predecessor_us=0.1 * _S),
    ]
    out = chop_trie_at_frontier(
        _parsed(nodes, edges),
        t_star_us=0.0,
        executed=frozenset({"a"}),
        return_wall_us={"a": 100.0},
        drain_end_wall_us=100.0,
        residual_cap_us=60 * _S,
    )
    assert out.graph.nodes["j"].min_start_delay_us == pytest.approx(60 * _S)


def test_chop_trie_at_frontier_kept_pred_residual_max_combines_with_node_delay():
    """An existing node-level min_start_delay_us survives when larger than the fold."""
    j = _llm(int(2 * _S))
    nodes = {
        "a": _llm(0),
        "b": _llm(int(1 * _S)),
        "j": msgspec.structs.replace(j, min_start_delay_us=5 * _S),
    }
    edges = [
        StaticEdge(source="START", target="a"),
        StaticEdge(source="START", target="b"),
        StaticEdge(source="a", target="j", delay_after_predecessor_us=0.2 * _S),
        StaticEdge(source="b", target="j", delay_after_predecessor_us=0.1 * _S),
    ]
    out = chop_trie_at_frontier(
        _parsed(nodes, edges),
        t_star_us=0.0,
        executed=frozenset({"a"}),
        return_wall_us={"a": 0.0},
        drain_end_wall_us=0.0,
    )
    assert out.graph.nodes["j"].min_start_delay_us == pytest.approx(5 * _S)


def test_chop_trie_at_frontier_full_execution_yields_empty_graph():
    out = chop_trie_at_frontier(
        _chain(),
        t_star_us=0.0,
        executed=frozenset({"a", "b", "c"}),
        return_wall_us={},
        drain_end_wall_us=0.0,
    )
    assert out.graph.nodes == {}
    assert out.graph.edges == []


def test_chop_trie_at_frontier_nothing_executed_keeps_nodes_zeroes_lead():
    """executed empty + t*=0: full node set; chain head re-roots at 0 lead.

    Pressure fires everything ASAP, so a not-yet-started template's recorded
    absolute lead is considered consumed; profiling resumes it immediately.
    """
    out = chop_trie_at_frontier(
        _chain(),
        t_star_us=0.0,
        executed=frozenset(),
        return_wall_us={},
        drain_end_wall_us=0.0,
    )
    assert set(out.graph.nodes) == {"a", "b", "c"}
    reroot = next(e for e in out.graph.edges if e.source == "START")
    assert reroot.target == "a"
    assert reroot.min_start_delay_us == 0.0


def test_chop_trie_at_frontier_start_anchored_delay_contributes_zero():
    """Start-anchored recorded delays never anchor a residual.

    The ledger wall is the predecessor's RETURN; debiting a dispatch-anchored
    delay from a return-anchored elapsed would over-delay by the pred's live
    service time (anchor mismatch), so start-anchored edges burst instead --
    the same 0.0 offset agentx gives pending handoff turns.
    """
    nodes = {"a": _llm(0), "b": _llm(int(1 * _S))}
    edges = [
        StaticEdge(source="START", target="a"),
        StaticEdge(source="a", target="b", delay_after_predecessor_start_us=5 * _S),
    ]
    out = chop_trie_at_frontier(
        _parsed(nodes, edges),
        t_star_us=0.0,
        executed=frozenset({"a"}),
        return_wall_us={"a": 100.0},
        drain_end_wall_us=100.0,
    )
    reroot = next(e for e in out.graph.edges if e.source == "START")
    assert reroot.min_start_delay_us == 0.0


def test_chop_trie_at_frontier_residual_clamped_to_cap():
    """A recorded gap beyond the cap resumes at the cap, not the full gap."""
    nodes = {"a": _llm(0), "b": _llm(int(1 * _S))}
    edges = [
        StaticEdge(source="START", target="a"),
        StaticEdge(source="a", target="b", delay_after_predecessor_us=300 * _S),
    ]
    out = chop_trie_at_frontier(
        _parsed(nodes, edges),
        t_star_us=0.0,
        executed=frozenset({"a"}),
        return_wall_us={"a": 100.0},
        drain_end_wall_us=100.0,
        residual_cap_us=60 * _S,
    )
    reroot = next(e for e in out.graph.edges if e.source == "START")
    assert reroot.min_start_delay_us == pytest.approx(60 * _S)
