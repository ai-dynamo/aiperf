# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct unit cover for the graph analysis core (elaborate_trace / compute_snapshot)."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pytest import param

from aiperf.dataset.graph.models import (
    ChannelRequirement,
    ChannelSpec,
    ChannelType,
    GraphRecord,
    LlmNode,
    ParsedGraph,
    ReducerName,
    StaticEdge,
    TraceRecord,
)
from aiperf.graph.analysis.snapshot import compute_snapshot
from aiperf.graph.analysis.timeline import GraphCycleError, elaborate_trace

# Graph builders stay module-level functions (not conftest fixtures) because
# @parametrize params reference them directly.


def _llm(output: str, *, offset: int | None = None) -> LlmNode:
    return LlmNode(prompt=[f"@{output}"], output=output, arrival_offset_us=offset)


def _parsed(
    nodes: dict[str, object],
    edges: list[StaticEdge],
    *,
    state: dict[str, ChannelSpec] | None = None,
) -> ParsedGraph:
    graph = GraphRecord(nodes=nodes, edges=edges, state=state or {})
    return ParsedGraph(graph=graph, traces=[TraceRecord(id="t")])


def _linear_chain_parsed() -> ParsedGraph:
    """START->A->B->C with one arrival every 100us."""
    return _parsed(
        {
            "A": _llm("a", offset=0),
            "B": _llm("b", offset=100),
            "C": _llm("c", offset=200),
        },
        [
            StaticEdge(source="START", target="A"),
            StaticEdge(source="A", target="B"),
            StaticEdge(source="B", target="C"),
        ],
    )


def _start_anchored_parsed() -> ParsedGraph:
    """The R1 review graph: START->a; a->b completion; a->c start-anchored; c->d completion."""
    return _parsed(
        {
            "a": _llm("a", offset=0),
            "b": _llm("b", offset=1_000_000),
            "c": _llm("c", offset=2_500_000),
            "d": _llm("d", offset=9_000_000),
        },
        [
            StaticEdge(source="START", target="a"),
            StaticEdge(source="a", target="b", delay_after_predecessor_us=0.0),
            StaticEdge(
                source="a", target="c", delay_after_predecessor_start_us=2_500_000.0
            ),
            StaticEdge(source="c", target="d", delay_after_predecessor_us=0.0),
        ],
    )


@pytest.mark.parametrize(
    ("build", "t_star_us", "warmup", "profiled_offsets"),
    [
        param(
            _linear_chain_parsed,
            100,
            {"A"},
            {"B": 0, "C": 100},
            id="linear-chain-inclusive-tstar-boundary",
        ),
        param(
            _start_anchored_parsed,
            2_500_000,
            {"a", "b"},
            {"c": 0, "d": 6_500_000},
            id="start-anchored-subtree-partition",
        ),
    ],
)  # fmt: skip
def test_compute_snapshot_partitions_at_t_star(
    build: Callable[[], ParsedGraph],
    t_star_us: int,
    warmup: set[str],
    profiled_offsets: dict[str, int],
) -> None:
    """arrival < t* is warmup; arrival >= t* is profiled, rebased so t* dispatches at 0."""
    parsed = build()

    snap = compute_snapshot(parsed, parsed.traces[0], t_star_us=t_star_us)

    assert {sf.firing.node_id for sf in snap.warmup} == warmup
    profiled = {sf.firing.node_id: sf for sf in snap.profiled}
    assert set(profiled) == set(profiled_offsets)
    assert {
        nid: sf.dispatch_offset_us for nid, sf in profiled.items()
    } == profiled_offsets


def test_elaborate_trace_depth_cap_raises_graph_cycle_error() -> None:
    """A firing count over depth_cap raises GraphCycleError."""
    parsed = _parsed(
        {"A": _llm("a"), "B": _llm("b")},
        [
            StaticEdge(source="START", target="A"),
            StaticEdge(source="A", target="B"),
        ],
    )
    with pytest.raises(GraphCycleError):
        elaborate_trace(parsed, parsed.traces[0], depth_cap=1)


def test_elaborate_trace_true_cycle_does_not_trip_the_stall_raise() -> None:
    """A genuine back-edge off START is NOT what the stall raise detects.

    Each node enters ``scheduled`` at most once, so ``A -> B -> A`` fires both
    nodes and drains ``pending``. Cycles are rejected upstream (the dynamo adapter's
    parent_link cycle guard) and by the executor's ``_schedule`` guard; the exit
    raise is a fan-in deadlock detector,
    which is why its message says so.
    """
    parsed = _parsed(
        {"A": _llm("A"), "B": _llm("B")},
        [
            StaticEdge(source="START", target="A"),
            StaticEdge(source="A", target="B"),
            StaticEdge(source="B", target="A"),
        ],
    )

    timeline = elaborate_trace(parsed, parsed.traces[0])

    assert [f.node_id for f in timeline.firings] == ["A", "B"]


def test_elaborate_trace_stall_raise_names_the_unsatisfied_nodes() -> None:
    """An AND-join needing more arrivals than reachable writers supply raises.

    The reader wants two arrivals on ``c`` but its only ancestor writer is
    ``w1``; the second writer ``w2`` is a DESCENDANT, so the count can never be
    met. The real ``VersionedChannelStore`` blocks forever on this same graph, so
    raising here is symmetric with runtime rather than a false positive.
    """
    reader = LlmNode(
        prompt=["@r"],
        output="r",
        inputs=[ChannelRequirement(channel="c", count=2)],
    )
    parsed = _parsed(
        {"w1": _llm("c"), "reader": reader, "w2": _llm("c")},
        [
            StaticEdge(source="START", target="w1"),
            StaticEdge(source="w1", target="reader"),
            StaticEdge(source="reader", target="w2"),
        ],
        state={
            "c": ChannelSpec(
                type=ChannelType.MESSAGES, reducer=ReducerName.ADD_MESSAGES
            ),
            "r": ChannelSpec(),
        },
    )
    with pytest.raises(GraphCycleError) as exc_info:
        elaborate_trace(parsed, parsed.traces[0])

    msg = str(exc_info.value)
    assert "reader" in msg
    # A cycle diagnosis would send readers hunting a back-edge that isn't there.
    assert "fan-in deadlock" in msg


def test_elaborate_trace_follows_start_anchored_edges() -> None:
    """R1: the whole subtree under a start-anchored edge fires in the dry-run."""
    parsed = _start_anchored_parsed()

    timeline = elaborate_trace(parsed, parsed.traces[0])

    # Consulting only ``successors_after`` stopped the timeline at ['a', 'b'], so
    # the anchored child and its completion successor never fired -- duration
    # under-measured and snapshot planning misclassified the subtree.
    assert [f.node_id for f in timeline.firings] == ["a", "b", "c", "d"]
    assert timeline.duration_us() == 9_000_000


def test_elaborate_trace_pending_fan_in_waits_for_late_arrival() -> None:
    """A scheduled node whose fan-in is unsatisfied stays pending, never dropped."""
    # Mirrors the runtime's parked ``await_inputs``: ``join`` is scheduled by a
    # start-anchored edge from ``a`` but requires ``b``'s output channel.
    join = LlmNode(
        prompt=["@j"],
        output="j",
        inputs=[ChannelRequirement(channel="b", count=1)],
    )
    parsed = _parsed(
        {
            "a": _llm("a", offset=0),
            "b": _llm("b", offset=100),
            "join": join,
        },
        [
            StaticEdge(source="START", target="a"),
            StaticEdge(source="a", target="b", delay_after_predecessor_us=0.0),
            StaticEdge(
                source="a", target="join", delay_after_predecessor_start_us=50.0
            ),
        ],
        state={"a": ChannelSpec(), "b": ChannelSpec(), "j": ChannelSpec()},
    )

    timeline = elaborate_trace(parsed, parsed.traces[0])

    order = [f.node_id for f in timeline.firings]
    assert set(order) == {"a", "b", "join"}
    assert order.index("join") > order.index("b")
