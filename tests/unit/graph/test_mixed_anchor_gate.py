# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R2/N7 -- a non-solo start-anchored in-edge is rejected loudly at graph load, never silently mis-gated at runtime."""

from __future__ import annotations

import asyncio
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
from aiperf.graph.analysis.timeline import elaborate_trace
from aiperf.graph.scheduler import Scheduler


def _llm(output: str) -> LlmNode:
    return LlmNode(prompt=[f"@{output}"], output=output)


def _mixed_graph() -> GraphRecord:
    """Node ``d`` fans in one start-anchored edge (a->d) and one completion edge (b->d)."""
    return GraphRecord(
        nodes={"a": _llm("a"), "b": _llm("b"), "d": _llm("d")},
        edges=[
            StaticEdge(source="START", target="a"),
            StaticEdge(source="a", target="b", delay_after_predecessor_us=0.0),
            StaticEdge(
                source="a", target="d", delay_after_predecessor_start_us=1_000.0
            ),
            StaticEdge(source="b", target="d", delay_after_predecessor_us=0.0),
        ],
        state={},
    )


@pytest.mark.parametrize("delay_us", [0.0, 1_000.0])
def test_scheduler_rejects_start_anchored_virtual_start_edge(
    delay_us: float,
) -> None:
    """Virtual START never dispatches, so its start-anchored successor cannot fire."""
    graph = GraphRecord(
        nodes={"a": _llm("a")},
        edges=[
            StaticEdge(
                source="START",
                target="a",
                delay_after_predecessor_start_us=delay_us,
            )
        ],
        state={},
    )

    with pytest.raises(NotImplementedError) as exc_info:
        Scheduler(graph)

    message = str(exc_info.value)
    assert message.startswith("edge 'START' -> 'a': ")
    assert "delay_after_predecessor_start_us" in message


def test_scheduler_rejects_mixed_anchor_fan_in_naming_node_and_edges() -> None:
    with pytest.raises(NotImplementedError) as exc_info:
        Scheduler(_mixed_graph())
    msg = str(exc_info.value)
    assert msg.startswith("node 'd': ")
    assert "mixed-anchor fan-in" in msg
    assert "'a' -> 'd'" in msg  # the start-anchored edge
    assert "'b' -> 'd'" in msg  # the completion edge
    # The remediation must NOT steer users toward uniform start-anchored
    # fan-in (equally unsupported); a start anchor must be the only in-edge.
    assert "same anchor kind" not in msg
    assert "ONLY in-edge" in msg


def test_scheduler_rejects_uniform_double_start_anchored_fan_in() -> None:
    """TWO start-anchored in-edges on one target are rejected as multi-start-anchored fan-in."""
    graph = GraphRecord(
        nodes={"a": _llm("a"), "b": _llm("b"), "d": _llm("d")},
        edges=[
            StaticEdge(source="START", target="a"),
            StaticEdge(source="a", target="b", delay_after_predecessor_us=0.0),
            StaticEdge(
                source="a", target="d", delay_after_predecessor_start_us=1_000.0
            ),
            StaticEdge(
                source="b", target="d", delay_after_predecessor_start_us=2_000.0
            ),
        ],
        state={},
    )
    # ``_compute_firing_gate_us`` would silently drop the not-yet-dispatched
    # second anchor and surface later as a spurious "cycle detected".
    with pytest.raises(NotImplementedError) as exc_info:
        Scheduler(graph)
    msg = str(exc_info.value)
    assert msg.startswith("node 'd': ")
    assert "multi-start-anchored fan-in" in msg
    assert "'a' -> 'd'" in msg
    assert "'b' -> 'd'" in msg
    assert "ONLY in-edge" in msg


def _construct_executor(graph: GraphRecord) -> None:
    from aiperf.graph.executor import TraceExecutor

    TraceExecutor(ParsedGraph(graph=graph, traces=[TraceRecord(id="t")]))


def _run_static_analysis(graph: GraphRecord) -> None:
    parsed = ParsedGraph(graph=graph, traces=[TraceRecord(id="t")])
    elaborate_trace(parsed, parsed.traces[0])


@pytest.mark.parametrize(
    "entry_point",
    [
        param(_construct_executor, id="executor_construction"),
        param(_run_static_analysis, id="elaborate_trace_static_analysis"),
    ],
)  # fmt: skip
def test_mixed_anchor_fan_in_rejected_at_every_scheduler_entry_point(
    entry_point: Callable[[GraphRecord], None],
) -> None:
    """Both the executor and static analysis build the Scheduler, so a mixed-anchor graph fails at load instead of firing early and spurious-cycling later."""
    with pytest.raises(NotImplementedError, match="mixed-anchor fan-in"):
        entry_point(_mixed_graph())


@pytest.mark.parametrize(
    "entry_point",
    [
        param(_construct_executor, id="executor_construction"),
        param(_run_static_analysis, id="elaborate_trace_static_analysis"),
    ],
)  # fmt: skip
def test_start_anchored_virtual_start_rejected_at_every_scheduler_entry_point(
    entry_point: Callable[[GraphRecord], None],
) -> None:
    """Every Scheduler consumer rejects a zero-delay start anchor on START."""
    graph = GraphRecord(
        nodes={"a": _llm("a")},
        edges=[
            StaticEdge(
                source="START",
                target="a",
                delay_after_predecessor_start_us=0.0,
            )
        ],
        state={},
    )

    with pytest.raises(NotImplementedError, match="virtual START"):
        entry_point(graph)


def test_start_edge_plus_start_anchored_edge_is_mixed_too() -> None:
    """A START in-edge counts as completion-kind, so pairing it with a start anchor is mixed."""
    graph = GraphRecord(
        nodes={"a": _llm("a"), "c": _llm("c")},
        edges=[
            StaticEdge(source="START", target="a"),
            StaticEdge(source="START", target="c"),
            StaticEdge(source="a", target="c", delay_after_predecessor_start_us=500.0),
        ],
        state={},
    )
    with pytest.raises(NotImplementedError, match="node 'c': mixed-anchor fan-in"):
        Scheduler(graph)


def test_supported_anchor_shapes_still_accepted() -> None:
    """All-completion fan-in and a SOLO start-anchored in-edge construct cleanly."""
    all_completion = GraphRecord(
        nodes={"a": _llm("a"), "b": _llm("b"), "d": _llm("d")},
        edges=[
            StaticEdge(source="START", target="a"),
            StaticEdge(source="START", target="b"),
            StaticEdge(source="a", target="d", delay_after_predecessor_us=0.0),
            StaticEdge(source="b", target="d", delay_after_predecessor_us=0.0),
        ],
        state={},
    )
    solo_start_anchored = GraphRecord(
        nodes={"a": _llm("a"), "c": _llm("c")},
        edges=[
            StaticEdge(source="START", target="a"),
            StaticEdge(source="a", target="c", delay_after_predecessor_start_us=500.0),
        ],
        state={},
    )
    Scheduler(all_completion)
    Scheduler(solo_start_anchored)


class _OrderedIssuer:
    """Stub issuer that holds ``c``'s dispatch until ``b`` has completed."""

    def __init__(self) -> None:
        self.b_dispatched = asyncio.Event()

    async def dispatch(
        self, node: object, request: object, ctx: object, **kw: object
    ) -> tuple[str, int | None, float | None, float | None]:
        nid = request.node_id  # type: ignore[attr-defined]
        if nid == "c":
            await self.b_dispatched.wait()
            for _ in range(3):  # let b's _fire task run to completion
                await asyncio.sleep(0)
        if nid == "b":
            self.b_dispatched.set()
        return f"ok::{nid}", None, None, None


@pytest.mark.asyncio
async def test_done_node_reschedule_error_mentions_mixed_anchor_cause() -> None:
    """A re-schedule of a still-reachable done node blames mixed-anchor fan-in in its cycle error."""
    from aiperf.dataset.graph.models import ChannelRequirement, ChannelSpec
    from aiperf.graph.executor import TraceExecutor

    graph = GraphRecord(
        nodes={
            "a": _llm("a"),
            "c": _llm("c"),
            "b": LlmNode(
                prompt=["@a"],
                output="b",
                inputs=[ChannelRequirement(channel="a", count=1)],
            ),
        },
        edges=[
            StaticEdge(source="START", target="a"),
            StaticEdge(source="START", target="c"),
            StaticEdge(source="a", target="b"),
            StaticEdge(source="c", target="b"),
        ],
        state={"a": ChannelSpec(), "b": ChannelSpec(), "c": ChannelSpec()},
    )
    parsed = ParsedGraph(graph=graph, traces=[TraceRecord(id="t")])
    executor = TraceExecutor(parsed, credit_issuer=_OrderedIssuer())

    with pytest.raises(ExceptionGroup) as exc_info:
        await executor.run(parsed.traces[0])
    cycle_errors = [e for e in exc_info.value.exceptions if isinstance(e, RuntimeError)]
    assert cycle_errors, exc_info.value.exceptions
    msg = str(cycle_errors[0])
    assert "cycle detected" in msg
    assert "mixed-anchor fan-in" in msg
