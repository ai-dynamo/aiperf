# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R2/N7 -- a non-solo start-anchored in-edge is rejected loudly at graph load.

Any fan-in involving a start-anchored in-edge was half-supported: the runtime
fired the target at the start anchor (silently ignoring the completion
predecessor's recorded ordering, or silently dropping a not-yet-dispatched
second start anchor from the firing gate), then died with a spurious "cycle
detected" when the other predecessor finished/dispatched. No shipped lowering
emits either shape (``apply_start_anchors`` replaces a node's WHOLE in-edge
set with exactly one edge), so ``Scheduler`` construction now rejects any
start-anchored in-edge that is not its target's ONLY in-edge, with a clear
``NotImplementedError`` naming the node and the offending edges.
"""

from __future__ import annotations

import asyncio

import pytest

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
    """START->a; a->d (completion); a->c (start-anchored); c->d (completion) is
    fine -- the mix is on ``d``: b->d completion + a->d start-anchored."""
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


def test_scheduler_rejects_mixed_anchor_fan_in_naming_node_and_edges():
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


def test_scheduler_rejects_uniform_double_start_anchored_fan_in():
    """TWO start-anchored in-edges on one target are rejected too.

    The runtime half-supports this shape: the target fires at its FIRST anchor
    parent's dispatch (`_compute_firing_gate_us` silently drops the
    not-yet-dispatched second anchor), then the second anchor parent's dispatch
    re-schedules the DONE target into the cycle guard (spurious "cycle
    detected") and the whole trace unwinds.
    """
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
    with pytest.raises(NotImplementedError) as exc_info:
        Scheduler(graph)
    msg = str(exc_info.value)
    assert msg.startswith("node 'd': ")
    assert "multi-start-anchored fan-in" in msg
    assert "'a' -> 'd'" in msg
    assert "'b' -> 'd'" in msg
    assert "ONLY in-edge" in msg


def test_executor_construction_rejects_mixed_anchor_fan_in():
    """The executor builds its Scheduler at construction, so a mixed-anchor
    graph fails at load instead of firing early + spurious-cycling later."""
    from aiperf.graph.executor import TraceExecutor

    parsed = ParsedGraph(graph=_mixed_graph(), traces=[TraceRecord(id="t")])
    with pytest.raises(NotImplementedError, match="mixed-anchor fan-in"):
        TraceExecutor(parsed)


def test_elaborate_trace_rejects_mixed_anchor_fan_in():
    """Static analysis shares the Scheduler, so it rejects the shape too."""
    parsed = ParsedGraph(graph=_mixed_graph(), traces=[TraceRecord(id="t")])
    with pytest.raises(NotImplementedError, match="mixed-anchor fan-in"):
        elaborate_trace(parsed, parsed.traces[0])


def test_start_edge_plus_start_anchored_edge_is_mixed_too():
    """A START in-edge is completion-kind (entry scheduling): mixing it with a
    start-anchored in-edge re-schedules the fired entry node the same way."""
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


def test_supported_anchor_shapes_still_accepted():
    """All-completion fan-in and a SOLO start-anchored in-edge construct
    cleanly -- the shapes shipped lowerings actually emit
    (``apply_start_anchors`` gives a start-anchored node exactly one in-edge).
    """
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
    ) -> str:
        nid = request.node_id  # type: ignore[attr-defined]
        if nid == "c":
            await self.b_dispatched.wait()
            for _ in range(3):  # let b's _fire task run to completion
                await asyncio.sleep(0)
        if nid == "b":
            self.b_dispatched.set()
        return f"ok::{nid}"


@pytest.mark.asyncio
async def test_done_node_reschedule_error_mentions_mixed_anchor_cause():
    """A still-reachable done-node re-schedule names mixed-anchor fan-in.

    ``b`` gates only on ``a``'s channel, so it fires and completes while its
    second completion predecessor ``c`` is still in flight; ``c``'s completion
    then re-schedules the done ``b`` into the cycle guard. The error message
    must point at mixed-anchor fan-in as a likely cause of this shape.
    """
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
