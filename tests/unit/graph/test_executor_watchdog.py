# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EXECUTOR_WATCHDOG_TIMEOUT converts a firing-loop deadlock into a failure.

DISPATCH_TIMEOUT only bounds a node that ISSUED a credit; a node wedged on an
unsatisfiable channel input never dispatched and has no per-dispatch guard. The
opt-in executor watchdog wraps `TraceExecutor.run` in `asyncio.wait_for` so such
a wedge raises `TimeoutError` instead of hanging. Default None preserves the
faithful unbounded idle-gap replay, so this test explicitly opts in.
"""

from __future__ import annotations

import asyncio

import pytest

from aiperf.common.environment import Environment
from aiperf.dataset.graph.models import (
    ChannelRequirement,
    ChannelSpec,
    GraphRecord,
    LlmNode,
    ParsedGraph,
    StaticEdge,
    TraceRecord,
)
from aiperf.graph.executor import TraceExecutor


async def _run_deadlock(executor: TraceExecutor, trace: TraceRecord) -> BaseException:
    """Drive a wedged run and return the exception it surfaced.

    A wedged ``run`` surfaces a body exception directly or wrapped in an
    ``ExceptionGroup`` depending on how the internal cancellation unwinds, so
    the caller inspects the returned exception rather than relying on the exact
    top-level type.
    """
    try:
        async with asyncio.TaskGroup():
            await executor.run(trace)
    except BaseException as exc:  # noqa: BLE001 - test inspects the surfaced type
        return exc
    raise AssertionError("run completed without raising; expected a deadlock")


def _has_timeout(exc: BaseException) -> bool:
    """True when ``exc`` is (or wraps, via ExceptionGroup) a TimeoutError."""
    if isinstance(exc, TimeoutError):
        return True
    nested = getattr(exc, "exceptions", None)
    if nested is not None:
        return any(_has_timeout(sub) for sub in nested)
    return False


def _mutual_deadlock_graph() -> ParsedGraph:
    """Two entry LLMs each AND-fan-in gated on the OTHER's output channel.

    A awaits ``b`` (produced only by B) and B awaits ``a`` (produced only by A),
    so neither ever becomes input-ready: a genuine liveness deadlock the orphan
    check cannot break (each channel has a live, never-completing producer).
    """
    nodes: dict[str, object] = {
        "A": LlmNode(
            prompt=["@a"], output="a", inputs=[ChannelRequirement(channel="b", count=1)]
        ),
        "B": LlmNode(
            prompt=["@b"], output="b", inputs=[ChannelRequirement(channel="a", count=1)]
        ),
    }
    edges = [
        StaticEdge(source="START", target="A"),
        StaticEdge(source="START", target="B"),
    ]
    graph = GraphRecord(
        nodes=nodes,
        edges=edges,
        state={"a": ChannelSpec(), "b": ChannelSpec()},
    )
    return ParsedGraph(graph=graph, traces=[TraceRecord(id="t")])


@pytest.mark.asyncio
async def test_watchdog_times_out_a_wedged_firing_loop(monkeypatch):
    """With the watchdog set, run() self-bounds the deadlock with a TimeoutError."""
    monkeypatch.setattr(
        Environment.GRAPH, "EXECUTOR_WATCHDOG_TIMEOUT", 0.2, raising=False
    )
    parsed = _mutual_deadlock_graph()
    executor = TraceExecutor(parsed)
    exc = await _run_deadlock(executor, parsed.traces[0])
    assert _has_timeout(exc)


@pytest.mark.asyncio
async def test_no_watchdog_by_default_does_not_self_bound(monkeypatch):
    """With the default (None) run() does NOT self-bound; only an external
    wait_for ends the deadlock (proving the watchdog, not some other guard, is
    what produced the timeout in the test above)."""
    monkeypatch.setattr(
        Environment.GRAPH, "EXECUTOR_WATCHDOG_TIMEOUT", None, raising=False
    )
    parsed = _mutual_deadlock_graph()
    executor = TraceExecutor(parsed)

    async def _drive() -> None:
        async with asyncio.TaskGroup():
            await executor.run(parsed.traces[0])

    # run() itself never returns; only our external wait_for breaks the wedge.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(_drive(), timeout=0.3)
