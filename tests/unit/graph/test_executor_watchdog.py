# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EXECUTOR_WATCHDOG_TIMEOUT converts a firing-loop deadlock into a failure."""

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
    """Drive a wedged run and return the exception it surfaced."""
    try:
        async with asyncio.TaskGroup():
            await executor.run(trace)
    except BaseException as exc:  # noqa: BLE001 - test inspects the surfaced type
        return exc
    raise AssertionError("run completed without raising; expected a deadlock")


def _has_timeout(exc: BaseException) -> bool:
    """True when ``exc`` is (or wraps, via ExceptionGroup) a TimeoutError."""
    return _find_timeout(exc) is not None


def _find_timeout(exc: BaseException) -> TimeoutError | None:
    """The first TimeoutError in ``exc`` or its (possibly nested) ExceptionGroup."""
    if isinstance(exc, TimeoutError):
        return exc
    for sub in getattr(exc, "exceptions", ()) or ():
        found = _find_timeout(sub)
        if found is not None:
            return found
    return None


def _mutual_deadlock_graph() -> ParsedGraph:
    """Two entry LLMs each AND-fan-in gated on the OTHER's output channel (never satisfiable)."""
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
async def test_watchdog_times_out_a_wedged_firing_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the watchdog set, run() self-bounds the deadlock with a TimeoutError."""
    monkeypatch.setattr(
        Environment.GRAPH, "EXECUTOR_WATCHDOG_TIMEOUT", 0.2, raising=False
    )
    parsed = _mutual_deadlock_graph()
    executor = TraceExecutor(parsed)
    exc = await _run_deadlock(executor, parsed.traces[0])
    assert _has_timeout(exc)


@pytest.mark.asyncio
async def test_watchdog_message_names_trace_and_pending_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The surfaced TimeoutError attributes the wedge -- trace id, pending nodes, env var."""
    # A bare TimeoutError names nothing, which defeats the guard's purpose.
    monkeypatch.setattr(
        Environment.GRAPH, "EXECUTOR_WATCHDOG_TIMEOUT", 0.2, raising=False
    )
    parsed = _mutual_deadlock_graph()
    executor = TraceExecutor(parsed)
    exc = await _run_deadlock(executor, parsed.traces[0])
    timeout = _find_timeout(exc)
    assert timeout is not None
    message = str(timeout)
    assert "'t'" in message
    assert "'A'" in message and "'B'" in message
    assert "AIPERF_GRAPH_EXECUTOR_WATCHDOG_TIMEOUT" in message


@pytest.mark.asyncio
async def test_no_watchdog_by_default_does_not_self_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the default (None) the wedged run never self-bounds -- only an external wait_for ends it."""
    # A/B counterpart to the test above: it proves the watchdog, and not some
    # other guard, is what produced that TimeoutError.
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
