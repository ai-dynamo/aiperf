# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tool dispatch: the injectable seam, the default sandbox-backed dispatcher,
and the executor wiring that keeps tool steps out of the credit path."""

from __future__ import annotations

import asyncio

import pytest

from aiperf.dataset.graph.models import (
    ChannelSpec,
    GraphRecord,
    LlmNode,
    ParsedGraph,
    StaticEdge,
    ToolNode,
    TraceRecord,
)
from aiperf.graph.executor import TraceExecutor
from aiperf.graph.sandbox.protocols import ToolResult
from aiperf.graph.tool_dispatch.protocols import ToolDispatchResult
from aiperf.graph.tool_dispatch.sandbox_dispatcher import SandboxToolDispatcher


class _StubSandbox:
    """Records commands, returns canned results."""

    def __init__(self) -> None:
        self.opened = False
        self.closed = False
        self.commands: list[str] = []

    async def open(self) -> None:
        self.opened = True

    async def run(self, command: str, timeout_s: float | None = None) -> ToolResult:
        self.commands.append(command)
        return ToolResult(
            stdout=f"out:{command}", returncode=0, duration_s=0.25, timed_out=False
        )

    async def close(self) -> None:
        self.closed = True


class _TimeoutSandbox(_StubSandbox):
    """Times out its first command to exercise dispatcher failure boundaries."""

    async def run(self, command: str, timeout_s: float | None = None) -> ToolResult:
        self.commands.append(command)
        return ToolResult(
            stdout=f"out:{command}", returncode=-1, duration_s=0.25, timed_out=True
        )


class _RecordingDispatcher:
    """A ToolDispatcher that executes nothing -- proves the seam is injectable."""

    def __init__(self) -> None:
        self.opened: list[str] = []
        self.closed: list[str] = []
        self.nodes: list[str] = []

    async def open_trace(self, trace_id: str) -> None:
        self.opened.append(trace_id)

    async def dispatch(self, node, request, ctx) -> ToolDispatchResult:
        self.nodes.append(request.node_id)
        return ToolDispatchResult(
            observation="stubbed", durations_s=[1.5], timed_out=False
        )

    async def close_trace(self, trace_id: str) -> None:
        self.closed.append(trace_id)


class _CountingIssuer:
    """Stands in for the credit pipeline; counts LLM dispatches."""

    def __init__(self) -> None:
        self.dispatches = 0

    async def dispatch(
        self, node, request, ctx, **kwargs
    ) -> tuple[str, int | None, float | None, float | None]:
        self.dispatches += 1
        return "placeholder", None, None, None


def _graph() -> ParsedGraph:
    graph = GraphRecord(
        state={
            "n0_out": ChannelSpec(),
            "t0_out": ChannelSpec(),
            "n1_out": ChannelSpec(),
        },
        nodes={
            "n0": LlmNode(prompt=[], output="n0_out", arrival_offset_us=0),
            "t0": ToolNode(
                commands=["echo one", "echo two"], output="t0_out", arrival_offset_us=1
            ),
            "n1": LlmNode(prompt=[], output="n1_out", arrival_offset_us=2),
        },
        edges=[
            StaticEdge(source="START", target="n0"),
            StaticEdge(source="n0", target="t0"),
            StaticEdge(source="t0", target="n1"),
            StaticEdge(source="n1", target="END"),
        ],
    )
    return ParsedGraph(graph=graph, traces=[TraceRecord(id="trace-1")])


async def test_default_dispatcher_runs_every_command_in_order() -> None:
    parsed = _graph()
    sandbox = _StubSandbox()
    executor = TraceExecutor(
        parsed,
        credit_issuer=_CountingIssuer(),
        tool_dispatcher=SandboxToolDispatcher(lambda trace_id: sandbox),
    )
    await executor.run(parsed.traces[0])
    assert sandbox.commands == ["echo one", "echo two"]


async def test_timed_out_tool_command_skips_the_remaining_node_commands() -> None:
    parsed = _graph()
    sandbox = _TimeoutSandbox()
    executor = TraceExecutor(
        parsed,
        credit_issuer=_CountingIssuer(),
        tool_dispatcher=SandboxToolDispatcher(lambda trace_id: sandbox),
    )

    result = await executor.run(parsed.traces[0])

    assert sandbox.commands == ["echo one"]
    assert result.channels["t0_out"] == "out:echo one"
    assert result.tool_durations_s == [0.25]


async def test_default_dispatcher_joins_command_output_as_the_observation() -> None:
    parsed = _graph()
    executor = TraceExecutor(
        parsed,
        credit_issuer=_CountingIssuer(),
        tool_dispatcher=SandboxToolDispatcher(lambda trace_id: _StubSandbox()),
    )
    result = await executor.run(parsed.traces[0])
    assert result.channels["t0_out"] == "out:echo one\nout:echo two"


async def test_tool_step_does_not_consume_a_credit() -> None:
    """Only the two LlmNodes may reach the issuer; a tool step is not a request."""
    parsed = _graph()
    issuer = _CountingIssuer()
    executor = TraceExecutor(
        parsed,
        credit_issuer=issuer,
        tool_dispatcher=SandboxToolDispatcher(lambda trace_id: _StubSandbox()),
    )
    await executor.run(parsed.traces[0])
    assert issuer.dispatches == 2


async def test_tool_durations_are_accumulated_for_reporting() -> None:
    parsed = _graph()
    executor = TraceExecutor(
        parsed,
        credit_issuer=_CountingIssuer(),
        tool_dispatcher=SandboxToolDispatcher(lambda trace_id: _StubSandbox()),
    )
    result = await executor.run(parsed.traces[0])
    assert result.tool_durations_s == [0.25, 0.25]


async def test_sandbox_is_opened_and_closed_around_the_trace() -> None:
    parsed = _graph()
    sandbox = _StubSandbox()
    executor = TraceExecutor(
        parsed,
        credit_issuer=_CountingIssuer(),
        tool_dispatcher=SandboxToolDispatcher(lambda trace_id: sandbox),
    )
    await executor.run(parsed.traces[0])
    assert sandbox.opened is True
    assert sandbox.closed is True


async def test_an_alternative_dispatcher_replaces_execution_entirely() -> None:
    """The seam is the point: a dispatcher that runs no commands still works.

    This is what makes later modes (live tool calls, recorded-delay replay,
    remote execution) new dispatchers rather than executor changes.
    """
    parsed = _graph()
    dispatcher = _RecordingDispatcher()
    executor = TraceExecutor(
        parsed, credit_issuer=_CountingIssuer(), tool_dispatcher=dispatcher
    )
    result = await executor.run(parsed.traces[0])
    assert dispatcher.nodes == ["t0"]
    assert dispatcher.opened == ["trace-1"]
    assert dispatcher.closed == ["trace-1"]
    assert result.channels["t0_out"] == "stubbed"
    assert result.tool_durations_s == [1.5]


async def test_tool_node_without_a_dispatcher_raises() -> None:
    parsed = _graph()
    executor = TraceExecutor(parsed, credit_issuer=_CountingIssuer())
    with pytest.raises(RuntimeError, match="no tool dispatcher"):
        await executor.run(parsed.traces[0])


async def test_trace_without_tool_nodes_needs_no_dispatcher() -> None:
    """A pure-LLM graph must not require tool wiring."""
    graph = GraphRecord(
        state={"n0_out": ChannelSpec()},
        nodes={"n0": LlmNode(prompt=[], output="n0_out", arrival_offset_us=0)},
        edges=[
            StaticEdge(source="START", target="n0"),
            StaticEdge(source="n0", target="END"),
        ],
    )
    parsed = ParsedGraph(graph=graph, traces=[TraceRecord(id="trace-2")])
    executor = TraceExecutor(parsed, credit_issuer=_CountingIssuer())
    result = await executor.run(parsed.traces[0])
    assert result.tool_durations_s == []


class _HalfOpenedDispatcher:
    """`open_trace` acquires a resource and THEN fails, like the Docker sandbox
    starting its container before spawning the exec session."""

    def __init__(self) -> None:
        self.opened: list[str] = []
        self.closed: list[str] = []

    async def open_trace(self, trace_id: str) -> None:
        self.opened.append(trace_id)
        raise RuntimeError("sandbox session never came up")

    async def dispatch(self, node, request, ctx) -> ToolDispatchResult:
        raise AssertionError("no node may fire after a failed open_trace")

    async def close_trace(self, trace_id: str) -> None:
        self.closed.append(trace_id)


async def test_failed_open_trace_still_tears_the_trace_down() -> None:
    """A part-way open already owns a container; skipping teardown leaks it."""
    parsed = _graph()
    dispatcher = _HalfOpenedDispatcher()
    executor = TraceExecutor(
        parsed, credit_issuer=_CountingIssuer(), tool_dispatcher=dispatcher
    )
    with pytest.raises(RuntimeError, match="never came up"):
        await executor.run(parsed.traces[0])
    assert dispatcher.closed == ["trace-1"]


class _FailingTeardownDispatcher:
    """Teardown blows up; the trace's own error must still reach the caller."""

    def __init__(self) -> None:
        self.closed: list[str] = []

    async def open_trace(self, trace_id: str) -> None:
        pass

    async def dispatch(self, node, request, ctx) -> ToolDispatchResult:
        raise RuntimeError("the real cause")

    async def close_trace(self, trace_id: str) -> None:
        self.closed.append(trace_id)
        raise RuntimeError("teardown noise")


async def test_teardown_failure_does_not_mask_the_trace_error() -> None:
    parsed = _graph()
    dispatcher = _FailingTeardownDispatcher()
    executor = TraceExecutor(
        parsed, credit_issuer=_CountingIssuer(), tool_dispatcher=dispatcher
    )
    with pytest.raises(BaseException) as excinfo:
        await executor.run(parsed.traces[0])
    assert "the real cause" in repr(excinfo.value)
    assert "teardown noise" not in repr(excinfo.value)
    assert dispatcher.closed == ["trace-1"]


class _CancelledSandbox(_StubSandbox):
    """Parks in `run`, then parks again in `close` so the test can re-cancel
    the trace WHILE teardown is in flight -- what a `wait_for` / TaskGroup
    cascade does."""

    def __init__(self) -> None:
        super().__init__()
        self.never = asyncio.Event()
        self.closing = asyncio.Event()
        self.release_close = asyncio.Event()

    async def run(self, command: str, timeout_s: float | None = None) -> ToolResult:
        self.commands.append(command)
        await self.never.wait()
        raise AssertionError("unreachable")

    async def close(self) -> None:
        self.closing.set()
        await self.release_close.wait()
        self.closed = True


async def test_teardown_survives_cancellation_of_the_trace() -> None:
    """Issuer refusal and run-cancel cancel the trace mid-flight; a container
    that outlives the worker must still be removed -- even when the cancel is
    re-delivered while `close` is itself awaiting."""
    parsed = _graph()
    sandbox = _CancelledSandbox()
    executor = TraceExecutor(
        parsed,
        credit_issuer=_CountingIssuer(),
        tool_dispatcher=SandboxToolDispatcher(lambda trace_id: sandbox),
    )
    task = asyncio.create_task(executor.run(parsed.traces[0]))
    while not sandbox.commands:
        await asyncio.sleep(0)

    task.cancel()
    await sandbox.closing.wait()
    task.cancel()  # the re-cancel a cancellation cascade delivers
    sandbox.release_close.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert sandbox.closed is True


async def test_sandbox_dispatcher_refuses_a_second_concurrent_trace() -> None:
    """One instance serves one trace; sharing would cross-contaminate sandboxes."""
    dispatcher = SandboxToolDispatcher(lambda trace_id: _StubSandbox())
    await dispatcher.open_trace("trace-a")
    with pytest.raises(RuntimeError, match="already holds an open sandbox"):
        await dispatcher.open_trace("trace-b")
