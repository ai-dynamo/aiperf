# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cancellation tests for the Kubernetes benchmark API collector."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from tests.kubernetes.helpers.kubectl import KubectlClient


class _BlockingRequest:
    """HTTP request context that blocks until its owner task is cancelled."""

    def __init__(self, started: asyncio.Event) -> None:
        self.started = started

    async def __aenter__(self) -> Any:
        self.started.set()
        await asyncio.Event().wait()

    async def __aexit__(self, *_args: object) -> None:
        return None


class _FakeSession:
    """ClientSession test double that records structured context cleanup."""

    def __init__(self, started: asyncio.Event, exited: asyncio.Event) -> None:
        self.started = started
        self.exited = exited

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *_args: object) -> None:
        self.exited.set()

    def get(self, *_args: object, **_kwargs: object) -> _BlockingRequest:
        return _BlockingRequest(self.started)


class _FakeProcess:
    """Subprocess test double used to verify startup cancellation cleanup."""

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.stderr = None
        self.terminated = False
        self.waited = False

    def terminate(self) -> None:
        self.terminated = True

    async def wait(self) -> None:
        self.waited = True


@pytest.mark.asyncio
async def test_wait_for_benchmark_api_cancellation_closes_session_and_port_forward() -> (
    None
):
    """Cancelling the API collector unwinds both nested async contexts."""
    client = KubectlClient()
    request_started = asyncio.Event()
    session_exited = asyncio.Event()
    port_forward_exited = asyncio.Event()
    session = _FakeSession(request_started, session_exited)

    @asynccontextmanager
    async def fake_port_forward(*_args: object, **_kwargs: object):
        try:
            yield 12345
        finally:
            port_forward_exited.set()

    with (
        patch.object(client, "port_forward", new=fake_port_forward),
        patch(
            "aiperf.transports.aiohttp_client.create_tcp_connector",
            return_value=object(),
        ),
        patch(
            "tests.kubernetes.helpers.kubectl.aiohttp.ClientSession",
            return_value=session,
        ),
    ):
        task = asyncio.create_task(
            client.wait_for_benchmark_api("controller", "bench-ns")
        )
        await request_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert session_exited.is_set()
    assert port_forward_exited.is_set()


@pytest.mark.asyncio
async def test_port_forward_startup_cancellation_terminates_process() -> None:
    """Cancellation during the readiness delay cannot orphan kubectl."""
    client = KubectlClient()
    process = _FakeProcess()
    process_created = asyncio.Event()
    startup_waiting = asyncio.Event()

    async def create_process(*_args: object, **_kwargs: object) -> _FakeProcess:
        process_created.set()
        return process

    async def block_startup_delay(_delay: float) -> None:
        startup_waiting.set()
        await asyncio.Event().wait()

    with (
        patch(
            "tests.kubernetes.helpers.kubectl.asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=create_process),
        ),
        patch(
            "tests.kubernetes.helpers.kubectl.asyncio.sleep",
            new=AsyncMock(side_effect=block_startup_delay),
        ),
    ):
        context = client.port_forward(
            "controller", 9090, local_port=12345, namespace="bench-ns"
        )
        task = asyncio.create_task(context.__aenter__())
        await process_created.wait()
        await startup_waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert process.terminated
    assert process.waited
