# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.control_hooks import (
    PreparedEndpointControlHooks,
    stop_server_profiler,
)
from aiperf.common.control_plane_http import ControlPlaneHttpError
from aiperf.common.enums import CreditPhase
from aiperf.timing.phase_orchestrator import run_phase_with_server_profiler


@pytest.mark.asyncio
async def test_profiler_starts_before_profiling_runner_and_stops_after() -> None:
    trace: list[str] = []

    async def run_phase() -> None:
        trace.append("profiling_run")

    async def start(_hooks, _headers) -> None:
        trace.append("profiler_start")

    async def stop(_hooks, _headers) -> None:
        trace.append("profiler_stop")

    hooks = MagicMock(
        profiler_start_urls=["http://h/start"],
        profiler_stop_urls=["http://h/stop"],
    )
    await run_phase_with_server_profiler(
        phase=CreditPhase.PROFILING,
        hooks=hooks,
        headers={},
        run_phase=run_phase,
        start_fn=start,
        stop_fn=stop,
        warn_fn=lambda _msg: None,
    )
    assert trace == ["profiler_start", "profiling_run", "profiler_stop"]


@pytest.mark.asyncio
async def test_defer_stop_leaves_stop_to_caller_after_run() -> None:
    trace: list[str] = []

    async def run_phase() -> None:
        trace.append("profiling_run")

    async def start(_hooks, _headers) -> None:
        trace.append("profiler_start")

    async def stop(_hooks, _headers) -> None:
        trace.append("profiler_stop")

    hooks = MagicMock(
        profiler_start_urls=["http://h/start"],
        profiler_stop_urls=["http://h/stop"],
    )
    owed = await run_phase_with_server_profiler(
        phase=CreditPhase.PROFILING,
        hooks=hooks,
        headers={},
        run_phase=run_phase,
        start_fn=start,
        stop_fn=stop,
        warn_fn=lambda _msg: None,
        defer_stop=True,
    )
    assert owed is True
    assert trace == ["profiler_start", "profiling_run"]
    await stop(hooks, {})
    assert trace == ["profiler_start", "profiling_run", "profiler_stop"]


@pytest.mark.asyncio
async def test_defer_stop_still_stops_on_run_failure() -> None:
    async def run_phase() -> None:
        raise RuntimeError("phase boom")

    start = AsyncMock()
    stop = AsyncMock()
    hooks = MagicMock(
        profiler_start_urls=["http://h/start"],
        profiler_stop_urls=["http://h/stop"],
    )
    with pytest.raises(RuntimeError, match="phase boom"):
        await run_phase_with_server_profiler(
            phase=CreditPhase.PROFILING,
            hooks=hooks,
            headers={},
            run_phase=run_phase,
            start_fn=start,
            stop_fn=stop,
            warn_fn=lambda _msg: None,
            defer_stop=True,
        )
    start.assert_awaited_once()
    stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_warmup_phase_does_not_invoke_profiler() -> None:
    trace: list[str] = []

    async def run_phase() -> None:
        trace.append("warmup_run")

    start = AsyncMock()
    stop = AsyncMock()
    hooks = MagicMock(
        profiler_start_urls=["http://h/start"],
        profiler_stop_urls=["http://h/stop"],
    )
    await run_phase_with_server_profiler(
        phase=CreditPhase.WARMUP,
        hooks=hooks,
        headers={},
        run_phase=run_phase,
        start_fn=start,
        stop_fn=stop,
        warn_fn=lambda _msg: None,
    )
    assert trace == ["warmup_run"]
    start.assert_not_awaited()
    stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_profiler_stop_failure_warns_but_does_not_raise() -> None:
    warnings: list[str] = []

    async def run_phase() -> None:
        return None

    start = AsyncMock()
    stop = AsyncMock(side_effect=ControlPlaneHttpError("stop failed"))
    hooks = MagicMock(
        profiler_start_urls=["http://h/start"],
        profiler_stop_urls=["http://h/stop"],
    )
    await run_phase_with_server_profiler(
        phase=CreditPhase.PROFILING,
        hooks=hooks,
        headers={},
        run_phase=run_phase,
        start_fn=start,
        stop_fn=stop,
        warn_fn=warnings.append,
    )
    assert any("stop failed" in w for w in warnings)


@pytest.mark.asyncio
async def test_profiler_start_failure_is_fatal_before_profiling_run() -> None:
    ran = False

    async def run_phase() -> None:
        nonlocal ran
        ran = True

    start = AsyncMock(side_effect=ControlPlaneHttpError("start failed"))
    stop = AsyncMock()
    hooks = MagicMock(
        profiler_start_urls=["http://h/start"],
        profiler_stop_urls=["http://h/stop"],
    )
    with pytest.raises(ControlPlaneHttpError, match="start failed"):
        await run_phase_with_server_profiler(
            phase=CreditPhase.PROFILING,
            hooks=hooks,
            headers={},
            run_phase=run_phase,
            start_fn=start,
            stop_fn=stop,
            warn_fn=lambda _msg: None,
        )
    assert ran is False


@pytest.mark.asyncio
async def test_profiler_stop_network_error_warns_and_does_not_raise() -> None:
    """Transport failures from stop must become ControlPlaneHttpError and warn."""
    warnings: list[str] = []
    # Port 1 is almost never listening → ClientConnectorError wrapped at boundary.
    hooks = PreparedEndpointControlHooks(
        timeout_s=0.5,
        reset_urls=[],
        profiler_start_urls=["http://127.0.0.1:1/start_profile"],
        profiler_stop_urls=["http://127.0.0.1:1/stop_profile"],
        profiler_timeout_s=0.5,
    )
    await run_phase_with_server_profiler(
        phase=CreditPhase.PROFILING,
        hooks=hooks,
        headers={},
        run_phase=AsyncMock(),
        start_fn=AsyncMock(),
        stop_fn=stop_server_profiler,
        warn_fn=warnings.append,
    )
    assert any("server_profiler stop failed" in w for w in warnings)
