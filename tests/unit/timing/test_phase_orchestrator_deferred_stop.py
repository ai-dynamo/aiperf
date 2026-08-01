# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deferred seamless profiler stops must complete before a run reports success."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.control_hooks import PreparedEndpointControlHooks


def _hooks() -> PreparedEndpointControlHooks:
    return PreparedEndpointControlHooks(
        timeout_s=1.0,
        reset_urls=[],
        profiler_start_urls=["http://a:8000/start_profile"],
        profiler_stop_urls=["http://a:8000/stop_profile"],
        profiler_timeout_s=1.0,
    )


@pytest.mark.asyncio
async def test_drain_awaits_stop_task_spawned_by_phase_complete_callback(
    create_orchestrator_harness,
) -> None:
    """A stop already in flight must be awaited, not left dangling."""
    orch = create_orchestrator_harness([("c1", 1)]).orchestrator
    orch._control_hooks = _hooks()
    runner = MagicMock()
    orch._server_profiler_owners.add(runner)

    released = asyncio.Event()
    stop_completed = False

    async def slow_stop(*_args: object) -> None:
        nonlocal stop_completed
        await released.wait()
        stop_completed = True

    with patch(
        "aiperf.timing.phase_orchestrator.stop_server_profiler",
        new_callable=AsyncMock,
        side_effect=slow_stop,
    ):
        # Fire the phase-complete callback the way _on_return_wait_complete does.
        orch._phase_runner_cleanup_and_stop_profiler_callback(runner)()
        assert orch._deferred_profiler_stops, "stop task should be tracked"

        drain = asyncio.create_task(orch._drain_deferred_profiler_stops())
        await asyncio.sleep(0)
        assert not drain.done(), "drain must block on the in-flight stop"

        released.set()
        await drain

    assert stop_completed
    assert not orch._server_profiler_owners


@pytest.mark.asyncio
async def test_drain_stops_owner_whose_callback_never_fired(
    create_orchestrator_harness,
) -> None:
    """Ownership held with no callback yet must still be stopped directly."""
    orch = create_orchestrator_harness([("c1", 1)]).orchestrator
    orch._control_hooks = _hooks()
    runner = MagicMock()
    orch._server_profiler_owners.add(runner)

    with patch(
        "aiperf.timing.phase_orchestrator.stop_server_profiler",
        new_callable=AsyncMock,
    ) as stop:
        await orch._drain_deferred_profiler_stops()

    stop.assert_awaited_once()
    assert not orch._server_profiler_owners


@pytest.mark.asyncio
async def test_drain_does_not_double_stop_when_callback_already_ran(
    create_orchestrator_harness,
) -> None:
    """The tracked-task path and the direct path must not both issue a stop."""
    orch = create_orchestrator_harness([("c1", 1)]).orchestrator
    orch._control_hooks = _hooks()
    runner = MagicMock()
    orch._server_profiler_owners.add(runner)

    with patch(
        "aiperf.timing.phase_orchestrator.stop_server_profiler",
        new_callable=AsyncMock,
    ) as stop:
        orch._phase_runner_cleanup_and_stop_profiler_callback(runner)()
        await orch._drain_deferred_profiler_stops()

    stop.assert_awaited_once()
