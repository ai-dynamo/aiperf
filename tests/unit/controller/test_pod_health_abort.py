# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests that Kubernetes pod-health failures actually abort a benchmark.

``KubernetesServiceManager`` detects crashed worker pods, but detection is only
useful if the controller consumes it. These tests pin the two consumption
points: the pre-PROFILE_START health gate, and the abort waiter that cancels
profiling when the pod-failure threshold is breached mid-run.
"""

import asyncio
import contextlib
from unittest.mock import AsyncMock

import pytest

from aiperf.common.enums import SystemState
from aiperf.common.exceptions import LifecycleOperationError, ServiceProcessDiedError
from aiperf.controller.protocols import KubernetesServiceManagerProtocol
from aiperf.controller.system_controller import SystemController
from aiperf.credit.messages import CreditsCompleteMessage


def test_protocol_declares_pod_health_surface() -> None:
    """The controller may only call what the protocol promises."""
    assert hasattr(KubernetesServiceManagerProtocol, "check_pods_healthy")
    assert hasattr(KubernetesServiceManagerProtocol, "get_pod_summary")
    annotations = KubernetesServiceManagerProtocol.__annotations__
    assert "pod_failure_abort_event" in annotations
    assert "pod_failure_abort_reason" in annotations


@pytest.mark.asyncio
async def test_pod_health_gate_runs_before_profile_start(
    system_controller: SystemController,
    mock_service_manager: AsyncMock,
) -> None:
    """A pod that died after registering must be caught before PROFILE_START."""
    order: list[str] = []

    async def _check() -> None:
        order.append("check_pods_healthy")

    mock_service_manager.check_pods_healthy = AsyncMock(side_effect=_check)
    system_controller._start_profiling_all_services = AsyncMock(
        side_effect=lambda: order.append("profile_start")
    )

    await system_controller._verify_pods_healthy()
    await system_controller._start_profiling_all_services()

    assert order == ["check_pods_healthy", "profile_start"]


@pytest.mark.asyncio
async def test_pod_health_gate_failure_stops_the_controller(
    system_controller: SystemController,
    mock_service_manager: AsyncMock,
) -> None:
    """A terminal pod raises out of the gate rather than profiling regardless."""
    mock_service_manager.check_pods_healthy = AsyncMock(
        side_effect=ServiceProcessDiedError(
            service_id="worker_3_a1b2", service_type="worker", exit_code=-9
        )
    )
    system_controller._start_profiling_all_services = AsyncMock()

    # try_operation_or_stop records the failure and re-raises, unwinding
    # _start_services before PROFILE_START is ever sent.
    with pytest.raises(LifecycleOperationError):
        await system_controller._verify_pods_healthy()

    assert system_controller._exit_errors
    system_controller._start_profiling_all_services.assert_not_awaited()


@pytest.mark.asyncio
async def test_pod_failure_abort_event_cancels_profiling(
    system_controller: SystemController,
    mock_service_manager: AsyncMock,
) -> None:
    """Breaching the pod-failure threshold mid-run cancels the benchmark."""
    event = asyncio.Event()
    mock_service_manager.pod_failure_abort_event = event
    mock_service_manager.pod_failure_abort_reason = "2/2 worker pods failed"
    system_controller._cancel_profiling = AsyncMock()

    watcher = asyncio.ensure_future(system_controller._watch_pod_failure_abort())
    await asyncio.sleep(0)
    event.set()
    await watcher

    system_controller._cancel_profiling.assert_awaited_once()


@pytest.mark.asyncio
async def test_pod_failure_abort_noop_after_cancellation(
    system_controller: SystemController,
    mock_service_manager: AsyncMock,
) -> None:
    """A late threshold breach during shutdown must not re-cancel."""
    event = asyncio.Event()
    event.set()
    mock_service_manager.pod_failure_abort_event = event
    mock_service_manager.pod_failure_abort_reason = "1/1 worker pods failed"
    system_controller._was_cancelled = True
    system_controller._cancel_profiling = AsyncMock()

    await system_controller._watch_pod_failure_abort()

    system_controller._cancel_profiling.assert_not_awaited()


@pytest.mark.asyncio
async def test_pod_failure_during_profiling_cancels_via_watcher_task(
    system_controller: SystemController,
    mock_service_manager: AsyncMock,
) -> None:
    """Baseline for the disarm tests: mid-profiling breach still cancels.

    Exercises the same real ``_pod_failure_watcher_task`` path the disarm
    tests use, so a disarm bug cannot be mistaken for the watcher never
    having been armed in the first place.
    """
    event = asyncio.Event()
    mock_service_manager.pod_failure_abort_event = event
    mock_service_manager.pod_failure_abort_reason = "2/2 worker pods failed"
    system_controller._cancel_profiling = AsyncMock()

    system_controller._pod_failure_watcher_task = asyncio.ensure_future(
        system_controller._watch_pod_failure_abort()
    )
    await asyncio.sleep(0)
    event.set()
    await system_controller._pod_failure_watcher_task

    system_controller._cancel_profiling.assert_awaited_once()


@pytest.mark.asyncio
async def test_pod_failure_after_credits_complete_does_not_cancel(
    system_controller: SystemController,
    mock_service_manager: AsyncMock,
) -> None:
    """Pods exit legitimately during teardown; that must not cancel a good run.

    Once credits are complete the load phase is over and remaining work is
    record aggregation. A worker pod terminating then is normal teardown, not
    a failure, so the watcher must already be disarmed.
    """
    event = asyncio.Event()
    mock_service_manager.pod_failure_abort_event = event
    mock_service_manager.pod_failure_abort_reason = "2/2 worker pods failed"
    system_controller._cancel_profiling = AsyncMock()

    system_controller._pod_failure_watcher_task = asyncio.ensure_future(
        system_controller._watch_pod_failure_abort()
    )
    await asyncio.sleep(0)

    await system_controller._process_credits_complete_message(
        CreditsCompleteMessage(service_id="timing_manager")
    )
    event.set()
    await asyncio.sleep(0)
    with contextlib.suppress(asyncio.CancelledError):
        await system_controller._pod_failure_watcher_task

    system_controller._cancel_profiling.assert_not_awaited()


@pytest.mark.asyncio
async def test_pod_failure_after_shutdown_check_does_not_cancel(
    system_controller: SystemController,
    mock_service_manager: AsyncMock,
) -> None:
    """Entering the shutdown check disarms the watcher too.

    ``_check_and_trigger_shutdown`` is the other entry into teardown, and it
    can trigger shutdown without a ``CREDITS_COMPLETE`` message ever having
    been handled by the controller.
    """
    event = asyncio.Event()
    mock_service_manager.pod_failure_abort_event = event
    mock_service_manager.pod_failure_abort_reason = "2/2 worker pods failed"
    system_controller._cancel_profiling = AsyncMock()
    system_controller._system_state = SystemState.PROFILING

    system_controller._pod_failure_watcher_task = asyncio.ensure_future(
        system_controller._watch_pod_failure_abort()
    )
    await asyncio.sleep(0)

    await system_controller._check_and_trigger_shutdown()
    assert system_controller._shutdown_triggered
    event.set()
    await asyncio.sleep(0)
    with contextlib.suppress(asyncio.CancelledError):
        await system_controller._pod_failure_watcher_task

    system_controller._cancel_profiling.assert_not_awaited()


@pytest.mark.asyncio
async def test_pending_result_domain_does_not_disarm_watcher(
    system_controller: SystemController,
    mock_service_manager: AsyncMock,
) -> None:
    """A readiness check that does not trigger shutdown leaves the watcher armed.

    ``_check_and_trigger_shutdown`` runs on every result/status/error message,
    long before teardown. Disarming on entry rather than on the branch that
    actually starts shutdown would silently switch pod-failure abort off for
    the rest of the load phase.
    """
    system_controller._result_join_coordinator.register("records", "records_manager")
    event = asyncio.Event()
    mock_service_manager.pod_failure_abort_event = event
    mock_service_manager.pod_failure_abort_reason = "2/2 worker pods failed"
    system_controller._cancel_profiling = AsyncMock()

    system_controller._pod_failure_watcher_task = asyncio.ensure_future(
        system_controller._watch_pod_failure_abort()
    )
    await asyncio.sleep(0)

    await system_controller._check_and_trigger_shutdown()
    assert not system_controller._shutdown_triggered
    event.set()
    await system_controller._pod_failure_watcher_task

    system_controller._cancel_profiling.assert_awaited_once()


@pytest.mark.asyncio
async def test_abort_completes_even_though_cancel_disarms_the_watcher(
    system_controller: SystemController,
    mock_service_manager: AsyncMock,
) -> None:
    """The abort must finish its work, not cancel itself half way through.

    ``_watch_pod_failure_abort`` *is* ``_pod_failure_watcher_task``, and the
    cancel path it invokes disarms that same task. Cancelling the currently
    running task delivers CancelledError at the next await, so everything after
    the first suspension point -- sending ProfileCancelCommand, harvesting
    results, stopping -- is silently skipped and the benchmark runs on.
    """
    event = asyncio.Event()
    mock_service_manager.pod_failure_abort_event = event
    mock_service_manager.pod_failure_abort_reason = "2/2 worker pods failed"

    reached_the_end = False

    async def cancel_profiling() -> None:
        # Mirrors the real path: disarm first, then do async teardown work.
        system_controller._disarm_pod_failure_watcher()
        await asyncio.sleep(0)
        nonlocal reached_the_end
        reached_the_end = True

    system_controller._cancel_profiling = cancel_profiling
    system_controller._pod_failure_watcher_task = asyncio.ensure_future(
        system_controller._watch_pod_failure_abort()
    )
    await asyncio.sleep(0)
    event.set()
    with contextlib.suppress(asyncio.CancelledError):
        await system_controller._pod_failure_watcher_task

    assert reached_the_end, "abort self-cancelled: work after the first await never ran"


@pytest.mark.asyncio
async def test_disarm_still_cancels_the_watcher_from_another_task(
    system_controller: SystemController,
    mock_service_manager: AsyncMock,
) -> None:
    """Disarming from a different task must still cancel the waiter."""
    event = asyncio.Event()
    mock_service_manager.pod_failure_abort_event = event

    system_controller._pod_failure_watcher_task = asyncio.ensure_future(
        system_controller._watch_pod_failure_abort()
    )
    await asyncio.sleep(0)

    system_controller._disarm_pod_failure_watcher()
    await asyncio.sleep(0)

    assert system_controller._pod_failure_watcher_task.cancelled()
    assert system_controller._pod_failure_watch_disarmed
