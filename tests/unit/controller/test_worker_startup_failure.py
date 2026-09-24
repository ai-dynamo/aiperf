# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A worker that fails to start must be judged by whether any worker is left.

Workers publish their start-up failure as SERVICE_ERROR (``BaseService.
reports_startup_failure``), but they do so *before registering*, so the
controller does not know them yet. The generic handler treats an unknown sender
as required, which would cancel the run over a single flaky worker, and treats
a known worker as optional, which would never cancel at all.

Neither is right. In multi-process mode workers are spawned later through
``SPAWN_WORKERS`` and are not required services, so one failing while another
can still start is a degraded run. But once no worker can start, nothing will
ever send a request: waiting out ``PhaseOrchestrator``'s 30s credit-router
timeout only buries the workers' own error under "No workers registered with
the credit router".
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import LifecycleState, ServiceRegistrationStatus, SystemState
from aiperf.common.messages import BaseServiceErrorMessage
from aiperf.common.models import ErrorDetails, ServiceRunInfo
from aiperf.controller.system_controller import SystemController
from aiperf.plugin.enums import ServiceType

_CAUSE = (
    "SigV4RequestSigner._init_credentials: No AWS credentials found. Configure "
    "via environment variables, ~/.aws/credentials, or IAM role."
)


def _local_workers(
    system_controller: SystemController,
    *,
    spawned: set[str],
    dead: frozenset[str] = frozenset(),
    state: SystemState = SystemState.PROFILING,
) -> None:
    """Workers spawned as local processes, none of them registered yet."""
    manager = system_controller.service_manager
    manager.service_id_map = {}
    manager.spawned_worker_ids = MagicMock(return_value=frozenset(spawned))
    manager.get_service_liveness = MagicMock(side_effect=lambda sid: sid not in dead)
    system_controller._system_state = state
    system_controller._cancel_profiling = AsyncMock()
    system_controller._check_and_trigger_shutdown = AsyncMock()


async def _report(system_controller: SystemController, service_id: str) -> None:
    await system_controller._process_service_error_message(
        BaseServiceErrorMessage(
            service_id=service_id, error=ErrorDetails(message=_CAUSE)
        )
    )


@pytest.mark.asyncio
async def test_the_only_worker_failing_to_start_cancels_with_its_real_error(
    system_controller: SystemController,
) -> None:
    _local_workers(system_controller, spawned={"worker_a"})

    await _report(system_controller, "worker_a")

    system_controller._cancel_profiling.assert_awaited_once()
    assert [e.service_id for e in system_controller._exit_errors] == ["worker_a"]
    # The worker's own message is what reaches the exit-errors panel, rather
    # than the credit-router timeout it used to be buried under.
    assert (
        "No AWS credentials found"
        in system_controller._exit_errors[0].error_details.message
    )


@pytest.mark.asyncio
async def test_a_worker_failing_while_another_can_still_start_is_tolerated(
    system_controller: SystemController,
) -> None:
    """Unknown senders are treated as required by the generic handler, which
    would cancel here over one flaky worker. A partial loss keeps today's
    behavior: continue, and do not turn the run into a failure."""
    _local_workers(system_controller, spawned={"worker_a", "worker_b"})

    await _report(system_controller, "worker_a")

    system_controller._cancel_profiling.assert_not_awaited()
    assert system_controller._exit_errors == []


@pytest.mark.asyncio
async def test_the_last_of_several_workers_failing_cancels_and_reports_all(
    system_controller: SystemController,
) -> None:
    """Workers failing on configuration usually fail together, but their
    reports arrive one at a time. Only the last one decides, and the earlier
    ones are not lost -- the panel groups identical errors across services."""
    _local_workers(system_controller, spawned={"worker_a", "worker_b"})

    await _report(system_controller, "worker_a")
    system_controller._cancel_profiling.assert_not_awaited()

    await _report(system_controller, "worker_b")

    system_controller._cancel_profiling.assert_awaited_once()
    assert sorted(e.service_id for e in system_controller._exit_errors) == [
        "worker_a",
        "worker_b",
    ]


@pytest.mark.asyncio
async def test_a_worker_that_died_without_reporting_is_not_counted_as_viable(
    system_controller: SystemController,
) -> None:
    """A worker can die without publishing (a crash, or comms not up yet).
    Process liveness is ground truth in multi-process mode, so a dead one
    must not hold the run open."""
    _local_workers(
        system_controller,
        spawned={"worker_a", "worker_b"},
        dead=frozenset({"worker_b"}),
    )

    await _report(system_controller, "worker_a")

    system_controller._cancel_profiling.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_cancel_once_the_system_is_already_stopping(
    system_controller: SystemController,
) -> None:
    """Mirrors the required-service path: cancelling during shutdown would
    race the shutdown already in progress."""
    _local_workers(system_controller, spawned={"worker_a"}, state=SystemState.STOPPING)

    await _report(system_controller, "worker_a")

    system_controller._cancel_profiling.assert_not_awaited()
    assert [e.service_id for e in system_controller._exit_errors] == ["worker_a"]


@pytest.mark.asyncio
async def test_a_registered_worker_keeps_the_existing_optional_path(
    system_controller: SystemController,
) -> None:
    """The new path is only for workers that failed before registering. A
    registered worker's error still goes through the generic handler, where in
    multi-process mode a worker is optional and does not cancel the run."""
    assert ServiceType.WORKER not in system_controller.required_services
    manager = system_controller.service_manager
    manager.service_id_map = {
        "worker_a": ServiceRunInfo(
            service_id="worker_a",
            service_type=ServiceType.WORKER,
            registration_status=ServiceRegistrationStatus.REGISTERED,
            state=LifecycleState.RUNNING,
        )
    }
    manager.spawned_worker_ids = MagicMock(return_value=frozenset({"worker_a"}))
    manager.get_service_liveness = MagicMock(return_value=True)
    system_controller._system_state = SystemState.PROFILING
    system_controller._cancel_profiling = AsyncMock()
    system_controller._check_and_trigger_shutdown = AsyncMock()

    await _report(system_controller, "worker_a")

    system_controller._cancel_profiling.assert_not_awaited()
    system_controller._check_and_trigger_shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_second_report_from_a_failed_worker_does_not_replace_the_cause(
    system_controller: SystemController,
) -> None:
    """A worker can publish twice: its start-up failure, then ``_kill``'s
    generic "entered FAILED state" if its stop is invoked while stopping. The
    first message is the cause; the second must neither replace it nor add a
    second exit error, and must not cancel again."""
    _local_workers(system_controller, spawned={"worker_a"})

    await _report(system_controller, "worker_a")
    await system_controller._process_service_error_message(
        BaseServiceErrorMessage(
            service_id="worker_a",
            error=ErrorDetails(
                message="Service worker_a entered FAILED state and is being killed"
            ),
        )
    )

    system_controller._cancel_profiling.assert_awaited_once()
    assert len(system_controller._exit_errors) == 1
    assert (
        "No AWS credentials found"
        in system_controller._exit_errors[0].error_details.message
    )
