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

import asyncio
from collections import Counter
from multiprocessing import Process
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from aiperf.common.control_structs import Command
from aiperf.common.enums import (
    CommandType,
    LifecycleState,
    ServiceRegistrationStatus,
    SystemState,
)
from aiperf.common.messages import BaseServiceErrorMessage
from aiperf.common.models import ErrorDetails, ServiceRunInfo
from aiperf.controller.multiprocess_service_manager import MultiProcessServiceManager
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
    exit_codes: dict[str, int] | None = None,
    state: SystemState = SystemState.PROFILING,
) -> None:
    manager = system_controller.service_manager
    manager.service_id_map = {}
    manager.spawned_worker_ids = MagicMock(return_value=frozenset(spawned))
    manager.get_service_liveness = MagicMock(side_effect=lambda sid: sid not in dead)
    manager.get_service_exit_code = MagicMock(
        side_effect=lambda sid: (exit_codes or {}).get(sid)
    )
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
    assert (
        "No AWS credentials found"
        in system_controller._exit_errors[0].error_details.message
    )


@pytest.mark.asyncio
async def test_a_worker_failing_while_another_can_still_start_is_tolerated(
    system_controller: SystemController,
) -> None:
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


@pytest.mark.asyncio
async def test_a_worker_reaped_before_its_error_arrives_is_still_tolerated(
    system_controller: SystemController,
    benchmark_run,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registration-wait reaper and the SERVICE_ERROR race. With the real
    manager, reaping first used to erase the worker's identity, so its error
    took the generic path and cancelled the run although another worker was
    healthy."""
    monkeypatch.setattr(
        "aiperf.controller.multiprocess_service_manager.Process",
        MagicMock(side_effect=lambda **_: MagicMock(spec=Process)),
    )
    manager = MultiProcessServiceManager(required_services={}, run=benchmark_run)
    await manager.run_service(ServiceType.WORKER, num_replicas=2)
    dead, alive = manager.multi_process_info
    dead.process.is_alive.return_value = False
    alive.process.is_alive.return_value = True
    manager._reap_dead_processes_during_registration(Counter({ServiceType.WORKER: 2}))

    system_controller.service_manager = manager
    system_controller._system_state = SystemState.PROFILING
    system_controller._cancel_profiling = AsyncMock()
    system_controller._check_and_trigger_shutdown = AsyncMock()

    await _report(system_controller, dead.service_id)

    system_controller._cancel_profiling.assert_not_awaited()
    assert system_controller._exit_errors == []


def _on_each_tick(monkeypatch: pytest.MonkeyPatch, callback) -> None:
    """Run ``callback(tick)`` at every watch poll, then yield as sleep would."""
    real_sleep = asyncio.sleep
    ticks = 0

    async def tick(_delay: float) -> None:
        nonlocal ticks
        ticks += 1
        await callback(ticks)
        await real_sleep(0)

    monkeypatch.setattr("aiperf.controller.system_controller.asyncio.sleep", tick)


class TestWorkersThatDieWithoutReporting:
    """Workers start after the registration-wait reaper has finished, so one
    that dies without publishing SERVICE_ERROR -- killed, or crashed before its
    comms were up -- is seen by nothing until PhaseOrchestrator's 30s
    credit-router timeout. The watch polls process liveness until every
    spawned worker has registered or failed."""

    @pytest.mark.asyncio
    async def test_the_only_worker_dying_silently_cancels_with_its_exit_code(
        self, system_controller: SystemController
    ) -> None:
        _local_workers(
            system_controller,
            spawned={"worker_a"},
            dead=frozenset({"worker_a"}),
            exit_codes={"worker_a": -9},
        )

        await system_controller._watch_workers_until_registered()

        system_controller._cancel_profiling.assert_awaited_once()
        [error] = system_controller._exit_errors
        assert error.service_id == "worker_a"
        assert "exited before registering" in error.error_details.message
        assert "exit code -9" in error.error_details.message

    @pytest.mark.asyncio
    async def test_a_report_arriving_within_the_grace_keeps_the_real_cause(
        self, system_controller: SystemController, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing worker publishes and then exits, so the watch can see it
        dead before its report lands. Acting on the first sighting would bury
        the real cause under "exited before registering"."""
        _local_workers(
            system_controller,
            spawned={"worker_a"},
            dead=frozenset({"worker_a"}),
            exit_codes={"worker_a": 1},
        )

        async def report_on_first_tick(tick: int) -> None:
            if tick == 1:
                await _report(system_controller, "worker_a")

        _on_each_tick(monkeypatch, report_on_first_tick)

        await system_controller._watch_workers_until_registered()

        [error] = system_controller._exit_errors
        assert error.error_details.message == _CAUSE

    @pytest.mark.asyncio
    async def test_a_silent_death_while_another_worker_starts_is_tolerated(
        self, system_controller: SystemController, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _local_workers(
            system_controller,
            spawned={"worker_a", "worker_b"},
            dead=frozenset({"worker_a"}),
            exit_codes={"worker_a": -11},
        )
        manager = system_controller.service_manager

        async def worker_b_registers_late(tick: int) -> None:
            if tick == 10:
                manager.service_id_map["worker_b"] = MagicMock()

        _on_each_tick(monkeypatch, worker_b_registers_late)

        await system_controller._watch_workers_until_registered()

        system_controller._cancel_profiling.assert_not_awaited()
        assert "worker_a" in system_controller._worker_startup_failures
        assert system_controller._exit_errors == []

    @pytest.mark.asyncio
    async def test_the_watch_ends_once_every_worker_has_registered(
        self, system_controller: SystemController
    ) -> None:
        _local_workers(system_controller, spawned={"worker_a"})
        system_controller.service_manager.service_id_map["worker_a"] = MagicMock()

        await system_controller._watch_workers_until_registered()

        system_controller._cancel_profiling.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_watch_ends_when_the_system_stops(
        self, system_controller: SystemController
    ) -> None:
        _local_workers(
            system_controller,
            spawned={"worker_a"},
            dead=frozenset({"worker_a"}),
            state=SystemState.STOPPING,
        )

        await system_controller._watch_workers_until_registered()

        system_controller._cancel_profiling.assert_not_awaited()
        assert system_controller._exit_errors == []


class TestSpawningStartsTheWatch:
    @staticmethod
    async def _spawn(system_controller: SystemController, spawned: set[str]) -> None:
        manager = system_controller.service_manager
        manager.spawned_worker_ids = MagicMock(return_value=frozenset(spawned))
        system_controller._watch_workers_until_registered = AsyncMock()
        system_controller.scale_record_processors_with_workers = False
        await system_controller._handle_spawn_workers_command(
            Command(
                cid="c-1",
                cmd=CommandType.SPAWN_WORKERS,
                payload=orjson.dumps({"num_workers": len(spawned) or 1}),
            )
        )
        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_spawning_local_workers_starts_the_watch(
        self, system_controller: SystemController
    ) -> None:
        await self._spawn(system_controller, {"worker_a"})

        system_controller._watch_workers_until_registered.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_watch_without_local_workers(
        self, system_controller: SystemController
    ) -> None:
        """Under Kubernetes the manager spawns no local processes and has no
        liveness to poll; pod failures have their own watcher."""
        await self._spawn(system_controller, set())

        system_controller._watch_workers_until_registered.assert_not_awaited()
