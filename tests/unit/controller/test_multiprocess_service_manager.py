# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from multiprocessing import Process
from unittest.mock import MagicMock

import pytest

from aiperf.common.exceptions import AIPerfError
from aiperf.controller.multiprocess_service_manager import (
    MultiProcessRunInfo,
    MultiProcessServiceManager,
)
from aiperf.plugin.enums import ServiceType


class TestForkProcessRemovalSmokeTest:
    """Bug 1 regression: ``ForkProcess`` import was Linux-only.

    The original ``Process | SpawnProcess | ForkProcess | None`` field type
    pulled ``ForkProcess`` from ``multiprocessing.context``, which fails at
    module-load time on Windows (``ImportError: cannot import name
    'ForkProcess'``). Replaced with ``Process | None`` — every Process
    subclass inherits from Process, no subclass dispatch in the codebase.
    """

    def test_module_imports_on_any_platform(self) -> None:
        """Importing this module must succeed on every platform AIPerf
        supports — ForkProcess is no longer in the import chain."""
        from aiperf.controller import multiprocess_service_manager

        assert multiprocess_service_manager.MultiProcessRunInfo is not None

    def test_field_accepts_a_plain_process(self) -> None:
        """The ``process`` field accepts any subclass of Process, including
        SpawnProcess (the actual runtime type AIPerf produces)."""
        from multiprocessing import Process

        info = MultiProcessRunInfo.model_construct(
            process=Process(target=lambda: None),
            service_type=ServiceType.SYSTEM_CONTROLLER,
            run_id="test",
        )
        assert info.process is not None
        # Don't actually start; just confirm assignment works.


class TestMultiProcessServiceManager:
    """Test MultiProcessServiceManager process failure scenarios."""

    @pytest.fixture
    def mock_dead_process(self) -> MagicMock:
        """Create a mock process that appears dead."""
        mock_process = MagicMock(spec=Process)
        mock_process.is_alive.return_value = False
        mock_process.pid = 12345
        return mock_process

    @pytest.fixture
    def mock_alive_process(self) -> MagicMock:
        """Create a mock process that appears alive."""
        mock_process = MagicMock(spec=Process)
        mock_process.is_alive.return_value = True
        mock_process.pid = 54321
        return mock_process

    @pytest.fixture
    def service_manager(self, benchmark_run) -> MultiProcessServiceManager:
        """Create a MultiProcessServiceManager instance for testing."""
        return MultiProcessServiceManager(
            required_services={
                ServiceType.DATASET_MANAGER: 1,
                ServiceType.TIMING_MANAGER: 1,
            },
            run=benchmark_run,
        )

    @pytest.mark.asyncio
    async def test_process_dies_before_registration_raises_error(
        self, service_manager: MultiProcessServiceManager, mock_dead_process: MagicMock
    ):
        """Test that MultiProcessServiceManager raises AIPerfError when a process dies before registering.

        This test verifies the critical safety mechanism where:
        1. A process is started but dies before it can register with the system controller
        2. During the registration wait loop, the service manager detects the dead process
        3. An AIPerfError is raised with a descriptive message about the failed process

        This prevents the system from hanging indefinitely waiting for a dead process to register.
        """
        # Create a process info with a dead process
        dead_process_info = MultiProcessRunInfo.model_construct(
            process=mock_dead_process,
            service_type=ServiceType.DATASET_MANAGER,
            service_id="dead_service_123",
        )
        service_manager.multi_process_info = [dead_process_info]

        # Expect an error due to the dead REQUIRED process. (DATASET_MANAGER
        # is in required_services via the service_manager fixture.)
        with pytest.raises(
            AIPerfError,
            match="Required service dead_service_123 died before registering",
        ):
            await service_manager.wait_for_all_services_registration(
                stop_event=asyncio.Event(),
                timeout_seconds=1.0,
            )

    @pytest.mark.asyncio
    async def test_mixed_alive_and_dead_processes_raises_error_for_dead_one(
        self,
        service_manager: MultiProcessServiceManager,
        mock_alive_process: MagicMock,
        mock_dead_process: MagicMock,
    ):
        """Test that the manager raises error for dead process even when other processes are alive."""
        # Create mix of alive and dead processes
        alive_process_info = MultiProcessRunInfo.model_construct(
            process=mock_alive_process,
            service_type=ServiceType.TIMING_MANAGER,
            service_id="alive_service_456",
        )
        dead_process_info = MultiProcessRunInfo.model_construct(
            process=mock_dead_process,
            service_type=ServiceType.DATASET_MANAGER,
            service_id="dead_service_789",
        )
        service_manager.multi_process_info = [alive_process_info, dead_process_info]

        # Should raise error about the dead REQUIRED process. (DATASET_MANAGER
        # is in required_services via the service_manager fixture.)
        with pytest.raises(
            AIPerfError,
            match="Required service dead_service_789 died before registering",
        ):
            await service_manager.wait_for_all_services_registration(
                stop_event=asyncio.Event(), timeout_seconds=1.0
            )

    @pytest.mark.asyncio
    async def test_none_process_raises_error(
        self, service_manager: MultiProcessServiceManager
    ):
        """Test that a None process (failed to start) is treated as dead."""
        # Create a process info with None process (failed to start)
        none_process_info = MultiProcessRunInfo.model_construct(
            process=None,
            service_type=ServiceType.DATASET_MANAGER,
            service_id="failed_to_start_service",
        )
        service_manager.multi_process_info = [none_process_info]

        # Should raise error: None process counts as dead, and DATASET_MANAGER
        # is in required_services. Optional services with None process would
        # be dropped with a warning instead (see test_optional_dead_drops_and_continues).
        with pytest.raises(
            AIPerfError,
            match="Required service failed_to_start_service died before registering",
        ):
            await service_manager.wait_for_all_services_registration(
                stop_event=asyncio.Event(), timeout_seconds=1.0
            )

    @pytest.mark.asyncio
    async def test_optional_dead_drops_and_continues(
        self,
        service_manager: MultiProcessServiceManager,
        mock_alive_process: MagicMock,
        mock_dead_process: MagicMock,
    ):
        """Pins F-04: an optional service (not in required_services) dying
        before registering must drop the service from the wait set and let
        the benchmark continue, NOT raise AIPerfError. Required services
        dying remains fatal (covered by other tests in this class).
        """
        # GPU_TELEMETRY is not in required_services (fixture has only
        # DATASET_MANAGER + TIMING_MANAGER) — so a dead one should be
        # warning + drop, not fatal.
        alive_dataset = MultiProcessRunInfo.model_construct(
            process=mock_alive_process,
            service_type=ServiceType.DATASET_MANAGER,
            service_id="dataset_alive",
        )
        alive_timing = MultiProcessRunInfo.model_construct(
            process=mock_alive_process,
            service_type=ServiceType.TIMING_MANAGER,
            service_id="timing_alive",
        )
        # SERVER_METRICS_MANAGER is an actual optional service started via
        # run_service() — not in the fixture's required_services.
        dead_optional = MultiProcessRunInfo.model_construct(
            process=mock_dead_process,
            service_type=ServiceType.SERVER_METRICS_MANAGER,
            service_id="server_metrics_dead",
        )
        service_manager.multi_process_info = [
            alive_dataset,
            alive_timing,
            dead_optional,
        ]
        # Mark the required services as registered so the wait loop succeeds
        # once the dead optional is dropped.
        from aiperf.common.enums import ServiceRegistrationStatus

        for info in (alive_dataset, alive_timing):
            registered = MagicMock()
            registered.service_type = info.service_type
            registered.registration_status = ServiceRegistrationStatus.REGISTERED
            service_manager.service_id_map[info.service_id] = registered

        # Should NOT raise — dead optional gets dropped, wait returns cleanly.
        await service_manager.wait_for_all_services_registration(
            stop_event=asyncio.Event(), timeout_seconds=2.0
        )

        # Dead optional was removed from the wait set.
        assert dead_optional not in service_manager.multi_process_info

    @pytest.mark.asyncio
    async def test_wait_blocks_until_optional_services_register(
        self, service_manager: MultiProcessServiceManager, mock_alive_process: MagicMock
    ):
        """Regression: optional services started via run_service() must also
        be waited for before ProfileConfigureCommand is broadcast.

        Failure mode: ServerMetricsManager (an optional service started via
        run_service, not part of required_services) registers ~1s later than
        the core services on slow Windows VDI. The SystemController previously
        only waited for required_services; it broadcast ProfileConfigureCommand
        before ServerMetricsManager had subscribed, leaving it un-configured
        and the JSON export file missing on disk.

        Now wait_for_all_services_registration derives its wait set from
        multi_process_info (every spawned service) instead of just
        required_services.
        """
        from aiperf.common.enums import ServiceRegistrationStatus
        from aiperf.common.models.service_models import ServiceRunInfo

        # required service already registered
        required_info = MultiProcessRunInfo.model_construct(
            process=mock_alive_process,
            service_type=ServiceType.DATASET_MANAGER,
            service_id="dataset_manager",
        )
        # optional service spawned via run_service() but not yet registered
        optional_info = MultiProcessRunInfo.model_construct(
            process=mock_alive_process,
            service_type=ServiceType.SERVER_METRICS_MANAGER,
            service_id="server_metrics_manager",
        )
        service_manager.multi_process_info = [required_info, optional_info]
        # Mark only the required one as REGISTERED — optional is still spawning.
        service_manager.service_id_map = {
            "dataset_manager": ServiceRunInfo(
                service_type=ServiceType.DATASET_MANAGER,
                registration_status=ServiceRegistrationStatus.REGISTERED,
                service_id="dataset_manager",
            ),
            "timing_manager": ServiceRunInfo(
                service_type=ServiceType.TIMING_MANAGER,
                registration_status=ServiceRegistrationStatus.REGISTERED,
                service_id="timing_manager",
            ),
        }

        # Pre-fix: wait would return ~immediately because required_services
        # are all registered. Post-fix: wait times out because the optional
        # SERVER_METRICS_MANAGER in multi_process_info isn't in service_id_map.
        with pytest.raises(AIPerfError, match="failed to register within timeout"):
            await service_manager.wait_for_all_services_registration(
                stop_event=asyncio.Event(), timeout_seconds=1.0
            )

    @pytest.mark.asyncio
    async def test_stop_event_cancels_registration_wait(
        self, service_manager: MultiProcessServiceManager, mock_alive_process: MagicMock
    ):
        """Test that setting the stop event cancels the registration wait gracefully."""
        # Sleep for a fraction of the time for faster test execution
        # Create an alive process that won't register (to test cancellation)
        alive_process_info = MultiProcessRunInfo.model_construct(
            process=mock_alive_process,
            service_type=ServiceType.DATASET_MANAGER,
            service_id="alive_but_not_registering",
        )
        service_manager.multi_process_info = [alive_process_info]

        stop_event = asyncio.Event()

        # Set the stop event after a short delay (use longer delay for CI stability)
        async def set_stop_event():
            await asyncio.sleep(0.1)
            stop_event.set()

        asyncio.create_task(set_stop_event())

        # This should exit early when the stop event is set, not wait for full timeout
        await service_manager.wait_for_all_services_registration(
            stop_event=stop_event, timeout_seconds=5.0
        )


class TestWaitForProcess:
    """Test _wait_for_process graceful shutdown and SIGKILL escalation."""

    @pytest.fixture
    def service_manager(self, benchmark_run) -> MultiProcessServiceManager:
        return MultiProcessServiceManager(
            required_services={ServiceType.DATASET_MANAGER: 1},
            run=benchmark_run,
        )

    @pytest.fixture
    def _make_process_info(self) -> "callable":
        def _factory(
            *, is_alive_sequence: list[bool], pid: int = 12345
        ) -> MultiProcessRunInfo:
            mock_process = MagicMock(spec=Process)
            mock_process.is_alive.side_effect = is_alive_sequence
            mock_process.pid = pid
            mock_process.join.return_value = None
            return MultiProcessRunInfo.model_construct(
                process=mock_process,
                service_type=ServiceType.DATASET_MANAGER,
                service_id="test_service",
            )

        return _factory

    @pytest.mark.asyncio
    async def test_skips_already_dead_process(
        self, service_manager: MultiProcessServiceManager
    ):
        """Process that is already dead should be skipped entirely."""
        info = MultiProcessRunInfo.model_construct(
            process=MagicMock(spec=Process, is_alive=MagicMock(return_value=False)),
            service_type=ServiceType.DATASET_MANAGER,
            service_id="already_dead",
        )
        await service_manager._wait_for_process(info)
        info.process.terminate.assert_not_called()
        info.process.kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_none_process(
        self, service_manager: MultiProcessServiceManager
    ):
        """None process (never started) should be skipped entirely."""
        info = MultiProcessRunInfo.model_construct(
            process=None,
            service_type=ServiceType.DATASET_MANAGER,
            service_id="none_process",
        )
        await service_manager._wait_for_process(info)

    @pytest.mark.asyncio
    async def test_terminate_succeeds_no_kill(
        self, service_manager: MultiProcessServiceManager, _make_process_info
    ):
        """Process that exits after SIGTERM should not be killed."""
        # First is_alive=True (guard check), second is_alive=False (after join)
        info = _make_process_info(is_alive_sequence=[True, False])

        await service_manager._wait_for_process(info)

        info.process.terminate.assert_called_once()
        info.process.join.assert_called_once()
        info.process.kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_terminate_fails_escalates_to_kill(
        self, service_manager: MultiProcessServiceManager, _make_process_info
    ):
        """Process still alive after join timeout should be killed with SIGKILL."""
        # First is_alive=True (guard check), second is_alive=True (after join — still running)
        info = _make_process_info(is_alive_sequence=[True, True])

        await service_manager._wait_for_process(info)

        info.process.terminate.assert_called_once()
        info.process.join.assert_called_once()
        info.process.kill.assert_called_once()
