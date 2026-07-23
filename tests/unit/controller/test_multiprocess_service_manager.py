# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
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
    """Bug 1 regression: ``ForkProcess`` import was Linux-only."""

    def test_module_imports_on_any_platform(self) -> None:
        """Importing this module must succeed on every platform AIPerf"""
        from aiperf.controller import multiprocess_service_manager

        assert multiprocess_service_manager.MultiProcessRunInfo is not None

    def test_field_accepts_a_plain_process(self) -> None:
        """The ``process`` field accepts any subclass of Process, including"""
        from multiprocessing import Process

        info = MultiProcessRunInfo.model_construct(
            process=Process(target=lambda: None),
            service_type=ServiceType.SYSTEM_CONTROLLER,
            run_id="test",
        )
        assert info.process is not None


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
        """Test that MultiProcessServiceManager raises AIPerfError when a process dies before registering."""
        dead_process_info = MultiProcessRunInfo.model_construct(
            process=mock_dead_process,
            service_type=ServiceType.DATASET_MANAGER,
            service_id="dead_service_123",
        )
        service_manager.multi_process_info = [dead_process_info]

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
        none_process_info = MultiProcessRunInfo.model_construct(
            process=None,
            service_type=ServiceType.DATASET_MANAGER,
            service_id="failed_to_start_service",
        )
        service_manager.multi_process_info = [none_process_info]

        with pytest.raises(
            AIPerfError,
            match="Required service failed_to_start_service died before registering",
        ):
            await service_manager.wait_for_all_services_registration(
                stop_event=asyncio.Event(), timeout_seconds=1.0
            )

    @pytest.mark.asyncio
    async def test_run_service_passes_controller_pid_for_pdeathsig_guard(
        self,
        service_manager: MultiProcessServiceManager,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Every child Process must receive the controller's PID so bootstrap"""
        mock_process_cls = MagicMock(return_value=MagicMock(spec=Process))
        monkeypatch.setattr(
            "aiperf.controller.multiprocess_service_manager.Process",
            mock_process_cls,
        )

        await service_manager.run_service(ServiceType.DATASET_MANAGER)

        mock_process_cls.assert_called_once()
        launch_kwargs = mock_process_cls.call_args.kwargs
        assert launch_kwargs["daemon"] is True
        assert launch_kwargs["kwargs"]["controller_pid"] == os.getpid()
        mock_process_cls.return_value.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_optional_dead_drops_and_continues(
        self,
        service_manager: MultiProcessServiceManager,
        mock_alive_process: MagicMock,
        mock_dead_process: MagicMock,
    ):
        """Pins F-04: an optional service (not in required_services) dying"""
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
        from aiperf.common.enums import ServiceRegistrationStatus

        for info in (alive_dataset, alive_timing):
            registered = MagicMock()
            registered.service_type = info.service_type
            registered.registration_status = ServiceRegistrationStatus.REGISTERED
            service_manager.service_id_map[info.service_id] = registered

        await service_manager.wait_for_all_services_registration(
            stop_event=asyncio.Event(), timeout_seconds=2.0
        )

        assert dead_optional not in service_manager.multi_process_info

    @pytest.mark.asyncio
    async def test_wait_blocks_until_optional_services_register(
        self, service_manager: MultiProcessServiceManager, mock_alive_process: MagicMock
    ):
        """Regression: optional services started via run_service() must also"""
        from aiperf.common.enums import ServiceRegistrationStatus
        from aiperf.common.models.service_models import ServiceRunInfo

        required_info = MultiProcessRunInfo.model_construct(
            process=mock_alive_process,
            service_type=ServiceType.DATASET_MANAGER,
            service_id="dataset_manager",
        )
        optional_info = MultiProcessRunInfo.model_construct(
            process=mock_alive_process,
            service_type=ServiceType.SERVER_METRICS_MANAGER,
            service_id="server_metrics_manager",
        )
        service_manager.multi_process_info = [required_info, optional_info]
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

        with pytest.raises(AIPerfError, match="failed to register within timeout"):
            await service_manager.wait_for_all_services_registration(
                stop_event=asyncio.Event(), timeout_seconds=1.0
            )

    @pytest.mark.asyncio
    async def test_stop_event_cancels_registration_wait(
        self, service_manager: MultiProcessServiceManager, mock_alive_process: MagicMock
    ):
        """Test that setting the stop event cancels the registration wait gracefully."""
        alive_process_info = MultiProcessRunInfo.model_construct(
            process=mock_alive_process,
            service_type=ServiceType.DATASET_MANAGER,
            service_id="alive_but_not_registering",
        )
        service_manager.multi_process_info = [alive_process_info]

        stop_event = asyncio.Event()

        async def set_stop_event():
            await asyncio.sleep(0.1)
            stop_event.set()

        asyncio.create_task(set_stop_event())

        await service_manager.wait_for_all_services_registration(
            stop_event=stop_event, timeout_seconds=5.0
        )


class TestWaitForProcess:
    """Test _wait_for_process force-kill after bus shutdown grace."""

    @pytest.fixture
    def service_manager(self, benchmark_run) -> MultiProcessServiceManager:
        return MultiProcessServiceManager(
            required_services={ServiceType.DATASET_MANAGER: 1},
            run=benchmark_run,
        )

    @pytest.fixture
    def _make_process_info(self) -> "callable":
        def _factory(*, is_alive: bool = True, pid: int = 12345) -> MultiProcessRunInfo:
            mock_process = MagicMock(spec=Process)
            mock_process.is_alive.return_value = is_alive
            mock_process.pid = pid
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
    async def test_alive_process_skips_terminate_goes_straight_to_kill(
        self, service_manager: MultiProcessServiceManager, _make_process_info
    ):
        """Alive straggler is killed immediately; terminate must not run."""
        info = _make_process_info(is_alive=True)
        info.process.is_alive.side_effect = [True, False]

        await service_manager._wait_for_process(info)

        info.process.terminate.assert_not_called()
        info.process.kill.assert_called_once()
        info.process.join.assert_called_once()
        method_names = [c[0] for c in info.process.method_calls]
        assert method_names.index("kill") < method_names.index("join")
