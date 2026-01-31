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
    def service_manager(
        self, service_config, user_config
    ) -> MultiProcessServiceManager:
        """Create a MultiProcessServiceManager instance for testing."""
        return MultiProcessServiceManager(
            required_services={
                ServiceType.DATASET_MANAGER: 1,
                ServiceType.TIMING_MANAGER: 1,
            },
            service_config=service_config,
            user_config=user_config,
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

        # Expect an error due to the dead process
        with pytest.raises(
            AIPerfError,
            match="Service process dead_service_123 died before registering",
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

        # Should raise error about the dead process
        with pytest.raises(
            AIPerfError,
            match="Service process dead_service_789 died before registering",
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

        # Should raise error about the failed process
        with pytest.raises(
            AIPerfError,
            match="Service process failed_to_start_service died before registering",
        ):
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
