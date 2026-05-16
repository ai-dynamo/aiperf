# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from multiprocessing import Process
from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.enums import LifecycleState
from aiperf.common.environment import Environment
from aiperf.common.exceptions import (
    ServiceProcessDiedError,
    ServiceRegistrationTimeoutError,
)
from aiperf.common.service_registry import ServiceRegistry
from aiperf.common.subprocess_manager import SubprocessInfo
from aiperf.controller.multiprocess_service_manager import (
    MultiProcessServiceManager,
)
from aiperf.plugin.enums import ServiceType


class TestMultiProcessServiceManager:
    """Test MultiProcessServiceManager process lifecycle and monitoring."""

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
        self,
        run,
    ) -> MultiProcessServiceManager:
        """Create a MultiProcessServiceManager instance for testing."""
        return MultiProcessServiceManager(
            required_services={
                ServiceType.DATASET_MANAGER: 1,
                ServiceType.TIMING_MANAGER: 1,
            },
            run=run,
        )

    @pytest.mark.asyncio
    async def test_monitor_detects_dead_required_process_and_fails_registry(
        self, service_manager: MultiProcessServiceManager, mock_dead_process: MagicMock
    ):
        """Test that _monitor_processes detects a dead required process and calls fail_service."""
        ServiceRegistry.expect_service("dead_service_123", ServiceType.DATASET_MANAGER)

        service_manager._subprocess_manager.subprocesses = [
            SubprocessInfo(
                process=mock_dead_process,
                service_type=ServiceType.DATASET_MANAGER,
                service_id="dead_service_123",
            )
        ]

        mock_metadata = MagicMock()
        mock_metadata.required = True

        with patch(
            "aiperf.plugin.plugins.get_service_metadata",
            return_value=mock_metadata,
        ):
            await service_manager._monitor_processes()

        # The process should have been cleaned up
        assert len(service_manager._subprocess_manager.subprocesses) == 0

        # The registry should have a failure recorded
        with pytest.raises(ServiceProcessDiedError, match="dead_service_123"):
            ServiceRegistry._raise_on_failure()

    @pytest.mark.asyncio
    async def test_monitor_detects_dead_optional_process_and_forgets(
        self, service_manager: MultiProcessServiceManager, mock_dead_process: MagicMock
    ):
        """Test that _monitor_processes forgets optional dead processes without failing."""
        ServiceRegistry.expect_service("optional_123", ServiceType.DATASET_MANAGER)

        service_manager._subprocess_manager.subprocesses = [
            SubprocessInfo(
                process=mock_dead_process,
                service_type=ServiceType.DATASET_MANAGER,
                service_id="optional_123",
            )
        ]

        mock_metadata = MagicMock()
        mock_metadata.required = False

        with patch(
            "aiperf.plugin.plugins.get_service_metadata",
            return_value=mock_metadata,
        ):
            await service_manager._monitor_processes()

        assert len(service_manager._subprocess_manager.subprocesses) == 0
        assert ServiceRegistry.get_service("optional_123") is None

    @pytest.mark.asyncio
    @pytest.mark.looptime
    async def test_wait_for_all_times_out_when_no_services_register(
        self, service_manager: MultiProcessServiceManager
    ):
        """Test that wait_for_all raises TimeoutError when services don't register."""
        ServiceRegistry.expect_services({ServiceType.DATASET_MANAGER: 1})

        with pytest.raises(ServiceRegistrationTimeoutError):
            await service_manager.wait_for_all_services_registration(
                timeout_seconds=1,
            )

    @pytest.mark.asyncio
    async def test_wait_for_all_succeeds_when_services_register(
        self, service_manager: MultiProcessServiceManager
    ):
        """Test that wait_for_all completes when expected services register."""
        ServiceRegistry.expect_services({ServiceType.DATASET_MANAGER: 1})

        # Simulate registration after a short delay
        async def register_service():
            await asyncio.sleep(0.05)
            ServiceRegistry.register(
                "dm_001",
                ServiceType.DATASET_MANAGER,
                first_seen_ns=1000,
                state=LifecycleState.RUNNING,
            )

        asyncio.create_task(register_service())

        await service_manager.wait_for_all_services_registration(
            timeout_seconds=2.0,
        )

    @pytest.mark.asyncio
    async def test_run_service_calls_expect_service(
        self, service_manager: MultiProcessServiceManager
    ):
        """Test that run_service registers expectations with ServiceRegistry."""
        mock_metadata = MagicMock()
        mock_metadata.replicable = False

        mock_proc = MagicMock()
        mock_proc.pid = 9999
        mock_context = MagicMock()
        mock_context.Process.return_value = mock_proc

        with (
            patch(
                "aiperf.plugin.plugins.get_service_metadata",
                return_value=mock_metadata,
            ),
            patch(
                "aiperf.common.subprocess_manager.get_mp_context",
                return_value=mock_context,
            ),
        ):
            await service_manager.run_service(ServiceType.DATASET_MANAGER)

        # ServiceRegistry should have the expectation
        assert ServiceType.DATASET_MANAGER in ServiceRegistry.expected_by_type
        assert ServiceRegistry.expected_by_type[ServiceType.DATASET_MANAGER] == 1

    @pytest.mark.asyncio
    async def test_heartbeat_monitor_detects_stale_services(
        self, service_manager: MultiProcessServiceManager
    ):
        """Stale services should be marked as failed by heartbeat monitor
        after two consecutive stale ticks (two-strike verification)."""
        import time

        old_ns = time.time_ns() - 60_000_000_000
        ServiceRegistry.expect_services({ServiceType.DATASET_MANAGER: 1})
        ServiceRegistry.register(
            "dm_0",
            ServiceType.DATASET_MANAGER,
            first_seen_ns=old_ns,
            state=LifecycleState.RUNNING,
        )

        service_manager.activate_heartbeat_monitoring()
        # First tick: suspected but not failed (first strike).
        await service_manager._monitor_heartbeats()
        info = ServiceRegistry.get_service("dm_0")
        assert info is not None
        assert info.state == LifecycleState.RUNNING
        assert "dm_0" in service_manager._suspected_stale

        # Second tick: now failed (second strike).
        await service_manager._monitor_heartbeats()
        info = ServiceRegistry.get_service("dm_0")
        assert info is not None
        assert info.state == LifecycleState.FAILED

    @pytest.mark.asyncio
    async def test_heartbeat_monitor_two_strike_clears_on_fresh_heartbeat(
        self, service_manager: MultiProcessServiceManager
    ):
        """A service stale on one tick then heartbeats before next tick is NOT failed."""
        import time

        old_ns = time.time_ns() - 60_000_000_000
        ServiceRegistry.expect_services({ServiceType.DATASET_MANAGER: 1})
        ServiceRegistry.register(
            "dm_0",
            ServiceType.DATASET_MANAGER,
            first_seen_ns=old_ns,
            state=LifecycleState.RUNNING,
        )

        service_manager.activate_heartbeat_monitoring()
        # First tick: first strike recorded, service NOT failed.
        await service_manager._monitor_heartbeats()
        assert "dm_0" in service_manager._suspected_stale
        info = ServiceRegistry.get_service("dm_0")
        assert info is not None
        assert info.state == LifecycleState.RUNNING

        # Simulate a fresh heartbeat arriving before the next tick.
        ServiceRegistry.update_service(
            "dm_0",
            ServiceType.DATASET_MANAGER,
            last_seen_ns=time.time_ns(),
            state=LifecycleState.RUNNING,
        )

        # Second tick: service is no longer stale — strike cleared, still not failed.
        await service_manager._monitor_heartbeats()
        assert "dm_0" not in service_manager._suspected_stale
        info = ServiceRegistry.get_service("dm_0")
        assert info is not None
        assert info.state == LifecycleState.RUNNING

    @pytest.mark.asyncio
    async def test_heartbeat_monitor_skips_on_catch_up_tick(
        self,
        service_manager: MultiProcessServiceManager,
        caplog: pytest.LogCaptureFixture,
    ):
        """If the monitor tick was delayed (controller stall), death decisions
        are skipped for this iteration and a WARNING is logged."""
        import logging
        import time

        old_ns = time.time_ns() - 60_000_000_000
        ServiceRegistry.expect_services({ServiceType.DATASET_MANAGER: 1})
        ServiceRegistry.register(
            "dm_0",
            ServiceType.DATASET_MANAGER,
            first_seen_ns=old_ns,
            state=LifecycleState.RUNNING,
        )

        service_manager.activate_heartbeat_monitoring()
        # First tick: records strike 1 and the tick time.
        await service_manager._monitor_heartbeats()
        assert "dm_0" in service_manager._suspected_stale

        # Simulate a long stall: backdate the last tick time so the gap
        # exceeds HEARTBEAT_INTERVAL * 2.
        interval_ns = int(Environment.SERVICE.HEARTBEAT_INTERVAL * 1_000_000_000)
        service_manager._last_heartbeat_tick_ns = time.time_ns() - (interval_ns * 5)

        with (
            caplog.at_level(logging.WARNING),
            patch.object(ServiceRegistry, "fail_service") as mock_fail,
        ):
            await service_manager._monitor_heartbeats()
            assert mock_fail.call_count == 0

        assert any(
            "watchdog tick delayed" in record.message.lower()
            for record in caplog.records
        )
        # Suspected-stale state must be cleared so we don't carry forward
        # a bogus strike from the stalled window.
        assert service_manager._suspected_stale == {}
        info = ServiceRegistry.get_service("dm_0")
        assert info is not None
        assert info.state == LifecycleState.RUNNING
