# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for KubernetesServiceManager.

Tests the Kubernetes service manager that treats controller services and
worker services as externally managed Kubernetes runtimes.
"""

import asyncio
import time
from multiprocessing import Process
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param

from aiperf.common.enums import LifecycleState
from aiperf.common.exceptions import (
    ServiceProcessDiedError,
    ServiceRegistrationTimeoutError,
)
from aiperf.common.service_registry import ServiceRegistry
from aiperf.common.subprocess_manager import SubprocessInfo, SubprocessManager
from aiperf.config import BenchmarkRun
from aiperf.controller.kubernetes_service_manager import (
    EXTERNAL_K8S_SERVICES,
    KubernetesServiceManager,
)
from aiperf.plugin.enums import ServiceType


class TestIsExternalService:
    """Test _is_external_service classification for all service types."""

    @pytest.fixture
    def manager(self, run: BenchmarkRun) -> KubernetesServiceManager:
        return KubernetesServiceManager(
            required_services={ServiceType.DATASET_MANAGER: 1},
            run=run,
        )

    @pytest.mark.parametrize(
        "service_type",
        [
            param(ServiceType.API, id="api"),
            param(ServiceType.DATASET_MANAGER, id="dataset_manager"),
            param(ServiceType.GPU_TELEMETRY_MANAGER, id="gpu_telemetry_manager"),
            param(ServiceType.RECORDS_MANAGER, id="records_manager"),
            param(ServiceType.SERVER_METRICS_MANAGER, id="server_metrics_manager"),
            param(ServiceType.TIMING_MANAGER, id="timing_manager"),
            param(ServiceType.WORKER, id="worker"),
            param(ServiceType.WORKER_MANAGER, id="worker_manager"),
            param(ServiceType.RECORD_PROCESSOR, id="record_processor"),
            param(ServiceType.WORKER_POD_MANAGER, id="worker_pod_manager"),
        ],
    )  # fmt: skip
    def test_external_services(
        self, manager: KubernetesServiceManager, service_type: ServiceType
    ) -> None:
        """External K8s services should return True."""
        assert manager._is_external_service(service_type) is True

    @pytest.mark.parametrize(
        "service_type",
        [
            param(ServiceType.SYSTEM_CONTROLLER, id="system_controller"),
        ],
    )  # fmt: skip
    def test_internal_services(
        self, manager: KubernetesServiceManager, service_type: ServiceType
    ) -> None:
        """Services not launched by the manager should return False."""
        assert manager._is_external_service(service_type) is False

    def test_external_set_is_frozen(self) -> None:
        """EXTERNAL_K8S_SERVICES should be an immutable frozenset."""
        assert isinstance(EXTERNAL_K8S_SERVICES, frozenset)
        assert len(EXTERNAL_K8S_SERVICES) == 10


class TestRunService:
    """Test run_service delegates or no-ops based on service type."""

    @pytest.fixture
    def manager(self, run: BenchmarkRun) -> KubernetesServiceManager:
        return KubernetesServiceManager(
            required_services={
                ServiceType.DATASET_MANAGER: 1,
                ServiceType.WORKER: 5,
            },
            run=run,
        )

    @pytest.mark.asyncio
    async def test_run_external_service_does_not_spawn(
        self, manager: KubernetesServiceManager
    ) -> None:
        """Running an external service should not spawn a subprocess."""
        with patch.object(
            manager._subprocess_manager, "spawn_service", new_callable=AsyncMock
        ) as mock_spawn:
            await manager.run_service(ServiceType.WORKER, num_replicas=5)
            mock_spawn.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_external_service_sets_registry_expectations(
        self, manager: KubernetesServiceManager
    ) -> None:
        """Running an external service should set expected count in ServiceRegistry."""
        await manager.run_service(ServiceType.WORKER, num_replicas=5)
        assert ServiceRegistry.expected_by_type.get(ServiceType.WORKER) == 5

    @pytest.mark.asyncio
    async def test_run_external_service_expectations_are_additive(
        self, manager: KubernetesServiceManager
    ) -> None:
        """Multiple run_service calls for external types should accumulate expectations."""
        await manager.run_service(ServiceType.WORKER, num_replicas=3)
        await manager.run_service(ServiceType.WORKER, num_replicas=2)
        assert ServiceRegistry.expected_by_type.get(ServiceType.WORKER) == 5

    @pytest.mark.asyncio
    async def test_run_internal_service_delegates_to_parent(
        self, manager: KubernetesServiceManager
    ) -> None:
        """Non-external services should still delegate to MultiProcessServiceManager."""
        with patch(
            "aiperf.controller.multiprocess_service_manager.MultiProcessServiceManager.run_service",
            new_callable=AsyncMock,
        ) as mock_parent:
            await manager.run_service(ServiceType.SYSTEM_CONTROLLER, num_replicas=1)
            mock_parent.assert_awaited_once_with(ServiceType.SYSTEM_CONTROLLER, 1)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "service_type",
        [
            param(ServiceType.DATASET_MANAGER, id="dataset_manager"),
            param(ServiceType.WORKER, id="worker"),
            param(ServiceType.RECORD_PROCESSOR, id="record_processor"),
            param(ServiceType.WORKER_POD_MANAGER, id="worker_pod_manager"),
        ],
    )  # fmt: skip
    async def test_all_external_types_do_not_spawn(
        self, manager: KubernetesServiceManager, service_type: ServiceType
    ) -> None:
        """All external service types should not spawn subprocesses."""
        with patch.object(
            manager._subprocess_manager, "spawn_service", new_callable=AsyncMock
        ) as mock_spawn:
            await manager.run_service(service_type)
            mock_spawn.assert_not_called()
            assert ServiceRegistry.expected_by_type.get(service_type) == 1


class TestStopService:
    """Test stop_service delegates or no-ops based on service type."""

    @pytest.fixture
    def manager(self, run: BenchmarkRun) -> KubernetesServiceManager:
        return KubernetesServiceManager(
            required_services={
                ServiceType.DATASET_MANAGER: 1,
                ServiceType.WORKER: 5,
            },
            run=run,
        )

    @pytest.mark.asyncio
    async def test_stop_external_service_returns_empty(
        self, manager: KubernetesServiceManager
    ) -> None:
        """Stopping an external service should return empty list (no-op)."""
        result = await manager.stop_service(ServiceType.WORKER)
        assert result == []

    @pytest.mark.asyncio
    async def test_stop_external_service_does_not_touch_subprocesses(
        self, manager: KubernetesServiceManager
    ) -> None:
        """Stopping an external service should not interact with subprocess manager."""
        with patch.object(
            manager._subprocess_manager, "stop_service", new_callable=AsyncMock
        ) as mock_stop:
            await manager.stop_service(ServiceType.RECORD_PROCESSOR)
            mock_stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_internal_service_delegates_to_parent(
        self, manager: KubernetesServiceManager
    ) -> None:
        """Stopping a non-external service should delegate to parent."""
        with patch(
            "aiperf.controller.multiprocess_service_manager.MultiProcessServiceManager.stop_service",
            new_callable=AsyncMock,
            return_value=[None],
        ) as mock_parent:
            result = await manager.stop_service(ServiceType.SYSTEM_CONTROLLER)
            mock_parent.assert_awaited_once_with(ServiceType.SYSTEM_CONTROLLER, None)
            assert result == [None]

    @pytest.mark.asyncio
    async def test_stop_internal_service_with_id(
        self, manager: KubernetesServiceManager
    ) -> None:
        """Stopping a specific service by ID should pass through to parent."""
        with patch(
            "aiperf.controller.multiprocess_service_manager.MultiProcessServiceManager.stop_service",
            new_callable=AsyncMock,
            return_value=[None],
        ) as mock_parent:
            await manager.stop_service(
                ServiceType.SYSTEM_CONTROLLER, service_id="controller_001"
            )
            mock_parent.assert_awaited_once_with(
                ServiceType.SYSTEM_CONTROLLER, "controller_001"
            )


class TestShutdownAllServices:
    """Test shutdown_all_services stops all except API."""

    @pytest.fixture
    def mock_process(self) -> MagicMock:
        proc = MagicMock(spec=Process)
        proc.is_alive.return_value = True
        proc.pid = 100
        return proc

    @pytest.fixture
    def manager(self, run: BenchmarkRun) -> KubernetesServiceManager:
        return KubernetesServiceManager(
            required_services={
                ServiceType.DATASET_MANAGER: 1,
                ServiceType.TIMING_MANAGER: 1,
                ServiceType.API: 1,
            },
            run=run,
        )

    @pytest.mark.asyncio
    async def test_shutdown_skips_api_service(
        self, manager: KubernetesServiceManager, mock_process: MagicMock
    ) -> None:
        """shutdown_all_services should stop all subprocesses EXCEPT API."""
        api_info = SubprocessInfo(
            process=mock_process, service_type=ServiceType.API, service_id="api_0"
        )
        dataset_info = SubprocessInfo(
            process=mock_process,
            service_type=ServiceType.DATASET_MANAGER,
            service_id="dm_0",
        )
        timing_info = SubprocessInfo(
            process=mock_process,
            service_type=ServiceType.TIMING_MANAGER,
            service_id="tm_0",
        )
        manager._subprocess_manager.subprocesses = [api_info, dataset_info, timing_info]

        with patch.object(
            manager._subprocess_manager, "stop_process", new_callable=AsyncMock
        ) as mock_stop:
            await manager.shutdown_all_services()

            # Should stop dataset and timing, but NOT api
            assert mock_stop.await_count == 2
            stopped_types = {
                call.args[0].service_type for call in mock_stop.await_args_list
            }
            assert ServiceType.API not in stopped_types
            assert ServiceType.DATASET_MANAGER in stopped_types
            assert ServiceType.TIMING_MANAGER in stopped_types

            # API should remain in subprocess list
            assert api_info in manager._subprocess_manager.subprocesses
            assert dataset_info not in manager._subprocess_manager.subprocesses
            assert timing_info not in manager._subprocess_manager.subprocesses

    @pytest.mark.asyncio
    async def test_shutdown_with_only_api_does_nothing(
        self, manager: KubernetesServiceManager, mock_process: MagicMock
    ) -> None:
        """If only API is running, shutdown should not stop anything."""
        api_info = SubprocessInfo(
            process=mock_process, service_type=ServiceType.API, service_id="api_0"
        )
        manager._subprocess_manager.subprocesses = [api_info]

        with patch.object(
            manager._subprocess_manager, "stop_process", new_callable=AsyncMock
        ) as mock_stop:
            await manager.shutdown_all_services()
            mock_stop.assert_not_awaited()
            assert api_info in manager._subprocess_manager.subprocesses

    @pytest.mark.asyncio
    async def test_shutdown_with_no_subprocesses(
        self, manager: KubernetesServiceManager
    ) -> None:
        """Shutdown with no subprocesses should return empty results."""
        manager._subprocess_manager.subprocesses = []
        results = await manager.shutdown_all_services()
        assert results == []


class TestWaitForApiSubprocess:
    """Test wait_for_api_subprocess blocking behavior."""

    @pytest.fixture
    def manager(self, run: BenchmarkRun) -> KubernetesServiceManager:
        return KubernetesServiceManager(
            required_services={ServiceType.API: 1},
            run=run,
        )

    @pytest.mark.asyncio
    async def test_no_api_process_returns_immediately(
        self, manager: KubernetesServiceManager
    ) -> None:
        """If no API subprocess exists, should return without blocking."""
        manager._subprocess_manager.subprocesses = []
        await manager.wait_for_api_subprocess()  # Should not hang

    @pytest.mark.asyncio
    async def test_api_process_none_returns_immediately(
        self, manager: KubernetesServiceManager
    ) -> None:
        """If API subprocess has process=None, should return without blocking."""
        info = SubprocessInfo(
            process=None, service_type=ServiceType.API, service_id="api_0"
        )
        manager._subprocess_manager.subprocesses = [info]
        await manager.wait_for_api_subprocess()

    @pytest.mark.asyncio
    async def test_waits_until_api_process_exits(
        self, manager: KubernetesServiceManager
    ) -> None:
        """Should poll is_alive() until API process terminates."""
        mock_process = MagicMock(spec=Process)
        # Process is alive for 2 checks, then dies
        mock_process.is_alive.side_effect = [True, True, False]
        mock_process.pid = 99999

        info = SubprocessInfo(
            process=mock_process, service_type=ServiceType.API, service_id="api_0"
        )
        manager._subprocess_manager.subprocesses = [info]

        await manager.wait_for_api_subprocess()
        assert mock_process.is_alive.call_count == 3

    @pytest.mark.asyncio
    async def test_ignores_non_api_subprocesses(
        self, manager: KubernetesServiceManager
    ) -> None:
        """Should only wait for API subprocess, not others."""
        mock_alive = MagicMock(spec=Process)
        mock_alive.is_alive.return_value = True
        mock_alive.pid = 11111

        non_api = SubprocessInfo(
            process=mock_alive,
            service_type=ServiceType.DATASET_MANAGER,
            service_id="dm_0",
        )
        manager._subprocess_manager.subprocesses = [non_api]

        # Should return immediately since no API process is found
        await manager.wait_for_api_subprocess()


class TestWaitForAllServicesRegistration:
    """Test registration wait using ServiceRegistry."""

    @pytest.fixture
    def manager(self, run: BenchmarkRun) -> KubernetesServiceManager:
        return KubernetesServiceManager(
            required_services={
                ServiceType.DATASET_MANAGER: 1,
                ServiceType.TIMING_MANAGER: 1,
                ServiceType.WORKER: 1,
            },
            run=run,
        )

    @pytest.mark.asyncio
    async def test_all_services_already_registered(
        self, manager: KubernetesServiceManager
    ) -> None:
        """Should return immediately when all required services are registered."""
        ServiceRegistry.expect_services(
            {
                ServiceType.DATASET_MANAGER: 1,
                ServiceType.TIMING_MANAGER: 1,
                ServiceType.WORKER: 1,
            }
        )
        for sid, stype in [
            ("dm_0", ServiceType.DATASET_MANAGER),
            ("tm_0", ServiceType.TIMING_MANAGER),
            ("w_0", ServiceType.WORKER),
        ]:
            ServiceRegistry.register(
                sid, stype, first_seen_ns=1000, state=LifecycleState.RUNNING
            )

        await manager.wait_for_all_services_registration(timeout_seconds=2.0)

    @pytest.mark.asyncio
    async def test_subprocess_dies_raises_error(
        self, manager: KubernetesServiceManager
    ) -> None:
        """If a required service process dies, ServiceRegistry should raise."""
        ServiceRegistry.expect_service("dead_dm", ServiceType.DATASET_MANAGER)

        ServiceRegistry.fail_service("dead_dm", ServiceType.DATASET_MANAGER)

        with pytest.raises(ServiceProcessDiedError, match="dead_dm"):
            await manager.wait_for_all_services_registration(timeout_seconds=2.0)

    @pytest.mark.asyncio
    @pytest.mark.looptime
    async def test_timeout_when_services_missing(
        self, manager: KubernetesServiceManager
    ) -> None:
        """Should raise TimeoutError when services don't register in time."""
        ServiceRegistry.expect_services({ServiceType.DATASET_MANAGER: 1})

        with pytest.raises(ServiceRegistrationTimeoutError):
            await manager.wait_for_all_services_registration(timeout_seconds=1)

    @pytest.mark.asyncio
    async def test_gradual_registration_succeeds(
        self, manager: KubernetesServiceManager
    ) -> None:
        """Services registering over time should eventually satisfy the wait."""
        ServiceRegistry.expect_services(
            {
                ServiceType.DATASET_MANAGER: 1,
                ServiceType.WORKER: 1,
            }
        )

        # Register one immediately
        ServiceRegistry.register(
            "dm_0",
            ServiceType.DATASET_MANAGER,
            first_seen_ns=1000,
            state=LifecycleState.RUNNING,
        )

        async def register_remaining() -> None:
            await asyncio.sleep(0.05)
            ServiceRegistry.register(
                "w_0",
                ServiceType.WORKER,
                first_seen_ns=2000,
                state=LifecycleState.RUNNING,
            )

        asyncio.create_task(register_remaining())

        await manager.wait_for_all_services_registration(timeout_seconds=2.0)


class TestKubernetesServiceManagerInit:
    """Test construction and inheritance."""

    def test_inherits_from_multiprocess(self, run: BenchmarkRun) -> None:
        """KubernetesServiceManager should be a subclass of MultiProcessServiceManager."""
        from aiperf.controller.multiprocess_service_manager import (
            MultiProcessServiceManager,
        )

        manager = KubernetesServiceManager(
            required_services={ServiceType.DATASET_MANAGER: 1},
            run=run,
        )
        assert isinstance(manager, MultiProcessServiceManager)

    def test_has_subprocess_manager(self, run: BenchmarkRun) -> None:
        """Should have a SubprocessManager from parent init."""
        manager = KubernetesServiceManager(
            required_services={ServiceType.DATASET_MANAGER: 1},
            run=run,
        )
        assert isinstance(manager._subprocess_manager, SubprocessManager)


class TestHeartbeatMonitoring:
    """Test _monitor_heartbeats detects stale external services."""

    @pytest.fixture
    def manager(self, run: BenchmarkRun) -> KubernetesServiceManager:
        mgr = KubernetesServiceManager(
            required_services={ServiceType.DATASET_MANAGER: 1},
            run=run,
        )
        mgr.activate_heartbeat_monitoring()
        return mgr

    @pytest.mark.asyncio
    async def test_stale_external_service_triggers_failure(
        self, manager: KubernetesServiceManager
    ) -> None:
        """External service with missed heartbeats should be marked as failed."""
        old_ns = time.time_ns() - 60_000_000_000  # 60 seconds ago
        ServiceRegistry.expect_services({ServiceType.WORKER: 1})
        ServiceRegistry.register(
            "worker_001",
            ServiceType.WORKER,
            first_seen_ns=old_ns,
            state=LifecycleState.RUNNING,
        )

        await manager._monitor_heartbeats()

        info = ServiceRegistry.get_service("worker_001")
        assert info is not None
        assert info.state == LifecycleState.FAILED

    @pytest.mark.asyncio
    async def test_fresh_external_service_not_affected(
        self, manager: KubernetesServiceManager
    ) -> None:
        """External service with recent heartbeat should not be marked as failed."""
        ServiceRegistry.expect_services({ServiceType.WORKER: 1})
        ServiceRegistry.register(
            "worker_001",
            ServiceType.WORKER,
            first_seen_ns=time.time_ns(),
            state=LifecycleState.RUNNING,
        )

        await manager._monitor_heartbeats()

        assert ServiceRegistry.is_registered("worker_001")

    @pytest.mark.asyncio
    async def test_stale_subprocess_service_also_detected(
        self, manager: KubernetesServiceManager
    ) -> None:
        """Stale control-plane subprocesses should also be caught by heartbeat monitor.

        A subprocess can be alive but hung (not sending heartbeats).
        _monitor_processes only catches crashed processes.
        """
        old_ns = time.time_ns() - 60_000_000_000
        ServiceRegistry.expect_services({ServiceType.DATASET_MANAGER: 1})
        ServiceRegistry.register(
            "dm_0",
            ServiceType.DATASET_MANAGER,
            first_seen_ns=old_ns,
            state=LifecycleState.RUNNING,
        )

        await manager._monitor_heartbeats()

        info = ServiceRegistry.get_service("dm_0")
        assert info is not None
        assert info.state == LifecycleState.FAILED


class TestMonitorWorkerPods:
    """Test _monitor_worker_pods detects failed Kubernetes pods."""

    @pytest.fixture
    def manager(self, run: BenchmarkRun) -> KubernetesServiceManager:
        mgr = KubernetesServiceManager(
            required_services={ServiceType.DATASET_MANAGER: 1},
            run=run,
        )
        mgr.activate_heartbeat_monitoring()
        return mgr

    @pytest.mark.asyncio
    async def test_skips_when_no_env_vars(
        self, manager: KubernetesServiceManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should be a no-op when AIPERF_NAMESPACE/AIPERF_JOB_ID are not set."""
        monkeypatch.delenv("AIPERF_NAMESPACE", raising=False)
        monkeypatch.delenv("AIPERF_JOB_ID", raising=False)
        # Should return without error
        await manager._monitor_worker_pods()

    @pytest.mark.asyncio
    async def test_skips_when_shutdown(
        self, manager: KubernetesServiceManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should skip monitoring during shutdown."""
        monkeypatch.setenv("AIPERF_NAMESPACE", "test-ns")
        monkeypatch.setenv("AIPERF_JOB_ID", "test-job")
        manager._shutdown_complete = True
        # Should return without querying K8s API
        await manager._monitor_worker_pods()

    @pytest.mark.asyncio
    async def test_failed_pod_marks_services_failed(
        self, manager: KubernetesServiceManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Failed K8s pod should mark its services as failed in ServiceRegistry."""
        monkeypatch.setenv("AIPERF_NAMESPACE", "test-ns")
        monkeypatch.setenv("AIPERF_JOB_ID", "test-job")

        ServiceRegistry.expect_services({ServiceType.WORKER: 2})
        for i in range(2):
            ServiceRegistry.register(
                f"worker_0_{i}",
                ServiceType.WORKER,
                first_seen_ns=time.time_ns(),
                state=LifecycleState.RUNNING,
                pod_index="0",
            )

        mock_pod = MagicMock()
        mock_pod.raw = {
            "metadata": {
                "name": "aiperf-test-worker-0-0-abc",
                "labels": {"jobset.sigs.k8s.io/job-index": "0"},
            },
            "status": {"phase": "Failed"},
        }

        mock_client = AsyncMock()
        mock_client.job_selector.return_value = (
            "app=aiperf,aiperf.nvidia.com/job-id=test-job"
        )
        mock_client.get_pods.return_value = [mock_pod]

        with patch(
            "aiperf.kubernetes.client.AIPerfKubeClient.create",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            await manager._monitor_worker_pods()

        for i in range(2):
            info = ServiceRegistry.get_service(f"worker_0_{i}")
            assert info is not None
            assert info.state == LifecycleState.FAILED

    @pytest.mark.asyncio
    async def test_replacement_running_pod_prevents_stale_failed_pod_from_failing_services(
        self, manager: KubernetesServiceManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stale failed pod should be ignored when the same pod index has a running replacement."""
        monkeypatch.setenv("AIPERF_NAMESPACE", "test-ns")
        monkeypatch.setenv("AIPERF_JOB_ID", "test-job")

        ServiceRegistry.expect_services({ServiceType.WORKER: 1})
        ServiceRegistry.register(
            "worker_0_0",
            ServiceType.WORKER,
            first_seen_ns=time.time_ns(),
            state=LifecycleState.RUNNING,
            pod_index="0",
        )

        failed_pod = MagicMock()
        failed_pod.raw = {
            "metadata": {
                "name": "aiperf-test-worker-0-0-old",
                "labels": {"jobset.sigs.k8s.io/job-index": "0"},
            },
            "status": {"phase": "Failed"},
        }

        running_pod = MagicMock()
        running_pod.raw = {
            "metadata": {
                "name": "aiperf-test-worker-0-0-new",
                "labels": {"jobset.sigs.k8s.io/job-index": "0"},
            },
            "status": {"phase": "Running"},
        }

        mock_client = AsyncMock()
        mock_client.job_selector.return_value = "app=aiperf"
        mock_client.get_pods.return_value = [failed_pod, running_pod]

        with patch(
            "aiperf.kubernetes.client.AIPerfKubeClient.create",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            await manager._monitor_worker_pods()

        info = ServiceRegistry.get_service("worker_0_0")
        assert info is not None
        assert info.state == LifecycleState.RUNNING

    @pytest.mark.asyncio
    async def test_running_pod_does_not_affect_services(
        self, manager: KubernetesServiceManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Running K8s pod should not mark services as failed."""
        monkeypatch.setenv("AIPERF_NAMESPACE", "test-ns")
        monkeypatch.setenv("AIPERF_JOB_ID", "test-job")

        ServiceRegistry.expect_services({ServiceType.WORKER: 1})
        ServiceRegistry.register(
            "worker_0_0",
            ServiceType.WORKER,
            first_seen_ns=time.time_ns(),
            state=LifecycleState.RUNNING,
            pod_index="0",
        )

        mock_pod = MagicMock()
        mock_pod.raw = {
            "metadata": {
                "name": "aiperf-test-worker-0-0-abc",
                "labels": {"jobset.sigs.k8s.io/job-index": "0"},
            },
            "status": {"phase": "Running"},
        }

        mock_client = AsyncMock()
        mock_client.job_selector.return_value = "app=aiperf"
        mock_client.get_pods.return_value = [mock_pod]

        with patch(
            "aiperf.kubernetes.client.AIPerfKubeClient.create",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            await manager._monitor_worker_pods()

        assert ServiceRegistry.is_registered("worker_0_0")

    @pytest.mark.asyncio
    async def test_api_error_is_swallowed(
        self, manager: KubernetesServiceManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """K8s API errors should be caught and not propagate."""
        monkeypatch.setenv("AIPERF_NAMESPACE", "test-ns")
        monkeypatch.setenv("AIPERF_JOB_ID", "test-job")

        with patch(
            "aiperf.kubernetes.client.AIPerfKubeClient.create",
            new_callable=AsyncMock,
            side_effect=RuntimeError("k8s unreachable"),
        ):
            await manager._monitor_worker_pods()
