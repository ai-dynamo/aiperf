# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for WorkerPodManager service.

WorkerPodManager runs in worker pods and spawns workers and record processors
as subprocesses, enabling efficient resource sharing within a pod.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param

from aiperf.common.control_structs import Command
from aiperf.common.enums import CommandType, WorkerStartupState
from aiperf.common.environment import Environment
from aiperf.common.messages import DatasetConfiguredNotification
from aiperf.common.messages.worker_messages import WorkerStartupStateMessage
from aiperf.common.models import (
    DatasetMetadata,
    MemoryMapClientMetadata,
    ProcessHealth,
    WorkerStats,
    WorkerTaskStats,
)
from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.controller.proxy_manager import ProxyManager
from aiperf.plugin.enums import DatasetSamplingStrategy, ServiceType
from aiperf.workers.worker_pod_manager import WorkerPodManager

# =============================================================================
# Helpers
# =============================================================================

_MINIMAL_CONFIG_KWARGS = {
    "models": ["test-model"],
    "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
    "datasets": {
        "main": {
            "type": "synthetic",
            "entries": 10,
            "prompts": {"isl": 32, "osl": 16},
        }
    },
    "phases": {
        "default": {
            "type": "concurrency",
            "requests": 10,
            "concurrency": 1,
        }
    },
}


def _make_run(
    workers_per_pod: int | None = None,
    record_processors_per_pod: int | None = None,
) -> BenchmarkRun:
    """Build a BenchmarkRun with optional worker pod settings."""
    runtime: dict = {}
    if workers_per_pod is not None:
        runtime["workers_per_pod"] = workers_per_pod
    if record_processors_per_pod is not None:
        runtime["record_processors_per_pod"] = record_processors_per_pod
    cfg = AIPerfConfig(
        **_MINIMAL_CONFIG_KWARGS,
        **({"runtime": runtime} if runtime else {}),
    )
    return BenchmarkRun(
        benchmark_id="test",
        cfg=cfg,
        artifact_dir=Path("/tmp/test"),
    )


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def run() -> BenchmarkRun:
    """Create a minimal BenchmarkRun for testing."""
    return _make_run()


@pytest.fixture
def run_with_workers() -> BenchmarkRun:
    """Create a BenchmarkRun with explicit worker settings."""
    return _make_run(workers_per_pod=8, record_processors_per_pod=2)


@pytest.fixture
def worker_pod_manager(run: BenchmarkRun) -> WorkerPodManager:
    """Create a WorkerPodManager instance for testing."""
    with (
        patch.object(WorkerPodManager, "debug"),
        patch.object(WorkerPodManager, "info"),
        patch.object(WorkerPodManager, "warning"),
        patch.object(WorkerPodManager, "error"),
    ):
        manager = WorkerPodManager(
            run=run,
            service_id="test-pod-manager",
        )
        return manager


@pytest.fixture
def worker_pod_manager_custom(run_with_workers: BenchmarkRun) -> WorkerPodManager:
    """Create a WorkerPodManager with custom worker configuration."""
    with (
        patch.object(WorkerPodManager, "debug"),
        patch.object(WorkerPodManager, "info"),
        patch.object(WorkerPodManager, "warning"),
        patch.object(WorkerPodManager, "error"),
    ):
        manager = WorkerPodManager(
            run=run_with_workers,
            service_id="test-pod-manager",
        )
        return manager


@pytest.fixture
def dataset_notification() -> DatasetConfiguredNotification:
    """Create a valid DatasetConfiguredNotification for testing."""
    return DatasetConfiguredNotification(
        service_id="test-dataset-manager",
        metadata=DatasetMetadata(
            conversations=[],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        ),
        client_metadata=MemoryMapClientMetadata(
            data_file_path=Path("/tmp/test_data.mmap"),
            index_file_path=Path("/tmp/test_index.mmap"),
            conversation_count=0,
            total_size_bytes=0,
        ),
    )


@pytest.fixture
def shutdown_command() -> Command:
    """Create a valid shutdown Command for testing."""
    return Command(cid="test", cmd=CommandType.SHUTDOWN)


# =============================================================================
# Initialization Tests
# =============================================================================


class TestWorkerPodManagerInit:
    """Tests for WorkerPodManager initialization."""

    def test_default_workers_per_pod(
        self, worker_pod_manager: WorkerPodManager
    ) -> None:
        """Test default workers_per_pod uses environment setting."""
        expected = Environment.WORKER.DEFAULT_WORKERS_PER_POD
        assert worker_pod_manager.workers_per_pod == expected

    def test_custom_workers_per_pod(
        self, worker_pod_manager_custom: WorkerPodManager
    ) -> None:
        """Test custom workers_per_pod from config."""
        assert worker_pod_manager_custom.workers_per_pod == 8

    def test_custom_record_processors_per_pod(
        self, worker_pod_manager_custom: WorkerPodManager
    ) -> None:
        """Test custom record_processors_per_pod from config."""
        assert worker_pod_manager_custom.record_processors_per_pod == 2

    @pytest.mark.parametrize(
        ("workers", "expected_rps"),
        [
            param(1, 1, id="min_one_rp"),
            param(2, 1, id="two_workers"),
            param(4, 1, id="four_workers"),
            param(8, 2, id="eight_workers"),
            param(12, 3, id="twelve_workers"),
            param(16, 4, id="sixteen_workers"),
        ],
    )  # fmt: skip
    def test_default_record_processors_calculation(
        self, workers: int, expected_rps: int
    ) -> None:
        """Test record processors default to workers / PROCESSOR_SCALE_FACTOR."""
        test_run = _make_run(workers_per_pod=workers)

        with (
            patch.object(WorkerPodManager, "debug"),
            patch.object(WorkerPodManager, "info"),
        ):
            manager = WorkerPodManager(
                run=test_run,
                service_id="test",
            )

        assert manager.record_processors_per_pod == expected_rps

    def test_subprocess_manager_created(
        self, worker_pod_manager: WorkerPodManager
    ) -> None:
        """Test SubprocessManager is created during init."""
        assert worker_pod_manager._subprocess_manager is not None

    def test_initial_state(self, worker_pod_manager: WorkerPodManager) -> None:
        """Test initial state is correct."""
        assert worker_pod_manager._dataset_downloaded is False
        assert worker_pod_manager.worker_health == {}

    def test_proxy_manager_created(self, worker_pod_manager: WorkerPodManager) -> None:
        """Test ProxyManager is created during init."""
        assert isinstance(worker_pod_manager._proxy_manager, ProxyManager)

    def test_proxy_manager_enables_only_raw_inference(
        self, worker_pod_manager: WorkerPodManager
    ) -> None:
        """Test ProxyManager only enables the raw inference proxy."""
        pm = worker_pod_manager._proxy_manager
        assert pm._enable_raw_inference is True
        assert pm._enable_event_bus is False
        assert pm._enable_dataset_manager is False


# =============================================================================
# Subprocess Spawning Tests
# =============================================================================


class TestSubprocessSpawning:
    """Tests for spawning worker and record processor subprocesses."""

    @pytest.mark.asyncio
    async def test_spawn_correct_number_of_workers(
        self, worker_pod_manager_custom: WorkerPodManager
    ) -> None:
        """Test spawning creates correct number of worker subprocesses."""
        manager = worker_pod_manager_custom
        manager._subprocess_manager.spawn_service = AsyncMock()

        await manager._spawn_subprocesses()

        # Count worker spawn calls
        worker_calls = [
            c
            for c in manager._subprocess_manager.spawn_service.call_args_list
            if c.kwargs.get("service_type") == ServiceType.WORKER
        ]
        assert len(worker_calls) == 8

    @pytest.mark.asyncio
    async def test_spawn_correct_number_of_record_processors(
        self, worker_pod_manager_custom: WorkerPodManager
    ) -> None:
        """Test spawning creates correct number of record processor subprocesses."""
        manager = worker_pod_manager_custom
        manager._subprocess_manager.spawn_service = AsyncMock()

        await manager._spawn_subprocesses()

        # Count RP spawn calls
        rp_calls = [
            c
            for c in manager._subprocess_manager.spawn_service.call_args_list
            if c.kwargs.get("service_type") == ServiceType.RECORD_PROCESSOR
        ]
        assert len(rp_calls) == 2

    @pytest.mark.asyncio
    async def test_service_ids_share_pod_id(
        self, worker_pod_manager_custom: WorkerPodManager
    ) -> None:
        """Test all spawned service IDs share the same pod_id segment."""
        manager = worker_pod_manager_custom
        manager._subprocess_manager.spawn_service = AsyncMock()

        await manager._spawn_subprocesses()

        # IDs are "{type}_{pod_id}_{index}" where type may contain underscores
        # (e.g. "record_processor_abc123_0", "worker_abc123_0")
        # The pod_id is the second-to-last segment when splitting on "_"
        calls = manager._subprocess_manager.spawn_service.call_args_list
        pod_ids = set()
        for call in calls:
            service_id = call.kwargs.get("service_id")
            parts = service_id.rsplit("_", 2)
            assert len(parts) >= 3 or parts[-1].isdigit()
            pod_ids.add(parts[-2])
        assert len(pod_ids) == 1, (
            f"Expected single pod_id across all services, got {pod_ids}"
        )

    @pytest.mark.asyncio
    async def test_record_processors_spawned_before_workers(
        self, worker_pod_manager_custom: WorkerPodManager
    ) -> None:
        """Test record processors are spawned before workers."""
        manager = worker_pod_manager_custom
        spawn_order = []

        async def track_spawn(**kwargs):
            spawn_order.append(kwargs.get("service_type"))

        manager._subprocess_manager.spawn_service = AsyncMock(side_effect=track_spawn)

        await manager._spawn_subprocesses()

        # First 2 should be record processors, then 8 workers
        rp_count = manager.record_processors_per_pod
        for i in range(rp_count):
            assert spawn_order[i] == ServiceType.RECORD_PROCESSOR
        for i in range(rp_count, len(spawn_order)):
            assert spawn_order[i] == ServiceType.WORKER


# =============================================================================
# Dataset Handling Tests
# =============================================================================


class TestDatasetHandling:
    """Tests for dataset configuration and download handling."""

    @staticmethod
    def _create_mock_path(size: int = 1024) -> MagicMock:
        """Create a mock Path object with stat support."""
        mock_path = MagicMock(spec=Path)
        mock_stat = MagicMock()
        mock_stat.st_size = size
        mock_path.stat.return_value = mock_stat
        return mock_path

    @pytest.mark.asyncio
    async def test_dataset_notification_triggers_download(
        self,
        worker_pod_manager: WorkerPodManager,
        dataset_notification: DatasetConfiguredNotification,
    ) -> None:
        """Test dataset configured notification triggers download."""
        manager = worker_pod_manager
        mock_data_path = self._create_mock_path(1024)
        mock_index_path = self._create_mock_path(256)
        manager._download_dataset = AsyncMock(
            return_value=(mock_data_path, mock_index_path)
        )
        manager.publish = AsyncMock()

        await manager._on_dataset_configured(dataset_notification)

        manager._download_dataset.assert_called_once()

    @pytest.mark.asyncio
    async def test_dataset_downloaded_flag_set(
        self,
        worker_pod_manager: WorkerPodManager,
        dataset_notification: DatasetConfiguredNotification,
    ) -> None:
        """Test _dataset_downloaded flag is set after notification."""
        manager = worker_pod_manager
        mock_data_path = self._create_mock_path(1024)
        mock_index_path = self._create_mock_path(256)
        manager._download_dataset = AsyncMock(
            return_value=(mock_data_path, mock_index_path)
        )
        manager.publish = AsyncMock()

        await manager._on_dataset_configured(dataset_notification)

        assert manager._dataset_downloaded is True

    @pytest.mark.asyncio
    async def test_duplicate_dataset_notification_ignored(
        self,
        worker_pod_manager: WorkerPodManager,
        dataset_notification: DatasetConfiguredNotification,
    ) -> None:
        """Test duplicate dataset notifications are ignored."""
        manager = worker_pod_manager
        mock_data_path = self._create_mock_path(1024)
        mock_index_path = self._create_mock_path(256)
        manager._download_dataset = AsyncMock(
            return_value=(mock_data_path, mock_index_path)
        )
        manager.publish = AsyncMock()

        # First notification
        await manager._on_dataset_configured(dataset_notification)
        # Second notification (should be ignored)
        await manager._on_dataset_configured(dataset_notification)

        assert manager._download_dataset.call_count == 1

    @pytest.mark.asyncio
    async def test_missing_dataset_api_url_raises_error(
        self, worker_pod_manager: WorkerPodManager
    ) -> None:
        """Test missing dataset_api_base_url raises RuntimeError."""
        manager = worker_pod_manager
        manager.run.cfg.runtime.dataset_api_base_url = None

        with pytest.raises(RuntimeError, match="dataset_api_base_url"):
            await manager._download_dataset()


# =============================================================================
# Health Monitoring Tests
# =============================================================================


class TestHealthMonitoring:
    """Tests for subprocess health monitoring."""

    @pytest.mark.asyncio
    async def test_worker_health_tracked(
        self, worker_pod_manager: WorkerPodManager
    ) -> None:
        """Test worker health messages are tracked correctly."""
        from aiperf.common.enums import WorkerStatus
        from aiperf.common.messages import WorkerHealthMessage

        manager = worker_pod_manager

        health_msg = WorkerHealthMessage(
            service_id="test-pod-manager_worker_0",
            health=ProcessHealth(
                create_time=1000.0,
                uptime=100.0,
                cpu_usage=50.0,
                memory_usage=1024 * 1024 * 100,
            ),
            task_stats=WorkerTaskStats(total=10, completed=5, failed=0),
        )

        await manager._on_worker_health(health_msg)

        # Verify worker is tracked
        assert "test-pod-manager_worker_0" in manager.worker_health
        stats = manager.worker_health["test-pod-manager_worker_0"]
        assert stats.worker_id == "test-pod-manager_worker_0"
        assert stats.health.cpu_usage == 50.0
        assert stats.status == WorkerStatus.HEALTHY
        assert stats.task_stats.total == 10

    @pytest.mark.asyncio
    async def test_worker_startup_state_tracked(
        self, worker_pod_manager: WorkerPodManager
    ) -> None:
        """Test worker startup-state messages are tracked correctly."""
        manager = worker_pod_manager

        await manager._on_worker_startup_state(
            WorkerStartupStateMessage(
                service_id="test-pod-manager_worker_0",
                startup_state=WorkerStartupState.WAITING_FOR_DATASET,
            )
        )

        assert (
            manager.worker_health["test-pod-manager_worker_0"].startup_state
            == WorkerStartupState.WAITING_FOR_DATASET
        )

    @pytest.mark.asyncio
    async def test_dead_subprocess_detected(
        self, worker_pod_manager: WorkerPodManager
    ) -> None:
        """Test dead subprocesses are detected and removed."""
        manager = worker_pod_manager
        manager.warning = MagicMock()

        # Create a mock SubprocessInfo-like object
        dead_info = MagicMock()
        dead_info.process = MagicMock()
        dead_info.process.exitcode = 1
        dead_info.service_type = ServiceType.WORKER
        dead_info.service_id = "test_worker_0"

        manager._subprocess_manager.check_alive = MagicMock(return_value=[dead_info])
        manager._subprocess_manager.remove = MagicMock()
        manager._subprocess_manager.get_by_type = MagicMock(
            return_value=[MagicMock()]  # At least one worker remaining
        )

        await manager._monitor_subprocesses()

        manager._subprocess_manager.remove.assert_called_once_with(dead_info)
        manager.warning.assert_called()

    @pytest.mark.asyncio
    async def test_dead_preready_worker_logs_startup_state(
        self, worker_pod_manager: WorkerPodManager
    ) -> None:
        """Dead workers should log their last preready startup state when known."""
        manager = worker_pod_manager
        manager.warning = MagicMock()

        manager.worker_health["test_worker_0"] = WorkerStats(
            worker_id="test_worker_0",
            startup_state=WorkerStartupState.WAITING_FOR_DATASET,
        )

        dead_info = MagicMock()
        dead_info.process = MagicMock()
        dead_info.process.exitcode = 1
        dead_info.service_type = ServiceType.WORKER
        dead_info.service_id = "test_worker_0"

        manager._subprocess_manager.check_alive = MagicMock(return_value=[dead_info])
        manager._subprocess_manager.remove = MagicMock()
        manager._subprocess_manager.get_by_type = MagicMock(return_value=[MagicMock()])

        await manager._monitor_subprocesses()

        warning_message = manager.warning.call_args[0][0]
        assert "before becoming ready" in warning_message
        assert "waiting_for_dataset" in warning_message

    @pytest.mark.asyncio
    async def test_all_workers_dead_triggers_stop(
        self, worker_pod_manager: WorkerPodManager
    ) -> None:
        """Test pod stops when all workers are dead."""
        manager = worker_pod_manager
        manager.error = MagicMock()
        manager.stop = AsyncMock()

        # Create a mock SubprocessInfo-like object
        dead_info = MagicMock()
        dead_info.process = MagicMock()
        dead_info.process.exitcode = 1
        dead_info.service_type = ServiceType.WORKER
        dead_info.service_id = "test_worker_0"

        manager._subprocess_manager.check_alive = MagicMock(return_value=[dead_info])
        manager._subprocess_manager.remove = MagicMock()
        manager._subprocess_manager.get_by_type = MagicMock(
            return_value=[]
        )  # No workers

        await manager._monitor_subprocesses()

        manager.error.assert_called()
        manager.stop.assert_called_once()


# =============================================================================
# Shutdown Tests
# =============================================================================


class TestShutdown:
    """Tests for WorkerPodManager shutdown behavior."""

    @pytest.mark.asyncio
    async def test_shutdown_command_triggers_stop(
        self, worker_pod_manager: WorkerPodManager, shutdown_command: Command
    ) -> None:
        """Test shutdown command triggers stop."""
        manager = worker_pod_manager
        manager.stop = AsyncMock()

        await manager._on_shutdown_command(shutdown_command)

        manager.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_cleans_up_subprocesses(
        self, worker_pod_manager: WorkerPodManager
    ) -> None:
        """Test stop cleans up all subprocesses."""
        manager = worker_pod_manager
        manager._subprocess_manager.stop_all = AsyncMock()
        manager._subprocess_manager.clear = MagicMock()

        await manager._stop_worker_pod_manager()

        manager._subprocess_manager.stop_all.assert_called_once()
        manager._subprocess_manager.clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_stops_proxy_manager(
        self, worker_pod_manager: WorkerPodManager
    ) -> None:
        """Test stop calls proxy_manager.stop() after subprocesses are cleaned up."""
        manager = worker_pod_manager
        manager._subprocess_manager.stop_all = AsyncMock()
        manager._subprocess_manager.clear = MagicMock()
        manager._proxy_manager.stop = AsyncMock()

        await manager._stop_worker_pod_manager()

        manager._proxy_manager.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_order_subprocesses_before_proxy(
        self, worker_pod_manager: WorkerPodManager
    ) -> None:
        """Test subprocesses are stopped before the proxy manager."""
        manager = worker_pod_manager
        call_order = []
        manager._subprocess_manager.stop_all = AsyncMock(
            side_effect=lambda: call_order.append("subprocesses")
        )
        manager._subprocess_manager.clear = MagicMock()
        manager._proxy_manager.stop = AsyncMock(
            side_effect=lambda: call_order.append("proxy")
        )

        await manager._stop_worker_pod_manager()

        assert call_order == ["subprocesses", "proxy"]


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestWorkerPodManagerIntegration:
    """Integration-style tests for WorkerPodManager lifecycle."""

    @pytest.mark.asyncio
    async def test_initialize_proxy_starts_proxy_manager(
        self, worker_pod_manager: WorkerPodManager
    ) -> None:
        """Test on_init hook initializes and starts the proxy manager."""
        manager = worker_pod_manager
        manager._proxy_manager.initialize_and_start = AsyncMock()

        await manager._initialize_proxy()

        manager._proxy_manager.initialize_and_start.assert_called_once()

    @pytest.mark.asyncio
    async def test_full_lifecycle(
        self,
        dataset_notification: DatasetConfiguredNotification,
    ) -> None:
        """Test full WorkerPodManager lifecycle from init to shutdown."""
        test_run = _make_run(workers_per_pod=2, record_processors_per_pod=1)

        with (
            patch.object(WorkerPodManager, "debug"),
            patch.object(WorkerPodManager, "info"),
            patch.object(WorkerPodManager, "warning"),
        ):
            manager = WorkerPodManager(
                run=test_run,
                service_id="lifecycle-test",
            )

        # Verify initialization
        assert manager.workers_per_pod == 2
        assert manager.record_processors_per_pod == 1
        assert manager._dataset_downloaded is False

        # Mock subprocess spawning, tokenizer prefetch, and download
        manager._subprocess_manager.spawn_service = AsyncMock()
        manager._subprocess_manager.stop_all = AsyncMock()
        manager._subprocess_manager.clear = MagicMock()
        manager._proxy_manager.stop = AsyncMock()
        manager._prefetch_tokenizers = AsyncMock()

        # Create mock paths that support stat()
        mock_data_path = MagicMock(spec=Path)
        mock_data_path.stat.return_value = MagicMock(st_size=1024)
        mock_index_path = MagicMock(spec=Path)
        mock_index_path.stat.return_value = MagicMock(st_size=256)
        manager._download_dataset = AsyncMock(
            return_value=(mock_data_path, mock_index_path)
        )
        manager.publish = AsyncMock()

        # Simulate startup (prefetches tokenizers, then spawns subprocesses)
        await manager._start_worker_pod_manager()
        manager._prefetch_tokenizers.assert_called_once()
        assert (
            manager._subprocess_manager.spawn_service.call_count == 3
        )  # 2 workers + 1 RP

        # Simulate dataset configured (triggers download and notification)
        await manager._on_dataset_configured(dataset_notification)
        assert manager._dataset_downloaded is True
        manager._download_dataset.assert_called_once()
        manager.publish.assert_called_once()  # DatasetDownloadedNotification

        # Simulate shutdown
        await manager._stop_worker_pod_manager()

        manager._subprocess_manager.clear.assert_called_once()
        manager._proxy_manager.stop.assert_called_once()

    @pytest.mark.parametrize(
        ("workers", "rps", "expected_total"),
        [
            param(1, 1, 2, id="minimal"),
            param(4, 1, 5, id="four_workers"),
            param(8, 2, 10, id="eight_workers"),
            param(16, 4, 20, id="sixteen_workers"),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_spawn_total_subprocesses(
        self, workers: int, rps: int, expected_total: int
    ) -> None:
        """Test total subprocess count matches workers + record processors."""
        test_run = _make_run(workers_per_pod=workers, record_processors_per_pod=rps)

        with (
            patch.object(WorkerPodManager, "debug"),
            patch.object(WorkerPodManager, "info"),
        ):
            manager = WorkerPodManager(
                run=test_run,
                service_id="spawn-test",
            )

        manager._subprocess_manager.spawn_service = AsyncMock()

        await manager._spawn_subprocesses()

        assert manager._subprocess_manager.spawn_service.call_count == expected_total
