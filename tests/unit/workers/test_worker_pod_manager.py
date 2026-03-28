# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for WorkerPodManager service.

WorkerPodManager runs in worker pods and coordinates shared pod infrastructure
while worker and record-processor services run as sibling containers.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param

from aiperf.common.control_structs import Command
from aiperf.common.enums import CommandType, WorkerStartupState, WorkerStatus
from aiperf.common.environment import Environment
from aiperf.common.messages import (
    DatasetConfiguredNotification,
    WorkerHealthMessage,
)
from aiperf.common.messages.worker_messages import (
    WorkerStartupStateMessage,
    WorkerStatusSummaryMessage,
)
from aiperf.common.models import (
    DatasetMetadata,
    MemoryMapClientMetadata,
    ProcessHealth,
    WorkerTaskStats,
)
from aiperf.common.pod_lifecycle_structs import (
    PodPeerHello,
    PodPeerShutdown,
    PodWorkerHealth,
    PodWorkerStartupState,
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
        manager._pod_index = "0"
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
        manager._pod_index = "0"
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
# Startup Tests
# =============================================================================


class TestStartup:
    """Tests for WorkerPodManager startup behavior."""

    @pytest.mark.asyncio
    async def test_start_prefetches_tokenizers_in_background(
        self, worker_pod_manager_custom: WorkerPodManager
    ) -> None:
        """Startup should kick off tokenizer prefetch without blocking registration."""
        manager = worker_pod_manager_custom
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_prefetch() -> None:
            started.set()
            await release.wait()

        manager._prefetch_tokenizers = AsyncMock(side_effect=slow_prefetch)

        await manager._start_worker_pod_manager()
        await asyncio.sleep(0)

        assert started.is_set()
        assert manager._tokenizer_prefetch_task is not None
        assert not manager._tokenizer_prefetch_task.done()

        release.set()
        await manager._tokenizer_prefetch_task
        manager._prefetch_tokenizers.assert_called_once()


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
    async def test_success_notification_includes_pod_index(
        self,
        worker_pod_manager: WorkerPodManager,
        dataset_notification: DatasetConfiguredNotification,
    ) -> None:
        """Successful dataset download notifications should be scoped to the current pod."""
        manager = worker_pod_manager
        mock_data_path = self._create_mock_path(1024)
        mock_index_path = self._create_mock_path(256)
        manager._download_dataset = AsyncMock(
            return_value=(mock_data_path, mock_index_path)
        )
        manager.publish = AsyncMock()
        manager._notify_registered_workers_of_dataset = AsyncMock()

        await manager._on_dataset_configured(dataset_notification)

        manager._notify_registered_workers_of_dataset.assert_awaited_once()
        kwargs = manager._notify_registered_workers_of_dataset.await_args.kwargs
        assert kwargs["success"] is True

    @pytest.mark.asyncio
    async def test_failure_notification_includes_pod_index(
        self,
        worker_pod_manager: WorkerPodManager,
        dataset_notification: DatasetConfiguredNotification,
    ) -> None:
        """Failed dataset download notifications should be scoped to the current pod."""
        manager = worker_pod_manager
        manager._download_dataset = AsyncMock(side_effect=RuntimeError("boom"))
        manager.publish = AsyncMock()
        manager._notify_registered_workers_of_dataset = AsyncMock()

        with pytest.raises(RuntimeError, match="boom"):
            await manager._on_dataset_configured(dataset_notification)

        manager._notify_registered_workers_of_dataset.assert_awaited_once()
        kwargs = manager._notify_registered_workers_of_dataset.await_args.kwargs
        assert kwargs["success"] is False

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
        manager._notify_registered_workers_of_dataset = AsyncMock()

        # First notification
        await manager._on_dataset_configured(dataset_notification)
        # Second notification (should trigger a re-send, not a re-download)
        await manager._on_dataset_configured(dataset_notification)

        assert manager._download_dataset.call_count == 1
        assert manager._notify_registered_workers_of_dataset.await_count == 2

    @pytest.mark.asyncio
    async def test_concurrent_dataset_notifications_do_not_overlap_downloads(
        self,
        worker_pod_manager: WorkerPodManager,
        dataset_notification: DatasetConfiguredNotification,
    ) -> None:
        """Concurrent rebroadcasts should share one in-flight dataset download."""
        manager = worker_pod_manager
        started = asyncio.Event()
        release = asyncio.Event()
        mock_data_path = self._create_mock_path(1024)
        mock_index_path = self._create_mock_path(256)

        async def slow_download() -> tuple[MagicMock, MagicMock]:
            started.set()
            await release.wait()
            return mock_data_path, mock_index_path

        manager._download_dataset = AsyncMock(side_effect=slow_download)
        manager.publish = AsyncMock()
        manager._notify_registered_workers_of_dataset = AsyncMock()

        task1 = asyncio.create_task(
            manager._on_dataset_configured(dataset_notification)
        )
        await started.wait()
        task2 = asyncio.create_task(
            manager._on_dataset_configured(dataset_notification)
        )
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(task1, task2)

        assert manager._download_dataset.await_count == 1
        assert manager._notify_registered_workers_of_dataset.await_count == 1

    @pytest.mark.asyncio
    async def test_download_dataset_starts_data_and_index_downloads_concurrently(
        self, worker_pod_manager: WorkerPodManager, tmp_path: Path, monkeypatch
    ) -> None:
        """Dataset data and index downloads should start concurrently."""
        manager = worker_pod_manager
        manager.run.cfg.runtime.dataset_api_base_url = "http://controller/api/dataset"
        monkeypatch.setattr(Environment.DATASET, "MMAP_BASE_PATH", tmp_path)

        started: list[str] = []
        release = asyncio.Event()

        async def download_file(_session, _url: str, dest_path: Path) -> None:
            started.append(dest_path.name)
            if len(started) == 2:
                release.set()
            await release.wait()
            dest_path.write_bytes(
                b"x" * (1024 if dest_path.name == "dataset.dat" else 256)
            )

        manager._download_file = AsyncMock(side_effect=download_file)

        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = False
            mock_session_cls.return_value = mock_session

            data_path, index_path = await manager._download_dataset()

        assert started == ["dataset.dat", "index.dat"]
        assert data_path.name == "dataset.dat"
        assert index_path.name == "index.dat"
        assert data_path.stat().st_size == 1024
        assert index_path.stat().st_size == 256
        assert manager._download_file.await_count == 2

    @pytest.mark.asyncio
    async def test_download_dataset_retries_when_one_parallel_download_fails(
        self, worker_pod_manager: WorkerPodManager, tmp_path: Path, monkeypatch
    ) -> None:
        """A failure in either parallel download should retry the whole dataset fetch."""
        manager = worker_pod_manager
        manager.run.cfg.runtime.dataset_api_base_url = "http://controller/api/dataset"
        monkeypatch.setattr(Environment.DATASET, "MMAP_BASE_PATH", tmp_path)
        index_failures = {"count": 0}

        async def download_file(_session, url: str, dest_path: Path) -> None:
            if url.endswith("/index") and index_failures["count"] == 0:
                index_failures["count"] += 1
                raise RuntimeError("index failed")
            dest_path.write_bytes(b"x" * (1024 if url.endswith("/data") else 256))

        manager._download_file = AsyncMock(side_effect=download_file)

        with (
            patch("aiohttp.ClientSession") as mock_session_cls,
            patch("asyncio.sleep", new=AsyncMock()) as mock_sleep,
        ):
            mock_session = AsyncMock()
            mock_session.__aenter__.return_value = mock_session
            mock_session.__aexit__.return_value = False
            mock_session_cls.side_effect = [mock_session, mock_session]

            data_path, index_path = await manager._download_dataset()

        assert mock_sleep.await_count == 1
        assert manager._download_file.await_count == 4
        assert data_path.name == "dataset.dat"
        assert index_path.name == "index.dat"
        assert data_path.stat().st_size == 1024
        assert index_path.stat().st_size == 256

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
    """Tests for worker health tracking across sibling containers."""

    @pytest.mark.asyncio
    async def test_worker_registration_over_pod_lifecycle_channel(
        self, worker_pod_manager: WorkerPodManager
    ) -> None:
        """WorkerPodManager should track sibling registrations on the direct ROUTER channel."""
        ack = await worker_pod_manager._on_pod_lifecycle_message(
            "worker-identity",
            PodPeerHello(
                service_id="worker_0",
                service_type=str(ServiceType.WORKER),
                pod_index="0",
            ),
        )

        assert ack is not None
        assert worker_pod_manager._pod_peer_identities["worker_0"] == "worker-identity"
        assert worker_pod_manager._pod_peer_types["worker_0"] == str(ServiceType.WORKER)

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
        manager.publish = AsyncMock()

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
        summary = manager.publish.await_args.args[0]
        assert isinstance(summary, WorkerStatusSummaryMessage)
        assert summary.worker_startup_states == {
            "test-pod-manager_worker_0": WorkerStartupState.WAITING_FOR_DATASET
        }

    @pytest.mark.asyncio
    async def test_direct_worker_health_struct_updates_status(
        self, worker_pod_manager: WorkerPodManager
    ) -> None:
        """Pod-local worker health structs should feed the existing aggregation logic."""
        await worker_pod_manager._on_pod_lifecycle_message(
            "worker-identity",
            PodWorkerHealth(
                service_id="worker_0",
                create_time=1000.0,
                uptime=100.0,
                cpu_usage=10.0,
                memory_usage=1024 * 1024 * 100,
                task_total=10,
                task_failed=0,
                task_completed=5,
            ),
        )

        assert (
            worker_pod_manager.worker_health["worker_0"].status == WorkerStatus.HEALTHY
        )

    @pytest.mark.asyncio
    async def test_direct_worker_startup_state_struct_updates_summary(
        self, worker_pod_manager: WorkerPodManager
    ) -> None:
        """Pod-local startup-state structs should update the tracked worker summary."""
        worker_pod_manager.publish = AsyncMock()

        await worker_pod_manager._on_pod_lifecycle_message(
            "worker-identity",
            PodWorkerStartupState(
                service_id="worker_0",
                startup_state=str(WorkerStartupState.WAITING_FOR_DATASET),
                request_ns=123,
            ),
        )

        summary = worker_pod_manager.publish.await_args.args[0]
        assert isinstance(summary, WorkerStatusSummaryMessage)
        assert summary.worker_startup_states == {
            "worker_0": WorkerStartupState.WAITING_FOR_DATASET
        }

    @pytest.mark.asyncio
    async def test_report_worker_status_summary_command_publishes_summary(
        self, worker_pod_manager: WorkerPodManager
    ) -> None:
        """Controller refresh requests should trigger an immediate worker summary publish."""
        manager = worker_pod_manager
        manager.publish = AsyncMock()
        await manager._on_worker_health(
            WorkerHealthMessage(
                service_id="worker_0",
                health=ProcessHealth(
                    create_time=1000.0,
                    uptime=100.0,
                    cpu_usage=10.0,
                    memory_usage=1024 * 1024 * 100,
                ),
                task_stats=WorkerTaskStats(total=10, completed=0, failed=0),
            )
        )
        await manager._on_worker_startup_state(
            WorkerStartupStateMessage(
                service_id="worker_0",
                startup_state=WorkerStartupState.WAITING_FOR_DATASET,
            )
        )
        manager.publish.reset_mock()

        await manager._on_report_worker_status_summary(
            Command(cmd=CommandType.REPORT_WORKER_STATUS_SUMMARY, cid="cid")
        )

        summary = manager.publish.await_args.args[0]
        assert isinstance(summary, WorkerStatusSummaryMessage)
        assert summary.worker_statuses == {"worker_0": WorkerStatus.HEALTHY}
        assert summary.worker_startup_states == {
            "worker_0": WorkerStartupState.WAITING_FOR_DATASET
        }


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
    async def test_record_processor_shutdowns_are_tracked_over_local_channel(
        self, worker_pod_manager: WorkerPodManager
    ) -> None:
        """WorkerPodManager should track local record-processor shutdown notifications."""
        await worker_pod_manager._on_pod_lifecycle_message(
            "rp-identity",
            PodPeerShutdown(
                service_id="record_processor_0",
                service_type=str(ServiceType.RECORD_PROCESSOR),
            ),
        )

        assert "record_processor_0" in worker_pod_manager._record_processors_shutdown

    @pytest.mark.asyncio
    async def test_stop_waits_for_raw_record_files(
        self, worker_pod_manager: WorkerPodManager
    ) -> None:
        """Test stop waits for sibling record processors to flush files."""
        manager = worker_pod_manager
        manager._wait_for_record_processor_shutdowns = AsyncMock()
        manager._wait_for_raw_record_files = AsyncMock()
        manager._proxy_manager.stop = AsyncMock()
        manager._upload_raw_records = AsyncMock()

        await manager._stop_worker_pod_manager()

        manager._wait_for_record_processor_shutdowns.assert_called_once()
        manager._wait_for_raw_record_files.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_stops_proxy_manager(
        self, worker_pod_manager: WorkerPodManager
    ) -> None:
        """Test stop calls proxy_manager.stop()."""
        manager = worker_pod_manager
        manager._wait_for_record_processor_shutdowns = AsyncMock()
        manager._wait_for_raw_record_files = AsyncMock()
        manager._proxy_manager.stop = AsyncMock()
        manager._upload_raw_records = AsyncMock()

        await manager._stop_worker_pod_manager()

        manager._proxy_manager.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_order_wait_before_proxy_before_upload(
        self, worker_pod_manager: WorkerPodManager
    ) -> None:
        """Test shutdown waits, then stops proxy, then uploads."""
        manager = worker_pod_manager
        call_order = []
        manager._wait_for_record_processor_shutdowns = AsyncMock(
            side_effect=lambda: call_order.append("record_processors")
        )
        manager._wait_for_raw_record_files = AsyncMock(
            side_effect=lambda: call_order.append("wait")
        )
        manager._proxy_manager.stop = AsyncMock(
            side_effect=lambda: call_order.append("proxy")
        )
        manager._upload_raw_records = AsyncMock(
            side_effect=lambda: call_order.append("upload")
        )

        await manager._stop_worker_pod_manager()

        assert call_order == ["record_processors", "wait", "proxy", "upload"]

    @pytest.mark.asyncio
    async def test_wait_for_raw_record_files_returns_when_files_stabilize(
        self, worker_pod_manager_custom: WorkerPodManager, tmp_path: Path
    ) -> None:
        """Raw-record wait completes once the expected files exist and stop growing."""
        manager = worker_pod_manager_custom
        manager.run.cfg.output.raw = True
        manager.run.cfg.output.dir = tmp_path

        raw_dir = tmp_path / "raw_records"
        raw_dir.mkdir()
        for idx in range(manager.record_processors_per_pod):
            (raw_dir / f"raw_records_{idx}.jsonl").write_text('{"ok": true}\n')

        with patch("asyncio.sleep", new=AsyncMock()):
            await manager._wait_for_raw_record_files()


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

        # Mock tokenizer prefetch, raw record coordination, proxy stop, and download
        manager._proxy_manager.stop = AsyncMock()
        manager._prefetch_tokenizers = AsyncMock()
        manager._wait_for_record_processor_shutdowns = AsyncMock()
        manager._wait_for_raw_record_files = AsyncMock()
        manager._upload_raw_records = AsyncMock()

        # Create mock paths that support stat()
        mock_data_path = MagicMock(spec=Path)
        mock_data_path.stat.return_value = MagicMock(st_size=1024)
        mock_index_path = MagicMock(spec=Path)
        mock_index_path.stat.return_value = MagicMock(st_size=256)
        manager._download_dataset = AsyncMock(
            return_value=(mock_data_path, mock_index_path)
        )
        manager.publish = AsyncMock()
        manager._notify_registered_workers_of_dataset = AsyncMock()

        # Simulate startup (prefetches shared tokenizer cache in the background)
        await manager._start_worker_pod_manager()
        assert manager._tokenizer_prefetch_task is not None
        await manager._tokenizer_prefetch_task
        manager._prefetch_tokenizers.assert_called_once()

        # Simulate dataset configured (triggers download and direct worker notification)
        await manager._on_dataset_configured(dataset_notification)
        assert manager._dataset_downloaded is True
        manager._download_dataset.assert_called_once()
        manager._notify_registered_workers_of_dataset.assert_called_once()

        # Simulate shutdown
        await manager._stop_worker_pod_manager()

        manager._wait_for_raw_record_files.assert_called_once()
        manager._proxy_manager.stop.assert_called_once()
