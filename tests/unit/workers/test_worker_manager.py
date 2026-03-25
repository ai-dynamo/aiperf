# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for WorkerManager health monitoring and worker scaling calculations.
"""

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.enums import WorkerStartupState, WorkerStatus
from aiperf.common.environment import Environment
from aiperf.common.messages import WorkerHealthMessage
from aiperf.common.messages.worker_messages import (
    WorkerStartupStateMessage,
    WorkerStatusSummaryMessage,
)
from aiperf.common.models import ProcessHealth, WorkerTaskStats
from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.workers.scaling import calculate_worker_count
from aiperf.workers.worker_manager import WorkerManager, WorkerStatusInfo

DEFAULT_MEMORY = 1024 * 1024 * 100
WORKER_ID = "test-worker-1"


def _make_config(
    concurrency: int | None = None,
    max_workers: int | None = None,
    request_rate: float | None = None,
    request_count: int | None = None,
) -> AIPerfConfig:
    """Helper to build an AIPerfConfig for WorkerManager tests."""
    if concurrency is not None:
        # Concurrency mode
        req = request_count or max(concurrency, 10)
        phases = {
            "default": {
                "type": "concurrency",
                "requests": req,
                "concurrency": concurrency,
            }
        }
    elif request_rate is not None:
        # Request rate mode (no concurrency cap on workers)
        phases = {
            "default": {
                "type": "poisson",
                "requests": request_count or 100,
                "rate": request_rate,
            }
        }
    else:
        # Default: concurrency of 1
        phases = {
            "default": {
                "type": "concurrency",
                "requests": request_count or 10,
                "concurrency": 1,
            }
        }

    runtime = {}
    if max_workers is not None:
        runtime["workers"] = max_workers

    return AIPerfConfig(
        models=["test-model"],
        endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
        datasets={
            "default": {
                "type": "synthetic",
                "entries": 100,
                "prompts": {"isl": 128, "osl": 64},
            }
        },
        phases=phases,
        **({"runtime": runtime} if runtime else {}),
    )


def _make_run(**config_overrides) -> BenchmarkRun:
    """Helper to build a BenchmarkRun for WorkerManager tests."""
    config = _make_config(**config_overrides)
    return BenchmarkRun(
        benchmark_id="test",
        cfg=config,
        artifact_dir=Path("/tmp/test"),
    )


@pytest.fixture
def worker_manager() -> WorkerManager:
    """Create a WorkerManager instance for testing."""
    run = _make_run(concurrency=10, max_workers=4)
    with patch("aiperf.workers.scaling.multiprocessing.cpu_count", return_value=8):
        manager = WorkerManager(
            run=run,
            service_id="test-worker-manager",
        )
        manager.warning = MagicMock()
        return manager


@pytest.fixture
def worker_info() -> WorkerStatusInfo:
    """Create a WorkerStatusInfo instance for testing."""
    return WorkerStatusInfo(
        worker_id=WORKER_ID,
        last_update_ns=time.time_ns(),
        status=WorkerStatus.HEALTHY,
        health=ProcessHealth(
            create_time=time.time(),
            uptime=100.0,
            cpu_usage=50.0,
            memory_usage=DEFAULT_MEMORY,
        ),
        task_stats=WorkerTaskStats(total=10, completed=5, failed=0),
    )


def create_health_message(
    cpu_usage: float = 50.0,
    memory_usage: int | None = None,
    uptime: float = 100.0,
    total: int = 10,
    completed: int = 5,
    failed: int = 0,
) -> WorkerHealthMessage:
    """Helper to create WorkerHealthMessage with default values."""
    return WorkerHealthMessage(
        service_id=WORKER_ID,
        health=ProcessHealth(
            create_time=time.time(),
            uptime=uptime,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage or DEFAULT_MEMORY,
        ),
        task_stats=WorkerTaskStats(total=total, completed=completed, failed=failed),
    )


class TestCalculateWorkerCount:
    """Test the calculate_worker_count function in aiperf.workers.scaling."""

    @pytest.mark.parametrize(
        "cpus,concurrency,max_workers,expected",
        [
            (10, 100, None, 6),  # CPU-based limit: 10 * 0.75 - 1 = 6
            (10, 100, 4, 4),  # max_workers setting limits to 4
            (10, None, None, 1),  # Concurrency defaults to 1, which limits workers to 1
            (10, 3, None, 3),  # Low concurrency (3) limits workers below CPU calculation
            (10, 8, 3, 3),  # max_workers (3) overrides higher concurrency (8)
            (10, 10, 5, 5),  # max_workers (5) overrides higher concurrency (10)
            (224, 1000, None, 32),  # High CPU count with hard cap at 32 workers
            (32, 1000, None, 23),  # CPU-based limit: 32 * 0.75 - 1 = 23
            (1, 100, None, 1),  # Single CPU system, should default to 1 worker minimum
            (2, 100, None, 1),  # Two CPUs: 2 * 0.75 - 1 = 0.5, rounded up to 1
            (4, 100, None, 2),  # Four CPUs: 4 * 0.75 - 1 = 2
            (44, 1000, None, 32),  # CPU count that would exceed 32 limit: 44 * 0.75 - 1 = 32
            (45, 1000, None, 32),  # CPU count that hits the hard cap: 45 * 0.75 - 1 = 32.75
            (4, 100, 100, 100),  # Very high max_workers, not limited by CPU calculation
            (64, 1, None, 1),  # Concurrency of 1 limits to 1 worker regardless of CPUs
        ],
    )  # fmt: skip
    def test_max_workers_combinations(self, cpus, concurrency, max_workers, expected):
        """Test max workers calculation with different CPU counts, concurrency, and max_workers settings."""
        with patch(
            "aiperf.workers.scaling.multiprocessing.cpu_count", return_value=cpus
        ):
            request_count = max(concurrency or 10, 10)
            run = _make_run(
                concurrency=concurrency,
                max_workers=max_workers,
                request_count=request_count,
            )

            assert calculate_worker_count(run.cfg) == expected

    @pytest.mark.parametrize(
        "cpus,request_rate,max_workers,expected",
        [
            (10, 50, None, 6),  # CPU-based limit: 10 * 0.75 - 1 = 6 (no concurrency limit)
            (10, 100, 4, 4),  # max_workers setting limits to 4
            (4, 10, None, 2),  # Low CPU count: 4 * 0.75 - 1 = 2
            (2, 50, None, 1),  # Very low CPU: 2 * 0.75 - 1 = 0.5, rounded up to 1
            (1, 100, None, 1),  # Single CPU system minimum
            (64, 500, None, 32),  # High CPU count with hard cap at 32 workers
            (10, 1, None, 6),  # Very low request rate still uses CPU calculation
            (10, 1000, 8, 8),  # High request rate with max_workers override
            (8, 50, 20, 20),  # max_workers higher than CPU calc
        ],
    )  # fmt: skip
    def test_max_workers_with_request_rate_combinations(
        self, cpus, request_rate, max_workers, expected
    ):
        """Test max workers calculation with request_rate mode where concurrency is 0/None."""
        with patch(
            "aiperf.workers.scaling.multiprocessing.cpu_count", return_value=cpus
        ):
            run = _make_run(
                request_rate=request_rate,
                max_workers=max_workers,
            )

            assert calculate_worker_count(run.cfg) == expected


class TestHighCPUWarning:
    """Test the high CPU warning functionality in WorkerManager."""

    def test_high_cpu_triggers_warning(
        self,
        worker_manager: WorkerManager,
        worker_info: WorkerStatusInfo,
    ):
        """Test that exceeding CPU threshold triggers warning and sets HIGH_LOAD status."""
        message = create_health_message(cpu_usage=90.0)

        worker_manager._update_worker_status(worker_info, message)

        assert worker_info.status == WorkerStatus.HIGH_LOAD
        assert worker_info.last_high_load_ns is not None
        worker_manager.warning.assert_called_once()
        warning_msg = worker_manager.warning.call_args[0][0]
        assert "CPU usage" in warning_msg
        assert "90%" in warning_msg
        assert WORKER_ID in warning_msg

    def test_cpu_at_threshold_boundary(
        self,
        worker_manager: WorkerManager,
        worker_info: WorkerStatusInfo,
    ):
        """Test that CPU exactly at 85% threshold does not trigger warning (boundary condition)."""
        message = create_health_message(
            cpu_usage=Environment.WORKER.HIGH_LOAD_CPU_USAGE
        )

        worker_manager._update_worker_status(worker_info, message)

        assert worker_info.status != WorkerStatus.HIGH_LOAD
        worker_manager.warning.assert_not_called()

    def test_first_high_cpu_with_none_last_high_load(
        self,
        worker_manager: WorkerManager,
        worker_info: WorkerStatusInfo,
    ):
        """Test high CPU detection when last_high_load_ns is None (first occurrence)."""
        worker_info.last_high_load_ns = None
        worker_info.status = WorkerStatus.HEALTHY

        message = create_health_message(cpu_usage=95.0)

        worker_manager._update_worker_status(worker_info, message)

        assert worker_info.status == WorkerStatus.HIGH_LOAD
        assert worker_info.last_high_load_ns is not None
        worker_manager.warning.assert_called_once()

    def test_multiple_consecutive_high_cpu_messages(
        self,
        worker_manager: WorkerManager,
        worker_info: WorkerStatusInfo,
    ):
        """Test multiple consecutive high CPU messages update timestamp and trigger warnings."""
        worker_manager._update_worker_status(
            worker_info, create_health_message(cpu_usage=90.0)
        )

        first_timestamp = worker_info.last_high_load_ns
        assert worker_info.status == WorkerStatus.HIGH_LOAD
        assert worker_manager.warning.call_count == 1

        worker_manager._update_worker_status(
            worker_info,
            create_health_message(cpu_usage=92.0, uptime=102.0, total=12, completed=6),
        )

        assert worker_info.status == WorkerStatus.HIGH_LOAD
        assert worker_info.last_high_load_ns != first_timestamp
        assert worker_manager.warning.call_count == 2

    def test_error_status_takes_precedence_over_high_cpu(
        self,
        worker_manager: WorkerManager,
        worker_info: WorkerStatusInfo,
    ):
        """Test that ERROR status takes precedence over HIGH_LOAD status."""
        worker_info.task_stats = WorkerTaskStats(total=10, completed=5, failed=2)
        worker_info.last_error_ns = time.time_ns()

        message = create_health_message(cpu_usage=95.0, failed=3)

        worker_manager._update_worker_status(worker_info, message)

        assert worker_info.status == WorkerStatus.ERROR
        worker_manager.warning.assert_not_called()

    def test_high_cpu_with_idle_worker(
        self,
        worker_manager: WorkerManager,
        worker_info: WorkerStatusInfo,
    ):
        """Test that HIGH_LOAD status takes precedence over IDLE when CPU is high."""
        message = create_health_message(cpu_usage=90.0, total=0, completed=0)

        worker_manager._update_worker_status(worker_info, message)

        assert worker_info.status == WorkerStatus.HIGH_LOAD
        worker_manager.warning.assert_called_once()

    def test_high_cpu_updates_health_info(
        self,
        worker_manager: WorkerManager,
        worker_info: WorkerStatusInfo,
    ):
        """Test that worker health info is updated even when high CPU is detected."""
        message = create_health_message(
            cpu_usage=92.5,
            memory_usage=DEFAULT_MEMORY * 2,
            uptime=150.0,
            total=20,
            completed=15,
        )

        worker_manager._update_worker_status(worker_info, message)

        assert worker_info.status == WorkerStatus.HIGH_LOAD
        assert worker_info.health.cpu_usage == 92.5
        assert worker_info.health.memory_usage == DEFAULT_MEMORY * 2
        assert worker_info.task_stats.total == 20
        assert worker_info.task_stats.completed == 15


class TestWorkerStartupStates:
    """Test worker startup-state tracking in WorkerManager."""

    @pytest.mark.asyncio
    async def test_startup_state_creates_worker_and_publishes_summary(
        self, worker_manager: WorkerManager
    ) -> None:
        """Startup-state messages should create worker info and publish a summary."""
        worker_manager.publish = AsyncMock()
        message = WorkerStartupStateMessage(
            service_id=WORKER_ID,
            startup_state=WorkerStartupState.WAITING_FOR_DATASET,
        )

        await worker_manager._on_worker_startup_state(message)

        info = worker_manager.worker_infos[WORKER_ID]
        assert info.startup_state == WorkerStartupState.WAITING_FOR_DATASET

        summary = worker_manager.publish.await_args.args[0]
        assert isinstance(summary, WorkerStatusSummaryMessage)
        assert summary.worker_statuses == {WORKER_ID: WorkerStatus.IDLE}
        assert summary.worker_startup_states == {
            WORKER_ID: WorkerStartupState.WAITING_FOR_DATASET
        }

    @pytest.mark.asyncio
    async def test_recent_startup_state_does_not_mark_worker_stale(
        self, worker_manager: WorkerManager
    ) -> None:
        """Startup-state activity should count as liveness before the first health tick."""
        worker_manager.worker_infos[WORKER_ID] = WorkerStatusInfo(
            worker_id=WORKER_ID,
            status=WorkerStatus.IDLE,
            startup_state=WorkerStartupState.WAITING_FOR_DATASET,
            startup_state_updated_ns=time.time_ns(),
        )

        await worker_manager._worker_status_loop()

        assert worker_manager.worker_infos[WORKER_ID].status == WorkerStatus.IDLE
