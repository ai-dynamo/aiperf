# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Simple test for WorkerManager max workers functionality.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.config.loadgen_config import LoadGeneratorConfig
from aiperf.common.config.worker_config import WorkersConfig
from aiperf.common.enums import WorkerStatus
from aiperf.common.environment import Environment
from aiperf.common.messages import WorkerHealthMessage
from aiperf.common.models import ProcessHealth, WorkerTaskStats
from aiperf.workers.worker_manager import WorkerManager, WorkerStatusInfo

DEFAULT_MEMORY = 1024 * 1024 * 100
WORKER_ID = "test-worker-1"


@pytest.fixture
def worker_manager() -> WorkerManager:
    """Create a WorkerManager instance for testing."""
    service_config = ServiceConfig(workers=WorkersConfig(max=4))
    user_config = UserConfig(
        endpoint=EndpointConfig(model_names=["test-model"]),
        loadgen=LoadGeneratorConfig(concurrency=10),
    )
    with patch(
        "aiperf.workers.worker_manager.multiprocessing.cpu_count", return_value=8
    ):
        manager = WorkerManager(
            service_config=service_config,
            user_config=user_config,
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


class TestMaxWorkers:
    """Test the max workers calculation logic in WorkerManager."""

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
            "aiperf.workers.worker_manager.multiprocessing.cpu_count", return_value=cpus
        ):
            service_config = ServiceConfig(workers=WorkersConfig(max=max_workers))
            # Set request_count to be >= concurrency to avoid validation error
            request_count = max(concurrency or 10, 10)
            user_config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                loadgen=LoadGeneratorConfig(
                    concurrency=concurrency, request_count=request_count
                ),
            )

            worker_manager = WorkerManager(
                service_config=service_config,
                user_config=user_config,
                service_id="test-worker-manager",
            )

            assert worker_manager.max_workers == expected

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
            "aiperf.workers.worker_manager.multiprocessing.cpu_count", return_value=cpus
        ):
            service_config = ServiceConfig(workers=WorkersConfig(max=max_workers))
            user_config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                loadgen=LoadGeneratorConfig(request_rate=request_rate),
            )

            worker_manager = WorkerManager(
                service_config=service_config,
                user_config=user_config,
                service_id="test-worker-manager",
            )

            assert worker_manager.max_workers == expected


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

        # Simulate time passing by manually updating the worker_info timestamp
        # (the actual implementation uses time.time_ns() internally)
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
