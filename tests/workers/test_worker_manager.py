# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Simple test for WorkerManager max workers functionality.
"""

from unittest.mock import patch

import pytest

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.config.loadgen_config import LoadGeneratorConfig
from aiperf.common.config.worker_config import WorkersConfig
from aiperf.workers.worker_manager import WorkerManager


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
            user_config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                loadgen=LoadGeneratorConfig(concurrency=concurrency),
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
