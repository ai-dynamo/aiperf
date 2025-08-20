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
        "concurrency,request_rate,max_workers,expected",
        [
            (None, 1000, None, 9),  # Default case (10 fake CPUs - 1)
            (None, 1000, 4, 4),  # Only max set
            (None, None, None, 1),  # Concurrency defaults to 1
            (3, None, None, 3),  # Only concurrency set
            (2, None, 5, 2),  # Concurrency limits max
            (8, None, 3, 3),  # Max limits concurrency
            (10, 1000, 5, 5),  # Normal case with all values
        ],
    )
    def test_max_workers_combinations(
        self, concurrency, request_rate, max_workers, expected
    ):
        """Test various combinations of configuration values."""
        with patch(
            "aiperf.workers.worker_manager.multiprocessing.cpu_count", return_value=10
        ):
            service_config = ServiceConfig(workers=WorkersConfig(max=max_workers))
            user_config = UserConfig(
                endpoint=EndpointConfig(model_names=["test-model"]),
                loadgen=LoadGeneratorConfig(
                    concurrency=concurrency, request_rate=request_rate
                ),
            )

            worker_manager = WorkerManager(
                service_config=service_config,
                user_config=user_config,
                service_id="test-worker-manager",
            )

            assert worker_manager.max_workers == expected
