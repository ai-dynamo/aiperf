# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for dashboard UI mode with duration-based termination.

Dashboard mode with request-count termination is tested elsewhere:
- test_stress.py::test_high_worker_count_streaming
- test_gpu_telemetry.py
- test_server_metrics.py
"""

import pytest

from tests.harness.utils import AIPerfCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestDashboardUI:
    """Tests for dashboard UI mode with duration-based termination."""

    async def test_duration_based_termination(
        self, cli: AIPerfCLI, mock_server_factory
    ):
        """Dashboard UI with duration-based benchmark termination produces correct output."""
        # Use faster mock server settings for reliability
        async with mock_server_factory(ttft=10, itl=5) as aiperf_mock_server:
            result = await cli.run(
                f"""
                aiperf profile \
                    --model {defaults.model} \
                    --url {aiperf_mock_server.url} \
                    --tokenizer gpt2 \
                    --endpoint-type chat \
                    --ui dashboard \
                    --benchmark-duration 5 \
                    --benchmark-grace-period 10 \
                    --concurrency 3 \
                    --image-width-mean 64 \
                    --image-height-mean 64 \
                    --audio-length-mean 0.1
                """
            )
            # Verify benchmark completed and CSV contains duration config
            assert result.request_count >= 1
            assert "Benchmark Duration" in result.csv
