# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for warmup phase functionality."""

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults
from tests.integration.models import AIPerfMockServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestWarmup:
    """Tests for warmup phase functionality."""

    async def test_warmup_phase(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Warmup requests excluded from profiling metrics."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --warmup-request-count 5 \
                --request-count 15 \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == 15

    async def test_warmup_with_streaming(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Warmup with streaming enabled."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --streaming \
                --warmup-request-count 10 \
                --request-count 20 \
                --concurrency 4 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == 20
        assert result.has_streaming_metrics
