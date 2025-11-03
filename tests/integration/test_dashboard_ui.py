# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for dashboard UI mode with different configurations."""

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults
from tests.integration.models import AIPerfMockServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestDashboardUI:
    """Tests for dashboard UI mode with different configurations."""

    async def test_with_request_count(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Dashboard with fixed request count."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --tokenizer gpt2 \
                --ui dashboard \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --audio-length-mean 0.1
            """
        )
        assert result.request_count == defaults.request_count

    async def test_with_duration(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Dashboard with time-based limit and streaming."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --ui dashboard \
                --benchmark-duration 10 \
                --streaming \
                --concurrency 3 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --audio-length-mean 0.1
            """
        )
        assert result.request_count >= 3
        assert result.has_streaming_metrics
        assert "Benchmark Duration" in result.csv
