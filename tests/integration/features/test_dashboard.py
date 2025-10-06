# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dashboard UI integration tests."""

import pytest

from tests.integration.helpers import AIPerfCLI, FakeAIServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestDashboardUI:
    """Tests for dashboard UI mode."""

    async def test_with_request_count(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """Dashboard with request count limit."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --ui dashboard \
                --request-count 10 \
                --concurrency 2 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --audio-length-mean 0.1
            """
        )
        assert result.request_count == 10

    async def test_with_duration(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Dashboard with duration-based limit."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --ui dashboard \
                --benchmark-duration 10 \
                --streaming \
                --concurrency 3 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --audio-length-mean 0.1
            """,
            timeout=30.0,
        )
        assert result.request_count >= 3
        assert result.has_streaming_metrics
        assert "Benchmark Duration" in result.csv_content

    async def test_streaming_dashboard(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """Dashboard with streaming metrics."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --ui dashboard \
                --streaming \
                --request-count 20 \
                --concurrency 5
            """
        )
        assert result.request_count == 20
        assert result.has_streaming_metrics
