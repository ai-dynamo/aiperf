# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for dashboard UI mode with different configurations."""

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.models import FakeAIServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestDashboardUI:
    """Tests for dashboard UI mode with different configurations."""

    async def test_with_request_count(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """Dashboard with fixed request count."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model Qwen/Qwen2.5-Coder-7B-Instruct \
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
        """Dashboard with time-based limit and streaming."""
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
        assert "Benchmark Duration" in result.csv
