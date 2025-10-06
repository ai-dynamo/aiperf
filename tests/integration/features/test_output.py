# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Output format and export integration tests."""

import pytest

from tests.integration.helpers import AIPerfCLI, FakeAIServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestOutputFormats:
    """Tests for different output formats and exports."""

    async def test_csv_export(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """CSV export format validation."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-count 10 \
                --concurrency 2 \
                --ui simple
            """
        )
        assert "Metric" in result.csv_content
        assert "Request Latency" in result.csv_content

    async def test_json_export(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """JSON export format validation."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --ui simple
            """
        )
        assert result.metrics is not None
        assert "request_count" in result.metrics
        assert "request_latency" in result.metrics


@pytest.mark.integration
@pytest.mark.asyncio
class TestUIOptions:
    """Tests for different UI options."""

    async def test_simple_ui(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Simple UI mode."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --ui simple
            """
        )
        assert result.request_count == 10

    async def test_none_ui(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """None UI mode (no output)."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --ui none
            """
        )
        assert result.request_count == 10


@pytest.mark.integration
@pytest.mark.asyncio
class TestWarmup:
    """Tests for warmup phase."""

    async def test_warmup_phase(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Warmup requests before profiling."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --warmup-request-count 5 \
                --request-count 15 \
                --concurrency 2 \
                --ui simple
            """
        )
        # Should only count the 15 profiling requests, not the 5 warmup
        assert result.request_count == 15

    async def test_warmup_with_streaming(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """Warmup with streaming."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-20b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --streaming \
                --warmup-request-count 10 \
                --request-count 20 \
                --concurrency 4 \
                --ui simple
            """
        )
        assert result.request_count == 20
        assert result.has_streaming_metrics
