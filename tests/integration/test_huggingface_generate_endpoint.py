# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults
from tests.integration.models import AIPerfMockServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestHuggingFaceGenerateEndpoint:
    """Integration tests for Hugging Face /generate and /generate_stream endpoints."""

    async def test_basic_generate(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Non-streaming text generation request."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model HuggingFaceH4/zephyr-7b-beta \
                --url {aiperf_mock_server.url} \
                --endpoint-type huggingface_generate \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.request_count == defaults.request_count
        assert not result.has_streaming_metrics
        assert "generate" in result.endpoint.lower()

    async def test_streaming_generate(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Streaming text generation request (/generate_stream)."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model HuggingFaceH4/zephyr-7b-beta \
                --url {aiperf_mock_server.url} \
                --endpoint-type huggingface_generate \
                --streaming \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.request_count == defaults.request_count
        assert result.has_streaming_metrics
        assert "generate" in result.endpoint.lower()
