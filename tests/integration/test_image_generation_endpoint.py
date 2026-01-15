# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for /v1/images/generations endpoint.

Based on: docs/tutorials/sglang-image-generation.md
"""

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestImageGenerationEndpoint:
    """Tests for /v1/images/generations endpoint."""

    async def test_image_generation_produces_no_streaming_metrics(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Image generation completes requests without token-based streaming metrics.

        Unlike text generation endpoints, image generation does not produce
        time-to-first-token or inter-token-latency metrics since there is no
        token streaming - the entire image is returned as a single response.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model black-forest-labs/FLUX.1-dev \
                --tokenizer gpt2 \
                --url {aiperf_mock_server.url} \
                --endpoint-type image_generation \
                --synthetic-input-tokens-mean 150 \
                --synthetic-input-tokens-stddev 30 \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count

        # Image generation should not have token-based streaming metrics
        assert result.json.time_to_first_token is None
        assert result.json.inter_token_latency is None
        assert result.json.time_to_second_token is None

        # But should have basic request metrics
        assert result.json.request_latency is not None
        assert result.json.request_throughput is not None
