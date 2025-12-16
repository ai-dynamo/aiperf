# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for /v1/images/generations endpoint.

Based on: docs/tutorials/sglang-image-generation.md
"""

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults
from tests.integration.models import AIPerfMockServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestImageGenerationEndpoint:
    """Tests for /v1/images/generations endpoint.

    These tests mirror the usage patterns documented in the
    SGLang Image Generation tutorial.
    """

    async def test_synthetic_image_generation(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Text-to-image generation using synthetic inputs.

        Based on tutorial example:
        aiperf profile --model black-forest-labs/FLUX.1-dev
                       --endpoint-type image_generation
                       --synthetic-input-tokens-mean 150
                       --synthetic-input-tokens-stddev 30
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
        # Image generation doesn't produce tokens, so no token-based streaming metrics
        assert (
            not hasattr(result.json, "time_to_first_token")
            or result.json.time_to_first_token is None
        )

    async def test_image_generation_with_size_quality(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Image generation with size and quality parameters.

        Based on tutorial example:
        aiperf profile --endpoint-type image_generation
                       --extra-inputs size:512x512 quality:standard
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model black-forest-labs/FLUX.1-dev \
                --tokenizer gpt2 \
                --url {aiperf_mock_server.url} \
                --endpoint-type image_generation \
                --extra-inputs size:512x512 quality:standard \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count

    async def test_image_generation_basic(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Basic image generation without extra parameters.

        Simplified test for basic functionality.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model black-forest-labs/FLUX.1-dev \
                --tokenizer gpt2 \
                --url {aiperf_mock_server.url} \
                --endpoint-type image_generation \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count
