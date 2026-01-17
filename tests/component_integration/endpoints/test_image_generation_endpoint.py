# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for /v1/images/generations endpoint.

Based on: docs/tutorials/sglang-image-generation.md
"""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestImageGenerationEndpoint:
    """Tests for /v1/images/generations endpoint.

    These tests mirror the usage patterns documented in the
    SGLang Image Generation tutorial.
    """

    def test_synthetic_image_generation(self, cli: AIPerfCLI):
        """Text-to-image generation using synthetic inputs.

        Based on tutorial example:
        aiperf profile --model black-forest-labs/FLUX.1-dev
                       --endpoint-type image_generation
                       --synthetic-input-tokens-mean 150
                       --synthetic-input-tokens-stddev 30
        """
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model black-forest-labs/FLUX.1-dev \
                --tokenizer gpt2 \
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

    def test_image_generation_with_extra_inputs(self, cli: AIPerfCLI):
        """Image generation with extra-inputs for size and quality.

        Based on tutorial example:
        aiperf profile --endpoint-type image_generation
                       --extra-inputs size:512x512 quality:standard

        Validates that extra-inputs are accepted for image generation endpoints.
        """
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model black-forest-labs/FLUX.1-dev \
                --tokenizer gpt2 \
                --endpoint-type image_generation \
                --extra-inputs size:512x512 quality:standard \
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
