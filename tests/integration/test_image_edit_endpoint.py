# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for /v1/images/edits endpoint.

Based on: docs/tutorials/sglang-image-edit.md
"""

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestImageEditEndpoint:
    """Tests for /v1/images/edits endpoint."""

    async def test_image_edit_produces_no_streaming_metrics(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ) -> None:
        """Image edit completes requests without token-based streaming metrics.

        The endpoint POSTs a prompt plus a reference image as multipart/form-data.
        request_content_type auto-defaults to multipart for image_edit, so the
        CLI matches the documented tutorial usage.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model black-forest-labs/FLUX.2-klein-4B \
                --tokenizer gpt2 \
                --url {aiperf_mock_server.url} \
                --endpoint-type image_edit \
                --image-batch-size 1 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --synthetic-input-tokens-mean 50 \
                --synthetic-input-tokens-stddev 10 \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count

        assert result.json.time_to_first_token is None
        assert result.json.inter_token_latency is None
        assert result.json.time_to_second_token is None

        assert result.json.request_latency is not None
        assert result.json.request_throughput is not None

    async def test_image_edit_extra_inputs_pass_through(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ) -> None:
        """Diffusion-specific extras (size, num_inference_steps, guidance_scale)
        flow through to the multipart form fields without affecting the metric pipeline.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model black-forest-labs/FLUX.2-klein-4B \
                --tokenizer gpt2 \
                --url {aiperf_mock_server.url} \
                --endpoint-type image_edit \
                --image-batch-size 1 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --extra-inputs size:512x512 num_inference_steps:4 guidance_scale:1.0 \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count
        assert result.json.request_latency is not None
        assert result.json.request_throughput is not None
