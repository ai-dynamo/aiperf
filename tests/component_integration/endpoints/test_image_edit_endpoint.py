# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for /v1/images/edits endpoint (image_edit endpoint type)."""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestImageEditEndpoint:
    """End-to-end smoke tests for the image_edit endpoint."""

    def test_synthetic_image_edit(self, cli: AIPerfCLI):
        """Image edit with synthetic prompt and reference image."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model black-forest-labs/FLUX.2-klein-4B \
                --tokenizer gpt2 \
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
        assert (
            not hasattr(result.json, "time_to_first_token")
            or result.json.time_to_first_token is None
        )

    def test_image_edit_with_extra_inputs(self, cli: AIPerfCLI):
        """--extra-inputs flow through to multipart form fields."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model black-forest-labs/FLUX.2-klein-4B \
                --tokenizer gpt2 \
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
