# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for NIM Image Retrieval endpoint (/v1/image/infer)."""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestImageRetrievalEndpoint:
    """Tests for NIM Image Retrieval endpoint."""

    def test_basic_image_retrieval(self, cli: AIPerfCLI):
        """Image retrieval with synthetic images completes expected requests."""
        # NOTE: mock server endpoint path for image retrieval is different from the default
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model nvidia/page-elements-v2 \
                --endpoint-type image_retrieval \
                --endpoint /v1/image/infer \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count
        assert result.json.time_to_first_token is None
        assert result.json.request_latency is not None
        assert result.json.image_throughput is not None
        assert result.json.image_latency is not None
