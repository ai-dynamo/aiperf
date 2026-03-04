# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for NIM Image Retrieval endpoint (/v1/image/infer)."""

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestImageRetrievalEndpoint:
    """Integration tests for NIM Image Retrieval endpoint."""

    async def test_basic_image_retrieval(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Image retrieval with synthetic images completes expected requests."""
        # NOTE: mock server endpoint path for image retrieval is different from the default
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/page-elements-v2 \
                --url {aiperf_mock_server.url} \
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
        assert result.json.request_throughput is not None
        assert result.json.image_throughput is not None
        assert result.json.image_latency is not None
